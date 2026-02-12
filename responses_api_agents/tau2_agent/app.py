# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""τ²-bench Response Agent (Orchestrator) for NeMo Gym.

Implements the SPEC's turn-based orchestration loop:
  1. Call Target-Agent (policy LLM) with conversation history + tool definitions
  2. If output is a tool call → forward to resources_server /execute_tool → append result → repeat
  3. If output is a normal assistant message → check termination → call User-Simulator
  4. Append user message → repeat until termination

The response_agent is NOT the policy model — it NEVER generates NL responses itself.
Target-Agent and User-Simulator are strictly separated.

Message management mirrors τ²-bench:
  - agent_messages: agent's accumulated view (system_messages + messages)
  - user_messages: user's accumulated view (system_messages + flip_roles(messages))
  - trajectory (all_outputs): complete unfiltered history

Orchestration follows τ²-bench Orchestrator.step() pattern:
  - State machine with (from_role, to_role) routing
  - USER/ENV → AGENT: call Target-Agent LLM
  - AGENT → ENV: execute pending tool calls
  - AGENT/ENV → USER: call User-Simulator LLM
"""
import json
import logging
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import Request, Response
from pydantic import ConfigDict, Field, ValidationError

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import get_response_json, raise_for_status

logger = logging.getLogger(__name__)

# Default first agent message (matches τ²-bench DEFAULT_FIRST_AGENT_MESSAGE)
DEFAULT_FIRST_AGENT_MESSAGE = "Hi! How can I help you today?"

# Agent system prompt (matches τ²-bench LLMAgent)
AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()

AGENT_SYSTEM_PROMPT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()

# Termination markers (from τ²-bench user/base.py)
STOP_MARKERS = {"###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###"}


# ---------------------------------------------------------------------------
# Role enum (mirrors τ²-bench orchestrator.Role)
# ---------------------------------------------------------------------------


class Role(str, Enum):
    AGENT = "agent"
    USER = "user"
    ENV = "env"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class Tau2AgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef  # Target-Agent policy LLM
    user_model_server: ModelServerRef  # User-Simulator LLM
    domain: str = "airline"
    task_id: Optional[str] = None  # Can be passed per-request via run()
    task_split_name: Optional[str] = "base"
    max_steps: int = 100
    max_errors: int = 10


# ---------------------------------------------------------------------------
# Run / Verify request/response models
# ---------------------------------------------------------------------------


class Tau2AgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    domain: Optional[str] = None
    task_id: Optional[str] = None
    task_split_name: Optional[str] = None
    task_data: Optional[dict] = None  # Single task entry from tasks.json
    max_steps: Optional[int] = None
    evaluation_type: str = "all"


class Tau2AgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    reward_info: Optional[dict] = None


# ---------------------------------------------------------------------------
# Episode state tracking
# ---------------------------------------------------------------------------


class EpisodeState:
    """Tracks the full state of an orchestration episode.

    Mirrors τ²-bench's separate state management:
      - agent_messages: messages the agent LLM has seen/produced
      - user_messages: messages the user-sim LLM has seen/produced (original roles, flipped at call time)
      - tau2_messages: complete trajectory for evaluation

    Orchestration state (mirrors τ²-bench Orchestrator):
      - from_role / to_role: current routing direction
      - done: whether the episode has terminated
    """

    def __init__(self, episode_id: str, domain: str, task_id: str):
        self.episode_id = episode_id
        self.domain = domain
        self.task_id = task_id
        self.turn_index = 0
        self.error_count = 0
        self.consecutive_errors = 0
        self.termination_reason: Optional[str] = None

        # Separate message histories (mirrors τ²-bench agent_state / user_state)
        self.agent_messages: List[Any] = []
        self.user_messages: List[Any] = []

        # τ² message history (serialized dicts for verification)
        self.tau2_messages: List[dict] = []

        # Tool definitions from environment
        self.tools: List[dict] = []

        # Environment info
        self.environment_info: Optional[dict] = None
        self.user_scenario: Optional[dict] = None
        self.simulation_guidelines: str = ""
        self.agent_system_prompt: str = ""

        # Orchestration state (mirrors τ²-bench Orchestrator)
        self.from_role: Optional[Role] = None
        self.to_role: Optional[Role] = None
        self.done: bool = False
        self.all_outputs: List[Any] = []
        self.last_agent_resp: Optional[NeMoGymResponse] = None
        self._pending_fn_calls: List[NeMoGymResponseFunctionToolCall] = []

        # Per-server cookies (managed across steps)
        self.model_server_cookies: Optional[dict] = None
        self.user_model_cookies: Optional[dict] = None
        self.resources_server_cookies: Optional[dict] = None


# ---------------------------------------------------------------------------
# Message format conversion utilities
# ---------------------------------------------------------------------------


def _extract_assistant_text(output: NeMoGymResponseOutputMessage) -> Optional[str]:
    """Extract text content from an assistant output message."""
    texts = []
    for content_item in output.content:
        if isinstance(content_item, NeMoGymResponseOutputText):
            texts.append(content_item.text)
    return "\n".join(texts) if texts else None


def _is_stop_message(text: Optional[str]) -> bool:
    """Check if text signals termination (matches τ²-bench 'in' check)."""
    if text is None:
        return False
    return any(marker in text for marker in STOP_MARKERS)


def _build_user_sim_system_prompt(
    user_scenario: Optional[dict],
    simulation_guidelines: str = "",
) -> str:
    """Build user-simulator system prompt matching τ²-bench format."""
    instructions = ""
    if user_scenario:
        instructions = user_scenario.get("instructions", "")
    return f"{simulation_guidelines}\n\n<scenario>\n{instructions}\n</scenario>"


def _flip_user_messages(user_messages: List[Any]) -> List[Any]:
    """Flip roles in user_messages for the user-simulator LLM.

    Mirrors τ²-bench UserState.flip_roles():
      - AssistantMessage (agent text) → role="user" (input to sim LLM)
      - UserMessage (user text) → role="assistant" (sim LLM's prior output)
      - function_call_output (user tool result) → kept as-is
    """
    flipped = []
    for msg in user_messages:
        if hasattr(msg, "type") and msg.type == "function_call_output":
            # User's tool result — keep as-is
            flipped.append(msg)
            continue

        if hasattr(msg, "role"):
            if msg.role == "assistant":
                content = msg.content if isinstance(msg.content, str) else ""
                flipped.append(NeMoGymEasyInputMessage(role="user", content=content))
            elif msg.role == "user":
                content = msg.content if isinstance(msg.content, str) else ""
                flipped.append(NeMoGymEasyInputMessage(role="assistant", content=content))
    return flipped


# ---------------------------------------------------------------------------
# Agent (Orchestrator)
# ---------------------------------------------------------------------------


class Tau2Agent(SimpleResponsesAPIAgent):
    config: Tau2AgentConfig

    async def _step(self, episode: EpisodeState) -> None:
        """Perform one step of the orchestration.

        Mirrors τ²-bench Orchestrator.step() — a state machine that routes
        messages between AGENT, USER, and ENV based on (from_role, to_role):

          - USER/ENV → AGENT: call Target-Agent LLM
          - AGENT → ENV: execute pending tool calls
          - AGENT/ENV → USER: call User-Simulator LLM

        Updates episode.from_role, episode.to_role, and episode.done.
        """
        if episode.done:
            return

        logger.debug(
            f"Step {episode.turn_index}: {episode.from_role} → {episode.to_role}"
        )

        # ── USER/ENV → AGENT: Call Target-Agent LLM ──────────────────────
        if episode.to_role == Role.AGENT:
            agent_input: List[Any] = []
            if episode.agent_system_prompt:
                agent_input.append(
                    NeMoGymEasyInputMessage(role="system", content=episode.agent_system_prompt)
                )
            agent_input.extend(episode.agent_messages)

            agent_body = NeMoGymResponseCreateParamsNonStreaming(
                input=agent_input,
                tools=episode.tools,
            )

            agent_response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=agent_body,
                cookies=episode.model_server_cookies,
            )
            await raise_for_status(agent_response)
            agent_response_json = await get_response_json(agent_response)
            episode.model_server_cookies = agent_response.cookies

            try:
                agent_resp = NeMoGymResponse.model_validate(agent_response_json)
            except ValidationError:
                logger.error(f"Invalid model response: {json.dumps(agent_response_json)}")
                episode.termination_reason = "agent_error"
                episode.done = True
                return

            episode.last_agent_resp = agent_resp

            # Classify outputs
            fn_calls: List[NeMoGymResponseFunctionToolCall] = [
                o for o in agent_resp.output if o.type == "function_call"
            ]
            output_messages: List[NeMoGymResponseOutputMessage] = [
                o for o in agent_resp.output if o.type == "message" and o.role == "assistant"
            ]

            if fn_calls:
                # Agent made tool calls → record them, route to ENV
                # tau2_messages: ONE assistant message with ALL tool_calls
                # (get_actions_from_messages expects tool_calls[] then ToolMessages immediately after)
                tau2_tool_calls = []
                for fn_call in fn_calls:
                    episode.agent_messages.append(fn_call)
                    episode.all_outputs.append(fn_call)

                    tool_args = json.loads(fn_call.arguments) if isinstance(fn_call.arguments, str) else fn_call.arguments
                    tau2_tool_calls.append({
                        "id": fn_call.call_id,
                        "name": fn_call.name,
                        "arguments": tool_args,
                        "requestor": "assistant",
                    })

                episode.tau2_messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tau2_tool_calls,
                })

                episode._pending_fn_calls = fn_calls
                episode.from_role = Role.AGENT
                episode.to_role = Role.ENV

            elif output_messages:
                # Agent sent text → record it, route to USER
                for msg in output_messages:
                    episode.agent_messages.append(msg)
                    episode.all_outputs.append(msg)

                    assistant_text = _extract_assistant_text(msg)
                    if assistant_text:
                        episode.user_messages.append(
                            NeMoGymEasyInputMessage(role="assistant", content=assistant_text)
                        )

                    episode.tau2_messages.append({
                        "role": "assistant",
                        "content": assistant_text,
                    })

                    if _is_stop_message(assistant_text):
                        episode.termination_reason = "agent_stop"
                        episode.done = True
                        return

                episode.from_role = Role.AGENT
                episode.to_role = Role.USER

            else:
                # No tool calls and no output messages — should not happen
                logger.warning("Target-Agent produced neither tool calls nor messages.")
                episode.termination_reason = "agent_error"
                episode.done = True
                return

        # ── AGENT/USER → ENV: Execute pending tool calls ────────────────
        elif episode.to_role == Role.ENV:
            episode.consecutive_errors = 0
            # requestor tracks who made the tool call (mirrors τ²-bench ToolMessage.requestor)
            requestor = "assistant" if episode.from_role == Role.AGENT else "user"

            for fn_call in episode._pending_fn_calls:
                tool_args = json.loads(fn_call.arguments) if isinstance(fn_call.arguments, str) else fn_call.arguments

                tool_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path="/execute_tool",
                    json={
                        "tool_name": fn_call.name,
                        "arguments": tool_args,
                        "tool_call_id": fn_call.call_id,
                        "requestor": requestor,
                    },
                    cookies=episode.resources_server_cookies,
                )
                episode.resources_server_cookies = tool_response.cookies
                tool_result = await get_response_json(tool_response)

                tool_content = tool_result.get("content", "")
                tool_error = tool_result.get("error", False)
                tool_call_id = tool_result.get("tool_call_id", fn_call.call_id)

                if tool_error:
                    episode.error_count += 1
                    episode.consecutive_errors += 1
                    if episode.consecutive_errors >= self.config.max_errors:
                        episode.termination_reason = "too_many_errors"
                        episode.done = True
                        break

                tool_output = NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=tool_call_id,
                    output=tool_content,
                )
                # Tool result goes to whoever made the call
                if episode.from_role == Role.AGENT:
                    episode.agent_messages.append(tool_output)
                else:
                    episode.user_messages.append(tool_output)
                episode.all_outputs.append(tool_output)

                episode.tau2_messages.append({
                    "role": "tool",
                    "id": tool_call_id,
                    "content": tool_content,
                    "requestor": requestor,
                    "error": tool_error,
                })

            episode._pending_fn_calls = []
            # Route back to whoever made the call (mirrors τ²-bench: self.to_role = self.from_role)
            episode.to_role = episode.from_role
            episode.from_role = Role.ENV

        # ── AGENT/ENV → USER: Call User-Simulator ────────────────────────
        elif episode.to_role == Role.USER:
            user_system_prompt = _build_user_sim_system_prompt(
                user_scenario=episode.user_scenario,
                simulation_guidelines=episode.simulation_guidelines,
            )
            user_input: List[Any] = [
                NeMoGymEasyInputMessage(role="system", content=user_system_prompt)
            ]
            user_input.extend(_flip_user_messages(episode.user_messages))

            user_sim_body = NeMoGymResponseCreateParamsNonStreaming(
                input=user_input,
                tools=[],
            )

            user_response = await self.server_client.post(
                server_name=self.config.user_model_server.name,
                url_path="/v1/responses",
                json=user_sim_body,
                cookies=episode.user_model_cookies,
            )
            await raise_for_status(user_response)
            user_response_json = await get_response_json(user_response)
            episode.user_model_cookies = user_response.cookies

            try:
                user_resp = NeMoGymResponse.model_validate(user_response_json)
            except ValidationError:
                logger.error(f"Invalid user model response: {json.dumps(user_response_json)}")
                episode.termination_reason = "user_error"
                episode.done = True
                return

            # Classify user outputs (mirrors agent classification in AGENT step)
            user_fn_calls: List[NeMoGymResponseFunctionToolCall] = [
                o for o in user_resp.output if o.type == "function_call"
            ]
            user_output_messages: List[NeMoGymResponseOutputMessage] = [
                o for o in user_resp.output if o.type == "message"
            ]

            if user_fn_calls:
                # User made tool calls → record them, route to ENV
                # tau2_messages: ONE user message with ALL tool_calls
                tau2_tool_calls = []
                for fn_call in user_fn_calls:
                    episode.user_messages.append(fn_call)
                    episode.all_outputs.append(fn_call)

                    tool_args = json.loads(fn_call.arguments) if isinstance(fn_call.arguments, str) else fn_call.arguments
                    tau2_tool_calls.append({
                        "id": fn_call.call_id,
                        "name": fn_call.name,
                        "arguments": tool_args,
                        "requestor": "user",
                    })

                episode.tau2_messages.append({
                    "role": "user",
                    "content": None,
                    "tool_calls": tau2_tool_calls,
                })

                episode._pending_fn_calls = user_fn_calls
                episode.from_role = Role.USER
                episode.to_role = Role.ENV

            elif user_output_messages:
                # User sent text → extract, check stop, route to AGENT
                # (mirrors τ²-bench: not user_msg.is_tool_call() → to_role = AGENT)
                user_text = None
                for out in user_output_messages:
                    user_text = _extract_assistant_text(out)
                    if user_text:
                        break
                if user_text is None:
                    user_text = ""

                user_msg = NeMoGymEasyInputMessage(role="user", content=user_text)
                episode.user_messages.append(user_msg)
                episode.agent_messages.append(user_msg)
                episode.all_outputs.append(user_msg)

                episode.tau2_messages.append({"role": "user", "content": user_text})

                if _is_stop_message(user_text):
                    episode.termination_reason = "user_stop"
                    episode.done = True
                    return

                episode.from_role = Role.USER
                episode.to_role = Role.AGENT

            else:
                logger.warning("User-Simulator produced neither tool calls nor messages.")
                episode.termination_reason = "user_error"
                episode.done = True
                return

        else:
            raise ValueError(
                f"Invalid role combination: from={episode.from_role}, to={episode.to_role}"
            )

        episode.turn_index += 1

    async def _orchestrate(
        self,
        episode: EpisodeState,
        max_steps: Optional[int] = None,
    ) -> tuple[NeMoGymResponse, EpisodeState]:
        """Core orchestration loop — calls _step() until done.

        Mirrors τ²-bench Orchestrator.run(): repeatedly calls step() and
        checks termination conditions after each non-ENV step.
        """
        effective_max_steps = max_steps or self.config.max_steps

        while not episode.done:
            await self._step(episode)

            # Check max_steps/max_errors only when not routing to ENV
            # (tool execution steps are "free", matching τ²-bench behavior)
            if episode.to_role == Role.ENV:
                continue
            if episode.turn_index >= effective_max_steps:
                episode.done = True
                episode.termination_reason = "max_steps"
            if episode.error_count >= self.config.max_errors:
                episode.done = True
                episode.termination_reason = "too_many_errors"

        # Build final response using the last agent response as a base
        if episode.last_agent_resp is not None:
            final_response = episode.last_agent_resp
        else:
            final_response = NeMoGymResponse.model_validate({
                "id": str(uuid.uuid4()),
                "created_at": 0,
                "model": "",
                "object": "response",
                "output": [],
            })
        final_response.output = episode.all_outputs

        return final_response, episode

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        """Implements the /v1/responses endpoint.

        For standalone use (without run()), this runs the orchestration loop
        with a minimal episode. For full τ²-bench evaluation, use /run instead.
        """
        body = body.model_copy(deep=True)

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        episode = EpisodeState(
            episode_id=str(uuid.uuid4()),
            domain=self.config.domain,
            task_id=self.config.task_id or "",
        )
        episode.tools = body.tools if body.tools else []
        episode.resources_server_cookies = request.cookies

        # Populate agent_messages from input
        for msg in body.input:
            episode.agent_messages.append(msg)

        # Initial state: input provided → agent should respond next
        episode.from_role = Role.USER
        episode.to_role = Role.AGENT

        final_response, _ = await self._orchestrate(episode=episode)

        return final_response

    async def run(self, request: Request, body: Tau2AgentRunRequest) -> Tau2AgentVerifyResponse:
        """Full pipeline: seed → orchestrate → verify → reward."""
        cookies = request.cookies

        # Resolve domain/task_id from request or config
        domain = body.domain or self.config.domain
        task_data = body.task_data
        task_split_name = body.task_split_name or self.config.task_split_name
        max_steps = body.max_steps or self.config.max_steps

        # If task_data is provided (single entry from tasks.json), extract task_id from it
        if task_data is not None:
            task_id = task_data.get("id", body.task_id or self.config.task_id)
        else:
            task_id = body.task_id or self.config.task_id

        if not task_id and task_data is None:
            raise ValueError("task_id or task_data must be provided either in the request or in the agent config.")

        # -----------------------------------------------------------
        # 1. Seed session
        # -----------------------------------------------------------
        seed_request: dict = {
            "domain": domain,
            "task_id": task_id,
            "task_split_name": task_split_name,
        }
        if task_data is not None:
            seed_request["task_data"] = task_data

        seed_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=seed_request,
            cookies=cookies,
        )
        await raise_for_status(seed_response)
        cookies = seed_response.cookies
        seed_data = await get_response_json(seed_response)

        # -----------------------------------------------------------
        # 2. Build episode state
        # -----------------------------------------------------------
        episode = EpisodeState(
            episode_id=str(uuid.uuid4()),
            domain=domain,
            task_id=task_id,
        )
        episode.tools = seed_data.get("tools", [])
        episode.environment_info = seed_data.get("environment_info")
        episode.user_scenario = seed_data.get("user_scenario")
        episode.simulation_guidelines = seed_data.get("simulation_guidelines", "")
        episode.resources_server_cookies = cookies

        # Build agent system prompt (matches τ²-bench LLMAgent.system_prompt)
        env_info = seed_data.get("environment_info", {})
        domain_policy = env_info.get("policy", "")
        episode.agent_system_prompt = AGENT_SYSTEM_PROMPT.format(
            agent_instruction=AGENT_INSTRUCTION,
            domain_policy=domain_policy,
        )

        # -----------------------------------------------------------
        # 3. Initialize message histories (mirrors τ²-bench initialize())
        # -----------------------------------------------------------
        initial_messages = seed_data.get("initial_messages", [])

        # Pre-populate τ² messages from task initial state
        for msg_dict in initial_messages:
            episode.tau2_messages.append(msg_dict)

        if initial_messages:
            # Populate agent_messages and user_messages from initial history
            # Mirrors τ²-bench: filter by is_valid_agent/user_history_message
            for msg_dict in initial_messages:
                role = msg_dict.get("role", "user")
                content = msg_dict.get("content")
                requestor = msg_dict.get("requestor", "assistant")

                if role == "assistant" and content:
                    msg = NeMoGymEasyInputMessage(role="assistant", content=content)
                    episode.agent_messages.append(msg)
                    # User sees agent text (non-tool-call only)
                    if "tool_calls" not in msg_dict or not msg_dict["tool_calls"]:
                        episode.user_messages.append(msg)
                elif role == "user" and content:
                    msg = NeMoGymEasyInputMessage(role="user", content=content)
                    episode.agent_messages.append(msg)
                    episode.user_messages.append(msg)
                elif role == "tool":
                    tool_output = NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=msg_dict.get("id", ""),
                        output=content or "",
                    )
                    if requestor == "assistant":
                        episode.agent_messages.append(tool_output)
                    elif requestor == "user":
                        episode.user_messages.append(tool_output)
        else:
            # Default case: no initial messages
            # τ²-bench: agent sends greeting, user_state is empty, then user responds
            greeting_msg = NeMoGymEasyInputMessage(
                role="assistant", content=DEFAULT_FIRST_AGENT_MESSAGE
            )

            # Agent has sent the greeting
            episode.agent_messages.append(greeting_msg)
            episode.tau2_messages.append({
                "role": "assistant",
                "content": DEFAULT_FIRST_AGENT_MESSAGE,
            })

            # Call User-Simulator to respond to greeting
            if episode.user_scenario:
                # User receives greeting (append to user_messages before calling)
                episode.user_messages.append(greeting_msg)

                first_user_text = await self._get_initial_user_message(
                    episode=episode,
                    cookies=cookies,
                )
                first_user_msg = NeMoGymEasyInputMessage(role="user", content=first_user_text)

                # User's output → append to user_messages
                episode.user_messages.append(first_user_msg)
                # Agent's next input → append to agent_messages
                episode.agent_messages.append(first_user_msg)

                episode.tau2_messages.append({"role": "user", "content": first_user_text})

        # -----------------------------------------------------------
        # 4. Run orchestration loop
        # -----------------------------------------------------------
        # After initialization, the last message is from USER → agent responds next
        episode.from_role = Role.USER
        episode.to_role = Role.AGENT

        response_result, episode = await self._orchestrate(
            episode=episode,
            max_steps=max_steps,
        )

        # -----------------------------------------------------------
        # 5. Verify
        # -----------------------------------------------------------
        verify_request_data = {
            "domain": domain,
            "task_id": task_id,
            "task_split_name": task_split_name,
            "episode_messages": episode.tau2_messages,
            "termination_reason": episode.termination_reason,
            "evaluation_type": body.evaluation_type,
            "responses_create_params": body.responses_create_params.model_dump()
                if hasattr(body.responses_create_params, "model_dump")
                else body.responses_create_params,
            "response": response_result.model_dump(),
        }

        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request_data,
            cookies=cookies,
        )
        await raise_for_status(verify_response)

        return Tau2AgentVerifyResponse.model_validate(await get_response_json(verify_response))

    async def _get_initial_user_message(
        self,
        episode: EpisodeState,
        cookies: Optional[dict] = None,
    ) -> str:
        """Generate the initial user message via the User-Simulator LLM.

        Uses episode.user_messages (which should contain the agent greeting)
        and flips roles before sending to the user-sim LLM.
        """
        user_system_prompt = _build_user_sim_system_prompt(
            user_scenario=episode.user_scenario,
            simulation_guidelines=episode.simulation_guidelines,
        )
        user_input: List[Any] = [
            NeMoGymEasyInputMessage(role="system", content=user_system_prompt)
        ]
        user_input.extend(_flip_user_messages(episode.user_messages))

        user_sim_body = NeMoGymResponseCreateParamsNonStreaming(
            input=user_input,
            tools=[],
        )

        user_response = await self.server_client.post(
            server_name=self.config.user_model_server.name,
            url_path="/v1/responses",
            json=user_sim_body,
            cookies=cookies,
        )
        await raise_for_status(user_response)
        user_response_json = await get_response_json(user_response)

        try:
            user_resp = NeMoGymResponse.model_validate(user_response_json)
            for out in user_resp.output:
                if out.type == "message":
                    text = _extract_assistant_text(out)
                    if text:
                        return text
        except (ValidationError, Exception) as e:
            logger.warning(f"Failed to get initial user message from simulator: {e}")

        # Fallback: derive from user scenario
        instructions = episode.user_scenario.get("instructions", "") if episode.user_scenario else ""
        if isinstance(instructions, dict):
            return instructions.get("reason_for_call", "Hello, I need help.")
        elif isinstance(instructions, str):
            return instructions
        return "Hello, I need help."


if __name__ == "__main__":
    Tau2Agent.run_webserver()
