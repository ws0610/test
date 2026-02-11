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
"""
import json
import logging
import uuid
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
    max_steps: Optional[int] = None
    evaluation_type: str = "all"


class Tau2AgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    reward_info: Optional[dict] = None


# ---------------------------------------------------------------------------
# Episode state tracking
# ---------------------------------------------------------------------------


class EpisodeState:
    """Tracks the full state of an orchestration episode."""

    def __init__(self, episode_id: str, domain: str, task_id: str):
        self.episode_id = episode_id
        self.domain = domain
        self.task_id = task_id
        self.turn_index = 0
        self.error_count = 0
        self.consecutive_errors = 0
        self.termination_reason: Optional[str] = None

        # τ² message history (serialized dicts for verification)
        self.tau2_messages: List[dict] = []

        # Tool definitions from environment
        self.tools: List[dict] = []

        # Environment info
        self.environment_info: Optional[dict] = None
        self.user_scenario: Optional[dict] = None
        self.simulation_guidelines: str = ""
        self.agent_system_prompt: str = ""


# ---------------------------------------------------------------------------
# Message format conversion utilities
# ---------------------------------------------------------------------------


class _TaggedMessage:
    """Wraps a conversation message with a requestor tag.

    NeMoGymFunctionCallOutput has no requestor field, so we wrap it
    to distinguish agent vs user tool interactions in the conversation.
    The inner `msg` is the actual NeMoGym type sent to the LLM.
    """

    def __init__(self, msg: Any, requestor: str = "assistant"):
        self.msg = msg
        self.requestor = requestor
        # Proxy type/role so filter functions can inspect without unwrapping
        self.type = getattr(msg, "type", None)
        self.role = getattr(msg, "role", None)


def _build_agent_input(
    conversation: List[Any],
    agent_system_prompt: str,
) -> List[Any]:
    """Build filtered input for Target-Agent LLM call.

    Mirrors τ²-bench is_valid_agent_history_message:
      - AssistantMessage (text + tool calls)
      - UserMessage (text only, NOT user tool calls)
      - ToolMessage with requestor="assistant" only
    """
    agent_input: List[Any] = []
    if agent_system_prompt:
        agent_input.append(
            NeMoGymEasyInputMessage(role="system", content=agent_system_prompt)
        )

    for item in conversation:
        msg = item.msg if isinstance(item, _TaggedMessage) else item
        requestor = item.requestor if isinstance(item, _TaggedMessage) else "assistant"

        # function_call (agent's tool call) → agent sees it
        if hasattr(msg, "type") and msg.type == "function_call":
            agent_input.append(msg)
            continue

        # function_call_output → agent only sees its own (requestor="assistant")
        if hasattr(msg, "type") and msg.type == "function_call_output":
            if requestor == "assistant":
                agent_input.append(msg)
            continue

        if hasattr(msg, "role"):
            if msg.role == "system":
                continue
            elif msg.role == "assistant":
                agent_input.append(msg)
            elif msg.role == "user":
                agent_input.append(msg)

    return agent_input


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


def _build_user_simulator_input(
    conversation: List[Any],
    user_scenario: Optional[dict],
    simulation_guidelines: str = "",
) -> NeMoGymResponseCreateParamsNonStreaming:
    """Build the input for the User-Simulator LLM call.

    Mirrors τ²-bench's UserSimulator.system_prompt:
      {global_user_sim_guidelines}
      <scenario>
      {instructions}
      </scenario>

    The user-simulator receives the conversation history plus a system
    instruction containing simulation guidelines and the user scenario
    so it can role-play the user.
    """
    # Extract instructions from user_scenario (pass as-is, matching τ²-bench)
    instructions = ""
    if user_scenario:
        instructions = user_scenario.get("instructions", "")

    # Build system prompt matching τ²-bench format:
    # {global_user_sim_guidelines}\n\n<scenario>\n{instructions}\n</scenario>
    system_prompt = f"{simulation_guidelines}\n\n<scenario>\n{instructions}\n</scenario>"

    # Build input: system message + conversation history with flipped roles.
    # Mirrors τ²-bench UserState.flip_roles():
    #   - AssistantMessage (text only) → "user" (agent's words become input)
    #   - UserMessage → "assistant" (user's words become sim LLM's prior output)
    #   - ToolMessage requestor="user" → kept as tool (user's tool results)
    #   - Agent tool calls / tool results → skipped (not visible to user)
    #   - System messages → skipped (sim has its own system prompt)
    input_messages = [NeMoGymEasyInputMessage(role="system", content=system_prompt)]

    for item in conversation:
        msg = item.msg if isinstance(item, _TaggedMessage) else item
        requestor = item.requestor if isinstance(item, _TaggedMessage) else "assistant"

        # function_call → always agent's, skip for user-sim
        if hasattr(msg, "type") and msg.type == "function_call":
            continue

        # function_call_output → user sees only its own (requestor="user")
        if hasattr(msg, "type") and msg.type == "function_call_output":
            if requestor == "user":
                input_messages.append(msg)
            continue

        if hasattr(msg, "role"):
            if msg.role == "system":
                continue
            elif msg.role == "assistant":
                # Agent's message → flip to "user" for the sim LLM
                content = msg.content if hasattr(msg, "content") else None
                if content is None and hasattr(msg, "text"):
                    content = msg.text
                # Extract text from NeMoGymResponseOutputMessage
                if content is None and hasattr(msg, "type") and msg.type == "message":
                    content = _extract_assistant_text(msg)
                if content:
                    input_messages.append(
                        NeMoGymEasyInputMessage(role="user", content=content)
                    )
            elif msg.role == "user":
                # User's message → flip to "assistant" for the sim LLM
                content = msg.content if hasattr(msg, "content") else ""
                input_messages.append(
                    NeMoGymEasyInputMessage(role="assistant", content=content)
                )

    return NeMoGymResponseCreateParamsNonStreaming(
        input=input_messages,
        tools=[],
    )


# ---------------------------------------------------------------------------
# Agent (Orchestrator)
# ---------------------------------------------------------------------------


class Tau2Agent(SimpleResponsesAPIAgent):
    config: Tau2AgentConfig

    async def _orchestrate(
        self,
        input_messages: List[Any],
        episode: EpisodeState,
        cookies: Optional[dict] = None,
        max_steps: Optional[int] = None,
    ) -> tuple[NeMoGymResponse, EpisodeState]:
        """Core orchestration loop implementing the τ²-bench turn-based interaction.

        Coordinates Target-Agent ↔ User-Simulator ↔ Environment.

        Returns:
            Tuple of (final NeMoGymResponse with all outputs, updated EpisodeState).
        """
        effective_max_steps = max_steps or self.config.max_steps
        model_server_cookies = None
        user_model_cookies = None
        resources_server_cookies = cookies

        conversation = list(input_messages)
        all_outputs = []
        last_agent_resp = None

        step = 0
        while step < effective_max_steps:
            step += 1
            episode.turn_index += 1

            # -----------------------------------------------------------
            # Step 1: Call Target-Agent (policy LLM)
            # Filter conversation for agent (matches τ²-bench: system_messages + agent-visible messages)
            # -----------------------------------------------------------
            agent_input = _build_agent_input(
                conversation=conversation,
                agent_system_prompt=episode.agent_system_prompt,
            )

            agent_body = NeMoGymResponseCreateParamsNonStreaming(
                input=agent_input,
                tools=episode.tools,
            )

            agent_response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=agent_body,
                cookies=model_server_cookies,
            )
            await raise_for_status(agent_response)
            agent_response_json = await get_response_json(agent_response)
            model_server_cookies = agent_response.cookies

            try:
                agent_resp = NeMoGymResponse.model_validate(agent_response_json)
            except ValidationError as e:
                logger.error(f"Invalid model response: {json.dumps(agent_response_json)}")
                episode.termination_reason = "agent_error"
                break

            last_agent_resp = agent_resp
            output = agent_resp.output

            # Classify outputs
            fn_calls: List[NeMoGymResponseFunctionToolCall] = [
                o for o in output if o.type == "function_call"
            ]
            output_messages: List[NeMoGymResponseOutputMessage] = [
                o for o in output if o.type == "message" and o.role == "assistant"
            ]

            # -----------------------------------------------------------
            # Step 2: If tool calls → forward to Environment, then loop
            # -----------------------------------------------------------
            if fn_calls:
                episode.consecutive_errors = 0

                for fn_call in fn_calls:
                    # Add function_call to conversation (tagged as agent's)
                    conversation.append(_TaggedMessage(fn_call, requestor="assistant"))
                    all_outputs.append(fn_call)

                    # Log to τ² messages
                    tool_args = json.loads(fn_call.arguments) if isinstance(fn_call.arguments, str) else fn_call.arguments
                    episode.tau2_messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": fn_call.call_id,
                            "name": fn_call.name,
                            "arguments": tool_args,
                            "requestor": "assistant",
                        }],
                    })

                    # Execute tool via resources server
                    tool_response = await self.server_client.post(
                        server_name=self.config.resources_server.name,
                        url_path="/execute_tool",
                        json={
                            "tool_name": fn_call.name,
                            "arguments": tool_args,
                            "tool_call_id": fn_call.call_id,
                            "requestor": "assistant",
                        },
                        cookies=resources_server_cookies,
                    )
                    resources_server_cookies = tool_response.cookies
                    tool_result = await get_response_json(tool_response)

                    tool_content = tool_result.get("content", "")
                    tool_error = tool_result.get("error", False)
                    tool_call_id = tool_result.get("tool_call_id", fn_call.call_id)

                    if tool_error:
                        episode.error_count += 1
                        episode.consecutive_errors += 1
                        if episode.consecutive_errors >= self.config.max_errors:
                            episode.termination_reason = "too_many_errors"
                            break

                    # Append tool result to conversation (tagged with requestor)
                    tool_output = NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=tool_call_id,
                        output=tool_content,
                    )
                    conversation.append(_TaggedMessage(tool_output, requestor="assistant"))
                    all_outputs.append(tool_output)

                    # Log to τ² messages
                    episode.tau2_messages.append({
                        "role": "tool",
                        "id": tool_call_id,
                        "content": tool_content,
                        "requestor": "assistant",
                        "error": tool_error,
                    })

                if episode.termination_reason == "too_many_errors":
                    break

                # Continue the ReAct loop — call Target-Agent again
                continue

            # -----------------------------------------------------------
            # Step 3: Normal assistant message → check DONE → call User-Sim
            # -----------------------------------------------------------
            if output_messages:
                terminated = False
                for msg in output_messages:
                    conversation.append(msg) #agent output (아마도 asssitant로 들어갈거다)
                    all_outputs.append(msg)

                    assistant_text = _extract_assistant_text(msg)

                    # Log to τ² messages
                    episode.tau2_messages.append({
                        "role": "assistant",
                        "content": assistant_text,
                    })

                    # Check for termination
                    if _is_stop_message(assistant_text):
                        episode.termination_reason = "agent_stop"
                        terminated = True
                        break

                if terminated:
                    break

                # -----------------------------------------------------------
                # Step 4: Call User-Simulator
                # -----------------------------------------------------------
                user_sim_body = _build_user_simulator_input(
                    conversation=conversation,
                    user_scenario=episode.user_scenario,
                    simulation_guidelines=episode.simulation_guidelines,
                )

                user_response = await self.server_client.post(
                    server_name=self.config.user_model_server.name,
                    url_path="/v1/responses",
                    json=user_sim_body,
                    cookies=user_model_cookies,
                )
                await raise_for_status(user_response)
                user_response_json = await get_response_json(user_response)
                user_model_cookies = user_response.cookies

                try:
                    user_resp = NeMoGymResponse.model_validate(user_response_json)
                except ValidationError as e:
                    logger.error(f"Invalid user model response: {json.dumps(user_response_json)}")
                    episode.termination_reason = "user_error"
                    break

                # Extract user text from the model output
                user_text = None
                for out in user_resp.output:
                    if out.type == "message":
                        user_text = _extract_assistant_text(out)
                        break

                if user_text is None:
                    user_text = ""

                # Log to τ² messages
                episode.tau2_messages.append({
                    "role": "user",
                    "content": user_text,
                })

                # Append user message to conversation
                user_msg = NeMoGymEasyInputMessage(role="user", content=user_text)
                conversation.append(user_msg)
                all_outputs.append(user_msg)

                # Check if user signals done
                if _is_stop_message(user_text):
                    episode.termination_reason = "user_stop"
                    break

                # Continue loop
                continue

            # No tool calls and no output messages — should not happen
            logger.warning("Target-Agent produced neither tool calls nor messages. Breaking.")
            episode.termination_reason = "agent_error"
            break

        # Check for max_steps termination
        if episode.termination_reason is None:
            episode.termination_reason = "max_steps"

        # Build final response using the last agent response as a base
        if last_agent_resp is not None:
            final_response = last_agent_resp
        else:
            final_response = NeMoGymResponse.model_validate({
                "id": str(uuid.uuid4()),
                "created_at": 0,
                "model": "",
                "object": "response",
                "output": [],
            })
        final_response.output = all_outputs

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

        final_response, _ = await self._orchestrate(
            input_messages=body.input,
            episode=episode,
            cookies=request.cookies,
        )

        return final_response

    async def run(self, request: Request, body: Tau2AgentRunRequest) -> Tau2AgentVerifyResponse:
        """Full pipeline: seed → orchestrate → verify → reward."""
        cookies = request.cookies

        # Resolve domain/task_id from request or config
        domain = body.domain or self.config.domain
        task_id = body.task_id or self.config.task_id
        task_split_name = body.task_split_name or self.config.task_split_name
        max_steps = body.max_steps or self.config.max_steps

        if not task_id:
            raise ValueError("task_id must be provided either in the request or in the agent config.")

        # -----------------------------------------------------------
        # 1. Seed session
        # -----------------------------------------------------------
        seed_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json={
                "domain": domain,
                "task_id": task_id,
                "task_split_name": task_split_name,
            },
            cookies=cookies,
        )
        await raise_for_status(seed_response)
        cookies = seed_response.cookies
        seed_data = await get_response_json(seed_response)

        # -----------------------------------------------------------
        # 2. Build initial conversation and episode state
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

        # Pre-populate τ² messages from task initial state
        initial_messages = seed_data.get("initial_messages", [])
        for msg_dict in initial_messages:
            episode.tau2_messages.append(msg_dict)

        # Build agent system prompt (matches τ²-bench LLMAgent.system_prompt)
        env_info = seed_data.get("environment_info", {})
        domain_policy = env_info.get("policy", "")
        episode.agent_system_prompt = AGENT_SYSTEM_PROMPT.format(
            agent_instruction=AGENT_INSTRUCTION,
            domain_policy=domain_policy,
        )

        input_messages: List[Any] = []

        # Add initial messages from task history as conversation context
        # Include all message types (assistant, user, tool) — filtering is done
        # at call time by _build_agent_input / _build_user_simulator_input
        for msg_dict in initial_messages:
            role = msg_dict.get("role", "user")
            content = msg_dict.get("content")
            if role == "tool":
                # Tool results from initial state (tagged with requestor)
                tool_output = NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=msg_dict.get("id", ""),
                    output=content or "",
                )
                requestor = msg_dict.get("requestor", "assistant")
                input_messages.append(_TaggedMessage(tool_output, requestor=requestor))
            elif role in ("user", "assistant") and content:
                input_messages.append(
                    NeMoGymEasyInputMessage(role=role, content=content)
                )

        # τ²-bench flow: Agent sends first greeting, then User-Simulator responds.
        # Only do this when there are no initial messages from the task.
        has_user_msg = any(
            hasattr(m, "role") and m.role == "user"
            for m in input_messages
        )
        if not has_user_msg:
            # Step A: Add default agent first message (matches τ²-bench)
            input_messages.append(
                NeMoGymEasyInputMessage(role="assistant", content=DEFAULT_FIRST_AGENT_MESSAGE)
            )
            episode.tau2_messages.append({
                "role": "assistant",
                "content": DEFAULT_FIRST_AGENT_MESSAGE,
            })

            # Step B: Call User-Simulator to generate the first user response
            if episode.user_scenario:
                first_user_msg = await self._get_initial_user_message(
                    input_messages=input_messages,
                    user_scenario=episode.user_scenario,
                    simulation_guidelines=episode.simulation_guidelines,
                    cookies=cookies,
                )
                input_messages.append(
                    NeMoGymEasyInputMessage(role="user", content=first_user_msg)
                )
                episode.tau2_messages.append({"role": "user", "content": first_user_msg})#user가 답변해서 assitant로 나왔지만 user로 등록..그러면 agent에 던질 때 수정안해도됨.

        # -----------------------------------------------------------
        # 3. Run orchestration loop
        # -----------------------------------------------------------
        response_result, episode = await self._orchestrate(
            input_messages=input_messages,
            episode=episode,
            cookies=cookies,
            max_steps=max_steps,
        )

        # -----------------------------------------------------------
        # 4. Verify
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
        input_messages: List[Any],
        user_scenario: dict,
        simulation_guidelines: str = "",
        cookies: Optional[dict] = None,
    ) -> str:
        """Generate the initial user message via the User-Simulator LLM.

        In τ²-bench, the agent sends "Hi! How can I help you today?" first,
        and the user simulator responds to it. input_messages should already
        contain the agent's greeting at this point.
        """
        user_sim_body = _build_user_simulator_input(
            conversation=input_messages,
            user_scenario=user_scenario,
            simulation_guidelines=simulation_guidelines,
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
        instructions = user_scenario.get("instructions", "")
        if isinstance(instructions, dict):
            return instructions.get("reason_for_call", "Hello, I need help.")
        elif isinstance(instructions, str):
            return instructions
        return "Hello, I need help."


if __name__ == "__main__":
    Tau2Agent.run_webserver()
