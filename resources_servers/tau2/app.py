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
"""τ²-bench Resources Server for NeMo Gym.

Wraps the τ² Environment and Evaluator as a NeMo Gym resources server
with session-based state isolation. Each session holds its own τ² Environment
instance, Task, and conversation history.
"""
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import RewardInfo, SimulationRun, TerminationReason
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment, EnvironmentInfo
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.registry import registry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class Tau2SeedSessionRequest(BaseSeedSessionRequest):
    domain: str
    task_id: str
    task_split_name: Optional[str] = "base"


class Tau2SeedSessionResponse(BaseSeedSessionResponse):
    session_id: str
    domain: str
    task_id: str
    tools: List[dict] = Field(default_factory=list)
    environment_info: Optional[dict] = None
    initial_messages: List[dict] = Field(default_factory=list)
    user_scenario: Optional[dict] = None


class Tau2ExecuteToolRequest(BaseModel):
    tool_name: str
    arguments: dict
    tool_call_id: Optional[str] = None
    requestor: str = "assistant"


class Tau2ExecuteToolResponse(BaseModel):
    content: str
    error: bool
    tool_call_id: str


class Tau2GetToolsResponse(BaseModel):
    tools: List[dict]


class Tau2GetEnvironmentInfoResponse(BaseModel):
    domain_name: str
    policy: str
    tool_descriptions: Optional[str] = None


class Tau2VerifyRequest(BaseVerifyRequest):
    domain: str
    task_id: str
    task_split_name: Optional[str] = "base"
    episode_messages: List[dict] = Field(default_factory=list)
    termination_reason: str = "agent_stop"
    evaluation_type: str = "all"


class Tau2VerifyResponse(BaseVerifyResponse):
    reward_info: Optional[dict] = None


# ---------------------------------------------------------------------------
# Per-session state
# ---------------------------------------------------------------------------


class SessionState:
    """Holds per-session τ² state."""

    def __init__(
        self,
        environment: Environment,
        task: Task,
        domain: str,
    ):
        self.environment = environment
        self.task = task
        self.domain = domain
        self.messages: List[Message] = []
        self.tool_trace: List[dict] = []


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class Tau2ResourcesServerConfig(BaseResourcesServerConfig):
    pass


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class Tau2ResourcesServer(SimpleResourcesServer):
    config: Tau2ResourcesServerConfig
    sessions: Dict[str, SessionState] = Field(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        app.post("/execute_tool")(self.execute_tool)
        app.post("/get_tools")(self.get_tools)
        app.post("/get_environment_info")(self.get_environment_info)

        return app

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_task(self, domain: str, task_id: str, task_split_name: Optional[str] = "base") -> Task:
        """Load a single task by domain and task_id."""
        tasks_loader = registry.get_tasks_loader(domain)
        tasks = tasks_loader(task_split_name)
        for task in tasks:
            if task.id == task_id:
                return task
        raise ValueError(f"Task '{task_id}' not found in domain '{domain}' split '{task_split_name}'")

    def _create_environment(self, domain: str) -> Environment:
        """Create a fresh τ² Environment for the given domain."""
        env_constructor = registry.get_env_constructor(domain)
        return env_constructor()

    def _get_tool_schemas(self, environment: Environment) -> List[dict]:
        """Get OpenAI-compatible tool schemas from environment."""
        tools = environment.get_tools()
        return [tool.openai_schema for tool in tools]

    def _get_session(self, session_id: str) -> SessionState:
        """Retrieve session state or raise."""
        if session_id not in self.sessions:
            raise ValueError(f"Session '{session_id}' not found. Call /seed_session first.")
        return self.sessions[session_id]

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    async def seed_session(self, request: Request, body: Tau2SeedSessionRequest) -> Tau2SeedSessionResponse:
        """Initialize a new τ² environment session for a specific domain/task."""
        session_id = request.session[SESSION_ID_KEY]

        # Create fresh environment and load task
        environment = self._create_environment(body.domain)
        task = self._load_task(body.domain, body.task_id, body.task_split_name)

        # Set initial state from task
        if task.initial_state is not None:
            environment.set_state(
                initialization_data=task.initial_state.initialization_data,
                initialization_actions=task.initial_state.initialization_actions,
                message_history=task.initial_state.message_history or [],
            )

        # Store session
        session = SessionState(
            environment=environment,
            task=task,
            domain=body.domain,
        )

        # Pre-populate message history from task initial state
        initial_messages_dicts = []
        if task.initial_state and task.initial_state.message_history:
            session.messages = list(task.initial_state.message_history)
            initial_messages_dicts = [msg.model_dump() for msg in task.initial_state.message_history]

        self.sessions[session_id] = session

        # Build response
        tool_schemas = self._get_tool_schemas(environment)
        env_info = environment.get_info(include_tool_info=False)

        return Tau2SeedSessionResponse(
            session_id=session_id,
            domain=body.domain,
            task_id=body.task_id,
            tools=tool_schemas,
            environment_info=env_info.model_dump(),
            initial_messages=initial_messages_dicts,
            user_scenario=task.user_scenario.model_dump(),
        )

    async def execute_tool(self, request: Request, body: Tau2ExecuteToolRequest) -> Tau2ExecuteToolResponse:
        """Execute a τ² environment tool call."""
        session_id = request.session[SESSION_ID_KEY]
        session = self._get_session(session_id)

        tool_call_id = body.tool_call_id or str(uuid.uuid4())

        # Build τ² ToolCall
        tc = ToolCall(
            id=tool_call_id,
            name=body.tool_name,
            arguments=body.arguments,
            requestor=body.requestor,
        )

        # Execute via environment
        tool_message: ToolMessage = session.environment.get_response(tc)

        # Log to session trace
        session.tool_trace.append({
            "tool_call_id": tool_call_id,
            "tool_name": body.tool_name,
            "arguments": body.arguments,
            "requestor": body.requestor,
            "content": tool_message.content,
            "error": tool_message.error,
        })

        return Tau2ExecuteToolResponse(
            content=tool_message.content or "",
            error=tool_message.error,
            tool_call_id=tool_call_id,
        )

    async def get_tools(self, request: Request) -> Tau2GetToolsResponse:
        """Return tool definitions for the current session's environment."""
        session_id = request.session[SESSION_ID_KEY]
        session = self._get_session(session_id)
        tool_schemas = self._get_tool_schemas(session.environment)
        return Tau2GetToolsResponse(tools=tool_schemas)

    async def get_environment_info(self, request: Request) -> Tau2GetEnvironmentInfoResponse:
        """Return domain policy and tool descriptions."""
        session_id = request.session[SESSION_ID_KEY]
        session = self._get_session(session_id)
        env = session.environment
        return Tau2GetEnvironmentInfoResponse(
            domain_name=env.get_domain_name(),
            policy=env.get_policy(),
            tool_descriptions=env.get_tools_description("assistant"),
        )

    async def verify(self, request: Request, body: Tau2VerifyRequest) -> Tau2VerifyResponse:
        """Evaluate a trajectory and return reward using τ² evaluator."""
        session_id = request.session[SESSION_ID_KEY]

        # Load task (may use session or reload)
        task = self._load_task(body.domain, body.task_id, body.task_split_name)

        # Reconstruct τ² messages from episode_messages
        messages = _deserialize_messages(body.episode_messages)

        # Build termination reason
        try:
            term_reason = TerminationReason(body.termination_reason)
        except ValueError:
            term_reason = TerminationReason.MAX_STEPS

        # Build SimulationRun for evaluator
        simulation = SimulationRun(
            id=str(uuid.uuid4()),
            task_id=task.id,
            start_time="",
            end_time="",
            duration=0.0,
            termination_reason=term_reason,
            messages=messages,
        )

        # Evaluate
        try:
            eval_type = EvaluationType(body.evaluation_type)
        except ValueError:
            eval_type = EvaluationType.ALL

        reward_info: RewardInfo = evaluate_simulation(
            simulation=simulation,
            task=task,
            evaluation_type=eval_type,
            solo_mode=False,
            domain=body.domain,
        )

        return Tau2VerifyResponse(
            **body.model_dump(),
            reward=reward_info.reward,
            reward_info=reward_info.model_dump(),
        )


# ---------------------------------------------------------------------------
# Message serialization helpers
# ---------------------------------------------------------------------------


def _deserialize_messages(message_dicts: List[dict]) -> List[Message]:
    """Convert a list of serialized message dicts back to τ² Message objects."""
    messages = []
    for md in message_dicts:
        role = md.get("role")
        if role == "system":
            messages.append(SystemMessage(**md))
        elif role == "assistant":
            messages.append(AssistantMessage(**md))
        elif role == "user":
            messages.append(UserMessage(**md))
        elif role == "tool":
            messages.append(ToolMessage(**md))
        else:
            logger.warning(f"Unknown message role '{role}', skipping.")
    return messages


if __name__ == "__main__":
    Tau2ResourcesServer.run_webserver()
