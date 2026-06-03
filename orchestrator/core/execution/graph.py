"""Builds the executor's LangGraph ReAct agent with middleware"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typing_extensions import Annotated, NotRequired, Required
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import before_model, wrap_model_call
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.runtime import Runtime

from orchestrator.shared.constants import DEFAULT_THREAD_ID, EXIT_KEY_REQUESTED

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool


# Agent state

class ExecutorAgentState(AgentState):
    """Agent state with an early-exit flag that the middleware can check to jump to the end node."""
    messages: Required[Annotated[list[AnyMessage], add_messages]]
    exit_requested: NotRequired[bool]


# Middleware

@wrap_model_call
def _force_sequential_tool_calls(request, handler):
    """Force tool_choice and disable parallel tool calls."""
    request.tool_choice = "required"
    request.model_settings = {
        **(request.model_settings or {}),
        "parallel_tool_calls": False,
    }
    return handler(request)


@before_model(state_schema=ExecutorAgentState, can_jump_to=["end"])
def _stop_on_exit(state: ExecutorAgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Jump to the end node once a tool has set ``exit_requested``."""
    if state.get(EXIT_KEY_REQUESTED):
        return {"jump_to": "end"}
    return None


# Builder

def build_executor_agent(
    llm: BaseChatModel,
    tools: list[BaseTool],
    system_prompt: str,
    *,
    thread_id: str = DEFAULT_THREAD_ID,
    recursion_limit: int = 100,
    checkpointer=None,
) -> tuple[Any, dict]:
    """Create the executor ReAct agent and its invoke config."""
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        state_schema=ExecutorAgentState,
        middleware=[_force_sequential_tool_calls, _stop_on_exit],
        checkpointer=checkpointer,
    )
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }
    return agent, config
