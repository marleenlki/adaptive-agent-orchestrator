"""Upfront planner ReAct agent.

The planner owns initial retrieval and plan creation. The executor then
uses that plan as its starting point and adapts during execution.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from orchestrator.core.execution.graph import build_executor_agent
from orchestrator.core.execution.tools.context import make_gather_context
from orchestrator.core.execution.tools.planning import make_planning_tools
from orchestrator.core.session_types import ToolCallRecord
from orchestrator.prompts.planner import PLANNER_PROMPT
from orchestrator.shared.constants import EXIT_KEY_REQUESTED, RECURSION_LIMIT

if TYPE_CHECKING:
    from orchestrator.core.session_types import OrchestratorSession

logger = logging.getLogger(__name__)


def _make_planning_complete(session: "OrchestratorSession"):
    @tool("planning_complete")
    def planning_complete(
        summary: str = "",
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> Command:
        """Finish the upfront planning phase after create_plan has been called.

        Args:
            summary: Brief summary of the created plan.
        """
        session.timeline.append(ToolCallRecord(
            tool_name="planning_complete",
            tool_input=summary[:200],
            tool_output=f"{len(session.plan_store.steps)} planned steps",
        ))
        return Command(update={
            EXIT_KEY_REQUESTED: True,
            "messages": [
                ToolMessage(content="Planning complete.", tool_call_id=tool_call_id)
            ],
        })

    return planning_complete


def run_planner(session: "OrchestratorSession", task: str) -> bool:
    """Retrieve planning context, create an upfront plan, and return whether one exists."""
    ctx = session.ctx
    create_plan_tool = next(
        tool_
        for tool_ in make_planning_tools(session, include_create_plan=True)
        if tool_.name == "create_plan"
    )
    tools = [
        make_gather_context(session),
        create_plan_tool,
        _make_planning_complete(session),
    ]

    agent, config = build_executor_agent(
        llm=ctx.llm,
        tools=tools,
        system_prompt=PLANNER_PROMPT.format(task=task),
        thread_id=f"{session.thread_id}::planner",
        recursion_limit=RECURSION_LIMIT,
    )

    try:
        agent.invoke({"messages": [{"role": "user", "content": task}]}, config=config)
    except GraphRecursionError:
        logger.warning("[planner] Recursion limit hit")
    except Exception:
        logger.warning("[planner] Upfront planning failed", exc_info=True)

    plan_created = bool(session.plan_store.steps)
    if not plan_created:
        logger.warning("[planner] No upfront plan produced; executor will continue reactively")
    return plan_created
