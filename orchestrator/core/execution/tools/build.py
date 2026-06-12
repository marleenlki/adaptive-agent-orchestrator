"""Assemble all executor tools from sub-modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

from orchestrator.core.execution.tools.context import make_gather_context
from orchestrator.core.execution.tools.delegation import make_delegate
from orchestrator.core.execution.tools.planning import make_planning_tools
from orchestrator.core.execution.tools.completion import make_task_complete

if TYPE_CHECKING:
    from orchestrator.core.session_types import OrchestratorSession


def build_tools(
    *,
    session: "OrchestratorSession",
    include_create_plan: bool = True,
) -> list:
    """Create all tools for the executor agent."""
    tools = [
        make_gather_context(session),
        *make_planning_tools(session, include_create_plan=include_create_plan),
        make_delegate(session),
        make_task_complete(session),
    ]

    if not session.ctx.enable_planning:
        plan_tool_names = {"create_plan", "view_plan", "update_step", "add_step"}
        tools = [t for t in tools if t.name not in plan_tool_names]

    return tools
