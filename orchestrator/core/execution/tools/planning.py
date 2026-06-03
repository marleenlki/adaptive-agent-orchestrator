"""Planning tools — create_plan, view_plan, update_step, add_step."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.tools import tool

from orchestrator.core.session_types import Plan, StepStatus, ToolCallRecord
from orchestrator.shared.constants import MAX_LABEL_PREVIEW, MAX_SHORT_PREVIEW

if TYPE_CHECKING:
    from orchestrator.core.session_types import OrchestratorSession
    from orchestrator.core.execution.plan_store import AdaptivePlanStore

logger = logging.getLogger(__name__)

_STEP_NOT_FOUND = "Error: step '{}' not found in plan."


def _render_plan(plan_store: "AdaptivePlanStore") -> str:
    if not plan_store.steps:
        return "Plan is empty."

    # Goal + deliverables ride along as a runtime alignment aid for the
    # executor — shown only when set.
    header: list[str] = []
    if plan_store.goal:
        header.append(f"Goal: {plan_store.goal}")
    if plan_store.deliverables:
        header.append(
            "Deliverables:\n" + "\n".join(f"  - {d}" for d in plan_store.deliverables)
        )

    lines: list[str] = []
    for step in plan_store.steps:
        reasoning = (
            f"\n    reasoning: {step.reasoning[:MAX_LABEL_PREVIEW]}"
            if step.reasoning else ""
        )
        instruction = f"\n    instruction: {step.instruction}" if step.instruction else ""
        output = (
            f"\n    actual_output: {step.actual_output[:MAX_SHORT_PREVIEW]}"
            if step.actual_output else ""
        )
        lines.append(
            f"[{step.step_id}] [{step.status}]\n"
            f"    agent: {step.agent}{reasoning}{instruction}{output}"
        )
    body = "\n\n".join(lines)
    if not header:
        return body
    return "\n".join(header) + "\n\n" + body


def make_planning_tools(session: "OrchestratorSession") -> list:
    """Build planning tools"""
    ctx = session.ctx
    plan_store = session.plan_store

    @tool("create_plan")
    def create_plan(plan: Plan) -> str:
        """Create an execution plan for complex tasks.

        Plans are advisory and they help organize multi-step work but
        never constrain you. You can always delegate() without a plan.

        Args:
            plan: The full execution plan.
        """

        plan_store.load_plan(plan)
        result = _render_plan(plan_store)

        session.timeline.append(ToolCallRecord(
            tool_name="create_plan",
            tool_input=f"goal={plan.goal[:80]} steps={len(plan.steps)}",
            tool_output=f"{len(plan.steps)} steps, {len(plan.deliverables)} deliverables",
        ))
        logger.info("[create_plan] %d steps loaded", len(plan.steps))
        return result

    @tool("update_step")
    def update_step(
        step_id: str,
        status: str,
        comment: str = "",
    ) -> str:
        """Update a plan step's status after reviewing its output.

        Args:
            step_id: The step to update.
            status: New status "done", "skipped", or "failed".
            comment: Brief description of outcome or reason for skip/fail.
        """
        step = plan_store.get_step(step_id)
        if step is None:
            return _STEP_NOT_FOUND.format(step_id)

        status_lower = status.lower().strip()
        if comment:
            step.comment = comment

        if status_lower == "done":
            if step.status != StepStatus.DONE:
                plan_store.mark_step_done(step_id, comment=comment)
        elif status_lower == "skipped":
            plan_store.mark_step_skipped(step_id)
        elif status_lower == "failed":
            plan_store.mark_step_failed(step_id, comment=comment)
        else:
            return f"Error: status must be 'done', 'skipped', or 'failed'. Got '{status}'."

        session.timeline.append(ToolCallRecord(
            tool_name="update_step",
            tool_input=f"step_id={step_id} status={status_lower}",
            tool_output=f"step [{step_id}] marked {status_lower}",
        ))
        return f"step [{step_id}] marked {status_lower}"

    @tool("add_step")
    def add_step(
        instruction: str,
        agent: str = "",
        reasoning: str = "",
        after_step_id: str = "",
        before_step_id: str = "",
    ) -> str:
        """Add a new step to the plan.

        Args:
            instruction: What this step should do.
            agent: Which agent you plan to use (optional).
            reasoning: Why this step is needed and how it helps achieve the goal.
            after_step_id: Insert after this step (default: append).
            before_step_id: Insert before this step.
        """
        new_step = plan_store.add_step(
            instruction=instruction,
            agent=agent,
            reasoning=reasoning,
            after_step_id=after_step_id,
            before_step_id=before_step_id,
        )

        step_ids = [s.step_id for s in plan_store.steps]
        pos = step_ids.index(new_step.step_id) + 1
        result = (
            f"Added step [{new_step.step_id}] at position {pos}/{len(step_ids)}: {instruction[:80]}"
        )

        session.timeline.append(ToolCallRecord(
            tool_name="add_step",
            tool_input=f"step_id={new_step.step_id}",
            tool_output=result,
        ))
        return result

    @tool("view_plan")
    def view_plan() -> str:
        """Show the current plan with step statuses and outputs.

        Use to re-orient yourself during long tasks — see which steps
        are done, pending, or failed, and what each step produced.
        Works for both explicit plans and ad-hoc delegation tracking.
        """
        if not plan_store.steps:
            return "No plan exists. Use create_plan() or delegate directly."
        result = _render_plan(plan_store)
        session.timeline.append(ToolCallRecord(
            tool_name="view_plan",
            tool_input="",
            tool_output=f"{len(plan_store.steps)} steps",
        ))
        return result

    return [create_plan, update_step, add_step, view_plan]
