"""Plan detector for completeness and non-redundancy checks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from orchestrator.core.execution.tools.planning import render_plan
from orchestrator.core.session_types import Plan, PlanDetectorDecision, ToolCallRecord
from orchestrator.prompts.plan_detector import PLAN_DETECTOR_PROMPT
from orchestrator.shared.constants import StepStatus

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from orchestrator.core.session_types import OrchestratorSession

logger = logging.getLogger(__name__)


def _sanitize_refined_plan(plan: Plan) -> Plan:
    """Reset runtime-only fields before loading a detector-refined plan."""
    for step in plan.steps:
        step.status = StepStatus.PENDING
        step.actual_output = ""
        step.comment = ""
    return plan


def _detect_plan(
    llm: "BaseChatModel",
    *,
    task: str,
    plan_text: str,
) -> PlanDetectorDecision:
    return llm.with_structured_output(PlanDetectorDecision).invoke([
        {"role": "system", "content": PLAN_DETECTOR_PROMPT},
        {"role": "user", "content": (
            f"# Task\n{task}\n\n"
            f"# Plan\n{plan_text}\n"
        )},
    ])


def run_plan_detector(session: "OrchestratorSession", task: str) -> bool:
    """Refine the upfront plan when the detector finds missing or redundant work."""
    if not session.plan_store.steps:
        return False

    initial_step_count = len(session.plan_store.steps)
    initial_plan_text = render_plan(session.plan_store)
    try:
        decision = _detect_plan(
            llm=session.ctx.llm,
            task=task,
            plan_text=initial_plan_text,
        )
    except Exception:
        logger.warning("[plan_detector] Structured detector call failed", exc_info=True)
        session.timeline.append(ToolCallRecord(
            tool_name="plan_detector",
            tool_input=f"steps={len(session.plan_store.steps)}",
            tool_output="detector failed; original plan kept",
        ))
        return False

    needs_refinement = (
        not decision.satisfies_completeness
        or not decision.satisfies_non_redundancy
        or bool(decision.suggestions)
    )
    refined = (
        needs_refinement
        and decision.refined_plan is not None
        and bool(decision.refined_plan.steps)
    )

    if refined:
        session.plan_store.load_plan(_sanitize_refined_plan(decision.refined_plan))
        logger.info(
            "[plan_detector] Refined plan loaded: %d steps",
            len(session.plan_store.steps),
        )

    session.timeline.append(ToolCallRecord(
        tool_name="plan_detector",
        tool_input=f"steps={initial_step_count}",
        tool_output=(
            f"complete={decision.satisfies_completeness} "
            f"non_redundant={decision.satisfies_non_redundancy} "
            f"refined={'yes' if refined else 'no'} "
            f"suggestions={'; '.join(decision.suggestions)[:300]}"
        ),
    ))

    if needs_refinement and not refined:
        logger.warning(
            "[plan_detector] Detector suggested changes but did not return a refined plan"
        )
    return bool(refined)
