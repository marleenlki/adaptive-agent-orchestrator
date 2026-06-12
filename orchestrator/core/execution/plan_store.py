from __future__ import annotations

import logging

from orchestrator.core.session_types import Plan, PlanStep, StepStatus

logger = logging.getLogger(__name__)


class AdaptivePlanStore:
    """In-memory plan for the executor"""

    def __init__(self) -> None:
        self._steps: list[PlanStep] = []
        self.goal: str = ""
        self.deliverables: list[str] = []
        self._step_counter: int = 0

    def load_plan(self, plan: Plan) -> None:
        """Replace the current plan with a complete new plan."""
        self._steps = []
        self._step_counter = 0
        for step in plan.steps:
            step.step_id = self._next_step_id()
            self._steps.append(step)

        self.goal = plan.goal
        self.deliverables = list(plan.deliverables)

        logger.info("Plan loaded: %d steps", len(plan.steps))

    @property
    def steps(self) -> list[PlanStep]:
        """Read-only: the step list is managed via load_plan/add_step/mark_*."""
        return self._steps

    def get_step(self, step_id: str) -> PlanStep | None:
        return next((s for s in self._steps if s.step_id == step_id), None)

    def _index_of(self, step_id: str) -> int | None:
        return next((i for i, s in enumerate(self._steps) if s.step_id == step_id), None)

    def _next_step_id(self) -> str:
        """Return the next unused id, skipping any already in the plan."""
        existing = {s.step_id for s in self._steps}
        self._step_counter += 1
        while f"s{self._step_counter}" in existing:
            self._step_counter += 1
        return f"s{self._step_counter}"

    def _require_step(self, step_id: str) -> PlanStep:
        step = self.get_step(step_id)
        if step is None:
            raise ValueError(f"step '{step_id}' not found in plan.")
        return step

    def mark_step_done(self, step_id: str, comment: str = "") -> PlanStep:
        step = self._require_step(step_id)
        step.status = StepStatus.DONE
        if comment:
            step.comment = comment
        return step

    def mark_step_failed(self, step_id: str, comment: str = "") -> PlanStep:
        step = self._require_step(step_id)
        step.status = StepStatus.FAILED
        if comment:
            step.comment = comment
        return step

    def mark_step_skipped(self, step_id: str) -> PlanStep:
        step = self._require_step(step_id)
        step.status = StepStatus.SKIPPED
        return step

    def add_step(
        self,
        *,
        agent: str,
        instruction: str = "",
        after_step_id: str = "",
        before_step_id: str = "",
        reasoning: str = "",
    ) -> PlanStep:
        """Append a new step or insert it relative to another step.

        Raises ValueError if before_step_id is given but not found;
        an unknown after_step_id falls back to appending.
        """
        new_step = PlanStep(
            step_id=self._next_step_id(),
            agent=agent,
            instruction=instruction,
            reasoning=reasoning,
            status=StepStatus.PENDING,
        )

        if before_step_id:
            idx = self._index_of(before_step_id)
            if idx is None:
                raise ValueError(f"before_step_id '{before_step_id}' not found in plan.")
            self._steps.insert(idx, new_step)
        elif after_step_id and (idx := self._index_of(after_step_id)) is not None:
            self._steps.insert(idx + 1, new_step)
        else:
            self._steps.append(new_step)

        return new_step
