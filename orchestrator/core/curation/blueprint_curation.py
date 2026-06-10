"""Episode-level blueprint curator."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from pydantic import Field, create_model

from orchestrator.prompts.episode_curator import EPISODE_CURATOR_SUCCESS_PROMPT
from orchestrator.core.session_types import DelegationExchange, ToolCallRecord
from orchestrator.core.curation.types import BlueprintStepOutput, SuccessCuratorOutput
from orchestrator.memory.records import (
    BlueprintRecord,
    DelegationBlueprint,
    DelegationStep,
)
from orchestrator.shared.constants import JUDGE_STEP_ID

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from orchestrator.core.session_types import OrchestratorSession

logger = logging.getLogger(__name__)
OUTPUT_PREVIEW_LIMIT = 800


# --- Orchestration ---

def run_blueprint_curation(
    llm: "BaseChatModel",
    session: "OrchestratorSession",
    success: bool = True,
    *,
    known_agents: set[str] | None = None,
) -> BlueprintRecord | None:
    """Run blueprint curation and return a reusable BlueprintRecord.

    Only runs on successful episodes.
    """
    if not success:
        return None

    try:
        prompt = _build_success_prompt(session, known_agents=known_agents)
        schema = _success_schema(known_agents) if known_agents else SuccessCuratorOutput
        raw = llm.with_structured_output(schema).invoke(prompt)
        record = _success_to_blueprint(raw, session.task)
        logger.info(
            "[blueprint_curation] success=%s steps=%d",
            success, len(record.blueprint.steps) if record.blueprint else 0,
        )
        return record

    except Exception:
        logger.warning("[blueprint_curation] LLM call failed", exc_info=True)
        return None


def _build_success_prompt(
    session: "OrchestratorSession",
    *,
    known_agents: set[str] | None,
) -> str:
    available_agents = ", ".join(sorted(known_agents)) if known_agents else "(unknown)"
    return EPISODE_CURATOR_SUCCESS_PROMPT.format(
        task=session.task,
        execution_timeline=_format_episode_timeline(session),
        total_delegations=len(session.history),
        retrieved_blueprint=session.retrieved_blueprint_text or "(none)",
        available_agents=available_agents,
        judge_rejections=session.judge_rejections,
    )


def _success_schema(known_agents: set[str]) -> type[SuccessCuratorOutput]:
    """Build a SuccessCuratorOutput variant whose step.agent is restricted to known_agents."""
    agent_type = Literal[tuple(sorted(known_agents))]  # type: ignore[valid-type]
    step = create_model(
        "ConstrainedBlueprintStep",
        __base__=BlueprintStepOutput,
        agent=(agent_type, Field(description="Must be one of the available agents.")),
    )
    return create_model(
        "ConstrainedSuccessCuratorOutput",
        __base__=SuccessCuratorOutput,
        ideal_steps=(list[step], ...),  # type: ignore[valid-type]
    )


def _success_to_blueprint(raw: SuccessCuratorOutput, task_fallback: str) -> BlueprintRecord:
    """Convert a SuccessCuratorOutput to a BlueprintRecord."""
    steps = [
        DelegationStep(
            agent=step.agent,
            does=step.does,
            receives=step.receives,
            produces=step.produces,
        )
        for step in raw.ideal_steps
        if step.agent != JUDGE_STEP_ID
    ]
    blueprint = DelegationBlueprint(steps=steps)
    return BlueprintRecord(
        task=raw.task.strip() or task_fallback,
        blueprint=blueprint,
        agents_involved=sorted({s.agent for s in steps}),
        refines_retrieved=raw.refines_retrieved,
    )


def _format_episode_timeline(session: "OrchestratorSession") -> str:
    blocks: list[str] = []
    delegation_number = 0

    for entry in session.timeline:
        if isinstance(entry, DelegationExchange):
            if entry.agent == JUDGE_STEP_ID:
                continue
            delegation_number += 1
            blocks.append(_format_delegation(entry, delegation_number))
        elif isinstance(entry, ToolCallRecord):
            blocks.append(_format_tool_call(entry))

    judge_feedback = _format_judge_feedback(session)
    if judge_feedback:
        blocks.append(judge_feedback)

    return "\n\n".join(blocks) or "(no delegations)"


def _format_delegation(record: DelegationExchange, delegation_number: int) -> str:
    lines = [f"[{delegation_number}] {record.agent}"]
    if record.instruction:
        lines.append(f"  instruction: {record.instruction[:200]}")
    lines.append(f"  output: {_preview(record.actual_output)}")
    return "\n".join(lines)


def _format_tool_call(record: ToolCallRecord) -> str:
    return f"  ⚙ {record.tool_name}({record.tool_input[:80]}) → {record.tool_output[:120]}"


def _format_judge_feedback(session: "OrchestratorSession") -> str:
    judge_records = [record for record in session.history if record.agent == JUDGE_STEP_ID]
    if not judge_records:
        return ""

    lines = ["## Judge Feedback"]
    for index, record in enumerate(judge_records, start=1):
        verdict = "ACCEPTED" if record.success else "REJECTED"
        lines.append(_format_judge_record(index, verdict, record))

    if session.judge_force_accepted:
        lines.append("  → Force-accepted after max rejections")
    return "\n".join(lines)


def _format_judge_record(index: int, verdict: str, record: DelegationExchange) -> str:
    lines = [f"  {index}. {verdict}: {record.feedback}"]
    for tag, value in _extract_judge_fields(record.actual_output or ""):
        lines.append(f"     {tag.capitalize()}: {value}")
    return "\n".join(lines)


def _extract_judge_fields(actual_output: str) -> list[tuple[str, str]]:
    fields: list[tuple[str, str]] = []
    for tag in ("HINT", "REASONING"):
        marker = f"| {tag}:"
        if marker not in actual_output:
            continue
        value = actual_output.split(marker, 1)[1].split("|")[0].strip()
        fields.append((tag, value))
    return fields


def _preview(text: str) -> str:
    preview = text[:OUTPUT_PREVIEW_LIMIT]
    return f"{preview}…" if len(text) > OUTPUT_PREVIEW_LIMIT else preview
