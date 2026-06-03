"""task_complete tool — final-answer submission with optional judge gate."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

from orchestrator.prompts import VERIFICATION_GATE_PROMPT
from orchestrator.core.session_types import DelegationExchange, JudgeDecision, ToolCallRecord
from orchestrator.instrumentation.recorder import EpisodeMetricsRecorder
from orchestrator.shared.constants import EXIT_KEY_REQUESTED, JUDGE_STEP_ID

if TYPE_CHECKING:
    from orchestrator.core.session_types import OrchestratorSession

logger = logging.getLogger(__name__)


def _judge_trajectory(
    history: list[DelegationExchange],
    timeline: list[DelegationExchange | ToolCallRecord],
) -> str:
    lines: list[str] = []
    delegation_index = 0
    for record in history:
        if record.agent == JUDGE_STEP_ID:
            continue
        delegation_index += 1
        agent_line = (
            f"{delegation_index}. {record.instruction[:100]}\n"
            f"   Agent: {record.agent}"
        )
        if record.reasoning:
            agent_line += f"\n   Reasoning: {record.reasoning[:100]}"
        if record.actual_output:
            agent_line += f"\n   Output:\n{record.actual_output}"
        lines.append(agent_line)

    tool_lines = [
        f"- {record.tool_name}({record.tool_input[:100]}) -> {record.tool_output[:100]}"
        for record in timeline
        if isinstance(record, ToolCallRecord)
    ]
    if tool_lines:
        lines.append("## Orchestrator tool calls\n" + "\n".join(tool_lines))

    return "\n\n".join(lines) if lines else "No delegations recorded."


def _evaluate_answer(
    llm: BaseChatModel,
    task: str,
    answer: str,
    trajectory: str,
) -> JudgeDecision:
    """Evaluate a candidate answer against the execution trajectory."""
    candidate = answer.strip()
    if not candidate:
        return JudgeDecision(
            reasoning="Empty answer provided.",
            accepted=False,
            feedback="Empty final answer.",
        )

    result = llm.with_structured_output(JudgeDecision).invoke([
        {"role": "system", "content": VERIFICATION_GATE_PROMPT},
        {"role": "user", "content": (
            f"# Task\n{task}\n\n"
            f"# Candidate Answer\n{candidate}\n\n"
            f"# Trajectory\n{trajectory}\n"
        )},
    ])
    feedback = result.feedback.strip() or (
        "Accepted." if result.accepted else "Not ready yet."
    )
    return JudgeDecision(
        reasoning=result.reasoning,
        accepted=result.accepted,
        feedback=feedback,
        task_summary=(result.task_summary or "").strip(),
    )


def make_task_complete(session: "OrchestratorSession"):
    """Build the task_complete tool closed over *session*."""
    ctx = session.ctx

    @tool("task_complete")
    def task_complete(
        final_answer: str,
        justification: str = "",
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> Command | str:
        """Submit the final answer when the task is complete.

        Args:
            final_answer: The complete answer for the user.
            justification: Map deliverables to evidence from agent outputs.
        """
        def accept(message: str) -> Command:
            session.final_answer = final_answer
            return Command(update={
                EXIT_KEY_REQUESTED: True,
                "messages": [ToolMessage(content=message, tool_call_id=tool_call_id)],
            })

        def record_verdict(verdict: str, *, success: bool) -> None:
            detail = decision.feedback
            if decision.reasoning:
                detail += f" | REASONING: {decision.reasoning}"
            session.timeline.append(DelegationExchange(
                step_number=len(session.history) + 1,
                agent=JUDGE_STEP_ID,
                instruction="Validate final answer",
                actual_output=f"{verdict}: {detail}",
                success=success,
                feedback=decision.feedback,
            ))

        session.timeline.append(ToolCallRecord(
            tool_name="task_complete",
            tool_input=f"answer={final_answer[:200]}",
            tool_output="",
        ))
        logger.info("[task_complete] answer=%s", final_answer[:200])

        if not ctx.enable_judge:
            return accept("Answer accepted (judge disabled).")

        candidate = f"{final_answer}\n\n---\nJustification:\n{justification}" if justification else final_answer

        try:
            decision = _evaluate_answer(
                llm=ctx.judge_llm,
                task=session.task,
                answer=candidate,
                trajectory=_judge_trajectory(session.history, session.timeline),
            )
            session.timeline.append(ToolCallRecord(
                tool_name="judge_review",
                tool_input=f"answer={candidate!r}",
                tool_output=f"accepted={decision.accepted}, feedback={decision.feedback}",
            ))
        except Exception:
            logger.warning("Judge LLM call failed — auto-accepting", exc_info=True)
            decision = JudgeDecision(accepted=True, feedback="Judge error — auto-accepted.")

        session.submission_attempts += 1
        if decision.task_summary:
            session.judge_task_summary = decision.task_summary

        if decision.accepted:
            record_verdict("ACCEPTED", success=True)
            return accept("Answer accepted.")

        session.judge_rejections += 1
        record_verdict("REJECTED", success=False)

        if session.judge_rejections >= ctx.max_judge_rejections:
            logger.warning("[task_complete] Max judge rejections — force-accepting")
            session.judge_force_accepted = True
            EpisodeMetricsRecorder(session.metrics).record_force_accept()
            return accept("Answer force-accepted after max rejections.")

        return (
            f"REJECTED by verification judge ({session.judge_rejections}/{ctx.max_judge_rejections}).\n"
            f"Feedback: {decision.feedback}\n\n"
            f"Fix the issues and call task_complete() again."
        )

    return task_complete
