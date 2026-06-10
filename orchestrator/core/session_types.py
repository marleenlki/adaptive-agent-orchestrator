"""Pydantic schemas for the adaptive orchestrator.

Architecture:
  A single unified ReAct agent handles the full orchestration lifecycle:
  context gathering, planning, delegation, adaptation, and completion.
  An answer judge validates the final answer before returning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from orchestrator.shared.constants import StepStatus
from orchestrator.instrumentation.metrics import EpisodeMetrics
from orchestrator.instrumentation.trajectory import DelegationExchange

if TYPE_CHECKING:
    from orchestrator.core.resources import OrchestratorResources
else:
    OrchestratorResources = Any


class AgentContext(BaseModel):
    """Bundled context for one agent returned by assemble_plan_context."""

    name: str = Field(description="Agent name (matches registry key).")
    description: str = Field(default="", description="Agent card description.")
    skills: list[str] = Field(
        default_factory=list,
        description="Skill names from the agent card.",
    )
    score: float = Field(
        default=0.0,
        description="Relevance score (0–1) from memory or embedding search.",
    )
    source: str = Field(
        default="",
        description="How the agent was found, e.g. 'embedding', 'capability', or 'blueprint'.",
    )
    matched_capabilities: list[str] = Field(
        default_factory=list,
        description="Which capability queries matched this agent.",
    )
    playbook: str = Field(
        default="",
        description="Confirmed playbook bullets from observed behavior.",
    )
    capability_score: float = Field(
        default=0.0,
        description="Best capability-bullet similarity (0-1) from playbook.",
    )


class PlanContext(BaseModel):
    """Full context assembled for the planner in one tool call."""

    blueprint_text: str = Field(
        default="",
        description="Formatted delegation blueprint retrieved for planning, or empty.",
    )
    agents: list[AgentContext] = Field(
        default_factory=list,
        description="Agent packages with cards, playbooks, and examples.",
    )
    unmatched_capabilities: list[str] = Field(
        default_factory=list,
        description="Capabilities for which no agents were found.",
    )
    blueprint_id: str = Field(
        default="",
        description="Blueprint UUID retrieved, for retrieval outcome tracking.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class PlanStep(BaseModel):
    """One step in the execution plan — a scratchpad for the orchestrator's thinking.

    Fields are free-form notes. The plan does not drive delegation;
    the executor delegates freely and updates steps to track progress.
    """

    step_id: str = Field(description="Unique step identifier (e.g. 's1', 's2').")
    agent: str = Field(default="", description="Which agent is planned for this step.")
    instruction: str = Field(default="", description="What this step should do.")
    reasoning: str = Field(default="", description="Why this step is the right next action.")
    status: StepStatus = Field(default=StepStatus.PENDING, description="Step lifecycle state.")
    actual_output: str = Field(default="", description="Summary of what actually happened.")
    comment: str = Field(default="", description="Outcome note — why done, skipped, or failed.")


class Plan(BaseModel):
    """The full execution plan produced by the initial planner."""

    goal: str = Field(
        description=(
            "The overall goal the plan aims to achieve. Incorporate constraints from "
            "planning context, scope from retrieved blueprints, and capabilities of "
            "available agents. Do NOT add specifics about the task that you have not "
            "observed or that are not grounded — keep unverified aspects generic."
        ),
    )
    deliverables: list[str] = Field(
        default_factory=list,
        description=(
            "Concrete, falsifiable outcomes that the TASK requires — not intermediate "
            "evidence or formatting expectations. Each entry describes one thing that "
            "must be true when the task is complete. Do NOT specify data formats, "
            "field names, or evidence structures that agents may not support. "
            "Focus on what the end-user asked for, not on how agents produce it."
        ),
    )
    steps: list[PlanStep] = Field(
        default_factory=list,
        description="Ordered list of planned steps.",
    )


class ToolCallRecord(BaseModel):
    """One internal orchestrator tool call (gather_context, create_plan, etc.)."""

    tool_name: str = Field(description="Name of the tool called.")
    tool_input: str = Field(default="", description="Summary of the arguments.")
    tool_output: str = Field(default="", description="Summary of the result.")


class JudgeDecision(BaseModel):
    """Structured decision returned by the answer judge."""

    reasoning: str = Field(
        description=(
            "Step-by-step evaluation: (1) list every concrete requirement from the TASK, "
            "(2) for each requirement state whether the trajectory provides evidence it was met, "
            "(3) note any mismatch between claimed and actual file paths, targets, or values."
        ),
    )
    accepted: bool = Field(
        description="Whether the proposed answer is supported well enough to send to the user.",
    )
    feedback: str = Field(
        default="",
        description=(
            "If accepted, confirm what was satisfied. If rejected, state which specific "
            "TASK requirement was not met and suggest one concrete recovery action."
        ),
    )
    task_summary: str = Field(
        default="",
        description=(
            "Always produce a 1-2 sentence domain-agnostic summary of what the task "
            "required and which types of agent capabilities were critical for solving it. "
            "This summary is used for post-episode learning."
        ),
    )


class OrchestratorSession(BaseModel):
    """Runtime state for a single task execution.

    Combines mutable per-task state (plan, history, answer) with a
    reference to the shared resources (LLM, stores, config flags).
    Every subsystem receives just ``session`` — no separate ``ctx``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Shared resources (LLM, stores, config) — set once, read-only during execution.
    # SkipValidation keeps the precise type for readers and IDEs while bypassing
    # Pydantic's deep-validation of the LLM/store handles (they are not models).
    ctx: SkipValidation[OrchestratorResources]

    thread_id: str
    task: str
    plan_store: Any = None  # AdaptivePlanStore — Any avoids a circular import for Pydantic
    timeline: list[DelegationExchange | ToolCallRecord] = Field(default_factory=list)
    final_answer: str = ""
    submission_attempts: int = 0
    judge_rejections: int = 0
    judge_force_accepted: bool = False

    context_gathered: bool = False
    task_analysis: str = ""
    judge_task_summary: str = ""
    # Agent discovery — persisted across plan/replan cycles
    seen_agents: set[str] = Field(default_factory=set)
    agent_matched_capabilities: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Per-agent list of capability queries that matched during discovery.",
    )
    retrieved_blueprint_id: str = Field(
        default="",
        description="Blueprint UUID retrieved this episode (for retrieval outcome tracking).",
    )
    retrieved_blueprint_text: str = Field(
        default="",
        description="Formatted blueprint text shown to the executor (for curator context).",
    )
    metrics: EpisodeMetrics | None = None

    @property
    def history(self) -> list[DelegationExchange]:
        """Delegation records only — backward-compatible view of the timeline."""
        return [e for e in self.timeline if isinstance(e, DelegationExchange)]
