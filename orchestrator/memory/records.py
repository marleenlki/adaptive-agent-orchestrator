"""Shared data types for the memory stores."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field


@dataclass
class StoredEpisodeStep:
    """One stored delegation step used for familiarity checks."""

    agent: str
    instruction: str = ""
    instruction_embedding: list[float] = field(default_factory=list)


@dataclass
class StoredEpisode:
    """One successful episode stored for familiarity checks."""

    task: str
    steps: list[StoredEpisodeStep] = field(default_factory=list)
    task_embedding: list[float] = field(default_factory=list)


@dataclass
class PlaybookBullet:
    """One bullet in an agent's playbook."""

    agent: str
    section: str
    rule: str
    n_confirmed: int = 1
    n_contradicted: int = 0
    embedding: list[float] = field(default_factory=list)
    bullet_id: str = ""


class TrajectoryStep(BaseModel):
    """A single interaction step in the episode trajectory."""

    role: str = Field(
        description="Who produced this message: 'user' | 'orchestrator' | 'agent' | 'tool'",
    )
    agent_name: str | None = Field(
        default=None,
        description="Agent involved (set for orchestrator→agent and agent→orchestrator steps)",
    )
    tool_name: str | None = Field(
        default=None,
        description="Tool involved (set for role='tool' steps)",
    )
    content: str = Field(description="Message content / summary")
    tool_input: str | None = Field(
        default=None,
        description="Full tool input / arguments (set for role='tool' steps)",
    )
    tool_output: str | None = Field(
        default=None,
        description="Full tool output / return value (set for role='tool' steps)",
    )


class Trajectory(BaseModel):
    """Full interaction trajectory for one episode."""

    episode_id: str = Field(default="", description="Shared episode ID for the run")
    timestamp: str = Field(default="", description="ISO-8601 creation timestamp")
    task: str = Field(description="The user's original task")
    steps: list[TrajectoryStep] = Field(default_factory=list, description="Ordered interaction steps")
    final_response: str = Field(default="", description="Orchestrator's final response to the user")


# Playbook delta types (used by step curator → playbook store)


class NewBullet(BaseModel):
    """A new playbook bullet to add."""

    section: str = Field(
        description="One of: capability, strategy, limitation.",
    )
    rule: str = Field(
        description=(
            "A concrete, reusable rule about this agent.\n\n"

            "Format by section:\n"
            "- capability: 'Can <action>' (≤15 words)\n"
            "- limitation: 'Cannot <action/scope>' (≤15 words). "
            "The inverse of a capability — a hard boundary the "
            "agent cannot cross.\n"
            "- strategy: A deep, actionable insight/ tip/ pattern about HOW to "
            "work with this agent effectively. (≤35 words)\n\n"

            "STRATEGY — what to capture:\n"
            "- Recovery patterns: How to recover from or prevent "
            "an observed failure\n"
            "- Input requirements: What the agent needs in the "
            "instruction to succeed (format, context, specificity)\n"
            "- Interaction patterns: When to split tasks, what "
            "order works, what to validate in the output\n"
            "- Workarounds: Non-obvious techniques that make the "
            "agent more reliable\n"
            "Each strategy must be grounded in observed behavior "
            "from this execution — not speculation.\n\n"

            "CAPABILITY/LIMITATION — what to capture:\n"
            "- Think like a search index: a future orchestrator will "
            "embed the capability bullet and compare it against a new "
            "task's required capabilities. Write the bullet so that "
            "cosine similarity is HIGH for tasks that need this agent.\n"
            "- Use the vocabulary a task-description would use, not "
            "internal tool names.\n"
            "- Each capability should cover ONE distinct skill; do not "
            "combine unrelated abilities in one bullet.\n"
            "- Limitations are the inverse: they prevent the agent "
            "from being selected for tasks it cannot handle.\n\n"

            "QUALITY RULES (all sections):\n"
            "- Must help someone who has NEVER seen the current task\n"
            "- Must reference concrete command, tool, format, or behavior\n"
            "- Must NOT mention task-specific names, paths, or entities\n"
            "- Must NOT be replaceable by 'give clear instructions'"
        ),
    )


class PlaybookDeltaOutput(BaseModel):
    """Curator's incremental playbook update."""

    confirmed_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Bullet IDs from the current playbook that were actively "
            "relevant in this execution and confirmed as helpful. Only include bullets "
            "the agent demonstrably relied on or matched. Do NOT confirm "
            "bullets that were merely 'not violated.'"
        ),
    )
    contradicted_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Bullet IDs from the current agent's playbook that this execution "
            "disproves. This means the agent behaved differently than the rule "
            "predicts, or the advice was wrong. Consider all bullets that could "
            "be harmful even if not applied in this execution. It's important to "
            "be strict to actively unlearn incorrect assumptions about the agent."
        ),
    )
    new_bullets: list[NewBullet] = Field(
        default_factory=list,
        description=(
            "New bullet to add. Max 2. Apply the bullet quality test to "
            "each: transferable, specific, not redundant with existing "
            "playbook, no task-specific identifiers, no output format bullet. "
            "If nothing transferable was learned, return empty list."
        ),
    )


class MergedBullet(BaseModel):
    """LLM output: one consolidated playbook bullet from a cluster."""

    rule: str = Field(
        description=(
            "A single, comprehensive playbook rule that captures all "
            "actionable information from the source bullets while "
            "removing redundancy. Preserve specific details (commands, "
            "flags, formats, tools). Follow the section's word limit: "
            "capability/limitation ≤15 words, strategy ≤35 words."
        ),
    )


# Delegation Blueprint types


@dataclass
class DelegationStep:
    """One step in a delegation blueprint — describes data flow."""

    agent: str
    does: str        # What this agent should do
    receives: str    # Input + from whom
    produces: str    # Output + for whom


@dataclass
class DelegationBlueprint:
    """An ideal, minimal agent chain for a task type."""

    steps: list[DelegationStep]


@dataclass
class BlueprintRecord:
    """Coordination knowledge extracted from a completed episode."""

    task: str
    blueprint: DelegationBlueprint | None
    agents_involved: list[str] = field(default_factory=list)
    task_embedding: list[float] = field(default_factory=list)
    blueprint_id: str = ""  # DB UUID, populated on retrieval for tracking
    refines_retrieved: bool = False  # curator signal: UPDATE the retrieved blueprint vs INSERT new
