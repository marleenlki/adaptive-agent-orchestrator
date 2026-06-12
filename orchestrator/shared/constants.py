"""Shared constants used across the orchestrator subsystems.

All tuning parameters, thresholds, limits, and configuration constants
live here.  Individual modules import what they need rather than
defining their own magic numbers.
"""

from enum import StrEnum

# Agent Scoring
AGENT_SEARCH_MIN_SCORE = 0.75
"""Minimum cosine similarity for agent search (card and capability)."""

AGENT_SEARCH_TOP_K = 3
"""Maximum agents returned per capability."""


# Familiarity scoring
EPISODE_FAMILIARITY_MIN_SEEN = 3
"""Skip episode curation when this many similar episodes already exist."""

STEP_FAMILIARITY_MIN_SEEN = 2
"""Skip agent-step curation when this many similar steps already exist."""

INSTRUCTION_MATCH_MIN_SIMILARITY = 0.8
"""Minimum cosine similarity for an agent step to count as similar."""

EPISODE_MATCH_MIN_SIMILARITY = 0.9
"""Minimum cosine similarity for an episode to count as similar."""

# Planning

MEMORY_INJECT_SCORE_CAP = 0.75
"""Max score for agents injected into the pool by memory signals (blueprints)
but lacking any capability match.  Keeps dependency agents reachable without
displacing capability-matched agents (which typically score 0.75-0.88)."""

# Execution

STEP_FAILED_PREFIX = "STEP_FAILED"
"""Sentinel prefix prepended to agent error responses."""

# LangGraph executor exit signals
EXIT_KEY_REQUESTED = "exit_requested"
"""State key: whether the executor requests an early exit."""

# Judge
JUDGE_STEP_ID = "judge"
"""Pseudo step-ID used for judge records in the execution history."""

# Thread
DEFAULT_THREAD_ID = "default"
"""Fallback thread identifier when none is supplied."""

MAX_JUDGE_REJECTIONS = 2
"""Default max judge rejections per execution turn before force-accept."""

RECURSION_LIMIT = 120
"""LangGraph recursion limit for the executor ReAct agent."""

# Memory — retrieval

MEMORY_SEARCH_MIN_SIMILARITY = 0.75
"""Minimum cosine similarity for memory retrieval (blueprints)."""

# Memory — blueprints (delegation chains)

BLUEPRINT_HARM_THRESHOLD = 2
"""Auto-prune threshold: a blueprint with this many contradictions is removed."""

BLUEPRINT_CONFIRMATION_CAP = 5
"""Maximum n_confirmed for a blueprint."""

# Memory — playbooks (agent capability bullets)

PLAYBOOK_HARM_THRESHOLD = 2
"""Auto-prune threshold: a bullet with this many contradictions is removed."""

PLAYBOOK_CONFIRMATION_CAP = 5
"""Maximum n_confirmed for a playbook bullet — prevents runaway confirmation."""

PLAYBOOK_CONSOLIDATION_THRESHOLD = 10
"""Bullet count per agent above which LLM-based consolidation is triggered."""

PLAYBOOK_MERGE_SIMILARITY = 0.9
"""Cosine similarity for grouping bullets into merge clusters (below dedup, above noise)."""

# Agent profiling

PROFILING_N_QUERIES = 20
"""Number of test queries generated per agent during profiling."""

PROFILING_MAX_BULLETS = 3
"""Maximum capability bullets generated per agent during profiling."""

# LLM-facing text truncation

MAX_SHORT_PREVIEW = 1000
"""Short preview truncation limit."""

MAX_LABEL_PREVIEW = 200
"""Label/reasoning preview truncation limit."""


# Delegation source labels

ORCHESTRATOR_DELEGATION_SOURCE = "orchestrator"
"""Source label for orchestrator-to-agent calls shown in agent-level views."""


# Enums — role/status labels


class TrajectoryRole(StrEnum):
    """Role values for trajectory step records."""

    USER = "user"
    ORCHESTRATOR = "orchestrator"
    AGENT = "agent"
    TOOL = "tool"


class StepStatus(StrEnum):
    """Lifecycle states for a plan step."""

    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"
