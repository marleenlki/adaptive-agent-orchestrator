"""Prompt templates for the adaptive orchestrator."""

from orchestrator.prompts.episode_curator import (
    EPISODE_CURATOR_SUCCESS_PROMPT,
)
from orchestrator.prompts.verification_gate import (
    VERIFICATION_GATE_PROMPT,
)
from orchestrator.prompts.playbook_consolidation import (
    PLAYBOOK_MERGE_PROMPT,
)
from orchestrator.prompts.executor import (
    EXECUTOR_PROMPT,
)
from orchestrator.prompts.agent_curator import (
    AGENT_CURATOR_PROMPT,
)

__all__ = [
    "AGENT_CURATOR_PROMPT",
    "VERIFICATION_GATE_PROMPT",
    "EPISODE_CURATOR_SUCCESS_PROMPT",
    "EXECUTOR_PROMPT",
    "PLAYBOOK_MERGE_PROMPT",
]
