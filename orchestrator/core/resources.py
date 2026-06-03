"""OrchestratorResources — shared resource bundle for the executor and tools.

Subsystems receive ``ctx: OrchestratorResources`` for data, config flags,
and memory store handles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from orchestrator.instrumentation.trajectory import AgentCallTracker
    from orchestrator.memory.stores import (
        PostgresBlueprintStore,
        PostgresEpisodeStore,
        PostgresPlaybookStore,
    )
    from orchestrator.registration.registry import AgentRegistry
    from orchestrator.shared.embedder import Embedder


@dataclass(slots=True, frozen=True)
class OrchestratorResources:
    """Shared resources passed to the executor, tools, and finalization.

    Built once per orchestrator and never mutated during a task — ``frozen``
    makes that contract enforced rather than merely documented.
    """

    # Core
    llm: BaseChatModel
    curator_llm: BaseChatModel
    judge_llm: BaseChatModel
    registry: AgentRegistry
    call_tracker: AgentCallTracker

    # Config flags
    enable_subagent_memory: bool
    max_judge_rejections: int
    enable_judge: bool = True

    # Embedder — always available, independent of memory stores
    embedder: Embedder | None = None

    # Curation flags — when False, the episode runs read-only (no memory writes)
    memory_writes_enabled: bool = True

    # Agent filtering (False = return all agents without scoring)
    enable_agent_filtering: bool = True

    # Planning (False = remove create_plan, view_plan, update_step, add_step tools)
    enable_planning: bool = True

    # Optional memory stores
    episode_store: PostgresEpisodeStore | None = None
    blueprint_store: PostgresBlueprintStore | None = None
    playbook_store: PostgresPlaybookStore | None = None
