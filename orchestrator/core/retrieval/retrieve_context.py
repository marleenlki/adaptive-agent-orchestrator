"""Context assembly — combines blueprint and agent search into a single
PlanContext for the executor.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from orchestrator.core.retrieval.search_agents import search_agents_for_capabilities
from orchestrator.core.retrieval.search_blueprints import search_blueprint
from orchestrator.core.session_types import AgentContext, PlanContext
from orchestrator.instrumentation.recorder import EpisodeMetricsRecorder
from orchestrator.shared.constants import MEMORY_INJECT_SCORE_CAP

if TYPE_CHECKING:
    from orchestrator.memory.stores import (
        PostgresBlueprintStore,
        PostgresEpisodeStore,
        PostgresPlaybookStore,
    )
    from orchestrator.registration.registry import AgentRegistry

logger = logging.getLogger(__name__)


def construct_context(
    goal: str,
    capabilities: list[str],
    registry: "AgentRegistry",
    episode_store: "PostgresEpisodeStore | None",
    blueprint_store: "PostgresBlueprintStore | None",
    playbook_store: "PostgresPlaybookStore | None",
    embedder=None,
    metrics=None,
    enable_agent_filtering: bool = True,
) -> PlanContext:
    """Build the PlanContext from blueprint search + agent search."""
    goal_emb: list[float] | None = None
    if goal:
        if blueprint_store is not None:
            goal_emb = blueprint_store.embed(goal) or None
        elif episode_store is not None:
            goal_emb = episode_store.embed(goal) or None

    blueprint_text, blueprint_agent_names, blueprint_similarity, blueprint_id = search_blueprint(
        goal, registry, blueprint_store, goal_embedding=goal_emb,
    )

    metrics_recorder = EpisodeMetricsRecorder(metrics)
    metrics_recorder.record_blueprint_retrieval(
        blueprint_text,
        blueprint_id,
        blueprint_similarity,
    )

    agents: list[AgentContext] = []
    unmatched: list[str] = []
    if capabilities:
        agents, unmatched = search_agents_for_capabilities(
            capabilities, registry,
            episode_store=episode_store,
            playbook_store=playbook_store,
            embedder=embedder,
            enable_filtering=enable_agent_filtering,
        )

    bp_scores = {name: blueprint_similarity for name in blueprint_agent_names}
    _boost_memory_agents(agents, bp_scores, registry, "blueprint")
    agents.sort(key=lambda agent: agent.score, reverse=True)

    metrics_recorder.record_agent_retrieval(capabilities, unmatched, agents)

    return PlanContext(
        blueprint_text=blueprint_text,
        agents=agents,
        unmatched_capabilities=unmatched,
        blueprint_id=blueprint_id,
    )


def _boost_memory_agents(
    agents: list[AgentContext],
    memory_agent_scores: dict[str, float],
    registry: "AgentRegistry",
    source: str,
) -> None:
    """Boost agents from a memory signal (blueprint).

    Agents already in the pool have their score raised to the memory
    similarity when it exceeds the current score; missing agents are injected
    with a capped score (MEMORY_INJECT_SCORE_CAP) so they stay reachable
    without displacing capability-matched agents.
    """
    if not memory_agent_scores:
        return

    for agent in agents:
        mem_score = memory_agent_scores.get(agent.name)
        if mem_score is not None and mem_score > agent.score:
            agent.score = mem_score
            agent.source = f"{agent.source}+{source}" if agent.source else source

    found = {a.name for a in agents}
    all_cards = {c["name"]: c for c in registry.list_agents()}
    for name in sorted(set(memory_agent_scores) - found):
        card = all_cards.get(name)
        if card is None:
            logger.debug("Skipping unregistered agent '%s' from %s", name, source)
            continue
        agents.append(AgentContext(
            name=name,
            description=card.get("description", ""),
            skills=[s.get("name", "") for s in card.get("skills", [])],
            score=min(memory_agent_scores[name], MEMORY_INJECT_SCORE_CAP),
            source=source,
            matched_capabilities=[],
        ))
