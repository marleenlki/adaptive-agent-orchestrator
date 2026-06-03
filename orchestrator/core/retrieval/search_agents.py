"""Capability-based agent search."""

from __future__ import annotations

from typing import TYPE_CHECKING

from orchestrator.core.retrieval import card_scoring
from orchestrator.core.session_types import AgentContext
from orchestrator.memory.pg_helpers import embed_text
from orchestrator.shared.constants import (
    AGENT_SEARCH_MIN_SCORE,
    AGENT_SEARCH_TOP_K,
)

if TYPE_CHECKING:
    from orchestrator.memory.stores import PostgresEpisodeStore, PostgresPlaybookStore
    from orchestrator.registration.registry import AgentRegistry
    from orchestrator.shared.embedder import Embedder


def search_agents_for_capabilities(
    capabilities: list[str],
    registry: "AgentRegistry",
    episode_store: "PostgresEpisodeStore | None" = None,
    playbook_store: "PostgresPlaybookStore | None" = None,
    embedder: "Embedder | None" = None,
    enable_filtering: bool = True,
) -> tuple[list[AgentContext], list[str]]:
    """Find matching agents and capability queries with no matches."""
    all_agents = registry.list_agents()

    if not enable_filtering:
        return _unfiltered_agents(all_agents, capabilities), []

    embedder = (
        embedder
        or getattr(episode_store, "_embedder", None)
        or getattr(playbook_store, "_embedder", None)
    )
    if embedder is None:
        return [], list(capabilities)

    name_to_card = {a["name"]: a for a in all_agents}
    best: dict[str, AgentContext] = {}
    unmatched: list[str] = []

    for capability in capabilities:
        hits = _rank_single_capability(
            capability,
            all_agents,
            name_to_card,
            embedder,
            playbook_store,
        )
        if not hits:
            unmatched.append(capability)
            continue

        for hit in hits:
            _merge_hit(best, hit, capability)

    agents = sorted(best.values(), key=lambda a: a.score, reverse=True)
    _attach_playbooks(agents, playbook_store)
    return agents, unmatched


def _rank_single_capability(
    capability: str,
    all_agents: list[dict],
    name_to_card: dict[str, dict],
    embedder: "Embedder",
    playbook_store: "PostgresPlaybookStore | None",
) -> list[AgentContext]:
    capability_embedding = embed_text(embedder, capability)
    card_scores = card_scoring.score_agents_by_card(
        capability,
        all_agents,
        embedder,
        query_embedding=capability_embedding or None,
    )
    capability_scores = {}
    if playbook_store is not None and capability_embedding:
        capability_scores = {
            name: similarity
            for name, similarity, _count
            in playbook_store.search_capability_bullets(capability_embedding)
        }

    hits: list[AgentContext] = []
    seen: set[str] = set()

    for agent_score in card_scores:
        hit = _agent_context(
            agent_score.name,
            name_to_card.get(agent_score.name, {}),
            card_score=agent_score.score,
            capability_score=capability_scores.get(agent_score.name, 0.0),
        )
        if hit is None:
            continue
        hits.append(hit)
        seen.add(agent_score.name)

    for agent_name, capability_score in capability_scores.items():
        if agent_name in seen:
            continue
        hit = _agent_context(
            agent_name,
            name_to_card.get(agent_name, {}),
            card_score=0.0,
            capability_score=capability_score,
        )
        if hit is not None:
            hits.append(hit)

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:AGENT_SEARCH_TOP_K]


def _agent_context(
    agent_name: str,
    card: dict,
    *,
    card_score: float,
    capability_score: float,
) -> AgentContext | None:
    score = max(card_score, capability_score)
    if score < AGENT_SEARCH_MIN_SCORE:
        return None
    return AgentContext(
        name=agent_name,
        description=card.get("description", ""),
        skills=[s.get("name", "") for s in card.get("skills", [])],
        score=score,
        source=_discovery_source(card_score, capability_score),
        matched_capabilities=[],
        capability_score=capability_score,
    )


def _merge_hit(best: dict[str, AgentContext], hit: AgentContext, capability: str) -> None:
    """Keep the higher-scoring hit per agent and accumulate matched capabilities."""
    existing = best.get(hit.name)
    if existing is None:
        winner = hit
    elif hit.score > existing.score:
        hit.matched_capabilities = list(existing.matched_capabilities)
        winner = hit
    else:
        winner = existing

    if capability not in winner.matched_capabilities:
        winner.matched_capabilities.append(capability)
    best[hit.name] = winner


def _attach_playbooks(
    agents: list[AgentContext],
    playbook_store: "PostgresPlaybookStore | None",
) -> None:
    if playbook_store is None:
        return
    for agent in agents:
        agent.playbook = playbook_store.get_confirmed_playbook(agent.name)


def _unfiltered_agents(
    all_agents: list[dict],
    capabilities: list[str],
) -> list[AgentContext]:
    return sorted(
        (
            AgentContext(
                name=card.get("name", ""),
                description=card.get("description", ""),
                skills=[s.get("name", "") for s in card.get("skills", [])],
                score=1.0,
                source="no_filter",
                matched_capabilities=list(capabilities),
            )
            for card in all_agents
        ),
        key=lambda a: a.name,
    )


def _discovery_source(card_score: float, cap_score: float) -> str:
    card_ok = card_score >= AGENT_SEARCH_MIN_SCORE
    cap_ok = cap_score >= AGENT_SEARCH_MIN_SCORE
    if card_ok and cap_ok:
        return "embedding+capability"
    if card_ok:
        return "embedding"
    return "capability"
