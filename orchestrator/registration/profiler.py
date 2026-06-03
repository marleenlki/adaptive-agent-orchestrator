"""Find Agent Card retrieval gaps and seed unconfirmed capability bullets."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from orchestrator.core.retrieval.card_scoring import score_agents_by_card
from orchestrator.memory.pg_helpers import embed_text
from orchestrator.prompts.profiler import (
    CAPABILITY_BULLET_PROMPT,
    QUERY_GENERATION_PROMPT,
)
from orchestrator.shared.constants import (
    AGENT_SEARCH_MIN_SCORE,
    PROFILING_MAX_BULLETS,
    PROFILING_N_QUERIES,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from orchestrator.memory.stores import PostgresPlaybookStore
    from orchestrator.registration.registry import AgentRegistry
    from orchestrator.shared.embedder import Embedder

logger = logging.getLogger(__name__)


class QueryOutput(BaseModel):
    queries: list[str] = Field(
        description="Realistic user queries that should retrieve this agent.",
    )


class BulletOutput(BaseModel):
    bullets: list[str] = Field(
        description="Capability bullets, each starting with 'Can '. ≤15 words each.",
    )


@dataclass
class ProfileResult:
    queries: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    bullets: list[str] = field(default_factory=list)
    scores_before: dict[str, float] = field(default_factory=dict)
    scores_after: dict[str, float] = field(default_factory=dict)


def profile_agents(
    registry: "AgentRegistry",
    playbook_store: "PostgresPlaybookStore | None",
    llm: "BaseChatModel",
    embedder: "Embedder | None" = None,
) -> dict[str, ProfileResult]:
    """Profile every unprofiled registered agent."""
    if playbook_store is None:
        logger.warning("[profiling] No playbook store available — skipping profiling")
        return {}

    embedder = embedder or getattr(playbook_store, "_embedder", None)
    if embedder is None:
        logger.warning("[profiling] No embedder available — skipping profiling")
        return {}

    agent_cards = registry.list_agents()
    results: dict[str, ProfileResult] = {}

    for agent_card in agent_cards:
        agent_name = agent_card["name"]
        if playbook_store.is_profiled(agent_name):
            logger.debug("[profiling] Skipping '%s' — already profiled", agent_name)
            continue

        logger.info("[profiling] Profiling agent '%s'", agent_name)
        result = _profile_agent(agent_card, agent_cards, llm, embedder)
        results[agent_name] = result

        if result.bullets:
            playbook_store.insert_profiled_bullets(agent_name, result.bullets)
            result.scores_after = _score_queries(
                result.queries,
                agent_name,
                agent_cards,
                embedder,
                playbook_store=playbook_store,
            )

        playbook_store.mark_profiled(
            agent_name,
            queries_tested=result.queries,
            scores_before=result.scores_before,
            scores_after=result.scores_after,
        )

    profiled_count = sum(1 for r in results.values() if r.bullets)
    logger.info(
        "[profiling] Profiling complete: %d agents, %d with bullets",
        len(results), profiled_count,
    )
    return results


def _profile_agent(
    agent_card: dict,
    agent_cards: list[dict],
    llm: "BaseChatModel",
    embedder: "Embedder",
) -> ProfileResult:
    agent_name = agent_card["name"]
    skill_names = ", ".join(
        skill.get("name", "").strip()
        for skill in agent_card.get("skills", [])
        if skill.get("name", "").strip()
    ) or "(none listed)"
    queries = _generate_queries(llm, agent_card, skill_names)
    scores_before = _score_queries(queries, agent_name, agent_cards, embedder)
    missed_queries = [
        query for query, score in scores_before.items()
        if score < AGENT_SEARCH_MIN_SCORE
    ]
    result = ProfileResult(
        queries=queries,
        gaps=missed_queries,
        scores_before=scores_before,
    )

    if not missed_queries:
        logger.debug("[profiling] '%s' — no gaps, card is fine", agent_name)
        return result

    result.bullets = _generate_bullets(
        llm, agent_card, skill_names, missed_queries,
    )
    logger.info(
        "[profiling] '%s': %d gaps → %d bullets",
        agent_name, len(missed_queries), len(result.bullets),
    )
    return result


def _generate_queries(
    llm: "BaseChatModel",
    agent_card: dict,
    skill_names: str,
) -> list[str]:
    prompt = QUERY_GENERATION_PROMPT.format(
        agent_name=agent_card["name"],
        agent_description=agent_card.get("description", "(no description)"),
        agent_skills=skill_names,
        n_queries=PROFILING_N_QUERIES,
    )

    try:
        result: QueryOutput = llm.with_structured_output(QueryOutput).invoke(prompt)
        return [query.strip() for query in result.queries if query.strip()]
    except Exception:
        logger.warning(
            "[profiling] Query generation failed for '%s'",
            agent_card["name"],
            exc_info=True,
        )
        return []


def _score_queries(
    queries: list[str],
    agent_name: str,
    agent_cards: list[dict],
    embedder: "Embedder",
    playbook_store: "PostgresPlaybookStore | None" = None,
) -> dict[str, float]:
    return {
        query: round(
            _best_score(query, agent_name, agent_cards, embedder, playbook_store),
            4,
        )
        for query in queries
    }


def _best_score(
    query: str,
    agent_name: str,
    agent_cards: list[dict],
    embedder: "Embedder",
    playbook_store: "PostgresPlaybookStore | None" = None,
) -> float:
    card_score = next(
        (
            score.score
            for score in score_agents_by_card(query, agent_cards, embedder)
            if score.name == agent_name
        ),
        0.0,
    )
    capability_score = 0.0
    if playbook_store is not None:
        query_embedding = embed_text(embedder, query)
        if query_embedding:
            capability_score = next(
                (
                    similarity
                    for name, similarity, _count
                    in playbook_store.search_capability_bullets(query_embedding)
                    if name == agent_name
                ),
                0.0,
            )
    return max(card_score, capability_score)


def _generate_bullets(
    llm: "BaseChatModel",
    agent_card: dict,
    skill_names: str,
    missed_queries: list[str],
) -> list[str]:
    prompt = CAPABILITY_BULLET_PROMPT.format(
        agent_name=agent_card["name"],
        agent_description=agent_card.get("description", "(no description)"),
        agent_skills=skill_names,
        gap_queries="\n".join(f"- {query}" for query in missed_queries),
        n_bullets=PROFILING_MAX_BULLETS,
    )

    try:
        result: BulletOutput = llm.with_structured_output(BulletOutput).invoke(prompt)
        bullets = [bullet.strip() for bullet in result.bullets if bullet.strip()]
        return bullets[:PROFILING_MAX_BULLETS]
    except Exception:
        logger.warning(
            "[profiling] Bullet generation failed for '%s'",
            agent_card["name"],
            exc_info=True,
        )
        return []
