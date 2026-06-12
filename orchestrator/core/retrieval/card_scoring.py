"""Cosine-similarity scoring for registered agent cards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from orchestrator.memory.pg_helpers import embed_text

if TYPE_CHECKING:
    from orchestrator.shared.embedder import Embedder


@dataclass
class AgentCardScore:
    name: str
    score: float


# (agent_name, card-text hash) -> embedding. Keying on the text hash means a
# changed card yields a fresh entry instead of silently reusing a stale embedding.
_card_embedding_cache: dict[tuple[str, int], list[float]] = {}


def score_agents_by_card(
    query: str,
    agent_cards: list[dict],
    embedder: "Embedder | None",
    query_embedding: list[float] | None = None,
) -> list[AgentCardScore]:
    """Score agents by cosine similarity against their card text."""
    query_emb = (
        list(query_embedding)
        if query_embedding is not None and len(query_embedding) > 0
        else embed_text(embedder, query)
    )
    if not query_emb:
        return [AgentCardScore(name=c.get("name", ""), score=0.0) for c in agent_cards]

    query_vec = np.array(query_emb, dtype=np.float32)
    scored: list[AgentCardScore] = []

    for card in agent_cards:
        name = card.get("name", "")
        text = _card_text(card)
        card_emb = _cached_embed(
            name,
            text,
            embedder,
            precomputed_embedding=card.get("card_embedding"),
        )
        card_vec = np.array(card_emb, dtype=np.float32) if card_emb else None
        scored.append(
            AgentCardScore(
                name=name,
                score=_cosine_similarity(query_vec, card_vec),
            ),
        )

    return sorted(scored, key=lambda e: e.score, reverse=True)


def _card_text(card: dict) -> str:
    parts = [card.get("name", ""), card.get("description", "")]
    for skill in card.get("skills", []):
        if isinstance(skill, dict):
            parts.append(skill.get("name", ""))
            parts.append(skill.get("description", ""))
        else:
            parts.append(str(skill))
    return " ".join(p for p in parts if p)


def _cached_embed(
    agent_name: str,
    text: str,
    embedder: "Embedder | None",
    *,
    precomputed_embedding: list[float] | None = None,
) -> list[float]:
    key = (agent_name, hash(text))
    if key in _card_embedding_cache:
        return _card_embedding_cache[key]

    embedding = (
        list(precomputed_embedding)
        if precomputed_embedding is not None
        else []
    ) or embed_text(embedder, text)
    if embedding:
        _card_embedding_cache[key] = embedding
    return embedding


def _cosine_similarity(
    query_vec: np.ndarray,
    card_vec: np.ndarray | None,
) -> float:
    if card_vec is None or query_vec.size == 0 or card_vec.size == 0:
        return 0.0
    if query_vec.shape != card_vec.shape:
        return 0.0

    denominator = float(np.linalg.norm(query_vec) * np.linalg.norm(card_vec))
    if denominator <= 0.0:
        return 0.0

    return max(0.0, min(1.0, float(np.dot(query_vec, card_vec) / denominator)))
