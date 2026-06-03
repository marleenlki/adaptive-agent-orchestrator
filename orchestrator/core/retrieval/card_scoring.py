"""Hybrid scoring for registered agent cards."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from rank_bm25 import BM25Okapi

from orchestrator.memory.pg_helpers import embed_text
from orchestrator.shared.constants import HYBRID_ALPHA

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
    """Score agents by hybrid similarity against their card text."""
    query_emb = (
        list(query_embedding)
        if query_embedding is not None and len(query_embedding) > 0
        else embed_text(embedder, query)
    )
    if not query_emb:
        return [AgentCardScore(name=c.get("name", ""), score=0.0) for c in agent_cards]

    query_vec = np.array(query_emb, dtype=np.float32)
    card_texts = [_card_text(c) for c in agent_cards]
    scored: list[AgentCardScore] = []

    for card, text, bm25_score in zip(
        agent_cards,
        card_texts,
        _bm25_scores(query, card_texts),
    ):
        name = card.get("name", "")
        card_emb = _cached_embed(
            name,
            text,
            embedder,
            precomputed_embedding=card.get("card_embedding"),
        )
        cosine_sim = (
            max(0.0, float(np.dot(query_vec, np.array(card_emb, dtype=np.float32))))
            if card_emb else 0.0
        )
        score = HYBRID_ALPHA * cosine_sim + (1.0 - HYBRID_ALPHA) * bm25_score
        scored.append(AgentCardScore(name=name, score=score))

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


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _bm25_scores(query: str, candidates: list[str]) -> list[float]:
    query_tokens = _tokenize(query)
    if not candidates or not query_tokens:
        return [0.0] * len(candidates)

    bm25 = BM25Okapi([_tokenize(doc) for doc in candidates])
    raw = bm25.get_scores(query_tokens)

    max_score = float(max(raw)) if len(raw) > 0 else 0.0
    if max_score <= 0.0:
        return [0.0] * len(candidates)
    return [float(s) / max_score for s in raw]
