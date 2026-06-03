"""Shared utilities for PostgreSQL + pgvector stores."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator.shared.embedder import Embedder


def vec_literal(vec: list[float]) -> str:
    """Format a vector as a pgvector literal string ``[0.1,0.2,…]``."""
    return "[" + ",".join(f"{v:.8f}" for v in vec) + "]"


def parse_embedding(emb_val) -> list[float]:
    """Parse a pgvector column value (string or sequence) to a float list."""
    if isinstance(emb_val, str):
        return [float(x) for x in emb_val.strip("[]").split(",")]
    return list(emb_val)


def embed_text(embedder: "Embedder | None", text: str) -> list[float]:
    """Embed text, failing fast when no embedder is configured."""
    if not text:
        return []
    if embedder is None:
        raise RuntimeError("Embedding requested but no embedder is configured.")
    return embedder.encode([text])[0].tolist()


def cluster_by_cosine(
    embeddings: list[list[float]], threshold: float,
) -> list[list[int]]:
    """Single-linkage clustering by cosine similarity.

    Returns a list of index-lists (one per cluster).
    Singletons are included — callers filter by cluster size.

    Embeddings are assumed pre-normalised (OpenAI ada), so
    dot-product == cosine similarity.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    import numpy as np

    vecs = np.array(embeddings, dtype=np.float32)
    n = len(vecs)
    if n < 2:
        return [list(range(n))]

    distances = pdist(vecs, metric="cosine")
    Z = linkage(distances, method="single")
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")

    clusters: dict[int, list[int]] = {}
    for idx, lbl in enumerate(labels):
        clusters.setdefault(lbl, []).append(idx)
    return list(clusters.values())
