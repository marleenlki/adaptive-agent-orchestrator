"""Shared base class for PostgreSQL + pgvector stores."""

from __future__ import annotations

from typing import TYPE_CHECKING

from orchestrator.memory.pg_helpers import embed_text

if TYPE_CHECKING:
    from orchestrator.shared.embedder import Embedder


class BaseStore:
    """Connection-pool and embedding shared by all stores."""

    def __init__(
        self,
        conninfo: str,
        embedder: "Embedder | None" = None,
        *,
        pool_min: int = 1,
        pool_max: int = 5,
    ) -> None:
        from psycopg_pool import ConnectionPool

        self._embedder = embedder
        self._pool = ConnectionPool(
            conninfo, min_size=pool_min, max_size=pool_max, open=True,
        )

    def close(self) -> None:
        self._pool.close()

    def embed(self, text: str) -> list[float]:
        """Return an embedding vector for text."""
        return embed_text(self._embedder, text)
