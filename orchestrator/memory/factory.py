"""Factory for creating all PostgreSQL-backed memory stores at once."""

from __future__ import annotations

from orchestrator.memory.stores.episode_store import PostgresEpisodeStore
from orchestrator.memory.stores.blueprint_store import PostgresBlueprintStore
from orchestrator.memory.stores.playbook_store import PostgresPlaybookStore
from orchestrator.memory.stores.trajectory_store import PostgresTrajectoryStore
from orchestrator.memory.setup import apply_search_path, ensure_schema


def create_pg_stores(
    conninfo: str,
    embedder=None,
    *,
    pool_min: int = 1,
    pool_max: int = 5,
    schema: str | None = None,
):
    """Create all PostgreSQL-backed memory stores sharing one connection pool and embedder."""
    if schema is not None:
        ensure_schema(conninfo, schema)
        conninfo = apply_search_path(conninfo, schema)

    episode_store = PostgresEpisodeStore(conninfo, embedder, pool_min=pool_min, pool_max=pool_max)
    blueprint_store = PostgresBlueprintStore(conninfo, embedder, pool_min=pool_min, pool_max=pool_max)
    playbook_store = PostgresPlaybookStore(conninfo, embedder, pool_min=pool_min, pool_max=pool_max)
    trajectory_store = PostgresTrajectoryStore(conninfo, pool_min=pool_min, pool_max=pool_max)
    return episode_store, blueprint_store, playbook_store, trajectory_store
