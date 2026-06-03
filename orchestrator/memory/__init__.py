"""Memory subsystem: episode familiarity, blueprints, playbooks, trajectories."""

from orchestrator.memory.factory import create_pg_stores
from orchestrator.memory.setup import apply_search_path, ensure_schema
from orchestrator.memory.stores.episode_store import PostgresEpisodeStore
from orchestrator.memory.stores.blueprint_store import PostgresBlueprintStore
from orchestrator.memory.stores.playbook_store import PostgresPlaybookStore
from orchestrator.memory.stores.trajectory_store import PostgresTrajectoryStore
from orchestrator.memory.records import (
    BlueprintRecord,
    DelegationBlueprint,
    DelegationStep,
    StoredEpisode,
    PlaybookBullet,
    StoredEpisodeStep,
    Trajectory,
    TrajectoryStep,
)

__all__ = [
    # Adapters (PostgreSQL implementations)
    "PostgresEpisodeStore",
    "PostgresBlueprintStore",
    "PostgresPlaybookStore",
    "PostgresTrajectoryStore",
    "create_pg_stores",
    "apply_search_path",
    "ensure_schema",
    # Data types
    "BlueprintRecord",
    "DelegationBlueprint",
    "DelegationStep",
    "StoredEpisode",
    "PlaybookBullet",
    "StoredEpisodeStep",
    "Trajectory",
    "TrajectoryStep",
]
