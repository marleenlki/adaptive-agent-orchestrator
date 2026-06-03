"""Persistence for episode memory and delegation blueprints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from orchestrator.memory.records import StoredEpisodeStep, StoredEpisode
from orchestrator.memory.pg_helpers import embed_text
from orchestrator.shared.constants import JUDGE_STEP_ID

if TYPE_CHECKING:
    from orchestrator.core.session_types import DelegationExchange, OrchestratorSession
    from orchestrator.memory.records import BlueprintRecord
    from orchestrator.shared.embedder import Embedder


def build_stored_episode_steps(
    history: list["DelegationExchange"],
    embedder: "Embedder | None" = None,
) -> list[StoredEpisodeStep]:
    """Turn each non-judge delegation into a stored episode step."""
    return [
        StoredEpisodeStep(
            agent=record.agent,
            instruction=record.instruction,
            instruction_embedding=embed_text(embedder, record.instruction),
        )
        for record in history
        if record.agent != JUDGE_STEP_ID
    ]


def store_episode_memory(
    episode_store,
    session: "OrchestratorSession",
) -> None:
    """Persist the full delegation sequence for the familiarity gate."""
    episode_store.add(StoredEpisode(
        task=session.task,
        steps=build_stored_episode_steps(
            session.history,
            embedder=getattr(episode_store, "_embedder", None),
        ),
    ))


def store_episode_result(
    episode_store,
    blueprint_store,
    session: "OrchestratorSession",
    result: "BlueprintRecord",
    success: bool,
) -> None:
    """Persist the episode result: stored episode (on success) and blueprint."""
    if success and episode_store is not None:
        store_episode_memory(episode_store, session)

    if blueprint_store is not None:
        blueprint_store.add(result)
