"""Persistence for episode memory and delegation blueprints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from orchestrator.memory.records import StoredEpisodeStep, StoredEpisode
from orchestrator.memory.pg_helpers import embed_text
from orchestrator.memory.stores.blueprint_store import PostgresBlueprintStore
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
    blueprint_store: "PostgresBlueprintStore | None",
    session: "OrchestratorSession",
    result: "BlueprintRecord",
    success: bool,
) -> None:
    """Persist the episode result: stored episode (on success) and blueprint."""
    if success and episode_store is not None:
        store_episode_memory(episode_store, session) # only successful episodes are stored in memory

    if blueprint_store is None:
        return

    # Refine the retrieved blueprint in place when the curator flagged it as an
    # improvement of that same pattern; otherwise store a fresh candidate.
    # update_blueprint returns False if that blueprint was meanwhile pruned —
    # then fall through to add() so the refinement is not lost.
    if result.refines_retrieved and session.retrieved_blueprint_id:
        if blueprint_store.update_blueprint(session.retrieved_blueprint_id, result):
            return
    blueprint_store.add(result)
