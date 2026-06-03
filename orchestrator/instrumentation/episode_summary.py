"""Populate and emit per-episode metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from orchestrator.instrumentation.recorder import EpisodeMetricsRecorder

if TYPE_CHECKING:
    from orchestrator.core.session_types import OrchestratorSession


def populate_metrics(session: "OrchestratorSession", success: bool) -> None:
    """Fill episode metrics from the final session state and log them."""
    recorder = EpisodeMetricsRecorder(session.metrics)
    recorder.record_episode_result(session, success)
    recorder.finalize()
