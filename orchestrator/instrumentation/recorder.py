"""No-op-safe writer for episode metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from orchestrator.shared.constants import JUDGE_STEP_ID

if TYPE_CHECKING:
    from orchestrator.core.session_types import OrchestratorSession
    from orchestrator.instrumentation.metrics import EpisodeMetrics
    from orchestrator.instrumentation.playbook_evolution import (
        PlaybookConsolidationEvent,
        PlaybookDeltaEvent,
    )


class EpisodeMetricsRecorder:
    """Centralizes writes to optional per-episode metrics."""

    def __init__(self, metrics: "EpisodeMetrics | None") -> None:
        self._metrics = metrics

    def record_blueprint_retrieval(
        self,
        blueprint_text: str,
        blueprint_id: str,
        blueprint_similarity: float,
    ) -> None:
        if self._metrics is None:
            return
        self._metrics.blueprint_matched = bool(blueprint_text)
        self._metrics.blueprint_id = blueprint_id
        self._metrics.blueprint_similarity = blueprint_similarity

    def record_agent_retrieval(
        self,
        capabilities: list[str],
        unmatched: list[str],
        agents: list[Any],
    ) -> None:
        if self._metrics is None:
            return
        self._metrics.capability_boosted_agents = sum(
            1
            for agent in agents
            if agent.capability_score > 0 and agent.capability_score >= agent.score
        )
        self._metrics.agent_retrieval_log.append({
            "call": len(self._metrics.agent_retrieval_log) + 1,
            "capabilities": capabilities,
            "unmatched": unmatched,
            "results": [
                {
                    "agent": agent.name,
                    "score": round(agent.score, 4),
                    "source": agent.source,
                    "capability_score": round(agent.capability_score, 4),
                    "rank": rank,
                    "matched_capabilities": agent.matched_capabilities,
                }
                for rank, agent in enumerate(agents, 1)
            ],
        })

    def record_agent_discovery(self, agent_name: str, source: str) -> None:
        if self._metrics is None:
            return
        self._metrics.agent_discovery_sources[agent_name] = source

    def record_curation_skipped(self) -> None:
        if self._metrics is None:
            return
        self._metrics.episode_curation_skipped = True

    def record_agent_curation(self, event: "PlaybookDeltaEvent | None") -> None:
        if self._metrics is None:
            return
        self._metrics.agent_curations_run += 1
        if event is not None:
            self._metrics.record_playbook_delta(event)

    def record_playbook_consolidation(
        self,
        event: "PlaybookConsolidationEvent | None",
    ) -> None:
        if self._metrics is None or event is None:
            return
        self._metrics.record_playbook_consolidation(event)

    def record_episode_familiarity(self, familiarity: int) -> None:
        if self._metrics is None:
            return
        self._metrics.episode_familiarity = familiarity

    def record_blueprint_curation(self, result: Any) -> None:
        if self._metrics is None:
            return
        self._metrics.episode_curation_run = True
        if result.blueprint is not None:
            self._metrics.optimal_sequence = [
                step.agent for step in result.blueprint.steps
            ]

    def record_force_accept(self) -> None:
        if self._metrics is None:
            return
        self._metrics.judge_force_accepted = True

    def record_episode_result(
        self,
        session: "OrchestratorSession",
        success: bool,
    ) -> None:
        if self._metrics is None:
            return
        plan_store = session.plan_store
        self._metrics.task = session.task
        self._metrics.success = success
        self._metrics.judge_accepted = success
        self._metrics.plan_created = bool(plan_store.steps)
        self._metrics.total_delegations = len(session.history)
        self._metrics.submission_attempts = session.submission_attempts
        self._metrics.judge_rejections = session.judge_rejections
        self._metrics.executed_agents = _deduplicated_executed_agents(session)

    def finalize(self) -> None:
        if self._metrics is not None:
            self._metrics.finalize_and_log()


def _deduplicated_executed_agents(
    session: "OrchestratorSession",
) -> list[str]:
    seen: set[str] = set()
    executed_agents: list[str] = []
    for record in session.history:
        if record.agent == JUDGE_STEP_ID or record.agent in seen:
            continue
        seen.add(record.agent)
        executed_agents.append(record.agent)
    return executed_agents
