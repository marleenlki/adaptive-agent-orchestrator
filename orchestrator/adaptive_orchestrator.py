"""Adaptive orchestrator entry point and main loop."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.language_models import BaseChatModel

from orchestrator.core.execution.plan_store import AdaptivePlanStore
from orchestrator.core.execution.loop import run_executor
from orchestrator.core.curation.pipeline import finalize_episode
from orchestrator.core.resources import OrchestratorResources
from orchestrator.core.session_types import OrchestratorSession
from orchestrator.instrumentation.metrics import EpisodeMetrics
from orchestrator.instrumentation.trajectory import AgentCallTracker
from orchestrator.registration.profiler import profile_agents
from orchestrator.registration.registry import AgentRegistry
from orchestrator.shared.constants import (
    DEFAULT_THREAD_ID,
    MAX_JUDGE_REJECTIONS,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from orchestrator.memory.stores import (
        PostgresBlueprintStore,
        PostgresEpisodeStore,
        PostgresPlaybookStore,
        PostgresTrajectoryStore,
    )


class AdaptiveOrchestrator:
    """ReAct orchestrator with adaptive memory."""

    def __init__(
        self,
        llm: BaseChatModel,
        registry: AgentRegistry,
        trajectory_store: PostgresTrajectoryStore | None = None,
        max_judge_rejections: int = MAX_JUDGE_REJECTIONS,
        enable_subagent_memory: bool = True,
        episode_store: PostgresEpisodeStore | None = None,
        blueprint_store: PostgresBlueprintStore | None = None,
        playbook_store: PostgresPlaybookStore | None = None,
        curator_llm: BaseChatModel | None = None,
        judge_llm: BaseChatModel | None = None,
        # Testing ablations
        enable_playbooks: bool = True,
        enable_blueprints: bool = True,
        enable_trajectory: bool = True,
        enable_judge: bool = True,
        enable_agent_filtering: bool = True,
        enable_planning: bool = True,
        memory_mode: str = "train",
    ) -> None:
        self.registry = registry
        self.call_tracker = AgentCallTracker()
        self._memory_mode = memory_mode

        self.trajectory_store = trajectory_store if enable_trajectory else None

        # Resolve embedder from whichever store is available
        embedder = None
        for store in (episode_store, blueprint_store, playbook_store):
            if store is not None and hasattr(store, "_embedder"):
                embedder = store._embedder
                break

        logger.info(
            "[orchestrator] Memory units: playbooks=%s blueprints=%s trajectory=%s | "
            "judge=%s agent_filtering=%s mode=%s",
            enable_playbooks, enable_blueprints, enable_trajectory,
            enable_judge, enable_agent_filtering, memory_mode,
        )

        self._ctx = OrchestratorResources(
            llm=llm,
            curator_llm=curator_llm or llm,
            judge_llm=judge_llm or curator_llm or llm,
            registry=registry,
            call_tracker=self.call_tracker,
            enable_subagent_memory=enable_subagent_memory,
            max_judge_rejections=max_judge_rejections,
            enable_judge=enable_judge,
            embedder=embedder,
            enable_agent_filtering=enable_agent_filtering,
            enable_planning=enable_planning,
            episode_store=episode_store,
            blueprint_store=blueprint_store if enable_blueprints else None,
            playbook_store=playbook_store if enable_playbooks else None,
            memory_writes_enabled=memory_mode == "train",
        )

        self.last_metrics: EpisodeMetrics | None = None

    def profile_agents(self) -> dict[str, list[str]]:
        """Conduct zero-shot profiling of all registered agents and return the results.
        """
        self.seed_profiled_bullets()
        results = profile_agents(
            registry=self.registry,
            playbook_store=self._ctx.playbook_store,
            llm=self._ctx.curator_llm,
        )
        return {name: r.bullets for name, r in results.items()}

    def seed_profiled_bullets(self) -> dict[str, int]:
        """Insert pre-computed bullets from the registry into the playbook store.
        """
        store = self._ctx.playbook_store
        if store is None:
            return {}

        seeded: dict[str, int] = {}
        for name, bullets in self.registry.agents_with_profiled_bullets().items():
            if store.is_profiled(name):
                continue
            n = store.insert_profiled_bullets(name, bullets)
            store.mark_profiled(name, queries_tested=[], scores_before={}, scores_after={})
            seeded[name] = n
            logger.info("[seed] Seeded %d bullets for '%s' from registration data", n, name)
        return seeded

    def solve(self, task: str, thread_id: str = DEFAULT_THREAD_ID) -> str:
        """Run the full orchestration loop and return the final answer."""
        self.call_tracker.record_message("user", task)

        metrics = EpisodeMetrics(started_at=time.monotonic())
        with get_usage_metadata_callback() as usage_cb:
            session = OrchestratorSession(
                ctx=self._ctx,
                thread_id=thread_id,
                task=task,
                plan_store=AdaptivePlanStore(),
                metrics=metrics,
            )
            metrics._usage_cb = usage_cb
            metrics._embedder = self._ctx.embedder

            answer = run_executor(session, task)

            self.call_tracker.record_message("orchestrator", answer)
            if self.trajectory_store is not None:
                # Bookend the session timeline with the user task and the
                # orchestrator's final answer (recorded on the tracker).
                from orchestrator.instrumentation.trajectory import MessageRecord
                messages = [
                    r for r in self.call_tracker.get_timeline()
                    if isinstance(r, MessageRecord)
                ]
                user_msgs = [m for m in messages if m.role == "user"]
                orch_msgs = [m for m in messages if m.role == "orchestrator"]
                merged = user_msgs[:1] + list(session.timeline) + orch_msgs[-1:]
                self.trajectory_store.save(
                    task=task,
                    timeline=merged,
                    final_response=answer,
                    episode_id=thread_id,
                )
            self.call_tracker.reset(thread_id=thread_id)
            finalize_episode(session)

            self.last_metrics = metrics
            return answer

    def get_run_metadata(self) -> dict[str, Any]:
        return {
            "orchestrator": self.__class__.__name__,
            "memory_mode": self._memory_mode,
            "memory_writes_enabled": self._memory_mode == "train",
            "max_judge_rejections": self._ctx.max_judge_rejections,
        }
