"""Episode finalization pipeline.

After the executor completes, this module records memory, runs episode
curation, consolidates playbooks, and populates metrics.
"""

from __future__ import annotations

import logging

from orchestrator.core.curation.agent_curation import (
    apply_playbook_delta,
    consolidate_playbook,
    run_agent_curation,
)
from orchestrator.core.curation.blueprint_curation import run_blueprint_curation
from orchestrator.core.curation.episode_storage import (
    store_episode_memory,
    store_episode_result,
)
from orchestrator.core.session_types import OrchestratorSession
from orchestrator.instrumentation.episode_summary import populate_metrics
from orchestrator.instrumentation.playbook_evolution import build_playbook_consolidation_event
from orchestrator.instrumentation.recorder import EpisodeMetricsRecorder
from orchestrator.shared.constants import (
    EPISODE_FAMILIARITY_MIN_SEEN,
    JUDGE_STEP_ID,
)

logger = logging.getLogger(__name__)


def finalize_episode(
    session: OrchestratorSession,
    *,
    success_override: bool | None = None,
) -> None:
    """Run the full post-episode finalization pipeline.

    Called by the orchestrator after the executor loop is done
    """
    ctx = session.ctx
    success = (
        success_override
        if success_override is not None
        else session.judge_rejections < ctx.max_judge_rejections
    )

    try:
        if not ctx.memory_writes_enabled:
            EpisodeMetricsRecorder(session.metrics).record_curation_skipped()
            populate_metrics(session, success)
            return

        if not session.plan_store.goal:
            session.plan_store.goal = session.task

        # Agents involved this episode (the judge isn't a real agent).
        agent_names = sorted({
            r.agent for r in session.history if r.agent != JUDGE_STEP_ID
        })
        _curate_agents(session, success, agent_names)
        _curate_episode(session, success, agent_names)
        populate_metrics(session, success)
    except Exception:
        logger.warning("[finalize_episode] Failed — non-critical", exc_info=True)
        populate_metrics(session, success)



def _curate_agents(
    session: OrchestratorSession,
    success: bool,
    agent_names: list[str],
) -> None:
    """Run per-agent curation and record typed playbook evolution events."""
    ctx = session.ctx
    if ctx.playbook_store is None:
        return

    metrics = EpisodeMetricsRecorder(session.metrics)
    for agent_name in agent_names:
        result = run_agent_curation(
            ctx.curator_llm,
            agent_name,
            session,
            playbook_store=ctx.playbook_store,
            episode_store=ctx.episode_store,
            registry=ctx.registry,
            success=success,
        )
        if result is None:
            continue

        event = apply_playbook_delta(
            agent_name,
            result.playbook_delta,
            ctx.playbook_store,
        )
        metrics.record_agent_curation(event)

    for agent_name in agent_names:
        merges = consolidate_playbook(ctx.curator_llm, agent_name, ctx.playbook_store)
        event = build_playbook_consolidation_event(agent_name, merges)
        if event is None:
            continue
        logger.info("[consolidation] %s: merged %d cluster(s)", agent_name, merges)
        metrics.record_playbook_consolidation(event)


def _curate_episode(
    session: OrchestratorSession,
    success: bool,
    agent_names: list[str],
) -> None:
    """Run episode-level blueprint curation when memory gates allow it."""
    ctx = session.ctx
    metrics = EpisodeMetricsRecorder(session.metrics)
    if ctx.blueprint_store is None:
        metrics.record_curation_skipped()
        return

    familiarity = (
        ctx.episode_store.count_similar_episodes(session.task)
        if ctx.episode_store is not None else 0
    )
    logger.info("[episode_familiarity] similar_episodes=%d", familiarity)
    metrics.record_episode_familiarity(familiarity)

    if familiarity >= EPISODE_FAMILIARITY_MIN_SEEN and success:
        logger.info(
            "[blueprint_curation] Skipping — episode familiar (%d >= %d similar episodes)",
            familiarity, EPISODE_FAMILIARITY_MIN_SEEN,
        )
        metrics.record_curation_skipped()
        if ctx.episode_store is not None:
            store_episode_memory(ctx.episode_store, session)
    else:
        _curate_blueprint(session, success, agent_names)


def _curate_blueprint(
    session: OrchestratorSession,
    success: bool,
    agent_names: list[str],
) -> None:
    """Run LLM blueprint curation and persist the result."""
    ctx = session.ctx
    metrics = EpisodeMetricsRecorder(session.metrics)

    try:
        result = run_blueprint_curation(
            ctx.curator_llm,
            session,
            success=success,
            known_agents=set(agent_names),
        )
        if result is None:
            return

        store_episode_result(
            ctx.episode_store,
            ctx.blueprint_store,
            session,
            result,
            success,
        )

        metrics.record_blueprint_curation(result)

    except Exception:
        logger.warning("[blueprint_curation] Failed — non-critical", exc_info=True)
