"""Per-agent episode-end curator"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from orchestrator.core.curation.types import CuratorOutput
from orchestrator.instrumentation.playbook_evolution import (
    PlaybookDeltaEvent,
    build_playbook_delta_event,
    snapshot_playbook,
)
from orchestrator.memory.pg_helpers import cluster_by_cosine
from orchestrator.memory.records import MergedBullet
from orchestrator.prompts.agent_curator import AGENT_CURATOR_PROMPT
from orchestrator.prompts.playbook_consolidation import PLAYBOOK_MERGE_PROMPT
from orchestrator.shared.constants import (
    PLAYBOOK_CONFIRMATION_CAP,
    PLAYBOOK_CONSOLIDATION_THRESHOLD,
    PLAYBOOK_MERGE_SIMILARITY,
    STEP_FAMILIARITY_MIN_SEEN,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from orchestrator.core.session_types import OrchestratorSession
    from orchestrator.memory.stores import PostgresEpisodeStore, PostgresPlaybookStore
    from orchestrator.memory.records import PlaybookDeltaOutput
    from orchestrator.registration.registry import AgentRegistry

logger = logging.getLogger(__name__)


def run_agent_curation(
    llm: "BaseChatModel",
    agent_name: str,
    session: "OrchestratorSession",
    *,
    playbook_store: "PostgresPlaybookStore | None" = None,
    episode_store: "PostgresEpisodeStore | None" = None,
    registry: "AgentRegistry | None" = None,
    success: bool = True,
) -> CuratorOutput | None:
    """Review all delegations to this agent and update the playbook if needed.

    Skips curation entirely when all delegations were already familiar
    and the episode succeeded.
    """
    agent_records = [r for r in session.history if r.agent == agent_name]

    # How often each delegation has been seen before 
    familiarity = [
        episode_store.count_similar_steps(agent_name, r.instruction) if episode_store else 0
        for r in agent_records
    ]

    # On success, skip when every delegation was already familiar -> nothing new to learn.
    # On failure, always curate so contradictions and mistakes get recorded.
    if success and all(n >= STEP_FAMILIARITY_MIN_SEEN for n in familiarity):
        logger.info("Skipping %s — all steps already familiar", agent_name)
        return None

    discovery_caps = session.agent_matched_capabilities.get(agent_name, [])
    cited = sorted({bullet for r in agent_records for bullet in r.cited_bullets})

    prompt = AGENT_CURATOR_PROMPT.format(
        agent_name=agent_name,
        agent_note=_agent_description(registry, agent_name),
        playbook_bullets=playbook_store.get_formatted_playbook(agent_name) or "(no playbook yet)",
        step_blocks=_format_step_blocks(agent_records, familiarity),
        discovery_note=(
            f"Found via capabilities: {', '.join(discovery_caps)}" if discovery_caps
            else "Found via static card match (no capability bullets matched)"
        ),
        cited_bullets_summary=(
            f"Bullet IDs cited by executor: {', '.join(cited)}" if cited
            else "The executor did not cite any playbook bullets for this agent."
        ),
        task_summary=session.judge_task_summary or "(no task summary available)",
        task_analysis=session.task_analysis or "(no task analysis available)",
    )

    result: CuratorOutput = llm.with_structured_output(CuratorOutput).invoke(prompt)
    return result

def apply_playbook_delta(
    agent_name: str,
    delta: "PlaybookDeltaOutput",
    playbook_store: "PostgresPlaybookStore",
) -> PlaybookDeltaEvent | None:
    """Apply one curator delta to the playbook and return its evolution event."""
    before = snapshot_playbook(playbook_store, agent_name)
    playbook_store.apply_delta(agent_name, delta)
    after = snapshot_playbook(playbook_store, agent_name)
    event = build_playbook_delta_event(agent_name, delta, before, after)
    return event if event.has_signal() else None


def consolidate_playbook(
    llm: "BaseChatModel",
    agent_name: str,
    playbook_store: "PostgresPlaybookStore",
) -> int:
    """Merge semantically similar bullets by section when a playbook grows too large."""
    bullets = playbook_store.fetch_bullets_for_consolidation(agent_name)
    if len(bullets) <= PLAYBOOK_CONSOLIDATION_THRESHOLD:
        return 0

    logger.info(
        "[consolidation] '%s' has %d bullets (threshold %d) — consolidating",
        agent_name, len(bullets), PLAYBOOK_CONSOLIDATION_THRESHOLD,
    )

    by_section: dict[str, list[dict]] = defaultdict(list)
    for bullet in bullets:
        by_section[bullet["section"]].append(bullet)

    total_merges = 0
    for section, section_bullets in by_section.items():
        total_merges += _merge_section(llm, agent_name, section, section_bullets, playbook_store)

    if total_merges:
        logger.info("[consolidation] '%s' complete: %d merges", agent_name, total_merges)
    return total_merges





def _merge_section(
    llm: "BaseChatModel",
    agent_name: str,
    section: str,
    section_bullets: list[dict],
    playbook_store: "PostgresPlaybookStore",
) -> int:
    """Cluster and merge similar bullets within one section. Returns merge count."""
    if len(section_bullets) < 2:
        return 0

    embeddings = [bullet["embedding"] for bullet in section_bullets]
    clusters = cluster_by_cosine(embeddings, PLAYBOOK_MERGE_SIMILARITY)
    merges = 0

    for cluster_indices in clusters:
        if len(cluster_indices) < 2:
            continue

        cluster = [section_bullets[i] for i in cluster_indices]
        bullets_text = "\n".join(
            f"- {bullet['rule']} (confirmed {bullet['n_confirmed']}x, contradicted {bullet['n_contradicted']}x)"
            for bullet in cluster
        )
        prompt = PLAYBOOK_MERGE_PROMPT.format(
            agent=agent_name, section=section, bullets_text=bullets_text,
        )

        try:
            result: MergedBullet = llm.with_structured_output(MergedBullet).invoke(prompt)
        except Exception:
            logger.warning(
                "[consolidation] LLM merge failed for %s/%s cluster of %d",
                agent_name, section, len(cluster), exc_info=True,
            )
            continue

        new_n_confirmed = min(sum(bullet["n_confirmed"] for bullet in cluster), PLAYBOOK_CONFIRMATION_CAP)
        new_n_contradicted = max(bullet["n_contradicted"] for bullet in cluster)
        new_emb = playbook_store.embed(result.rule)
        playbook_store.replace_cluster(
            agent_name, section, result.rule,
            new_n_confirmed, new_n_contradicted, new_emb,
            [bullet["id"] for bullet in cluster],
        )
        merges += 1
        logger.info(
            "[consolidation] Merged %d bullets in %s/%s → '%s'",
            len(cluster), agent_name, section, result.rule[:80],
        )

    return merges


def _agent_description(registry: "AgentRegistry | None", agent_name: str) -> str:
    """Look up an agent's card description."""
    cards = {c["name"]: c for c in registry.list_agents()}
    return cards.get(agent_name, {}).get("description", "(no description)")


def _format_step_blocks(agent_records: list, familiarity: list[int]) -> str:
    """Render one markdown block per delegation for the curator prompt."""
    blocks = []
    for record, n_similar in zip(agent_records, familiarity):
        label = "[familiar]" if n_similar >= STEP_FAMILIARITY_MIN_SEEN else "[novel]"
        status = "success" if record.success else "failed"
        blocks.append(
            f"### Delegation {record.step_number} {label} -- {status}\n"
            f"- Instruction: {record.instruction}\n"
            f"- Reasoning: {record.reasoning or 'none'}\n"
            f"- Output: {(record.actual_output or '')[:500]}"
        )
    return "\n\n".join(blocks)