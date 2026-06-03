"""Episode-level metrics — aggregated once per episode for thesis analysis."""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from orchestrator.instrumentation.playbook_evolution import (
    PlaybookConsolidationEvent,
    PlaybookDeltaEvent,
)

logger = logging.getLogger(__name__)


@dataclass
class EpisodeMetrics:
    """Aggregated metrics for a single episode — emitted once at episode end."""

    # Timing
    started_at: float = 0.0
    wall_clock_seconds: float = 0.0

    # Token usage — per-model breakdown (from UsageMetadataCallbackHandler)
    # e.g. {"gpt-4o-2024-08-06": {"input_tokens": 500, "output_tokens": 200, "total_tokens": 700}}
    token_usage_by_model: dict[str, dict[str, int]] = field(default_factory=dict)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0

    # Embedding usage
    embedding_calls: int = 0
    embedding_tokens: int = 0

    # Execution shape
    task: str = ""
    success: bool = False
    plan_created: bool = False
    total_delegations: int = 0
    submission_attempts: int = 0
    judge_rejections: int = 0

    # Blueprint retrieval hits (should INCREASE with memory)
    blueprint_matched: bool = False
    blueprint_id: str = ""
    blueprint_similarity: float = 0.0

    # Curation overhead (should DECREASE with memory)
    agent_curations_run: int = 0       # per-agent playbook curations executed
    episode_curation_run: bool = False
    episode_curation_skipped: bool = False
    episode_familiarity: int | None = None

    # Agent selection
    executed_agents: list[str] = field(default_factory=list)   # agents actually delegated to (in order, deduped, excludes judge)
    optimal_sequence: list[str] = field(default_factory=list)  # minimal necessary agents (from curator blueprint; empty if no curation)
    capability_boosted_agents: int = 0  # agents whose score was boosted by capability bullets

    # Agent discovery sources 
    agent_discovery_sources: dict[str, str] = field(default_factory=dict)

    # Agent retrieval log 
    agent_retrieval_log: list[dict[str, Any]] = field(default_factory=list)

    # Playbook memory evolution 
    playbook_confirm_votes: int = 0
    playbook_contradict_votes: int = 0
    playbook_bullets_added: int = 0
    playbook_bullets_pruned: int = 0
    playbook_unconfirmed_prunes: int = 0
    playbook_harm_prunes: int = 0
    playbook_consolidation_merges: int = 0
    playbook_evolution: list[dict[str, Any]] = field(default_factory=list)

    # Judge decisions (for ground truth comparison)
    judge_accepted: bool | None = None
    judge_force_accepted: bool = False

    # Internal: stashed token-usage callback (not serialized)
    _usage_cb: Any = field(default=None, repr=False)
    _embedder: Any = field(default=None, repr=False)

    def to_dict(self) -> dict:
        """Serialize for JSON logging."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__ if not k.startswith("_")}

    def record_playbook_delta(self, event: PlaybookDeltaEvent) -> None:
        """Accumulate one per-agent playbook evolution event."""
        self.playbook_evolution.append(event.model_dump(mode="json"))
        self.playbook_confirm_votes += event.confirm_vote_count
        self.playbook_contradict_votes += event.contradict_vote_count
        self.playbook_bullets_added += event.added_count
        self.playbook_bullets_pruned += event.pruned_count
        for pruned in event.pruned_bullets:
            if pruned.reason == "unconfirmed_contradiction":
                self.playbook_unconfirmed_prunes += 1
            elif pruned.reason == "harm_threshold":
                self.playbook_harm_prunes += 1

    def record_playbook_consolidation(
        self,
        event: PlaybookConsolidationEvent,
    ) -> None:
        """Record LLM-based playbook consolidation as memory evolution."""
        self.playbook_consolidation_merges += event.merged_clusters
        self.playbook_evolution.append(event.model_dump(mode="json"))

    def absorb_token_usage(self, usage_metadata: dict) -> None:
        """Absorb per-model token data from UsageMetadataCallbackHandler.

        Args:
            usage_metadata: dict mapping model name → UsageMetadata
                (with keys input_tokens, output_tokens, total_tokens).
        """
        for model, usage in usage_metadata.items():
            if isinstance(usage, dict):
                inp, out, tot = usage.get("input_tokens", 0), usage.get("output_tokens", 0), usage.get("total_tokens", 0)
            else:
                inp, out, tot = getattr(usage, "input_tokens", 0), getattr(usage, "output_tokens", 0), getattr(usage, "total_tokens", 0)
            self.token_usage_by_model[model] = {
                "input_tokens": inp,
                "output_tokens": out,
                "total_tokens": tot,
            }
            self.total_input_tokens += inp
            self.total_output_tokens += out
            self.total_tokens += tot

    def finalize_and_log(self) -> None:
        """Compute wall_clock, absorb pending token data, emit structured JSON log."""
        self.wall_clock_seconds = round(time.monotonic() - self.started_at, 1)
        # Absorb token usage from stashed callback if present
        cb = self._usage_cb
        if cb is not None and hasattr(cb, "usage_metadata"):
            self.absorb_token_usage(cb.usage_metadata)
        # Absorb embedding usage from stashed embedder if present
        emb = self._embedder
        if emb is not None:
            self.embedding_calls += emb.total_calls
            self.embedding_tokens += emb.total_tokens
            emb.reset_usage()
        summary = self.to_dict()
        summary["task"] = summary["task"][:120]
        summary.pop("started_at", None)
        logger.info("[episode_summary] %s", json.dumps(summary))
