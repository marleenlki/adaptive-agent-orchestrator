"""Typed playbook evolution events for episode metrics."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from orchestrator.memory.records import PlaybookDeltaOutput

PruneReason = Literal["unconfirmed_contradiction", "harm_threshold"]


class PlaybookBulletSnapshot(BaseModel):
    """Compact before/after view of one playbook bullet."""

    bullet_id: str = ""
    section: str
    rule: str
    n_confirmed: int = 0
    n_contradicted: int = 0


class NewPlaybookBulletCandidate(BaseModel):
    """Bullet proposed by the curator before store validation and dedup."""

    section: str
    rule: str


class AddedPlaybookBullet(BaseModel):
    """Bullet that actually appeared in the playbook after the delta."""

    bullet_id: str
    section: str
    rule: str


class PrunedPlaybookBullet(BaseModel):
    """Bullet removed by contradiction voting."""

    bullet_id: str
    section: str
    rule: str
    reason: PruneReason
    n_confirmed_before: int
    n_contradicted_before: int
    n_contradicted_after_vote: int


class PlaybookDeltaEvent(BaseModel):
    """One per-agent playbook update event emitted after store application."""

    kind: Literal["playbook_delta"] = "playbook_delta"
    agent: str
    before_count: int
    after_count: int
    confirm_vote_count: int
    contradict_vote_count: int
    confirmed_ids: list[str] = Field(default_factory=list)
    contradicted_ids: list[str] = Field(default_factory=list)
    ignored_confirmed_ids: list[str] = Field(default_factory=list)
    ignored_contradicted_ids: list[str] = Field(default_factory=list)
    new_bullet_candidates: list[NewPlaybookBulletCandidate] = Field(default_factory=list)
    added_count: int = 0
    added_bullets: list[AddedPlaybookBullet] = Field(default_factory=list)
    pruned_count: int = 0
    pruned_bullets: list[PrunedPlaybookBullet] = Field(default_factory=list)

    def has_signal(self) -> bool:
        """Return whether this event contains useful learning evidence."""
        return any((
            self.confirm_vote_count,
            self.contradict_vote_count,
            self.ignored_confirmed_ids,
            self.ignored_contradicted_ids,
            self.new_bullet_candidates,
            self.before_count != self.after_count,
        ))


class PlaybookConsolidationEvent(BaseModel):
    """Event for LLM-based playbook consolidation."""

    kind: Literal["playbook_consolidation"] = "playbook_consolidation"
    agent: str
    merged_clusters: int


def snapshot_playbook(playbook_store, agent_name: str) -> list[PlaybookBulletSnapshot]:
    """Read a JSON-safe playbook snapshot from a store."""
    return [
        PlaybookBulletSnapshot(
            bullet_id=bullet.bullet_id,
            section=bullet.section,
            rule=bullet.rule,
            n_confirmed=bullet.n_confirmed,
            n_contradicted=bullet.n_contradicted,
        )
        for bullet in playbook_store.get_playbook(agent_name)
    ]


def build_playbook_delta_event(
    agent_name: str,
    delta: "PlaybookDeltaOutput",
    before: list[PlaybookBulletSnapshot],
    after: list[PlaybookBulletSnapshot],
) -> PlaybookDeltaEvent:
    """Build one typed event from a curator delta and store before/after state."""
    before_by_id = {bullet.bullet_id: bullet for bullet in before}
    confirmed_ids = _known_ids(delta.confirmed_ids, before_by_id)
    contradicted_ids = _known_ids(delta.contradicted_ids, before_by_id)
    added_bullets = _find_added_bullets(before, after)
    pruned_bullets = _find_pruned_bullets(contradicted_ids, before_by_id, before, after)

    return PlaybookDeltaEvent(
        agent=agent_name,
        before_count=len(before),
        after_count=len(after),
        confirm_vote_count=len(confirmed_ids),
        contradict_vote_count=len(contradicted_ids),
        confirmed_ids=confirmed_ids,
        contradicted_ids=contradicted_ids,
        ignored_confirmed_ids=_unknown_ids(delta.confirmed_ids, before_by_id),
        ignored_contradicted_ids=_unknown_ids(delta.contradicted_ids, before_by_id),
        new_bullet_candidates=[
            NewPlaybookBulletCandidate(section=item.section, rule=item.rule)
            for item in delta.new_bullets
        ],
        added_count=len(added_bullets),
        added_bullets=added_bullets,
        pruned_count=len(pruned_bullets),
        pruned_bullets=pruned_bullets,
    )


def build_playbook_consolidation_event(
    agent_name: str,
    merged_clusters: int,
) -> PlaybookConsolidationEvent | None:
    """Build a consolidation event when consolidation changed the playbook."""
    if merged_clusters <= 0:
        return None
    return PlaybookConsolidationEvent(
        agent=agent_name,
        merged_clusters=merged_clusters,
    )


def _known_ids(
    bullet_ids: list[str],
    before_by_id: dict[str, PlaybookBulletSnapshot],
) -> list[str]:
    return [bullet_id for bullet_id in bullet_ids if bullet_id in before_by_id]


def _unknown_ids(
    bullet_ids: list[str],
    before_by_id: dict[str, PlaybookBulletSnapshot],
) -> list[str]:
    return [bullet_id for bullet_id in bullet_ids if bullet_id not in before_by_id]


def _find_pruned_bullets(
    contradicted_ids: list[str],
    before_by_id: dict[str, PlaybookBulletSnapshot],
    before: list[PlaybookBulletSnapshot],
    after: list[PlaybookBulletSnapshot],
) -> list[PrunedPlaybookBullet]:
    remaining_before = Counter(_bullet_key(bullet) for bullet in before)
    after_counts = Counter(_bullet_key(bullet) for bullet in after)
    pruned_bullets: list[PrunedPlaybookBullet] = []

    for bullet_id in contradicted_ids:
        bullet = before_by_id[bullet_id]
        key = _bullet_key(bullet)
        if after_counts[key] >= remaining_before[key]:
            continue
        pruned_bullets.append(_to_pruned_bullet(bullet))
        remaining_before[key] -= 1

    return pruned_bullets


def _to_pruned_bullet(bullet: PlaybookBulletSnapshot) -> PrunedPlaybookBullet:
    reason: PruneReason = (
        "unconfirmed_contradiction"
        if bullet.n_confirmed == 0
        else "harm_threshold"
    )
    return PrunedPlaybookBullet(
        bullet_id=bullet.bullet_id,
        section=bullet.section,
        rule=bullet.rule,
        reason=reason,
        n_confirmed_before=bullet.n_confirmed,
        n_contradicted_before=bullet.n_contradicted,
        n_contradicted_after_vote=bullet.n_contradicted + 1,
    )


def _find_added_bullets(
    before: list[PlaybookBulletSnapshot],
    after: list[PlaybookBulletSnapshot],
) -> list[AddedPlaybookBullet]:
    remaining_before = Counter(_bullet_key(bullet) for bullet in before)
    added_bullets: list[AddedPlaybookBullet] = []

    for bullet in after:
        key = _bullet_key(bullet)
        if remaining_before[key] > 0:
            remaining_before[key] -= 1
            continue
        added_bullets.append(AddedPlaybookBullet(
            bullet_id=bullet.bullet_id,
            section=bullet.section,
            rule=bullet.rule,
        ))

    return added_bullets


def _bullet_key(bullet: PlaybookBulletSnapshot) -> tuple[str, str]:
    return (bullet.section, bullet.rule)
