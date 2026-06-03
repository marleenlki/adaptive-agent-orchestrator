"""PostgreSQL-backed trajectory store (trajectory table)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Union

from orchestrator.shared.constants import DEFAULT_THREAD_ID, TrajectoryRole
from orchestrator.instrumentation.trajectory import (
    DelegationExchange, MessageRecord, TimelineRecord, ToolRecord,
)
from orchestrator.core.session_types import ToolCallRecord
from orchestrator.memory.base_store import BaseStore
from orchestrator.memory.records import Trajectory, TrajectoryStep

logger = logging.getLogger(__name__)

# Everything that can appear in a merged timeline.
AnyTimelineRecord = Union[TimelineRecord, ToolCallRecord]


def build_trajectory(
    *,
    task: str,
    timeline: list[AnyTimelineRecord],
    final_response: str,
    episode_id: str,
) -> Trajectory:
    """Build a canonical trajectory object from a timeline."""
    steps: list[TrajectoryStep] = []
    has_explicit_messages = any(isinstance(rec, MessageRecord) for rec in timeline)
    if not has_explicit_messages:
        steps.append(TrajectoryStep(role=TrajectoryRole.USER, content=task))

    for rec in timeline:
        if isinstance(rec, MessageRecord):
            steps.append(TrajectoryStep(role=rec.role, content=rec.content))
        elif isinstance(rec, DelegationExchange):
            steps.append(
                TrajectoryStep(
                    role=TrajectoryRole.ORCHESTRATOR,
                    agent_name=rec.agent,
                    content=rec.instruction or "",
                )
            )
            steps.append(
                TrajectoryStep(
                    role=TrajectoryRole.AGENT,
                    agent_name=rec.agent,
                    content=rec.actual_output or "",
                )
            )
        elif isinstance(rec, ToolRecord):
            steps.append(
                TrajectoryStep(
                    role=TrajectoryRole.TOOL,
                    tool_name=rec.tool_name,
                    content=f"Called {rec.tool_name}",
                    tool_input=rec.input,
                    tool_output=rec.output,
                )
            )
        elif isinstance(rec, ToolCallRecord):
            steps.append(
                TrajectoryStep(
                    role=TrajectoryRole.TOOL,
                    tool_name=rec.tool_name,
                    content=f"Called {rec.tool_name}",
                    tool_input=rec.tool_input,
                    tool_output=rec.tool_output,
                )
            )

    if not has_explicit_messages:
        steps.append(TrajectoryStep(role=TrajectoryRole.ORCHESTRATOR, content=final_response or ""))

    return Trajectory(
        episode_id=episode_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        task=task,
        steps=steps,
        final_response=final_response or "",
    )


class PostgresTrajectoryStore(BaseStore):
    """Persist run trajectories in PostgreSQL using a dedicated table."""

    def save(
        self,
        task: str,
        timeline: list[AnyTimelineRecord],
        final_response: str,
        episode_id: str = DEFAULT_THREAD_ID,
    ) -> str | None:
        """Build and persist a trajectory row, returning its id."""
        episode_id = episode_id or DEFAULT_THREAD_ID
        trajectory = build_trajectory(
            task=task,
            timeline=timeline,
            final_response=final_response,
            episode_id=episode_id,
        )
        steps_payload = json.dumps(
            [step.model_dump(mode="json") for step in trajectory.steps],
            ensure_ascii=False,
        )

        with self._pool.connection() as conn, conn.cursor() as cur:
            trajectory_id = self._persist_thread_trajectory(cur, trajectory, steps_payload)

        logger.info("[pg_trajectory] saved trajectory: %s -> %s", trajectory.episode_id, trajectory_id)
        return trajectory_id

    def _persist_thread_trajectory(self, cur, trajectory, steps_payload: str) -> str | None:
        existing_ids = self._find_existing_ids(cur, trajectory.episode_id)
        if not existing_ids:
            return self._insert_trajectory(cur, trajectory, steps_payload)

        keep_id = existing_ids[0]
        if len(existing_ids) > 1:
            self._delete_duplicate_rows(cur, existing_ids[1:])
        self._update_trajectory(cur, keep_id, trajectory, steps_payload)
        return keep_id

    @staticmethod
    def _find_existing_ids(cur, episode_id: str) -> list[str]:
        cur.execute(
            """
            SELECT id::text
            FROM trajectory
            WHERE episode_id = %s
            ORDER BY timestamp DESC, id DESC
            """,
            (episode_id,),
        )
        return [row[0] for row in cur.fetchall()]

    @staticmethod
    def _insert_trajectory(cur, trajectory, steps_payload: str) -> str | None:
        cur.execute(
            """
            INSERT INTO trajectory (episode_id, timestamp, task, steps, final_response)
            VALUES (%s, %s::timestamptz, %s, %s::jsonb, %s)
            RETURNING id::text
            """,
            (
                trajectory.episode_id,
                trajectory.timestamp,
                trajectory.task,
                steps_payload,
                trajectory.final_response,
            ),
        )
        row = cur.fetchone()
        return row[0] if row else None

    @staticmethod
    def _update_trajectory(cur, trajectory_id: str, trajectory, steps_payload: str) -> None:
        cur.execute(
            """
            UPDATE trajectory
            SET timestamp = %s::timestamptz,
                task = %s,
                steps = %s::jsonb,
                final_response = %s
            WHERE id = %s::uuid
            """,
            (
                trajectory.timestamp,
                trajectory.task,
                steps_payload,
                trajectory.final_response,
                trajectory_id,
            ),
        )

    @staticmethod
    def _delete_duplicate_rows(cur, duplicate_ids: list[str]) -> None:
        cur.execute(
            "DELETE FROM trajectory WHERE id = ANY(%s::uuid[])",
            (duplicate_ids,),
        )