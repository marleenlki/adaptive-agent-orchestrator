"""PostgreSQL-backed delegation blueprint store (append-only).
"""

from __future__ import annotations

import dataclasses
import json
import logging

from orchestrator.shared.constants import (
    BLUEPRINT_CONFIRMATION_CAP,
    BLUEPRINT_HARM_THRESHOLD,
    MEMORY_SEARCH_MIN_SIMILARITY,
)
from orchestrator.memory.base_store import BaseStore
from orchestrator.memory.records import (
    BlueprintRecord,
    DelegationBlueprint,
    DelegationStep,
)
from orchestrator.memory.pg_helpers import embed_text, vec_literal

logger = logging.getLogger(__name__)


class PostgresBlueprintStore(BaseStore):
    """Append-only PostgreSQL store for delegation blueprints."""

    def add(self, record: BlueprintRecord) -> None:
        """Insert a blueprint from one successful episode"""
        if record.blueprint is None:  
            return

        if not record.task_embedding:
            record.task_embedding = embed_text(self._embedder, record.task)

        task_emb_str = vec_literal(record.task_embedding) 
        blueprint_json = json.dumps(dataclasses.asdict(record.blueprint))

        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO delegation_blueprint
                    (task_summary, blueprint, agents_involved, task_embedding)
                VALUES (%s, %s::jsonb, %s, %s::vector)
                RETURNING id
                """,
                (
                    record.task,
                    blueprint_json,
                    record.agents_involved,
                    task_emb_str,
                ),
            )
            blueprint_id = cur.fetchone()[0]

            for idx, step in enumerate(record.blueprint.steps):
                does_emb = embed_text(self._embedder, step.does)
                does_emb_str = vec_literal(does_emb) if does_emb else None
                cur.execute(
                    """
                    INSERT INTO delegation_blueprint_step
                        (blueprint_id, step_index, agent, does,
                         receives, produces, does_embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
                    """,
                    (
                        blueprint_id, idx, step.agent, step.does,
                        step.receives, step.produces, does_emb_str,
                    ),
                )

            conn.commit()
            logger.info(
                "[blueprint_store] Stored blueprint_id=%s",
                blueprint_id,
            )

    def record_outcome(self, blueprint_id: str, success: bool) -> None:
        """Vote on a retrieved blueprint using this episode's judge outcome.
        """
        if not blueprint_id:
            return

        with self._pool.connection() as conn, conn.cursor() as cur:
            if success:
                cur.execute(
                    "UPDATE delegation_blueprint "
                    "SET n_confirmed = LEAST(n_confirmed + 1, %s) " # cap the confirmation count 
                    "WHERE id = %s::uuid RETURNING n_confirmed, n_contradicted",
                    (BLUEPRINT_CONFIRMATION_CAP, blueprint_id),
                )
            else:
                cur.execute(
                    "UPDATE delegation_blueprint "
                    "SET n_contradicted = n_contradicted + 1 "
                    "WHERE id = %s::uuid RETURNING n_confirmed, n_contradicted",
                    (blueprint_id,),
                )

            row = cur.fetchone()
            if row is None:
                return  

            n_confirmed, n_contradicted = row
            if n_contradicted >= BLUEPRINT_HARM_THRESHOLD:
                cur.execute(
                    "DELETE FROM delegation_blueprint WHERE id = %s::uuid",
                    (blueprint_id,),
                )
                logger.info(
                    "[blueprint_store] Pruned blueprint_id=%s (confirmed=%d contradicted=%d)",
                    blueprint_id, n_confirmed, n_contradicted,
                )
            conn.commit()

    def update_blueprint(self, blueprint_id: str, record: BlueprintRecord) -> bool:
        """Refine an existing blueprint in place: replace its content and reset
        the contradiction count (the old contradictions applied to the old text).
        """
        if record.blueprint is None or not blueprint_id:
            return False

        if not record.task_embedding:
            record.task_embedding = embed_text(self._embedder, record.task)
        task_emb_str = vec_literal(record.task_embedding)
        blueprint_json = json.dumps(dataclasses.asdict(record.blueprint))

        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE delegation_blueprint
                SET task_summary = %s, blueprint = %s::jsonb, agents_involved = %s,
                    task_embedding = %s::vector, n_contradicted = 0
                WHERE id = %s::uuid
                """,
                (
                    record.task, blueprint_json, record.agents_involved,
                    task_emb_str, blueprint_id,
                ),
            )
            if cur.rowcount == 0:
                conn.rollback()
                return False

            cur.execute(
                "DELETE FROM delegation_blueprint_step WHERE blueprint_id = %s::uuid",
                (blueprint_id,),
            )
            for idx, step in enumerate(record.blueprint.steps):
                does_emb = embed_text(self._embedder, step.does)
                does_emb_str = vec_literal(does_emb) if does_emb else None
                cur.execute(
                    """
                    INSERT INTO delegation_blueprint_step
                        (blueprint_id, step_index, agent, does,
                         receives, produces, does_embedding)
                    VALUES (%s::uuid, %s, %s, %s, %s, %s, %s::vector)
                    """,
                    (
                        blueprint_id, idx, step.agent, step.does,
                        step.receives, step.produces, does_emb_str,
                    ),
                )

            conn.commit()
            logger.info("[blueprint_store] Refined blueprint_id=%s", blueprint_id)
        return True

    # Read path

    def retrieve_blueprint(
        self,
        goal: str,
        planned_agents: set[str],
        goal_embedding: list[float] | None = None,
    ) -> tuple[BlueprintRecord, float] | None:
        """Find a delegation chain for a very similar task.

        High embedding threshold + agent pool overlap.
        Returns 0 or 1 result plus cosine similarity.
        """
        query_emb = goal_embedding or embed_text(self._embedder, goal)
        emb_str = vec_literal(query_emb)

        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, task_summary, blueprint, agents_involved,
                       1 - (task_embedding <=> %s::vector) AS sim
                FROM delegation_blueprint
                WHERE task_embedding IS NOT NULL
                  AND 1 - (task_embedding <=> %s::vector) >= %s
                ORDER BY n_confirmed DESC, sim DESC
                LIMIT 10
                """,
                (emb_str, emb_str, MEMORY_SEARCH_MIN_SIMILARITY),
            )
            rows = cur.fetchall()

        if not rows:
            return None

        for (bp_id, task_summary, blueprint_json, agents, sim) in rows:
            memo_agents = set(agents or [])
            if not memo_agents.issubset(planned_agents):
                continue
            blueprint = self._parse_blueprint(blueprint_json)
            return (
                BlueprintRecord(
                    task=task_summary or "",
                    blueprint=blueprint,
                    agents_involved=list(agents or []),
                    blueprint_id=str(bp_id),
                ),
                float(sim or 0.0),
            )

        return None

    # Helpers

    @staticmethod
    def _parse_blueprint(blueprint_json) -> DelegationBlueprint | None:
        """Deserialize JSONB → DelegationBlueprint."""
        if blueprint_json is None:
            return None
        bp_data = blueprint_json if isinstance(blueprint_json, dict) else json.loads(blueprint_json)
        return DelegationBlueprint(
            steps=[DelegationStep(**s) for s in bp_data.get("steps", [])]        )
