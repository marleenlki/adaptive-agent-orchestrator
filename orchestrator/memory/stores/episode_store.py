"""PostgreSQL-backed store for successful episodes used by familiarity gates."""

from __future__ import annotations

import logging

from orchestrator.shared.constants import (
    EPISODE_MATCH_MIN_SIMILARITY,
    INSTRUCTION_MATCH_MIN_SIMILARITY,
)
from orchestrator.memory.base_store import BaseStore
from orchestrator.memory.pg_helpers import embed_text, vec_literal
from orchestrator.memory.records import StoredEpisode

logger = logging.getLogger(__name__)


class PostgresEpisodeStore(BaseStore):
    """Append-only episode memory store."""

    def add(self, record: StoredEpisode) -> None:
        """Store one successful episode."""
        if not record.task_embedding:
            record.task_embedding = embed_text(self._embedder, record.task)

        emb_str = vec_literal(record.task_embedding)

        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO stored_episode
                    (task, task_embedding)
                VALUES (%s, %s::vector)
                RETURNING id
                """,
                (
                    record.task,
                    emb_str,
                ),
            )
            episode_id = cur.fetchone()[0]

            for index, step in enumerate(record.steps):
                cur.execute(
                    """
                    INSERT INTO stored_episode_step
                        (episode_id, step_index, agent,
                         instruction, instruction_embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        episode_id,
                        index,
                        step.agent,
                        step.instruction,
                        vec_literal(step.instruction_embedding)
                        if step.instruction_embedding else None,
                    ),
                )

            conn.commit()
            logger.info(
                "[episode_store] Stored episode (%d steps): %s",
                len(record.steps), record.task[:100],
            )

    def _count_similar(
        self,
        text: str,
        embedding: list[float] | None,
        table: str,
        emb_column: str,
        threshold: float,
        extra_where: str = "",
        extra_params: tuple = (),
    ) -> int:
        """Count rows with similarity >= *threshold*."""
        emb = embedding or embed_text(self._embedder, text)
        emb_str = vec_literal(emb)

        sql = (
            f"SELECT COUNT(*) FROM {table} "
            f"WHERE 1 - ({emb_column} <=> %s::vector) >= %s"
        )
        params: list = [emb_str, threshold]
        if extra_where:
            sql += f" AND {extra_where}"
            params.extend(extra_params)

        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            (n,) = cur.fetchone()

        return n

    def count_similar_steps(
        self,
        agent_name: str,
        instruction: str,
        instruction_embedding: list[float] | None = None,
    ) -> int:
        """Count stored steps similar to *instruction* for this agent."""
        return self._count_similar(
            text=instruction,
            embedding=instruction_embedding,
            table="stored_episode_step",
            emb_column="instruction_embedding",
            threshold=INSTRUCTION_MATCH_MIN_SIMILARITY,
            extra_where="agent = %s",
            extra_params=(agent_name,),
        )

    def count_similar_episodes(
        self,
        task: str,
        task_embedding: list[float] | None = None,
    ) -> int:
        """Count stored episodes similar to *task*."""
        return self._count_similar(
            text=task,
            embedding=task_embedding,
            table="stored_episode",
            emb_column="task_embedding",
            threshold=EPISODE_MATCH_MIN_SIMILARITY,
        )
