"""PostgreSQL-backed playbook store.

Uses the ``agent_playbook`` table. The store manages its own connection pool.
"""

from __future__ import annotations

import json
import logging

from orchestrator.shared.constants import (
    AGENT_SEARCH_MIN_SCORE,
    PLAYBOOK_CONFIRMATION_CAP,
    PLAYBOOK_HARM_THRESHOLD,
)
from orchestrator.memory.base_store import BaseStore
from orchestrator.memory.pg_helpers import embed_text, parse_embedding, vec_literal
from orchestrator.memory.records import (
    PlaybookDeltaOutput,
    PlaybookBullet,
)


logger = logging.getLogger(__name__)

_SECTION_LABELS = {
    "capability": "Capabilities",
    "strategy": "Rules & Strategies",
    "limitation": "Limitations",
}
SECTIONS: list[str] = ["capability", "strategy", "limitation"]

# Stable ordering shared by every query that maps bullets to positional
# display IDs (agent-1, agent-2, …). The rendering query and the
# delta-application query MUST order identically, or the IDs drift apart.
_BULLET_ORDER_BY = (
    "ORDER BY CASE section "
    + " ".join(f"WHEN '{s}' THEN {i}" for i, s in enumerate(SECTIONS))
    + " END, n_confirmed DESC, rule ASC, id ASC"
)


def _bullet_display_id(agent: str, index: int) -> str:
    return f"{agent}-{index}"


def _render_playbook(bullets: list[PlaybookBullet]) -> str:
    if not bullets:
        return ""

    lines: list[str] = []
    current_section = ""
    for index, bullet in enumerate(bullets, start=1):
        if bullet.section != current_section:
            current_section = bullet.section
            label = _SECTION_LABELS.get(current_section, f"{current_section.title()}s")
            lines.append(f"### {label}")

        confirmed = ""
        if bullet.n_confirmed == 0:
            confirmed = " (unconfirmed)"
        elif bullet.n_confirmed > 1:
            confirmed = f" (confirmed {bullet.n_confirmed}x)"

        contradicted = (
            f" (contradicted {bullet.n_contradicted}x)"
            if bullet.n_contradicted > 0 else ""
        )
        bullet_id = bullet.bullet_id or _bullet_display_id(bullet.agent, index)
        lines.append(f"[{bullet_id}] {bullet.rule}{confirmed}{contradicted}")

    return "\n".join(lines)


class PostgresPlaybookStore(BaseStore):
    """PostgreSQL + pgvector backed playbook store."""

    # Read

    def get_playbook(self, agent: str) -> list[PlaybookBullet]:
        """Return ALL bullets for an agent, ordered by section then n_confirmed desc."""
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT agent, section, rule, n_confirmed, n_contradicted
                FROM agent_playbook
                WHERE agent = %s
                {_BULLET_ORDER_BY}
                """,
                (agent,),
            )
            return [
                PlaybookBullet(
                    agent=row[0], section=row[1], rule=row[2],
                    n_confirmed=row[3], n_contradicted=row[4],
                    bullet_id=_bullet_display_id(agent, idx),
                )
                for idx, row in enumerate(cur.fetchall(), start=1)
            ]

    def get_formatted_playbook(self, agent: str) -> str:
        """Render the playbook as numbered markdown grouped by section."""
        return _render_playbook(self.get_playbook(agent))

    def get_confirmed_playbook(self, agent: str) -> str:
        """Render only confirmed bullets (n_confirmed > 0) as markdown."""
        confirmed = [b for b in self.get_playbook(agent) if b.n_confirmed > 0]
        return _render_playbook(confirmed)

    def search_capability_bullets(
        self,
        query_embedding: list[float],
        min_similarity: float = AGENT_SEARCH_MIN_SCORE,
        top_k: int = 5,
    ) -> list[tuple[str, float, int]]:
        """Find agents with capability bullets matching the query.

        Returns a list of ``(agent_name, best_similarity, n_confirmed)``
        tuples — one entry per agent (best matching bullet wins).
        """
        emb_str = vec_literal(query_embedding)
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT agent,
                       MAX(1 - (embedding <=> %s::vector)) AS best_sim,
                       MAX(n_confirmed) AS max_confirmed
                FROM agent_playbook
                WHERE section = 'capability'
                  AND embedding IS NOT NULL
                  AND 1 - (embedding <=> %s::vector) >= %s
                GROUP BY agent
                ORDER BY best_sim DESC
                LIMIT %s
                """,
                (emb_str, emb_str, min_similarity, top_k),
            )
            return [(row[0], float(row[1]), int(row[2])) for row in cur.fetchall()]

    # Write

    # -- profiling helpers ---------------------------------------------

    def is_profiled(self, agent: str) -> bool:
        """Check whether an agent has a profiling record."""
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM agent_profile WHERE agent = %s",
                (agent,),
            )
            return cur.fetchone() is not None

    def mark_profiled(
        self,
        agent: str,
        queries_tested: list[str],
        scores_before: dict[str, float] | None = None,
        scores_after: dict[str, float] | None = None,
    ) -> None:
        """Insert or update the profiling record for an agent."""
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO agent_profile
                    (agent, queries_tested, scores_before, scores_after)
                VALUES (%s, %s::jsonb, %s::jsonb, %s::jsonb)
                ON CONFLICT (agent) DO UPDATE
                    SET profiled_at    = now(),
                        queries_tested = EXCLUDED.queries_tested,
                        scores_before  = EXCLUDED.scores_before,
                        scores_after   = EXCLUDED.scores_after
                """,
                (
                    agent,
                    json.dumps(queries_tested),
                    json.dumps(scores_before or {}),
                    json.dumps(scores_after or {}),
                ),
            )
            conn.commit()

    def insert_profiled_bullets(
        self, agent: str, rules: list[str],
    ) -> int:
        """Insert profiled capability bullets with ``n_confirmed = 0``.

        Profiled bullets are hypotheses generated from card analysis,
        not from observed behaviour.  Setting ``n_confirmed = 0``
        distinguishes them from observation-backed bullets (≥ 1).

        Deduplicates against existing bullets.  Returns the number of
        bullets actually inserted.
        """
        if not rules:
            return 0

        with self._pool.connection() as conn, conn.cursor() as cur:
            existing_embs = self._load_existing_embeddings(cur, agent)
            inserted = 0
            for rule in rules:
                rule = rule.strip()
                if not rule:
                    continue
                if self._insert_bullet_if_new(
                    cur, agent, "capability", rule, n_confirmed=0,
                    existing_embs=existing_embs,
                ):
                    inserted += 1
            conn.commit()

        logger.info(
            "[pg_playbook] Profiled %d capability bullets for '%s'",
            inserted, agent,
        )
        return inserted

    def apply_delta(self, agent: str, delta: PlaybookDeltaOutput) -> None:
        """Apply incremental changes to an agent's playbook."""
        with self._pool.connection() as conn, conn.cursor() as cur:
            # Fetch current bullets (ordered) to map stable display IDs.
            cur.execute(
                f"""
                SELECT id, section, embedding, rule
                FROM agent_playbook
                WHERE agent = %s
                {_BULLET_ORDER_BY}
                """,
                (agent,),
            )
            rows = cur.fetchall()
            id_map = {
                _bullet_display_id(agent, idx): row[0]
                for idx, row in enumerate(rows, start=1)
            }

            # Confirm (capped to prevent irremovable bullets)
            for bullet_id in delta.confirmed_ids:
                row_id = id_map.get(bullet_id)
                if row_id:
                    cur.execute(
                        "UPDATE agent_playbook SET n_confirmed = LEAST(n_confirmed + 1, %s), "
                        "last_seen = now() WHERE id = %s",
                        (PLAYBOOK_CONFIRMATION_CAP, row_id),
                    )

            # Contradict (unconfirmed → immediate removal, confirmed → harm threshold)
            for bullet_id in delta.contradicted_ids:
                row_id = id_map.get(bullet_id)
                if row_id:
                    cur.execute(
                        "UPDATE agent_playbook SET n_contradicted = n_contradicted + 1, "
                        "last_seen = now() WHERE id = %s RETURNING n_confirmed, n_contradicted",
                        (row_id,),
                    )
                    result = cur.fetchone()
                    if result and (result[0] == 0 or result[1] >= PLAYBOOK_HARM_THRESHOLD):
                        cur.execute(
                            "DELETE FROM agent_playbook WHERE id = %s",
                            (row_id,),
                        )
                        logger.info(
                            "[pg_playbook] Auto-pruned %s bullet id=%s",
                            "unconfirmed" if result[0] == 0 else "harmful",
                            row_id,
                        )

            # Add new (with dedup)
            existing_embs = [
                parse_embedding(emb_val)
                for _, _, emb_val, _ in rows
                if emb_val is not None
            ]
            for item in delta.new_bullets:
                if not item.rule or item.section not in SECTIONS:
                    continue
                self._insert_bullet_if_new(
                    cur, agent, item.section, item.rule,
                    n_confirmed=1, existing_embs=existing_embs,
                )

            conn.commit()

    # Consolidation primitives (orchestrated by core.curation)

    def fetch_bullets_for_consolidation(self, agent: str) -> list[dict]:
        """Return embedded bullets as dicts for clustering/merging.
        """
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, section, rule, n_confirmed, n_contradicted, embedding
                FROM agent_playbook
                WHERE agent = %s AND embedding IS NOT NULL
                ORDER BY section, n_confirmed DESC
                """,
                (agent,),
            )
            rows = cur.fetchall()
        return [
            {
                "id": row_id, "section": section, "rule": rule,
                "n_confirmed": n_conf, "n_contradicted": n_contra,
                "embedding": parse_embedding(emb_val),
            }
            for row_id, section, rule, n_conf, n_contra, emb_val in rows
        ]

    def replace_cluster(
        self,
        agent: str,
        section: str,
        rule: str,
        n_confirmed: int,
        n_contradicted: int,
        embedding: list[float],
        delete_ids: list,
    ) -> None:
        """Atomically delete the source bullets and insert the merged replacement."""
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "DELETE FROM agent_playbook WHERE id = ANY(%s)",
                (delete_ids,),
            )
            cur.execute(
                "INSERT INTO agent_playbook "
                "(agent, section, rule, n_confirmed, n_contradicted, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s::vector)",
                (agent, section, rule, n_confirmed, n_contradicted, vec_literal(embedding)),
            )
            conn.commit()

    # Private helpers

    def _load_existing_embeddings(self, cur, agent: str) -> list[list[float]]:
        """Fetch all stored embeddings for *agent* from the current cursor."""
        cur.execute(
            "SELECT embedding FROM agent_playbook "
            "WHERE agent = %s AND embedding IS NOT NULL",
            (agent,),
        )
        return [parse_embedding(emb) for (emb,) in cur.fetchall()]

    def _insert_bullet_if_new(
        self,
        cur,
        agent: str,
        section: str,
        rule: str,
        *,
        n_confirmed: int,
        existing_embs: list[list[float]],
    ) -> bool:
        """Embed *rule*, dedup, then insert. Mutates *existing_embs*. Returns True if inserted."""
        new_emb = embed_text(self._embedder, rule)
        cur.execute(
            "INSERT INTO agent_playbook (agent, section, rule, n_confirmed, embedding) "
            "VALUES (%s, %s, %s, %s, %s::vector)",
            (agent, section, rule, n_confirmed, vec_literal(new_emb)),
        )
        existing_embs.append(new_emb)
        return True
