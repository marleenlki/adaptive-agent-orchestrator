"""PostgreSQL schema lifecycle for memory stores."""

from __future__ import annotations

import logging
import os
import sys
from importlib import resources
from urllib.parse import quote

from psycopg import sql

logger = logging.getLogger(__name__)

MEMORY_TABLES = (
    "stored_episode",
    "stored_episode_step",
    "delegation_blueprint",
    "delegation_blueprint_step",
    "agent_playbook",
    "agent_profile",
    "trajectory",
)


def _get_pool(conninfo: str):
    from psycopg_pool import ConnectionPool

    return ConnectionPool(conninfo, min_size=1, max_size=1, open=True)


def load_schema_sql() -> str:
    """Read the packaged SQL used by Python setup and Docker init."""
    return resources.files(__package__).joinpath("schema.sql").read_text()


def ensure_schema(conninfo: str, schema: str) -> None:
    """Create a PostgreSQL schema and populate it with memory tables."""
    pool = _get_pool(conninfo)
    try:
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                    sql.Identifier(schema),
                ),
            )
            cur.execute(
                sql.SQL("SET search_path TO {}, public").format(
                    sql.Identifier(schema),
                ),
            )
            cur.execute(load_schema_sql())
            conn.commit()
    finally:
        pool.close()
    logger.info("[schema_manager] Schema '%s' ready", schema)


def list_schemas(conninfo: str) -> list[str]:
    """List all user-created schemas."""
    pool = _get_pool(conninfo)
    try:
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT schema_name
                FROM information_schema.schemata
                WHERE schema_name NOT LIKE 'pg_%'
                  AND schema_name != 'information_schema'
                ORDER BY schema_name
                """,
            )
            return [row[0] for row in cur.fetchall()]
    finally:
        pool.close()


def rename_schema(conninfo: str, old_name: str, new_name: str) -> None:
    """Rename a schema, for example to archive an experiment."""
    pool = _get_pool(conninfo)
    try:
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                sql.SQL("ALTER SCHEMA {} RENAME TO {}").format(
                    sql.Identifier(old_name),
                    sql.Identifier(new_name),
                ),
            )
            conn.commit()
    finally:
        pool.close()
    logger.info("[schema_manager] Renamed schema '%s' to '%s'", old_name, new_name)


def schema_stats(conninfo: str, schema: str) -> dict[str, int]:
    """Return row counts for all memory tables in a schema."""
    counts: dict[str, int] = {}
    pool = _get_pool(conninfo)
    try:
        with pool.connection() as conn, conn.cursor() as cur:
            for table in MEMORY_TABLES:
                try:
                    cur.execute(
                        sql.SQL("SELECT count(*) FROM {}.{}").format(
                            sql.Identifier(schema),
                            sql.Identifier(table),
                        ),
                    )
                    counts[table] = cur.fetchone()[0]
                except Exception:
                    conn.rollback()
                    counts[table] = -1
    finally:
        pool.close()
    return counts


def apply_search_path(conninfo: str, schema: str) -> str:
    """Append a PostgreSQL search_path option to a conninfo string."""
    search_path = f"{_quote_search_path_identifier(schema)},public"
    encoded = quote(f"-c search_path={search_path}")
    separator = "&" if "?" in conninfo else "?"
    return f"{conninfo}{separator}options={encoded}"


def _quote_search_path_identifier(identifier: str) -> str:
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def main() -> None:
    """CLI: python -m orchestrator.memory.setup <command> [args]."""
    conninfo = os.environ.get(
        "DATABASE_URL",
        "postgresql://orchestrator:orchestrator@localhost/orchestrator",
    )

    if len(sys.argv) < 2:
        print("Usage: python -m orchestrator.memory.setup <command> [args]")
        print("Commands:")
        print("  list                        List all schemas")
        print("  create <name>               Create a schema with memory tables")
        print("  archive <old> <new>         Rename a schema")
        print("  stats <name>                Show row counts for a schema")
        sys.exit(1)

    command = sys.argv[1]
    if command == "list":
        for schema in list_schemas(conninfo):
            print(schema)
    elif command == "create":
        name = sys.argv[2] if len(sys.argv) > 2 else "exp_v1"
        ensure_schema(conninfo, name)
        print(f"Schema '{name}' created.")
    elif command == "archive":
        if len(sys.argv) < 4:
            print("Usage: archive <old_name> <new_name>")
            sys.exit(1)
        rename_schema(conninfo, sys.argv[2], sys.argv[3])
        print(f"Renamed '{sys.argv[2]}' to '{sys.argv[3]}'")
    elif command == "stats":
        name = sys.argv[2] if len(sys.argv) > 2 else "public"
        for table, count in schema_stats(conninfo, name).items():
            print(f"  {table}: {count}")
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
