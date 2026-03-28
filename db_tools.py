"""
tools/db_tools.py
==================
Database tool implementations: SQLite queries, schema inspection,
and a generic key-value store.
All tools self-register into ToolRegistry on import.

Production note: swap SQLite for PostgreSQL/MySQL via SQLAlchemy
by changing the connection string in DB_URL.
"""

from __future__ import annotations
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

from tools.tool_registry import registry

logger = logging.getLogger(__name__)

DB_URL = os.getenv("DB_URL", "./data/platform.db")


def _get_conn() -> sqlite3.Connection:
    Path(DB_URL).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_URL)
    conn.row_factory = sqlite3.Row
    return conn


# ── SQL query ─────────────────────────────────────────────────────────────────
@registry.register_tool(
    name="sql_query",
    description="Execute a read-only SQL SELECT query and return rows as dicts.",
    category="database",
)
def sql_query(query: str, params: tuple = ()) -> list[dict]:
    q = query.strip().upper()
    if not q.startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed via this tool.")
    with _get_conn() as conn:
        cur = conn.execute(query, params)
        rows = cur.fetchall()
        return [dict(row) for row in rows]


# ── Schema inspection ─────────────────────────────────────────────────────────
@registry.register_tool(
    name="list_tables",
    description="List all tables in the database.",
    category="database",
)
def list_tables() -> list[str]:
    with _get_conn() as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        return [r["name"] for r in rows]


@registry.register_tool(
    name="describe_table",
    description="Return column names and types for a given table.",
    category="database",
)
def describe_table(table_name: str) -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()  # noqa: S608
        return [{"column": r["name"], "type": r["type"], "nullable": not r["notnull"]} for r in rows]


# ── Key-value store ───────────────────────────────────────────────────────────
@registry.register_tool(
    name="kv_set",
    description="Store a key-value pair persistently.",
    category="database",
)
def kv_set(key: str, value: str) -> bool:
    with _get_conn() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS kv_store (key TEXT PRIMARY KEY, value TEXT)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)",
            (key, value),
        )
        conn.commit()
    return True


@registry.register_tool(
    name="kv_get",
    description="Retrieve a value by key from the persistent store.",
    category="database",
)
def kv_get(key: str) -> Any:
    with _get_conn() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS kv_store (key TEXT PRIMARY KEY, value TEXT)")
        row = conn.execute("SELECT value FROM kv_store WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None
