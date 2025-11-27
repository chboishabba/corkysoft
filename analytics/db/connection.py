<<<<<<< HEAD
"""Connection and schema helpers for the analytics SQLite database."""
=======
"""Core database connection utilities."""
from __future__ import annotations
>>>>>>> c3ed293 (Remove tracked pycache)
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Optional, Sequence

<<<<<<< HEAD
DEFAULT_DB_PATH = os.environ.get(
    "CORKYSOFT_DB", os.environ.get("ROUTES_DB", "routes.db")
)
=======
DEFAULT_DB_PATH = os.environ.get("CORKYSOFT_DB", os.environ.get("ROUTES_DB", "routes.db"))
>>>>>>> c3ed293 (Remove tracked pycache)


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Return a SQLite connection using WAL mode for better concurrency."""
<<<<<<< HEAD

=======
>>>>>>> c3ed293 (Remove tracked pycache)
    path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


@contextmanager
def connection_scope(db_path: Optional[str] = None):
<<<<<<< HEAD
    """Yield a SQLite connection and close it afterwards."""

=======
    """Context manager that yields a SQLite connection and closes it afterwards."""
>>>>>>> c3ed293 (Remove tracked pycache)
    conn = get_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()


<<<<<<< HEAD
def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    """Return True when ``table`` is present in the SQLite schema."""

    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, table: str) -> Sequence[str]:
    """Return column names for ``table`` preserving declared order."""

    columns = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [column["name"] for column in columns]


def _unique_index_columns(conn: sqlite3.Connection, table: str) -> list[list[str]]:
    """Return lists of columns participating in unique indexes for ``table``."""

    indexes = conn.execute(f"PRAGMA index_list({table})").fetchall()
    unique_columns: list[list[str]] = []
    for index in indexes:
        index_name, is_unique = index["name"], index["unique"]
        if is_unique:
            columns = conn.execute(f"PRAGMA index_info({index_name})").fetchall()
            unique_columns.append([column["name"] for column in columns])
    return unique_columns


def _create_table_if_missing(conn: sqlite3.Connection, ddl: str) -> None:
    """Execute ``ddl`` when the referenced table is absent."""

    conn.executescript(ddl)


def initialize_database(conn: Optional[sqlite3.Connection] = None) -> None:
    """Ensure core dashboard tables and global parameters exist."""

    close_conn = False
    working_conn = conn
    if working_conn is None:
        working_conn = get_connection()
        close_conn = True
    try:
        from .legacy import ensure_dashboard_tables
        from .parameters import ensure_global_parameters_table

        ensure_dashboard_tables(working_conn)
        ensure_global_parameters_table(working_conn)
    finally:
        if close_conn:
            working_conn.close()


__all__ = [
    "DEFAULT_DB_PATH",
    "connection_scope",
    "get_connection",
    "initialize_database",
    "_create_table_if_missing",
    "_table_columns",
    "_table_exists",
    "_unique_index_columns",
]
=======
def _table_columns(conn: sqlite3.Connection, table: str) -> Sequence[str]:
    """Return the column names for *table* in the current connection."""
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [row[1] for row in rows]


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    """Return True if the given table exists in the database."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None
>>>>>>> c3ed293 (Remove tracked pycache)
