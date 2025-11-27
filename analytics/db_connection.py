"""Connection helpers for the analytics SQLite database."""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Optional


DEFAULT_DB_PATH = os.environ.get("CORKYSOFT_DB", os.environ.get("ROUTES_DB", "routes.db"))


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Return a SQLite connection using WAL mode for better concurrency."""

    path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


@contextmanager
def connection_scope(db_path: Optional[str] = None):
    """Yield a SQLite connection and close it afterwards."""

    conn = get_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()
