"""Global parameter helpers used across analytics features."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Iterable, Optional


def ensure_global_parameters_table(conn: sqlite3.Connection) -> None:
    """Ensure the ``global_parameters`` table exists."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS global_parameters (
            key TEXT PRIMARY KEY,
            value_numeric REAL,
            value_text TEXT,
            description TEXT,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def get_parameter_value(
    conn: sqlite3.Connection, key: str, default: Optional[float] = None
) -> Optional[float]:
    """Return the numeric value for ``key`` from ``global_parameters``."""

    row = conn.execute(
        "SELECT value_numeric FROM global_parameters WHERE key = ?",
        (key,),
    ).fetchone()
    if row is None:
        return default
    return row[0]


def set_parameter_value(
    conn: sqlite3.Connection, key: str, value: float, description: Optional[str] = None
) -> None:
    """Insert or update a numeric parameter in ``global_parameters``."""

    conn.execute(
        """
        INSERT INTO global_parameters (key, value_numeric, description, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value_numeric = excluded.value_numeric,
            description = COALESCE(excluded.description, global_parameters.description),
            updated_at = excluded.updated_at
        """,
        (key, float(value), description, datetime.now(UTC).isoformat()),
    )
    conn.commit()


def bootstrap_parameters(
    conn: sqlite3.Connection, defaults: Iterable[tuple[str, float, str]]
) -> None:
    """Ensure default parameter values exist."""

    ensure_global_parameters_table(conn)
    for key, value, description in defaults:
        current = get_parameter_value(conn, key)
        if current is None:
            set_parameter_value(conn, key, value, description)
