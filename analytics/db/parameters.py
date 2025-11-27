<<<<<<< HEAD
"""Parameter storage helpers backed by ``global_parameters``."""
=======
"""Functions for managing global parameters in the database."""
from __future__ import annotations
>>>>>>> c3ed293 (Remove tracked pycache)
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Iterable, Optional


def ensure_global_parameters_table(conn: sqlite3.Connection) -> None:
<<<<<<< HEAD
    """Ensure the ``global_parameters`` table exists."""

=======
    """Ensure the global_parameters table exists."""
>>>>>>> c3ed293 (Remove tracked pycache)
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
<<<<<<< HEAD
    conn: sqlite3.Connection, key: str, default: Optional[float] = None
) -> Optional[float]:
    """Return the numeric value for ``key`` from ``global_parameters``."""

=======
    conn: sqlite3.Connection,
    key: str,
    default: Optional[float] = None,
) -> Optional[float]:
    """Return the numeric value for *key* from global_parameters."""
>>>>>>> c3ed293 (Remove tracked pycache)
    row = conn.execute(
        "SELECT value_numeric FROM global_parameters WHERE key = ?",
        (key,),
    ).fetchone()
    if row is None:
        return default
    return row[0]


def set_parameter_value(
<<<<<<< HEAD
    conn: sqlite3.Connection, key: str, value: float, description: Optional[str] = None
) -> None:
    """Insert or update a numeric parameter in ``global_parameters``."""

=======
    conn: sqlite3.Connection,
    key: str,
    value: float,
    description: Optional[str] = None,
) -> None:
    """Insert or update a numeric parameter in global_parameters."""
>>>>>>> c3ed293 (Remove tracked pycache)
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
<<<<<<< HEAD
    conn: sqlite3.Connection, defaults: Iterable[tuple[str, float, str]]
) -> None:
    """Ensure default parameter values exist."""

=======
    conn: sqlite3.Connection,
    defaults: Iterable[tuple[str, float, str]],
) -> None:
    """Ensure default parameter values exist."""
>>>>>>> c3ed293 (Remove tracked pycache)
    ensure_global_parameters_table(conn)
    for key, value, description in defaults:
        current = get_parameter_value(conn, key)
        if current is None:
            set_parameter_value(conn, key, value, description)
<<<<<<< HEAD


__all__ = [
    "bootstrap_parameters",
    "ensure_global_parameters_table",
    "get_parameter_value",
    "set_parameter_value",
]
=======
>>>>>>> c3ed293 (Remove tracked pycache)
