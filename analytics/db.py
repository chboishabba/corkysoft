"""Database helpers for analytics features."""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Iterable, Optional, Sequence

DEFAULT_DB_PATH = os.environ.get("CORKYSOFT_DB", os.environ.get("ROUTES_DB", "routes.db"))


_DASHBOARD_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS addresses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_input TEXT NOT NULL,
    normalized TEXT,
    street_number TEXT,
    street_name TEXT,
    street_type TEXT,
    unit_number TEXT,
    city TEXT,
    state TEXT,
    postcode TEXT,
    country TEXT,
    lon REAL,
    lat REAL,
    UNIQUE(normalized, country)
);

CREATE TABLE IF NOT EXISTS historical_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_date TEXT,
    client TEXT,
    corridor_display TEXT,
    price_per_m3 REAL,
    revenue_total REAL,
    revenue REAL,
    volume_m3 REAL,
    volume REAL,
    distance_km REAL,
    final_cost REAL,
    origin TEXT,
    destination TEXT,
    origin_postcode TEXT,
    destination_postcode TEXT,
    origin_address_id INTEGER,
    destination_address_id INTEGER,
    created_at TEXT,
    updated_at TEXT,
    FOREIGN KEY(origin_address_id) REFERENCES addresses(id),
    FOREIGN KEY(destination_address_id) REFERENCES addresses(id)
);

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_date TEXT,
    client TEXT,
    origin TEXT,
    destination TEXT,
    price_per_m3 REAL,
    revenue_total REAL,
    revenue REAL,
    volume_m3 REAL,
    volume REAL,
    distance_km REAL,
    final_cost REAL,
    origin_postcode TEXT,
    destination_postcode TEXT,
    origin_lat REAL,
    origin_lon REAL,
    dest_lat REAL,
    dest_lon REAL,
    updated_at TEXT
);
"""


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Return a SQLite connection using WAL mode for better concurrency."""
    path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


@contextmanager
def connection_scope(db_path: Optional[str] = None):
    """Context manager that yields a SQLite connection and closes it afterwards."""
    conn = get_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()


def ensure_global_parameters_table(conn: sqlite3.Connection) -> None:
    """Ensure the global_parameters table exists."""
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
    conn: sqlite3.Connection,
    key: str,
    default: Optional[float] = None,
) -> Optional[float]:
    """Return the numeric value for *key* from global_parameters."""
    row = conn.execute(
        "SELECT value_numeric FROM global_parameters WHERE key = ?",
        (key,),
    ).fetchone()
    if row is None:
        return default
    return row[0]


def set_parameter_value(
    conn: sqlite3.Connection,
    key: str,
    value: float,
    description: Optional[str] = None,
) -> None:
    """Insert or update a numeric parameter in global_parameters."""
    conn.execute(
        """
        INSERT INTO global_parameters (key, value_numeric, description, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value_numeric = excluded.value_numeric,
            description = COALESCE(excluded.description, global_parameters.description),
            updated_at = excluded.updated_at
        """,
        (key, float(value), description, datetime.utcnow().isoformat()),
    )
    conn.commit()


def bootstrap_parameters(
    conn: sqlite3.Connection,
    defaults: Iterable[tuple[str, float, str]],
) -> None:
    """Ensure default parameter values exist."""
    ensure_global_parameters_table(conn)
    for key, value, description in defaults:
        current = get_parameter_value(conn, key)
        if current is None:
            set_parameter_value(conn, key, value, description)


def ensure_dashboard_tables(conn: sqlite3.Connection) -> None:
    """Create empty dashboard tables so the UI can load before data imports."""

    conn.executescript(_DASHBOARD_SCHEMA_SQL)
    ensure_historical_job_routes_table(conn)
    conn.commit()


def _table_columns(conn: sqlite3.Connection, table: str) -> Sequence[str]:
    """Return the column names for *table* in the current connection."""
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [row[1] for row in rows]


def ensure_historical_job_routes_table(conn: sqlite3.Connection) -> None:
    """Ensure the table storing historical job route GeoJSON exists."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS historical_job_routes (
            historical_job_id INTEGER PRIMARY KEY,
            geojson TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT,
            FOREIGN KEY(historical_job_id) REFERENCES historical_jobs(id) ON DELETE CASCADE
        )
        """
    )
    conn.commit()


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def migrate_geojson_to_routes(conn: sqlite3.Connection) -> None:
    """Move embedded GeoJSON columns into the historical_job_routes table."""

    if not _table_exists(conn, "historical_jobs"):
        return

    ensure_historical_job_routes_table(conn)

    columns = _table_columns(conn, "historical_jobs")
    geojson_column = next(
        (name for name in ("route_geojson", "geojson") if name in columns),
        None,
    )
    if not geojson_column:
        return

    timestamp_sources: list[str] = []
    if "updated_at" in columns:
        timestamp_sources.append("updated_at")
    if "imported_at" in columns:
        timestamp_sources.append("imported_at")
    timestamp_sources.append("datetime('now')")
    created_at_expr = f"COALESCE({', '.join(timestamp_sources)})"
    updated_at_expr = "updated_at" if "updated_at" in columns else "NULL"

    insert_sql = f"""
        INSERT OR IGNORE INTO historical_job_routes (
            historical_job_id, geojson, created_at, updated_at
        )
        SELECT id, {geojson_column}, {created_at_expr}, {updated_at_expr}
        FROM historical_jobs
        WHERE {geojson_column} IS NOT NULL AND TRIM({geojson_column}) != ''
    """
    conn.execute(insert_sql)
    conn.execute(f"ALTER TABLE historical_jobs DROP COLUMN {geojson_column}")
    conn.commit()
