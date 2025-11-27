"""Database helpers for analytics features."""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import IO, Iterable, Optional, Sequence

import pandas as pd

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
    client_id INTEGER,
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
    client_id INTEGER,
    origin TEXT,
    destination TEXT,
    origin_resolved TEXT,
    destination_resolved TEXT,
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
    route_geojson TEXT,
    internal_cost_total REAL DEFAULT 0,
    internal_cost_updated_at TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS inventory_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    quantity INTEGER NOT NULL DEFAULT 0,
    unit TEXT DEFAULT 'unit',
    updated_at TEXT,
    UNIQUE(name)
);

CREATE TABLE IF NOT EXISTS workers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    role TEXT DEFAULT '',
    phone TEXT DEFAULT '',
    rate REAL,
    tickets INTEGER,
    active INTEGER NOT NULL DEFAULT 1,
    hired_at TEXT,
    updated_at TEXT,
    UNIQUE(name)
);

CREATE TABLE IF NOT EXISTS trucks (
    truck_id TEXT PRIMARY KEY,
    name TEXT,
    capacity_m3 REAL,
    active INTEGER NOT NULL DEFAULT 1,
    notes TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS shipments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER,
    historical_job_id INTEGER,
    inventory_item_id INTEGER,
    truck_id TEXT,
    worker_id INTEGER,
    status TEXT NOT NULL DEFAULT 'planned',
    scheduled_date TEXT,
    delivered_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT,
    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE SET NULL,
    FOREIGN KEY(historical_job_id) REFERENCES historical_jobs(id) ON DELETE SET NULL,
    FOREIGN KEY(inventory_item_id) REFERENCES inventory_items(id) ON DELETE SET NULL,
    FOREIGN KEY(truck_id) REFERENCES trucks(truck_id) ON DELETE SET NULL,
    FOREIGN KEY(worker_id) REFERENCES workers(id) ON DELETE SET NULL
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
        (key, float(value), description, datetime.now(UTC).isoformat()),
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
    hist_columns = _table_columns(conn, "historical_jobs")
    if "client_id" not in hist_columns:
        conn.execute("ALTER TABLE historical_jobs ADD COLUMN client_id INTEGER")

    job_columns = _table_columns(conn, "jobs")
    column_declarations = {
        "client_id": "INTEGER",
        "origin_resolved": "TEXT",
        "destination_resolved": "TEXT",
        "route_geojson": "TEXT",
        "internal_cost_total": "REAL DEFAULT 0",
        "internal_cost_updated_at": "TEXT",
    }
    for column, declaration in column_declarations.items():
        if column not in job_columns:
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} {declaration}")

    worker_columns = _table_columns(conn, "workers")
    worker_declarations = {
        "rate": "REAL",
        "tickets": "INTEGER",
    }
    for column, declaration in worker_declarations.items():
        if column not in worker_columns:
            conn.execute(f"ALTER TABLE workers ADD COLUMN {column} {declaration}")

    ensure_historical_job_routes_table(conn)
    conn.commit()

    for table_name in ("inventory_items", "workers", "trucks", "shipments"):
        if not _table_exists(conn, table_name):
            conn.execute(
                f"SELECT RAISE(FAIL, 'Failed to create {table_name} during bootstrap')"
            )


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


def upsert_inventory_item(
    conn: sqlite3.Connection,
    *,
    name: str,
    description: str = "",
    quantity: int = 0,
    unit: str = "unit",
) -> sqlite3.Row:
    """Create or update an inventory item and return the stored row."""

    timestamp = datetime.now(UTC).isoformat()
    conn.execute(
        """
        INSERT INTO inventory_items (name, description, quantity, unit, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            description = excluded.description,
            quantity = excluded.quantity,
            unit = excluded.unit,
            updated_at = excluded.updated_at
        """,
        (name, description, int(quantity), unit, timestamp),
    )
    conn.commit()
    return conn.execute(
        "SELECT * FROM inventory_items WHERE name = ?", (name,)
    ).fetchone()


def list_inventory(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return all inventory items ordered by name."""

    return list(conn.execute("SELECT * FROM inventory_items ORDER BY name"))


def upsert_truck(
    conn: sqlite3.Connection,
    *,
    truck_id: str,
    name: str | None = None,
    capacity_m3: float | None = None,
    active: bool = True,
    notes: str | None = None,
) -> sqlite3.Row:
    """Create or update a truck record."""

    timestamp = datetime.now(UTC).isoformat()
    conn.execute(
        """
        INSERT INTO trucks (truck_id, name, capacity_m3, active, notes, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(truck_id) DO UPDATE SET
            name = excluded.name,
            capacity_m3 = excluded.capacity_m3,
            active = excluded.active,
            notes = excluded.notes,
            updated_at = excluded.updated_at
        """,
        (truck_id, name, capacity_m3, int(active), notes, timestamp),
    )
    conn.commit()
    return conn.execute(
        "SELECT * FROM trucks WHERE truck_id = ?", (truck_id,)
    ).fetchone()


def upsert_worker(
    conn: sqlite3.Connection,
    *,
    name: str,
    role: str = "",
    phone: str = "",
    rate: float | None = None,
    tickets: int | None = None,
    active: bool = True,
) -> sqlite3.Row:
    """Create or update a worker record based on the unique name."""

    timestamp = datetime.now(UTC).isoformat()
    rate_value = float(rate) if rate is not None else None
    tickets_value = int(tickets) if tickets is not None else None
    conn.execute(
        """
        INSERT INTO workers (name, role, phone, rate, tickets, active, hired_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            role = excluded.role,
            phone = excluded.phone,
            rate = excluded.rate,
            tickets = excluded.tickets,
            active = excluded.active,
            updated_at = excluded.updated_at
        """,
        (
            name,
            role,
            phone,
            rate_value,
            tickets_value,
            int(active),
            timestamp,
            timestamp,
        ),
    )
    conn.commit()
    return conn.execute("SELECT * FROM workers WHERE name = ?", (name,)).fetchone()


def _coalesce_name(first_name: str | float | None, last_name: str | float | None) -> str:
    parts = [
        str(first_name).strip() if first_name is not None and not pd.isna(first_name) else "",
        str(last_name).strip() if last_name is not None and not pd.isna(last_name) else "",
    ]
    return " ".join(part for part in parts if part)


def import_workers_from_staff_sheet(
    conn: sqlite3.Connection,
    workbook: os.PathLike[str] | str | bytes | IO[bytes],
    *,
    sheet_name: str = "STAFF",
) -> tuple[int, int]:
    """Import or update worker records from a STAFF worksheet.

    Returns a ``(inserted, updated)`` tuple.
    """

    if hasattr(workbook, "seek"):
        try:
            workbook.seek(0)
        except Exception:
            pass

    df = pd.read_excel(workbook, sheet_name=sheet_name)
    inserted = 0
    updated = 0

    for _, row in df.iterrows():
        name = _coalesce_name(row.get("FIRST NAME"), row.get("LAST NAME"))
        if not name:
            continue

        existing = conn.execute(
            "SELECT id FROM workers WHERE name = ?",
            (name,),
        ).fetchone()

        rate = row.get("RATE")
        tickets = row.get("TICKETS")
        rate_value: float | None
        tickets_value: int | None
        try:
            rate_value = None if pd.isna(rate) else float(rate)
        except (TypeError, ValueError):
            rate_value = None
        try:
            tickets_value = None if pd.isna(tickets) else int(tickets)
        except (TypeError, ValueError):
            tickets_value = None
        upsert_worker(
            conn,
            name=name,
            role=str(row.get("ROLE", "") or ""),
            rate=rate_value,
            tickets=tickets_value,
        )

        if existing is None:
            inserted += 1
        else:
            updated += 1

    return inserted, updated


def create_shipment(
    conn: sqlite3.Connection,
    *,
    job_id: int | None = None,
    historical_job_id: int | None = None,
    inventory_item_id: int | None = None,
    truck_id: str | None = None,
    worker_id: int | None = None,
    status: str = "planned",
    scheduled_date: str | None = None,
    delivered_at: str | None = None,
) -> sqlite3.Row:
    """Insert a shipment linked to a job or historical job."""

    if job_id is None and historical_job_id is None:
        raise ValueError("Shipments must reference a job or historical job")

    timestamp = datetime.now(UTC).isoformat()
    conn.execute(
        """
        INSERT INTO shipments (
            job_id,
            historical_job_id,
            inventory_item_id,
            truck_id,
            worker_id,
            status,
            scheduled_date,
            delivered_at,
            created_at,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            job_id,
            historical_job_id,
            inventory_item_id,
            truck_id,
            worker_id,
            status,
            scheduled_date,
            delivered_at,
            timestamp,
            timestamp,
        ),
    )
    conn.commit()
    return conn.execute(
        "SELECT * FROM shipments WHERE id = last_insert_rowid()"
    ).fetchone()


def fetch_shipments_with_context(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return shipments joined with job, inventory, and assignment context."""

    query = """
        SELECT
            s.id,
            s.status,
            s.scheduled_date,
            s.delivered_at,
            s.job_id,
            s.historical_job_id,
            j.origin AS job_origin,
            j.destination AS job_destination,
            h.origin AS historical_origin,
            h.destination AS historical_destination,
            i.name AS inventory_name,
            i.quantity AS inventory_quantity,
            t.truck_id,
            t.name AS truck_name,
            w.name AS worker_name,
            w.role AS worker_role,
            w.rate AS worker_rate,
            w.tickets AS worker_tickets
        FROM shipments AS s
        LEFT JOIN jobs AS j ON s.job_id = j.id
        LEFT JOIN historical_jobs AS h ON s.historical_job_id = h.id
        LEFT JOIN inventory_items AS i ON s.inventory_item_id = i.id
        LEFT JOIN trucks AS t ON s.truck_id = t.truck_id
        LEFT JOIN workers AS w ON s.worker_id = w.id
        ORDER BY s.id
    """
    return list(conn.execute(query))
