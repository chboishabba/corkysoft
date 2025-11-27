"""Database helpers for analytics features."""
from __future__ import annotations

import os
import sqlite3
from datetime import UTC, datetime
from urllib.parse import quote_plus
from typing import IO, Iterable, Optional, Sequence

import pandas as pd

from .connection import DEFAULT_DB_PATH, connection_scope, get_connection
from .parameters import (
    bootstrap_parameters,
    ensure_global_parameters_table,
    get_parameter_value,
    set_parameter_value,
)


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
    job_number TEXT,
    job_date TEXT,
    client TEXT,
    client_reference TEXT,
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
    created_at TEXT,
    updated_at TEXT,
    UNIQUE(job_number)
);

CREATE TABLE IF NOT EXISTS suppliers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_name TEXT NOT NULL,
    contact_name TEXT,
    contact_number TEXT,
    email TEXT,
    notes TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(company_name)
);

CREATE TABLE IF NOT EXISTS inventory_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    quantity INTEGER NOT NULL DEFAULT 0,
    unit TEXT DEFAULT 'unit',
    supplier_id INTEGER,
    updated_at TEXT,
    UNIQUE(name),
    FOREIGN KEY(supplier_id) REFERENCES suppliers(id)
);

CREATE TABLE IF NOT EXISTS inventory_movements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    inventory_item_id INTEGER NOT NULL,
    shipment_id INTEGER,
    change_on_hand INTEGER NOT NULL DEFAULT 0,
    change_allocated INTEGER NOT NULL DEFAULT 0,
    reason TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY(inventory_item_id) REFERENCES inventory_items(id) ON DELETE CASCADE,
    FOREIGN KEY(shipment_id) REFERENCES shipments(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_inventory_movements_item
    ON inventory_movements(inventory_item_id);

CREATE TABLE IF NOT EXISTS workers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_code TEXT,
    name TEXT NOT NULL,
    role TEXT DEFAULT '',
    phone TEXT DEFAULT '',
    rate REAL,
    tickets INTEGER,
    active INTEGER NOT NULL DEFAULT 1,
    hired_at TEXT,
    created_at TEXT,
    updated_at TEXT,
    UNIQUE(employee_code),
    UNIQUE(name, phone)
);

CREATE TABLE IF NOT EXISTS job_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    segment_sequence INTEGER NOT NULL,
    origin TEXT,
    destination TEXT,
    mode TEXT,
    status TEXT,
    distance_km REAL,
    client_reference TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT,
    UNIQUE(job_id, segment_sequence),
    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS container_bookings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    booking_reference TEXT NOT NULL,
    job_id INTEGER,
    client_reference TEXT,
    status TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT,
    UNIQUE(booking_reference),
    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS containers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    container_number TEXT NOT NULL,
    booking_id INTEGER,
    job_id INTEGER,
    client_reference TEXT,
    status TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT,
    UNIQUE(container_number),
    FOREIGN KEY(booking_id) REFERENCES container_bookings(id) ON DELETE SET NULL,
    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS trucks (
    truck_id TEXT PRIMARY KEY,
    name TEXT,
    capacity_m3 REAL,
    active INTEGER NOT NULL DEFAULT 1,
    notes TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS vehicle_details (
    truck_id TEXT PRIMARY KEY,
    state TEXT,
    rego TEXT,
    rego_expiry TEXT,
    make TEXT,
    model TEXT,
    year INTEGER,
    body_type TEXT,
    description TEXT,
    nhv_code TEXT,
    insurance TEXT,
    odometer INTEGER,
    last_service TEXT,
    next_service TEXT,
    coi_number TEXT,
    coi_due TEXT,
    present_driver TEXT,
    daily_check_complete INTEGER,
    FOREIGN KEY(truck_id) REFERENCES trucks(truck_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS vehicle_repairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    truck_id TEXT NOT NULL,
    job_item TEXT NOT NULL,
    description TEXT,
    price REAL,
    supplier TEXT,
    service_date TEXT,
    next_service_date TEXT,
    notes TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT,
    FOREIGN KEY(truck_id) REFERENCES trucks(truck_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_vehicle_repairs_truck_date
    ON vehicle_repairs(truck_id, service_date);

CREATE TABLE IF NOT EXISTS shipments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER,
    historical_job_id INTEGER,
    inventory_item_id INTEGER,
    truck_id TEXT,
    worker_id INTEGER,
    segment_id INTEGER,
    quantity REAL NOT NULL DEFAULT 1,
    from_location TEXT,
    to_location TEXT,
    status TEXT NOT NULL DEFAULT 'planned',
    scheduled_date TEXT,
    delivered_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT,
    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE SET NULL,
    FOREIGN KEY(historical_job_id) REFERENCES historical_jobs(id) ON DELETE SET NULL,
    FOREIGN KEY(inventory_item_id) REFERENCES inventory_items(id) ON DELETE SET NULL,
    FOREIGN KEY(truck_id) REFERENCES trucks(truck_id) ON DELETE SET NULL,
    FOREIGN KEY(worker_id) REFERENCES workers(id) ON DELETE SET NULL,
    FOREIGN KEY(segment_id) REFERENCES job_segments(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS worker_roles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS worker_compliances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS job_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    segment_sequence INTEGER NOT NULL,
    from_location TEXT,
    to_location TEXT,
    planned_start TEXT,
    planned_end TEXT,
    status TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT,
    UNIQUE(job_id, segment_sequence),
    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS job_segment_workers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    segment_id INTEGER NOT NULL,
    worker_id INTEGER NOT NULL,
    start_time TEXT NOT NULL DEFAULT '',
    end_time TEXT NOT NULL DEFAULT '',
    role_id INTEGER,
    FOREIGN KEY(segment_id) REFERENCES job_segments(id) ON DELETE CASCADE,
    FOREIGN KEY(worker_id) REFERENCES workers(id) ON DELETE CASCADE,
    FOREIGN KEY(role_id) REFERENCES worker_roles(id) ON DELETE SET NULL,
    UNIQUE(segment_id, worker_id, start_time, end_time)
);

CREATE TABLE IF NOT EXISTS job_segment_vehicles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    segment_id INTEGER NOT NULL,
    truck_id TEXT NOT NULL,
    requirement_met INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY(segment_id) REFERENCES job_segments(id) ON DELETE CASCADE,
    FOREIGN KEY(truck_id) REFERENCES trucks(truck_id) ON DELETE CASCADE,
    UNIQUE(segment_id, truck_id)
);

CREATE TABLE IF NOT EXISTS containers (
    container_number TEXT PRIMARY KEY,
    type TEXT,
    tare REAL,
    payload REAL,
    ownership TEXT,
    status TEXT,
    location TEXT
);

CREATE TABLE IF NOT EXISTS container_bookings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    container_number TEXT NOT NULL,
    booking_ref TEXT NOT NULL,
    etd TEXT,
    eta TEXT,
    load_port TEXT,
    discharge_port TEXT,
    carrier TEXT,
    UNIQUE(container_number, booking_ref),
    FOREIGN KEY(container_number) REFERENCES containers(container_number) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS job_container_allocations (
    job_id INTEGER NOT NULL,
    booking_id INTEGER NOT NULL,
    segment_id INTEGER,
    volume_share REAL,
    weight_share REAL,
    UNIQUE(job_id, booking_id, segment_id),
    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE,
    FOREIGN KEY(booking_id) REFERENCES container_bookings(id) ON DELETE CASCADE,
    FOREIGN KEY(segment_id) REFERENCES job_segments(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS container_movements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    container_number TEXT NOT NULL,
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    source_ref TEXT,
    location TEXT,
    notes TEXT,
    UNIQUE(container_number, event_type, timestamp, source_ref),
    FOREIGN KEY(container_number) REFERENCES containers(container_number) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS container_seals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    container_number TEXT NOT NULL,
    seal_number TEXT NOT NULL,
    applied_at TEXT,
    removed_at TEXT,
    source_ref TEXT,
    UNIQUE(container_number, seal_number, applied_at, source_ref),
    FOREIGN KEY(container_number) REFERENCES containers(container_number) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS condition_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    container_number TEXT NOT NULL,
    report_time TEXT NOT NULL,
    condition TEXT,
    reporter TEXT,
    notes TEXT,
    source_ref TEXT,
    UNIQUE(container_number, report_time, source_ref),
    FOREIGN KEY(container_number) REFERENCES containers(container_number) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS container_charges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    container_number TEXT NOT NULL,
    booking_id INTEGER,
    charge_type TEXT NOT NULL,
    amount REAL,
    currency TEXT,
    effective_date TEXT,
    source_ref TEXT,
    UNIQUE(container_number, booking_id, charge_type, effective_date, source_ref),
    FOREIGN KEY(container_number) REFERENCES containers(container_number) ON DELETE CASCADE,
    FOREIGN KEY(booking_id) REFERENCES container_bookings(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS driver_shifts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    shift_date TEXT NOT NULL,
    truck_id TEXT,
    worker_id INTEGER,
    job_id INTEGER,
    shipment_id INTEGER,
    ticket_numbers TEXT,
    shift_start TEXT,
    shift_end TEXT,
    shift_window_start TEXT,
    shift_window_end TEXT,
    role TEXT,
    hours REAL,
    hourly_rate REAL,
    cost_total REAL,
    notes TEXT,
    source TEXT,
    imported_at TEXT NOT NULL,
    UNIQUE(shift_date, truck_id, worker_id, shift_start, shift_end, ticket_numbers),
    FOREIGN KEY(truck_id) REFERENCES trucks(truck_id) ON DELETE SET NULL,
    FOREIGN KEY(worker_id) REFERENCES workers(id) ON DELETE SET NULL,
    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE SET NULL,
    FOREIGN KEY(shipment_id) REFERENCES shipments(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_driver_shifts_date ON driver_shifts(shift_date);
CREATE INDEX IF NOT EXISTS idx_driver_shifts_job ON driver_shifts(job_id);
CREATE INDEX IF NOT EXISTS idx_driver_shifts_shipment ON driver_shifts(shipment_id);
"""
def ensure_suppliers_table(conn: sqlite3.Connection) -> None:
    """Create or migrate the suppliers table used by inventory features."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS suppliers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_name TEXT NOT NULL,
            contact_name TEXT,
            contact_number TEXT,
            email TEXT,
            notes TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(company_name)
        )
        """
    )

    column_declarations = {
        "contact_name": "TEXT",
        "contact_number": "TEXT",
        "email": "TEXT",
        "notes": "TEXT",
        "created_at": "TEXT",
        "updated_at": "TEXT",
    }

    for column, declaration in column_declarations.items():
        if column not in _table_columns(conn, "suppliers"):
            conn.execute(f"ALTER TABLE suppliers ADD COLUMN {column} {declaration}")

    conn.commit()


def ensure_dashboard_tables(conn: sqlite3.Connection) -> None:
    """Create empty dashboard tables so the UI can load before data imports."""

    conn.executescript(_DASHBOARD_SCHEMA_SQL)
    ensure_suppliers_table(conn)
    _ensure_inventory_movements_table(conn)
    _ensure_job_segment_tables(conn)

    hist_columns = _table_columns(conn, "historical_jobs")
    if "client_id" not in hist_columns:
        conn.execute("ALTER TABLE historical_jobs ADD COLUMN client_id INTEGER")

    job_columns = _table_columns(conn, "jobs")
    column_declarations = {
        "job_number": "TEXT",
        "client_id": "INTEGER",
        "client_reference": "TEXT",
        "origin_resolved": "TEXT",
        "destination_resolved": "TEXT",
        "route_geojson": "TEXT",
        "internal_cost_total": "REAL DEFAULT 0",
        "internal_cost_updated_at": "TEXT",
        "created_at": "TEXT",
    }
    for column, declaration in column_declarations.items():
        if column not in job_columns:
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} {declaration}")

    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_job_number ON jobs(job_number)"
    )

    _ensure_worker_schema(conn)

    ensure_historical_job_routes_table(conn)
    _ensure_vehicle_details_table(conn)
    _ensure_shipment_columns(conn)
    _backfill_job_segments(conn)
    conn.commit()

    for table_name in (
        "inventory_items",
        "inventory_movements",
        "workers",
        "trucks",
        "vehicle_repairs",
        "shipments",
        "containers",
        "container_bookings",
        "job_container_allocations",
        "container_movements",
        "container_seals",
        "condition_reports",
        "container_charges",
    ):
        if not _table_exists(conn, table_name):
            conn.execute(
                f"SELECT RAISE(FAIL, 'Failed to create {table_name} during bootstrap')"
            )


def _table_columns(conn: sqlite3.Connection, table: str) -> Sequence[str]:
    """Return the column names for *table* in the current connection."""
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [row[1] for row in rows]


def _unique_index_columns(conn: sqlite3.Connection, table: str) -> list[list[str]]:
    """Return column lists for all unique indexes on *table*."""

    indexes = conn.execute(f"PRAGMA index_list({table})").fetchall()
    columns: list[list[str]] = []
    for _, name, is_unique, *_ in indexes:
        if not is_unique:
            continue
        cols = conn.execute(f"PRAGMA index_info({name})").fetchall()
        columns.append([col[2] for col in cols])
    return columns


def _ensure_inventory_movements_table(conn: sqlite3.Connection) -> None:
    """Ensure the inventory_movements table exists for tracking adjustments."""

    if _table_exists(conn, "inventory_movements"):
        return

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS inventory_movements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            inventory_item_id INTEGER NOT NULL,
            shipment_id INTEGER,
            change_on_hand INTEGER NOT NULL DEFAULT 0,
            change_allocated INTEGER NOT NULL DEFAULT 0,
            reason TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY(inventory_item_id) REFERENCES inventory_items(id) ON DELETE CASCADE,
            FOREIGN KEY(shipment_id) REFERENCES shipments(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_inventory_movements_item
            ON inventory_movements(inventory_item_id);
        """
    )


def _rebuild_workers_table(conn: sqlite3.Connection) -> None:
    """Recreate the workers table to enforce new uniqueness and columns."""

    if not _table_exists(conn, "workers"):
        return

    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("ALTER TABLE workers RENAME TO workers_old")
    conn.execute(
        """
        CREATE TABLE workers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_code TEXT,
            name TEXT NOT NULL,
            role TEXT DEFAULT '',
            phone TEXT DEFAULT '',
            rate REAL,
            tickets INTEGER,
            active INTEGER NOT NULL DEFAULT 1,
            hired_at TEXT,
            created_at TEXT,
            updated_at TEXT,
            UNIQUE(employee_code),
            UNIQUE(name, phone)
        )
        """
    )

    conn.execute(
        """
        INSERT INTO workers (
            id, employee_code, name, role, phone, rate, tickets, active, hired_at, created_at, updated_at
        )
        SELECT
            id,
            NULL,
            name,
            role,
            phone,
            rate,
            tickets,
            active,
            hired_at,
            COALESCE(hired_at, updated_at, datetime('now')),
            updated_at
        FROM workers_old
        """
    )
    conn.execute("DROP TABLE workers_old")
    conn.execute("PRAGMA foreign_keys=ON")


def _ensure_worker_schema(conn: sqlite3.Connection) -> None:
    """Ensure worker columns and uniqueness constraints match import contracts."""

    if not _table_exists(conn, "workers"):
        return

    worker_columns = _table_columns(conn, "workers")
    unique_sets = [set(idx) for idx in _unique_index_columns(conn, "workers")]
    has_employee_code_unique = {"employee_code"} in unique_sets
    has_name_phone_unique = {"name", "phone"} in unique_sets
    has_expected_uniques = has_employee_code_unique and has_name_phone_unique

    needs_rebuild = "employee_code" not in worker_columns or "created_at" not in worker_columns
    needs_rebuild = needs_rebuild or not has_expected_uniques

    if needs_rebuild:
        _rebuild_workers_table(conn)
    else:
        worker_declarations = {
            "rate": "REAL",
            "tickets": "INTEGER",
        }
        for column, declaration in worker_declarations.items():
            if column not in worker_columns:
                conn.execute(f"ALTER TABLE workers ADD COLUMN {column} {declaration}")


def _ensure_shipment_columns(conn: sqlite3.Connection) -> None:
    """Ensure shipment columns for quantities and locations exist and are populated."""

    if not _table_exists(conn, "shipments"):
        return

    columns = set(_table_columns(conn, "shipments"))
    declarations = {
        "quantity": "REAL NOT NULL DEFAULT 1",
        "from_location": "TEXT",
        "to_location": "TEXT",
        "segment_id": "INTEGER",
    }
    for column, declaration in declarations.items():
        if column not in columns:
            conn.execute(f"ALTER TABLE shipments ADD COLUMN {column} {declaration}")

    conn.execute(
        "UPDATE shipments SET quantity = COALESCE(quantity, 1) WHERE quantity IS NULL"
    )
    conn.execute(
        """
        UPDATE shipments
        SET from_location = COALESCE(
            from_location,
            (SELECT origin FROM jobs WHERE jobs.id = shipments.job_id),
            (
                SELECT origin
                FROM historical_jobs
                WHERE historical_jobs.id = shipments.historical_job_id
            )
        )
        WHERE from_location IS NULL
        """
    )
    conn.execute(
        """
        UPDATE shipments
        SET to_location = COALESCE(
            to_location,
            (SELECT destination FROM jobs WHERE jobs.id = shipments.job_id),
            (
                SELECT destination
                FROM historical_jobs
                WHERE historical_jobs.id = shipments.historical_job_id
            )
        )
        WHERE to_location IS NULL
        """
    )


def _ensure_inventory_movements_table(conn: sqlite3.Connection) -> None:
    """Create the inventory_movements table and supporting index when missing."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS inventory_movements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            inventory_item_id INTEGER NOT NULL,
            shipment_id INTEGER,
            change_on_hand INTEGER NOT NULL DEFAULT 0,
            change_allocated INTEGER NOT NULL DEFAULT 0,
            reason TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY(inventory_item_id) REFERENCES inventory_items(id) ON DELETE CASCADE,
            FOREIGN KEY(shipment_id) REFERENCES shipments(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_inventory_movements_item
            ON inventory_movements(inventory_item_id)
        """
    )


def _ensure_job_segment_tables(conn: sqlite3.Connection) -> None:
    """Create tables used for segment-level scheduling and assignments."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS worker_roles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            description TEXT DEFAULT ''
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS worker_compliances (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            description TEXT DEFAULT ''
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS job_segments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL,
            segment_sequence INTEGER NOT NULL,
            from_location TEXT,
            to_location TEXT,
            planned_start TEXT,
            planned_end TEXT,
            status TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT,
            UNIQUE(job_id, segment_sequence),
            FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_job_segments_job_seq
            ON job_segments(job_id, segment_sequence)
        """
    )

    columns = _table_columns(conn, "job_segments")
    column_declarations = {
        "from_location": "TEXT",
        "to_location": "TEXT",
        "planned_start": "TEXT",
        "planned_end": "TEXT",
        "status": "TEXT",
        "updated_at": "TEXT",
    }
    for column, declaration in column_declarations.items():
        if column not in columns:
            conn.execute(f"ALTER TABLE job_segments ADD COLUMN {column} {declaration}")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS job_segment_workers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            segment_id INTEGER NOT NULL,
            worker_id INTEGER NOT NULL,
            start_time TEXT NOT NULL DEFAULT '',
            end_time TEXT NOT NULL DEFAULT '',
            role_id INTEGER,
            FOREIGN KEY(segment_id) REFERENCES job_segments(id) ON DELETE CASCADE,
            FOREIGN KEY(worker_id) REFERENCES workers(id) ON DELETE CASCADE,
            FOREIGN KEY(role_id) REFERENCES worker_roles(id) ON DELETE SET NULL,
            UNIQUE(segment_id, worker_id, start_time, end_time)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_job_segment_workers_segment
            ON job_segment_workers(segment_id)
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS job_segment_vehicles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            segment_id INTEGER NOT NULL,
            truck_id TEXT NOT NULL,
            requirement_met INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY(segment_id) REFERENCES job_segments(id) ON DELETE CASCADE,
            FOREIGN KEY(truck_id) REFERENCES trucks(truck_id) ON DELETE CASCADE,
            UNIQUE(segment_id, truck_id)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_job_segment_vehicles_segment
            ON job_segment_vehicles(segment_id)
        """
    )

    if _table_exists(conn, "shipments"):
        shipment_columns = set(_table_columns(conn, "shipments"))
        if "segment_id" not in shipment_columns:
            conn.execute("ALTER TABLE shipments ADD COLUMN segment_id INTEGER")


def _link_vehicle_to_segment(
    conn: sqlite3.Connection,
    *,
    segment_id: int,
    truck_id: str,
    requirement_met: bool = False,
) -> None:
    """Associate a vehicle with a job segment."""

    conn.execute(
        """
        INSERT INTO job_segment_vehicles (segment_id, truck_id, requirement_met)
        VALUES (?, ?, ?)
        ON CONFLICT(segment_id, truck_id) DO UPDATE SET
            requirement_met = excluded.requirement_met
        """,
        (segment_id, truck_id, int(requirement_met)),
    )


def _link_worker_to_segment(
    conn: sqlite3.Connection,
    *,
    segment_id: int,
    worker_id: int,
    start_time: str | None = None,
    end_time: str | None = None,
    role_id: int | None = None,
) -> None:
    """Associate a worker with a job segment."""

    conn.execute(
        """
        INSERT INTO job_segment_workers (
            segment_id, worker_id, start_time, end_time, role_id
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(segment_id, worker_id, start_time, end_time) DO UPDATE SET
            role_id = excluded.role_id,
            start_time = excluded.start_time,
            end_time = excluded.end_time
        """,
        (segment_id, worker_id, start_time or "", end_time or "", role_id),
    )


def _backfill_job_segments(conn: sqlite3.Connection) -> None:
    """Ensure each job has at least one segment and link legacy assignments."""

    _ensure_job_segment_tables(conn)

    timestamp = datetime.now(UTC).isoformat()
    for job_row in conn.execute("SELECT id, origin, destination FROM jobs"):
        job_id = int(job_row["id"])
        segment = conn.execute(
            "SELECT * FROM job_segments WHERE job_id = ? AND segment_sequence = 1",
            (job_id,),
        ).fetchone()

        if segment is None:
            conn.execute(
                """
                INSERT INTO job_segments (
                    job_id, segment_sequence, from_location, to_location, status, created_at
                ) VALUES (?, 1, ?, ?, 'planned', ?)
                """,
                (job_id, job_row["origin"], job_row["destination"], timestamp),
            )
            segment = conn.execute(
                "SELECT * FROM job_segments WHERE job_id = ? AND segment_sequence = 1",
                (job_id,),
            ).fetchone()
        else:
            if segment["from_location"] is None and job_row["origin"]:
                conn.execute(
                    "UPDATE job_segments SET from_location = ? WHERE id = ?",
                    (job_row["origin"], segment["id"]),
                )
            if segment["to_location"] is None and job_row["destination"]:
                conn.execute(
                    "UPDATE job_segments SET to_location = ? WHERE id = ?",
                    (job_row["destination"], segment["id"]),
                )

        conn.execute(
            "UPDATE shipments SET segment_id = COALESCE(segment_id, ?) WHERE job_id = ?",
            (segment["id"], job_id),
        )

        for shipment in conn.execute(
            "SELECT truck_id, worker_id FROM shipments WHERE job_id = ?",
            (job_id,),
        ):
            if shipment["truck_id"]:
                _link_vehicle_to_segment(
                    conn,
                    segment_id=int(segment["id"]),
                    truck_id=shipment["truck_id"],
                )
            if shipment["worker_id"]:
                _link_worker_to_segment(
                    conn,
                    segment_id=int(segment["id"]),
                    worker_id=int(shipment["worker_id"]),
                )


def get_or_create_job_segment(
    conn: sqlite3.Connection,
    *,
    job_id: int,
    segment_sequence: int = 1,
    from_location: str | None = None,
    to_location: str | None = None,
    planned_start: str | None = None,
    planned_end: str | None = None,
    status: str = "planned",
) -> sqlite3.Row:
    """Return the requested job segment, creating it if missing."""

    _ensure_job_segment_tables(conn)
    existing = conn.execute(
        "SELECT * FROM job_segments WHERE job_id = ? AND segment_sequence = ?",
        (job_id, segment_sequence),
    ).fetchone()
    if existing is None:
        timestamp = datetime.now(UTC).isoformat()
        conn.execute(
            """
            INSERT INTO job_segments (
                job_id, segment_sequence, from_location, to_location, planned_start,
                planned_end, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                segment_sequence,
                from_location,
                to_location,
                planned_start,
                planned_end,
                status,
                timestamp,
            ),
        )
    else:
        if existing["from_location"] is None and from_location:
            conn.execute(
                "UPDATE job_segments SET from_location = ? WHERE id = ?",
                (from_location, existing["id"]),
            )
        if existing["to_location"] is None and to_location:
            conn.execute(
                "UPDATE job_segments SET to_location = ? WHERE id = ?",
                (to_location, existing["id"]),
            )
    return conn.execute(
        "SELECT * FROM job_segments WHERE job_id = ? AND segment_sequence = ?",
        (job_id, segment_sequence),
    ).fetchone()

def _ensure_driver_shift_columns(conn: sqlite3.Connection) -> None:
    columns = _table_columns(conn, "driver_shifts") if _table_exists(conn, "driver_shifts") else []
    declarations = {
        "job_id": "INTEGER REFERENCES jobs(id) ON DELETE SET NULL",
        "shipment_id": "INTEGER REFERENCES shipments(id) ON DELETE SET NULL",
        "ticket_numbers": "TEXT",
        "shift_start": "TEXT",
        "shift_end": "TEXT",
        "shift_window_start": "TEXT",
        "shift_window_end": "TEXT",
        "role": "TEXT",
        "notes": "TEXT",
        "source": "TEXT",
        "imported_at": "TEXT NOT NULL DEFAULT ''",
    }
    for column, declaration in declarations.items():
        if column not in columns:
            conn.execute(f"ALTER TABLE driver_shifts ADD COLUMN {column} {declaration}")



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


def _ensure_vehicle_details_table(conn: sqlite3.Connection) -> None:
    """Create or migrate the vehicle details table used for fleet metadata."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS vehicle_details (
            truck_id TEXT PRIMARY KEY,
            state TEXT,
            rego TEXT,
            rego_expiry TEXT,
            make TEXT,
            model TEXT,
            year INTEGER,
            body_type TEXT,
            description TEXT,
            nhv_code TEXT,
            insurance TEXT,
            odometer INTEGER,
            last_service TEXT,
            next_service TEXT,
            coi_number TEXT,
            coi_due TEXT,
            present_driver TEXT,
            daily_check_complete INTEGER,
            FOREIGN KEY(truck_id) REFERENCES trucks(truck_id) ON DELETE CASCADE
        )
        """
    )

    columns = set(_table_columns(conn, "vehicle_details"))
    column_types = {
        "state": "TEXT",
        "rego": "TEXT",
        "rego_expiry": "TEXT",
        "make": "TEXT",
        "model": "TEXT",
        "year": "INTEGER",
        "body_type": "TEXT",
        "description": "TEXT",
        "nhv_code": "TEXT",
        "insurance": "TEXT",
        "odometer": "INTEGER",
        "last_service": "TEXT",
        "next_service": "TEXT",
        "coi_number": "TEXT",
        "coi_due": "TEXT",
        "present_driver": "TEXT",
        "daily_check_complete": "INTEGER",
    }
    for column, declaration in column_types.items():
        if column not in columns:
            conn.execute(f"ALTER TABLE vehicle_details ADD COLUMN {column} {declaration}")

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
    supplier_id: int | None = None,
) -> sqlite3.Row:
    """Create or update an inventory item and return the stored row."""

    timestamp = datetime.now(UTC).isoformat()
    conn.execute(
        """
        INSERT INTO inventory_items (name, description, quantity, unit, supplier_id, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            description = excluded.description,
            quantity = excluded.quantity,
            unit = excluded.unit,
            supplier_id = excluded.supplier_id,
            updated_at = excluded.updated_at
        """,
        (name, description, int(quantity), unit, supplier_id, timestamp),
    )
    conn.commit()
    return conn.execute(
        "SELECT * FROM inventory_items WHERE name = ?", (name,)
    ).fetchone()


def _inventory_balance_query(where_clause: str = "") -> str:
    balances_cte = """
        WITH movement_totals AS (
            SELECT
                inventory_item_id,
                COALESCE(SUM(change_on_hand), 0) AS delta_on_hand,
                COALESCE(SUM(change_allocated), 0) AS delta_allocated
            FROM inventory_movements
            GROUP BY inventory_item_id
        )
    """
    select_sql = f"""
        SELECT
            i.*,\n            i.quantity + COALESCE(m.delta_on_hand, 0) AS on_hand_quantity,\n            COALESCE(m.delta_allocated, 0) AS allocated_quantity,\n            i.quantity + COALESCE(m.delta_on_hand, 0) - COALESCE(m.delta_allocated, 0)
                AS available_quantity
        FROM inventory_items AS i
        LEFT JOIN movement_totals AS m ON m.inventory_item_id = i.id
        {where_clause}
        ORDER BY i.name
    """
    return balances_cte + select_sql


def list_inventory_balances(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return inventory items with on-hand, allocated, and available totals."""

    _ensure_inventory_movements_table(conn)
    return list(conn.execute(_inventory_balance_query()))


def get_inventory_balance(
    conn: sqlite3.Connection, inventory_item_id: int
) -> sqlite3.Row | None:
    """Return a single inventory balance row by item id."""

    _ensure_inventory_movements_table(conn)
    rows = conn.execute(
        _inventory_balance_query("WHERE i.id = ?"), (inventory_item_id,)
    ).fetchall()
    return rows[0] if rows else None


def record_inventory_movement(
    conn: sqlite3.Connection,
    *,
    inventory_item_id: int,
    shipment_id: int | None = None,
    change_on_hand: int = 0,
    change_allocated: int = 0,
    reason: str | None = None,
    commit: bool = True,
) -> sqlite3.Row:
    """Insert an inventory movement entry and return the stored row."""

    _ensure_inventory_movements_table(conn)
    timestamp = datetime.now(UTC).isoformat()
    conn.execute(
        """
        INSERT INTO inventory_movements (
            inventory_item_id,
            shipment_id,
            change_on_hand,
            change_allocated,
            reason,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            inventory_item_id,
            shipment_id,
            int(change_on_hand),
            int(change_allocated),
            reason or "",
            timestamp,
        ),
    )
    if commit:
        conn.commit()
    return conn.execute(
        "SELECT * FROM inventory_movements WHERE id = last_insert_rowid()"
    ).fetchone()


def list_inventory(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return all inventory items ordered by name."""

    return list(
        conn.execute(
            """
            SELECT
                i.*, 
                s.company_name AS supplier_company_name,
                s.contact_name AS supplier_contact_name,
                s.contact_number AS supplier_contact_number,
                s.email AS supplier_email,
                s.notes AS supplier_notes
            FROM inventory_items AS i
            LEFT JOIN suppliers AS s ON i.supplier_id = s.id
            ORDER BY i.name
            """
        )
    )


def upsert_supplier(
    conn: sqlite3.Connection,
    *,
    company_name: str,
    contact_name: str | None = None,
    contact_number: str | None = None,
    email: str | None = None,
    notes: str | None = None,
) -> sqlite3.Row:
    """Create or update a supplier record by company name."""

    if not company_name.strip():
        raise ValueError("Supplier company name is required")

    timestamp = datetime.now(UTC).isoformat()
    conn.execute(
        """
        INSERT INTO suppliers (
            company_name, contact_name, contact_number, email, notes, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(company_name) DO UPDATE SET
            contact_name = excluded.contact_name,
            contact_number = excluded.contact_number,
            email = excluded.email,
            notes = excluded.notes,
            updated_at = excluded.updated_at
        """,
        (
            company_name.strip(),
            _clean_optional_str(contact_name),
            _clean_optional_str(contact_number),
            _clean_optional_str(email),
            _clean_optional_str(notes),
            timestamp,
            timestamp,
        ),
    )
    conn.commit()
    return conn.execute(
        "SELECT * FROM suppliers WHERE company_name = ?", (company_name.strip(),)
    ).fetchone()


def list_suppliers(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return suppliers ordered by company name."""

    return list(
        conn.execute(
            "SELECT * FROM suppliers ORDER BY LOWER(company_name), company_name"
        )
    )


def import_suppliers_from_google_sheet(
    conn: sqlite3.Connection,
    *,
    sheet_id: str | None = None,
    sheet_name: str = "SUPPLIERS",
    csv_url: str | None = None,
    dataframe: pd.DataFrame | None = None,
) -> int:
    """Import suppliers from a Google Sheet export or DataFrame.

    Returns the number of supplier rows inserted or updated.
    """

    ensure_suppliers_table(conn)

    df = dataframe
    if df is None:
        resolved_url = csv_url or _build_suppliers_sheet_url(sheet_id, sheet_name)
        if resolved_url is None:
            raise ValueError("Provide a dataframe, csv_url, or sheet_id for import")
        df = pd.read_csv(resolved_url)

    if df.empty:
        return 0

    normalized = df.rename(columns=_normalize_supplier_column)
    required_field = "company_name"
    if required_field not in normalized.columns:
        raise ValueError("Suppliers sheet must include a company name column")

    imported = 0
    for _, row in normalized.iterrows():
        company = _clean_optional_str(row.get("company_name"))
        if not company:
            continue
        contact_name = _clean_optional_str(row.get("contact_name"))
        contact_number = _clean_optional_str(row.get("contact_number"))
        email = _clean_optional_str(row.get("email"))
        notes = _clean_optional_str(row.get("notes"))
        upsert_supplier(
            conn,
            company_name=company,
            contact_name=contact_name,
            contact_number=contact_number,
            email=email,
            notes=notes,
        )
        imported += 1
    return imported


def _clean_optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    value_str = str(value).strip()
    return value_str or None


def _normalize_supplier_column(column_name: str) -> str:
    normalized = column_name.strip().lower().replace(" ", "_")
    if normalized in {"company", "supplier"}:
        return "company_name"
    if normalized in {"contact", "contact_person"}:
        return "contact_name"
    if normalized in {"phone", "phone_number"}:
        return "contact_number"
    return normalized


def _build_suppliers_sheet_url(
    sheet_id: str | None, sheet_name: str, *, env_var: str = "SUPPLIERS_SHEET_ID"
) -> str | None:
    resolved_id = sheet_id or os.environ.get(env_var)
    if not resolved_id:
        explicit_url = os.environ.get("SUPPLIERS_SHEET_URL")
        return explicit_url
    return (
        f"https://docs.google.com/spreadsheets/d/{resolved_id}/gviz/tq?tqx=out:csv"
        f"&sheet={quote_plus(sheet_name)}"
    )


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


def upsert_vehicle_details(
    conn: sqlite3.Connection,
    *,
    truck_id: str,
    state: str | None = None,
    rego: str | None = None,
    rego_expiry: str | None = None,
    make: str | None = None,
    model: str | None = None,
    year: int | None = None,
    body_type: str | None = None,
    description: str | None = None,
    nhv_code: str | None = None,
    insurance: str | None = None,
    odometer: int | None = None,
    last_service: str | None = None,
    next_service: str | None = None,
    coi_number: str | None = None,
    coi_due: str | None = None,
    present_driver: str | None = None,
    daily_check_complete: bool | None = None,
) -> sqlite3.Row:
    """Create or update vehicle metadata for the given ``truck_id``."""

    _ensure_vehicle_details_table(conn)
    conn.execute(
        """
        INSERT INTO vehicle_details (
            truck_id,
            state,
            rego,
            rego_expiry,
            make,
            model,
            year,
            body_type,
            description,
            nhv_code,
            insurance,
            odometer,
            last_service,
            next_service,
            coi_number,
            coi_due,
            present_driver,
            daily_check_complete
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(truck_id) DO UPDATE SET
            state = excluded.state,
            rego = excluded.rego,
            rego_expiry = excluded.rego_expiry,
            make = excluded.make,
            model = excluded.model,
            year = excluded.year,
            body_type = excluded.body_type,
            description = excluded.description,
            nhv_code = excluded.nhv_code,
            insurance = excluded.insurance,
            odometer = excluded.odometer,
            last_service = excluded.last_service,
            next_service = excluded.next_service,
            coi_number = excluded.coi_number,
            coi_due = excluded.coi_due,
            present_driver = excluded.present_driver,
            daily_check_complete = excluded.daily_check_complete
        """,
        (
            truck_id,
            state,
            rego,
            rego_expiry,
            make,
            model,
            year,
            body_type,
            description,
            nhv_code,
            insurance,
            odometer,
            last_service,
            next_service,
            coi_number,
            coi_due,
            present_driver,
            None if daily_check_complete is None else int(bool(daily_check_complete)),
        ),
    )
    conn.commit()
    return conn.execute(
        "SELECT * FROM vehicle_details WHERE truck_id = ?", (truck_id,)
    ).fetchone()


def upsert_worker(
    conn: sqlite3.Connection,
    *,
    employee_code: str | None = None,
    name: str,
    role: str = "",
    phone: str = "",
    rate: float | None = None,
    tickets: int | None = None,
    active: bool = True,
    hired_at: str | None = None,
    created_at: str | None = None,
) -> sqlite3.Row:
    """Create or update a worker record based on the unique name."""

    timestamp = datetime.now(UTC).isoformat()
    created_timestamp = created_at or timestamp
    rate_value = float(rate) if rate is not None else None
    tickets_value = int(tickets) if tickets is not None else None
    clean_phone = phone.strip()
    conflict_target = "employee_code" if employee_code else "name, phone"
    hired_value = hired_at or timestamp
    conn.execute(
        f"""
        INSERT INTO workers (
            employee_code, name, role, phone, rate, tickets, active, hired_at, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT({conflict_target}) DO UPDATE SET
            employee_code = COALESCE(excluded.employee_code, workers.employee_code),
            role = excluded.role,
            phone = excluded.phone,
            rate = excluded.rate,
            tickets = excluded.tickets,
            active = excluded.active,
            hired_at = COALESCE(workers.hired_at, excluded.hired_at),
            updated_at = excluded.updated_at
        """,
        (
            employee_code,
            name,
            role,
            clean_phone,
            rate_value,
            tickets_value,
            int(active),
            hired_value,
            created_timestamp,
            timestamp,
        ),
    )
    conn.commit()
    where_clause = "employee_code = ?" if employee_code else "name = ? AND phone IS ?"
    params = (employee_code,) if employee_code else (name, clean_phone)
    return conn.execute(f"SELECT * FROM workers WHERE {where_clause}", params).fetchone()


def upsert_job_by_number(
    conn: sqlite3.Connection,
    *,
    job_number: str,
    job_date: str | None = None,
    client: str | None = None,
    client_reference: str | None = None,
    origin: str | None = None,
    destination: str | None = None,
    revenue_total: float | None = None,
    revenue: float | None = None,
    volume_m3: float | None = None,
    volume: float | None = None,
    distance_km: float | None = None,
    final_cost: float | None = None,
    origin_postcode: str | None = None,
    destination_postcode: str | None = None,
    origin_lat: float | None = None,
    origin_lon: float | None = None,
    dest_lat: float | None = None,
    dest_lon: float | None = None,
    created_at: str | None = None,
    updated_at: str | None = None,
) -> sqlite3.Row:
    """Insert or update a job keyed by the business ``job_number``."""

    ensure_dashboard_tables(conn)
    cleaned_job_number = str(job_number).strip()
    if not cleaned_job_number:
        raise ValueError("job_number is required for job upsert")

    timestamp = updated_at or datetime.now(UTC).isoformat()
    created_timestamp = created_at or timestamp

    conn.execute(
        """
        INSERT INTO jobs (
            job_number, job_date, client, client_reference, origin, destination,
            revenue_total, revenue, volume_m3, volume, distance_km, final_cost,
            origin_postcode, destination_postcode, origin_lat, origin_lon, dest_lat,
            dest_lon, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(job_number) DO UPDATE SET
            job_date = excluded.job_date,
            client = excluded.client,
            client_reference = excluded.client_reference,
            origin = excluded.origin,
            destination = excluded.destination,
            revenue_total = excluded.revenue_total,
            revenue = excluded.revenue,
            volume_m3 = excluded.volume_m3,
            volume = excluded.volume,
            distance_km = excluded.distance_km,
            final_cost = excluded.final_cost,
            origin_postcode = excluded.origin_postcode,
            destination_postcode = excluded.destination_postcode,
            origin_lat = excluded.origin_lat,
            origin_lon = excluded.origin_lon,
            dest_lat = excluded.dest_lat,
            dest_lon = excluded.dest_lon,
            updated_at = excluded.updated_at,
            created_at = COALESCE(jobs.created_at, excluded.created_at)
        """,
        (
            cleaned_job_number,
            job_date,
            client,
            client_reference,
            origin,
            destination,
            revenue_total,
            revenue,
            volume_m3,
            volume,
            distance_km,
            final_cost,
            origin_postcode,
            destination_postcode,
            origin_lat,
            origin_lon,
            dest_lat,
            dest_lon,
            created_timestamp,
            timestamp,
        ),
    )
    conn.commit()
    return conn.execute(
        "SELECT * FROM jobs WHERE job_number = ?", (cleaned_job_number,)
    ).fetchone()


def upsert_job_segment(
    conn: sqlite3.Connection,
    *,
    job_id: int,
    segment_sequence: int,
    origin: str | None = None,
    destination: str | None = None,
    mode: str | None = None,
    status: str | None = None,
    distance_km: float | None = None,
    client_reference: str | None = None,
    created_at: str | None = None,
    updated_at: str | None = None,
) -> sqlite3.Row:
    """Insert or update a job segment keyed by job and sequence."""

    timestamp = updated_at or datetime.now(UTC).isoformat()
    created_timestamp = created_at or timestamp
    conn.execute(
        """
        INSERT INTO job_segments (
            job_id, segment_sequence, origin, destination, mode, status,
            distance_km, client_reference, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(job_id, segment_sequence) DO UPDATE SET
            origin = excluded.origin,
            destination = excluded.destination,
            mode = excluded.mode,
            status = excluded.status,
            distance_km = excluded.distance_km,
            client_reference = excluded.client_reference,
            updated_at = excluded.updated_at,
            created_at = COALESCE(job_segments.created_at, excluded.created_at)
        """,
        (
            job_id,
            segment_sequence,
            origin,
            destination,
            mode,
            status,
            distance_km,
            client_reference,
            created_timestamp,
            timestamp,
        ),
    )
    conn.commit()
    return conn.execute(
        "SELECT * FROM job_segments WHERE job_id = ? AND segment_sequence = ?",
        (job_id, segment_sequence),
    ).fetchone()


def upsert_container_booking(
    conn: sqlite3.Connection,
    *,
    booking_reference: str,
    job_id: int | None = None,
    client_reference: str | None = None,
    status: str | None = None,
    created_at: str | None = None,
    updated_at: str | None = None,
) -> sqlite3.Row:
    """Insert or update a container booking keyed by ``booking_reference``."""

    timestamp = updated_at or datetime.now(UTC).isoformat()
    created_timestamp = created_at or timestamp
    conn.execute(
        """
        INSERT INTO container_bookings (
            booking_reference, job_id, client_reference, status, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(booking_reference) DO UPDATE SET
            job_id = COALESCE(excluded.job_id, container_bookings.job_id),
            client_reference = excluded.client_reference,
            status = excluded.status,
            updated_at = excluded.updated_at,
            created_at = COALESCE(container_bookings.created_at, excluded.created_at)
        """,
        (
            booking_reference,
            job_id,
            client_reference,
            status,
            created_timestamp,
            timestamp,
        ),
    )
    conn.commit()
    return conn.execute(
        "SELECT * FROM container_bookings WHERE booking_reference = ?",
        (booking_reference,),
    ).fetchone()


def upsert_container(
    conn: sqlite3.Connection,
    *,
    container_number: str,
    booking_id: int | None = None,
    job_id: int | None = None,
    client_reference: str | None = None,
    status: str | None = None,
    created_at: str | None = None,
    updated_at: str | None = None,
) -> sqlite3.Row:
    """Insert or update a container keyed by container number."""

    timestamp = updated_at or datetime.now(UTC).isoformat()
    created_timestamp = created_at or timestamp
    conn.execute(
        """
        INSERT INTO containers (
            container_number, booking_id, job_id, client_reference, status, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(container_number) DO UPDATE SET
            booking_id = COALESCE(excluded.booking_id, containers.booking_id),
            job_id = COALESCE(excluded.job_id, containers.job_id),
            client_reference = excluded.client_reference,
            status = excluded.status,
            updated_at = excluded.updated_at,
            created_at = COALESCE(containers.created_at, excluded.created_at)
        """,
        (
            container_number,
            booking_id,
            job_id,
            client_reference,
            status,
            created_timestamp,
            timestamp,
        ),
    )
    conn.commit()
    return conn.execute(
        "SELECT * FROM containers WHERE container_number = ?", (container_number,)
    ).fetchone()


def upsert_job_container_allocation(
    conn: sqlite3.Connection,
    *,
    job_id: int,
    booking_id: int,
    segment_id: int | None = None,
    volume_share: float | None = None,
    weight_share: float | None = None,
) -> sqlite3.Row:
    """Insert or update a job container allocation keyed by job, booking, and segment."""

    if segment_id is not None:
        segment_row = conn.execute(
            "SELECT job_id FROM job_segments WHERE id = ?", (segment_id,)
        ).fetchone()
        if segment_row is None:
            raise ValueError(f"Job segment {segment_id} does not exist")
        if segment_row[0] != job_id:
            raise ValueError("Segment does not belong to the specified job")

    conn.execute(
        """
        INSERT INTO job_container_allocations (
            job_id, booking_id, segment_id, volume_share, weight_share
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(job_id, booking_id, segment_id) DO UPDATE SET
            volume_share = excluded.volume_share,
            weight_share = excluded.weight_share
        """,
        (job_id, booking_id, segment_id, volume_share, weight_share),
    )
    conn.commit()
    return conn.execute(
        """
        SELECT job_id, booking_id, segment_id, volume_share, weight_share
        FROM job_container_allocations
        WHERE job_id = ? AND booking_id = ? AND (
            (segment_id IS NULL AND ? IS NULL) OR segment_id = ?
        )
        """,
        (job_id, booking_id, segment_id, segment_id),
    ).fetchone()


def _resolve_shift_job_id(
    conn: sqlite3.Connection, job_id: int | None, shipment_id: int | None
) -> int | None:
    """Validate and resolve the job linked to a driver shift."""

    resolved_job_id = job_id
    shipment_row = None

    if shipment_id is not None:
        shipment_row = conn.execute(
            "SELECT id, job_id FROM shipments WHERE id = ?", (shipment_id,)
        ).fetchone()
        if shipment_row is None:
            raise ValueError(f"Shipment {shipment_id} does not exist")
        shipment_job_id = shipment_row[1]
        if resolved_job_id is None:
            resolved_job_id = shipment_job_id
        elif shipment_job_id is not None and shipment_job_id != resolved_job_id:
            raise ValueError(
                "Shipment is linked to a different job than the shift specifies"
            )

    if resolved_job_id is not None:
        job_row = conn.execute("SELECT 1 FROM jobs WHERE id = ?", (resolved_job_id,)).fetchone()
        if job_row is None:
            raise ValueError(f"Job {resolved_job_id} does not exist")

    return resolved_job_id


def upsert_driver_shift(
    conn: sqlite3.Connection,
    *,
    shift_date: str,
    truck_id: str | None = None,
    worker_name: str | None = None,
    ticket_numbers: str | None = None,
    shift_start: str | None = None,
    shift_end: str | None = None,
    shift_window_start: str | None = None,
    shift_window_end: str | None = None,
    role: str | None = None,
    hours: float | None = None,
    hourly_rate: float | None = None,
    cost_total: float | None = None,
    job_id: int | None = None,
    shipment_id: int | None = None,
    notes: str | None = None,
    source: str | None = None,
    imported_at: str | None = None,
) -> tuple[sqlite3.Row, bool]:
    """Insert or update a driver shift entry keyed by date/vehicle/worker/time."""

    _ensure_driver_shift_columns(conn)

    worker_id: int | None = None
    if worker_name:
        worker = upsert_worker(conn, name=worker_name)
        worker_id = int(worker["id"])

    resolved_job_id = _resolve_shift_job_id(conn, job_id, shipment_id)

    calculated_cost = cost_total
    if calculated_cost is None and hours is not None and hourly_rate is not None:
        calculated_cost = float(hours) * float(hourly_rate)

    timestamp = imported_at or datetime.now(UTC).isoformat()
    previous_row = conn.execute(
        """
        SELECT * FROM driver_shifts
        WHERE shift_date = ? AND truck_id IS ? AND worker_id IS ?
          AND shift_start IS ? AND shift_end IS ? AND ticket_numbers IS ?
        """,
        (shift_date, truck_id, worker_id, shift_start, shift_end, ticket_numbers),
    ).fetchone()

    conn.execute(
        """
        INSERT INTO driver_shifts (
            shift_date, truck_id, worker_id, job_id, shipment_id, ticket_numbers,
            shift_start, shift_end, shift_window_start, shift_window_end, role,
            hours, hourly_rate, cost_total, notes, source, imported_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(shift_date, truck_id, worker_id, shift_start, shift_end, ticket_numbers)
        DO UPDATE SET
            job_id = excluded.job_id,
            shipment_id = excluded.shipment_id,
            shift_start = excluded.shift_start,
            shift_end = excluded.shift_end,
            shift_window_start = excluded.shift_window_start,
            shift_window_end = excluded.shift_window_end,
            role = excluded.role,
            hours = excluded.hours,
            hourly_rate = excluded.hourly_rate,
            cost_total = excluded.cost_total,
            notes = excluded.notes,
            source = excluded.source,
            imported_at = excluded.imported_at,
            truck_id = excluded.truck_id,
            worker_id = excluded.worker_id,
            ticket_numbers = excluded.ticket_numbers
        """,
        (
            shift_date,
            truck_id,
            worker_id,
            resolved_job_id,
            shipment_id,
            ticket_numbers,
            shift_start,
            shift_end,
            shift_window_start,
            shift_window_end,
            role,
            hours,
            hourly_rate,
            calculated_cost,
            notes,
            source,
            timestamp,
        ),
    )
    conn.commit()
    row = conn.execute(
        """
        SELECT * FROM driver_shifts
        WHERE shift_date = ? AND truck_id IS ? AND worker_id IS ?
          AND shift_start IS ? AND shift_end IS ? AND ticket_numbers IS ?
        """,
        (shift_date, truck_id, worker_id, shift_start, shift_end, ticket_numbers),
    ).fetchone()
    return row, previous_row is None


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


def _resolve_shipment_locations(
    conn: sqlite3.Connection,
    *,
    job_id: int | None,
    historical_job_id: int | None,
    from_location: str | None,
    to_location: str | None,
) -> tuple[str | None, str | None]:
    """Return shipment locations preferring explicit values then job origins/destinations."""

    if from_location is not None and to_location is not None:
        return from_location, to_location

    resolved_from = from_location
    resolved_to = to_location

    if resolved_from is None or resolved_to is None:
        if job_id is not None:
            row = conn.execute(
                "SELECT origin, destination FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is not None:
                resolved_from = resolved_from or row[0]
                resolved_to = resolved_to or row[1]
        if (resolved_from is None or resolved_to is None) and historical_job_id is not None:
            row = conn.execute(
                "SELECT origin, destination FROM historical_jobs WHERE id = ?",
                (historical_job_id,),
            ).fetchone()
            if row is not None:
                resolved_from = resolved_from or row[0]
                resolved_to = resolved_to or row[1]

    return resolved_from, resolved_to


def create_shipment(
    conn: sqlite3.Connection,
    *,
    job_id: int | None = None,
    historical_job_id: int | None = None,
    inventory_item_id: int | None = None,
    truck_id: str | None = None,
    worker_id: int | None = None,
    segment_id: int | None = None,
    segment_sequence: int | None = None,
    worker_role_id: int | None = None,
    worker_start_time: str | None = None,
    worker_end_time: str | None = None,
    vehicle_requirement_met: bool | None = None,
    quantity: float | None = None,
    from_location: str | None = None,
    to_location: str | None = None,
    status: str = "planned",
    scheduled_date: str | None = None,
    delivered_at: str | None = None,
    reserve_in_transit: bool = True,
) -> sqlite3.Row:
    """Insert a shipment linked to a job or historical job.

    If ``inventory_item_id`` is provided, ensure the requested ``quantity`` is
    available before creating the shipment and recording an inventory movement.
    When ``reserve_in_transit`` is True, the quantity is tracked as allocated in
    transit rather than deducted from on-hand stock.
    """

    _ensure_job_segment_tables(conn)

    if job_id is None and historical_job_id is None and segment_id is None:
        raise ValueError("Shipments must reference a job, segment, or historical job")

    segment_row: sqlite3.Row | None = None
    resolved_job_id = job_id
    if segment_id is not None:
        segment_row = conn.execute(
            "SELECT * FROM job_segments WHERE id = ?",
            (segment_id,),
        ).fetchone()
        if segment_row is None:
            raise ValueError(f"Segment {segment_id} does not exist")
        segment_job_id = int(segment_row["job_id"])
        if resolved_job_id is None:
            resolved_job_id = segment_job_id
        elif resolved_job_id != segment_job_id:
            raise ValueError("Segment belongs to a different job")

    if resolved_job_id is None and segment_row is None and segment_sequence is not None:
        raise ValueError("segment_sequence requires a job_id when no segment_id is provided")

    if segment_row is None and resolved_job_id is not None:
        segment_row = get_or_create_job_segment(
            conn,
            job_id=resolved_job_id,
            segment_sequence=segment_sequence or 1,
            from_location=from_location,
            to_location=to_location,
        )
    segment_id_value = int(segment_row["id"]) if segment_row is not None else None
    job_id = resolved_job_id

    resolved_from, resolved_to = _resolve_shipment_locations(
        conn,
        job_id=job_id,
        historical_job_id=historical_job_id,
        from_location=from_location,
        to_location=to_location,
    )

    if segment_row is not None:
        resolved_from = resolved_from or segment_row["from_location"]
        resolved_to = resolved_to or segment_row["to_location"]
        if segment_row["from_location"] is None and resolved_from is not None:
            conn.execute(
                "UPDATE job_segments SET from_location = ? WHERE id = ?",
                (resolved_from, segment_row["id"]),
            )
        if segment_row["to_location"] is None and resolved_to is not None:
            conn.execute(
                "UPDATE job_segments SET to_location = ? WHERE id = ?",
                (resolved_to, segment_row["id"]),
            )

    quantity_value = 1.0 if quantity is None else float(quantity)
    timestamp = datetime.now(UTC).isoformat()
    conn.execute(
        """
        INSERT INTO shipments (
            job_id,
            historical_job_id,
            inventory_item_id,
            truck_id,
            worker_id,
            segment_id,
            quantity,
            from_location,
            to_location,
            status,
            scheduled_date,
            delivered_at,
            created_at,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            job_id,
            historical_job_id,
            inventory_item_id,
            truck_id,
            worker_id,
            segment_id_value,
            quantity_value,
            resolved_from,
            resolved_to,
            status,
            scheduled_date,
            delivered_at,
            timestamp,
            timestamp,
        ),
    )
    shipment_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    if inventory_item_id is not None and reserve_in_transit:
        record_inventory_movement(
            conn,
            inventory_item_id=inventory_item_id,
            shipment_id=shipment_id,
            change_allocated=-int(quantity_value),
            reason="Shipment created",
            commit=False,
        )
    if segment_id_value is not None and truck_id:
        _link_vehicle_to_segment(
            conn,
            segment_id=segment_id_value,
            truck_id=truck_id,
            requirement_met=bool(vehicle_requirement_met)
            if vehicle_requirement_met is not None
            else False,
        )
    if segment_id_value is not None and worker_id:
        _link_worker_to_segment(
            conn,
            segment_id=segment_id_value,
            worker_id=worker_id,
            start_time=worker_start_time,
            end_time=worker_end_time,
            role_id=worker_role_id,
        )
    conn.commit()
    return conn.execute(
        "SELECT * FROM shipments WHERE id = ?", (shipment_id,)
    ).fetchone()


def fetch_driver_shifts(
    conn: sqlite3.Connection,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    worker_names: Sequence[str] | None = None,
    truck_ids: Sequence[str] | None = None,
) -> list[sqlite3.Row]:
    """Return driver shifts matching the provided filters."""

    filters: list[str] = []
    params: list[object] = []

    if start_date:
        filters.append("ds.shift_date >= ?")
        params.append(start_date)
    if end_date:
        filters.append("ds.shift_date <= ?")
        params.append(end_date)
    if worker_names:
        filters.append("w.name IN (" + ",".join(["?"] * len(worker_names)) + ")")
        params.extend(worker_names)
    if truck_ids:
        filters.append("ds.truck_id IN (" + ",".join(["?"] * len(truck_ids)) + ")")
        params.extend(truck_ids)

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    query = f"""
        SELECT
            ds.*, w.name AS worker_name, t.name AS truck_name,
            COALESCE(ds.job_id, s.job_id) AS linked_job_id,
            s.job_id AS shipment_job_id,
            j.origin AS job_origin,
            j.destination AS job_destination
        FROM driver_shifts ds
        LEFT JOIN workers w ON ds.worker_id = w.id
        LEFT JOIN trucks t ON ds.truck_id = t.truck_id
        LEFT JOIN shipments s ON ds.shipment_id = s.id
        LEFT JOIN jobs j ON COALESCE(ds.job_id, s.job_id) = j.id
        {where_clause}
        ORDER BY ds.shift_date DESC, ds.shift_start
    """
    return list(conn.execute(query, params))


def rollup_driver_shift_costs_by_job(
    conn: sqlite3.Connection,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[sqlite3.Row]:
    """Aggregate driver shift hours and costs grouped by job."""

    filters: list[str] = ["COALESCE(ds.job_id, s.job_id) IS NOT NULL"]
    params: list[object] = []

    if start_date:
        filters.append("ds.shift_date >= ?")
        params.append(start_date)
    if end_date:
        filters.append("ds.shift_date <= ?")
        params.append(end_date)

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    query = f"""
        SELECT
            COALESCE(ds.job_id, s.job_id) AS job_id,
            COUNT(*) AS shift_count,
            SUM(ds.hours) AS total_hours,
            SUM(ds.cost_total) AS total_cost
        FROM driver_shifts ds
        LEFT JOIN shipments s ON ds.shipment_id = s.id
        {where_clause}
        GROUP BY COALESCE(ds.job_id, s.job_id)
        ORDER BY job_id
    """
    return list(conn.execute(query, params))


def fetch_shipments_with_context(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return shipments joined with job, inventory, and assignment context."""

    query = """
        SELECT
            s.id,
            s.quantity,
            s.from_location,
            s.to_location,
            s.status,
            s.scheduled_date,
            s.delivered_at,
            s.job_id,
            s.historical_job_id,
            s.segment_id,
            js.segment_sequence,
            js.from_location AS segment_from_location,
            js.to_location AS segment_to_location,
            j.origin AS job_origin,
            j.destination AS job_destination,
            h.origin AS historical_origin,
            h.destination AS historical_destination,
            i.name AS inventory_name,
            i.quantity AS inventory_quantity,
            sup.company_name AS supplier_company_name,
            sup.contact_name AS supplier_contact_name,
            sup.contact_number AS supplier_contact_number,
            sup.email AS supplier_email,
            COALESCE(
                GROUP_CONCAT(DISTINCT seg_trucks.truck_id),
                legacy_truck.truck_id
            ) AS truck_id,
            COALESCE(
                GROUP_CONCAT(DISTINCT seg_trucks.name),
                legacy_truck.name
            ) AS truck_name,
            COALESCE(
                GROUP_CONCAT(DISTINCT workers.name),
                legacy_worker.name
            ) AS worker_name,
            COALESCE(
                GROUP_CONCAT(DISTINCT workers.role),
                legacy_worker.role
            ) AS worker_role,
            COALESCE(MAX(workers.rate), legacy_worker.rate) AS worker_rate,
            COALESCE(MAX(workers.tickets), legacy_worker.tickets) AS worker_tickets,
            GROUP_CONCAT(DISTINCT worker_roles.name) AS worker_role_names,
            GROUP_CONCAT(DISTINCT seg_trucks.truck_id) AS segment_truck_ids,
            GROUP_CONCAT(DISTINCT seg_trucks.name) AS segment_truck_names,
            GROUP_CONCAT(DISTINCT workers.name) AS segment_worker_names
        FROM shipments AS s
        LEFT JOIN jobs AS j ON s.job_id = j.id
        LEFT JOIN historical_jobs AS h ON s.historical_job_id = h.id
        LEFT JOIN inventory_items AS i ON s.inventory_item_id = i.id
        LEFT JOIN suppliers AS sup ON i.supplier_id = sup.id
        LEFT JOIN job_segments AS js ON s.segment_id = js.id
        LEFT JOIN job_segment_vehicles AS seg_vehicles ON js.id = seg_vehicles.segment_id
        LEFT JOIN trucks AS seg_trucks ON seg_vehicles.truck_id = seg_trucks.truck_id
        LEFT JOIN trucks AS legacy_truck ON s.truck_id = legacy_truck.truck_id
        LEFT JOIN job_segment_workers AS seg_workers ON js.id = seg_workers.segment_id
        LEFT JOIN workers ON seg_workers.worker_id = workers.id
        LEFT JOIN worker_roles ON seg_workers.role_id = worker_roles.id
        LEFT JOIN workers AS legacy_worker ON s.worker_id = legacy_worker.id
        GROUP BY s.id
        ORDER BY s.id
    """
    return list(conn.execute(query))
