"""Schema definitions and migration helpers for the dashboard database."""
from __future__ import annotations
from __future__ import annotations

import sqlite3

from .connection import _table_columns, _table_exists

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
    FOREIGN KEY(worker_id) REFERENCES workers(id) ON DELETE SET NULL
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
    # _ensure_inventory_movements_table(conn)

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
    _ensure_vehicle_details_table(conn)
    _ensure_shipment_columns(conn)
    conn.commit()

    for table_name in (
        "inventory_items",
        "inventory_movements",
        "workers",
        "trucks",
        "vehicle_repairs",
        "shipments",
    ):
        if not _table_exists(conn, table_name):
            conn.execute(
                f"SELECT RAISE(FAIL, 'Failed to create {table_name} during bootstrap')"
            )


def _ensure_shipment_columns(conn: sqlite3.Connection) -> None:
    """Ensure shipment columns for quantities and locations exist and are populated."""

    if not _table_exists(conn, "shipments"):
        return

    columns = set(_table_columns(conn, "shipments"))
    declarations = {
        "quantity": "REAL NOT NULL DEFAULT 1",
        "from_location": "TEXT",
        "to_location": "TEXT",
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
