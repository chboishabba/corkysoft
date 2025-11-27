"""Database helpers for analytics features."""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from urllib.parse import quote_plus
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


def _table_columns(conn: sqlite3.Connection, table: str) -> Sequence[str]:
    """Return the column names for *table* in the current connection."""
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [row[1] for row in rows]


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
    quantity: int = 1,
    truck_id: str | None = None,
    worker_id: int | None = None,
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

    if job_id is None and historical_job_id is None:
        raise ValueError("Shipments must reference a job or historical job")

    resolved_from, resolved_to = _resolve_shipment_locations(
        conn,
        job_id=job_id,
        historical_job_id=historical_job_id,
        from_location=from_location,
        to_location=to_location,
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
            quantity,
            from_location,
            to_location,
            status,
            scheduled_date,
            delivered_at,
            created_at,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            job_id,
            historical_job_id,
            inventory_item_id,
            truck_id,
            worker_id,
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
        LEFT JOIN suppliers AS sup ON i.supplier_id = sup.id
        LEFT JOIN trucks AS t ON s.truck_id = t.truck_id
        LEFT JOIN workers AS w ON s.worker_id = w.id
        ORDER BY s.id
    """
    return list(conn.execute(query))
