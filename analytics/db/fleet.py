"""Fleet and worker-related database functions."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import IO

import pandas as pd

from analytics.google_sheets import (
    build_google_sheet_xlsx_url,
    resolve_google_sheet_reference,
)
from .schema import _ensure_vehicle_details_table

__all__ = [
    "import_workers_from_dataframe",
    "import_workers_from_google_sheet",
    "upsert_truck",
    "upsert_vehicle_details",
    "upsert_worker",
    "import_workers_from_staff_sheet",
]


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
    source_system: str | None = None,
    source_sheet: str | None = None,
    source_imported_at: str | None = None,
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
            daily_check_complete,
            source_system,
            source_sheet,
            source_imported_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            daily_check_complete = excluded.daily_check_complete,
            source_system = COALESCE(excluded.source_system, vehicle_details.source_system),
            source_sheet = COALESCE(excluded.source_sheet, vehicle_details.source_sheet),
            source_imported_at = COALESCE(excluded.source_imported_at, vehicle_details.source_imported_at)
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
            source_system,
            source_sheet,
            source_imported_at,
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
    source_system: str | None = None,
    source_sheet: str | None = None,
    source_imported_at: str | None = None,
) -> sqlite3.Row:
    """Create or update a worker record based on the unique name."""

    timestamp = datetime.now(UTC).isoformat()
    rate_value = float(rate) if rate is not None else None
    tickets_value = int(tickets) if tickets is not None else None
    conn.execute(
        """
        INSERT INTO workers (
            name, role, phone, rate, tickets, active,
            source_system, source_sheet, source_imported_at, hired_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            role = excluded.role,
            phone = excluded.phone,
            rate = excluded.rate,
            tickets = excluded.tickets,
            active = excluded.active,
            source_system = COALESCE(excluded.source_system, workers.source_system),
            source_sheet = COALESCE(excluded.source_sheet, workers.source_sheet),
            source_imported_at = COALESCE(excluded.source_imported_at, workers.source_imported_at),
            updated_at = excluded.updated_at
        """,
        (
            name,
            role,
            phone,
            rate_value,
            tickets_value,
            int(active),
            source_system,
            source_sheet,
            source_imported_at,
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


def import_workers_from_dataframe(
    conn: sqlite3.Connection,
    dataframe: pd.DataFrame,
    *,
    source_sheet: str = "STAFF",
) -> tuple[int, int]:
    """Import or update worker records from a normalised STAFF dataframe."""

    inserted = 0
    updated = 0

    for _, row in dataframe.iterrows():
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
            role=str(row.get("ROLE") or ""),
            phone=str(row.get("PHONE") or ""),
            rate=rate_value,
            tickets=tickets_value,
            active=str(row.get("ACTIVE") or "Yes").strip().lower() != "no",
            source_system="google_sheets",
            source_sheet=source_sheet,
            source_imported_at=datetime.now(UTC).isoformat(),
        )

        if existing:
            updated += 1
        else:
            inserted += 1

    return inserted, updated


def import_workers_from_staff_sheet(
    conn: sqlite3.Connection,
    workbook: str | bytes | IO[bytes],
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
    return import_workers_from_dataframe(conn, df, source_sheet=sheet_name)


def import_workers_from_google_sheet(
    conn: sqlite3.Connection,
    *,
    sheet_id_or_url: str | None = None,
    sheet_name: str = "STAFF",
) -> tuple[int, int]:
    """Import staff from a Google Sheets workbook via XLSX export."""

    reference = resolve_google_sheet_reference(
        sheet_id_or_url,
        env_keys=(
            "STAFF_SHEET_ID",
            "STAFF_SHEET_URL",
            "OPERATIONS_WORKBOOK_SHEET_ID",
            "OPERATIONS_WORKBOOK_URL",
        ),
    )
    if not reference:
        raise ValueError(
            "Provide a STAFF sheet/workbook reference or set OPERATIONS_WORKBOOK_SHEET_ID/URL."
        )
    workbook_url = build_google_sheet_xlsx_url(reference)
    df = pd.read_excel(workbook_url, sheet_name=sheet_name)
    return import_workers_from_dataframe(conn, df, source_sheet=sheet_name)


__all__ = [
    "import_workers_from_dataframe",
    "import_workers_from_google_sheet",
    "upsert_truck",
    "upsert_vehicle_details",
    "upsert_worker",
    "import_workers_from_staff_sheet",
]
