"""Helpers for importing and reporting vehicle repair history."""
from __future__ import annotations

import os
import re
import sqlite3
from datetime import UTC, datetime
from typing import Iterable, Optional

import pandas as pd

from analytics.db import ensure_dashboard_tables


_SHEET_ENV_KEYS: tuple[str, ...] = ("VEHICLE_REPAIRS_SHEET_URL", "VEHICLE_REPAIRS_SHEET")
_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "truck_id": ("truck", "vehicle", "vehicle id", "truck id"),
    "job_item": ("job item", "item", "job"),
    "description": ("description", "details", "job description"),
    "price": ("price", "cost", "amount"),
    "supplier": ("supplier", "vendor", "workshop"),
    "service_date": ("service date", "date", "completed", "completed date"),
    "next_service_date": ("next service", "next due", "due date", "next date"),
    "notes": ("notes", "note", "comments"),
}


def _normalise_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.strip().lower()).strip()


def _resolve_column(columns: Iterable[str], aliases: tuple[str, ...]) -> Optional[str]:
    normalised = {_normalise_column_name(col): col for col in columns}
    for alias in aliases:
        if alias in normalised:
            return normalised[alias]
    return None


def _clean_string(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else None
    return str(value).strip() or None


def _parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        float_value = float(value)
    except (TypeError, ValueError):
        return None
    return float_value


def _parse_date_string(value: object) -> Optional[str]:
    if value is None:
        return None
    try:
        parsed = pd.to_datetime(value, errors="coerce")
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    if isinstance(parsed, pd.Timestamp):
        return parsed.date().isoformat()
    if isinstance(parsed, datetime):
        return parsed.date().isoformat()
    return None


def _load_sheet_dataframe(sheet_url: str) -> pd.DataFrame:
    try:
        return pd.read_csv(sheet_url)
    except Exception as csv_exc:
        try:
            return pd.read_excel(sheet_url)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load VEHICLE_REPAIRS sheet from {sheet_url}: {exc}"
            ) from csv_exc


def _build_existing_key(
    truck_id: str, job_item: str, service_date: Optional[str], price: Optional[float]
) -> tuple[str, str, str, Optional[float]]:
    return (
        truck_id.strip().lower(),
        job_item.strip().lower(),
        (service_date or "").strip(),
        price if price is None else float(price),
    )


def import_vehicle_repairs_from_dataframe(
    conn: sqlite3.Connection, df: pd.DataFrame
) -> tuple[int, int]:
    """Insert or update vehicle repair rows from ``df``.

    Returns ``(inserted, updated)`` counts. Rows missing a truck/vehicle ID or
    job item are skipped. Dates are stored as ISO strings when parsable.
    """

    ensure_dashboard_tables(conn)

    if df.empty:
        return 0, 0

    truck_col = _resolve_column(df.columns, ("truck_id",) + _COLUMN_ALIASES["truck_id"])
    job_col = _resolve_column(df.columns, ("job_item",) + _COLUMN_ALIASES["job_item"])
    if not truck_col or not job_col:
        raise ValueError("Sheet must include truck/vehicle and job item columns.")

    description_col = _resolve_column(df.columns, ("description",) + _COLUMN_ALIASES["description"])
    price_col = _resolve_column(df.columns, ("price",) + _COLUMN_ALIASES["price"])
    supplier_col = _resolve_column(df.columns, ("supplier",) + _COLUMN_ALIASES["supplier"])
    service_date_col = _resolve_column(
        df.columns, ("service_date",) + _COLUMN_ALIASES["service_date"]
    )
    next_service_col = _resolve_column(
        df.columns, ("next_service_date",) + _COLUMN_ALIASES["next_service_date"]
    )
    notes_col = _resolve_column(df.columns, ("notes",) + _COLUMN_ALIASES["notes"])

    existing_rows = conn.execute(
        "SELECT id, truck_id, job_item, service_date, price FROM vehicle_repairs"
    ).fetchall()
    existing_keys = {
        _build_existing_key(row[1], row[2], row[3], row[4]): row[0]
        for row in existing_rows
    }

    inserted = 0
    updated = 0
    timestamp = datetime.now(UTC).isoformat()

    for idx in range(len(df)):
        truck_value = _clean_string(df.iloc[idx][truck_col])
        job_item_value = _clean_string(df.iloc[idx][job_col])
        if not truck_value or not job_item_value:
            continue

        description_value = (
            _clean_string(df.iloc[idx][description_col]) if description_col else None
        )
        price_value = _parse_float(df.iloc[idx][price_col]) if price_col else None
        supplier_value = _clean_string(df.iloc[idx][supplier_col]) if supplier_col else None
        service_date_value = (
            _parse_date_string(df.iloc[idx][service_date_col])
            if service_date_col
            else None
        )
        next_service_value = (
            _parse_date_string(df.iloc[idx][next_service_col]) if next_service_col else None
        )
        notes_value = _clean_string(df.iloc[idx][notes_col]) if notes_col else None

        key = _build_existing_key(
            truck_value, job_item_value, service_date_value, price_value
        )
        payload = (
            truck_value,
            job_item_value,
            description_value,
            price_value,
            supplier_value,
            service_date_value,
            next_service_value,
            notes_value,
            timestamp,
            timestamp,
        )

        if key in existing_keys:
            conn.execute(
                """
                UPDATE vehicle_repairs
                SET description = ?,
                    price = ?,
                    supplier = ?,
                    service_date = ?,
                    next_service_date = ?,
                    notes = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    description_value,
                    price_value,
                    supplier_value,
                    service_date_value,
                    next_service_value,
                    notes_value,
                    timestamp,
                    existing_keys[key],
                ),
            )
            updated += 1
        else:
            conn.execute(
                """
                INSERT INTO vehicle_repairs (
                    truck_id,
                    job_item,
                    description,
                    price,
                    supplier,
                    service_date,
                    next_service_date,
                    notes,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
            inserted += 1
            new_row_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            existing_keys[key] = new_row_id

    conn.commit()
    return inserted, updated


def import_vehicle_repairs_from_sheet(
    conn: sqlite3.Connection, *, sheet_url: Optional[str] = None
) -> tuple[int, int]:
    """Fetch the VEHICLE_REPAIRS sheet and import its contents."""

    resolved_url = sheet_url or next(
        (os.environ.get(key) for key in _SHEET_ENV_KEYS if os.environ.get(key)), None
    )
    if not resolved_url:
        raise RuntimeError(
            "Provide a VEHICLE_REPAIRS sheet URL via sheet_url or environment variables."
        )

    df = _load_sheet_dataframe(resolved_url)
    return import_vehicle_repairs_from_dataframe(conn, df)


def load_vehicle_repairs(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return all recorded vehicle repairs ordered by service date."""

    ensure_dashboard_tables(conn)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM vehicle_repairs ORDER BY service_date DESC, created_at DESC",
            conn,
        )
    except Exception:
        return pd.DataFrame(
            columns=[
                "truck_id",
                "job_item",
                "description",
                "price",
                "supplier",
                "service_date",
                "next_service_date",
                "notes",
                "created_at",
                "updated_at",
            ]
        )
    return df


__all__ = [
    "import_vehicle_repairs_from_dataframe",
    "import_vehicle_repairs_from_sheet",
    "load_vehicle_repairs",
]
