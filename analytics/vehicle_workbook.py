"""Helpers for importing fleet vehicle details from Google Sheets workbooks."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Iterable, Mapping, Sequence

import pandas as pd

from analytics.db import ensure_dashboard_tables, upsert_truck, upsert_vehicle_details

GOOGLE_SHEET_EXPORT = "https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"
DEFAULT_VEHICLE_SHEET_HINTS: Sequence[str] = ("vehicle", "fleet")

_COLUMN_ALIASES: Mapping[str, str] = {
    "state": "state",
    "rego": "rego",
    "rego expiry date": "rego_expiry",
    "rego expiry date?": "rego_expiry",
    "make": "make",
    "model": "model",
    "year": "year",
    "body type": "body_type",
    "description": "description",
    "nhv charging code": "nhv_code",
    "nhv code": "nhv_code",
    "insurance type": "insurance",
    "insurance": "insurance",
    "odometer": "odometer",
    "last service": "last_service",
    "next service due": "next_service",
    "coi number": "coi_number",
    "certificate of inspection (coi) due date": "coi_due",
    "coi due date": "coi_due",
    "present driver": "present_driver",
    "daily complete": "daily_check_complete",
    "daily complete?": "daily_check_complete",
}


def _normalise_label(label: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", label.strip().lower())
    return cleaned.strip()


def _resolve_sheet_name(sheet_names: Sequence[str], hints: Sequence[str]) -> str:
    normalised = {_normalise_label(name): name for name in sheet_names}
    for hint in hints:
        target = _normalise_label(hint)
        for normalised_name, original in normalised.items():
            if normalised_name.startswith(target):
                return original
    raise ValueError(
        f"Could not find a vehicle worksheet matching {', '.join(hints)}; "
        f"available sheets: {', '.join(sheet_names)}"
    )


def _coerce_date(value: object) -> str | None:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    if isinstance(timestamp, pd.Timestamp):
        timestamp = timestamp.to_pydatetime()
    if isinstance(timestamp, datetime):
        return timestamp.date().isoformat()
    return None


def _coerce_int(value: object) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return int(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, (int, bool)):
        return bool(value)
    text = str(value).strip().lower()
    if not text or text in {"nan", "none"}:
        return None
    return text in {"y", "yes", "true", "1"}


def _extract_sheet_row(workbook: pd.ExcelFile, sheet_name: str) -> Mapping[str, object]:
    """Return a mapping of column labels to values from a two-row vehicle sheet.

    VEHICLE_DETAIL workbooks store each vehicle's metadata on its own worksheet
    with headers on the first row and values on the second. Columns are often
    offset (e.g., starting at column B) and may contain trailing blank columns.
    """

    sheet = workbook.parse(sheet_name, header=None)
    sheet = sheet.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if sheet.shape[0] < 2:
        return {}

    headers = sheet.iloc[0]
    values = sheet.iloc[1]
    row: dict[str, object] = {}
    for header, value in zip(headers, values):
        if isinstance(header, str) and header.strip():
            row[header] = value
    return row


def _collect_vehicle_rows_from_workbook(workbook: pd.ExcelFile) -> pd.DataFrame:
    """Gather vehicle rows from multi-sheet VEHICLE_DETAIL workbooks."""

    rows: list[Mapping[str, object]] = []
    for sheet_name in workbook.sheet_names:
        if sheet_name.strip().lower() in {"index", "log", "init"}:
            continue

        row = _extract_sheet_row(workbook, sheet_name)
        if not row:
            continue

        mapping = _canonical_column_mapping(row.keys())
        if not mapping:
            continue

        if "rego" not in mapping.values():
            row["REGO"] = sheet_name

        rows.append(row)

    return pd.DataFrame(rows)


def _canonical_column_mapping(columns: Iterable[str]) -> Mapping[str, str]:
    mapping = {}
    for column in columns:
        canonical = _COLUMN_ALIASES.get(_normalise_label(column))
        if canonical:
            mapping[column] = canonical
    return mapping


def _compose_truck_name(row: Mapping[str, object]) -> str | None:
    make = str(row.get("make") or "").strip()
    model = str(row.get("model") or "").strip()
    year = row.get("year")
    parts = [part for part in (make, model) if part]
    if year and not pd.isna(year):
        try:
            parts.append(str(int(year)))
        except (TypeError, ValueError):
            parts.append(str(year).strip())
    if parts:
        return " ".join(parts)
    description = row.get("description")
    if isinstance(description, str) and description.strip():
        return description.strip()
    return None


def import_vehicle_details_from_dataframe(
    conn,
    df: pd.DataFrame,
) -> int:
    """Import vehicle metadata from a worksheet DataFrame.

    The importer tolerates inconsistent column naming by normalising headers and
    mapping them to canonical database fields. ``rego`` acts as the ``truck_id``
    and is required for a row to be persisted.
    """

    ensure_dashboard_tables(conn)
    if df.empty:
        return 0

    column_mapping = _canonical_column_mapping(df.columns)
    if not column_mapping:
        raise ValueError("No recognised VEHICLE columns found in the workbook")

    prepared = df.rename(columns=column_mapping)
    inserted = 0
    for _, row in prepared.iterrows():
        rego = str(row.get("rego") or "").strip()
        if not rego:
            continue

        truck_name = _compose_truck_name(row)
        upsert_truck(conn, truck_id=rego, name=truck_name, active=True, notes=row.get("description"))

        upsert_vehicle_details(
            conn,
            truck_id=rego,
            state=(row.get("state") or None),
            rego=rego,
            rego_expiry=_coerce_date(row.get("rego_expiry")),
            make=(row.get("make") or None),
            model=(row.get("model") or None),
            year=_coerce_int(row.get("year")),
            body_type=(row.get("body_type") or None),
            description=(row.get("description") or None),
            nhv_code=(row.get("nhv_code") or None),
            insurance=(row.get("insurance") or None),
            odometer=_coerce_int(row.get("odometer")),
            last_service=_coerce_date(row.get("last_service")),
            next_service=_coerce_date(row.get("next_service")),
            coi_number=(row.get("coi_number") or None),
            coi_due=_coerce_date(row.get("coi_due")),
            present_driver=(row.get("present_driver") or None),
            daily_check_complete=_coerce_bool(row.get("daily_check_complete")),
        )
        inserted += 1

    return inserted


def import_vehicle_details_from_workbook(
    conn,
    workbook: pd.ExcelFile,
    *,
    sheet_hints: Sequence[str] = DEFAULT_VEHICLE_SHEET_HINTS,
) -> int:
    """Ingest vehicle metadata from a downloaded workbook."""

    try:
        sheet_name = _resolve_sheet_name(workbook.sheet_names, sheet_hints)
    except ValueError:
        vehicle_rows = _collect_vehicle_rows_from_workbook(workbook)
        if vehicle_rows.empty:
            raise
        return import_vehicle_details_from_dataframe(conn, vehicle_rows)

    worksheet = workbook.parse(sheet_name)
    return import_vehicle_details_from_dataframe(conn, worksheet)


def import_vehicle_details_from_google_sheet(
    conn,
    *,
    sheet_id: str,
    sheet_hints: Sequence[str] = DEFAULT_VEHICLE_SHEET_HINTS,
) -> int:
    """Download and ingest the VEHICLE worksheet from a Google Sheets workbook."""

    workbook_url = GOOGLE_SHEET_EXPORT.format(sheet_id=sheet_id)
    workbook = pd.ExcelFile(workbook_url, engine="openpyxl")
    return import_vehicle_details_from_workbook(conn, workbook, sheet_hints=sheet_hints)


__all__ = [
    "import_vehicle_details_from_dataframe",
    "import_vehicle_details_from_workbook",
    "import_vehicle_details_from_google_sheet",
    "GOOGLE_SHEET_EXPORT",
    "DEFAULT_VEHICLE_SHEET_HINTS",
]
