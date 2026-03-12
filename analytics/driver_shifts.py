"""Helpers for importing and auditing driver shift data."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

from analytics.db import fetch_driver_shifts, upsert_driver_shift
from analytics.google_sheets import build_google_sheet_csv_url

DEFAULT_DRIVER_SHEET_NAME = "VEHICLE_DRIVER"


@dataclass(slots=True)
class DriverShiftRecord:
    shift_date: str
    truck_id: str | None = None
    worker_name: str | None = None
    job_id: int | None = None
    shipment_id: int | None = None
    ticket_numbers: str | None = None
    shift_start: str | None = None
    shift_end: str | None = None
    shift_window_start: str | None = None
    shift_window_end: str | None = None
    role: str | None = None
    hours: float | None = None
    hourly_rate: float | None = None
    cost_total: float | None = None
    source: str | None = None


_COLUMN_ALIASES: dict[str, set[str]] = {
    "shift_date": {"date", "shiftdate"},
    "truck_id": {"truck", "vehicle", "vehicleid", "truckid"},
    "worker_name": {"driver", "worker", "employee", "staff"},
    "job_id": {"job", "jobid", "job_id"},
    "shipment_id": {"shipment", "shipmentid", "load", "load_id"},
    "ticket_numbers": {"ticket", "tickets", "ticketnumbers"},
    "shift_start": {"start", "starttime", "shiftstart", "from"},
    "shift_end": {"end", "finish", "finishtime", "shiftend", "to"},
    "shift_window_start": {"shiftwindowstart", "windowstart", "plannedstart", "rosteredstart"},
    "shift_window_end": {"shiftwindowend", "windowend", "plannedend", "rosteredend"},
    "role": {"role", "position", "assignment"},
    "hours": {"hours", "hrs", "totalhours"},
    "hourly_rate": {"rate", "hourlyrate", "payrate"},
    "cost_total": {"cost", "totalcost", "pay", "amount"},
}


def _normalise_column_name(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "", name.strip().lower())
    return cleaned


def _resolve_column(columns: Iterable[str], target: str) -> str | None:
    aliases = _COLUMN_ALIASES[target]
    for column in columns:
        if _normalise_column_name(column) in aliases:
            return column
    return None


def _normalise_date(value: object) -> str | None:
    if value is None:
        return None
    try:
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    if not isinstance(parsed, pd.Timestamp):
        return None
    return parsed.date().isoformat()


def _coerce_numeric(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def _coerce_int(value: object) -> int | None:
    number = _coerce_numeric(value)
    if number is None:
        return None
    return int(number)


def _prepare_shift_records(df: pd.DataFrame, *, source: str | None = None) -> list[DriverShiftRecord]:
    mappings: dict[str, str | None] = {
        key: _resolve_column(df.columns, key) for key in _COLUMN_ALIASES
    }
    if mappings["shift_date"] is None:
        raise ValueError("Google Sheet is missing a shift date column")

    records: list[DriverShiftRecord] = []
    for _, row in df.iterrows():
        shift_date = _normalise_date(row[mappings["shift_date"]])
        if not shift_date:
            continue

        hours = _coerce_numeric(row[mappings["hours"]]) if mappings["hours"] else None
        hourly_rate = _coerce_numeric(row[mappings["hourly_rate"]]) if mappings["hourly_rate"] else None
        cost_total = _coerce_numeric(row[mappings["cost_total"]]) if mappings["cost_total"] else None
        if cost_total is None and hours is not None and hourly_rate is not None:
            cost_total = hours * hourly_rate

        record = DriverShiftRecord(
            shift_date=shift_date,
            truck_id=(row[mappings["truck_id"]] if mappings["truck_id"] else None),
            worker_name=(row[mappings["worker_name"]] if mappings["worker_name"] else None),
            job_id=_coerce_int(row[mappings["job_id"]]) if mappings["job_id"] else None,
            shipment_id=(
                _coerce_int(row[mappings["shipment_id"]])
                if mappings["shipment_id"]
                else None
            ),
            ticket_numbers=(
                str(row[mappings["ticket_numbers"]]).strip()
                if mappings["ticket_numbers"] and not pd.isna(row[mappings["ticket_numbers"]])
                else None
            ),
            shift_start=(
                str(row[mappings["shift_start"]]).strip()
                if mappings["shift_start"] and not pd.isna(row[mappings["shift_start"]])
                else None
            ),
            shift_end=(
                str(row[mappings["shift_end"]]).strip()
                if mappings["shift_end"] and not pd.isna(row[mappings["shift_end"]])
                else None
            ),
            shift_window_start=(
                str(row[mappings["shift_window_start"]]).strip()
                if mappings["shift_window_start"]
                and not pd.isna(row[mappings["shift_window_start"]])
                else None
            ),
            shift_window_end=(
                str(row[mappings["shift_window_end"]]).strip()
                if mappings["shift_window_end"]
                and not pd.isna(row[mappings["shift_window_end"]])
                else None
            ),
            role=(
                str(row[mappings["role"]]).strip()
                if mappings["role"] and not pd.isna(row[mappings["role"]])
                else None
            ),
            hours=hours,
            hourly_rate=hourly_rate,
            cost_total=cost_total,
            source=source,
        )
        records.append(record)

    return records


def _build_sheet_url(sheet_id_or_url: str, sheet_name: str) -> str:
    if sheet_id_or_url.startswith("http") and "gviz/tq" in sheet_id_or_url:
        return sheet_id_or_url
    return build_google_sheet_csv_url(sheet_id_or_url, sheet_name)


def import_driver_shifts_from_sheet(
    conn,
    *,
    sheet_id: str | None = None,
    sheet_name: str = DEFAULT_DRIVER_SHEET_NAME,
    dataframe: pd.DataFrame | None = None,
) -> tuple[int, int]:
    """Import driver shifts from the VEHICLE_DRIVER Google Sheet."""

    if dataframe is not None:
        df = dataframe
    else:
        if not sheet_id:
            raise ValueError("Provide a sheet_id or a dataframe for shift import")
        sheet_url = _build_sheet_url(sheet_id, sheet_name)
        df = pd.read_csv(sheet_url)
    if df.empty:
        return (0, 0)

    records = _prepare_shift_records(df, source=sheet_name)
    inserted = 0
    updated = 0
    for record in records:
        _, created = upsert_driver_shift(
            conn,
            shift_date=record.shift_date,
            truck_id=str(record.truck_id).strip() if record.truck_id else None,
            worker_name=str(record.worker_name).strip() if record.worker_name else None,
            job_id=record.job_id,
            shipment_id=record.shipment_id,
            ticket_numbers=record.ticket_numbers,
            shift_start=record.shift_start,
            shift_end=record.shift_end,
            shift_window_start=record.shift_window_start,
            shift_window_end=record.shift_window_end,
            role=record.role,
            hours=record.hours,
            hourly_rate=record.hourly_rate,
            cost_total=record.cost_total,
            source=record.source,
        )
        if created:
            inserted += 1
        else:
            updated += 1
    return inserted, updated


def load_driver_shifts_dataframe(
    conn,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    workers: Sequence[str] | None = None,
    trucks: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load driver shifts into a dataframe for auditing."""

    rows = fetch_driver_shifts(
        conn,
        start_date=start_date,
        end_date=end_date,
        worker_names=workers,
        truck_ids=trucks,
    )
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame.from_records([dict(row) for row in rows])
    for column in ("hours", "hourly_rate", "cost_total"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


__all__ = [
    "DEFAULT_DRIVER_SHEET_NAME",
    "DriverShiftRecord",
    "import_driver_shifts_from_sheet",
    "load_driver_shifts_dataframe",
]
