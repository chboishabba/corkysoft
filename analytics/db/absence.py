"""Worker absence and leave record helpers."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Any

from .schema import _ensure_worker_absence_records_table

ABSENCE_RECORD_TYPES = (
    "sick",
    "annual_leave",
    "personal_leave",
    "unpaid_leave",
    "carers_leave",
    "other",
)

ABSENCE_RECORD_STATUSES = (
    "planned",
    "confirmed",
    "cancelled",
)


def create_worker_absence_record(
    conn: sqlite3.Connection,
    *,
    worker_id: int,
    start_date: str,
    end_date: str | None = None,
    absence_type: str = "other",
    status: str = "confirmed",
    hours_per_day: float | None = None,
    note: str | None = None,
    source: str | None = None,
    recorded_by: str | None = None,
) -> sqlite3.Row:
    """Create a worker absence/leave record."""

    _ensure_worker_absence_records_table(conn)
    normalized_type = absence_type.strip().lower()
    normalized_status = status.strip().lower()
    if normalized_type not in ABSENCE_RECORD_TYPES:
        raise ValueError(f"Unsupported absence type: {absence_type}")
    if normalized_status not in ABSENCE_RECORD_STATUSES:
        raise ValueError(f"Unsupported absence status: {status}")
    worker_row = conn.execute("SELECT id FROM workers WHERE id = ?", (worker_id,)).fetchone()
    if worker_row is None:
        raise ValueError(f"Worker {worker_id} does not exist")

    timestamp = datetime.now(UTC).isoformat()
    resolved_end_date = end_date or start_date
    cursor = conn.execute(
        """
        INSERT INTO worker_absence_records (
            worker_id,
            start_date,
            end_date,
            absence_type,
            status,
            hours_per_day,
            note,
            source,
            recorded_by,
            created_at,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            worker_id,
            start_date,
            resolved_end_date,
            normalized_type,
            normalized_status,
            float(hours_per_day) if hours_per_day is not None else None,
            note,
            source,
            recorded_by,
            timestamp,
            timestamp,
        ),
    )
    conn.commit()
    return conn.execute(
        """
        SELECT
            war.*,
            w.name AS worker_name
        FROM worker_absence_records AS war
        JOIN workers AS w ON w.id = war.worker_id
        WHERE war.id = ?
        """,
        (int(cursor.lastrowid),),
    ).fetchone()


def list_worker_absence_records(
    conn: sqlite3.Connection,
    *,
    worker_id: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    status: str | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """List recorded worker absences/leaves with basic filters."""

    _ensure_worker_absence_records_table(conn)
    query = """
        SELECT
            war.*,
            w.name AS worker_name
        FROM worker_absence_records AS war
        JOIN workers AS w ON w.id = war.worker_id
        WHERE 1 = 1
    """
    params: list[Any] = []
    if worker_id is not None:
        query += " AND war.worker_id = ?"
        params.append(int(worker_id))
    if start_date:
        query += " AND war.end_date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND war.start_date <= ?"
        params.append(end_date)
    if status:
        query += " AND war.status = ?"
        params.append(status.strip().lower())
    query += " ORDER BY war.start_date DESC, war.id DESC LIMIT ?"
    params.append(int(limit))
    rows = conn.execute(query, tuple(params)).fetchall()
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append(
            {
                "id": int(row["id"]),
                "workerId": int(row["worker_id"]),
                "workerName": row["worker_name"],
                "startDate": row["start_date"],
                "endDate": row["end_date"],
                "absenceType": row["absence_type"],
                "status": row["status"],
                "hoursPerDay": row["hours_per_day"],
                "note": row["note"],
                "source": row["source"],
                "recordedBy": row["recorded_by"],
                "createdAt": row["created_at"],
                "updatedAt": row["updated_at"],
            }
        )
    return normalized


__all__ = [
    "ABSENCE_RECORD_STATUSES",
    "ABSENCE_RECORD_TYPES",
    "create_worker_absence_record",
    "list_worker_absence_records",
]
