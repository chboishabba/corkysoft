"""Operational product bridge: quote conversion, job states, calendar, communications and crew workflow."""
from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from typing import Any, Iterable

from analytics.db.schema import ensure_dashboard_tables
from corkysoft.repo import ensure_schema as ensure_quote_schema

JOB_STATES = (
    "draft",
    "planned",
    "assigned",
    "acknowledged",
    "en_route",
    "arrived",
    "loading",
    "in_transit",
    "unloading",
    "completed",
    "exception",
)

JOB_STATE_TRANSITIONS: dict[str, set[str]] = {
    "draft": {"planned", "exception"},
    "planned": {"assigned", "exception"},
    "assigned": {"acknowledged", "planned", "exception"},
    "acknowledged": {"en_route", "exception"},
    "en_route": {"arrived", "exception"},
    "arrived": {"loading", "exception"},
    "loading": {"in_transit", "exception"},
    "in_transit": {"unloading", "exception"},
    "unloading": {"completed", "exception"},
    "completed": set(),
    "exception": {"planned", "assigned"},
}

CUSTOMER_COMMUNICATION_EVENTS = (
    "quote_sent",
    "booking_confirmed",
    "day_before_reminder_due",
    "crew_en_route",
    "delayed",
    "milestone_reached",
    "completed",
    "receipt_ready",
    "support_required",
)

PUBLIC_SAFE_PAYLOAD_FIELDS = {
    "customer_name",
    "job_number",
    "origin",
    "destination",
    "planned_start",
    "planned_end",
    "eta",
    "status",
    "milestone",
    "delay_minutes",
    "support_reference",
    "receipt_reference",
    "message",
}


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _ensure_column(conn: sqlite3.Connection, table: str, name: str, declaration: str) -> None:
    if name not in _columns(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")


def ensure_operations_platform_schema(conn: sqlite3.Connection) -> None:
    """Create the additive schema used by the operational bridge."""

    ensure_dashboard_tables(conn)
    ensure_quote_schema(conn)
    for name, declaration in {
        "status": "TEXT NOT NULL DEFAULT 'draft'",
        "accepted_job_id": "INTEGER",
        "accepted_at": "TEXT",
    }.items():
        _ensure_column(conn, "quotes", name, declaration)
    for name, declaration in {
        "job_number": "TEXT",
        "quote_id": "INTEGER",
        "status": "TEXT NOT NULL DEFAULT 'draft'",
        "planned_start": "TEXT",
        "planned_end": "TEXT",
        "internal_notes": "TEXT",
    }.items():
        _ensure_column(conn, "jobs", name, declaration)
    for name, declaration in {
        "assignment_status": "TEXT NOT NULL DEFAULT 'draft'",
        "warning_flags": "TEXT NOT NULL DEFAULT '[]'",
        "blocking_flags": "TEXT NOT NULL DEFAULT '[]'",
    }.items():
        _ensure_column(conn, "job_segments", name, declaration)

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS quote_requirements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            quote_id INTEGER NOT NULL,
            requirement_type TEXT NOT NULL,
            description TEXT NOT NULL,
            quantity REAL,
            unit TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY(quote_id) REFERENCES quotes(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_quote_requirements_quote
            ON quote_requirements(quote_id, requirement_type);

        CREATE TABLE IF NOT EXISTS job_requirements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL,
            segment_id INTEGER,
            source_quote_requirement_id INTEGER,
            requirement_type TEXT NOT NULL,
            description TEXT NOT NULL,
            quantity REAL,
            unit TEXT,
            status TEXT NOT NULL DEFAULT 'required',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE,
            FOREIGN KEY(segment_id) REFERENCES job_segments(id) ON DELETE CASCADE,
            FOREIGN KEY(source_quote_requirement_id) REFERENCES quote_requirements(id) ON DELETE SET NULL
        );
        CREATE INDEX IF NOT EXISTS idx_job_requirements_job
            ON job_requirements(job_id, status);

        CREATE TABLE IF NOT EXISTS job_state_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL,
            previous_state TEXT,
            new_state TEXT NOT NULL,
            actor TEXT NOT NULL,
            note TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_job_state_events_job
            ON job_state_events(job_id, created_at);

        CREATE TABLE IF NOT EXISTS customer_communication_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER,
            quote_id INTEGER,
            event_type TEXT NOT NULL,
            channel TEXT NOT NULL DEFAULT 'internal',
            status TEXT NOT NULL DEFAULT 'proposed',
            recipient TEXT,
            template_key TEXT,
            public_payload_json TEXT NOT NULL DEFAULT '{}',
            proposed_by TEXT NOT NULL,
            approved_by TEXT,
            approved_at TEXT,
            sent_at TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE,
            FOREIGN KEY(quote_id) REFERENCES quotes(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_customer_communication_events_job
            ON customer_communication_events(job_id, event_type, created_at);

        CREATE TABLE IF NOT EXISTS crew_acknowledgements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL,
            segment_id INTEGER,
            worker_id INTEGER NOT NULL,
            acknowledgement_type TEXT NOT NULL,
            status TEXT NOT NULL,
            note TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE,
            FOREIGN KEY(segment_id) REFERENCES job_segments(id) ON DELETE CASCADE,
            FOREIGN KEY(worker_id) REFERENCES workers(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_crew_ack_job_worker
            ON crew_acknowledgements(job_id, worker_id, created_at);
        """
    )
    conn.commit()


def add_quote_requirement(
    conn: sqlite3.Connection,
    *,
    quote_id: int,
    requirement_type: str,
    description: str,
    quantity: float | None = None,
    unit: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> int:
    ensure_operations_platform_schema(conn)
    cursor = conn.execute(
        """
        INSERT INTO quote_requirements (
            quote_id, requirement_type, description, quantity, unit, metadata_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            quote_id,
            requirement_type.strip(),
            description.strip(),
            quantity,
            unit,
            json.dumps(metadata or {}, sort_keys=True),
            _utc_now_iso(),
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def _quote_row(conn: sqlite3.Connection, quote_id: int) -> sqlite3.Row:
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM quotes WHERE id = ?", (quote_id,)).fetchone()
    if row is None:
        raise ValueError(f"Quote {quote_id} does not exist")
    return row


def accept_quote_as_job(
    conn: sqlite3.Connection,
    *,
    quote_id: int,
    actor: str,
    planned_start: str | None = None,
    planned_end: str | None = None,
) -> dict[str, Any]:
    """Accept a quote once and materialise a job, segment and copied requirements."""

    ensure_operations_platform_schema(conn)
    quote = _quote_row(conn, quote_id)
    if quote["accepted_job_id"] is not None:
        job_id = int(quote["accepted_job_id"])
        return {"quoteId": quote_id, "jobId": job_id, "created": False}

    now = _utc_now_iso()
    job_number = f"Q-{quote_id:06d}"
    stored_total = quote["manual_quote"] if quote["manual_quote"] is not None else quote["final_quote"]
    safe_volume = float(quote["cubic_m"] or 0.0)
    price_per_m3 = float(stored_total) / safe_volume if safe_volume > 0 else None

    cursor = conn.execute(
        """
        INSERT INTO jobs (
            job_number, job_date, client, client_id, origin, destination,
            origin_resolved, destination_resolved, price_per_m3, revenue_total,
            revenue, volume_m3, volume, distance_km, final_cost,
            origin_lat, origin_lon, dest_lat, dest_lon,
            quote_id, status, planned_start, planned_end, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            job_number,
            quote["quote_date"],
            quote["client_display"] or "Quote builder",
            quote["client_id"],
            quote["origin_input"],
            quote["destination_input"],
            quote["origin_resolved"],
            quote["destination_resolved"],
            price_per_m3,
            stored_total,
            stored_total,
            quote["cubic_m"],
            quote["cubic_m"],
            quote["distance_km"],
            quote["total_before_margin"],
            quote["origin_lat"],
            quote["origin_lon"],
            quote["dest_lat"],
            quote["dest_lon"],
            quote_id,
            "planned",
            planned_start,
            planned_end,
            now,
            now,
        ),
    )
    job_id = int(cursor.lastrowid)
    segment_cursor = conn.execute(
        """
        INSERT INTO job_segments (
            job_id, segment_sequence, from_location, to_location, mode,
            distance_km, planned_start, planned_end, status, assignment_status,
            created_at, updated_at
        ) VALUES (?, 1, ?, ?, 'road', ?, ?, ?, 'planned', 'planned', ?, ?)
        """,
        (
            job_id,
            quote["origin_resolved"] or quote["origin_input"],
            quote["destination_resolved"] or quote["destination_input"],
            quote["distance_km"],
            planned_start,
            planned_end,
            now,
            now,
        ),
    )
    segment_id = int(segment_cursor.lastrowid)

    requirements = conn.execute(
        "SELECT * FROM quote_requirements WHERE quote_id = ? ORDER BY id",
        (quote_id,),
    ).fetchall()
    if not requirements:
        requirements = [
            {
                "id": None,
                "requirement_type": "move_volume",
                "description": "Quoted move volume",
                "quantity": quote["cubic_m"],
                "unit": "m3",
                "metadata_json": "{}",
            }
        ]
    for requirement in requirements:
        conn.execute(
            """
            INSERT INTO job_requirements (
                job_id, segment_id, source_quote_requirement_id, requirement_type,
                description, quantity, unit, metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                segment_id,
                requirement["id"],
                requirement["requirement_type"],
                requirement["description"],
                requirement["quantity"],
                requirement["unit"],
                requirement["metadata_json"],
                now,
                now,
            ),
        )

    conn.execute(
        "UPDATE quotes SET status = 'accepted', accepted_job_id = ?, accepted_at = ? WHERE id = ?",
        (job_id, now, quote_id),
    )
    conn.execute(
        """
        INSERT INTO job_state_events (job_id, previous_state, new_state, actor, note, created_at)
        VALUES (?, 'draft', 'planned', ?, 'Quote accepted and operational job created.', ?)
        """,
        (job_id, actor.strip() or "operator", now),
    )
    conn.commit()
    return {
        "quoteId": quote_id,
        "jobId": job_id,
        "segmentId": segment_id,
        "created": True,
        "requirementCount": len(requirements),
    }


def transition_job_state(
    conn: sqlite3.Connection,
    *,
    job_id: int,
    new_state: str,
    actor: str,
    note: str | None = None,
) -> dict[str, Any]:
    ensure_operations_platform_schema(conn)
    if new_state not in JOB_STATES:
        raise ValueError(f"Unknown job state: {new_state}")
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        raise ValueError(f"Job {job_id} does not exist")
    previous = str(row["status"] or "draft")
    if new_state == previous:
        return {"jobId": job_id, "previousState": previous, "newState": new_state, "changed": False}
    if new_state not in JOB_STATE_TRANSITIONS.get(previous, set()):
        raise ValueError(f"Invalid job state transition: {previous} -> {new_state}")
    now = _utc_now_iso()
    conn.execute(
        "UPDATE jobs SET status = ?, updated_at = ? WHERE id = ?",
        (new_state, now, job_id),
    )
    conn.execute(
        "UPDATE job_segments SET status = ?, assignment_status = ?, updated_at = ? WHERE job_id = ?",
        (new_state, new_state, now, job_id),
    )
    conn.execute(
        """
        INSERT INTO job_state_events (job_id, previous_state, new_state, actor, note, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (job_id, previous, new_state, actor.strip() or "operator", note, now),
    )
    conn.commit()
    return {"jobId": job_id, "previousState": previous, "newState": new_state, "changed": True}


def _normalise_filter(values: Iterable[str] | None) -> list[str]:
    return [str(value).strip() for value in (values or []) if str(value).strip()]


def list_dispatch_calendar(
    conn: sqlite3.Connection,
    *,
    start: str,
    end: str,
    statuses: Iterable[str] | None = None,
    truck_ids: Iterable[str] | None = None,
    worker_ids: Iterable[int] | None = None,
    depot: str | None = None,
) -> list[dict[str, Any]]:
    """Return read-only daily/weekly calendar rows with assignments and readiness."""

    ensure_operations_platform_schema(conn)
    conn.row_factory = sqlite3.Row
    status_values = _normalise_filter(statuses)
    truck_values = _normalise_filter(truck_ids)
    worker_values = [int(value) for value in (worker_ids or [])]
    clauses = ["COALESCE(js.planned_start, j.planned_start, j.job_date) >= ?", "COALESCE(js.planned_start, j.planned_start, j.job_date) < ?"]
    params: list[Any] = [start, end]
    if status_values:
        clauses.append(f"j.status IN ({','.join('?' for _ in status_values)})")
        params.extend(status_values)
    if depot:
        clauses.append("(j.origin LIKE ? OR j.origin_resolved LIKE ?)")
        params.extend([f"%{depot}%", f"%{depot}%"])
    if truck_values:
        clauses.append(
            f"EXISTS (SELECT 1 FROM job_segment_vehicles fsv WHERE fsv.segment_id = js.id AND fsv.truck_id IN ({','.join('?' for _ in truck_values)}))"
        )
        params.extend(truck_values)
    if worker_values:
        clauses.append(
            f"EXISTS (SELECT 1 FROM job_segment_workers fsw WHERE fsw.segment_id = js.id AND fsw.worker_id IN ({','.join('?' for _ in worker_values)}))"
        )
        params.extend(worker_values)

    rows = conn.execute(
        f"""
        SELECT
            j.id AS job_id,
            j.job_number,
            j.client,
            j.origin,
            j.destination,
            j.status AS job_status,
            js.id AS segment_id,
            js.segment_sequence,
            COALESCE(js.planned_start, j.planned_start, j.job_date) AS planned_start,
            COALESCE(js.planned_end, j.planned_end) AS planned_end,
            COALESCE(js.blocking_flags, '[]') AS blocking_flags,
            COALESCE(js.warning_flags, '[]') AS warning_flags,
            GROUP_CONCAT(DISTINCT jsv.truck_id) AS truck_ids,
            GROUP_CONCAT(DISTINCT w.name) AS worker_names,
            GROUP_CONCAT(DISTINCT jsw.worker_id) AS worker_ids
        FROM jobs j
        JOIN job_segments js ON js.job_id = j.id
        LEFT JOIN job_segment_vehicles jsv ON jsv.segment_id = js.id
        LEFT JOIN job_segment_workers jsw ON jsw.segment_id = js.id
        LEFT JOIN workers w ON w.id = jsw.worker_id
        WHERE {' AND '.join(clauses)}
        GROUP BY j.id, js.id
        ORDER BY planned_start, j.job_number, js.segment_sequence
        """,
        tuple(params),
    ).fetchall()
    result: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["truckIds"] = [value for value in str(item.pop("truck_ids") or "").split(",") if value]
        item["workerNames"] = [value for value in str(item.pop("worker_names") or "").split(",") if value]
        item["workerIds"] = [int(value) for value in str(item.pop("worker_ids") or "").split(",") if value]
        item["blockingFlags"] = json.loads(item.pop("blocking_flags") or "[]")
        item["warningFlags"] = json.loads(item.pop("warning_flags") or "[]")
        item["ready"] = not item["blockingFlags"] and bool(item["truckIds"]) and bool(item["workerIds"])
        result.append(item)
    return result


def propose_customer_communication(
    conn: sqlite3.Connection,
    *,
    event_type: str,
    proposed_by: str,
    public_payload: dict[str, Any],
    job_id: int | None = None,
    quote_id: int | None = None,
    recipient: str | None = None,
    channel: str = "internal",
    template_key: str | None = None,
) -> int:
    ensure_operations_platform_schema(conn)
    if event_type not in CUSTOMER_COMMUNICATION_EVENTS:
        raise ValueError(f"Unknown customer communication event: {event_type}")
    if job_id is None and quote_id is None:
        raise ValueError("Customer communication must reference a job or quote")
    unsafe = sorted(set(public_payload) - PUBLIC_SAFE_PAYLOAD_FIELDS)
    if unsafe:
        raise ValueError("Unsafe customer payload fields: " + ", ".join(unsafe))
    cursor = conn.execute(
        """
        INSERT INTO customer_communication_events (
            job_id, quote_id, event_type, channel, status, recipient,
            template_key, public_payload_json, proposed_by, created_at
        ) VALUES (?, ?, ?, ?, 'proposed', ?, ?, ?, ?, ?)
        """,
        (
            job_id,
            quote_id,
            event_type,
            channel,
            recipient,
            template_key,
            json.dumps(public_payload, sort_keys=True),
            proposed_by.strip() or "operator",
            _utc_now_iso(),
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def list_worker_assignments(conn: sqlite3.Connection, *, worker_id: int) -> list[dict[str, Any]]:
    ensure_operations_platform_schema(conn)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT j.id AS job_id, j.job_number, j.client, j.origin, j.destination,
               j.status AS job_status, j.internal_notes, js.id AS segment_id,
               js.segment_sequence, js.planned_start, js.planned_end,
               GROUP_CONCAT(DISTINCT jsv.truck_id) AS truck_ids
        FROM job_segment_workers jsw
        JOIN job_segments js ON js.id = jsw.segment_id
        JOIN jobs j ON j.id = js.job_id
        LEFT JOIN job_segment_vehicles jsv ON jsv.segment_id = js.id
        WHERE jsw.worker_id = ? AND j.status != 'completed'
        GROUP BY j.id, js.id
        ORDER BY COALESCE(js.planned_start, j.job_date), j.job_number
        """,
        (worker_id,),
    ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["truckIds"] = [value for value in str(item.pop("truck_ids") or "").split(",") if value]
        result.append(item)
    return result


def record_crew_acknowledgement(
    conn: sqlite3.Connection,
    *,
    job_id: int,
    worker_id: int,
    acknowledgement_type: str = "assignment",
    status: str = "acknowledged",
    segment_id: int | None = None,
    note: str | None = None,
) -> int:
    ensure_operations_platform_schema(conn)
    cursor = conn.execute(
        """
        INSERT INTO crew_acknowledgements (
            job_id, segment_id, worker_id, acknowledgement_type, status, note, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (job_id, segment_id, worker_id, acknowledgement_type, status, note, _utc_now_iso()),
    )
    row = conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        raise ValueError(f"Job {job_id} does not exist")
    current = str(row[0] or "draft")
    if acknowledgement_type == "assignment" and status == "acknowledged" and current == "assigned":
        transition_job_state(conn, job_id=job_id, new_state="acknowledged", actor=f"worker:{worker_id}", note=note)
    else:
        conn.commit()
    return int(cursor.lastrowid)


def record_crew_job_status(
    conn: sqlite3.Connection,
    *,
    job_id: int,
    worker_id: int,
    new_state: str,
    note: str | None = None,
) -> dict[str, Any]:
    if new_state == "exception":
        record_crew_acknowledgement(
            conn,
            job_id=job_id,
            worker_id=worker_id,
            acknowledgement_type="exception",
            status="flagged",
            note=note,
        )
    return transition_job_state(
        conn,
        job_id=job_id,
        new_state=new_state,
        actor=f"worker:{worker_id}",
        note=note,
    )


__all__ = [
    "JOB_STATES",
    "JOB_STATE_TRANSITIONS",
    "CUSTOMER_COMMUNICATION_EVENTS",
    "PUBLIC_SAFE_PAYLOAD_FIELDS",
    "ensure_operations_platform_schema",
    "add_quote_requirement",
    "accept_quote_as_job",
    "transition_job_state",
    "list_dispatch_calendar",
    "propose_customer_communication",
    "list_worker_assignments",
    "record_crew_acknowledgement",
    "record_crew_job_status",
]
