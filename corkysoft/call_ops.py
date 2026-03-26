from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from datetime import UTC, datetime
from typing import Any, Optional, Sequence

from corkysoft.repo import ensure_schema as ensure_client_schema
from corkysoft.whisperx_adapter import (
    WhisperXAdapterError,
    fetch_task_status,
    submit_transcription,
)

CALL_EVENT_KINDS: tuple[str, ...] = (
    "client_call",
    "ops_call",
    "manager_call",
    "worker_call",
    "clock_on_call",
    "clock_off_call",
)
CALL_DIRECTIONS: tuple[str, ...] = ("inbound", "outbound", "internal")
CALL_STATUSES: tuple[str, ...] = (
    "ringing",
    "active",
    "completed",
    "failed",
    "needs_review",
)
CALL_SOURCE_CHANNELS: tuple[str, ...] = (
    "telephony",
    "whatsapp",
    "manual_note",
    "imported_recording",
)
CALL_ROUTING_EVENT_TYPES: tuple[str, ...] = (
    "call_received",
    "call_routed",
    "call_answered",
    "call_transferred",
    "call_consult_started",
    "call_consult_ended",
    "call_ended",
)
CALL_LEG_KINDS: tuple[str, ...] = (
    "primary",
    "transfer",
    "consult",
    "timesheet",
    "callback",
)
CALL_TRANSCRIPT_STATUSES: tuple[str, ...] = (
    "queued",
    "in_progress",
    "completed",
    "failed",
)
EXTRACTED_ACTION_STATUSES: tuple[str, ...] = ("pending", "accepted", "rejected")
WORKER_TIME_EVENT_TYPES: tuple[str, ...] = ("clock_on", "clock_off")
WORKER_TIME_CHANNELS: tuple[str, ...] = (
    "app",
    "whatsapp",
    "voice_call",
    "manual_supervisor",
)
WORKER_TIME_REVIEW_STATUSES: tuple[str, ...] = (
    "pending_review",
    "accepted",
    "rejected",
)
AUTHORITY_CLASSES: tuple[str, ...] = ("compiled_state", "observer_capture_ref")
AMBIENT_SESSION_STATUSES: tuple[str, ...] = ("active", "completed", "needs_review")

_PHONE_RE = re.compile(r"\D+")


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS call_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    root_call_event_id INTEGER,
    event_kind TEXT NOT NULL,
    direction TEXT NOT NULL,
    status TEXT NOT NULL,
    source_channel TEXT NOT NULL,
    title TEXT,
    caller_phone TEXT,
    caller_phone_normalized TEXT,
    callee_phone TEXT,
    callee_phone_normalized TEXT,
    client_id INTEGER,
    quote_id INTEGER,
    job_id INTEGER,
    segment_id INTEGER,
    worker_id INTEGER,
    operator_id TEXT,
    correlation_id TEXT,
    started_at TEXT,
    ended_at TEXT,
    captured_at TEXT,
    processed_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(root_call_event_id) REFERENCES call_events(id) ON DELETE SET NULL,
    FOREIGN KEY(client_id) REFERENCES clients(id) ON DELETE SET NULL,
    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE SET NULL,
    FOREIGN KEY(segment_id) REFERENCES job_segments(id) ON DELETE SET NULL,
    FOREIGN KEY(worker_id) REFERENCES workers(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_call_sessions_created_at ON call_sessions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_call_sessions_root_call_event ON call_sessions(root_call_event_id);
CREATE INDEX IF NOT EXISTS idx_call_sessions_job_id ON call_sessions(job_id);

CREATE TABLE IF NOT EXISTS call_legs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_session_id INTEGER NOT NULL,
    root_call_event_id INTEGER,
    leg_kind TEXT NOT NULL,
    direction TEXT NOT NULL,
    status TEXT NOT NULL,
    source_channel TEXT NOT NULL,
    destination_kind TEXT,
    destination_label TEXT,
    operator_id TEXT,
    caller_phone TEXT,
    caller_phone_normalized TEXT,
    callee_phone TEXT,
    callee_phone_normalized TEXT,
    started_at TEXT,
    answered_at TEXT,
    ended_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(call_session_id) REFERENCES call_sessions(id) ON DELETE CASCADE,
    FOREIGN KEY(root_call_event_id) REFERENCES call_events(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_call_legs_session_id ON call_legs(call_session_id, created_at DESC);

CREATE TABLE IF NOT EXISTS call_routing_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_session_id INTEGER NOT NULL,
    call_leg_id INTEGER,
    event_type TEXT NOT NULL,
    from_destination TEXT,
    to_destination TEXT,
    actor TEXT,
    detail TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(call_session_id) REFERENCES call_sessions(id) ON DELETE CASCADE,
    FOREIGN KEY(call_leg_id) REFERENCES call_legs(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_call_routing_events_session_id ON call_routing_events(call_session_id, created_at DESC);

CREATE TABLE IF NOT EXISTS call_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_kind TEXT NOT NULL,
    direction TEXT NOT NULL,
    status TEXT NOT NULL,
    source_channel TEXT NOT NULL,
    title TEXT,
    caller_phone TEXT,
    caller_phone_normalized TEXT,
    callee_phone TEXT,
    callee_phone_normalized TEXT,
    client_id INTEGER,
    quote_id INTEGER,
    job_id INTEGER,
    segment_id INTEGER,
    worker_id INTEGER,
    operator_id TEXT,
    correlation_id TEXT,
    started_at TEXT,
    ended_at TEXT,
    captured_at TEXT,
    processed_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(client_id) REFERENCES clients(id) ON DELETE SET NULL,
    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE SET NULL,
    FOREIGN KEY(segment_id) REFERENCES job_segments(id) ON DELETE SET NULL,
    FOREIGN KEY(worker_id) REFERENCES workers(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_call_events_created_at ON call_events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_call_events_job_id ON call_events(job_id);
CREATE INDEX IF NOT EXISTS idx_call_events_client_id ON call_events(client_id);

CREATE TABLE IF NOT EXISTS call_transcript_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_event_id INTEGER,
    call_session_id INTEGER,
    call_leg_id INTEGER,
    service_key TEXT NOT NULL,
    external_task_id TEXT,
    status TEXT NOT NULL,
    transcript_text TEXT,
    transcript_segments_json TEXT,
    diarization_json TEXT,
    confidence REAL,
    is_final INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(call_event_id) REFERENCES call_events(id) ON DELETE CASCADE,
    FOREIGN KEY(call_session_id) REFERENCES call_sessions(id) ON DELETE CASCADE,
    FOREIGN KEY(call_leg_id) REFERENCES call_legs(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_call_transcripts_call_event ON call_transcript_artifacts(call_event_id, created_at DESC);

CREATE TABLE IF NOT EXISTS call_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_event_id INTEGER,
    ambient_session_id INTEGER,
    author TEXT,
    note_kind TEXT NOT NULL DEFAULT 'operator',
    note_text TEXT NOT NULL,
    authoritative INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    FOREIGN KEY(call_event_id) REFERENCES call_events(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS call_extracted_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_event_id INTEGER,
    ambient_session_id INTEGER,
    transcript_artifact_id INTEGER,
    source_engine TEXT,
    action_text TEXT NOT NULL,
    span_start REAL,
    span_end REAL,
    status TEXT NOT NULL DEFAULT 'pending',
    decided_by TEXT,
    decision_note TEXT,
    created_at TEXT NOT NULL,
    decided_at TEXT,
    FOREIGN KEY(call_event_id) REFERENCES call_events(id) ON DELETE CASCADE,
    FOREIGN KEY(transcript_artifact_id) REFERENCES call_transcript_artifacts(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS call_link_resolutions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_event_id INTEGER,
    ambient_session_id INTEGER,
    actor TEXT,
    client_id INTEGER,
    quote_id INTEGER,
    job_id INTEGER,
    segment_id INTEGER,
    worker_id INTEGER,
    resolution_note TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(call_event_id) REFERENCES call_events(id) ON DELETE CASCADE,
    FOREIGN KEY(client_id) REFERENCES clients(id) ON DELETE SET NULL,
    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE SET NULL,
    FOREIGN KEY(segment_id) REFERENCES job_segments(id) ON DELETE SET NULL,
    FOREIGN KEY(worker_id) REFERENCES workers(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS worker_time_capture_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_event_id INTEGER,
    call_session_id INTEGER,
    call_leg_id INTEGER,
    worker_id INTEGER,
    worker_name_raw TEXT,
    employee_code_raw TEXT,
    event_type TEXT NOT NULL,
    channel TEXT NOT NULL,
    effective_timestamp TEXT,
    captured_timestamp TEXT NOT NULL,
    caller_phone TEXT,
    caller_phone_normalized TEXT,
    job_id INTEGER,
    segment_id INTEGER,
    truck_id TEXT,
    confidence REAL,
    review_status TEXT NOT NULL,
    reviewer TEXT,
    review_note TEXT,
    raw_payload TEXT,
    created_at TEXT NOT NULL,
    reviewed_at TEXT,
    FOREIGN KEY(call_event_id) REFERENCES call_events(id) ON DELETE SET NULL,
    FOREIGN KEY(call_session_id) REFERENCES call_sessions(id) ON DELETE SET NULL,
    FOREIGN KEY(call_leg_id) REFERENCES call_legs(id) ON DELETE SET NULL,
    FOREIGN KEY(worker_id) REFERENCES workers(id) ON DELETE SET NULL,
    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE SET NULL,
    FOREIGN KEY(segment_id) REFERENCES job_segments(id) ON DELETE SET NULL,
    FOREIGN KEY(truck_id) REFERENCES trucks(truck_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS state_egress_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    source_component TEXT NOT NULL,
    source_entity_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    correlation_id TEXT,
    causation_id TEXT,
    authority_class TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    ingested_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_state_egress_created ON state_egress_events(ingested_at DESC);

CREATE TABLE IF NOT EXISTS ambient_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    source_location TEXT,
    source_device TEXT,
    team_label TEXT,
    status TEXT NOT NULL,
    client_id INTEGER,
    quote_id INTEGER,
    job_id INTEGER,
    segment_id INTEGER,
    worker_id INTEGER,
    operator_id TEXT,
    correlation_id TEXT,
    started_at TEXT,
    ended_at TEXT,
    captured_at TEXT,
    processed_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(client_id) REFERENCES clients(id) ON DELETE SET NULL,
    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE SET NULL,
    FOREIGN KEY(segment_id) REFERENCES job_segments(id) ON DELETE SET NULL,
    FOREIGN KEY(worker_id) REFERENCES workers(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_ambient_sessions_created_at ON ambient_sessions(created_at DESC);

CREATE TABLE IF NOT EXISTS ambient_transcript_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ambient_session_id INTEGER NOT NULL,
    service_key TEXT NOT NULL,
    status TEXT NOT NULL,
    transcript_text TEXT,
    transcript_segments_json TEXT,
    diarization_json TEXT,
    confidence REAL,
    is_final INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(ambient_session_id) REFERENCES ambient_sessions(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_ambient_transcripts_session ON ambient_transcript_artifacts(ambient_session_id, created_at DESC);
"""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _normalize_phone(value: str | None) -> str | None:
    if not value:
        return None
    digits = _PHONE_RE.sub("", str(value))
    return digits or None


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    existing = {
        (row["name"] if hasattr(row, "keys") else row[1])
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


def ensure_call_ops_tables(conn: sqlite3.Connection) -> None:
    from analytics.db.schema import ensure_dashboard_tables

    ensure_dashboard_tables(conn)
    ensure_client_schema(conn)
    conn.executescript(SCHEMA_SQL)
    _ensure_column(conn, "call_transcript_artifacts", "call_session_id", "call_session_id INTEGER")
    _ensure_column(conn, "call_transcript_artifacts", "call_leg_id", "call_leg_id INTEGER")
    _ensure_column(conn, "call_notes", "ambient_session_id", "ambient_session_id INTEGER")
    _ensure_column(conn, "call_extracted_actions", "ambient_session_id", "ambient_session_id INTEGER")
    _ensure_column(conn, "call_link_resolutions", "ambient_session_id", "ambient_session_id INTEGER")
    _ensure_column(conn, "worker_time_capture_events", "call_session_id", "call_session_id INTEGER")
    _ensure_column(conn, "worker_time_capture_events", "call_leg_id", "call_leg_id INTEGER")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_call_transcripts_call_leg ON call_transcript_artifacts(call_leg_id, created_at DESC)"
    )
    conn.commit()


def _emit_state_event(
    conn: sqlite3.Connection,
    *,
    source_entity_id: str,
    event_type: str,
    payload: dict[str, Any],
    authority_class: str,
    correlation_id: str | None = None,
    causation_id: str | None = None,
    occurred_at: str | None = None,
) -> None:
    ensure_call_ops_tables(conn)
    occurred = occurred_at or _utc_now()
    ingested = _utc_now()
    payload_json = json.dumps(payload, sort_keys=True)
    payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    event_id = str(uuid.uuid4())
    idempotency_key = f"corkysoft:{event_type}:{source_entity_id}:{payload_hash}"
    conn.execute(
        """
        INSERT INTO state_egress_events (
            event_id,
            source_component,
            source_entity_id,
            event_type,
            idempotency_key,
            correlation_id,
            causation_id,
            authority_class,
            payload_json,
            payload_hash,
            occurred_at,
            ingested_at
        ) VALUES (?, 'corkysoft', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_id,
            source_entity_id,
            event_type,
            idempotency_key,
            correlation_id,
            causation_id,
            authority_class if authority_class in AUTHORITY_CLASSES else "compiled_state",
            payload_json,
            payload_hash,
            occurred,
            ingested,
        ),
    )


def _ensure_client_for_phone(
    conn: sqlite3.Connection,
    *,
    raw_phone: str | None,
    note: str | None = None,
) -> int | None:
    normalized = _normalize_phone(raw_phone)
    if not normalized:
        return None
    row = conn.execute(
        "SELECT id, phone FROM clients ORDER BY id"
    ).fetchall()
    for candidate in row:
        if _normalize_phone(candidate["phone"]) == normalized:
            return int(candidate["id"])
    timestamp = _utc_now()
    cursor = conn.execute(
        """
        INSERT INTO clients (
            first_name, last_name, company_name, email, phone,
            address_line1, address_line2, city, state, postcode, country, notes,
            created_at, updated_at
        ) VALUES (NULL, NULL, NULL, NULL, ?, NULL, NULL, NULL, NULL, NULL, NULL, ?, ?, ?)
        """,
        (raw_phone, note or "Auto-created from call intake.", timestamp, timestamp),
    )
    return int(cursor.lastrowid)


def _resolve_worker_by_phone(conn: sqlite3.Connection, phone: str | None) -> int | None:
    normalized = _normalize_phone(phone)
    if not normalized:
        return None
    rows = conn.execute("SELECT id, phone FROM workers WHERE active = 1 ORDER BY id").fetchall()
    for row in rows:
        if _normalize_phone(row["phone"]) == normalized:
            return int(row["id"])
    return None


def _insert_call_event_record(
    conn: sqlite3.Connection,
    *,
    event_kind: str,
    direction: str,
    status: str,
    source_channel: str,
    title: str | None,
    caller_phone: str | None,
    callee_phone: str | None,
    quote_id: int | None,
    job_id: int | None,
    segment_id: int | None,
    worker_id: int | None,
    operator_id: str | None,
    started_at: str | None,
    ended_at: str | None,
    captured_at: str | None,
    correlation_id: str,
    auto_create_client: bool,
) -> int:
    timestamp = _utc_now()
    client_id: int | None = None
    if auto_create_client:
        client_id = _ensure_client_for_phone(
            conn,
            raw_phone=caller_phone or callee_phone,
            note="Auto-created from call intake.",
        )
    cursor = conn.execute(
        """
        INSERT INTO call_events (
            event_kind, direction, status, source_channel, title,
            caller_phone, caller_phone_normalized,
            callee_phone, callee_phone_normalized,
            client_id, quote_id, job_id, segment_id, worker_id, operator_id,
            correlation_id, started_at, ended_at, captured_at, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_kind,
            direction,
            status,
            source_channel,
            title,
            caller_phone,
            _normalize_phone(caller_phone),
            callee_phone,
            _normalize_phone(callee_phone),
            client_id,
            quote_id,
            job_id,
            segment_id,
            worker_id,
            operator_id,
            correlation_id,
            started_at,
            ended_at,
            captured_at or timestamp,
            timestamp,
            timestamp,
        ),
    )
    event_id = int(cursor.lastrowid)
    _emit_state_event(
        conn,
        source_entity_id=f"call_event:{event_id}",
        event_type="call_event_created",
        payload={
            "callEventId": event_id,
            "eventKind": event_kind,
            "direction": direction,
            "sourceChannel": source_channel,
            "clientId": client_id,
            "jobId": job_id,
            "workerId": worker_id,
        },
        authority_class="compiled_state",
        correlation_id=correlation_id,
        occurred_at=timestamp,
    )
    return event_id


def _call_session_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "rootCallEventId": row["root_call_event_id"],
        "eventKind": row["event_kind"],
        "direction": row["direction"],
        "status": row["status"],
        "sourceChannel": row["source_channel"],
        "title": row["title"],
        "callerPhone": row["caller_phone"],
        "callerPhoneNormalized": row["caller_phone_normalized"],
        "calleePhone": row["callee_phone"],
        "calleePhoneNormalized": row["callee_phone_normalized"],
        "clientId": row["client_id"],
        "clientName": row["client_name"] if "client_name" in row.keys() else None,
        "quoteId": row["quote_id"],
        "jobId": row["job_id"],
        "segmentId": row["segment_id"],
        "workerId": row["worker_id"],
        "workerName": row["worker_name"] if "worker_name" in row.keys() else None,
        "operatorId": row["operator_id"],
        "correlationId": row["correlation_id"],
        "startedAt": row["started_at"],
        "endedAt": row["ended_at"],
        "capturedAt": row["captured_at"],
        "processedAt": row["processed_at"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
        "legCount": int(row["leg_count"] or 0) if "leg_count" in row.keys() else 0,
        "pendingActionCount": int(row["pending_action_count"] or 0) if "pending_action_count" in row.keys() else 0,
    }


def _call_leg_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "callSessionId": int(row["call_session_id"]),
        "rootCallEventId": row["root_call_event_id"],
        "legKind": row["leg_kind"],
        "direction": row["direction"],
        "status": row["status"],
        "sourceChannel": row["source_channel"],
        "destinationKind": row["destination_kind"],
        "destinationLabel": row["destination_label"],
        "operatorId": row["operator_id"],
        "callerPhone": row["caller_phone"],
        "callerPhoneNormalized": row["caller_phone_normalized"],
        "calleePhone": row["callee_phone"],
        "calleePhoneNormalized": row["callee_phone_normalized"],
        "startedAt": row["started_at"],
        "answeredAt": row["answered_at"],
        "endedAt": row["ended_at"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
        "latestTranscriptStatus": row["latest_transcript_status"] if "latest_transcript_status" in row.keys() else None,
    }


def _routing_event_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "callSessionId": int(row["call_session_id"]),
        "callLegId": row["call_leg_id"],
        "eventType": row["event_type"],
        "fromDestination": row["from_destination"],
        "toDestination": row["to_destination"],
        "actor": row["actor"],
        "detail": row["detail"],
        "createdAt": row["created_at"],
    }


def _ambient_session_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "title": row["title"],
        "sourceLocation": row["source_location"],
        "sourceDevice": row["source_device"],
        "teamLabel": row["team_label"],
        "status": row["status"],
        "clientId": row["client_id"],
        "clientName": row["client_name"] if "client_name" in row.keys() else None,
        "quoteId": row["quote_id"],
        "jobId": row["job_id"],
        "segmentId": row["segment_id"],
        "workerId": row["worker_id"],
        "workerName": row["worker_name"] if "worker_name" in row.keys() else None,
        "operatorId": row["operator_id"],
        "correlationId": row["correlation_id"],
        "startedAt": row["started_at"],
        "endedAt": row["ended_at"],
        "capturedAt": row["captured_at"],
        "processedAt": row["processed_at"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }


def log_call_routing_event(
    conn: sqlite3.Connection,
    *,
    call_session_id: int,
    event_type: str,
    call_leg_id: int | None = None,
    from_destination: str | None = None,
    to_destination: str | None = None,
    actor: str | None = None,
    detail: str | None = None,
) -> dict[str, Any]:
    ensure_call_ops_tables(conn)
    session = get_call_session(conn, call_session_id)
    normalized_event = event_type if event_type in CALL_ROUTING_EVENT_TYPES else "call_routed"
    timestamp = _utc_now()
    cursor = conn.execute(
        """
        INSERT INTO call_routing_events (
            call_session_id, call_leg_id, event_type, from_destination, to_destination, actor, detail, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (call_session_id, call_leg_id, normalized_event, from_destination, to_destination, actor, detail, timestamp),
    )
    routing_id = int(cursor.lastrowid)
    _emit_state_event(
        conn,
        source_entity_id=f"call_session:{call_session_id}",
        event_type=normalized_event,
        payload={
            "callSessionId": call_session_id,
            "callLegId": call_leg_id,
            "fromDestination": from_destination,
            "toDestination": to_destination,
            "actor": actor,
        },
        authority_class="compiled_state",
        correlation_id=session["correlationId"],
        occurred_at=timestamp,
    )
    return get_call_routing_event(conn, routing_id)


def create_call_leg(
    conn: sqlite3.Connection,
    *,
    call_session_id: int,
    leg_kind: str = "primary",
    direction: str = "inbound",
    status: str = "ringing",
    source_channel: str = "telephony",
    destination_kind: str | None = None,
    destination_label: str | None = None,
    operator_id: str | None = None,
    caller_phone: str | None = None,
    callee_phone: str | None = None,
    started_at: str | None = None,
    answered_at: str | None = None,
    ended_at: str | None = None,
    log_routed: bool = True,
) -> dict[str, Any]:
    ensure_call_ops_tables(conn)
    session = get_call_session(conn, call_session_id)
    timestamp = _utc_now()
    normalized_leg_kind = leg_kind if leg_kind in CALL_LEG_KINDS else "primary"
    normalized_direction = direction if direction in CALL_DIRECTIONS else session["direction"]
    normalized_status = status if status in CALL_STATUSES else "ringing"
    normalized_source = source_channel if source_channel in CALL_SOURCE_CHANNELS else session["sourceChannel"]
    cursor = conn.execute(
        """
        INSERT INTO call_legs (
            call_session_id, root_call_event_id, leg_kind, direction, status, source_channel,
            destination_kind, destination_label, operator_id,
            caller_phone, caller_phone_normalized, callee_phone, callee_phone_normalized,
            started_at, answered_at, ended_at, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            call_session_id,
            session["rootCallEventId"],
            normalized_leg_kind,
            normalized_direction,
            normalized_status,
            normalized_source,
            destination_kind,
            destination_label,
            operator_id,
            caller_phone or session["callerPhone"],
            _normalize_phone(caller_phone or session["callerPhone"]),
            callee_phone or session["calleePhone"],
            _normalize_phone(callee_phone or session["calleePhone"]),
            started_at or timestamp,
            answered_at,
            ended_at,
            timestamp,
            timestamp,
        ),
    )
    leg_id = int(cursor.lastrowid)
    if log_routed:
        log_call_routing_event(
            conn,
            call_session_id=call_session_id,
            call_leg_id=leg_id,
            event_type="call_routed",
            to_destination=destination_label or destination_kind,
            actor=operator_id,
            detail=f"Leg created as {normalized_leg_kind}.",
        )
    if answered_at or normalized_status in {"active", "completed"}:
        log_call_routing_event(
            conn,
            call_session_id=call_session_id,
            call_leg_id=leg_id,
            event_type="call_answered",
            to_destination=destination_label or destination_kind,
            actor=operator_id,
            detail="Leg answered; transcription may begin.",
        )
    conn.commit()
    return get_call_leg(conn, leg_id)


def create_call_session(
    conn: sqlite3.Connection,
    *,
    event_kind: str,
    direction: str,
    status: str = "completed",
    source_channel: str = "telephony",
    title: str | None = None,
    caller_phone: str | None = None,
    callee_phone: str | None = None,
    quote_id: int | None = None,
    job_id: int | None = None,
    segment_id: int | None = None,
    worker_id: int | None = None,
    operator_id: str | None = None,
    started_at: str | None = None,
    ended_at: str | None = None,
    captured_at: str | None = None,
    correlation_id: str | None = None,
    auto_create_client: bool = True,
    initial_destination_kind: str | None = None,
    initial_destination_label: str | None = None,
) -> dict[str, Any]:
    ensure_call_ops_tables(conn)
    timestamp = _utc_now()
    normalized_event_kind = event_kind if event_kind in CALL_EVENT_KINDS else "ops_call"
    normalized_direction = direction if direction in CALL_DIRECTIONS else "internal"
    normalized_status = status if status in CALL_STATUSES else "completed"
    normalized_source = source_channel if source_channel in CALL_SOURCE_CHANNELS else "telephony"
    correlation = correlation_id or str(uuid.uuid4())
    root_event_id = _insert_call_event_record(
        conn,
        event_kind=normalized_event_kind,
        direction=normalized_direction,
        status=normalized_status,
        source_channel=normalized_source,
        title=title,
        caller_phone=caller_phone,
        callee_phone=callee_phone,
        quote_id=quote_id,
        job_id=job_id,
        segment_id=segment_id,
        worker_id=worker_id,
        operator_id=operator_id,
        started_at=started_at,
        ended_at=ended_at,
        captured_at=captured_at,
        correlation_id=correlation,
        auto_create_client=auto_create_client,
    )
    root_event = get_call_event(conn, root_event_id)
    cursor = conn.execute(
        """
        INSERT INTO call_sessions (
            root_call_event_id, event_kind, direction, status, source_channel, title,
            caller_phone, caller_phone_normalized, callee_phone, callee_phone_normalized,
            client_id, quote_id, job_id, segment_id, worker_id, operator_id, correlation_id,
            started_at, ended_at, captured_at, processed_at, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            root_event_id,
            normalized_event_kind,
            normalized_direction,
            normalized_status,
            normalized_source,
            title,
            caller_phone,
            _normalize_phone(caller_phone),
            callee_phone,
            _normalize_phone(callee_phone),
            root_event["clientId"],
            quote_id,
            job_id,
            segment_id,
            worker_id,
            operator_id,
            correlation,
            started_at,
            ended_at,
            captured_at or timestamp,
            root_event["processedAt"],
            timestamp,
            timestamp,
        ),
    )
    session_id = int(cursor.lastrowid)
    _emit_state_event(
        conn,
        source_entity_id=f"call_session:{session_id}",
        event_type="call_session_created",
        payload={
            "callSessionId": session_id,
            "rootCallEventId": root_event_id,
            "eventKind": normalized_event_kind,
            "direction": normalized_direction,
        },
        authority_class="compiled_state",
        correlation_id=correlation,
        occurred_at=timestamp,
    )
    log_call_routing_event(
        conn,
        call_session_id=session_id,
        event_type="call_received",
        to_destination=initial_destination_label or initial_destination_kind or operator_id,
        actor=operator_id,
        detail="Call session received.",
    )
    initial_status = "ringing" if normalized_status == "ringing" else "active"
    answered_at_value = None if initial_status == "ringing" else (started_at or timestamp)
    create_call_leg(
        conn,
        call_session_id=session_id,
        leg_kind="timesheet" if normalized_event_kind in {"clock_on_call", "clock_off_call"} else "primary",
        direction=normalized_direction,
        status=initial_status,
        source_channel=normalized_source,
        destination_kind=initial_destination_kind or ("timesheet" if normalized_event_kind in {"clock_on_call", "clock_off_call"} else "operator"),
        destination_label=initial_destination_label or operator_id or "primary",
        operator_id=operator_id,
        caller_phone=caller_phone,
        callee_phone=callee_phone,
        started_at=started_at or timestamp,
        answered_at=answered_at_value,
        ended_at=ended_at if normalized_status == "completed" else None,
        log_routed=True,
    )
    conn.commit()
    return get_call_session(conn, session_id)


def create_call_event(
    conn: sqlite3.Connection,
    *,
    event_kind: str,
    direction: str,
    status: str = "completed",
    source_channel: str = "telephony",
    title: str | None = None,
    caller_phone: str | None = None,
    callee_phone: str | None = None,
    quote_id: int | None = None,
    job_id: int | None = None,
    segment_id: int | None = None,
    worker_id: int | None = None,
    operator_id: str | None = None,
    started_at: str | None = None,
    ended_at: str | None = None,
    captured_at: str | None = None,
    correlation_id: str | None = None,
    auto_create_client: bool = True,
) -> dict[str, Any]:
    session = create_call_session(
        conn,
        event_kind=event_kind,
        direction=direction,
        status=status,
        source_channel=source_channel,
        title=title,
        caller_phone=caller_phone,
        callee_phone=callee_phone,
        quote_id=quote_id,
        job_id=job_id,
        segment_id=segment_id,
        worker_id=worker_id,
        operator_id=operator_id,
        started_at=started_at,
        ended_at=ended_at,
        captured_at=captured_at,
        correlation_id=correlation_id,
        auto_create_client=auto_create_client,
    )
    return get_call_event(conn, int(session["rootCallEventId"]))


def list_call_sessions(
    conn: sqlite3.Connection,
    *,
    limit: int = 100,
    status: str | None = None,
    event_kind: str | None = None,
) -> list[dict[str, Any]]:
    ensure_call_ops_tables(conn)
    where: list[str] = []
    params: list[Any] = []
    if status:
        where.append("s.status = ?")
        params.append(status)
    if event_kind:
        where.append("s.event_kind = ?")
        params.append(event_kind)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    rows = conn.execute(
        f"""
        SELECT
            s.*,
            COALESCE(cl.company_name, TRIM(COALESCE(cl.first_name, '') || ' ' || COALESCE(cl.last_name, '')), '') AS client_name,
            w.name AS worker_name,
            (
                SELECT COUNT(*)
                FROM call_legs l
                WHERE l.call_session_id = s.id
            ) AS leg_count,
            (
                SELECT COUNT(*)
                FROM call_extracted_actions a
                WHERE a.call_event_id = s.root_call_event_id AND a.status = 'pending'
            ) AS pending_action_count
        FROM call_sessions s
        LEFT JOIN clients cl ON cl.id = s.client_id
        LEFT JOIN workers w ON w.id = s.worker_id
        {where_sql}
        ORDER BY s.created_at DESC, s.id DESC
        LIMIT ?
        """,
        (*params, limit),
    ).fetchall()
    return [_call_session_row_to_dict(row) for row in rows]


def get_call_session(conn: sqlite3.Connection, call_session_id: int) -> dict[str, Any]:
    ensure_call_ops_tables(conn)
    row = conn.execute(
        """
        SELECT
            s.*,
            COALESCE(cl.company_name, TRIM(COALESCE(cl.first_name, '') || ' ' || COALESCE(cl.last_name, '')), '') AS client_name,
            w.name AS worker_name,
            (
                SELECT COUNT(*)
                FROM call_legs l
                WHERE l.call_session_id = s.id
            ) AS leg_count,
            (
                SELECT COUNT(*)
                FROM call_extracted_actions a
                WHERE a.call_event_id = s.root_call_event_id AND a.status = 'pending'
            ) AS pending_action_count
        FROM call_sessions s
        LEFT JOIN clients cl ON cl.id = s.client_id
        LEFT JOIN workers w ON w.id = s.worker_id
        WHERE s.id = ?
        """,
        (call_session_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Unknown call session: {call_session_id}")
    return _call_session_row_to_dict(row)


def list_call_legs(conn: sqlite3.Connection, *, call_session_id: int) -> list[dict[str, Any]]:
    ensure_call_ops_tables(conn)
    rows = conn.execute(
        """
        SELECT
            l.*,
            (
                SELECT status
                FROM call_transcript_artifacts t
                WHERE t.call_leg_id = l.id
                ORDER BY t.created_at DESC, t.id DESC
                LIMIT 1
            ) AS latest_transcript_status
        FROM call_legs l
        WHERE l.call_session_id = ?
        ORDER BY l.created_at ASC, l.id ASC
        """,
        (call_session_id,),
    ).fetchall()
    return [_call_leg_row_to_dict(row) for row in rows]


def get_call_leg(conn: sqlite3.Connection, leg_id: int) -> dict[str, Any]:
    ensure_call_ops_tables(conn)
    row = conn.execute(
        """
        SELECT
            l.*,
            (
                SELECT status
                FROM call_transcript_artifacts t
                WHERE t.call_leg_id = l.id
                ORDER BY t.created_at DESC, t.id DESC
                LIMIT 1
            ) AS latest_transcript_status
        FROM call_legs l
        WHERE l.id = ?
        """,
        (leg_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Unknown call leg: {leg_id}")
    return _call_leg_row_to_dict(row)


def get_call_routing_event(conn: sqlite3.Connection, routing_event_id: int) -> dict[str, Any]:
    row = conn.execute(
        "SELECT * FROM call_routing_events WHERE id = ?",
        (routing_event_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Unknown call routing event: {routing_event_id}")
    return _routing_event_row_to_dict(row)


def list_call_routing_events(conn: sqlite3.Connection, *, call_session_id: int) -> list[dict[str, Any]]:
    ensure_call_ops_tables(conn)
    rows = conn.execute(
        "SELECT * FROM call_routing_events WHERE call_session_id = ? ORDER BY created_at ASC, id ASC",
        (call_session_id,),
    ).fetchall()
    return [_routing_event_row_to_dict(row) for row in rows]


def list_call_events(
    conn: sqlite3.Connection,
    *,
    limit: int = 100,
    status: str | None = None,
    event_kind: str | None = None,
    unresolved_only: bool = False,
) -> list[dict[str, Any]]:
    ensure_call_ops_tables(conn)
    where: list[str] = []
    params: list[Any] = []
    if status:
        where.append("c.status = ?")
        params.append(status)
    if event_kind:
        where.append("c.event_kind = ?")
        params.append(event_kind)
    if unresolved_only:
        where.append("(c.job_id IS NULL OR c.client_id IS NULL)")
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    rows = conn.execute(
        f"""
        SELECT
            c.*,
            COALESCE(cl.company_name, TRIM(COALESCE(cl.first_name, '') || ' ' || COALESCE(cl.last_name, '')), '') AS client_name,
            j.client AS job_client,
            j.origin AS job_origin,
            j.destination AS job_destination,
            w.name AS worker_name,
            (
                SELECT status
                FROM call_transcript_artifacts t
                WHERE t.call_event_id = c.id
                ORDER BY t.created_at DESC, t.id DESC
                LIMIT 1
            ) AS latest_transcript_status,
            (
                SELECT COUNT(*)
                FROM call_extracted_actions a
                WHERE a.call_event_id = c.id AND a.status = 'pending'
            ) AS pending_action_count
        FROM call_events c
        LEFT JOIN clients cl ON cl.id = c.client_id
        LEFT JOIN jobs j ON j.id = c.job_id
        LEFT JOIN workers w ON w.id = c.worker_id
        {where_sql}
        ORDER BY c.created_at DESC, c.id DESC
        LIMIT ?
        """,
        (*params, limit),
    ).fetchall()
    return [_call_row_to_dict(row) for row in rows]


def get_call_event(conn: sqlite3.Connection, call_event_id: int) -> dict[str, Any]:
    ensure_call_ops_tables(conn)
    row = conn.execute(
        """
        SELECT
            c.*,
            COALESCE(cl.company_name, TRIM(COALESCE(cl.first_name, '') || ' ' || COALESCE(cl.last_name, '')), '') AS client_name,
            j.client AS job_client,
            j.origin AS job_origin,
            j.destination AS job_destination,
            w.name AS worker_name
        FROM call_events c
        LEFT JOIN clients cl ON cl.id = c.client_id
        LEFT JOIN jobs j ON j.id = c.job_id
        LEFT JOIN workers w ON w.id = c.worker_id
        WHERE c.id = ?
        """,
        (call_event_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Unknown call event: {call_event_id}")
    return _call_row_to_dict(row)


def create_ambient_session(
    conn: sqlite3.Connection,
    *,
    title: str | None = None,
    source_location: str | None = None,
    source_device: str | None = None,
    team_label: str | None = None,
    status: str = "active",
    client_id: int | None = None,
    quote_id: int | None = None,
    job_id: int | None = None,
    segment_id: int | None = None,
    worker_id: int | None = None,
    operator_id: str | None = None,
    started_at: str | None = None,
    ended_at: str | None = None,
    captured_at: str | None = None,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    ensure_call_ops_tables(conn)
    timestamp = _utc_now()
    normalized_status = status if status in AMBIENT_SESSION_STATUSES else "active"
    cursor = conn.execute(
        """
        INSERT INTO ambient_sessions (
            title, source_location, source_device, team_label, status,
            client_id, quote_id, job_id, segment_id, worker_id, operator_id,
            correlation_id, started_at, ended_at, captured_at, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            title,
            source_location,
            source_device,
            team_label,
            normalized_status,
            client_id,
            quote_id,
            job_id,
            segment_id,
            worker_id,
            operator_id,
            correlation_id or str(uuid.uuid4()),
            started_at or timestamp,
            ended_at,
            captured_at or timestamp,
            timestamp,
            timestamp,
        ),
    )
    ambient_id = int(cursor.lastrowid)
    row = get_ambient_session(conn, ambient_id)
    _emit_state_event(
        conn,
        source_entity_id=f"ambient_session:{ambient_id}",
        event_type="ambient_session_created",
        payload={
            "ambientSessionId": ambient_id,
            "title": title,
            "jobId": job_id,
            "segmentId": segment_id,
        },
        authority_class="observer_capture_ref",
        correlation_id=row["correlationId"],
        occurred_at=timestamp,
    )
    conn.commit()
    return row


def get_ambient_session(conn: sqlite3.Connection, ambient_session_id: int) -> dict[str, Any]:
    ensure_call_ops_tables(conn)
    row = conn.execute(
        """
        SELECT
            a.*,
            COALESCE(cl.company_name, TRIM(COALESCE(cl.first_name, '') || ' ' || COALESCE(cl.last_name, '')), '') AS client_name,
            w.name AS worker_name
        FROM ambient_sessions a
        LEFT JOIN clients cl ON cl.id = a.client_id
        LEFT JOIN workers w ON w.id = a.worker_id
        WHERE a.id = ?
        """,
        (ambient_session_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Unknown ambient session: {ambient_session_id}")
    return _ambient_session_row_to_dict(row)


def list_ambient_sessions(conn: sqlite3.Connection, *, limit: int = 100) -> list[dict[str, Any]]:
    ensure_call_ops_tables(conn)
    rows = conn.execute(
        """
        SELECT
            a.*,
            COALESCE(cl.company_name, TRIM(COALESCE(cl.first_name, '') || ' ' || COALESCE(cl.last_name, '')), '') AS client_name,
            w.name AS worker_name
        FROM ambient_sessions a
        LEFT JOIN clients cl ON cl.id = a.client_id
        LEFT JOIN workers w ON w.id = a.worker_id
        ORDER BY a.created_at DESC, a.id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [_ambient_session_row_to_dict(row) for row in rows]


def _call_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "eventKind": row["event_kind"],
        "direction": row["direction"],
        "status": row["status"],
        "sourceChannel": row["source_channel"],
        "title": row["title"],
        "callerPhone": row["caller_phone"],
        "callerPhoneNormalized": row["caller_phone_normalized"],
        "calleePhone": row["callee_phone"],
        "calleePhoneNormalized": row["callee_phone_normalized"],
        "clientId": row["client_id"],
        "clientName": (row["client_name"] or None),
        "quoteId": row["quote_id"],
        "jobId": row["job_id"],
        "segmentId": row["segment_id"],
        "workerId": row["worker_id"],
        "workerName": row["worker_name"] if "worker_name" in row.keys() else None,
        "jobClient": row["job_client"] if "job_client" in row.keys() else None,
        "jobOrigin": row["job_origin"] if "job_origin" in row.keys() else None,
        "jobDestination": row["job_destination"] if "job_destination" in row.keys() else None,
        "operatorId": row["operator_id"],
        "correlationId": row["correlation_id"],
        "startedAt": row["started_at"],
        "endedAt": row["ended_at"],
        "capturedAt": row["captured_at"],
        "processedAt": row["processed_at"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
        "latestTranscriptStatus": row["latest_transcript_status"] if "latest_transcript_status" in row.keys() else None,
        "pendingActionCount": int(row["pending_action_count"] or 0) if "pending_action_count" in row.keys() else 0,
    }


def add_call_note(
    conn: sqlite3.Connection,
    *,
    call_event_id: int | None = None,
    ambient_session_id: int | None = None,
    author: str | None,
    note_text: str,
    note_kind: str = "operator",
    authoritative: bool = True,
) -> dict[str, Any]:
    ensure_call_ops_tables(conn)
    if not note_text.strip():
        raise ValueError("Call note text is required")
    if call_event_id is None and ambient_session_id is None:
        raise ValueError("A call event or ambient session is required")
    correlation_id: str | None = None
    if call_event_id is not None:
        get_call_event(conn, call_event_id)
        correlation_id = get_call_event(conn, call_event_id)["correlationId"]
    if ambient_session_id is not None:
        correlation_id = get_ambient_session(conn, ambient_session_id)["correlationId"]
    timestamp = _utc_now()
    cursor = conn.execute(
        """
        INSERT INTO call_notes (call_event_id, ambient_session_id, author, note_kind, note_text, authoritative, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (call_event_id, ambient_session_id, author, note_kind or "operator", note_text.strip(), 1 if authoritative else 0, timestamp),
    )
    note_id = int(cursor.lastrowid)
    _emit_state_event(
        conn,
        source_entity_id=f"ambient_session:{ambient_session_id}" if ambient_session_id is not None else f"call_event:{call_event_id}",
        event_type="call_note_added",
        payload={
            "callEventId": call_event_id,
            "ambientSessionId": ambient_session_id,
            "noteId": note_id,
            "author": author,
            "noteKind": note_kind or "operator",
            "authoritative": bool(authoritative),
        },
        authority_class="compiled_state" if authoritative else "observer_capture_ref",
        correlation_id=correlation_id,
        occurred_at=timestamp,
    )
    conn.commit()
    return {
        "id": note_id,
        "callEventId": call_event_id,
        "ambientSessionId": ambient_session_id,
        "author": author,
        "noteKind": note_kind or "operator",
        "noteText": note_text.strip(),
        "authoritative": bool(authoritative),
        "createdAt": timestamp,
    }


def list_call_notes(
    conn: sqlite3.Connection,
    *,
    call_event_id: int | None = None,
    ambient_session_id: int | None = None,
) -> list[dict[str, Any]]:
    ensure_call_ops_tables(conn)
    if call_event_id is None and ambient_session_id is None:
        raise ValueError("A call event or ambient session is required")
    if ambient_session_id is not None:
        rows = conn.execute(
            "SELECT * FROM call_notes WHERE ambient_session_id = ? ORDER BY created_at DESC, id DESC",
            (ambient_session_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM call_notes WHERE call_event_id = ? ORDER BY created_at DESC, id DESC",
            (call_event_id,),
        ).fetchall()
    return [
        {
            "id": int(row["id"]),
            "callEventId": int(row["call_event_id"]) if row["call_event_id"] is not None else None,
            "ambientSessionId": row["ambient_session_id"] if "ambient_session_id" in row.keys() else None,
            "author": row["author"],
            "noteKind": row["note_kind"],
            "noteText": row["note_text"],
            "authoritative": bool(row["authoritative"]),
            "createdAt": row["created_at"],
        }
        for row in rows
    ]


def add_extracted_action(
    conn: sqlite3.Connection,
    *,
    call_event_id: int | None = None,
    ambient_session_id: int | None = None,
    action_text: str,
    source_engine: str | None = None,
    transcript_artifact_id: int | None = None,
    span_start: float | None = None,
    span_end: float | None = None,
) -> dict[str, Any]:
    ensure_call_ops_tables(conn)
    if not action_text.strip():
        raise ValueError("Extracted action text is required")
    if call_event_id is None and ambient_session_id is None:
        raise ValueError("A call event or ambient session is required")
    correlation_id: str | None = None
    if call_event_id is not None:
        correlation_id = get_call_event(conn, call_event_id)["correlationId"]
    if ambient_session_id is not None:
        correlation_id = get_ambient_session(conn, ambient_session_id)["correlationId"]
    timestamp = _utc_now()
    cursor = conn.execute(
        """
        INSERT INTO call_extracted_actions (
            call_event_id, ambient_session_id, transcript_artifact_id, source_engine, action_text,
            span_start, span_end, status, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)
        """,
        (
            call_event_id,
            ambient_session_id,
            transcript_artifact_id,
            source_engine,
            action_text.strip(),
            span_start,
            span_end,
            timestamp,
        ),
    )
    action_id = int(cursor.lastrowid)
    _emit_state_event(
        conn,
        source_entity_id=f"ambient_session:{ambient_session_id}" if ambient_session_id is not None else f"call_event:{call_event_id}",
        event_type="extracted_action_created",
        payload={
            "callEventId": call_event_id,
            "ambientSessionId": ambient_session_id,
            "actionId": action_id,
            "sourceEngine": source_engine,
        },
        authority_class="observer_capture_ref",
        correlation_id=correlation_id,
        occurred_at=timestamp,
    )
    conn.commit()
    return get_extracted_action(conn, action_id)


def get_extracted_action(conn: sqlite3.Connection, action_id: int) -> dict[str, Any]:
    row = conn.execute(
        "SELECT * FROM call_extracted_actions WHERE id = ?",
        (action_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Unknown extracted action: {action_id}")
    return {
        "id": int(row["id"]),
        "callEventId": int(row["call_event_id"]) if row["call_event_id"] is not None else None,
        "ambientSessionId": row["ambient_session_id"] if "ambient_session_id" in row.keys() else None,
        "transcriptArtifactId": row["transcript_artifact_id"],
        "sourceEngine": row["source_engine"],
        "actionText": row["action_text"],
        "spanStart": row["span_start"],
        "spanEnd": row["span_end"],
        "status": row["status"],
        "decidedBy": row["decided_by"],
        "decisionNote": row["decision_note"],
        "createdAt": row["created_at"],
        "decidedAt": row["decided_at"],
    }


def list_extracted_actions(
    conn: sqlite3.Connection,
    *,
    call_event_id: int | None = None,
    ambient_session_id: int | None = None,
) -> list[dict[str, Any]]:
    if call_event_id is None and ambient_session_id is None:
        raise ValueError("A call event or ambient session is required")
    if ambient_session_id is not None:
        rows = conn.execute(
            "SELECT * FROM call_extracted_actions WHERE ambient_session_id = ? ORDER BY created_at DESC, id DESC",
            (ambient_session_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM call_extracted_actions WHERE call_event_id = ? ORDER BY created_at DESC, id DESC",
            (call_event_id,),
        ).fetchall()
    return [get_extracted_action(conn, int(row["id"])) for row in rows]


def decide_extracted_action(
    conn: sqlite3.Connection,
    *,
    action_id: int,
    status: str,
    decided_by: str | None,
    decision_note: str | None = None,
) -> dict[str, Any]:
    ensure_call_ops_tables(conn)
    normalized = status if status in EXTRACTED_ACTION_STATUSES else "pending"
    if normalized == "pending":
        raise ValueError("Extracted action decision must be accepted or rejected")
    action = get_extracted_action(conn, action_id)
    timestamp = _utc_now()
    conn.execute(
        """
        UPDATE call_extracted_actions
        SET status = ?, decided_by = ?, decision_note = ?, decided_at = ?
        WHERE id = ?
        """,
        (normalized, decided_by, decision_note, timestamp, action_id),
    )
    call = get_call_event(conn, int(action["callEventId"]))
    _emit_state_event(
        conn,
        source_entity_id=(
            f"ambient_session:{action['ambientSessionId']}"
            if action.get("ambientSessionId") is not None
            else f"call_event:{action['callEventId']}"
        ),
        event_type="extracted_action_decided",
        payload={
            "callEventId": action["callEventId"],
            "ambientSessionId": action.get("ambientSessionId"),
            "actionId": action_id,
            "status": normalized,
            "decidedBy": decided_by,
        },
        authority_class="compiled_state" if normalized == "accepted" else "observer_capture_ref",
        correlation_id=(
            get_ambient_session(conn, int(action["ambientSessionId"]))["correlationId"]
            if action.get("ambientSessionId") is not None
            else call["correlationId"]
        ),
        occurred_at=timestamp,
    )
    conn.commit()
    return get_extracted_action(conn, action_id)


def resolve_call_links(
    conn: sqlite3.Connection,
    *,
    call_event_id: int | None = None,
    ambient_session_id: int | None = None,
    actor: str | None = None,
    client_id: int | None = None,
    quote_id: int | None = None,
    job_id: int | None = None,
    segment_id: int | None = None,
    worker_id: int | None = None,
    resolution_note: str | None = None,
) -> dict[str, Any]:
    ensure_call_ops_tables(conn)
    if call_event_id is None and ambient_session_id is None:
        raise ValueError("A call event or ambient session is required")
    timestamp = _utc_now()
    correlation_id: str | None = None
    if call_event_id is not None:
        call = get_call_event(conn, call_event_id)
        correlation_id = call["correlationId"]
        conn.execute(
            """
            UPDATE call_events
            SET client_id = COALESCE(?, client_id),
                quote_id = COALESCE(?, quote_id),
                job_id = COALESCE(?, job_id),
                segment_id = COALESCE(?, segment_id),
                worker_id = COALESCE(?, worker_id),
                status = CASE WHEN status = 'needs_review' AND (? IS NOT NULL OR ? IS NOT NULL OR ? IS NOT NULL OR ? IS NOT NULL) THEN 'completed' ELSE status END,
                updated_at = ?
            WHERE id = ?
            """,
            (client_id, quote_id, job_id, segment_id, worker_id, client_id, job_id, segment_id, worker_id, timestamp, call_event_id),
        )
        conn.execute(
            """
            UPDATE call_sessions
            SET client_id = COALESCE(?, client_id),
                quote_id = COALESCE(?, quote_id),
                job_id = COALESCE(?, job_id),
                segment_id = COALESCE(?, segment_id),
                worker_id = COALESCE(?, worker_id),
                updated_at = ?
            WHERE root_call_event_id = ?
            """,
            (client_id, quote_id, job_id, segment_id, worker_id, timestamp, call_event_id),
        )
    if ambient_session_id is not None:
        ambient = get_ambient_session(conn, ambient_session_id)
        correlation_id = ambient["correlationId"]
        conn.execute(
            """
            UPDATE ambient_sessions
            SET client_id = COALESCE(?, client_id),
                quote_id = COALESCE(?, quote_id),
                job_id = COALESCE(?, job_id),
                segment_id = COALESCE(?, segment_id),
                worker_id = COALESCE(?, worker_id),
                updated_at = ?
            WHERE id = ?
            """,
            (client_id, quote_id, job_id, segment_id, worker_id, timestamp, ambient_session_id),
        )
    conn.execute(
        """
        INSERT INTO call_link_resolutions (
            call_event_id, ambient_session_id, actor, client_id, quote_id, job_id, segment_id, worker_id, resolution_note, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (call_event_id, ambient_session_id, actor, client_id, quote_id, job_id, segment_id, worker_id, resolution_note, timestamp),
    )
    _emit_state_event(
        conn,
        source_entity_id=f"ambient_session:{ambient_session_id}" if ambient_session_id is not None else f"call_event:{call_event_id}",
        event_type="call_link_resolved",
        payload={
            "callEventId": call_event_id,
            "ambientSessionId": ambient_session_id,
            "clientId": client_id,
            "quoteId": quote_id,
            "jobId": job_id,
            "segmentId": segment_id,
            "workerId": worker_id,
            "actor": actor,
        },
        authority_class="compiled_state",
        correlation_id=correlation_id,
        occurred_at=timestamp,
    )
    conn.commit()
    if ambient_session_id is not None:
        return get_ambient_session(conn, ambient_session_id)
    return get_call_event(conn, int(call_event_id))


def record_transcript_artifact(
    conn: sqlite3.Connection,
    *,
    call_event_id: int | None = None,
    call_leg_id: int | None = None,
    service_key: str,
    status: str,
    external_task_id: str | None = None,
    transcript_text: str | None = None,
    transcript_segments: Sequence[dict[str, Any]] | None = None,
    diarization: Sequence[dict[str, Any]] | None = None,
    confidence: float | None = None,
    is_final: bool = False,
    error_message: str | None = None,
) -> dict[str, Any]:
    ensure_call_ops_tables(conn)
    if call_event_id is None and call_leg_id is None:
        raise ValueError("A call event or call leg is required")
    session_id: int | None = None
    if call_leg_id is not None:
        leg = get_call_leg(conn, call_leg_id)
        session_id = int(leg["callSessionId"])
        if call_event_id is None:
            call_event_id = leg["rootCallEventId"]
    call = get_call_event(conn, int(call_event_id)) if call_event_id is not None else None
    if session_id is None and call_event_id is not None:
        session_row = conn.execute(
            "SELECT id FROM call_sessions WHERE root_call_event_id = ? ORDER BY id DESC LIMIT 1",
            (call_event_id,),
        ).fetchone()
        session_id = int(session_row["id"]) if session_row is not None else None
    normalized_status = status if status in CALL_TRANSCRIPT_STATUSES else "queued"
    timestamp = _utc_now()
    cursor = conn.execute(
        """
        INSERT INTO call_transcript_artifacts (
            call_event_id, call_session_id, call_leg_id, service_key, external_task_id, status, transcript_text,
            transcript_segments_json, diarization_json, confidence, is_final, error_message,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            call_event_id,
            session_id,
            call_leg_id,
            service_key,
            external_task_id,
            normalized_status,
            transcript_text,
            json.dumps(list(transcript_segments or [])),
            json.dumps(list(diarization or [])),
            confidence,
            1 if is_final else 0,
            error_message,
            timestamp,
            timestamp,
        ),
    )
    artifact_id = int(cursor.lastrowid)
    if normalized_status == "completed":
        if call_event_id is not None:
            conn.execute(
                "UPDATE call_events SET processed_at = ?, updated_at = ? WHERE id = ?",
                (timestamp, timestamp, call_event_id),
            )
        if session_id is not None:
            conn.execute(
                "UPDATE call_sessions SET processed_at = ?, updated_at = ? WHERE id = ?",
                (timestamp, timestamp, session_id),
            )
    _emit_state_event(
        conn,
        source_entity_id=f"call_leg:{call_leg_id}" if call_leg_id is not None else f"call_event:{call_event_id}",
        event_type="transcript_available" if normalized_status == "completed" else "transcript_task_updated",
        payload={
            "callEventId": call_event_id,
            "callSessionId": session_id,
            "callLegId": call_leg_id,
            "artifactId": artifact_id,
            "status": normalized_status,
            "serviceKey": service_key,
            "externalTaskId": external_task_id,
            "isFinal": bool(is_final),
        },
        authority_class="observer_capture_ref",
        correlation_id=call["correlationId"] if call is not None else None,
        occurred_at=timestamp,
    )
    conn.commit()
    return get_transcript_artifact(conn, artifact_id)


def generate_fake_transcript_artifact(
    conn: sqlite3.Connection,
    *,
    call_event_id: int | None = None,
    call_leg_id: int | None = None,
    scenario: str | None = None,
    operator_goal: str | None = None,
    service_key: str = "ops",
) -> dict[str, Any]:
    ensure_call_ops_tables(conn)
    if call_event_id is None and call_leg_id is None:
        raise ValueError("A call event or call leg is required")
    call: dict[str, Any]
    leg: dict[str, Any] | None = None
    if call_leg_id is not None:
        leg = get_call_leg(conn, call_leg_id)
        call = get_call_event(conn, int(leg["rootCallEventId"]))
    else:
        call = get_call_event(conn, int(call_event_id))
    lines: list[str] = []
    opener = {
        "client_call": "Caller explains the issue and asks what happens next.",
        "manager_call": "Manager provides updated operating direction for the job.",
        "worker_call": "Worker reports live site conditions and requests clarification.",
        "clock_on_call": "Worker states they are clocking on and confirms their identifier.",
        "clock_off_call": "Worker states they are clocking off and confirms the end time.",
    }.get(call["eventKind"], "Operator and caller discuss the current operational situation.")
    lines.append(opener)
    if call.get("clientName"):
        lines.append(f"Linked client context: {call['clientName']}.")
    if call.get("jobId"):
        route_bits = " to ".join(
            bit for bit in (call.get("jobOrigin"), call.get("jobDestination")) if bit
        )
        if route_bits:
            lines.append(f"Linked job #{call['jobId']} route context: {route_bits}.")
        else:
            lines.append(f"Linked job #{call['jobId']} is referenced in the conversation.")
    if call.get("workerName"):
        lines.append(f"Worker context: {call['workerName']}.")
    if leg is not None:
        lines.append(
            f"Leg context: {leg.get('legKind')} to {leg.get('destinationLabel') or leg.get('destinationKind') or 'unknown destination'}."
        )
    if scenario and str(scenario).strip():
        lines.append(f"Scenario note: {str(scenario).strip()}.")
    if operator_goal and str(operator_goal).strip():
        lines.append(f"Desired outcome: {str(operator_goal).strip()}.")
    lines.append("Operator records the agreed next action and keeps the call attached to the operational record.")
    transcript_text = " ".join(lines)
    return record_transcript_artifact(
        conn,
        call_event_id=call_event_id,
        call_leg_id=call_leg_id,
        service_key=service_key,
        status="completed",
        transcript_text=transcript_text,
        transcript_segments=[],
        diarization=[],
        confidence=0.5,
        is_final=True,
    )


def get_transcript_artifact(conn: sqlite3.Connection, artifact_id: int) -> dict[str, Any]:
    row = conn.execute(
        "SELECT * FROM call_transcript_artifacts WHERE id = ?",
        (artifact_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Unknown transcript artifact: {artifact_id}")
    return {
        "id": int(row["id"]),
        "callEventId": int(row["call_event_id"]) if row["call_event_id"] is not None else None,
        "callSessionId": row["call_session_id"] if "call_session_id" in row.keys() else None,
        "callLegId": row["call_leg_id"] if "call_leg_id" in row.keys() else None,
        "serviceKey": row["service_key"],
        "externalTaskId": row["external_task_id"],
        "status": row["status"],
        "transcriptText": row["transcript_text"],
        "transcriptSegments": json.loads(row["transcript_segments_json"] or "[]"),
        "diarization": json.loads(row["diarization_json"] or "[]"),
        "confidence": row["confidence"],
        "isFinal": bool(row["is_final"]),
        "errorMessage": row["error_message"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }


def list_transcript_artifacts(
    conn: sqlite3.Connection,
    *,
    call_event_id: int | None = None,
    call_leg_id: int | None = None,
    call_session_id: int | None = None,
) -> list[dict[str, Any]]:
    if call_event_id is None and call_leg_id is None and call_session_id is None:
        raise ValueError("A call event, session, or leg is required")
    if call_leg_id is not None:
        rows = conn.execute(
            "SELECT id FROM call_transcript_artifacts WHERE call_leg_id = ? ORDER BY created_at DESC, id DESC",
            (call_leg_id,),
        ).fetchall()
    elif call_session_id is not None:
        rows = conn.execute(
            "SELECT id FROM call_transcript_artifacts WHERE call_session_id = ? ORDER BY created_at DESC, id DESC",
            (call_session_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id FROM call_transcript_artifacts WHERE call_event_id = ? ORDER BY created_at DESC, id DESC",
            (call_event_id,),
        ).fetchall()
    return [get_transcript_artifact(conn, int(row["id"])) for row in rows]


def submit_call_audio_for_transcription(
    conn: sqlite3.Connection,
    *,
    call_event_id: int | None = None,
    call_leg_id: int | None = None,
    service_key: str,
    file_bytes: bytes,
    filename: str,
    language: str | None = None,
    diarize: bool = True,
) -> dict[str, Any]:
    ensure_call_ops_tables(conn)
    if call_event_id is None and call_leg_id is None:
        raise ValueError("A call event or call leg is required")
    if call_leg_id is not None:
        leg = get_call_leg(conn, call_leg_id)
        if call_event_id is None:
            call_event_id = leg["rootCallEventId"]
    if call_event_id is not None:
        get_call_event(conn, int(call_event_id))
    payload = submit_transcription(
        service_key=service_key,
        file_bytes=file_bytes,
        filename=filename,
        language=language,
        diarize=diarize,
    )
    return record_transcript_artifact(
        conn,
        call_event_id=call_event_id,
        call_leg_id=call_leg_id,
        service_key=service_key,
        status=str(payload.get("status") or "queued").lower(),
        external_task_id=str(payload.get("identifier")),
    )


def poll_transcript_artifact(conn: sqlite3.Connection, *, artifact_id: int) -> dict[str, Any]:
    ensure_call_ops_tables(conn)
    artifact = get_transcript_artifact(conn, artifact_id)
    if not artifact["externalTaskId"]:
        raise ValueError("Transcript artifact has no external task id")
    payload = fetch_task_status(
        service_key=str(artifact["serviceKey"]),
        identifier=str(artifact["externalTaskId"]),
    )
    normalized_status = str(payload.get("status") or "queued").lower()
    result = payload.get("result")
    transcript_text = None
    transcript_segments: list[dict[str, Any]] = []
    diarization: list[dict[str, Any]] = []
    confidence: float | None = None
    error_message = payload.get("message") if normalized_status == "failed" else None
    if isinstance(result, list):
        transcript_segments = [segment for segment in result if isinstance(segment, dict)]
        transcript_text = "\n".join(
            str(segment.get("text") or "").strip() for segment in transcript_segments if str(segment.get("text") or "").strip()
        ) or None
        diarization = [
            {
                "speaker": segment.get("speaker"),
                "start": segment.get("start"),
                "end": segment.get("end"),
            }
            for segment in transcript_segments
            if segment.get("speaker") is not None
        ]
        confidences = [
            float(segment.get("confidence"))
            for segment in transcript_segments
            if segment.get("confidence") is not None
        ]
        if confidences:
            confidence = round(sum(confidences) / len(confidences), 4)
    timestamp = _utc_now()
    conn.execute(
        """
        UPDATE call_transcript_artifacts
        SET status = ?, transcript_text = ?, transcript_segments_json = ?,
            diarization_json = ?, confidence = ?, is_final = ?, error_message = ?, updated_at = ?
        WHERE id = ?
        """,
        (
            normalized_status if normalized_status in CALL_TRANSCRIPT_STATUSES else "queued",
            transcript_text,
            json.dumps(transcript_segments),
            json.dumps(diarization),
            confidence,
            1 if normalized_status == "completed" else 0,
            error_message,
            timestamp,
            artifact_id,
        ),
    )
    if normalized_status == "completed":
        if artifact["callEventId"] is not None:
            conn.execute(
                "UPDATE call_events SET processed_at = ?, updated_at = ? WHERE id = ?",
                (timestamp, timestamp, artifact["callEventId"]),
            )
        if artifact.get("callSessionId") is not None:
            conn.execute(
                "UPDATE call_sessions SET processed_at = ?, updated_at = ? WHERE id = ?",
                (timestamp, timestamp, artifact["callSessionId"]),
            )
    call = get_call_event(conn, int(artifact["callEventId"])) if artifact["callEventId"] is not None else None
    _emit_state_event(
        conn,
        source_entity_id=f"call_leg:{artifact['callLegId']}" if artifact.get("callLegId") is not None else f"call_event:{artifact['callEventId']}",
        event_type="transcript_available" if normalized_status == "completed" else "transcript_task_updated",
        payload={
            "callEventId": artifact["callEventId"],
            "callSessionId": artifact.get("callSessionId"),
            "callLegId": artifact.get("callLegId"),
            "artifactId": artifact_id,
            "status": normalized_status,
            "externalTaskId": artifact["externalTaskId"],
        },
        authority_class="observer_capture_ref",
        correlation_id=call["correlationId"] if call is not None else None,
        occurred_at=timestamp,
    )
    conn.commit()
    return get_transcript_artifact(conn, artifact_id)


def record_ambient_transcript_artifact(
    conn: sqlite3.Connection,
    *,
    ambient_session_id: int,
    service_key: str,
    status: str,
    transcript_text: str | None = None,
    transcript_segments: Sequence[dict[str, Any]] | None = None,
    diarization: Sequence[dict[str, Any]] | None = None,
    confidence: float | None = None,
    is_final: bool = False,
    error_message: str | None = None,
) -> dict[str, Any]:
    ensure_call_ops_tables(conn)
    ambient = get_ambient_session(conn, ambient_session_id)
    normalized_status = status if status in CALL_TRANSCRIPT_STATUSES else "queued"
    timestamp = _utc_now()
    cursor = conn.execute(
        """
        INSERT INTO ambient_transcript_artifacts (
            ambient_session_id, service_key, status, transcript_text,
            transcript_segments_json, diarization_json, confidence, is_final, error_message,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ambient_session_id,
            service_key,
            normalized_status,
            transcript_text,
            json.dumps(list(transcript_segments or [])),
            json.dumps(list(diarization or [])),
            confidence,
            1 if is_final else 0,
            error_message,
            timestamp,
            timestamp,
        ),
    )
    artifact_id = int(cursor.lastrowid)
    if normalized_status == "completed":
        conn.execute(
            "UPDATE ambient_sessions SET processed_at = ?, updated_at = ? WHERE id = ?",
            (timestamp, timestamp, ambient_session_id),
        )
    _emit_state_event(
        conn,
        source_entity_id=f"ambient_session:{ambient_session_id}",
        event_type="ambient_transcript_available" if normalized_status == "completed" else "ambient_transcript_updated",
        payload={
            "ambientSessionId": ambient_session_id,
            "artifactId": artifact_id,
            "status": normalized_status,
            "serviceKey": service_key,
        },
        authority_class="observer_capture_ref",
        correlation_id=ambient["correlationId"],
        occurred_at=timestamp,
    )
    conn.commit()
    return get_ambient_transcript_artifact(conn, artifact_id)


def generate_fake_ambient_transcript_artifact(
    conn: sqlite3.Connection,
    *,
    ambient_session_id: int,
    scenario: str | None = None,
    operator_goal: str | None = None,
    service_key: str = "ops",
) -> dict[str, Any]:
    ambient = get_ambient_session(conn, ambient_session_id)
    lines = [
        "Ambient office discussion is recorded for operational review.",
    ]
    if ambient.get("teamLabel"):
        lines.append(f"Team context: {ambient['teamLabel']}.")
    if ambient.get("sourceLocation"):
        lines.append(f"Location context: {ambient['sourceLocation']}.")
    if ambient.get("jobId"):
        lines.append(f"Linked job #{ambient['jobId']} is discussed.")
    if scenario and str(scenario).strip():
        lines.append(f"Scenario note: {str(scenario).strip()}.")
    if operator_goal and str(operator_goal).strip():
        lines.append(f"Desired outcome: {str(operator_goal).strip()}.")
    lines.append("Accepted operational notes should be promoted explicitly after review.")
    return record_ambient_transcript_artifact(
        conn,
        ambient_session_id=ambient_session_id,
        service_key=service_key,
        status="completed",
        transcript_text=" ".join(lines),
        transcript_segments=[],
        diarization=[],
        confidence=0.5,
        is_final=True,
    )


def get_ambient_transcript_artifact(conn: sqlite3.Connection, artifact_id: int) -> dict[str, Any]:
    row = conn.execute(
        "SELECT * FROM ambient_transcript_artifacts WHERE id = ?",
        (artifact_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Unknown ambient transcript artifact: {artifact_id}")
    return {
        "id": int(row["id"]),
        "ambientSessionId": int(row["ambient_session_id"]),
        "serviceKey": row["service_key"],
        "status": row["status"],
        "transcriptText": row["transcript_text"],
        "transcriptSegments": json.loads(row["transcript_segments_json"] or "[]"),
        "diarization": json.loads(row["diarization_json"] or "[]"),
        "confidence": row["confidence"],
        "isFinal": bool(row["is_final"]),
        "errorMessage": row["error_message"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }


def list_ambient_transcript_artifacts(conn: sqlite3.Connection, *, ambient_session_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT id FROM ambient_transcript_artifacts WHERE ambient_session_id = ? ORDER BY created_at DESC, id DESC",
        (ambient_session_id,),
    ).fetchall()
    return [get_ambient_transcript_artifact(conn, int(row["id"])) for row in rows]


def record_worker_time_capture_event(
    conn: sqlite3.Connection,
    *,
    event_type: str,
    channel: str,
    effective_timestamp: str | None = None,
    captured_timestamp: str | None = None,
    worker_id: int | None = None,
    worker_name_raw: str | None = None,
    employee_code_raw: str | None = None,
    caller_phone: str | None = None,
    call_event_id: int | None = None,
    call_session_id: int | None = None,
    call_leg_id: int | None = None,
    job_id: int | None = None,
    segment_id: int | None = None,
    truck_id: str | None = None,
    confidence: float | None = None,
    raw_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ensure_call_ops_tables(conn)
    if call_leg_id is not None:
        leg = get_call_leg(conn, call_leg_id)
        call_session_id = call_session_id or int(leg["callSessionId"])
        call_event_id = call_event_id or leg["rootCallEventId"]
    normalized_type = event_type if event_type in WORKER_TIME_EVENT_TYPES else None
    if normalized_type is None:
        raise ValueError("Worker time event type must be clock_on or clock_off")
    normalized_channel = channel if channel in WORKER_TIME_CHANNELS else "manual_supervisor"
    resolved_worker_id = worker_id or _resolve_worker_by_phone(conn, caller_phone)
    score = float(confidence) if confidence is not None else 0.0
    anomaly_flags: list[str] = []
    effective_value = effective_timestamp or captured_timestamp or _utc_now()
    duplicate_row = None
    if resolved_worker_id is not None:
        duplicate_row = conn.execute(
            """
            SELECT id
            FROM worker_time_capture_events
            WHERE worker_id = ?
              AND event_type = ?
              AND COALESCE(effective_timestamp, captured_timestamp) = ?
              AND review_status != 'rejected'
            ORDER BY id DESC
            LIMIT 1
            """,
            (resolved_worker_id, normalized_type, effective_value),
        ).fetchone()
        if duplicate_row is not None:
            anomaly_flags.append("duplicate_event")
        if normalized_type == "clock_off":
            prior_clock_on = conn.execute(
                """
                SELECT 1
                FROM worker_time_capture_events
                WHERE worker_id = ?
                  AND event_type = 'clock_on'
                  AND review_status != 'rejected'
                  AND COALESCE(effective_timestamp, captured_timestamp) <= ?
                ORDER BY COALESCE(effective_timestamp, captured_timestamp) DESC, id DESC
                LIMIT 1
                """,
                (resolved_worker_id, effective_value),
            ).fetchone()
            if prior_clock_on is None:
                anomaly_flags.append("missing_prior_clock_on")
    review_status = (
        "accepted"
        if resolved_worker_id is not None and score >= 0.9 and effective_timestamp and not anomaly_flags
        else "pending_review"
    )
    timestamp = _utc_now()
    cursor = conn.execute(
        """
        INSERT INTO worker_time_capture_events (
            call_event_id, call_session_id, call_leg_id, worker_id, worker_name_raw, employee_code_raw,
            event_type, channel, effective_timestamp, captured_timestamp,
            caller_phone, caller_phone_normalized,
            job_id, segment_id, truck_id, confidence, review_status,
            raw_payload, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            call_event_id,
            call_session_id,
            call_leg_id,
            resolved_worker_id,
            worker_name_raw,
            employee_code_raw,
            normalized_type,
            normalized_channel,
            effective_timestamp,
            captured_timestamp or timestamp,
            caller_phone,
            _normalize_phone(caller_phone),
            job_id,
            segment_id,
            truck_id,
            score,
            review_status,
            json.dumps(
                {
                    **(raw_payload or {}),
                    "anomalyFlags": anomaly_flags,
                    "duplicateOfEventId": int(duplicate_row["id"]) if duplicate_row is not None else None,
                },
                sort_keys=True,
            ),
            timestamp,
        ),
    )
    event_id = int(cursor.lastrowid)
    correlation_id = None
    if call_session_id is not None:
        correlation_id = get_call_session(conn, call_session_id)["correlationId"]
    elif call_event_id is not None:
        correlation_id = get_call_event(conn, call_event_id)["correlationId"]
    _emit_state_event(
        conn,
        source_entity_id=f"worker_time:{event_id}",
        event_type="worker_time_capture_recorded",
        payload={
            "workerTimeEventId": event_id,
            "callSessionId": call_session_id,
            "callLegId": call_leg_id,
            "workerId": resolved_worker_id,
            "eventType": normalized_type,
            "channel": normalized_channel,
            "reviewStatus": review_status,
        },
        authority_class="compiled_state" if review_status == "accepted" else "observer_capture_ref",
        correlation_id=correlation_id,
        occurred_at=timestamp,
    )
    conn.commit()
    return get_worker_time_capture_event(conn, event_id)


def get_worker_time_capture_event(conn: sqlite3.Connection, event_id: int) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT e.*, w.name AS worker_name
        FROM worker_time_capture_events e
        LEFT JOIN workers w ON w.id = e.worker_id
        WHERE e.id = ?
        """,
        (event_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Unknown worker time capture event: {event_id}")
    return _worker_time_row_to_dict(row)


def list_worker_time_capture_events(
    conn: sqlite3.Connection,
    *,
    review_status: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    ensure_call_ops_tables(conn)
    where = "WHERE e.review_status = ?" if review_status else ""
    params: list[Any] = [review_status] if review_status else []
    rows = conn.execute(
        f"""
        SELECT e.*, w.name AS worker_name
        FROM worker_time_capture_events e
        LEFT JOIN workers w ON w.id = e.worker_id
        {where}
        ORDER BY e.created_at DESC, e.id DESC
        LIMIT ?
        """,
        (*params, limit),
    ).fetchall()
    return [_worker_time_row_to_dict(row) for row in rows]


def _worker_time_row_to_dict(row: sqlite3.Row | tuple[Any, ...]) -> dict[str, Any]:
    if hasattr(row, "keys"):
        record: dict[str, Any] = {key: row[key] for key in row.keys()}
    else:
        columns = [
            "id",
            "call_event_id",
            "call_session_id",
            "call_leg_id",
            "worker_id",
            "worker_name",
            "worker_name_raw",
            "employee_code_raw",
            "event_type",
            "channel",
            "effective_timestamp",
            "captured_timestamp",
            "caller_phone",
            "job_id",
            "segment_id",
            "truck_id",
            "confidence",
            "review_status",
            "reviewer",
            "review_note",
            "raw_payload",
            "created_at",
            "reviewed_at",
        ]
        record = dict(zip(columns, row, strict=False))
    return {
        "id": int(record["id"]),
        "callEventId": record.get("call_event_id"),
        "callSessionId": record.get("call_session_id"),
        "callLegId": record.get("call_leg_id"),
        "workerId": record.get("worker_id"),
        "workerName": record.get("worker_name"),
        "workerNameRaw": record.get("worker_name_raw"),
        "employeeCodeRaw": record.get("employee_code_raw"),
        "eventType": record.get("event_type"),
        "channel": record.get("channel"),
        "effectiveTimestamp": record.get("effective_timestamp"),
        "capturedTimestamp": record.get("captured_timestamp"),
        "callerPhone": record.get("caller_phone"),
        "jobId": record.get("job_id"),
        "segmentId": record.get("segment_id"),
        "truckId": record.get("truck_id"),
        "confidence": record.get("confidence"),
        "reviewStatus": record.get("review_status"),
        "reviewer": record.get("reviewer"),
        "reviewNote": record.get("review_note"),
        "rawPayload": json.loads(record.get("raw_payload") or "{}"),
        "createdAt": record.get("created_at"),
        "reviewedAt": record.get("reviewed_at"),
    }


def decide_worker_time_capture_event(
    conn: sqlite3.Connection,
    *,
    event_id: int,
    review_status: str,
    reviewer: str | None,
    review_note: str | None = None,
    worker_id: int | None = None,
    job_id: int | None = None,
    segment_id: int | None = None,
    truck_id: str | None = None,
) -> dict[str, Any]:
    ensure_call_ops_tables(conn)
    normalized = review_status if review_status in WORKER_TIME_REVIEW_STATUSES else None
    if normalized is None or normalized == "pending_review":
        raise ValueError("Worker time review status must be accepted or rejected")
    event = get_worker_time_capture_event(conn, event_id)
    timestamp = _utc_now()
    conn.execute(
        """
        UPDATE worker_time_capture_events
        SET review_status = ?, reviewer = ?, review_note = ?, reviewed_at = ?,
            worker_id = COALESCE(?, worker_id),
            job_id = COALESCE(?, job_id),
            segment_id = COALESCE(?, segment_id),
            truck_id = COALESCE(?, truck_id)
        WHERE id = ?
        """,
        (normalized, reviewer, review_note, timestamp, worker_id, job_id, segment_id, truck_id, event_id),
    )
    _emit_state_event(
        conn,
        source_entity_id=f"worker_time:{event_id}",
        event_type="worker_time_capture_reviewed",
        payload={
            "workerTimeEventId": event_id,
            "reviewStatus": normalized,
            "reviewer": reviewer,
        },
        authority_class="compiled_state" if normalized == "accepted" else "observer_capture_ref",
        correlation_id=(
            get_call_session(conn, int(event["callSessionId"]))["correlationId"]
            if event.get("callSessionId") is not None
            else (get_call_event(conn, event["callEventId"])["correlationId"] if event["callEventId"] else None)
        ),
        occurred_at=timestamp,
    )
    conn.commit()
    return get_worker_time_capture_event(conn, event_id)


def list_state_egress_events(conn: sqlite3.Connection, *, limit: int = 100) -> list[dict[str, Any]]:
    ensure_call_ops_tables(conn)
    rows = conn.execute(
        "SELECT * FROM state_egress_events ORDER BY ingested_at DESC, id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [
        {
            "id": int(row["id"]),
            "eventId": row["event_id"],
            "sourceComponent": row["source_component"],
            "sourceEntityId": row["source_entity_id"],
            "eventType": row["event_type"],
            "idempotencyKey": row["idempotency_key"],
            "correlationId": row["correlation_id"],
            "causationId": row["causation_id"],
            "authorityClass": row["authority_class"],
            "payload": json.loads(row["payload_json"]),
            "payloadHash": row["payload_hash"],
            "occurredAt": row["occurred_at"],
            "ingestedAt": row["ingested_at"],
        }
        for row in rows
    ]


__all__ = [
    "AMBIENT_SESSION_STATUSES",
    "CALL_DIRECTIONS",
    "CALL_EVENT_KINDS",
    "CALL_LEG_KINDS",
    "CALL_ROUTING_EVENT_TYPES",
    "CALL_SOURCE_CHANNELS",
    "CALL_STATUSES",
    "CALL_TRANSCRIPT_STATUSES",
    "EXTRACTED_ACTION_STATUSES",
    "WORKER_TIME_CHANNELS",
    "WORKER_TIME_EVENT_TYPES",
    "WORKER_TIME_REVIEW_STATUSES",
    "WhisperXAdapterError",
    "add_call_note",
    "add_extracted_action",
    "create_ambient_session",
    "create_call_event",
    "create_call_leg",
    "create_call_session",
    "decide_extracted_action",
    "decide_worker_time_capture_event",
    "ensure_call_ops_tables",
    "generate_fake_ambient_transcript_artifact",
    "generate_fake_transcript_artifact",
    "get_ambient_session",
    "get_ambient_transcript_artifact",
    "get_call_event",
    "get_call_leg",
    "get_call_routing_event",
    "get_call_session",
    "get_transcript_artifact",
    "get_worker_time_capture_event",
    "list_ambient_sessions",
    "list_ambient_transcript_artifacts",
    "list_call_events",
    "list_call_legs",
    "list_call_notes",
    "list_call_routing_events",
    "list_call_sessions",
    "list_extracted_actions",
    "list_state_egress_events",
    "list_transcript_artifacts",
    "list_worker_time_capture_events",
    "log_call_routing_event",
    "poll_transcript_artifact",
    "record_ambient_transcript_artifact",
    "record_transcript_artifact",
    "record_worker_time_capture_event",
    "resolve_call_links",
    "submit_call_audio_for_transcription",
]
