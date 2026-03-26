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


from corkysoft.call_ops_actions import (
    add_call_note,
    add_extracted_action,
    decide_extracted_action,
    get_extracted_action,
    list_call_notes,
    list_extracted_actions,
    resolve_call_links,
)
from corkysoft.call_ops_core import (
    create_ambient_session,
    create_call_event,
    create_call_leg,
    create_call_session,
    get_ambient_session,
    get_call_event,
    get_call_leg,
    get_call_routing_event,
    get_call_session,
    list_ambient_sessions,
    list_call_events,
    list_call_legs,
    list_call_routing_events,
    list_call_sessions,
    log_call_routing_event,
)
from corkysoft.call_ops_transcripts import (
    generate_fake_ambient_transcript_artifact,
    generate_fake_transcript_artifact,
    get_ambient_transcript_artifact,
    get_transcript_artifact,
    list_ambient_transcript_artifacts,
    list_transcript_artifacts,
    poll_transcript_artifact,
    record_ambient_transcript_artifact,
    record_transcript_artifact,
    submit_call_audio_for_transcription,
)
from corkysoft.call_ops_worker_time import (
    decide_worker_time_capture_event,
    get_worker_time_capture_event,
    list_worker_time_capture_events,
    record_worker_time_capture_event,
)


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
