from __future__ import annotations

import sqlite3
from typing import Any


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
    from .call_ops import _emit_state_event, _utc_now, ensure_call_ops_tables, get_ambient_session, get_call_event

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
    from .call_ops import ensure_call_ops_tables

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
    from .call_ops import _emit_state_event, _utc_now, ensure_call_ops_tables, get_ambient_session, get_call_event

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
    from .call_ops import (
        EXTRACTED_ACTION_STATUSES,
        _emit_state_event,
        _utc_now,
        ensure_call_ops_tables,
        get_ambient_session,
        get_call_event,
    )

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
    from .call_ops import _emit_state_event, _utc_now, ensure_call_ops_tables, get_ambient_session, get_call_event

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
