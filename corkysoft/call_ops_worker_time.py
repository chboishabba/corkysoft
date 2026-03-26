from __future__ import annotations

import json
import sqlite3
from typing import Any


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
    from .call_ops import (
        WORKER_TIME_CHANNELS,
        WORKER_TIME_EVENT_TYPES,
        _emit_state_event,
        _normalize_phone,
        _resolve_worker_by_phone,
        _utc_now,
        ensure_call_ops_tables,
        get_call_event,
        get_call_leg,
        get_call_session,
    )

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
    from .call_ops import ensure_call_ops_tables

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
    from .call_ops import (
        WORKER_TIME_REVIEW_STATUSES,
        _emit_state_event,
        _utc_now,
        ensure_call_ops_tables,
        get_call_event,
        get_call_session,
    )

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
