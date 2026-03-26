from __future__ import annotations

import sqlite3
import uuid
from typing import Any


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
        "clientName": (row["client_name"] or None),
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
        "callSessionId": row["call_session_id"],
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
        "callSessionId": row["call_session_id"],
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
    from .call_ops import CALL_ROUTING_EVENT_TYPES, _emit_state_event, _utc_now, ensure_call_ops_tables

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
    from .call_ops import (
        CALL_DIRECTIONS,
        CALL_LEG_KINDS,
        CALL_SOURCE_CHANNELS,
        CALL_STATUSES,
        _normalize_phone,
        _utc_now,
        ensure_call_ops_tables,
    )

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
    from .call_ops import (
        CALL_DIRECTIONS,
        CALL_EVENT_KINDS,
        CALL_SOURCE_CHANNELS,
        CALL_STATUSES,
        _emit_state_event,
        _insert_call_event_record,
        _normalize_phone,
        _utc_now,
        ensure_call_ops_tables,
    )

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


def create_call_event(conn: sqlite3.Connection, **kwargs: Any) -> dict[str, Any]:
    session = create_call_session(conn, **kwargs)
    return get_call_event(conn, int(session["rootCallEventId"]))


def list_call_sessions(
    conn: sqlite3.Connection,
    *,
    limit: int = 100,
    status: str | None = None,
    event_kind: str | None = None,
) -> list[dict[str, Any]]:
    from .call_ops import ensure_call_ops_tables

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
    from .call_ops import ensure_call_ops_tables

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
    from .call_ops import ensure_call_ops_tables

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
    from .call_ops import ensure_call_ops_tables

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
    from .call_ops import ensure_call_ops_tables

    ensure_call_ops_tables(conn)
    rows = conn.execute(
        "SELECT * FROM call_routing_events WHERE call_session_id = ? ORDER BY created_at ASC, id ASC",
        (call_session_id,),
    ).fetchall()
    return [_routing_event_row_to_dict(row) for row in rows]


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


def list_call_events(
    conn: sqlite3.Connection,
    *,
    limit: int = 100,
    status: str | None = None,
    event_kind: str | None = None,
    unresolved_only: bool = False,
) -> list[dict[str, Any]]:
    from .call_ops import ensure_call_ops_tables

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
    from .call_ops import ensure_call_ops_tables

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
    from .call_ops import AMBIENT_SESSION_STATUSES, _emit_state_event, _utc_now, ensure_call_ops_tables

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
    from .call_ops import ensure_call_ops_tables

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
    from .call_ops import ensure_call_ops_tables

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
