from __future__ import annotations

import json
import sqlite3
from typing import Any, Sequence

from .call_ops import (
    CALL_TRANSCRIPT_STATUSES,
    _emit_state_event,
    _utc_now,
    ensure_call_ops_tables,
    get_ambient_session,
    get_call_event,
    get_call_leg,
)


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
    from . import call_ops as call_ops_module

    payload = call_ops_module.submit_transcription(
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
    from . import call_ops as call_ops_module

    payload = call_ops_module.fetch_task_status(
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
