from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hashlib
import json

from analytics.db import ensure_dashboard_tables, upsert_worker
from corkysoft.call_ops import (
    _emit_state_event,
    add_call_note,
    add_extracted_action,
    create_ambient_session,
    create_call_event,
    create_call_leg,
    create_call_session,
    decide_extracted_action,
    decide_worker_time_capture_event,
    ensure_call_ops_tables,
    generate_fake_ambient_transcript_artifact,
    generate_fake_transcript_artifact,
    get_ambient_session,
    get_call_event,
    get_call_session,
    list_ambient_transcript_artifacts,
    list_extracted_actions,
    list_call_legs,
    list_call_notes,
    list_call_routing_events,
    list_call_events,
    list_state_egress_events,
    poll_transcript_artifact,
    record_worker_time_capture_event,
    resolve_call_links,
    submit_call_audio_for_transcription,
)


def _job(conn: sqlite3.Connection) -> int:
    return int(
        conn.execute(
            "INSERT INTO jobs (client, origin, destination, updated_at) VALUES (?, ?, ?, ?)",
            ("Client A", "Brisbane", "Cairns", "2026-03-13T00:00:00+00:00"),
        ).lastrowid
    )


def test_call_event_auto_creates_client_and_notes_actions_flow(monkeypatch) -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)
    job_id = _job(conn)

    call = create_call_event(
        conn,
        event_kind="client_call",
        direction="inbound",
        caller_phone="0400 111 222",
        job_id=job_id,
        title="Client called about access",
    )
    assert call["clientId"] is not None
    assert call["callerPhoneNormalized"] == "0400111222"

    note = add_call_note(conn, call_event_id=call["id"], author="ops-1", note_text="Client says access is via rear lane.")
    assert note["authoritative"] is True

    action = add_extracted_action(
        conn,
        call_event_id=call["id"],
        action_text="Update access note and notify crew.",
        source_engine="statibaker",
    )
    accepted = decide_extracted_action(
        conn,
        action_id=action["id"],
        status="accepted",
        decided_by="ops-1",
    )
    assert accepted["status"] == "accepted"

    egress = list_state_egress_events(conn)
    event_types = {row["eventType"] for row in egress}
    assert "call_event_created" in event_types
    assert "call_note_added" in event_types
    assert "extracted_action_decided" in event_types


def test_worker_time_capture_review_flow() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)
    worker = upsert_worker(conn, name="Riley Worker", phone="0400123123")

    event = record_worker_time_capture_event(
        conn,
        event_type="clock_on",
        channel="voice_call",
        caller_phone="0400 123 123",
        effective_timestamp="2026-03-13T06:30:00+10:00",
        confidence=0.4,
    )
    assert event["reviewStatus"] == "pending_review"

    reviewed = decide_worker_time_capture_event(
        conn,
        event_id=event["id"],
        review_status="accepted",
        reviewer="labor-1",
        worker_id=int(worker["id"]),
    )
    assert reviewed["reviewStatus"] == "accepted"
    assert reviewed["workerId"] == int(worker["id"])


def test_resolve_call_links_updates_existing_call() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)
    job_id = _job(conn)
    call = create_call_event(
        conn,
        event_kind="ops_call",
        direction="internal",
        status="needs_review",
        title="Manager call",
    )
    updated = resolve_call_links(
        conn,
        call_event_id=call["id"],
        actor="dispatcher-1",
        job_id=job_id,
        resolution_note="Linked after ops review",
    )
    assert updated["jobId"] == job_id


def test_whisperx_submission_and_poll_are_adapter_driven(monkeypatch) -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)
    call = create_call_event(conn, event_kind="ops_call", direction="internal")

    def fake_submit(**kwargs):
        return {"identifier": "task-123", "status": "QUEUED", "message": "queued"}

    def fake_fetch(*, service_key: str, identifier: str):
        assert service_key == "ops"
        assert identifier == "task-123"
        return {
            "identifier": identifier,
            "status": "COMPLETED",
            "result": [
                {"text": "Manager says switch to dock two", "start": 0.0, "end": 1.2, "speaker": "SPEAKER_00", "confidence": 0.91},
                {"text": "Crew confirmed", "start": 1.3, "end": 1.8, "speaker": "SPEAKER_01", "confidence": 0.84},
            ],
        }

    monkeypatch.setattr("corkysoft.call_ops.submit_transcription", fake_submit)
    monkeypatch.setattr("corkysoft.call_ops.fetch_task_status", fake_fetch)

    artifact = submit_call_audio_for_transcription(
        conn,
        call_event_id=call["id"],
        service_key="ops",
        file_bytes=b"audio-bytes",
        filename="call.wav",
    )
    assert artifact["externalTaskId"] == "task-123"
    assert artifact["status"] == "queued"

    completed = poll_transcript_artifact(conn, artifact_id=artifact["id"])
    assert completed["status"] == "completed"
    assert "Manager says switch to dock two" in (completed["transcriptText"] or "")

    refreshed = get_call_event(conn, call["id"])
    assert refreshed["processedAt"] is not None
    assert list_call_events(conn, limit=5)[0]["latestTranscriptStatus"] == "completed"


def test_generate_fake_transcript_artifact_produces_completed_text() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)
    job_id = _job(conn)
    call = create_call_event(
        conn,
        event_kind="manager_call",
        direction="internal",
        job_id=job_id,
        title="Manager redirect",
    )
    artifact = generate_fake_transcript_artifact(
        conn,
        call_event_id=call["id"],
        scenario="Manager says the access path has changed.",
        operator_goal="Update the crew and client immediately.",
    )
    assert artifact["status"] == "completed"
    assert artifact["isFinal"] is True
    assert "Manager says the access path has changed." in (artifact["transcriptText"] or "")
    assert "Update the crew and client immediately." in (artifact["transcriptText"] or "")


def test_call_session_tracks_legs_and_routing_chain() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)
    session = create_call_session(
        conn,
        event_kind="client_call",
        direction="inbound",
        title="Client to operator",
        caller_phone="0400555000",
        operator_id="ops-1",
        initial_destination_kind="operator",
        initial_destination_label="desk-1",
    )
    assert session["rootCallEventId"] is not None

    legs = list_call_legs(conn, call_session_id=session["id"])
    assert len(legs) == 1
    consult = create_call_leg(
        conn,
        call_session_id=session["id"],
        leg_kind="consult",
        direction="internal",
        status="active",
        destination_kind="manager",
        destination_label="boss",
        operator_id="mgr-1",
        answered_at="2026-03-13T01:00:00+00:00",
    )
    assert consult["legKind"] == "consult"
    routing = list_call_routing_events(conn, call_session_id=session["id"])
    event_types = {row["eventType"] for row in routing}
    assert "call_received" in event_types
    assert "call_routed" in event_types
    assert "call_answered" in event_types
    fetched = get_call_session(conn, session["id"])
    assert fetched["legCount"] >= 2


def test_state_egress_event_payload_contract() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)
    job_id = _job(conn)

    call = create_call_event(
        conn,
        event_kind="client_call",
        direction="inbound",
        caller_phone="0400 111 222",
        job_id=job_id,
        title="Call for egress contract",
    )

    egress = list_state_egress_events(conn, limit=5)
    call_event = next(
        row for row in egress if row["eventType"] == "call_event_created"
    )

    payload = call_event["payload"]
    assert payload["callEventId"] == call["id"]
    assert payload["jobId"] == job_id
    assert call_event["authorityClass"] == "compiled_state"

    expected_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    assert call_event["payloadHash"] == expected_hash
    expected_idempotency = f"corkysoft:call_event_created:{call_event['sourceEntityId']}:{expected_hash}"
    assert call_event["idempotencyKey"] == expected_idempotency


def test_call_note_observer_authority_class() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)
    job_id = _job(conn)

    call = create_call_event(
        conn,
        event_kind="ops_call",
        direction="internal",
        title="Note authority",
        job_id=job_id,
    )

    add_call_note(
        conn,
        call_event_id=call["id"],
        author="ops-1",
        note_text="Observer note",
        authoritative=False,
    )

    note_event = next(
        row for row in list_state_egress_events(conn, limit=5) if row["eventType"] == "call_note_added"
    )
    assert note_event["authorityClass"] == "observer_capture_ref"
    assert note_event["payload"]["authoritative"] is False


def test_state_egress_idempotency_key_repeatable() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_call_ops_tables(conn)

    payload = {"test": "repeatable"}
    source = "test:ipc"
    event_type = "idempotent_test"

    _emit_state_event(
        conn,
        source_entity_id=source,
        event_type=event_type,
        payload=payload,
        authority_class="observer_capture_ref",
    )
    _emit_state_event(
        conn,
        source_entity_id=source,
        event_type=event_type,
        payload=payload,
        authority_class="observer_capture_ref",
    )

    rows = [row for row in list_state_egress_events(conn, limit=5) if row["eventType"] == event_type]
    assert len(rows) == 2
    assert rows[0]["idempotencyKey"] == rows[1]["idempotencyKey"]
    assert rows[0]["ingestedAt"] >= rows[1]["ingestedAt"]


def test_call_session_conflicting_manager_update_stays_traceable() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)
    job_id = _job(conn)
    session = create_call_session(
        conn,
        event_kind="client_call",
        direction="inbound",
        title="Access issue escalation",
        caller_phone="0400444555",
        job_id=job_id,
        operator_id="ops-1",
        initial_destination_label="desk-1",
    )
    leg = list_call_legs(conn, call_session_id=session["id"])[0]
    first_artifact = generate_fake_transcript_artifact(
        conn,
        call_leg_id=int(leg["id"]),
        scenario="Manager says use dock two.",
    )
    first_action = add_extracted_action(
        conn,
        call_event_id=int(session["rootCallEventId"]),
        transcript_artifact_id=int(first_artifact["id"]),
        action_text="Use dock two.",
        source_engine="statibaker",
    )
    decide_extracted_action(conn, action_id=int(first_action["id"]), status="accepted", decided_by="ops-1")

    consult_leg = create_call_leg(
        conn,
        call_session_id=int(session["id"]),
        leg_kind="consult",
        direction="internal",
        status="active",
        destination_kind="manager",
        destination_label="boss",
        operator_id="mgr-1",
        answered_at="2026-03-13T01:00:00+00:00",
    )
    second_artifact = generate_fake_transcript_artifact(
        conn,
        call_leg_id=int(consult_leg["id"]),
        scenario="Manager corrects prior advice and says use rear lane instead.",
    )
    second_action = add_extracted_action(
        conn,
        call_event_id=int(session["rootCallEventId"]),
        transcript_artifact_id=int(second_artifact["id"]),
        action_text="Use rear lane instead of dock two.",
        source_engine="statibaker",
    )
    accepted = decide_extracted_action(conn, action_id=int(second_action["id"]), status="accepted", decided_by="ops-1")

    assert accepted["status"] == "accepted"
    actions = list_extracted_actions(conn, call_event_id=int(session["rootCallEventId"]))
    assert len([row for row in actions if row["status"] == "accepted"]) == 2
    assert any("rear lane" in row["actionText"] for row in actions)
    routing = list_call_routing_events(conn, call_session_id=int(session["id"]))
    assert any(row["eventType"] == "call_answered" and row["callLegId"] == int(consult_leg["id"]) for row in routing)


def test_call_link_correction_keeps_audit_history() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)
    wrong_job = _job(conn)
    right_job = _job(conn)
    call = create_call_event(conn, event_kind="ops_call", direction="internal", title="Wrong job first")
    resolve_call_links(conn, call_event_id=int(call["id"]), actor="ops-1", job_id=wrong_job, resolution_note="Initial link")
    corrected = resolve_call_links(conn, call_event_id=int(call["id"]), actor="ops-1", job_id=right_job, resolution_note="Corrected link")
    assert corrected["jobId"] == right_job
    history = conn.execute(
        "SELECT job_id, resolution_note FROM call_link_resolutions WHERE call_event_id = ? ORDER BY id ASC",
        (int(call["id"]),),
    ).fetchall()
    assert [int(row["job_id"]) for row in history] == [wrong_job, right_job]
    assert history[1]["resolution_note"] == "Corrected link"


def test_ambient_session_fake_transcript_flow() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)
    ambient = create_ambient_session(
        conn,
        title="Office planning",
        source_location="Brisbane office",
        team_label="Ops desk",
        job_id=_job(conn),
    )
    artifact = generate_fake_ambient_transcript_artifact(
        conn,
        ambient_session_id=ambient["id"],
        scenario="Manager and operator discuss updated site conditions.",
        operator_goal="Send revised instructions to the crew.",
    )
    assert artifact["status"] == "completed"
    assert "updated site conditions" in (artifact["transcriptText"] or "")
    listed = list_ambient_transcript_artifacts(conn, ambient_session_id=ambient["id"])
    assert len(listed) == 1
    assert get_ambient_session(conn, ambient["id"])["title"] == "Office planning"


def test_duplicate_worker_time_and_missing_clock_on_are_flagged_for_review() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)
    worker = upsert_worker(conn, name="Riley Worker", phone="0400123123")

    first = record_worker_time_capture_event(
        conn,
        worker_id=int(worker["id"]),
        event_type="clock_on",
        channel="voice_call",
        effective_timestamp="2026-03-13T06:30:00+10:00",
        confidence=0.95,
    )
    assert first["reviewStatus"] == "accepted"

    duplicate = record_worker_time_capture_event(
        conn,
        worker_id=int(worker["id"]),
        event_type="clock_on",
        channel="voice_call",
        effective_timestamp="2026-03-13T06:30:00+10:00",
        confidence=0.95,
    )
    assert duplicate["reviewStatus"] == "pending_review"
    assert "duplicate_event" in duplicate["rawPayload"]["anomalyFlags"]
    assert duplicate["rawPayload"]["duplicateOfEventId"] == first["id"]

    off_without_on = record_worker_time_capture_event(
        conn,
        worker_id=int(worker["id"]),
        event_type="clock_off",
        channel="voice_call",
        effective_timestamp="2026-03-12T05:30:00+10:00",
        confidence=0.95,
    )
    assert off_without_on["reviewStatus"] == "pending_review"
    assert "missing_prior_clock_on" in off_without_on["rawPayload"]["anomalyFlags"]
