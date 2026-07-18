from __future__ import annotations

import sqlite3

import pytest

from corkysoft.operations_platform import (
    JOB_STATES,
    accept_quote_as_job,
    add_quote_requirement,
    ensure_operations_platform_schema,
    list_dispatch_calendar,
    list_worker_assignments,
    propose_customer_communication,
    record_crew_acknowledgement,
    record_crew_job_status,
    transition_job_state,
)


def _connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    ensure_operations_platform_schema(conn)
    return conn


def _insert_quote(conn: sqlite3.Connection) -> int:
    cursor = conn.execute(
        """
        INSERT INTO quotes (
            created_at, quote_date, origin_input, destination_input,
            origin_resolved, destination_resolved, origin_lon, origin_lat,
            dest_lon, dest_lat, distance_km, duration_hr, cubic_m,
            pricing_model, base_subtotal, base_components, modifiers_applied,
            modifiers_total, seasonal_multiplier, seasonal_label,
            total_before_margin, margin_percent, client_display,
            manual_quote, final_quote, summary
        ) VALUES (
            '2026-07-18T00:00:00+00:00', '2026-07-21', 'Brisbane', 'Cairns',
            'Brisbane QLD', 'Cairns QLD', 153.0, -27.4,
            145.7, -16.9, 1680, 20, 40,
            'lane', 7000, '{}', '[]', 0, 1, 'standard',
            7000, 20, 'Example Customer', NULL, 8400, 'Example quote'
        )
        """
    )
    conn.commit()
    return int(cursor.lastrowid)


def test_quote_acceptance_creates_job_segment_and_copies_requirements() -> None:
    conn = _connection()
    quote_id = _insert_quote(conn)
    requirement_id = add_quote_requirement(
        conn,
        quote_id=quote_id,
        requirement_type="special_item",
        description="Upright piano",
        quantity=1,
        unit="item",
        metadata={"handling": "specialist"},
    )

    result = accept_quote_as_job(
        conn,
        quote_id=quote_id,
        actor="dispatcher@example.test",
        planned_start="2026-07-21T08:00:00+10:00",
        planned_end="2026-07-21T16:00:00+10:00",
    )

    assert result["created"] is True
    assert result["requirementCount"] == 1
    job = conn.execute("SELECT * FROM jobs WHERE id = ?", (result["jobId"],)).fetchone()
    assert job["quote_id"] == quote_id
    assert job["status"] == "planned"
    assert job["job_number"] == f"Q-{quote_id:06d}"
    segment = conn.execute(
        "SELECT * FROM job_segments WHERE job_id = ?", (result["jobId"],)
    ).fetchone()
    assert segment["status"] == "planned"
    copied = conn.execute(
        "SELECT * FROM job_requirements WHERE job_id = ?", (result["jobId"],)
    ).fetchone()
    assert copied["source_quote_requirement_id"] == requirement_id
    assert copied["description"] == "Upright piano"
    quote = conn.execute("SELECT status, accepted_job_id FROM quotes WHERE id = ?", (quote_id,)).fetchone()
    assert quote["status"] == "accepted"
    assert quote["accepted_job_id"] == result["jobId"]

    repeated = accept_quote_as_job(conn, quote_id=quote_id, actor="another")
    assert repeated == {"quoteId": quote_id, "jobId": result["jobId"], "created": False}
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1


def test_default_move_volume_requirement_is_created() -> None:
    conn = _connection()
    quote_id = _insert_quote(conn)
    result = accept_quote_as_job(conn, quote_id=quote_id, actor="dispatcher")
    requirement = conn.execute(
        "SELECT requirement_type, quantity, unit FROM job_requirements WHERE job_id = ?",
        (result["jobId"],),
    ).fetchone()
    assert tuple(requirement) == ("move_volume", 40.0, "m3")


def test_canonical_job_state_progression_and_invalid_jump() -> None:
    conn = _connection()
    job_id = accept_quote_as_job(conn, quote_id=_insert_quote(conn), actor="dispatcher")["jobId"]
    expected = [state for state in JOB_STATES if state not in {"draft", "exception"}]
    current = "planned"
    for state in expected[1:]:
        result = transition_job_state(
            conn, job_id=job_id, new_state=state, actor="operator"
        )
        assert result["previousState"] == current
        current = state
    assert current == "completed"
    assert conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()[0] == "completed"
    with pytest.raises(ValueError, match="Invalid job state transition"):
        transition_job_state(conn, job_id=job_id, new_state="planned", actor="operator")


def test_dispatch_calendar_exposes_assignments_and_readiness_filters() -> None:
    conn = _connection()
    job = accept_quote_as_job(
        conn,
        quote_id=_insert_quote(conn),
        actor="dispatcher",
        planned_start="2026-07-21T08:00:00+10:00",
        planned_end="2026-07-21T16:00:00+10:00",
    )
    segment_id = job["segmentId"]
    worker_id = conn.execute(
        "INSERT INTO workers (employee_code, name, role, active) VALUES ('W1', 'Crew One', 'removalist', 1)"
    ).lastrowid
    conn.execute(
        "INSERT INTO trucks (truck_id, name, capacity_m3, active) VALUES ('T1', 'Truck One', 60, 1)"
    )
    conn.execute(
        "INSERT INTO job_segment_workers (segment_id, worker_id, start_time, end_time) VALUES (?, ?, '', '')",
        (segment_id, worker_id),
    )
    conn.execute(
        "INSERT INTO job_segment_vehicles (segment_id, truck_id, requirement_met) VALUES (?, 'T1', 1)",
        (segment_id,),
    )
    conn.commit()

    rows = list_dispatch_calendar(
        conn,
        start="2026-07-21T00:00:00+10:00",
        end="2026-07-22T00:00:00+10:00",
        statuses=["planned"],
        truck_ids=["T1"],
        worker_ids=[int(worker_id)],
        depot="Brisbane",
    )
    assert len(rows) == 1
    assert rows[0]["truckIds"] == ["T1"]
    assert rows[0]["workerNames"] == ["Crew One"]
    assert rows[0]["ready"] is True


def test_customer_communication_is_proposed_and_public_safe() -> None:
    conn = _connection()
    job_id = accept_quote_as_job(conn, quote_id=_insert_quote(conn), actor="dispatcher")["jobId"]
    event_id = propose_customer_communication(
        conn,
        event_type="booking_confirmed",
        proposed_by="dispatcher",
        job_id=job_id,
        recipient="customer@example.test",
        channel="email",
        public_payload={
            "customer_name": "Example Customer",
            "job_number": "Q-000001",
            "status": "planned",
            "message": "Your booking is confirmed.",
        },
    )
    row = conn.execute(
        "SELECT status, channel, sent_at FROM customer_communication_events WHERE id = ?",
        (event_id,),
    ).fetchone()
    assert tuple(row) == ("proposed", "email", None)
    with pytest.raises(ValueError, match="Unsafe customer payload fields"):
        propose_customer_communication(
            conn,
            event_type="delayed",
            proposed_by="dispatcher",
            job_id=job_id,
            public_payload={"internal_cost_total": 1000},
        )


def test_crew_acknowledges_and_advances_assigned_work_or_flags_exception() -> None:
    conn = _connection()
    job = accept_quote_as_job(conn, quote_id=_insert_quote(conn), actor="dispatcher")
    worker_id = int(
        conn.execute(
            "INSERT INTO workers (employee_code, name, role, active) VALUES ('W1', 'Crew One', 'removalist', 1)"
        ).lastrowid
    )
    conn.execute(
        "INSERT INTO job_segment_workers (segment_id, worker_id, start_time, end_time) VALUES (?, ?, '', '')",
        (job["segmentId"], worker_id),
    )
    conn.commit()
    transition_job_state(conn, job_id=job["jobId"], new_state="assigned", actor="dispatcher")

    assignments = list_worker_assignments(conn, worker_id=worker_id)
    assert len(assignments) == 1
    record_crew_acknowledgement(
        conn,
        job_id=job["jobId"],
        segment_id=job["segmentId"],
        worker_id=worker_id,
        note="Received",
    )
    assert conn.execute("SELECT status FROM jobs WHERE id = ?", (job["jobId"],)).fetchone()[0] == "acknowledged"

    record_crew_job_status(
        conn,
        job_id=job["jobId"],
        worker_id=worker_id,
        new_state="exception",
        note="Access blocked",
    )
    assert conn.execute("SELECT status FROM jobs WHERE id = ?", (job["jobId"],)).fetchone()[0] == "exception"
    exception = conn.execute(
        "SELECT acknowledgement_type, status, note FROM crew_acknowledgements WHERE job_id = ? ORDER BY id DESC LIMIT 1",
        (job["jobId"],),
    ).fetchone()
    assert tuple(exception) == ("exception", "flagged", "Access blocked")
