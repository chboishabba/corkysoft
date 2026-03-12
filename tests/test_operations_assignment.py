from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.db import ensure_dashboard_tables, upsert_truck, upsert_vehicle_details, upsert_worker
from analytics.operations_assignment import (
    assign_segment_resources,
    ensure_segment,
    evaluate_segment_readiness,
    list_operational_conflicts,
    list_segments_for_truck,
    list_segments_for_worker,
    list_segment_readiness,
    list_truck_assignment_summary,
    list_worker_assignment_summary,
)


def _job(conn: sqlite3.Connection, client: str, origin: str, destination: str) -> int:
    cursor = conn.execute(
        """
        INSERT INTO jobs (
            client,
            origin,
            destination,
            updated_at
        ) VALUES (?, ?, ?, ?)
        """,
        (client, origin, destination, "2026-03-12T00:00:00+00:00"),
    )
    return int(cursor.lastrowid)


def test_list_segment_readiness_backfills_default_segments() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)

    job_id = _job(conn, "Kent", "Brisbane", "Townsville")
    rows = list_segment_readiness(conn, job_id=job_id)

    assert len(rows) == 1
    row = rows[0]
    assert row["jobId"] == job_id
    assert row["segmentSequence"] == 1
    assert row["assignmentStatus"] == "draft"
    assert "segment:no_truck_assigned" in row["warningFlags"]
    assert "segment:no_worker_assigned" in row["warningFlags"]


def test_assign_segment_resources_requires_override_for_overlaps() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)

    upsert_truck(conn, truck_id="TRK-1", name="Truck 1")
    upsert_vehicle_details(
        conn,
        truck_id="TRK-1",
        rego="TRK-1",
        rego_expiry="2099-12-31",
        coi_due="2099-12-31",
        next_service="2099-12-31",
        daily_check_complete=True,
        source_system="google_sheets",
        source_sheet="FLEET",
        source_imported_at="2026-03-12T00:00:00+00:00",
    )
    worker = upsert_worker(
        conn,
        name="Alex Loader",
        source_system="google_sheets",
        source_sheet="STAFF",
        source_imported_at="2026-03-12T00:00:00+00:00",
    )

    job_one = _job(conn, "Client A", "Brisbane", "Toowoomba")
    job_two = _job(conn, "Client B", "Gold Coast", "Lismore")

    segment_one = ensure_segment(
        conn,
        job_id=job_one,
        segment_sequence=1,
        planned_start="2026-03-12T08:00:00+00:00",
        planned_end="2026-03-12T12:00:00+00:00",
    )
    segment_two = ensure_segment(
        conn,
        job_id=job_two,
        segment_sequence=1,
        planned_start="2026-03-12T09:00:00+00:00",
        planned_end="2026-03-12T11:00:00+00:00",
    )

    first = assign_segment_resources(
        conn,
        segment_id=int(segment_one["id"]),
        truck_ids=["TRK-1"],
        worker_assignments=[{"workerId": int(worker["id"])}],
    )
    assert first["assignmentStatus"] == "planned"

    with pytest.raises(ValueError, match="requires override"):
        assign_segment_resources(
            conn,
            segment_id=int(segment_two["id"]),
            truck_ids=["TRK-1"],
            worker_assignments=[{"workerId": int(worker["id"])}],
        )

    second = assign_segment_resources(
        conn,
        segment_id=int(segment_two["id"]),
        truck_ids=["TRK-1"],
        worker_assignments=[{"workerId": int(worker["id"])}],
        override=True,
        override_reason_code="manual_ops_override",
        override_note="Keep the same crew on the lane",
    )
    assert second["assignmentStatus"] == "overridden"
    assert any(flag.startswith("truck_conflict:TRK-1") for flag in second["overrideableFlags"])
    assert any(flag.startswith("worker_conflict:") for flag in second["overrideableFlags"])

    conflicts = list_operational_conflicts(conn)
    assert any(item["segmentId"] == int(segment_one["id"]) for item in conflicts)
    assert any(item["segmentId"] == int(segment_two["id"]) for item in conflicts)


def test_evaluate_segment_readiness_blocks_expired_rego() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)

    upsert_truck(conn, truck_id="TRK-2", name="Truck 2")
    upsert_vehicle_details(
        conn,
        truck_id="TRK-2",
        rego="TRK-2",
        rego_expiry="2000-01-01",
        coi_due="2099-12-31",
        next_service="2099-12-31",
        daily_check_complete=True,
    )
    job_id = _job(conn, "Client C", "Sydney", "Melbourne")
    segment = ensure_segment(
        conn,
        job_id=job_id,
        segment_sequence=1,
        planned_start="2026-03-12T08:00:00+00:00",
        planned_end="2026-03-12T10:00:00+00:00",
    )

    readiness = evaluate_segment_readiness(
        conn,
        segment_id=int(segment["id"]),
        truck_ids=["TRK-2"],
        worker_assignments=[],
    )

    assert readiness["assignmentStatus"] == "blocked"
    assert "truck:TRK-2:rego_expired" in readiness["blockingFlags"]


def test_assignment_summary_helpers_reflect_segment_plans() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)

    upsert_truck(conn, truck_id="TRK-3", name="Truck 3")
    upsert_vehicle_details(
        conn,
        truck_id="TRK-3",
        rego="TRK-3",
        rego_expiry="2099-12-31",
        coi_due="2099-12-31",
        next_service="2099-12-31",
        daily_check_complete=True,
        source_system="google_sheets",
        source_sheet="FLEET",
        source_imported_at="2026-03-12T00:00:00+00:00",
    )
    worker = upsert_worker(
        conn,
        name="Jamie Planner",
        source_system="google_sheets",
        source_sheet="STAFF",
        source_imported_at="2026-03-12T00:00:00+00:00",
    )
    job_id = _job(conn, "Client D", "Adelaide", "Perth")
    segment = ensure_segment(
        conn,
        job_id=job_id,
        segment_sequence=1,
        planned_start="2026-03-12T13:00:00+00:00",
        planned_end="2026-03-12T20:00:00+00:00",
    )
    assign_segment_resources(
        conn,
        segment_id=int(segment["id"]),
        truck_ids=["TRK-3"],
        worker_assignments=[{"workerId": int(worker["id"])}],
    )

    worker_summary = list_worker_assignment_summary(conn)
    assert worker_summary[int(worker["id"])]["plannedSegmentCount"] == 1
    assert worker_summary[int(worker["id"])]["plannedJobCount"] == 1
    assert worker_summary[int(worker["id"])]["plannedTrucks"] == ["TRK-3"]

    truck_summary = list_truck_assignment_summary(conn)
    assert truck_summary["TRK-3"]["plannedSegmentCount"] == 1
    assert truck_summary["TRK-3"]["plannedJobCount"] == 1
    assert truck_summary["TRK-3"]["plannedWorkers"] == ["Jamie Planner"]

    worker_segments = list_segments_for_worker(conn, worker_id=int(worker["id"]))
    assert len(worker_segments) == 1
    assert worker_segments[0]["jobId"] == job_id

    truck_segments = list_segments_for_truck(conn, truck_id="TRK-3")
    assert len(truck_segments) == 1
    assert truck_segments[0]["segmentId"] == int(segment["id"])
