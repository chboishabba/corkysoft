from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.db import (
    create_worker_absence_record,
    ensure_dashboard_tables,
    upsert_driver_shift,
    upsert_truck,
    upsert_worker,
)
from analytics.labor_analytics import build_payroll_labor_analytics
from analytics.operations_assignment import assign_segment_resources, ensure_segment
from corkysoft.call_ops import record_worker_time_capture_event


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


def test_build_payroll_labor_analytics_rolls_up_forecast_overtime_absence_and_confidence() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)

    upsert_truck(conn, truck_id="TRK-LAB-1", name="Labor Truck")
    worker = upsert_worker(conn, name="Labor Worker", phone="0400111222")
    worker_id = int(worker["id"])
    job_id = _job(conn, "Payroll Client", "Brisbane", "Ipswich")
    segment = ensure_segment(
        conn,
        job_id=job_id,
        segment_sequence=1,
        planned_start="2026-03-14T08:00:00+10:00",
        planned_end="2026-03-14T12:00:00+10:00",
        from_location="Brisbane",
        to_location="Ipswich",
    )
    assign_segment_resources(
        conn,
        segment_id=int(segment["id"]),
        truck_ids=["TRK-LAB-1"],
        worker_assignments=[{"workerId": worker_id}],
    )
    upsert_driver_shift(
        conn,
        shift_date="2026-03-14",
        truck_id="TRK-LAB-1",
        worker_name="Labor Worker",
        shift_window_start="08:00",
        shift_window_end="12:00",
        hours=10.0,
        hourly_rate=42.0,
        job_id=job_id,
        source="VEHICLE_DRIVER",
        imported_at="2026-03-14T00:00:00+10:00",
    )
    record_worker_time_capture_event(
        conn,
        event_type="clock_on",
        channel="manual_supervisor",
        worker_id=worker_id,
        worker_name_raw="Labor Worker",
        effective_timestamp="2026-03-14T08:30:00+10:00",
        truck_id="TRK-LAB-1",
        job_id=job_id,
        segment_id=int(segment["id"]),
        confidence=0.95,
    )
    record_worker_time_capture_event(
        conn,
        event_type="clock_on",
        channel="voice_call",
        worker_name_raw="Unknown Worker",
        effective_timestamp="2026-03-14T09:15:00+10:00",
        confidence=0.2,
        raw_payload={"anomalyFlags": ["duplicate_event"]},
    )
    create_worker_absence_record(
        conn,
        worker_id=worker_id,
        start_date="2026-03-15",
        end_date="2026-03-15",
        absence_type="sick",
        status="confirmed",
        hours_per_day=8.0,
        note="Recorded illness",
        source="manager_manual",
        recorded_by="ops-manager",
    )

    payload = build_payroll_labor_analytics(
        conn,
        start_date="2026-03-14",
        end_date="2026-03-15",
        overtime_daily_hours=8.0,
    )

    assert payload["summary"]["plannedExposure"] == pytest.approx(168.0)
    assert payload["summary"]["importedCost"] == pytest.approx(420.0)
    assert payload["summary"]["reviewedActualCost"] == pytest.approx(420.0)
    assert payload["summary"]["absenceModelStatus"] == "basic_recorded"
    assert payload["summary"]["absenceRecordCount"] == 1
    assert payload["summary"]["confirmedAbsenceCount"] == 1

    forecast_row = next(
        row for row in payload["payForecastRows"] if row["workerName"] == "Labor Worker"
    )
    assert forecast_row["plannedHours"] == pytest.approx(4.0)
    assert forecast_row["importedHours"] == pytest.approx(10.0)
    assert forecast_row["acceptedEventCount"] == 1
    assert forecast_row["absenceDays"] == pytest.approx(1.0)
    assert forecast_row["absenceHours"] == pytest.approx(8.0)

    overtime_row = next(
        row for row in payload["overtimeRows"] if row["workerName"] == "Labor Worker"
    )
    assert overtime_row["overtimeHours"] == pytest.approx(2.0)

    assert payload["confidence"]["pendingReviewCount"] == 1
    assert payload["confidence"]["acceptedEventCount"] == 1
    assert payload["confidence"]["acceptedUnmatchedCount"] == 0
    assert payload["absenceSummary"]["sickDays"] == pytest.approx(1.0)
    assert payload["exportReadyLaborSummaries"][0]["workerName"] == "Labor Worker"

    worker_driver = next(
        row for row in payload["laborCostDrivers"]["worker"] if row["dimensionValue"] == "Labor Worker"
    )
    assert worker_driver["totalCost"] == pytest.approx(420.0)
