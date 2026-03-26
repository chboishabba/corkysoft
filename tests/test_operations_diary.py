from __future__ import annotations

import sqlite3

from analytics.db import ensure_dashboard_tables, upsert_driver_shift, upsert_truck, upsert_worker
from analytics.operations_assignment import assign_segment_resources, ensure_segment
from analytics.operations_diary import (
    build_job_usage_details,
    build_operations_diary,
    build_reconciliation_exposure_summary,
    delete_operations_diary_task,
    export_operations_diary_observer_events,
    list_observer_outbox_events,
    list_operations_diary_tasks,
    upsert_customer_invoice_review,
    upsert_operations_diary_task,
    upsert_subcontractor_bill_review,
)


def _seed_conn() -> tuple[sqlite3.Connection, dict[str, int]]:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    ensure_dashboard_tables(conn)
    upsert_truck(conn, truck_id="TRK-1", name="Truck 1", capacity_m3=50.0)
    worker = upsert_worker(conn, name="Alex Planner")
    job1 = conn.execute(
        """
        INSERT INTO jobs (
            client, origin, destination, origin_resolved, destination_resolved, job_date,
            distance_km, volume_m3, origin_lat, origin_lon, dest_lat, dest_lon, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "Kent",
            "Brisbane",
            "Cairns",
            "Brisbane",
            "Cairns",
            "2026-03-20",
            1700.0,
            36.0,
            -27.47,
            153.02,
            -16.92,
            145.77,
            "2026-03-12T00:00:00+00:00",
        ),
    ).lastrowid
    job2 = conn.execute(
        """
        INSERT INTO jobs (
            client, origin, destination, origin_resolved, destination_resolved, job_date,
            distance_km, volume_m3, origin_lat, origin_lon, dest_lat, dest_lon, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "Alt Client",
            "Brisbane",
            "Sydney",
            "Brisbane",
            "Sydney",
            "2026-03-21",
            900.0,
            20.0,
            -27.47,
            153.02,
            -33.87,
            151.21,
            "2026-03-12T00:00:00+00:00",
        ),
    ).lastrowid
    segment = ensure_segment(
        conn,
        job_id=int(job1),
        segment_sequence=1,
        from_location="Brisbane",
        to_location="Cairns",
        planned_start="2026-03-20T08:00:00+00:00",
        planned_end="2026-03-20T12:00:00+00:00",
    )
    assign_segment_resources(
        conn,
        segment_id=int(segment["id"]),
        truck_ids=["TRK-1"],
        worker_assignments=[{"workerId": int(worker["id"])}],
    )
    ensure_segment(
        conn,
        job_id=int(job2),
        segment_sequence=1,
        from_location="Brisbane",
        to_location="Sydney",
        planned_start="2026-03-21T09:00:00+00:00",
        planned_end="2026-03-21T13:00:00+00:00",
    )
    supplier_id = conn.execute(
        """
        INSERT INTO suppliers (company_name, created_at, updated_at)
        VALUES (?, ?, ?)
        """,
        ("Holiday Carrier", "2026-03-12T00:00:00+00:00", "2026-03-12T00:00:00+00:00"),
    ).lastrowid
    upsert_driver_shift(
        conn,
        shift_date="2026-03-20",
        truck_id="TRK-1",
        worker_name="Alex Planner",
        shift_start="2026-03-20T08:00:00+00:00",
        shift_end="2026-03-20T12:00:00+00:00",
        hours=4.0,
        hourly_rate=35.0,
        job_id=int(job1),
        source="import",
    )
    conn.commit()
    return conn, {
        "supplier_id": int(supplier_id),
        "job1": int(job1),
        "job2": int(job2),
    }


def test_diary_task_crud_and_filters() -> None:
    conn, ids = _seed_conn()
    created = upsert_operations_diary_task(
        conn,
        job_id=ids["job1"],
        task_date="2026-03-20",
        task_scope="day",
        title="Review holiday subcontractor bill",
    )
    updated = upsert_operations_diary_task(
        conn,
        task_id=int(created["id"]),
        job_id=ids["job1"],
        task_date="2026-03-20",
        task_scope="job",
        title="Review holiday subcontractor bill",
        status="in_progress",
    )

    assert updated["task_scope"] == "job"
    assert updated["status"] == "in_progress"

    rows = list_operations_diary_tasks(conn, start_date="2026-03-20", end_date="2026-03-20")
    assert len(rows) == 1

    delete_operations_diary_task(conn, task_id=int(created["id"]))
    assert list_operations_diary_tasks(conn, start_date="2026-03-20", end_date="2026-03-20") == []


def test_build_operations_diary_surfaces_invoice_and_bill_status() -> None:
    conn, ids = _seed_conn()
    upsert_operations_diary_task(
        conn,
        job_id=ids["job1"],
        task_date="2026-03-20",
        title="Call customer before invoicing",
    )
    upsert_customer_invoice_review(
        conn,
        job_id=ids["job1"],
        invoice_status="ready_to_invoice",
        invoice_reference="INV-1001",
        invoice_amount=2400.0,
    )
    upsert_subcontractor_bill_review(
        conn,
        job_id=ids["job1"],
        supplier_id=ids["supplier_id"],
        bill_status="bill_received",
        bill_reference="BILL-77",
        amount=900.0,
    )

    diary = build_operations_diary(conn, anchor_date="2026-03-20", view_mode="day", focus_job_id=ids["job1"])

    assert diary["summary"]["jobCount"] == 1
    assert diary["summary"]["taskCount"] == 1
    row = diary["jobs"][0]
    assert row["jobId"] == ids["job1"]
    assert row["invoiceStatus"] == "ready_to_invoice"
    assert row["billStatus"] == "bill_received"
    assert row["isFocusJob"] is True


def test_build_job_usage_details_includes_vehicle_and_staff_usage() -> None:
    conn, ids = _seed_conn()
    details = build_job_usage_details(conn, job_id=ids["job1"])

    assert details["job"]["jobId"] == ids["job1"]
    assert len(details["vehicleUsage"]) == 1
    assert details["vehicleUsage"][0]["truckId"] == "TRK-1"
    assert details["vehicleUsage"][0]["actualShiftCount"] == 1
    assert len(details["staffUsage"]) == 1
    assert details["staffUsage"][0]["workerName"] == "Alex Planner"
    assert details["staffUsage"][0]["actualShiftCount"] == 1


def test_reconciliation_exposure_summary_uses_bill_receipt_for_age_and_job_for_latency() -> None:
    conn, ids = _seed_conn()
    upsert_subcontractor_bill_review(
        conn,
        job_id=ids["job1"],
        supplier_id=ids["supplier_id"],
        bill_status="bill_received",
        bill_reference="BILL-99",
        bill_date="2026-03-25",
        amount=30000.0,
    )
    upsert_customer_invoice_review(
        conn,
        job_id=ids["job1"],
        invoice_status="ready_to_invoice",
        invoice_reference="INV-1002",
        invoice_date="2026-03-24",
        invoice_amount=45000.0,
    )

    summary = build_reconciliation_exposure_summary(conn, as_of_date="2026-04-10")

    assert summary["supplierUnresolvedTotal"] == 30000.0
    assert summary["customerOpenTotal"] == 45000.0
    assert summary["oldestSupplierAgeDays"] == 16
    assert summary["longestSupplierLatencyDays"] == 5
    row = summary["activeSupplierRows"][0]
    assert row["jobExecutionDate"] == "2026-03-20"
    assert row["receivedDate"] == "2026-03-25"
    assert row["latencyDays"] == 5
    assert row["unresolvedAgeDays"] == 16
    assert row["signedAmount"] == -30000.0


def test_reconciliation_resolution_timestamps_close_when_review_resolves() -> None:
    conn, ids = _seed_conn()
    bill = upsert_subcontractor_bill_review(
        conn,
        job_id=ids["job1"],
        supplier_id=ids["supplier_id"],
        bill_status="bill_received",
        bill_date="2026-03-25",
        amount=900.0,
    )
    assert bill["resolved_at"] is None

    reconciled = upsert_subcontractor_bill_review(
        conn,
        bill_id=int(bill["id"]),
        job_id=ids["job1"],
        supplier_id=ids["supplier_id"],
        bill_status="bill_reconciled",
        bill_date="2026-03-25",
        amount=900.0,
    )
    assert reconciled["resolved_at"] is not None

    invoice = upsert_customer_invoice_review(
        conn,
        job_id=ids["job1"],
        invoice_status="ready_to_invoice",
        invoice_date="2026-03-24",
        invoice_amount=1200.0,
    )
    assert invoice["resolved_at"] is None


def test_diary_and_review_write_paths_emit_observer_events() -> None:
    conn, ids = _seed_conn()
    created = upsert_operations_diary_task(
        conn,
        job_id=ids["job1"],
        task_date="2026-03-20",
        title="Review invoice release",
        actor_ref="ops-manager",
    )
    upsert_operations_diary_task(
        conn,
        task_id=int(created["id"]),
        job_id=ids["job1"],
        task_date="2026-03-20",
        title="Review invoice release",
        status="in_progress",
        actor_ref="ops-manager",
    )
    delete_operations_diary_task(conn, task_id=int(created["id"]), actor_ref="ops-manager")
    upsert_customer_invoice_review(
        conn,
        job_id=ids["job1"],
        invoice_status="reconciliation_warning",
        invoice_reference="INV-2001",
        reviewed_by="ops-manager",
    )
    upsert_subcontractor_bill_review(
        conn,
        job_id=ids["job1"],
        supplier_id=ids["supplier_id"],
        bill_status="bill_received",
        bill_reference="BILL-2001",
        reviewed_by="ops-manager",
    )

    rows = list_observer_outbox_events(conn, limit=20)
    families = [row["eventFamily"] for row in rows]
    assert families.count("diary_task_event") == 3
    assert "customer_invoice_review" in families
    assert "subcontractor_bill_review" in families


def test_explicit_diary_export_emits_snapshot_and_exception_rows_idempotently() -> None:
    conn, ids = _seed_conn()
    upsert_customer_invoice_review(
        conn,
        job_id=ids["job1"],
        invoice_status="reconciliation_warning",
        reviewed_by="ops-manager",
    )
    upsert_subcontractor_bill_review(
        conn,
        job_id=ids["job1"],
        supplier_id=ids["supplier_id"],
        bill_status="bill_exception",
        bill_reference="BILL-404",
        bill_date="2026-03-25",
        amount=1200.0,
        reviewed_by="ops-manager",
    )

    export = export_operations_diary_observer_events(
        conn,
        anchor_date="2026-03-20",
        view_mode="day",
        actor_ref="ops-manager",
    )
    repeat = export_operations_diary_observer_events(
        conn,
        anchor_date="2026-03-20",
        view_mode="day",
        actor_ref="ops-manager",
    )

    assert export["byFamily"]["planning_snapshot"] == 1
    assert export["byFamily"]["reconciliation_exception"] >= 2
    assert repeat["emittedCount"] == export["emittedCount"]
    rows = list_observer_outbox_events(conn, limit=20)
    families = [row["eventFamily"] for row in rows]
    assert "planning_snapshot" in families
    assert "reconciliation_exception" in families

    invoiced = upsert_customer_invoice_review(
        conn,
        job_id=ids["job1"],
        invoice_status="invoiced",
        invoice_date="2026-03-26",
        invoice_amount=1200.0,
    )
    assert invoiced["resolved_at"] == "2026-03-26"
