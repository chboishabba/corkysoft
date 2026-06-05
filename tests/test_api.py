from __future__ import annotations

import json
import sqlite3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from fastapi.testclient import TestClient

import corkysoft.api as api
from analytics.db import (
    create_worker_absence_record,
    ensure_dashboard_tables,
    upsert_driver_shift,
    upsert_truck,
    upsert_vehicle_details,
    upsert_worker,
)
from analytics.operations_assignment import assign_segment_resources, ensure_segment
from analytics.operations_diary import (
    upsert_customer_invoice_review,
    upsert_operations_diary_task,
    upsert_subcontractor_bill_review,
)
from corkysoft.call_ops import record_worker_time_capture_event
from corkysoft.whisperx_adapter import WhisperXAdapterError


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    """Provision an isolated SQLite database for API tests."""

    db_path = tmp_path / "api.db"
    monkeypatch.setenv("CORKYSOFT_DB", str(db_path))
    monkeypatch.setenv("CORKYSOFT_API_TOKEN", "test-token")
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_date TEXT,
                client TEXT,
                origin TEXT,
                destination TEXT,
                revenue_total REAL,
                revenue REAL,
                volume_m3 REAL,
                volume REAL,
                distance_km REAL,
                final_cost REAL,
                updated_at TEXT,
                billing_name TEXT,
                billing_email TEXT,
                service_code TEXT,
                service_text TEXT
            );
            """
        )
        conn.commit()
    yield db_path


AUTH_HEADERS = {"X-Corkysoft-Api-Key": "test-token"}


def _set_service_credentials(monkeypatch, credentials):
    monkeypatch.setenv(
        "CORKYSOFT_SERVICE_CREDENTIALS_JSON",
        json.dumps({"credentials": credentials}),
    )


def _create_job(conn: sqlite3.Connection) -> int:
    cursor = conn.execute(
        """
        INSERT INTO jobs (
            job_date,
            client,
            origin,
            destination,
            revenue_total,
            revenue,
            volume_m3,
            volume,
            distance_km,
            final_cost,
            updated_at,
            billing_name,
            billing_email,
            service_code,
            service_text
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "2018-08-30T00:00+10:00",
            "SYD Customer",
            "1p, 2p",
            "3p 4p",
            1250.0,
            1250.0,
            50.0,
            50.0,
            900.5,
            975.0,
            "2018-09-01T12:30:00+10:00",
            "SYD Customer",
            "luke.pitcher@moveconnect.com",
            "LTL",
            "Less than truck load",
        ),
    )
    return int(cursor.lastrowid)


def test_get_job_by_id_returns_payload(isolated_db):
    with sqlite3.connect(isolated_db) as conn:
        job_id = _create_job(conn)
        conn.commit()

    client = TestClient(api.app)
    response = client.get(f"/jobs/{job_id}", headers=AUTH_HEADERS)
    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == str(job_id)
    assert payload["billing"]["name"] == "SYD Customer"
    assert payload["billing"]["email"] == "luke.pitcher@moveconnect.com"
    assert payload["service"]["code"] == "LTL"


def test_get_job_by_id_missing_returns_404(isolated_db):
    client = TestClient(api.app)
    response = client.get("/jobs/9999", headers=AUTH_HEADERS)
    assert response.status_code == 404
    assert response.json()["detail"] == "Job not found"


def test_top_level_sensitive_read_routes_require_api_token(isolated_db):
    client = TestClient(api.app)

    job_response = client.get("/jobs/9999")
    assert job_response.status_code == 401
    assert job_response.json()["detail"] == "Invalid internal API token"

    shifts_response = client.get("/driver-shifts")
    assert shifts_response.status_code == 401
    assert shifts_response.json()["detail"] == "Invalid internal API token"

    operations_response = client.get("/operations/policy")
    assert operations_response.status_code == 401
    assert operations_response.json()["detail"] == "Invalid internal API token"

    labor_response = client.get("/labor-analytics/summary")
    assert labor_response.status_code == 401
    assert labor_response.json()["detail"] == "Invalid internal API token"

    kent_response = client.get("/kent-ams/config")
    assert kent_response.status_code == 401
    assert kent_response.json()["detail"] == "Invalid internal API token"

    calls_response = client.get("/calls/events")
    assert calls_response.status_code == 401
    assert calls_response.json()["detail"] == "Invalid internal API token"


def test_moveware_importer_returns_summary(isolated_db):
    client = TestClient(api.app)
    response = client.post(
        "/importers/moveware/jobs",
        json={
            "records": [
                {"id": "100006", "externalId": "X001-D"},
                {"id": "100007", "externalId": "X002-D"},
            ],
            "dry_run": True,
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "resource": "jobs",
        "imported": 2,
        "dry_run": True,
    }


def test_mutating_internal_routes_require_api_token(isolated_db):
    client = TestClient(api.app)
    response = client.post(
        "/importers/kent-ams/jobs",
        json={
            "records": [
                {"moveId": "KENT-1001", "origin": "Brisbane", "destination": "Cairns"},
            ],
            "dry_run": True,
        },
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid internal API token"


def test_operations_policy_segment_assignment_and_conflicts(isolated_db):
    with sqlite3.connect(isolated_db) as conn:
        conn.row_factory = sqlite3.Row
        ensure_dashboard_tables(conn)
        job_id = _create_job(conn)
        upsert_truck(conn, truck_id="TRK-OPS-1", name="Ops Truck")
        upsert_vehicle_details(
            conn,
            truck_id="TRK-OPS-1",
            rego="TRK-OPS-1",
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
            name="Ops Worker",
            source_system="google_sheets",
            source_sheet="STAFF",
            source_imported_at="2026-03-12T00:00:00+00:00",
        )
        conn.commit()

    client = TestClient(api.app)

    response = client.get("/operations/policy", headers=AUTH_HEADERS)
    assert response.status_code == 200
    assert response.json()["regoWarningDays"] == 30

    response = client.put(
        "/operations/policy",
        json={
            "regoWarningDays": 21,
            "coiWarningDays": 14,
            "serviceWarningDays": 10,
            "complianceWarningDays": 7,
            "serviceOverdueBlocks": True,
            "conflictBlocks": True,
            "serviceOverrideAllowed": True,
            "conflictOverrideAllowed": True,
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["regoWarningDays"] == 21

    response = client.post(
        "/operations/segments",
        json={
            "jobId": job_id,
            "segmentSequence": 1,
            "fromLocation": "Brisbane",
            "toLocation": "Townsville",
            "plannedStart": "2026-03-12T08:00:00+00:00",
            "plannedEnd": "2026-03-12T12:00:00+00:00",
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    segment = response.json()
    assert segment["assignmentStatus"] == "draft"
    assert segment["warningFlags"] == [
        "segment:no_truck_assigned",
        "segment:no_worker_assigned",
    ]

    response = client.post(
        f"/operations/segments/{segment['segmentId']}/assign",
        json={
            "truckIds": ["TRK-OPS-1"],
            "workerAssignments": [{"workerId": int(worker['id'])}],
            "override": False,
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    assigned = response.json()
    assert assigned["assignmentStatus"] == "planned"
    assert assigned["truckAssignments"][0]["truckId"] == "TRK-OPS-1"
    assert assigned["truckAssignments"][0]["sourceImportedAt"] == "2026-03-12T00:00:00+00:00"
    assert assigned["workerAssignments"][0]["workerId"] == int(worker["id"])
    assert assigned["workerAssignments"][0]["sourceImportedAt"] == "2026-03-12T00:00:00+00:00"

    response = client.get(f"/operations/segments/readiness?job_id={job_id}", headers=AUTH_HEADERS)
    assert response.status_code == 200
    rows = response.json()
    assert len(rows) == 1
    assert rows[0]["assignmentStatus"] == "planned"

    response = client.get("/operations/conflicts", headers=AUTH_HEADERS)
    assert response.status_code == 200
    assert response.json() == []


def test_labor_analytics_summary_export_absence_and_cost_drivers(isolated_db):
    with sqlite3.connect(isolated_db) as conn:
        conn.row_factory = sqlite3.Row
        ensure_dashboard_tables(conn)
        upsert_truck(conn, truck_id="TRK-PAY-1", name="Payroll Truck")
        worker = upsert_worker(
            conn,
            name="Payroll Worker",
            source_system="google_sheets",
            source_sheet="STAFF",
            source_imported_at="2026-03-12T00:00:00+00:00",
        )
        job_id = _create_job(conn)
        segment = ensure_segment(
            conn,
            job_id=job_id,
            segment_sequence=1,
            from_location="Brisbane",
            to_location="Ipswich",
            planned_start="2026-03-14T08:00:00+10:00",
            planned_end="2026-03-14T12:00:00+10:00",
        )
        assign_segment_resources(
            conn,
            segment_id=int(segment["id"]),
            truck_ids=["TRK-PAY-1"],
            worker_assignments=[{"workerId": int(worker["id"])}],
        )
        upsert_driver_shift(
            conn,
            shift_date="2026-03-14",
            truck_id="TRK-PAY-1",
            worker_name="Payroll Worker",
            shift_window_start="08:00",
            shift_window_end="12:00",
            hours=9.5,
            hourly_rate=40.0,
            job_id=job_id,
            source="VEHICLE_DRIVER",
            imported_at="2026-03-14T00:00:00+10:00",
        )
        record_worker_time_capture_event(
            conn,
            event_type="clock_on",
            channel="manual_supervisor",
            worker_id=int(worker["id"]),
            worker_name_raw="Payroll Worker",
            effective_timestamp="2026-03-14T08:15:00+10:00",
            truck_id="TRK-PAY-1",
            job_id=job_id,
            segment_id=int(segment["id"]),
            confidence=0.95,
        )
        create_worker_absence_record(
            conn,
            worker_id=int(worker["id"]),
            start_date="2026-03-14",
            end_date="2026-03-14",
            absence_type="annual_leave",
            status="confirmed",
            hours_per_day=8.0,
            source="manager_manual",
            recorded_by="payroll-admin",
        )
        conn.commit()

    client = TestClient(api.app)

    response = client.get(
        "/labor-analytics/summary?start_date=2026-03-14&end_date=2026-03-14",
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    summary = response.json()
    assert summary["plannedExposure"] == pytest.approx(160.0)
    assert summary["importedCost"] == pytest.approx(380.0)
    assert summary["reviewedActualCost"] == pytest.approx(380.0)
    assert summary["absenceModelStatus"] == "basic_recorded"
    assert summary["absenceRecordCount"] == 1
    assert summary["confirmedAbsenceCount"] == 1

    response = client.get(
        "/labor-analytics/export-summary?start_date=2026-03-14&end_date=2026-03-14",
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    export_rows = response.json()
    assert any(
        row["workerName"] == "Payroll Worker"
        and row["importedCost"] == pytest.approx(380.0)
        and row["absenceDays"] == pytest.approx(1.0)
        for row in export_rows
    )

    response = client.get(
        "/labor-analytics/absence?start_date=2026-03-14&end_date=2026-03-14",
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    absence = response.json()
    assert absence["annualLeaveDays"] == pytest.approx(1.0)
    assert absence["recordCount"] == 1

    response = client.get(
        "/labor-analytics/cost-drivers?dimension=worker&start_date=2026-03-14&end_date=2026-03-14",
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    drivers = response.json()
    assert any(
        row["dimension"] == "worker"
        and row["dimensionValue"] == "Payroll Worker"
        and row["totalCost"] == pytest.approx(380.0)
        for row in drivers
    )


def test_worker_absence_record_api_round_trip(isolated_db):
    with sqlite3.connect(isolated_db) as conn:
        conn.row_factory = sqlite3.Row
        ensure_dashboard_tables(conn)
        worker = upsert_worker(conn, name="Leave Worker")
        conn.commit()

    client = TestClient(api.app)
    response = client.post(
        "/worker-absence/records",
        json={
            "workerId": int(worker["id"]),
            "startDate": "2026-03-20",
            "endDate": "2026-03-21",
            "absenceType": "sick",
            "status": "confirmed",
            "hoursPerDay": 7.5,
            "source": "manual_manager",
            "recordedBy": "ops-manager",
            "note": "Two-day illness",
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    created = response.json()
    assert created["workerName"] == "Leave Worker"
    assert created["absenceType"] == "sick"

    response = client.get(f"/worker-absence/records?worker_id={int(worker['id'])}", headers=AUTH_HEADERS)
    assert response.status_code == 200
    rows = response.json()
    assert len(rows) == 1
    assert rows[0]["endDate"] == "2026-03-21"


def test_operations_sync_route_dispatches_shared_workbook(monkeypatch, isolated_db):
    client = TestClient(api.app)

    def fake_sync(conn, *, sheet_id_or_url=None):
        assert sheet_id_or_url == "ops-shared-sheet"
        return {
            "fleetImported": 5,
            "staffInserted": 2,
            "staffUpdated": 1,
            "suppliersImported": 3,
            "staffSheetName": "STAFF",
            "suppliersSheetName": "SUPPLIERS",
        }

    monkeypatch.setattr(api, "sync_operations_workbook", fake_sync)

    response = client.post(
        "/operations/sync",
        json={"reference": "ops-shared-sheet"},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["fleetImported"] == 5
    assert response.json()["staffUpdated"] == 1


def test_operations_readiness_and_worker_compliance_routes(isolated_db):
    with sqlite3.connect(isolated_db) as conn:
        conn.row_factory = sqlite3.Row
        ensure_dashboard_tables(conn)
        upsert_truck(conn, truck_id="TRK-OPS-2", name="Ops Truck 2")
        upsert_vehicle_details(
            conn,
            truck_id="TRK-OPS-2",
            rego="TRK-OPS-2",
            rego_expiry="2000-01-01",
            coi_due="2099-12-31",
            next_service="2099-12-31",
            daily_check_complete=True,
            source_system="google_sheets",
            source_sheet="FLEET",
            source_imported_at="2026-03-12T00:00:00+00:00",
        )
        worker = upsert_worker(
            conn,
            name="Compliance Worker API",
            source_system="google_sheets",
            source_sheet="STAFF",
            source_imported_at="2026-03-12T00:00:00+00:00",
        )
        conn.commit()

    client = TestClient(api.app)

    response = client.post(
        f"/operations/workers/{int(worker['id'])}/roles",
        json={"roleName": "Driver"},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["workerId"] == int(worker["id"])

    response = client.post(
        f"/operations/workers/{int(worker['id'])}/compliances",
        json={"complianceName": "MSIC", "expiryDate": "2000-01-01"},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["workerId"] == int(worker["id"])

    response = client.get("/operations/readiness/resources", headers=AUTH_HEADERS)
    assert response.status_code == 200
    rows = response.json()
    assert any(
        row["resourceType"] == "vehicle"
        and row["resourceId"] == "TRK-OPS-2"
        and row["status"] == "blocked"
        for row in rows
    )
    assert any(
        row["resourceType"] == "worker"
        and row["resourceId"] == str(worker["id"])
        and row["status"] == "blocked"
        for row in rows
    )


def test_operations_labor_roster_and_reconciliation_routes(isolated_db):
    with sqlite3.connect(isolated_db) as conn:
        conn.row_factory = sqlite3.Row
        ensure_dashboard_tables(conn)
        job_id = _create_job(conn)
        upsert_truck(conn, truck_id="TRK-OPS-3", name="Ops Truck 3")
        upsert_vehicle_details(
            conn,
            truck_id="TRK-OPS-3",
            rego="TRK-OPS-3",
            rego_expiry="2099-12-31",
            coi_due="2099-12-31",
            next_service="2099-12-31",
            daily_check_complete=True,
        )
        worker = upsert_worker(conn, name="Roster API Worker")
        conn.commit()

    client = TestClient(api.app)
    response = client.post(
        "/operations/segments",
        json={
            "jobId": job_id,
            "segmentSequence": 1,
            "plannedStart": "2026-03-14T08:00:00+00:00",
            "plannedEnd": "2026-03-14T11:00:00+00:00",
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    segment_id = response.json()["segmentId"]

    response = client.post(
        f"/operations/segments/{segment_id}/assign",
        json={
            "truckIds": ["TRK-OPS-3"],
            "workerAssignments": [{"workerId": int(worker["id"])}],
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200

    with sqlite3.connect(isolated_db) as conn:
        conn.execute(
            """
            INSERT INTO driver_shifts (
                shift_date, truck_id, worker_id, shift_window_start, shift_window_end, source, imported_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("2026-03-14", "TRK-OPS-3", int(worker["id"]), "08:00", "11:00", "VEHICLE_DRIVER", "2026-03-14T00:00:00+00:00"),
        )
        conn.commit()

    response = client.get("/operations/labor/roster?start_date=2026-03-14&end_date=2026-03-14", headers=AUTH_HEADERS)
    assert response.status_code == 200
    roster = response.json()
    assert len(roster) == 1
    assert roster[0]["workerName"] == "Roster API Worker"
    assert roster[0]["truckIds"] == ["TRK-OPS-3"]

    response = client.get("/operations/labor/reconciliation?start_date=2026-03-14&end_date=2026-03-14", headers=AUTH_HEADERS)
    assert response.status_code == 200
    reconciliation = response.json()
    assert any(
        row["status"] == "matched"
        and row["workerName"] == "Roster API Worker"
        for row in reconciliation
    )


def test_operations_inventory_segment_routes(isolated_db):
    with sqlite3.connect(isolated_db) as conn:
        conn.row_factory = sqlite3.Row
        ensure_dashboard_tables(conn)
        job_id = _create_job(conn)
        supplier = conn.execute(
            """
            INSERT INTO suppliers (company_name, created_at, updated_at)
            VALUES (?, ?, ?)
            """,
            ("Segment Supplier", "2026-03-12T00:00:00+00:00", "2026-03-12T00:00:00+00:00"),
        ).lastrowid
        item = conn.execute(
            """
            INSERT INTO inventory_items (name, quantity, unit, supplier_id, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("Pads", 10, "ea", supplier, "2026-03-12T00:00:00+00:00"),
        ).lastrowid
        conn.commit()

    client = TestClient(api.app)
    response = client.post(
        "/operations/segments",
        json={
            "jobId": job_id,
            "segmentSequence": 1,
            "fromLocation": "Depot",
            "toLocation": "Customer",
            "plannedStart": "2026-03-15T09:00:00+00:00",
            "plannedEnd": "2026-03-15T12:00:00+00:00",
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    segment_id = response.json()["segmentId"]

    response = client.post(
        f"/operations/inventory/segments/{segment_id}/allocate",
        json={
            "inventoryItemId": int(item),
            "quantity": 2,
            "status": "staged",
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["segmentId"] == segment_id

    response = client.get(f"/operations/inventory/segments?job_id={job_id}", headers=AUTH_HEADERS)
    assert response.status_code == 200
    rows = response.json()
    assert len(rows) == 1
    assert rows[0]["shipmentCount"] == 1
    assert rows[0]["inventoryNames"] == ["Pads"]
    assert rows[0]["supplierNames"] == ["Segment Supplier"]


def test_operations_jobs_board_route(isolated_db):
    with sqlite3.connect(isolated_db) as conn:
        conn.row_factory = sqlite3.Row
        ensure_dashboard_tables(conn)
        job_id = _create_job(conn)
        supplier = conn.execute(
            """
            INSERT INTO suppliers (company_name, created_at, updated_at)
            VALUES (?, ?, ?)
            """,
            ("Board API Supplier", "2026-03-12T00:00:00+00:00", "2026-03-12T00:00:00+00:00"),
        ).lastrowid
        item = conn.execute(
            """
            INSERT INTO inventory_items (name, quantity, unit, supplier_id, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("Trolleys", 3, "ea", supplier, "2026-03-12T00:00:00+00:00"),
        ).lastrowid
        upsert_truck(conn, truck_id="TRK-OPS-BOARD", name="Ops Board Truck")
        upsert_vehicle_details(
            conn,
            truck_id="TRK-OPS-BOARD",
            rego="TRK-OPS-BOARD",
            rego_expiry="2099-12-31",
            coi_due="2099-12-31",
            next_service="2099-12-31",
            daily_check_complete=True,
        )
        worker = upsert_worker(conn, name="Board API Worker")
        conn.commit()

    client = TestClient(api.app)
    response = client.post(
        "/operations/segments",
        json={
            "jobId": job_id,
            "segmentSequence": 1,
            "plannedStart": "2026-03-16T08:00:00+00:00",
            "plannedEnd": "2026-03-16T12:00:00+00:00",
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    segment_id = response.json()["segmentId"]
    response = client.post(
        f"/operations/segments/{segment_id}/assign",
        json={
            "truckIds": ["TRK-OPS-BOARD"],
            "workerAssignments": [{"workerId": int(worker["id"])}],
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    response = client.post(
        f"/operations/inventory/segments/{segment_id}/allocate",
        json={
            "inventoryItemId": int(item),
            "quantity": 1,
            "status": "staged",
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200

    response = client.get(f"/operations/jobs/board?job_id={job_id}", headers=AUTH_HEADERS)
    assert response.status_code == 200
    rows = response.json()
    assert len(rows) == 1
    assert rows[0]["jobId"] == job_id
    assert rows[0]["truckIds"] == ["TRK-OPS-BOARD"]
    assert rows[0]["workerNames"] == ["Board API Worker"]
    assert rows[0]["inventoryNames"] == ["Trolleys"]
    assert rows[0]["supplierNames"] == ["Board API Supplier"]


def test_operations_cutover_workflow_routes(isolated_db):
    client = TestClient(api.app)

    response = client.get("/operations/cutover/workflows", headers=AUTH_HEADERS)
    assert response.status_code == 200
    rows = response.json()
    assert any(row["workflowKey"] == "dispatch_execution" for row in rows)

    response = client.put(
        "/operations/cutover/workflows/dispatch_execution",
        json={
            "cutoverStatus": "fallback_only",
            "ownerRole": "dispatcher",
            "snapshotMode": "daily",
            "snapshotFields": ["jobId", "jobStatus"],
            "fallbackMode": "manual_csv",
            "cutoverTargetPercent": 95.0,
            "nativeReady": True,
            "dualRunComplete": True,
            "fallbackDrillComplete": True,
            "operatorTrained": True,
            "rollbackInstructions": "Use the latest CSV snapshot while imports recover.",
            "notes": "Drill passed.",
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["workflowKey"] == "dispatch_execution"
    assert payload["cutoverStatus"] == "fallback_only"
    assert payload["snapshotMode"] == "daily"
    assert payload["metrics"]["cutoverTargetPercent"] == 95.0
    assert payload["checklist"]["fallbackDrillComplete"] is True
    response = client.post(
        "/operations/cutover/workflows/dispatch_execution/events",
        json={
            "eventType": "review",
            "actor": "dispatcher",
            "createdAt": "2026-03-12T09:00:00+00:00",
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    response = client.post(
        "/operations/cutover/workflows/dispatch_execution/events",
        json={
            "eventType": "fallback_drill",
            "actor": "dispatcher",
            "createdAt": "2026-03-12T08:00:00+00:00",
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    response = client.post(
        "/operations/cutover/workflows/dispatch_execution/events",
        json={
            "eventType": "snapshot_issued",
            "actor": "dispatcher",
            "eventValue": "ops-team",
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    response = client.get("/operations/cutover/events?workflow_key=dispatch_execution", headers=AUTH_HEADERS)
    assert response.status_code == 200
    events = response.json()
    assert len(events) >= 3
    response = client.get("/operations/cutover/workflows", headers=AUTH_HEADERS)
    assert response.status_code == 200
    rows = response.json()
    dispatch_row = next(row for row in rows if row["workflowKey"] == "dispatch_execution")
    assert dispatch_row["metrics"]["lastReviewAt"] == "2026-03-12T09:00:00+00:00"
    assert dispatch_row["lastDrillAt"] == "2026-03-12T08:00:00+00:00"
    assert dispatch_row["metrics"]["snapshotConsumerCount"] == 1
    assert "recommendation" in dispatch_row


def test_operations_cutover_scoped_credentials_bind_actor_and_receipt(monkeypatch, isolated_db):
    _set_service_credentials(
        monkeypatch,
        [
            {
                "id": "ops-cutover-writer",
                "token": "writer-token",
                "actor": "credential-ops-manager",
                "scopes": ["api:read", "operations.cutover:write"],
            }
        ],
    )
    headers = {
        "X-Corkysoft-Api-Key": "writer-token",
        "X-Corkysoft-Request-Id": "req-cutover-001",
    }
    client = TestClient(api.app)

    response = client.post(
        "/operations/cutover/workflows/dispatch_execution/events",
        json={
            "eventType": "review",
            "actor": "spoofed-body-actor",
            "createdAt": "2026-03-12T09:00:00+00:00",
        },
        headers=headers,
    )

    assert response.status_code == 200
    assert response.json()["actor"] == "credential-ops-manager"
    with sqlite3.connect(isolated_db) as conn:
        conn.row_factory = sqlite3.Row
        event = conn.execute(
            """
            SELECT actor
            FROM operations_cutover_events
            WHERE workflow_key = ? AND event_type = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            ("dispatch_execution", "review"),
        ).fetchone()
        receipt = conn.execute(
            """
            SELECT credential_id, actor, scopes_json, action, resource_type,
                   resource_id, request_id, route, method
            FROM api_write_receipts
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()

    assert event["actor"] == "credential-ops-manager"
    assert receipt["credential_id"] == "ops-cutover-writer"
    assert receipt["actor"] == "credential-ops-manager"
    assert "operations.cutover:write" in json.loads(receipt["scopes_json"])
    assert receipt["action"] == "operations_cutover_event:review"
    assert receipt["resource_type"] == "operations_cutover_workflow"
    assert receipt["resource_id"] == "dispatch_execution"
    assert receipt["request_id"] == "req-cutover-001"
    assert receipt["route"] == "/operations/cutover/workflows/dispatch_execution/events"
    assert receipt["method"] == "POST"


def test_operations_cutover_wrong_scope_fails_closed(monkeypatch, isolated_db):
    _set_service_credentials(
        monkeypatch,
        [
            {
                "id": "reader-only",
                "token": "reader-token",
                "actor": "reader",
                "scopes": ["api:read"],
            }
        ],
    )
    client = TestClient(api.app)

    response = client.post(
        "/operations/cutover/workflows/dispatch_execution/events",
        json={"eventType": "review", "actor": "spoofed-body-actor"},
        headers={"X-Corkysoft-Api-Key": "reader-token"},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Credential scope is not authorized"
    with sqlite3.connect(isolated_db) as conn:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'api_write_receipts'"
        ).fetchone()
        receipt_count = 0
        if exists is not None:
            receipt_count = conn.execute("SELECT COUNT(*) FROM api_write_receipts").fetchone()[0]
    assert receipt_count == 0


def test_kent_scoped_credentials_bind_operator_and_receipt(monkeypatch, isolated_db):
    _set_service_credentials(
        monkeypatch,
        [
            {
                "id": "kent-writer",
                "token": "kent-token",
                "actor": "credential-kent-operator",
                "scopes": ["api:read", "kent:write"],
            }
        ],
    )
    with sqlite3.connect(isolated_db) as conn:
        conn.row_factory = sqlite3.Row
        ensure_dashboard_tables(conn)
        api.import_kent_ams_records(
            conn,
            "tenders",
            [
                {
                    "tenderId": "T-SCOPED",
                    "moveId": "JOB-SCOPED",
                    "clientName": "Scoped Kent",
                    "origin": "Brisbane QLD",
                    "destination": "Cairns QLD",
                    "expectedRevenue": 5000.0,
                    "estimatedCost": 5200.0,
                    "requiredTrucks": 1,
                    "requiredWorkers": 2,
                    "dueAt": "2026-03-20T06:00:00+00:00",
                    "transferRuleViolated": True,
                }
            ],
            dry_run=False,
        )
        conn.commit()

    client = TestClient(api.app)
    response = client.post(
        "/kent-ams/tenders/T-SCOPED/override",
        json={
            "action": "pursue",
            "operatorId": "spoofed-body-operator",
            "reasonCode": "retention",
            "note": "Scoped credential should bind the actor.",
        },
        headers={
            "X-Corkysoft-Api-Key": "kent-token",
            "X-Corkysoft-Request-Id": "req-kent-001",
        },
    )

    assert response.status_code == 200
    assert response.json()["operatorId"] == "credential-kent-operator"
    with sqlite3.connect(isolated_db) as conn:
        conn.row_factory = sqlite3.Row
        receipt = conn.execute(
            """
            SELECT credential_id, actor, action, resource_type, request_id
            FROM api_write_receipts
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
    assert receipt["credential_id"] == "kent-writer"
    assert receipt["actor"] == "credential-kent-operator"
    assert receipt["action"] == "kent.tender_override.create"
    assert receipt["resource_type"] == "kent_tender_override"
    assert receipt["request_id"] == "req-kent-001"


def test_worker_time_scoped_credentials_bind_reviewer_and_receipt(monkeypatch, isolated_db):
    _set_service_credentials(
        monkeypatch,
        [
            {
                "id": "worker-time-reviewer",
                "token": "worker-time-token",
                "actor": "credential-payroll-reviewer",
                "scopes": ["api:read", "worker_time:write"],
            }
        ],
    )
    with sqlite3.connect(isolated_db) as conn:
        conn.row_factory = sqlite3.Row
        ensure_dashboard_tables(conn)
        worker = upsert_worker(conn, name="Scoped Worker", phone="0400123456")
        event = record_worker_time_capture_event(
            conn,
            worker_id=int(worker["id"]),
            worker_name_raw="Scoped Worker",
            event_type="clock_on",
            channel="voice_call",
            effective_timestamp="2026-03-13T06:30:00+10:00",
            confidence=0.6,
        )
        event_id = int(event["id"])
        conn.commit()

    client = TestClient(api.app)
    response = client.post(
        f"/worker-time/events/{event_id}/decision",
        json={
            "reviewStatus": "accepted",
            "reviewer": "spoofed-body-reviewer",
            "reviewNote": "Accept from scoped credential.",
        },
        headers={
            "X-Corkysoft-Api-Key": "worker-time-token",
            "X-Corkysoft-Request-Id": "req-worker-time-001",
        },
    )

    assert response.status_code == 200
    assert response.json()["reviewer"] == "credential-payroll-reviewer"
    with sqlite3.connect(isolated_db) as conn:
        conn.row_factory = sqlite3.Row
        receipt = conn.execute(
            """
            SELECT credential_id, actor, action, resource_type, resource_id, request_id
            FROM api_write_receipts
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
    assert receipt["credential_id"] == "worker-time-reviewer"
    assert receipt["actor"] == "credential-payroll-reviewer"
    assert receipt["action"] == "worker_time.event.decide"
    assert receipt["resource_type"] == "worker_time_event"
    assert receipt["resource_id"] == str(event_id)
    assert receipt["request_id"] == "req-worker-time-001"


def test_operations_scoped_credentials_create_segment_and_receipt(monkeypatch, isolated_db):
    _set_service_credentials(
        monkeypatch,
        [
            {
                "id": "operations-writer",
                "token": "operations-token",
                "actor": "credential-operations-planner",
                "scopes": ["api:read", "operations:write"],
            }
        ],
    )
    with sqlite3.connect(isolated_db) as conn:
        conn.row_factory = sqlite3.Row
        ensure_dashboard_tables(conn)
        job_id = _create_job(conn)
        conn.commit()

    client = TestClient(api.app)
    response = client.post(
        "/operations/segments",
        json={
            "jobId": job_id,
            "segmentSequence": 1,
            "fromLocation": "Depot",
            "toLocation": "Customer",
            "plannedStart": "2026-03-16T08:00:00+00:00",
            "plannedEnd": "2026-03-16T12:00:00+00:00",
        },
        headers={
            "X-Corkysoft-Api-Key": "operations-token",
            "X-Corkysoft-Request-Id": "req-operations-001",
        },
    )

    assert response.status_code == 200
    segment_id = response.json()["segmentId"]
    with sqlite3.connect(isolated_db) as conn:
        conn.row_factory = sqlite3.Row
        receipt = conn.execute(
            """
            SELECT credential_id, actor, action, resource_type, resource_id, request_id
            FROM api_write_receipts
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
    assert receipt["credential_id"] == "operations-writer"
    assert receipt["actor"] == "credential-operations-planner"
    assert receipt["action"] == "operations.segment.ensure"
    assert receipt["resource_type"] == "job_segment"
    assert receipt["resource_id"] == str(segment_id)
    assert receipt["request_id"] == "req-operations-001"


def test_operations_write_wrong_scope_fails_closed(monkeypatch, isolated_db):
    _set_service_credentials(
        monkeypatch,
        [
            {
                "id": "operations-reader",
                "token": "operations-reader-token",
                "actor": "read-only-ops",
                "scopes": ["api:read"],
            }
        ],
    )
    with sqlite3.connect(isolated_db) as conn:
        ensure_dashboard_tables(conn)
        job_id = _create_job(conn)
        conn.commit()

    client = TestClient(api.app)
    response = client.post(
        "/operations/segments",
        json={"jobId": job_id, "segmentSequence": 1},
        headers={"X-Corkysoft-Api-Key": "operations-reader-token"},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Credential scope is not authorized"


def test_operations_cutover_apply_recommendation_route(isolated_db):
    client = TestClient(api.app)

    response = client.put(
        "/operations/cutover/workflows/dispatch_execution",
        json={
            "cutoverStatus": "dual_run",
            "ownerRole": "dispatcher",
            "snapshotMode": "on_demand",
            "snapshotFields": ["jobId"],
            "fallbackMode": "import_only",
            "cutoverTargetPercent": 0.0,
            "nativeReady": True,
            "dualRunComplete": True,
            "fallbackDrillComplete": True,
            "operatorTrained": True,
            "rollbackInstructions": "Rollback",
            "notes": "Ready",
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    response = client.post(
        "/operations/cutover/workflows/dispatch_execution/apply-recommendation",
        json={"actor": "dispatcher", "note": "Promote"},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 400
    assert "approval" in response.json()["detail"].lower()

    response = client.post(
        "/operations/cutover/workflows/dispatch_execution/request-promotion",
        json={"actor": "ops-manager", "note": "Evidence looks good."},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["approval"]["status"] == "requested"

    response = client.post(
        "/operations/cutover/workflows/dispatch_execution/approve-promotion",
        json={"actor": "commercial-owner", "note": "Approved."},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["approval"]["status"] == "approved"

    response = client.post(
        "/operations/cutover/workflows/dispatch_execution/apply-recommendation",
        json={"actor": "dispatcher", "note": "Promote"},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["cutoverStatus"] == "native_primary"
    assert payload["recommendation"]["recommendedStatus"] in {"native_primary", "fallback_only"}


def test_operations_cutover_reject_promotion_route(isolated_db):
    client = TestClient(api.app)

    response = client.put(
        "/operations/cutover/workflows/dispatch_execution",
        json={
            "cutoverStatus": "dual_run",
            "ownerRole": "dispatcher",
            "snapshotMode": "on_demand",
            "snapshotFields": ["jobId"],
            "fallbackMode": "import_only",
            "cutoverTargetPercent": 0.0,
            "nativeReady": True,
            "dualRunComplete": True,
            "fallbackDrillComplete": True,
            "operatorTrained": True,
            "rollbackInstructions": "Rollback",
            "notes": "Ready",
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    response = client.post(
        "/operations/cutover/workflows/dispatch_execution/request-promotion",
        json={"actor": "ops-manager", "note": "Requesting promotion."},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    response = client.post(
        "/operations/cutover/workflows/dispatch_execution/reject-promotion",
        json={"actor": "commercial-owner", "note": "Need one more drill."},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["approval"]["status"] == "rejected"
    assert payload["recommendation"]["blockedByApproval"] is True


def test_kent_ams_importer_returns_summary(isolated_db):
    client = TestClient(api.app)
    response = client.post(
        "/importers/kent-ams/jobs",
        json={
            "records": [
                {"moveId": "KENT-1001", "origin": "Brisbane", "destination": "Cairns"},
                {"moveId": "KENT-1002", "origin": "Sydney", "destination": "Melbourne"},
            ],
            "dry_run": True,
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "resource": "jobs",
        "imported": 2,
        "dry_run": True,
    }


def test_kent_ams_tender_importer_returns_summary(isolated_db):
    client = TestClient(api.app)
    response = client.post(
        "/importers/kent-ams/tenders",
        json={
            "records": [
                {
                    "tenderId": "T-1001",
                    "moveId": "KENT-2001",
                    "expectedRevenue": 18000,
                    "estimatedDistanceKm": 1200,
                    "estimatedVolumeM3": 34,
                    "requiredTrucks": 2,
                    "requiredWorkers": 4,
                },
                {
                    "tenderId": "T-1002",
                    "moveId": "KENT-2002",
                    "expectedRevenue": 6200,
                    "estimatedDistanceKm": 180,
                    "estimatedVolumeM3": 18,
                    "requiredTrucks": 1,
                    "requiredWorkers": 2,
                },
            ],
            "dry_run": True,
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "resource": "tenders",
        "imported": 2,
        "dry_run": True,
    }


def test_get_prioritized_kent_tenders_returns_ranked_rows(isolated_db):
    with sqlite3.connect(isolated_db) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS kent_job_tenders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tender_external_id TEXT NOT NULL UNIQUE,
                job_number TEXT NOT NULL,
                client_name TEXT,
                origin TEXT,
                destination TEXT,
                expected_revenue REAL,
                estimated_cost REAL,
                required_trucks INTEGER,
                required_workers INTEGER,
                due_at TEXT,
                move_date TEXT,
                tender_status TEXT NOT NULL DEFAULT 'open',
                score_total REAL NOT NULL DEFAULT 0,
                score_profitability REAL NOT NULL DEFAULT 0,
                score_capacity REAL NOT NULL DEFAULT 0,
                score_urgency REAL NOT NULL DEFAULT 0,
                score_seasonality REAL NOT NULL DEFAULT 0,
                score_route_location REAL NOT NULL DEFAULT 0,
                score_spare_capacity REAL NOT NULL DEFAULT 0,
                overrideable_flags TEXT NOT NULL DEFAULT '[]',
                hard_block_flags TEXT NOT NULL DEFAULT '[]',
                recommended_action TEXT NOT NULL DEFAULT 'review',
                updated_at TEXT
            );
            """
        )
        conn.execute(
            """
            INSERT INTO kent_job_tenders (
                tender_external_id, job_number, client_name, tender_status,
                score_total, score_profitability, score_capacity, score_urgency,
                score_seasonality, score_route_location, score_spare_capacity,
                overrideable_flags, hard_block_flags, recommended_action, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("T-A", "J-A", "Client A", "open", 91.5, 88.0, 75.0, 95.0, 90.0, 85.0, 92.0, '[]', '[]', "pursue_now", "2026-03-12T00:00:00+00:00"),
        )
        conn.execute(
            """
            INSERT INTO kent_job_tenders (
                tender_external_id, job_number, client_name, tender_status,
                score_total, score_profitability, score_capacity, score_urgency,
                score_seasonality, score_route_location, score_spare_capacity,
                overrideable_flags, hard_block_flags, recommended_action, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("T-B", "J-B", "Client B", "open", 65.0, 62.0, 60.0, 70.0, 55.0, 40.0, 35.0, '["sla_risk_increased"]', '[]', "review_today", "2026-03-12T00:00:00+00:00"),
        )
        conn.commit()

    client = TestClient(api.app)
    response = client.get("/kent-ams/tenders/prioritized?status=open&limit=10", headers=AUTH_HEADERS)
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 2
    assert payload[0]["tenderExternalId"] == "T-A"
    assert payload[1]["tenderExternalId"] == "T-B"
    assert payload[0]["scoreSpareCapacity"] == 92.0
    assert payload[0]["policyMatched"] is False
    assert payload[0]["profitRuleMode"] == "EITHER"
    assert payload[1]["overrideableFlags"] == ["sla_risk_increased"]


def test_get_and_update_kent_tender_config(isolated_db):
    client = TestClient(api.app)

    response = client.get("/kent-ams/config", headers=AUTH_HEADERS)
    assert response.status_code == 200
    payload = response.json()
    assert payload["ruleMode"] == "EITHER"
    assert payload["absoluteMarginThreshold"] == 750.0

    response = client.put(
        "/kent-ams/config",
        json={
            "ruleMode": "BOTH",
            "absoluteMarginThreshold": 1200.0,
            "marginPercentThreshold": 18.0,
            "lossAlertFloor": -250.0,
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ruleMode"] == "BOTH"
    assert payload["absoluteMarginThreshold"] == 1200.0
    assert payload["marginPercentThreshold"] == 18.0
    assert payload["lossAlertFloor"] == -250.0


def test_kent_tender_override_reasons_and_history(isolated_db):
    with sqlite3.connect(isolated_db) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS kent_job_tenders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tender_external_id TEXT NOT NULL UNIQUE,
                job_number TEXT NOT NULL,
                client_name TEXT,
                expected_revenue REAL,
                estimated_cost REAL,
                tender_status TEXT NOT NULL DEFAULT 'open',
                score_total REAL NOT NULL DEFAULT 0,
                score_profitability REAL NOT NULL DEFAULT 0,
                score_capacity REAL NOT NULL DEFAULT 0,
                score_urgency REAL NOT NULL DEFAULT 0,
                score_seasonality REAL NOT NULL DEFAULT 0,
                score_route_location REAL NOT NULL DEFAULT 0,
                score_spare_capacity REAL NOT NULL DEFAULT 0,
                overrideable_flags TEXT NOT NULL DEFAULT '[]',
                hard_block_flags TEXT NOT NULL DEFAULT '[]',
                recommended_action TEXT NOT NULL DEFAULT 'review',
                created_at TEXT,
                updated_at TEXT
            );
            """
        )
        conn.execute(
            """
            INSERT INTO kent_job_tenders (
                tender_external_id,
                job_number,
                client_name,
                expected_revenue,
                estimated_cost,
                tender_status,
                score_total,
                score_profitability,
                score_capacity,
                score_urgency,
                score_seasonality,
                score_route_location,
                score_spare_capacity,
                overrideable_flags,
                hard_block_flags,
                recommended_action,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "T-OVR",
                "JOB-OVR",
                "Kent Client",
                5000.0,
                5200.0,
                "open",
                44.0,
                40.0,
                45.0,
                50.0,
                55.0,
                60.0,
                65.0,
                '["beyond_transfer_rule"]',
                "[]",
                "review_if_strategic",
                "2026-03-12T00:00:00+00:00",
                "2026-03-12T00:00:00+00:00",
            ),
        )
        conn.commit()

    client = TestClient(api.app)

    response = client.get("/kent-ams/override-reasons", headers=AUTH_HEADERS)
    assert response.status_code == 200
    reasons = response.json()
    assert any(row["code"] == "retention" for row in reasons)

    response = client.put(
        "/kent-ams/override-reasons/manual_dispatch",
        json={
            "code": "manual_dispatch",
            "label": "Manual dispatch",
            "description": "Planner-led exception",
            "active": True,
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["code"] == "manual_dispatch"

    response = client.post(
        "/kent-ams/tenders/T-OVR/override",
        json={
            "action": "pursue",
            "operatorId": "ops-1",
            "reasonCode": "retention",
            "note": "Needed for customer retention",
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["tenderExternalId"] == "T-OVR"
    assert payload["reasonCode"] == "retention"
    assert payload["policyMatched"] is False
    assert payload["lossAlert"] is True

    response = client.get("/kent-ams/tenders/T-OVR/overrides", headers=AUTH_HEADERS)
    assert response.status_code == 200
    history = response.json()
    assert len(history) == 1
    assert history[0]["operatorId"] == "ops-1"
    assert history[0]["reasonCode"] == "retention"


def test_get_kent_tender_calibration_returns_band_metrics(isolated_db):
    with sqlite3.connect(isolated_db) as conn:
        conn.executescript(
            """
            ALTER TABLE jobs ADD COLUMN job_number TEXT;
            CREATE TABLE IF NOT EXISTS kent_job_tenders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tender_external_id TEXT NOT NULL UNIQUE,
                job_number TEXT NOT NULL,
                expected_revenue REAL,
                estimated_cost REAL,
                tender_status TEXT NOT NULL DEFAULT 'open',
                score_total REAL NOT NULL DEFAULT 0,
                created_at TEXT,
                updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS kent_job_awards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                award_external_id TEXT NOT NULL UNIQUE,
                job_number TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            INSERT INTO jobs (job_number, revenue_total, final_cost, origin, destination)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("J-A", 15000.0, 11000.0, "Brisbane", "Cairns"),
        )
        conn.execute(
            """
            INSERT INTO jobs (job_number, revenue_total, final_cost, origin, destination)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("J-B", 9000.0, 7600.0, "Sydney", "Melbourne"),
        )
        conn.execute(
            """
            INSERT INTO kent_job_tenders (
                tender_external_id, job_number, expected_revenue, estimated_cost,
                tender_status, score_total, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("T-A", "J-A", 14000.0, 10000.0, "awarded", 92.0, "2026-03-11T00:00:00+00:00", "2026-03-11T00:00:00+00:00"),
        )
        conn.execute(
            """
            INSERT INTO kent_job_tenders (
                tender_external_id, job_number, expected_revenue, estimated_cost,
                tender_status, score_total, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("T-B", "J-B", 8000.0, 7000.0, "open", 68.0, "2026-03-11T00:00:00+00:00", "2026-03-11T00:00:00+00:00"),
        )
        conn.execute(
            """
            INSERT INTO kent_job_awards (award_external_id, job_number)
            VALUES (?, ?)
            """,
            ("A-100", "J-A"),
        )
        conn.commit()

    client = TestClient(api.app)
    response = client.get("/kent-ams/tenders/calibration?lookback_days=365", headers=AUTH_HEADERS)
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["tenders"] == 2
    assert payload["summary"]["wins"] == 1
    assert payload["summary"]["overallWinRate"] == 0.5
    top_band = next(row for row in payload["bands"] if row["scoreBand"] == "90-100")
    assert top_band["tenders"] == 1
    assert top_band["wins"] == 1


def test_kent_dry_run_does_not_persist_schema_or_config(isolated_db):
    client = TestClient(api.app)
    response = client.post(
        "/importers/kent-ams/tenders",
        json={
            "records": [
                {
                    "tenderId": "T-DRY-1",
                    "moveId": "JOB-DRY-1",
                    "expectedRevenue": 4000,
                    "estimatedCost": 3500,
                }
            ],
            "dry_run": True,
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200

    with sqlite3.connect(isolated_db) as conn:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='kent_job_tenders'"
        ).fetchone()
        assert row is None
        params = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='global_parameters'"
        ).fetchone()
        assert params is None


def test_prioritized_kent_tenders_ignores_unknown_hard_block_flags(isolated_db):
    with sqlite3.connect(isolated_db) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS kent_job_tenders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tender_external_id TEXT NOT NULL UNIQUE,
                job_number TEXT NOT NULL,
                client_name TEXT,
                origin TEXT,
                destination TEXT,
                expected_revenue REAL,
                estimated_cost REAL,
                required_trucks INTEGER,
                required_workers INTEGER,
                due_at TEXT,
                move_date TEXT,
                tender_status TEXT NOT NULL DEFAULT 'open',
                score_total REAL NOT NULL DEFAULT 0,
                score_profitability REAL NOT NULL DEFAULT 0,
                score_capacity REAL NOT NULL DEFAULT 0,
                score_urgency REAL NOT NULL DEFAULT 0,
                score_seasonality REAL NOT NULL DEFAULT 0,
                score_route_location REAL NOT NULL DEFAULT 0,
                score_spare_capacity REAL NOT NULL DEFAULT 0,
                overrideable_flags TEXT NOT NULL DEFAULT '[]',
                hard_block_flags TEXT NOT NULL DEFAULT '[]',
                recommended_action TEXT NOT NULL DEFAULT 'review',
                created_at TEXT,
                updated_at TEXT
            );
            """
        )
        conn.execute(
            """
            INSERT INTO kent_job_tenders (
                tender_external_id, job_number, client_name, expected_revenue, estimated_cost,
                tender_status, score_total, score_profitability, score_capacity, score_urgency,
                score_seasonality, score_route_location, score_spare_capacity,
                overrideable_flags, hard_block_flags, recommended_action, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "T-HARD",
                "JOB-HARD",
                "Client Hard",
                2000.0,
                1500.0,
                "open",
                80.0,
                75.0,
                75.0,
                75.0,
                75.0,
                75.0,
                75.0,
                "[]",
                '["not_a_real_hard_block"]',
                "pursue_now",
                "2026-03-12T00:00:00+00:00",
                "2026-03-12T00:00:00+00:00",
            ),
        )
        conn.commit()

    client = TestClient(api.app)
    response = client.get("/kent-ams/tenders/prioritized?status=open&limit=10", headers=AUTH_HEADERS)
    assert response.status_code == 200
    payload = response.json()
    assert payload[0]["hardBlockFlags"] == []


def test_prioritized_kent_tenders_returns_true_top_n_after_sort(isolated_db):
    with sqlite3.connect(isolated_db) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS kent_job_tenders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tender_external_id TEXT NOT NULL UNIQUE,
                job_number TEXT NOT NULL,
                client_name TEXT,
                origin TEXT,
                destination TEXT,
                expected_revenue REAL,
                estimated_cost REAL,
                required_trucks INTEGER,
                required_workers INTEGER,
                due_at TEXT,
                move_date TEXT,
                tender_status TEXT NOT NULL DEFAULT 'open',
                score_total REAL NOT NULL DEFAULT 0,
                score_profitability REAL NOT NULL DEFAULT 0,
                score_capacity REAL NOT NULL DEFAULT 0,
                score_urgency REAL NOT NULL DEFAULT 0,
                score_seasonality REAL NOT NULL DEFAULT 0,
                score_route_location REAL NOT NULL DEFAULT 0,
                score_spare_capacity REAL NOT NULL DEFAULT 0,
                overrideable_flags TEXT NOT NULL DEFAULT '[]',
                hard_block_flags TEXT NOT NULL DEFAULT '[]',
                recommended_action TEXT NOT NULL DEFAULT 'review',
                created_at TEXT,
                updated_at TEXT
            );
            """
        )
        for idx in range(10):
            conn.execute(
                """
                INSERT INTO kent_job_tenders (
                    tender_external_id, job_number, client_name, expected_revenue, estimated_cost,
                    tender_status, score_total, score_profitability, score_capacity, score_urgency,
                    score_seasonality, score_route_location, score_spare_capacity,
                    overrideable_flags, hard_block_flags, recommended_action, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"T-LOW-{idx}",
                    f"JOB-LOW-{idx}",
                    "Low Client",
                    1000.0,
                    950.0,
                    "open",
                    20.0 + idx,
                    20.0,
                    20.0,
                    20.0,
                    20.0,
                    20.0,
                    20.0,
                    "[]",
                    "[]",
                    "defer",
                    "2026-03-12T00:00:00+00:00",
                    "2026-03-12T00:00:00+00:00",
                ),
            )
        conn.execute(
            """
            INSERT INTO kent_job_tenders (
                tender_external_id, job_number, client_name, expected_revenue, estimated_cost,
                tender_status, score_total, score_profitability, score_capacity, score_urgency,
                score_seasonality, score_route_location, score_spare_capacity,
                overrideable_flags, hard_block_flags, recommended_action, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "T-TOP",
                "JOB-TOP",
                "Top Client",
                5000.0,
                3000.0,
                "open",
                95.0,
                95.0,
                95.0,
                95.0,
                95.0,
                95.0,
                95.0,
                "[]",
                "[]",
                "pursue_now",
                "2026-03-12T00:00:00+00:00",
                "2026-03-12T00:00:00+00:00",
            ),
        )
        conn.commit()

    client = TestClient(api.app)
    response = client.get("/kent-ams/tenders/prioritized?status=open&limit=1", headers=AUTH_HEADERS)
    assert response.status_code == 200
    payload = response.json()
    assert payload[0]["tenderExternalId"] == "T-TOP"


def test_call_event_routes_create_notes_actions_and_worker_time(isolated_db):
    with sqlite3.connect(isolated_db) as conn:
        conn.row_factory = sqlite3.Row
        ensure_dashboard_tables(conn)
        job_id = _create_job(conn)
        worker = upsert_worker(conn, name="Call Worker", phone="0400111222")
        conn.commit()

    client = TestClient(api.app)

    response = client.post(
        "/calls/events",
        json={
            "eventKind": "client_call",
            "direction": "inbound",
            "sourceChannel": "telephony",
            "callerPhone": "0400 111 222",
            "jobId": job_id,
            "title": "Client asking about dock access",
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    call_event = response.json()
    assert call_event["clientId"] is not None
    assert call_event["jobId"] == job_id

    response = client.post(
        f"/calls/events/{call_event['id']}/notes",
        json={"author": "ops-1", "noteText": "Use dock two after 9am."},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["authoritative"] is True

    response = client.post(
        f"/calls/events/{call_event['id']}/extracted-actions",
        json={"actionText": "Notify crew about dock two.", "sourceEngine": "statibaker"},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    action_id = response.json()["id"]

    response = client.post(
        f"/calls/extracted-actions/{action_id}/decision",
        json={"status": "accepted", "decidedBy": "ops-1"},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["status"] == "accepted"

    response = client.post(
        "/worker-time/events",
        json={
            "callEventId": call_event["id"],
            "eventType": "clock_on",
            "channel": "voice_call",
            "callerPhone": "0400 111 222",
            "effectiveTimestamp": "2026-03-13T06:30:00+10:00",
            "confidence": 0.95,
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["reviewStatus"] == "accepted"
    assert response.json()["workerId"] == int(worker["id"])

    response = client.get("/state-egress/events?limit=20", headers=AUTH_HEADERS)
    assert response.status_code == 200
    event_types = {row["eventType"] for row in response.json()}
    assert "call_event_created" in event_types
    assert "call_note_added" in event_types
    assert "extracted_action_decided" in event_types
    assert "worker_time_capture_recorded" in event_types


def test_call_session_and_ambient_routes(isolated_db):
    client = TestClient(api.app)

    response = client.post(
        "/calls/sessions",
        json={
            "eventKind": "client_call",
            "direction": "inbound",
            "sourceChannel": "telephony",
            "callerPhone": "0400 222 333",
            "title": "Client routed to operator",
            "initialDestinationKind": "operator",
            "initialDestinationLabel": "desk-1",
            "operatorId": "ops-1",
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    session = response.json()
    assert session["rootCallEventId"] is not None

    response = client.get(f"/calls/sessions/{session['id']}/legs", headers=AUTH_HEADERS)
    assert response.status_code == 200
    legs = response.json()
    assert len(legs) == 1
    initial_leg = legs[0]

    response = client.post(
        f"/calls/legs/{initial_leg['id']}/transcripts/fake",
        json={"scenario": "Client explains access constraints.", "operatorGoal": "Record the constraint and notify crew."},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200

    assert response.json()["callLegId"] == initial_leg["id"]

    response = client.post(
        f"/calls/sessions/{session['id']}/legs",
        json={
            "legKind": "consult",
            "direction": "internal",
            "status": "active",
            "sourceChannel": "telephony",
            "destinationKind": "manager",
            "destinationLabel": "boss",
            "operatorId": "mgr-1",
            "answeredAt": "2026-03-13T00:00:00+00:00",
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["legKind"] == "consult"

    response = client.get(f"/calls/sessions/{session['id']}/routing-events", headers=AUTH_HEADERS)
    assert response.status_code == 200
    event_types = {row["eventType"] for row in response.json()}
    assert "call_received" in event_types
    assert "call_routed" in event_types

    response = client.post(
        "/calls/ambient-sessions",
        json={
            "title": "Office coordination",
            "sourceLocation": "Brisbane office",
            "teamLabel": "Ops desk",
            "status": "active",
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    ambient = response.json()

    response = client.post(
        f"/calls/ambient-sessions/{ambient['id']}/transcripts/fake",
        json={"scenario": "Operator and manager discuss revised site instructions."},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["ambientSessionId"] == ambient["id"]



def test_call_transcription_upload_and_poll_routes(monkeypatch, isolated_db):
    client = TestClient(api.app)
    response = client.post(
        "/calls/events",
        json={"eventKind": "ops_call", "direction": "internal", "sourceChannel": "imported_recording"},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    call_event_id = response.json()["id"]

    monkeypatch.setattr(
        api,
        "submit_call_audio_for_transcription",
        lambda conn, **kwargs: {
            "id": 1,
            "callEventId": call_event_id,
            "serviceKey": "ops",
            "externalTaskId": "task-xyz",
            "status": "queued",
            "transcriptText": None,
            "transcriptSegments": [],
            "diarization": [],
            "confidence": None,
            "isFinal": False,
            "errorMessage": None,
            "dataClassification": "observer_capture_transcript",
            "authorityClass": "observer_capture_ref",
            "failureKind": None,
            "createdAt": "2026-03-13T00:00:00+00:00",
            "updatedAt": "2026-03-13T00:00:00+00:00",
        },
    )
    monkeypatch.setattr(
        api,
        "poll_transcript_artifact",
        lambda conn, artifact_id: {
            "id": artifact_id,
            "callEventId": call_event_id,
            "serviceKey": "ops",
            "externalTaskId": "task-xyz",
            "status": "completed",
            "transcriptText": "Manager called with updated instructions.",
            "transcriptSegments": [{"text": "Manager called with updated instructions."}],
            "diarization": [],
            "confidence": 0.92,
            "isFinal": True,
            "errorMessage": None,
            "dataClassification": "observer_capture_transcript",
            "authorityClass": "observer_capture_ref",
            "failureKind": None,
            "createdAt": "2026-03-13T00:00:00+00:00",
            "updatedAt": "2026-03-13T00:05:00+00:00",
        },
    )

    response = client.post(
        f"/calls/events/{call_event_id}/transcripts/upload",
        json={
            "serviceKey": "ops",
            "filename": "call.wav",
            "contentBase64": "ZmFrZS1hdWRpbw==",
            "diarize": True,
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    artifact_id = response.json()["id"]
    assert response.json()["externalTaskId"] == "task-xyz"

    response = client.post(
        f"/calls/transcripts/{artifact_id}/poll",
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert "updated instructions" in response.json()["transcriptText"]
    assert response.json()["dataClassification"] == "observer_capture_transcript"
    assert response.json()["authorityClass"] == "observer_capture_ref"


def test_call_transcription_upload_rejects_invalid_base64(isolated_db):
    client = TestClient(api.app)
    response = client.post(
        "/calls/events",
        json={
            "eventKind": "ops_call",
            "direction": "internal",
            "sourceChannel": "imported_recording",
        },
        headers=AUTH_HEADERS,
    )
    call_event_id = response.json()["id"]

    response = client.post(
        f"/calls/events/{call_event_id}/transcripts/upload",
        json={
            "serviceKey": "ops",
            "filename": "call.wav",
            "contentBase64": "not valid base64",
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "contentBase64 must contain valid base64 data"


def test_call_transcription_upload_rejects_empty_audio(isolated_db):
    client = TestClient(api.app)
    response = client.post(
        "/calls/events",
        json={
            "eventKind": "ops_call",
            "direction": "internal",
            "sourceChannel": "imported_recording",
        },
        headers=AUTH_HEADERS,
    )
    call_event_id = response.json()["id"]

    response = client.post(
        f"/calls/events/{call_event_id}/transcripts/upload",
        json={
            "serviceKey": "ops",
            "filename": "call.wav",
            "contentBase64": "",
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "contentBase64 is required"


def test_call_transcription_upload_rejects_unsupported_extension(isolated_db):
    client = TestClient(api.app)
    response = client.post(
        "/calls/events",
        json={
            "eventKind": "ops_call",
            "direction": "internal",
            "sourceChannel": "imported_recording",
        },
        headers=AUTH_HEADERS,
    )
    call_event_id = response.json()["id"]

    response = client.post(
        f"/calls/events/{call_event_id}/transcripts/upload",
        json={
            "serviceKey": "ops",
            "filename": "call.txt",
            "contentBase64": "ZmFrZS1hdWRpbw==",
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "filename must use one of: m4a, mp3, mp4, ogg, wav"


def test_call_leg_transcription_upload_maps_adapter_error(monkeypatch, isolated_db):
    client = TestClient(api.app)
    session = client.post(
        "/calls/sessions",
        json={
            "eventKind": "ops_call",
            "direction": "internal",
            "sourceChannel": "imported_recording",
        },
        headers=AUTH_HEADERS,
    ).json()
    leg = client.post(
        f"/calls/sessions/{session['id']}/legs",
        json={"legKind": "primary", "direction": "inbound"},
        headers=AUTH_HEADERS,
    ).json()

    def fail_submitter(conn, **kwargs):
        raise WhisperXAdapterError("backend down")

    monkeypatch.setattr(api, "submit_call_audio_for_transcription", fail_submitter)

    response = client.post(
        f"/calls/legs/{leg['id']}/transcripts/upload",
        json={
            "serviceKey": "ops",
            "filename": "call.wav",
            "contentBase64": "ZmFrZS1hdWRpbw==",
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 502
    assert response.json()["detail"] == "backend down"


def test_fake_call_transcript_route(isolated_db):
    client = TestClient(api.app)
    response = client.post(
        "/calls/events",
        json={
            "eventKind": "worker_call",
            "direction": "internal",
            "sourceChannel": "manual_note",
            "title": "Worker needs direction",
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    call_event_id = response.json()["id"]

    response = client.post(
        f"/calls/events/{call_event_id}/transcripts/fake",
        json={
            "scenario": "Worker reports site conditions are different from the plan.",
            "operatorGoal": "Escalate to manager and confirm the revised unloading sequence.",
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert "different from the plan" in (payload["transcriptText"] or "")


def test_worker_time_duplicate_event_stays_pending_review(isolated_db):
    with sqlite3.connect(isolated_db) as conn:
        conn.row_factory = sqlite3.Row
        ensure_dashboard_tables(conn)
        worker = upsert_worker(conn, name="Dup Worker", phone="0400999888")
        conn.commit()

    client = TestClient(api.app)
    payload = {
        "workerId": int(worker["id"]),
        "eventType": "clock_on",
        "channel": "voice_call",
        "effectiveTimestamp": "2026-03-13T06:30:00+10:00",
        "confidence": 0.95,
    }
    first = client.post("/worker-time/events", json=payload, headers=AUTH_HEADERS)
    assert first.status_code == 200
    assert first.json()["reviewStatus"] == "accepted"

    duplicate = client.post("/worker-time/events", json=payload, headers=AUTH_HEADERS)
    assert duplicate.status_code == 200
    duplicate_payload = duplicate.json()
    assert duplicate_payload["reviewStatus"] == "pending_review"
    assert "duplicate_event" in duplicate_payload["rawPayload"]["anomalyFlags"]


def test_call_link_correction_and_leg_transcript_routes(isolated_db):
    with sqlite3.connect(isolated_db) as conn:
        conn.row_factory = sqlite3.Row
        ensure_dashboard_tables(conn)
        first_job = _create_job(conn)
        second_job = _create_job(conn)
        conn.commit()

    client = TestClient(api.app)
    session = client.post(
        "/calls/sessions",
        json={
            "eventKind": "client_call",
            "direction": "inbound",
            "sourceChannel": "telephony",
            "callerPhone": "0400111222",
            "title": "Correction flow",
            "initialDestinationKind": "operator",
            "initialDestinationLabel": "desk-1",
        },
        headers=AUTH_HEADERS,
    ).json()

    response = client.post(
        f"/calls/events/{session['rootCallEventId']}/resolve",
        json={"actor": "ops-1", "jobId": first_job, "resolutionNote": "First link"},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    response = client.post(
        f"/calls/events/{session['rootCallEventId']}/resolve",
        json={"actor": "ops-1", "jobId": second_job, "resolutionNote": "Corrected link"},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["jobId"] == second_job

    legs = client.get(f"/calls/sessions/{session['id']}/legs", headers=AUTH_HEADERS).json()
    leg_id = legs[0]["id"]
    transcript = client.post(
        f"/calls/legs/{leg_id}/transcripts/fake",
        json={"scenario": "Client restates the access issue after routing."},
        headers=AUTH_HEADERS,
    )
    assert transcript.status_code == 200
    assert transcript.json()["callLegId"] == leg_id


def test_state_egress_combines_observer_and_call_ops_streams(isolated_db):
    with sqlite3.connect(isolated_db) as conn:
        ensure_dashboard_tables(conn)
        job_id = _create_job(conn)
        upsert_operations_diary_task(
            conn,
            job_id=job_id,
            task_date="2018-08-30",
            title="Review invoice release",
            actor_ref="ops-manager",
        )
        record_worker_time_capture_event(
            conn,
            event_type="clock_on",
            channel="voice_call",
            worker_name_raw="Alex Planner",
            caller_phone="0400 222 333",
            effective_timestamp="2018-08-30T06:30:00+10:00",
            job_id=job_id,
            confidence=0.9,
        )
        conn.commit()

    client = TestClient(api.app)
    response = client.get(f"/state-egress/events?surface=observer&jobId={job_id}&limit=20", headers=AUTH_HEADERS)
    assert response.status_code == 200
    payload = response.json()
    assert any(row["eventFamily"] == "diary_task_event" for row in payload)
    assert all(row["surface"] == "observer" for row in payload)

    response = client.get("/state-egress/events?surface=all&limit=20", headers=AUTH_HEADERS)
    assert response.status_code == 200
    surfaces = {row["surface"] for row in response.json()}
    assert "observer" in surfaces
    assert "call_ops" in surfaces


def test_operations_diary_export_endpoint_emits_snapshot_and_exceptions(isolated_db):
    with sqlite3.connect(isolated_db) as conn:
        ensure_dashboard_tables(conn)
        job_id = _create_job(conn)
        supplier_id = conn.execute(
            """
            INSERT INTO suppliers (company_name, created_at, updated_at)
            VALUES (?, ?, ?)
            """,
            ("Holiday Carrier", "2018-09-01T12:30:00+10:00", "2018-09-01T12:30:00+10:00"),
        ).lastrowid
        upsert_customer_invoice_review(
            conn,
            job_id=job_id,
            invoice_status="reconciliation_warning",
            reviewed_by="ops-manager",
        )
        upsert_subcontractor_bill_review(
            conn,
            job_id=job_id,
            supplier_id=int(supplier_id),
            bill_status="bill_exception",
            bill_reference="BILL-99",
            bill_date="2018-08-31",
            amount=500.0,
            reviewed_by="ops-manager",
        )
        conn.commit()

    client = TestClient(api.app)
    response = client.post(
        "/state-egress/operations-diary-export",
        json={
            "anchorDate": "2018-08-30",
            "viewMode": "day",
            "actorRef": "ops-manager",
            "includePlanningSnapshot": True,
            "includeReconciliationExceptions": True,
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["byFamily"]["planning_snapshot"] == 1
    assert payload["byFamily"]["reconciliation_exception"] >= 1
