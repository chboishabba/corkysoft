from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.db import ensure_dashboard_tables, upsert_truck, upsert_vehicle_details, upsert_worker
from analytics.db.inventory import (
    allocate_inventory_to_segment,
    decide_inventory_substitution,
    request_inventory_substitution,
    upsert_inventory_item,
    upsert_inventory_requirement,
)
from analytics.operational_signals import upsert_job_operational_signal
from analytics.operations_assignment import (
    DISPATCH_SHARE_ACTION_STATUSES,
    approve_operations_cutover_promotion,
    apply_operations_cutover_recommendation,
    assign_worker_compliance,
    assign_worker_role,
    assign_segment_resources,
    ensure_segment,
    ensure_worker_compliance,
    ensure_worker_role,
    evaluate_segment_readiness,
    list_dispatch_share_actions,
    list_job_operations_board,
    list_labor_reconciliation,
    list_operations_cutover_events,
    list_operations_cutover_rollout,
    list_operations_cutover_workflows,
    list_operational_readiness_items,
    list_operational_share_opportunities,
    list_operational_conflicts,
    list_planned_labor_assignments,
    list_segments_for_truck,
    list_segments_for_worker,
    list_segment_readiness,
    list_truck_assignment_summary,
    list_worker_assignment_summary,
    reject_operations_cutover_promotion,
    record_dispatch_share_action,
    record_operations_cutover_event,
    request_operations_cutover_promotion,
    upsert_operations_cutover_workflow,
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


def test_operational_readiness_items_cover_vehicle_and_worker_alerts() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)

    upsert_truck(conn, truck_id="TRK-4", name="Truck 4")
    upsert_vehicle_details(
        conn,
        truck_id="TRK-4",
        rego="TRK-4",
        rego_expiry="2000-01-01",
        coi_due="2099-12-31",
        next_service="2000-02-01",
        daily_check_complete=True,
        source_imported_at="2026-03-12T00:00:00+00:00",
    )
    worker = upsert_worker(conn, name="Casey Compliance")
    compliance_id = ensure_worker_compliance(conn, name="MSIC")
    role_id = ensure_worker_role(conn, name="Driver")
    assign_worker_role(conn, worker_id=int(worker["id"]), role_id=role_id)
    assign_worker_compliance(
        conn,
        worker_id=int(worker["id"]),
        compliance_id=compliance_id,
        expiry_date="2000-01-01",
    )

    readiness_items = list_operational_readiness_items(conn)
    assert any(
        item["resourceType"] == "vehicle"
        and item["resourceId"] == "TRK-4"
        and item["ruleType"] == "rego"
        and item["status"] == "blocked"
        for item in readiness_items
    )
    assert any(
        item["resourceType"] == "worker"
        and item["resourceId"] == str(worker["id"])
        and item["ruleType"] == "compliance"
        and item["status"] == "blocked"
        for item in readiness_items
    )


def test_planned_labor_and_reconciliation_compare_with_imported_shifts() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)

    upsert_truck(conn, truck_id="TRK-5", name="Truck 5")
    upsert_vehicle_details(
        conn,
        truck_id="TRK-5",
        rego="TRK-5",
        rego_expiry="2099-12-31",
        coi_due="2099-12-31",
        next_service="2099-12-31",
        daily_check_complete=True,
    )
    worker = upsert_worker(conn, name="Roster Worker")
    job_id = _job(conn, "Client E", "Brisbane", "Roma")
    segment = ensure_segment(
        conn,
        job_id=job_id,
        segment_sequence=1,
        planned_start="2026-03-13T08:00:00+00:00",
        planned_end="2026-03-13T12:00:00+00:00",
    )
    assign_segment_resources(
        conn,
        segment_id=int(segment["id"]),
        truck_ids=["TRK-5"],
        worker_assignments=[{"workerId": int(worker["id"])}],
    )

    planned = list_planned_labor_assignments(conn, start_date="2026-03-13", end_date="2026-03-13")
    assert len(planned) == 1
    assert planned[0]["workerName"] == "Roster Worker"
    assert planned[0]["truckIds"] == ["TRK-5"]

    conn.execute(
        """
        INSERT INTO driver_shifts (
            shift_date, truck_id, worker_id, shift_window_start, shift_window_end, source, imported_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("2026-03-13", "TRK-5", int(worker["id"]), "08:00", "12:00", "VEHICLE_DRIVER", "2026-03-13T00:00:00+00:00"),
    )
    conn.execute(
        """
        INSERT INTO driver_shifts (
            shift_date, truck_id, worker_id, shift_window_start, shift_window_end, source, imported_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("2026-03-13", "TRK-X", None, "09:00", "17:00", "VEHICLE_DRIVER", "2026-03-13T00:00:00+00:00"),
    )
    conn.commit()

    reconciliation = list_labor_reconciliation(conn, start_date="2026-03-13", end_date="2026-03-13")
    assert any(
        row["status"] == "matched"
        and row["workerName"] == "Roster Worker"
        and row["jobId"] == job_id
        for row in reconciliation
    )
    assert any(
        row["status"] == "imported_only"
        and row["truckIds"] == ["TRK-X"]
        for row in reconciliation
    )


def test_job_operations_board_rolls_up_segments_labor_and_inventory() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)

    supplier = conn.execute(
        """
        INSERT INTO suppliers (company_name, created_at, updated_at)
        VALUES (?, ?, ?)
        """,
        ("Board Supplier", "2026-03-12T00:00:00+00:00", "2026-03-12T00:00:00+00:00"),
    ).lastrowid
    item = conn.execute(
        """
        INSERT INTO inventory_items (name, quantity, unit, supplier_id, updated_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("Blankets", 5, "ea", supplier, "2026-03-12T00:00:00+00:00"),
    ).lastrowid
    upsert_truck(conn, truck_id="TRK-BOARD", name="Board Truck")
    upsert_vehicle_details(
        conn,
        truck_id="TRK-BOARD",
        rego="TRK-BOARD",
        rego_expiry="2099-12-31",
        coi_due="2099-12-31",
        next_service="2099-12-31",
        daily_check_complete=True,
    )
    worker = upsert_worker(conn, name="Board Worker")
    job_id = _job(conn, "Board Client", "Alpha", "Beta")
    segment = ensure_segment(
        conn,
        job_id=job_id,
        segment_sequence=1,
        planned_start="2026-03-16T08:00:00+00:00",
        planned_end="2026-03-16T12:00:00+00:00",
    )
    assign_segment_resources(
        conn,
        segment_id=int(segment["id"]),
        truck_ids=["TRK-BOARD"],
        worker_assignments=[{"workerId": int(worker["id"])}],
    )
    from analytics.db.inventory import allocate_inventory_to_segment

    allocate_inventory_to_segment(
        conn,
        segment_id=int(segment["id"]),
        inventory_item_id=int(item),
        quantity=2,
        status="staged",
    )

    rows = list_job_operations_board(conn, job_id=job_id)
    assert len(rows) == 1
    row = rows[0]
    assert row["jobId"] == job_id
    assert row["jobStatus"] == "planned"
    assert row["truckIds"] == ["TRK-BOARD"]
    assert row["workerNames"] == ["Board Worker"]
    assert row["inventoryNames"] == ["Blankets"]
    assert row["supplierNames"] == ["Board Supplier"]
    assert row["segments"][0]["shipmentCount"] == 1


def test_inventory_shortage_blocks_segment_readiness_when_not_substitutable() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)

    from analytics.db.inventory import upsert_inventory_item, upsert_inventory_requirement

    item = upsert_inventory_item(
        conn,
        name="Container Pod",
        quantity=2,
        architecture="container",
    )
    job_id = _job(conn, "Container Client", "Depot", "Site F")
    segment = ensure_segment(
        conn,
        job_id=job_id,
        segment_sequence=1,
        planned_start="2026-03-17T08:00:00+00:00",
        planned_end="2026-03-17T12:00:00+00:00",
    )
    upsert_inventory_requirement(
        conn,
        job_id=job_id,
        segment_id=int(segment["id"]),
        inventory_item_id=int(item["id"]),
        requirement_name="Container Pod",
        required_quantity=4,
        substitution_allowed=False,
        architecture="container",
    )

    readiness = evaluate_segment_readiness(conn, segment_id=int(segment["id"]))
    assert any("segment:inventory_shortage:Container Pod" in flag for flag in readiness["blockingFlags"])
    assert readiness["inventoryShortages"][0]["shortageQuantity"] == 4.0
    assert readiness["assignmentStatus"] == "blocked"


def test_substitutable_inventory_shortage_requires_override_not_block() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)

    from analytics.db.inventory import upsert_inventory_item, upsert_inventory_requirement

    item = upsert_inventory_item(
        conn,
        name="Moving Blanket",
        quantity=1,
        architecture="consumable",
    )
    job_id = _job(conn, "Blanket Client", "Depot", "Site G")
    segment = ensure_segment(
        conn,
        job_id=job_id,
        segment_sequence=1,
        planned_start="2026-03-18T08:00:00+00:00",
        planned_end="2026-03-18T12:00:00+00:00",
    )
    upsert_inventory_requirement(
        conn,
        job_id=job_id,
        segment_id=int(segment["id"]),
        inventory_item_id=int(item["id"]),
        requirement_name="Moving Blanket",
        required_quantity=3,
        substitution_allowed=True,
        architecture="consumable",
    )

    readiness = evaluate_segment_readiness(conn, segment_id=int(segment["id"]))
    assert readiness["blockingFlags"] == []
    assert any("segment:inventory_shortage:Moving Blanket" in flag for flag in readiness["overrideableFlags"])
    assert readiness["overrideRequired"] is True


def test_job_operations_board_rolls_up_inventory_shortages() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)

    from analytics.db.inventory import upsert_inventory_item, upsert_inventory_requirement

    item = upsert_inventory_item(
        conn,
        name="Packing Crate",
        quantity=10,
        architecture="container",
    )
    job_id = _job(conn, "Ops Client", "Depot", "Site H")
    segment = ensure_segment(
        conn,
        job_id=job_id,
        segment_sequence=1,
        planned_start="2026-03-19T08:00:00+00:00",
        planned_end="2026-03-19T12:00:00+00:00",
    )
    upsert_inventory_requirement(
        conn,
        job_id=job_id,
        segment_id=int(segment["id"]),
        inventory_item_id=int(item["id"]),
        requirement_name="Packing Crate",
        required_quantity=6,
        substitution_allowed=False,
        architecture="container",
    )
    from analytics.db.inventory import allocate_inventory_to_segment
    allocate_inventory_to_segment(
        conn,
        segment_id=int(segment["id"]),
        inventory_item_id=int(item["id"]),
        quantity=2,
    )

    board = list_job_operations_board(conn, job_id=job_id)
    assert len(board) == 1
    row = board[0]
    assert row["requiredQuantity"] == 6.0
    assert row["allocatedQuantity"] == 2.0
    assert row["shortageQuantity"] == 4.0
    assert row["inventoryShortageCount"] == 1
    assert row["segments"][0]["shortageQuantity"] == 4.0


def test_job_operations_board_includes_spare_capacity_and_container_rollups() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)
    conn.execute("ALTER TABLE jobs ADD COLUMN job_number TEXT")

    item = upsert_inventory_item(
        conn,
        name="Container Pod",
        quantity=5,
        architecture="container",
    )
    job_number = "JOB-OPS-1"
    job_id = _job(conn, "Ops Signal Client", "Depot", "Site Z")
    conn.execute("UPDATE jobs SET job_number = ? WHERE id = ?", (job_number, job_id))
    segment = ensure_segment(
        conn,
        job_id=job_id,
        segment_sequence=1,
        planned_start="2026-03-22T08:00:00+00:00",
        planned_end="2026-03-22T12:00:00+00:00",
    )
    upsert_inventory_requirement(
        conn,
        job_id=job_id,
        segment_id=int(segment["id"]),
        inventory_item_id=int(item["id"]),
        requirement_name="Container Pod",
        required_quantity=3,
        substitution_allowed=False,
        architecture="container",
    )
    allocate_inventory_to_segment(
        conn,
        segment_id=int(segment["id"]),
        inventory_item_id=int(item["id"]),
        quantity=1,
    )
    upsert_job_operational_signal(
        conn,
        job_number=job_number,
        origin="Depot",
        destination="Site Z",
        estimated_volume_m3=18.0,
        source="planning",
    )

    board = list_job_operations_board(conn, job_id=job_id)

    assert len(board) == 1
    row = board[0]
    assert row["jobNumber"] == job_number
    assert row["spareCapacityLabel"] == "constrained"
    assert row["spareCapacityScore"] == 40.0
    assert row["operationalSignalSource"] == "planning"
    assert row["containerRequirementCount"] == 1
    assert row["containerRequiredQuantity"] == 3.0
    assert row["containerAllocatedQuantity"] == 1.0
    assert row["containerShortageQuantity"] == 2.0
    assert row["segments"][0]["containerRequirementCount"] == 1
    assert row["segments"][0]["containerShortageQuantity"] == 2.0


def test_operational_share_opportunities_flag_container_reallocation() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)
    conn.execute("ALTER TABLE jobs ADD COLUMN job_number TEXT")

    upsert_truck(conn, truck_id="TRK-SIGNAL", name="Signal Truck", capacity_m3=50.0)
    job_signal_id = _job(conn, "Signal Load", "Depot", "Site Q")
    conn.execute("UPDATE jobs SET job_number = ? WHERE id = ?", ("SIG-1", job_signal_id))
    signal_segment = ensure_segment(
        conn,
        job_id=job_signal_id,
        segment_sequence=1,
        from_location="Depot",
        to_location="Site Q",
        planned_start="2026-03-21T08:00:00+00:00",
        planned_end="2026-03-21T12:00:00+00:00",
    )
    allocate_inventory_to_segment(
        conn,
        segment_id=int(signal_segment["id"]),
        inventory_item_id=int(
            upsert_inventory_item(conn, name="Signal Container", quantity=5, architecture="container")["id"]
        ),
        quantity=10,
        status="assigned",
    )
    conn.execute(
        "UPDATE shipments SET truck_id = ?, from_location = ?, to_location = ? WHERE segment_id = ?",
        ("TRK-SIGNAL", "Depot", "Site Q", int(signal_segment["id"])),
    )

    item = upsert_inventory_item(
        conn,
        name="Container Pod",
        quantity=5,
        architecture="container",
    )
    job_id = _job(conn, "Reallocation Client", "Depot", "Site Q")
    conn.execute("UPDATE jobs SET job_number = ? WHERE id = ?", ("REALLOC-1", job_id))
    segment = ensure_segment(
        conn,
        job_id=job_id,
        segment_sequence=1,
        planned_start="2026-03-22T08:00:00+00:00",
        planned_end="2026-03-22T12:00:00+00:00",
    )
    upsert_inventory_requirement(
        conn,
        job_id=job_id,
        segment_id=int(segment["id"]),
        inventory_item_id=int(item["id"]),
        requirement_name="Container Pod",
        required_quantity=3,
        substitution_allowed=False,
        architecture="container",
    )
    allocate_inventory_to_segment(
        conn,
        segment_id=int(segment["id"]),
        inventory_item_id=int(item["id"]),
        quantity=1,
    )
    upsert_job_operational_signal(
        conn,
        job_number="REALLOC-1",
        origin="Depot",
        destination="Site Q",
        estimated_volume_m3=10.0,
        source="planning",
    )

    opportunities = list_operational_share_opportunities(conn, job_id=job_id)

    assert len(opportunities) == 1
    row = opportunities[0]
    assert row["opportunityType"] == "container_reallocation"
    assert row["utilizationState"] == "pressure_with_relief_option"
    assert row["utilizationResponse"] == "over_utilised"
    assert row["recommendedActionKey"] == "reallocate_container"
    assert row["operatorActions"][0] == "reallocate_container"
    assert row["spareCapacityLabel"] == "favorable"
    assert row["containerShortageQuantity"] == 2.0


def test_operational_share_opportunities_flag_backhaul_share_candidate() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)
    conn.execute("ALTER TABLE jobs ADD COLUMN job_number TEXT")

    upsert_truck(conn, truck_id="TRK-BACKHAUL", name="Backhaul Truck", capacity_m3=50.0)
    route_job_id = _job(conn, "Signal Load", "Alpha", "Beta")
    signal_segment = ensure_segment(
        conn,
        job_id=route_job_id,
        segment_sequence=1,
        from_location="Alpha",
        to_location="Beta",
        planned_start="2026-03-21T08:00:00+00:00",
        planned_end="2026-03-21T12:00:00+00:00",
    )
    allocate_inventory_to_segment(
        conn,
        segment_id=int(signal_segment["id"]),
        inventory_item_id=int(
            upsert_inventory_item(conn, name="Signal Item", quantity=5, architecture="general")["id"]
        ),
        quantity=5,
        status="assigned",
    )
    conn.execute(
        "UPDATE shipments SET truck_id = ?, from_location = ?, to_location = ? WHERE segment_id = ?",
        ("TRK-BACKHAUL", "Alpha", "Beta", int(signal_segment["id"])),
    )

    item = upsert_inventory_item(
        conn,
        name="Blankets",
        quantity=10,
        architecture="general",
    )
    job_id = _job(conn, "Backhaul Client", "Alpha", "Beta")
    conn.execute("UPDATE jobs SET job_number = ? WHERE id = ?", ("BACKHAUL-1", job_id))
    segment = ensure_segment(
        conn,
        job_id=job_id,
        segment_sequence=1,
        planned_start="2026-03-22T08:00:00+00:00",
        planned_end="2026-03-22T12:00:00+00:00",
    )
    upsert_inventory_requirement(
        conn,
        job_id=job_id,
        segment_id=int(segment["id"]),
        inventory_item_id=int(item["id"]),
        requirement_name="Blankets",
        required_quantity=2,
        substitution_allowed=False,
        architecture="general",
    )
    allocate_inventory_to_segment(
        conn,
        segment_id=int(segment["id"]),
        inventory_item_id=int(item["id"]),
        quantity=2,
    )
    upsert_job_operational_signal(
        conn,
        job_number="BACKHAUL-1",
        origin="Alpha",
        destination="Beta",
        estimated_volume_m3=8.0,
        source="planning",
    )

    opportunities = list_operational_share_opportunities(conn, job_id=job_id)

    assert len(opportunities) == 1
    row = opportunities[0]
    assert row["opportunityType"] == "backhaul_share_candidate"
    assert row["utilizationState"] == "under_utilised"
    assert row["utilizationResponse"] == "under_utilised"
    assert row["recommendedActionKey"] == "offer_share_capacity"
    assert row["operatorActions"][0] == "offer_share_capacity"
    assert row["spareCapacityLabel"] == "favorable"
    assert "backhaul positioning" in row["recommendedAction"]


def test_record_dispatch_share_action_is_reflected_in_latest_opportunity_response() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)
    conn.execute("ALTER TABLE jobs ADD COLUMN job_number TEXT")

    upsert_truck(conn, truck_id="TRK-BACKHAUL", name="Backhaul Truck", capacity_m3=50.0)
    route_job_id = _job(conn, "Signal Load", "Alpha", "Beta")
    signal_segment = ensure_segment(
        conn,
        job_id=route_job_id,
        segment_sequence=1,
        from_location="Alpha",
        to_location="Beta",
        planned_start="2026-03-21T08:00:00+00:00",
        planned_end="2026-03-21T12:00:00+00:00",
    )
    allocate_inventory_to_segment(
        conn,
        segment_id=int(signal_segment["id"]),
        inventory_item_id=int(
            upsert_inventory_item(conn, name="Signal Item", quantity=5, architecture="general")["id"]
        ),
        quantity=5,
        status="assigned",
    )
    conn.execute(
        "UPDATE shipments SET truck_id = ?, from_location = ?, to_location = ? WHERE segment_id = ?",
        ("TRK-BACKHAUL", "Alpha", "Beta", int(signal_segment["id"])),
    )

    item = upsert_inventory_item(conn, name="Blankets", quantity=10, architecture="general")
    job_id = _job(conn, "Backhaul Client", "Alpha", "Beta")
    conn.execute("UPDATE jobs SET job_number = ? WHERE id = ?", ("BACKHAUL-2", job_id))
    segment = ensure_segment(
        conn,
        job_id=job_id,
        segment_sequence=1,
        planned_start="2026-03-22T08:00:00+00:00",
        planned_end="2026-03-22T12:00:00+00:00",
    )
    upsert_inventory_requirement(
        conn,
        job_id=job_id,
        segment_id=int(segment["id"]),
        inventory_item_id=int(item["id"]),
        requirement_name="Blankets",
        required_quantity=2,
        substitution_allowed=False,
        architecture="general",
    )
    allocate_inventory_to_segment(
        conn,
        segment_id=int(segment["id"]),
        inventory_item_id=int(item["id"]),
        quantity=2,
    )
    upsert_job_operational_signal(
        conn,
        job_number="BACKHAUL-2",
        origin="Alpha",
        destination="Beta",
        estimated_volume_m3=8.0,
        source="planning",
    )

    created = record_dispatch_share_action(
        conn,
        job_id=job_id,
        opportunity_type="backhaul_share_candidate",
        utilization_state="under_utilised",
        action_type="offer_share_capacity",
        action_status="in_progress",
        actor="dispatch-1",
        note="Hold truck for add-on load.",
    )

    assert created["actionType"] == "offer_share_capacity"
    assert created["actionStatus"] == "in_progress"
    assert created["actor"] == "dispatch-1"
    assert "in_progress" in DISPATCH_SHARE_ACTION_STATUSES

    actions = list_dispatch_share_actions(conn, job_id=job_id, limit=10)
    assert len(actions) == 1
    assert actions[0]["note"] == "Hold truck for add-on load."

    opportunities = list_operational_share_opportunities(conn, job_id=job_id)
    assert len(opportunities) == 1
    row = opportunities[0]
    assert row["latestActionType"] == "offer_share_capacity"
    assert row["latestActionStatus"] == "in_progress"
    assert row["latestActionActor"] == "dispatch-1"


def test_approved_inventory_substitution_clears_readiness_block_and_board_rollup() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)

    from analytics.db.inventory import (
        decide_inventory_substitution,
        request_inventory_substitution,
        upsert_inventory_item,
        upsert_inventory_requirement,
    )

    item = upsert_inventory_item(
        conn,
        name="Container Pod",
        quantity=2,
        architecture="container",
    )
    substitute = upsert_inventory_item(
        conn,
        name="Spare Container Pod",
        quantity=4,
        architecture="container",
    )
    job_id = _job(conn, "Substitution Client", "Depot", "Site J")
    segment = ensure_segment(
        conn,
        job_id=job_id,
        segment_sequence=1,
        planned_start="2026-03-20T08:00:00+00:00",
        planned_end="2026-03-20T12:00:00+00:00",
    )
    requirement = upsert_inventory_requirement(
        conn,
        job_id=job_id,
        segment_id=int(segment["id"]),
        inventory_item_id=int(item["id"]),
        requirement_name="Container Pod",
        required_quantity=4,
        substitution_allowed=True,
        architecture="container",
    )

    pending = request_inventory_substitution(
        conn,
        requirement_id=int(requirement["id"]),
        requested_quantity=2,
        requested_by="warehouse-1",
        reason_code="stock_shortage",
        substitute_inventory_item_id=int(substitute["id"]),
    )
    readiness_pending = evaluate_segment_readiness(conn, segment_id=int(segment["id"]))
    assert readiness_pending["blockingFlags"] == []
    assert readiness_pending["overrideRequired"] is True
    assert readiness_pending["inventoryShortages"][0]["shortageQuantity"] == 4.0

    decide_inventory_substitution(
        conn,
        substitution_id=int(pending["id"]),
        status="approved",
        approved_by="dispatch-1",
        approved_role="dispatcher",
        approved_quantity=4,
        substitute_inventory_item_id=int(substitute["id"]),
    )
    readiness_approved = evaluate_segment_readiness(conn, segment_id=int(segment["id"]))
    assert readiness_approved["overrideRequired"] is False
    assert readiness_approved["inventoryShortages"] == []

    board = list_job_operations_board(conn, job_id=job_id)
    assert len(board) == 1
    row = board[0]
    assert row["approvedSubstitutionQuantity"] == 4.0
    assert row["shortageQuantity"] == 0.0
    assert row["pendingSubstitutionCount"] == 0


def test_operations_cutover_workflows_can_be_listed_and_updated() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)

    rows = list_operations_cutover_workflows(conn)
    assert any(row["workflowKey"] == "dispatch_execution" for row in rows)

    updated = upsert_operations_cutover_workflow(
        conn,
        workflow_key="dispatch_execution",
        cutover_status="fallback_only",
        owner_role="dispatcher",
        snapshot_mode="daily",
        snapshot_fields=["jobId", "jobStatus"],
        fallback_mode="manual_csv",
        cutover_target_percent=95.0,
        native_ready=True,
        dual_run_complete=True,
        fallback_drill_complete=True,
        operator_trained=True,
        rollback_instructions="Use the last exported dispatch snapshot while imports recover.",
        notes="Validated cutover drill.",
    )
    record_operations_cutover_event(
        conn,
        workflow_key="dispatch_execution",
        event_type="review",
        actor="dispatcher",
        created_at="2026-03-12T09:00:00+00:00",
    )
    record_operations_cutover_event(
        conn,
        workflow_key="dispatch_execution",
        event_type="fallback_drill",
        actor="dispatcher",
        created_at="2026-03-12T08:00:00+00:00",
    )
    record_operations_cutover_event(
        conn,
        workflow_key="dispatch_execution",
        event_type="snapshot_issued",
        actor="dispatcher",
        event_value="ops-team",
    )
    updated = next(
        row for row in list_operations_cutover_rollout(conn) if row["workflowKey"] == "dispatch_execution"
    )
    assert updated["cutoverStatus"] == "fallback_only"
    assert updated["snapshotMode"] == "daily"
    assert updated["metrics"]["cutoverTargetPercent"] == 95.0
    assert updated["metrics"]["lastReviewAt"] == "2026-03-12T09:00:00+00:00"
    assert updated["lastDrillAt"] == "2026-03-12T08:00:00+00:00"
    assert updated["metrics"]["snapshotConsumerCount"] == 1
    assert updated["checklist"]["fallbackDrillComplete"] is True
    assert updated["allChecksComplete"] is True
    assert updated["recommendation"]["recommendedStatus"] == "fallback_only"
    assert updated["recommendation"]["actionable"] is False
    events = list_operations_cutover_events(conn, workflow_key="dispatch_execution", limit=10)
    assert len(events) >= 3


def test_apply_operations_cutover_recommendation_transitions_and_logs() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)

    upsert_operations_cutover_workflow(
        conn,
        workflow_key="dispatch_execution",
        cutover_status="dual_run",
        owner_role="dispatcher",
        snapshot_mode="on_demand",
        snapshot_fields=["jobId"],
        fallback_mode="import_only",
        cutover_target_percent=0.0,
        native_ready=True,
        dual_run_complete=True,
        fallback_drill_complete=True,
        operator_trained=True,
        rollback_instructions="Rollback",
        notes="Ready",
    )
    with pytest.raises(ValueError, match="approval"):
        apply_operations_cutover_recommendation(
            conn,
            workflow_key="dispatch_execution",
            actor="dispatcher",
            note="Promotion",
        )

    requested = request_operations_cutover_promotion(
        conn,
        workflow_key="dispatch_execution",
        actor="ops-manager",
        note="Metrics support promotion.",
    )
    assert requested["approval"]["status"] == "requested"

    approved = approve_operations_cutover_promotion(
        conn,
        workflow_key="dispatch_execution",
        actor="commercial-owner",
        note="Approved for rollout.",
    )
    assert approved["approval"]["status"] == "approved"

    updated = apply_operations_cutover_recommendation(
        conn,
        workflow_key="dispatch_execution",
        actor="dispatcher",
        note="Promotion",
    )
    assert updated["cutoverStatus"] == "native_primary"
    events = list_operations_cutover_events(conn, workflow_key="dispatch_execution", limit=10)
    assert any(item["eventType"] == "status_transition" for item in events)


def test_cutover_promotion_rejection_blocks_apply_until_new_request() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)

    upsert_operations_cutover_workflow(
        conn,
        workflow_key="dispatch_execution",
        cutover_status="dual_run",
        owner_role="dispatcher",
        snapshot_mode="on_demand",
        snapshot_fields=["jobId"],
        fallback_mode="import_only",
        cutover_target_percent=0.0,
        native_ready=True,
        dual_run_complete=True,
        fallback_drill_complete=True,
        operator_trained=True,
        rollback_instructions="Rollback",
        notes="Ready",
    )
    requested = request_operations_cutover_promotion(
        conn,
        workflow_key="dispatch_execution",
        actor="ops-manager",
        note="Requesting promotion.",
    )
    assert requested["approval"]["status"] == "requested"

    rejected = reject_operations_cutover_promotion(
        conn,
        workflow_key="dispatch_execution",
        actor="commercial-owner",
        note="Need another drill.",
    )
    assert rejected["approval"]["status"] == "rejected"
    assert rejected["recommendation"]["blockedByApproval"] is True

    with pytest.raises(ValueError, match="Address feedback"):
        apply_operations_cutover_recommendation(
            conn,
            workflow_key="dispatch_execution",
            actor="dispatcher",
        )


def test_inventory_substitution_changes_segment_readiness() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)

    job_id = _job(conn, "Client E", "Depot", "Site M")
    segment = ensure_segment(
        conn,
        job_id=job_id,
        segment_sequence=1,
        planned_start="2026-03-12T08:00:00+00:00",
        planned_end="2026-03-12T10:00:00+00:00",
    )
    item = upsert_inventory_item(conn, name="Container Pod", quantity=1, architecture="container")
    substitute = upsert_inventory_item(conn, name="Spare Pod", quantity=5, architecture="container")
    requirement = upsert_inventory_requirement(
        conn,
        job_id=job_id,
        segment_id=int(segment["id"]),
        inventory_item_id=int(item["id"]),
        requirement_name="Container Pod",
        required_quantity=3,
        substitution_allowed=True,
        architecture="container",
    )
    allocate_inventory_to_segment(conn, segment_id=int(segment["id"]), inventory_item_id=int(item["id"]), quantity=1, status="staged")

    before = evaluate_segment_readiness(conn, segment_id=int(segment["id"]), truck_ids=[], worker_assignments=[])
    assert before["assignmentStatus"] == "override_required"
    assert any(flag.startswith("segment:inventory_shortage:Container Pod") for flag in before["overrideableFlags"])

    request = request_inventory_substitution(
        conn,
        requirement_id=int(requirement["id"]),
        requested_quantity=2,
        requested_by="warehouse-1",
        reason_code="stock_shortage",
        substitute_inventory_item_id=int(substitute["id"]),
    )
    pending = evaluate_segment_readiness(conn, segment_id=int(segment["id"]), truck_ids=[], worker_assignments=[])
    assert pending["assignmentStatus"] == "override_required"

    decide_inventory_substitution(
        conn,
        substitution_id=int(request["id"]),
        status="approved",
        approved_by="dispatch-1",
        approved_role="dispatcher",
        approved_quantity=2,
        substitute_inventory_item_id=int(substitute["id"]),
    )
    after = evaluate_segment_readiness(conn, segment_id=int(segment["id"]), truck_ids=[], worker_assignments=[])
    assert not any(flag.startswith("segment:inventory_shortage:Container Pod") for flag in after["overrideableFlags"])
