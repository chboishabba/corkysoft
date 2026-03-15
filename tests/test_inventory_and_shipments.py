import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analytics import db
from routes_to_sqlite import ensure_schema, migrate_schema


def test_schema_includes_logistics_tables(tmp_path):
    conn = sqlite3.connect(tmp_path / "routes.db")
    try:
        ensure_schema(conn)
        migrate_schema(conn)

        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }

        assert {
            "inventory_items",
            "inventory_movements",
            "workers",
            "trucks",
            "shipments",
            "job_segments",
            "job_segment_workers",
            "job_segment_vehicles",
            "worker_roles",
            "worker_compliances",
            "worker_role_assignments",
            "worker_compliance_assignments",
        }.issubset(tables)

        worker_columns = {row[1] for row in conn.execute("PRAGMA table_info(workers)")}
        assert {"rate", "tickets"}.issubset(worker_columns)

        shipment_columns = {row[1] for row in conn.execute("PRAGMA table_info(shipments)")}
        assert "segment_id" in shipment_columns
    finally:
        conn.close()


def test_shipments_backfilled_for_jobs(tmp_path):
    conn = sqlite3.connect(tmp_path / "routes.db")
    conn.row_factory = sqlite3.Row
    try:
        ensure_schema(conn)
        conn.execute(
            "INSERT INTO jobs (origin, destination, hourly_rate, per_km_rate, country, provider)"
            " VALUES ('A', 'B', 1.0, 1.0, 'AU', 'ors')"
        )
        conn.execute(
            "INSERT INTO historical_jobs (job_date, origin, destination, client, quoted_price, imported_at)"
            " VALUES ('2023-01-01', 'C', 'D', 'Client', 10.0, 'now')"
        )

        migrate_schema(conn)

        shipments = conn.execute(
            """
            SELECT job_id, historical_job_id, status, quantity, from_location, to_location, segment_id
            FROM shipments
            ORDER BY id
            """
        ).fetchall()

        assert len(shipments) == 2
        assert shipments[0]["job_id"] is not None
        segment = conn.execute(
            "SELECT id, job_id, segment_sequence FROM job_segments WHERE job_id = ?",
            (shipments[0]["job_id"],),
        ).fetchone()
        assert segment is not None
        assert shipments[0]["status"] == "planned"
        assert shipments[0]["quantity"] == 1
        assert shipments[0]["from_location"] == "A"
        assert shipments[0]["to_location"] == "B"
        assert shipments[0]["segment_id"] == segment["id"]
        assert shipments[1]["historical_job_id"] is not None
        assert shipments[1]["status"] == "delivered"
        assert shipments[1]["quantity"] == 1
        assert shipments[1]["from_location"] == "C"
        assert shipments[1]["to_location"] == "D"
        assert shipments[1]["segment_id"] is None
    finally:
        conn.close()


def test_crud_helpers_and_views():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        db.ensure_dashboard_tables(conn)

        job_id = conn.execute(
            "INSERT INTO jobs (origin, destination) VALUES ('Origin', 'Dest')"
        ).lastrowid

        item = db.upsert_inventory_item(conn, name="Boxes", quantity=10, unit="ea")
        truck = db.upsert_truck(conn, truck_id="TR-1", name="Prime Mover", capacity_m3=50)
        worker = db.upsert_worker(
            conn, name="Alex Driver", role="Driver", rate=42.5, tickets=3
        )

        shipment = db.create_shipment(
            conn,
            job_id=job_id,
            inventory_item_id=item["id"],
            truck_id=truck["truck_id"],
            worker_id=worker["id"],
            scheduled_date="2024-01-01",
        )

        assert shipment["job_id"] == job_id
        assert shipment["segment_id"] is not None

        segment_vehicle = conn.execute(
            "SELECT * FROM job_segment_vehicles WHERE segment_id = ?",
            (shipment["segment_id"],),
        ).fetchone()
        segment_worker = conn.execute(
            "SELECT * FROM job_segment_workers WHERE segment_id = ?",
            (shipment["segment_id"],),
        ).fetchone()

        assert segment_vehicle["truck_id"] == truck["truck_id"]
        assert segment_worker["worker_id"] == worker["id"]

        rows = db.fetch_shipments_with_context(conn)
        assert len(rows) == 1
        row = rows[0]
        assert row["job_origin"] == "Origin"
        assert row["inventory_name"] == "Boxes"
        assert row["truck_name"] == "Prime Mover"
        assert row["worker_name"] == "Alex Driver"
        assert row["segment_worker_names"] == "Alex Driver"
        assert row["segment_truck_names"] == "Prime Mover"
        assert row["worker_rate"] == 42.5
        assert row["worker_tickets"] == 3
    finally:
        conn.close()


def test_import_workers_from_staff_sheet():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        pytest.importorskip("openpyxl")
        db.ensure_dashboard_tables(conn)
        workbook_path = Path(__file__).resolve().parents[1] / "Crusader.xlsx"

        inserted, updated = db.import_workers_from_staff_sheet(conn, workbook_path)

        assert inserted == 1
        assert updated == 0

        worker = conn.execute(
            "SELECT name, rate, tickets FROM workers WHERE name = ?",
            ("Johl Brown",),
        ).fetchone()
        assert worker["name"] == "Johl Brown"
        assert worker["rate"] == 0
        assert worker["tickets"] is None
    finally:
        conn.close()


def test_supplier_import_and_linkage():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        db.ensure_dashboard_tables(conn)

        dataframe = pd.DataFrame(
            [
                {
                    "company": "Parts Co",
                    "contact_name": "Sam Supplier",
                    "phone_number": "555-1234",
                    "email": "sam@example.com",
                    "notes": "Preferred for hydraulics",
                },
                {"company": "", "contact_name": "Ignored"},
            ]
        )

        imported = db.import_suppliers_from_google_sheet(conn, dataframe=dataframe)
        assert imported == 1

        supplier = db.list_suppliers(conn)[0]
        item = db.upsert_inventory_item(
            conn, name="Hydraulic Pump", quantity=2, supplier_id=supplier["id"]
        )

        job_id = conn.execute(
            "INSERT INTO jobs (origin, destination) VALUES ('Site A', 'Site B')"
        ).lastrowid
        db.create_shipment(conn, job_id=job_id, inventory_item_id=item["id"])

        shipment = db.fetch_shipments_with_context(conn)[0]
        assert shipment["supplier_company_name"] == "Parts Co"
        assert shipment["supplier_contact_name"] == "Sam Supplier"
        assert shipment["supplier_contact_number"] == "555-1234"
        assert shipment["supplier_email"] == "sam@example.com"
    finally:
        conn.close()


def test_segment_inventory_coordination_and_allocation():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        db.ensure_dashboard_tables(conn)

        supplier = db.upsert_supplier(conn, company_name="Ops Supplier")
        item = db.upsert_inventory_item(
            conn,
            name="Crates",
            quantity=12,
            supplier_id=supplier["id"],
            unit="ea",
        )
        job_id = conn.execute(
            "INSERT INTO jobs (origin, destination) VALUES ('Depot', 'Site C')"
        ).lastrowid
        segment = db.upsert_job_segment(
            conn,
            job_id=job_id,
            segment_sequence=1,
            from_location="Depot",
            to_location="Site C",
            planned_start="2026-03-15T08:00:00+00:00",
            planned_end="2026-03-15T10:00:00+00:00",
        )

        shipment = db.allocate_inventory_to_segment(
            conn,
            segment_id=int(segment["id"]),
            inventory_item_id=int(item["id"]),
            quantity=3,
            status="staged",
        )
        assert shipment["segment_id"] == int(segment["id"])

        rows = db.list_segment_inventory_coordination(conn, job_id=job_id)
        assert len(rows) == 1
        row = rows[0]
        assert row["segmentId"] == int(segment["id"])
        assert row["shipmentCount"] == 1
        assert row["allocatedQuantity"] == 3.0
        assert row["inventoryNames"] == ["Crates"]
        assert row["supplierNames"] == ["Ops Supplier"]
    finally:
        conn.close()


def test_inventory_requirements_compute_shortages_and_architecture() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        db.ensure_dashboard_tables(conn)
        item = db.upsert_inventory_item(
            conn,
            name="Module Container",
            quantity=8,
            unit="ea",
            architecture="container",
        )
        job_id = conn.execute(
            "INSERT INTO jobs (origin, destination) VALUES ('Depot', 'Site D')"
        ).lastrowid
        segment = db.upsert_job_segment(
            conn,
            job_id=job_id,
            segment_sequence=1,
            from_location="Depot",
            to_location="Site D",
        )
        db.upsert_inventory_requirement(
            conn,
            job_id=int(job_id),
            segment_id=int(segment["id"]),
            inventory_item_id=int(item["id"]),
            requirement_name="Module Container",
            required_quantity=5,
            substitution_allowed=False,
            architecture="container",
        )
        db.allocate_inventory_to_segment(
            conn,
            segment_id=int(segment["id"]),
            inventory_item_id=int(item["id"]),
            quantity=3,
            status="staged",
        )

        requirements = db.list_inventory_requirements(conn, segment_id=int(segment["id"]))
        assert len(requirements) == 1
        requirement = requirements[0]
        assert requirement["architecture"] == "container"
        assert requirement["requiredQuantity"] == 5.0
        assert requirement["allocatedQuantity"] == 3.0
        assert requirement["shortageQuantity"] == 2.0

        coordination = db.list_segment_inventory_coordination(conn, job_id=int(job_id))[0]
        assert coordination["requirementCount"] == 1
        assert coordination["requiredQuantity"] == 5.0
        assert coordination["allocatedQuantity"] == 3.0
        assert coordination["shortageQuantity"] == 2.0
        assert coordination["blockingShortageQuantity"] == 2.0
        assert coordination["warningShortageQuantity"] == 0.0
        assert coordination["architectures"] == ["container"]
    finally:
        conn.close()


def test_inventory_custody_updates_follow_movements() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        db.ensure_dashboard_tables(conn)
        item = db.upsert_inventory_item(
            conn,
            name="Dolly",
            quantity=2,
            unit="ea",
            architecture="reusable_asset",
        )

        db.record_inventory_movement(
            conn,
            inventory_item_id=int(item["id"]),
            reason="loaded_to_truck",
            state="loaded",
            location_type="truck",
            location_ref="TRK-9",
            location_label="Truck 9",
        )
        refreshed = db.get_inventory_balance(conn, int(item["id"]))
        assert refreshed is not None
        assert refreshed["custody_location_type"] == "truck"
        assert refreshed["custody_location_ref"] == "TRK-9"
        assert refreshed["custody_location_label"] == "Truck 9"

        movements = db.list_inventory_movements(conn, limit=5)
        assert movements[0]["location_type_value"] == "truck"
        assert movements[0]["location_label_value"] == "Truck 9"
    finally:
        conn.close()


def test_inventory_execution_events_and_substitutions_affect_requirement_view() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        db.ensure_dashboard_tables(conn)
        item = db.upsert_inventory_item(
            conn,
            name="Container Pod",
            quantity=4,
            unit="ea",
            architecture="container",
        )
        substitute = db.upsert_inventory_item(
            conn,
            name="Spare Container Pod",
            quantity=2,
            unit="ea",
            architecture="container",
        )
        job_id = conn.execute(
            "INSERT INTO jobs (origin, destination) VALUES ('Depot', 'Site E')"
        ).lastrowid
        segment = db.upsert_job_segment(
            conn,
            job_id=job_id,
            segment_sequence=1,
            from_location="Depot",
            to_location="Site E",
        )
        requirement = db.upsert_inventory_requirement(
            conn,
            job_id=int(job_id),
            segment_id=int(segment["id"]),
            inventory_item_id=int(item["id"]),
            requirement_name="Container Pod",
            required_quantity=5,
            substitution_allowed=True,
            architecture="container",
        )
        db.allocate_inventory_to_segment(
            conn,
            segment_id=int(segment["id"]),
            inventory_item_id=int(item["id"]),
            quantity=3,
            status="staged",
        )

        for stage in ("picked", "packed", "loaded"):
            db.record_inventory_execution_event(
                conn,
                job_id=int(job_id),
                segment_id=int(segment["id"]),
                requirement_id=int(requirement["id"]),
                inventory_item_id=int(item["id"]),
                stage=stage,
                quantity=3,
                actor="warehouse-1",
                container_ref="CONT-1",
                truck_id=None,
                location_type="container",
                location_ref="CONT-1",
                location_label="Container 1",
            )
        requested = db.request_inventory_substitution(
            conn,
            requirement_id=int(requirement["id"]),
            requested_quantity=2,
            requested_by="warehouse-1",
            reason_code="stock_shortage",
            substitute_inventory_item_id=int(substitute["id"]),
        )
        pending = db.list_inventory_requirements(conn, segment_id=int(segment["id"]))[0]
        assert pending["executionStage"] == "loaded"
        assert pending["hasPendingSubstitution"] is True
        assert pending["approvedSubstitutionQuantity"] == 0.0
        assert pending["requestedSubstitutionQuantity"] == 2.0
        assert pending["shortageQuantity"] == 2.0

        db.decide_inventory_substitution(
            conn,
            substitution_id=int(requested["id"]),
            status="approved",
            approved_by="dispatch-1",
            approved_role="dispatcher",
            approved_quantity=2,
            substitute_inventory_item_id=int(substitute["id"]),
            note="Approve equivalent container pod",
        )
        approved = db.list_inventory_requirements(conn, segment_id=int(segment["id"]))[0]
        assert approved["approvedSubstitutionQuantity"] == 2.0
        assert approved["effectiveFulfilledQuantity"] == 5.0
        assert approved["shortageQuantity"] == 0.0

        coordination = db.list_segment_inventory_coordination(conn, job_id=int(job_id))[0]
        assert coordination["approvedSubstitutionQuantity"] == 2.0
        assert coordination["pendingSubstitutionCount"] == 0
        assert coordination["executionStages"] == ["loaded"]

        events = db.list_inventory_execution_events(conn, segment_id=int(segment["id"]))
        assert events[0]["stage"] == "loaded"
        assert events[0]["containerRef"] == "CONT-1"
    finally:
        conn.close()


def test_inventory_execution_stages_and_reason_catalog_are_enforced() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        db.ensure_dashboard_tables(conn)
        reasons = db.list_inventory_substitution_reason_codes(conn, active_only=True)
        assert any(row["code"] == "stock_shortage" for row in reasons)

        item = db.upsert_inventory_item(
            conn,
            name="Packing Crate",
            quantity=2,
            architecture="container",
        )
        substitute = db.upsert_inventory_item(
            conn,
            name="Alternate Packing Crate",
            quantity=3,
            architecture="container",
        )
        job_id = conn.execute(
            "INSERT INTO jobs (origin, destination) VALUES ('Depot', 'Site K')"
        ).lastrowid
        segment = db.upsert_job_segment(
            conn,
            job_id=job_id,
            segment_sequence=1,
            from_location="Depot",
            to_location="Site K",
        )
        requirement = db.upsert_inventory_requirement(
            conn,
            job_id=int(job_id),
            segment_id=int(segment["id"]),
            inventory_item_id=int(item["id"]),
            requirement_name="Packing Crate",
            required_quantity=2,
            substitution_allowed=True,
            architecture="container",
        )

        with pytest.raises(ValueError, match="not allowed from 'required'"):
            db.record_inventory_execution_event(
                conn,
                job_id=int(job_id),
                segment_id=int(segment["id"]),
                requirement_id=int(requirement["id"]),
                inventory_item_id=int(item["id"]),
                stage="loaded",
                quantity=2,
                actor="warehouse-1",
            )

        request = db.request_inventory_substitution(
            conn,
            requirement_id=int(requirement["id"]),
            requested_quantity=1,
            requested_by="warehouse-1",
            reason_code="stock_shortage",
            substitute_inventory_item_id=int(substitute["id"]),
        )

        with pytest.raises(ValueError, match="approval role must be one of"):
            db.decide_inventory_substitution(
                conn,
                substitution_id=int(request["id"]),
                status="approved",
                approved_by="planner-1",
                approved_role="warehouse",
                approved_quantity=1,
                substitute_inventory_item_id=int(substitute["id"]),
            )
    finally:
        conn.close()


def test_rejected_substitution_keeps_shortage_active() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        db.ensure_dashboard_tables(conn)
        item = db.upsert_inventory_item(conn, name="Crew Box", quantity=1, architecture="container")
        substitute = db.upsert_inventory_item(conn, name="Alt Crew Box", quantity=2, architecture="container")
        job_id = conn.execute(
            "INSERT INTO jobs (origin, destination) VALUES ('Depot', 'Site L')"
        ).lastrowid
        segment = db.upsert_job_segment(conn, job_id=job_id, segment_sequence=1, from_location="Depot", to_location="Site L")
        requirement = db.upsert_inventory_requirement(
            conn,
            job_id=int(job_id),
            segment_id=int(segment["id"]),
            inventory_item_id=int(item["id"]),
            requirement_name="Crew Box",
            required_quantity=3,
            substitution_allowed=True,
            architecture="container",
        )
        db.allocate_inventory_to_segment(conn, segment_id=int(segment["id"]), inventory_item_id=int(item["id"]), quantity=1, status="staged")
        request = db.request_inventory_substitution(
            conn,
            requirement_id=int(requirement["id"]),
            requested_quantity=2,
            requested_by="warehouse-1",
            reason_code="stock_shortage",
            substitute_inventory_item_id=int(substitute["id"]),
        )

        db.decide_inventory_substitution(
            conn,
            substitution_id=int(request["id"]),
            status="rejected",
            approved_by="dispatch-1",
            approved_role="dispatcher",
            note="Wait for original stock",
        )
        requirement_view = db.list_inventory_requirements(conn, segment_id=int(segment["id"]))[0]
        assert requirement_view["approvedSubstitutionQuantity"] == 0.0
        assert requirement_view["requestedSubstitutionQuantity"] == 0.0
        assert requirement_view["shortageQuantity"] == 2.0
        substitutions = db.list_inventory_substitutions(conn, segment_id=int(segment["id"]))
        assert substitutions[0]["status"] == "rejected"
    finally:
        conn.close()


def test_inventory_custody_latest_location_remains_singular() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        db.ensure_dashboard_tables(conn)
        item = db.upsert_inventory_item(conn, name="Reusable Crate", quantity=1, architecture="reusable_asset")
        db.record_inventory_movement(
            conn,
            inventory_item_id=int(item["id"]),
            reason="loaded_to_truck",
            state="loaded",
            location_type="truck",
            location_ref="TRK-1",
            location_label="Truck 1",
        )
        db.record_inventory_movement(
            conn,
            inventory_item_id=int(item["id"]),
            reason="arrived_on_site",
            state="delivered",
            location_type="site",
            location_ref="SITE-1",
            location_label="Site 1",
        )
        refreshed = db.get_inventory_balance(conn, int(item["id"]))
        assert refreshed["custody_location_type"] == "site"
        assert refreshed["custody_location_ref"] == "SITE-1"
        assert refreshed["custody_location_label"] == "Site 1"
        movements = [
            row
            for row in db.list_inventory_movements(conn, limit=10)
            if int(row["inventory_item_id"]) == int(item["id"])
        ]
        assert movements[0]["location_type_value"] == "site"
        assert movements[1]["location_type_value"] == "truck"
    finally:
        conn.close()


def test_segment_worker_requires_assigned_role():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        db.ensure_dashboard_tables(conn)
        job_id = conn.execute(
            "INSERT INTO jobs (origin, destination) VALUES ('Origin', 'Destination')"
        ).lastrowid
        worker = db.upsert_worker(conn, name="Roleless Worker")
        role_id = conn.execute(
            "INSERT INTO worker_roles (name) VALUES ('Loader')"
        ).lastrowid

        with pytest.raises(ValueError):
            db.create_shipment(
                conn,
                job_id=job_id,
                worker_id=worker["id"],
                worker_role_id=role_id,
            )
        conn.rollback()

        conn.execute(
            """
            INSERT INTO worker_role_assignments (worker_id, role_id, assigned_at)
            VALUES (?, ?, '2024-01-01T00:00:00Z')
            """,
            (worker["id"], role_id),
        )

        shipment = db.create_shipment(
            conn, job_id=job_id, worker_id=worker["id"], worker_role_id=role_id
        )
        segment_worker = conn.execute(
            "SELECT role_id FROM job_segment_workers WHERE segment_id = ?",
            (shipment["segment_id"],),
        ).fetchone()

        assert segment_worker["role_id"] == role_id
    finally:
        conn.close()


def test_segment_worker_requires_valid_compliance():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        db.ensure_dashboard_tables(conn)
        worker = db.upsert_worker(conn, name="Compliance Worker")

        compliance_id = conn.execute(
            "INSERT INTO worker_compliances (name) VALUES ('MSIC')"
        ).lastrowid
        job_id = conn.execute(
            "INSERT INTO jobs (origin, destination) VALUES ('Origin', 'Destination')"
        ).lastrowid

        with pytest.raises(ValueError):
            db.create_shipment(
                conn,
                job_id=job_id,
                worker_id=worker["id"],
                required_compliance_ids=[compliance_id],
            )
        conn.rollback()

        job_id = conn.execute(
            "INSERT INTO jobs (origin, destination) VALUES ('Origin', 'Destination')"
        ).lastrowid
        compliance_id = conn.execute(
            "INSERT OR IGNORE INTO worker_compliances (name) VALUES ('MSIC')"
        ).lastrowid or conn.execute(
            "SELECT id FROM worker_compliances WHERE name = 'MSIC'"
        ).fetchone()[0]
        conn.execute(
            """
            INSERT INTO worker_compliance_assignments (
                worker_id, compliance_id, expiry_date, assigned_at
            ) VALUES (?, ?, '2000-01-01', '2024-01-01T00:00:00Z')
            """,
            (worker["id"], compliance_id),
        )

        with pytest.raises(ValueError):
            db.create_shipment(
                conn,
                job_id=job_id,
                worker_id=worker["id"],
                required_compliance_ids=[compliance_id],
            )
        conn.rollback()

        job_id = conn.execute(
            "INSERT INTO jobs (origin, destination) VALUES ('Origin', 'Destination')"
        ).lastrowid
        compliance_id = conn.execute(
            "INSERT OR IGNORE INTO worker_compliances (name) VALUES ('MSIC')"
        ).lastrowid or conn.execute(
            "SELECT id FROM worker_compliances WHERE name = 'MSIC'"
        ).fetchone()[0]
        conn.execute(
            """
            INSERT OR REPLACE INTO worker_compliance_assignments (
                worker_id, compliance_id, expiry_date, assigned_at
            ) VALUES (?, ?, '2099-01-01', '2024-01-01T00:00:00Z')
            """,
            (worker["id"], compliance_id),
        )

        shipment = db.create_shipment(
            conn,
            job_id=job_id,
            worker_id=worker["id"],
            required_compliance_ids=[compliance_id],
        )

        segment_worker = conn.execute(
            "SELECT worker_id FROM job_segment_workers WHERE segment_id = ?",
            (shipment["segment_id"],),
        ).fetchone()
        assert segment_worker["worker_id"] == worker["id"]
    finally:
        conn.close()


def test_partial_shipments_track_quantity_and_locations():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        db.ensure_dashboard_tables(conn)
        job_id = conn.execute(
            "INSERT INTO jobs (origin, destination) VALUES ('Warehouse', 'Site Alpha')"
        ).lastrowid
        item = db.upsert_inventory_item(conn, name="Widgets", quantity=20, unit="ea")

        first = db.create_shipment(
            conn,
            job_id=job_id,
            inventory_item_id=item["id"],
            quantity=5,
            from_location="Warehouse",
            to_location="Staging",
        )
        second = db.create_shipment(
            conn,
            job_id=job_id,
            inventory_item_id=item["id"],
            quantity=7.5,
            status="in_transit",
            scheduled_date="2024-02-02",
        )

        rows = db.fetch_shipments_with_context(conn)
        by_id = {row["id"]: row for row in rows}

        assert by_id[first["id"]]["quantity"] == 5
        assert by_id[first["id"]]["from_location"] == "Warehouse"
        assert by_id[first["id"]]["to_location"] == "Staging"
        assert by_id[first["id"]]["segment_id"] is not None

        assert by_id[second["id"]]["quantity"] == 7.5
        assert by_id[second["id"]]["from_location"] == "Warehouse"
        assert by_id[second["id"]]["to_location"] == "Site Alpha"
        assert by_id[second["id"]]["status"] == "in_transit"
        assert by_id[second["id"]]["segment_id"] is not None
    finally:
        conn.close()


def test_shipments_capture_historical_locations_by_default(tmp_path):
    conn = sqlite3.connect(tmp_path / "routes.db")
    conn.row_factory = sqlite3.Row
    try:
        ensure_schema(conn)
        conn.execute(
            "INSERT INTO historical_jobs (job_date, origin, destination, client, quoted_price, imported_at)"
            " VALUES ('2023-02-02', 'Depot', 'Client Site', 'Client', 25.0, 'now')"
        )

        migrate_schema(conn)

        shipment = conn.execute("SELECT * FROM shipments").fetchone()
        assert shipment["quantity"] == 1
        assert shipment["from_location"] == "Depot"
        assert shipment["to_location"] == "Client Site"
        assert shipment["segment_id"] is None
    finally:
        conn.close()
