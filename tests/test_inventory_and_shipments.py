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
