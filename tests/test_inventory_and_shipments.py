import sqlite3
import sys
from pathlib import Path

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

        assert {"inventory_items", "workers", "trucks", "shipments"}.issubset(tables)
    finally:
        conn.close()


def test_shipments_backfilled_for_jobs(tmp_path):
    conn = sqlite3.connect(tmp_path / "routes.db")
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
            "SELECT job_id, historical_job_id, status FROM shipments ORDER BY id"
        ).fetchall()

        assert len(shipments) == 2
        assert shipments[0][0] is not None
        assert shipments[0][2] == "planned"
        assert shipments[1][1] is not None
        assert shipments[1][2] == "delivered"
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
        worker = db.upsert_worker(conn, name="Alex Driver", role="Driver")

        shipment = db.create_shipment(
            conn,
            job_id=job_id,
            inventory_item_id=item["id"],
            truck_id=truck["truck_id"],
            worker_id=worker["id"],
            scheduled_date="2024-01-01",
        )

        assert shipment["job_id"] == job_id

        rows = db.fetch_shipments_with_context(conn)
        assert len(rows) == 1
        row = rows[0]
        assert row["job_origin"] == "Origin"
        assert row["inventory_name"] == "Boxes"
        assert row["truck_name"] == "Prime Mover"
        assert row["worker_name"] == "Alex Driver"
    finally:
        conn.close()
