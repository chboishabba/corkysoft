import sqlite3

from analytics.db import ensure_dashboard_tables, upsert_driver_shift, upsert_truck, upsert_worker


def test_upsert_driver_shift_records_hours() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    ensure_dashboard_tables(conn)
    upsert_truck(conn, truck_id="T1", name="Test Truck")
    worker = upsert_worker(conn, name="Casey Driver")

    shift, created = upsert_driver_shift(
        conn,
        shift_date="2024-01-01",
        truck_id="T1",
        worker_name=worker["name"],
        hours=8.0,
        hourly_rate=10.0,
    )

    assert created is True
    assert shift["cost_total"] == 80.0
