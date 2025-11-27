import sqlite3

from analytics.db import create_shipment, ensure_dashboard_tables, upsert_job_by_number


def test_create_shipment_links_to_job() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    ensure_dashboard_tables(conn)
    job = upsert_job_by_number(conn, job_number="JOB-1", origin="Yard", destination="Site")
    shipment = create_shipment(conn, job_id=job["id"], quantity=2)

    assert shipment["job_id"] == job["id"]
    assert shipment["quantity"] == 2
