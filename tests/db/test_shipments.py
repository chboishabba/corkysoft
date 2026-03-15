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


def test_upsert_job_by_number_attempts_route_geometry_enrichment(monkeypatch) -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)

    calls: list[tuple[list[int], str]] = []

    def fake_populate(conn_arg, job_ids, *, dataset, client=None, provider=None):
        calls.append((list(job_ids), dataset))
        return 0

    monkeypatch.setattr("analytics.routes_map.populate_route_geometry", fake_populate)

    job = upsert_job_by_number(
        conn,
        job_number="JOB-2",
        origin="Depot",
        destination="Site",
        origin_lat=-27.4698,
        origin_lon=153.0251,
        dest_lat=-26.6500,
        dest_lon=153.0667,
    )

    assert job["job_number"] == "JOB-2"
    assert calls == [([int(job["id"])], "live")]
