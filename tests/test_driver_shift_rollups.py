import sqlite3
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from analytics.db import (
    create_shipment,
    ensure_dashboard_tables,
    fetch_driver_shifts,
    rollup_driver_shift_costs_by_job,
)
from analytics.driver_shifts import import_driver_shifts_from_sheet


def _build_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)
    return conn


def test_shift_import_links_jobs_and_shipments():
    conn = _build_conn()
    job_id = conn.execute(
        "INSERT INTO jobs (origin, destination, job_date) VALUES (?, ?, ?)",
        ("Depot", "Site A", "2024-02-01"),
    ).lastrowid
    other_job_id = conn.execute(
        "INSERT INTO jobs (origin, destination, job_date) VALUES (?, ?, ?)",
        ("Depot", "Site B", "2024-02-02"),
    ).lastrowid

    shipment = create_shipment(
        conn,
        job_id=job_id,
        historical_job_id=None,
        inventory_item_id=None,
        truck_id=None,
        worker_id=None,
        status="planned",
        scheduled_date="2024-02-01",
        delivered_at=None,
    )

    df = pd.DataFrame(
        [
            {
                "shift_date": "2024-02-01",
                "worker": "Alice",
                "hours": 8,
                "hourly_rate": 50,
                "job_id": job_id,
                "shift_window_start": "06:00",
                "shift_window_end": "14:00",
                "role": "Driver",
            },
            {
                "shift_date": "2024-02-01",
                "worker": "Bob",
                "hours": 6,
                "hourly_rate": 40,
                "shipment_id": shipment["id"],
                "shift_start": "07:00",
                "shift_end": "13:00",
            },
            {
                "shift_date": "2024-02-02",
                "worker": "Casey",
                "hours": 4,
                "hourly_rate": 55,
                "job_id": other_job_id,
            },
        ]
    )

    inserted, updated = import_driver_shifts_from_sheet(
        conn, dataframe=df, sheet_id=None
    )
    assert inserted == 3
    assert updated == 0

    rows = fetch_driver_shifts(conn)
    linked_jobs = {row["worker_name"]: row["linked_job_id"] for row in rows}
    assert linked_jobs["Alice"] == job_id
    assert linked_jobs["Bob"] == job_id  # resolved via shipment link
    assert linked_jobs["Casey"] == other_job_id

    alice_row = next(row for row in rows if row["worker_name"] == "Alice")
    assert alice_row["role"] == "Driver"
    assert alice_row["shift_window_start"] == "06:00"
    assert alice_row["shift_window_end"] == "14:00"

    rollups = rollup_driver_shift_costs_by_job(conn)
    summary = {row["job_id"]: row for row in rollups}
    assert summary[job_id]["total_hours"] == pytest.approx(14)
    assert summary[job_id]["total_cost"] == pytest.approx((8 * 50) + (6 * 40))
    assert summary[other_job_id]["total_hours"] == 4
    assert summary[other_job_id]["total_cost"] == pytest.approx(4 * 55)

    conn.close()
