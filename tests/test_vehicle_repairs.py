from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analytics import db
from analytics.vehicle_repairs import (
    import_vehicle_repairs_from_dataframe,
    load_vehicle_repairs,
)


def test_vehicle_repairs_table_bootstrap():
    conn = sqlite3.connect(":memory:")
    try:
        db.ensure_dashboard_tables(conn)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(vehicle_repairs)")}

        assert {
            "truck_id",
            "job_item",
            "price",
            "supplier",
            "service_date",
            "next_service_date",
            "notes",
            "created_at",
            "updated_at",
        }.issubset(columns)
    finally:
        conn.close()


def test_vehicle_repairs_import_and_update():
    conn = sqlite3.connect(":memory:")
    try:
        db.ensure_dashboard_tables(conn)

        df = pd.DataFrame(
            {
                "Truck": ["TR-1", "TR-1", "TR-2"],
                "Job item": ["Oil change", "Oil change", "Tyres"],
                "Price": [120.5, 120.5, 300.0],
                "Supplier": ["Bob's Workshop", "Bob's Workshop", "TyreCo"],
                "Service date": ["2024-01-01", "2024-01-01", "2024-02-03"],
                "Next due": ["2024-06-01", "2024-06-01", ""],
                "Notes": ["Initial entry", "Updated notes", "Rotation"],
            }
        )

        inserted, updated = import_vehicle_repairs_from_dataframe(conn, df)
        assert inserted == 2
        assert updated == 1

        stored = load_vehicle_repairs(conn)
        assert len(stored) == 2
        oil_change = stored[stored["job_item"] == "Oil change"].iloc[0]
        assert oil_change["service_date"] == "2024-01-01"
        assert oil_change["notes"] == "Updated notes"
    finally:
        conn.close()

