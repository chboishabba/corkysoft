from __future__ import annotations

import sqlite3
import sys
from io import BytesIO
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analytics.db import ensure_dashboard_tables
from analytics.live_data import _ensure_live_tables, load_truck_positions
from analytics.vehicle_workbook import (
    import_vehicle_details_from_dataframe,
    import_vehicle_details_from_workbook,
)


def test_import_vehicle_details_from_dataframe_populates_tables() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)

    df = pd.DataFrame(
        {
            "STATE": ["QLD"],
            "REGO": ["ABC123"],
            "REGO EXPIRY DATE ": ["2025-12-31"],
            "MAKE": ["Volvo"],
            "MODEL": ["FH"],
            "YEAR": [2020],
            "BODY TYPE": ["Prime mover"],
            "DESCRIPTION": ["Heavy haul"],
            "NHV CHARGING CODE": ["HVC"],
            "INSURANCE TYPE": ["Full"],
            "ODOMETER": ["120,000"],
            "LAST SERVICE": ["2024-01-01"],
            "NEXT SERVICE DUE": ["2024-06-01"],
            "CERTIFICATE OF INSPECTION (COI) DUE DATE ": ["2024-12-31"],
            "COI NUMBER": ["COI-001"],
            "PRESENT DRIVER": ["Alex"],
            "DAILY COMPLETE?": ["Yes"],
        }
    )

    inserted = import_vehicle_details_from_dataframe(conn, df)
    assert inserted == 1

    detail = conn.execute(
        "SELECT rego_expiry, nhv_code, daily_check_complete FROM vehicle_details WHERE truck_id='ABC123'"
    ).fetchone()
    assert detail[0] == "2025-12-31"
    assert detail[1] == "HVC"
    assert detail[2] == 1

    truck = conn.execute("SELECT truck_id, name FROM trucks WHERE truck_id='ABC123'").fetchone()
    assert truck is not None
    assert "Volvo" in truck[1]


def test_load_truck_positions_includes_vehicle_metadata() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)
    _ensure_live_tables(conn)

    conn.execute(
        "INSERT INTO truck_positions (truck_id, lat, lon, status, updated_at) VALUES (?, ?, ?, ?, ?)",
        ("XYZ789", -27.0, 153.0, "loading", "2024-01-01T00:00:00Z"),
    )
    import_vehicle_details_from_dataframe(
        conn,
        pd.DataFrame(
            {
                "REGO": ["XYZ789"],
                "PRESENT DRIVER": ["Jamie"],
                "DAILY COMPLETE?": [0],
                "NEXT SERVICE DUE": ["2024-02-02"],
            }
        ),
    )

    frame = load_truck_positions(conn)
    assert "present_driver" in frame.columns
    assert frame.loc[0, "present_driver"] == "Jamie"
    assert frame.loc[0, "next_service"] == "2024-02-02"


def test_import_vehicle_details_from_multisheet_workbook() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame({"REGO": ["INDEX"], "GIDs (sheet ID)": [123]}).to_excel(
            writer, sheet_name="INDEX", index=False
        )
        pd.DataFrame(
            [
                [None, "STATE", "REGO", "MAKE", "MODEL", "YEAR"],
                [None, "QLD", "ABC123", "Volvo", "FH", 2020],
            ]
        ).to_excel(writer, sheet_name="ABC123", header=False, index=False)
        pd.DataFrame(
            [
                [None, "STATE", "REGO EXPIRY DATE ", "DESCRIPTION"],
                [None, "NSW", "2025-12-31", "Vacuum truck"],
            ]
        ).to_excel(writer, sheet_name="DEF456", header=False, index=False)

    buffer.seek(0)
    workbook = pd.ExcelFile(buffer)

    inserted = import_vehicle_details_from_workbook(conn, workbook)
    assert inserted == 2

    vehicles = conn.execute(
        "SELECT truck_id, state, make, rego_expiry FROM vehicle_details ORDER BY truck_id"
    ).fetchall()
    assert vehicles[0][0] == "ABC123"
    assert vehicles[0][2] == "Volvo"
    assert vehicles[1][0] == "DEF456"
    assert vehicles[1][3] == "2025-12-31"
