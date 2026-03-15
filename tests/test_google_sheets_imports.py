from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.db import ensure_dashboard_tables
from analytics.db.fleet import import_workers_from_google_sheet
from analytics.db.inventory import import_suppliers_from_google_sheet
from analytics.operations_assignment import assign_segment_resources, ensure_segment, list_worker_assignment_summary


def test_import_workers_from_google_sheet_uses_operations_workbook_env(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)

    captured: dict[str, object] = {}

    def fake_read_excel(path, sheet_name):
        captured["path"] = path
        captured["sheet_name"] = sheet_name
        return pd.DataFrame(
            [
                {
                    "FIRST NAME": "Johl",
                    "LAST NAME": "Brown",
                    "RATE": 25,
                    "TICKETS": 2,
                }
            ]
        )

    monkeypatch.setenv("OPERATIONS_WORKBOOK_SHEET_ID", "sheet-123")
    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    inserted, updated = import_workers_from_google_sheet(conn)

    assert inserted == 1
    assert updated == 0
    assert captured["sheet_name"] == "STAFF"
    assert "sheet-123" in str(captured["path"])
    worker = conn.execute(
        "SELECT name, rate, tickets, source_system, source_sheet, source_imported_at FROM workers"
    ).fetchone()
    assert worker["name"] == "Johl Brown"
    assert worker["rate"] == 25
    assert worker["tickets"] == 2
    assert worker["source_system"] == "google_sheets"
    assert worker["source_sheet"] == "STAFF"
    assert worker["source_imported_at"]


def test_import_suppliers_from_google_sheet_uses_operations_workbook_env(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)

    captured: dict[str, object] = {}

    def fake_read_csv(url):
        captured["url"] = url
        return pd.DataFrame(
            [
                {
                    "COMPANY NAME": "Parts Co",
                    "CONTACT NAME": "Sam",
                    "CONTACT NUMBER": "555-1234",
                }
            ]
        )

    monkeypatch.setenv("OPERATIONS_WORKBOOK_SHEET_ID", "sheet-456")
    monkeypatch.setattr(pd, "read_csv", fake_read_csv)

    imported = import_suppliers_from_google_sheet(conn, sheet_name="SUPPLIERS")

    assert imported == 1
    assert "sheet-456" in str(captured["url"])
    assert "sheet=SUPPLIERS" in str(captured["url"])
    supplier = conn.execute(
        """
        SELECT company_name, source_system, source_sheet, source_imported_at
        FROM suppliers
        """
    ).fetchone()
    assert supplier["company_name"] == "Parts Co"
    assert supplier["source_system"] == "google_sheets"
    assert supplier["source_sheet"] == "SUPPLIERS"
    assert supplier["source_imported_at"]


def test_worker_import_refreshes_metadata_without_clobbering_assignments(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)

    worker_id = int(
        conn.execute(
            """
            INSERT INTO workers (name, role, phone, active, hired_at, updated_at)
            VALUES (?, ?, ?, 1, ?, ?)
            """,
            ("Johl Brown", "Driver", "0400000000", "2026-03-01T00:00:00+00:00", "2026-03-01T00:00:00+00:00"),
        ).lastrowid
    )
    job_id = int(
        conn.execute(
            "INSERT INTO jobs (client, origin, destination, updated_at) VALUES (?, ?, ?, ?)",
            ("Client A", "Depot", "Site N", "2026-03-12T00:00:00+00:00"),
        ).lastrowid
    )
    segment = ensure_segment(conn, job_id=job_id, segment_sequence=1)
    assign_segment_resources(conn, segment_id=int(segment["id"]), truck_ids=[], worker_assignments=[{"workerId": worker_id}])

    def fake_read_excel(path, sheet_name):
        return pd.DataFrame(
            [{"FIRST NAME": "Johl", "LAST NAME": "Brown", "RATE": 30, "TICKETS": 4, "PHONE": "0400999999"}]
        )

    monkeypatch.setenv("OPERATIONS_WORKBOOK_SHEET_ID", "sheet-123")
    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    inserted, updated = import_workers_from_google_sheet(conn)
    assert inserted == 0
    assert updated == 1

    refreshed = conn.execute(
        "SELECT phone, rate, tickets, source_system, source_sheet, source_imported_at FROM workers WHERE id = ?",
        (worker_id,),
    ).fetchone()
    assert refreshed["phone"] == "0400999999"
    assert refreshed["rate"] == 30
    assert refreshed["tickets"] == 4
    assert refreshed["source_system"] == "google_sheets"
    assert refreshed["source_sheet"] == "STAFF"
    assert refreshed["source_imported_at"]
    summary = list_worker_assignment_summary(conn)
    assert summary[worker_id]["plannedSegmentCount"] == 1
