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
