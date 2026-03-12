from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.operations_workbook import sync_operations_workbook


def test_sync_operations_workbook_dispatches_all_imports(monkeypatch):
    conn = sqlite3.connect(":memory:")

    calls: dict[str, object] = {}

    def fake_vehicle_import(connection, *, sheet_id):
        calls["fleet"] = (connection, sheet_id)
        return 4

    def fake_staff_import(connection, *, sheet_id_or_url, sheet_name="STAFF"):
        calls["staff"] = (connection, sheet_id_or_url, sheet_name)
        return (2, 1)

    def fake_supplier_import(connection, *, sheet_id=None, sheet_name="SUPPLIERS", csv_url=None, dataframe=None):
        calls["suppliers"] = (connection, sheet_id, sheet_name)
        return 3

    monkeypatch.setattr(
        "analytics.operations_workbook.import_vehicle_details_from_google_sheet",
        fake_vehicle_import,
    )
    monkeypatch.setattr(
        "analytics.operations_workbook.import_workers_from_google_sheet",
        fake_staff_import,
    )
    monkeypatch.setattr(
        "analytics.operations_workbook.import_suppliers_from_google_sheet",
        fake_supplier_import,
    )
    monkeypatch.setenv("OPERATIONS_STAFF_SHEET_NAME", "STAFF")
    monkeypatch.setenv("OPERATIONS_SUPPLIERS_SHEET_NAME", "SUPPLIERS")

    summary = sync_operations_workbook(conn, sheet_id_or_url="ops-sheet")

    assert calls["fleet"][1] == "ops-sheet"
    assert calls["staff"][1:] == ("ops-sheet", "STAFF")
    assert calls["suppliers"][1:] == ("ops-sheet", "SUPPLIERS")
    assert summary == {
        "fleetImported": 4,
        "staffInserted": 2,
        "staffUpdated": 1,
        "suppliersImported": 3,
        "staffSheetName": "STAFF",
        "suppliersSheetName": "SUPPLIERS",
    }


def test_sync_operations_workbook_requires_reference(monkeypatch):
    conn = sqlite3.connect(":memory:")
    monkeypatch.delenv("OPERATIONS_WORKBOOK_URL", raising=False)
    monkeypatch.delenv("OPERATIONS_WORKBOOK_SHEET_ID", raising=False)

    try:
        sync_operations_workbook(conn)
    except ValueError as exc:
        assert "OPERATIONS_WORKBOOK_SHEET_ID/URL" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError when no workbook reference is configured")
