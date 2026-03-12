"""Helpers for syncing the shared operations workbook into local tables."""
from __future__ import annotations

import os
import sqlite3
from typing import Any

from analytics.db.fleet import import_workers_from_google_sheet
from analytics.db.inventory import import_suppliers_from_google_sheet
from analytics.vehicle_workbook import import_vehicle_details_from_google_sheet


def sync_operations_workbook(
    conn: sqlite3.Connection,
    *,
    sheet_id_or_url: str | None = None,
) -> dict[str, Any]:
    """Refresh fleet, staff, and suppliers from the shared operations workbook."""

    reference = (
        (sheet_id_or_url or "").strip()
        or os.environ.get("OPERATIONS_WORKBOOK_URL", "").strip()
        or os.environ.get("OPERATIONS_WORKBOOK_SHEET_ID", "").strip()
    )
    if not reference:
        raise ValueError(
            "Provide an operations workbook reference or set OPERATIONS_WORKBOOK_SHEET_ID/URL."
        )

    staff_sheet_name = os.environ.get("OPERATIONS_STAFF_SHEET_NAME", "STAFF").strip() or "STAFF"
    suppliers_sheet_name = (
        os.environ.get("OPERATIONS_SUPPLIERS_SHEET_NAME", "SUPPLIERS").strip()
        or "SUPPLIERS"
    )

    fleet_imported = import_vehicle_details_from_google_sheet(conn, sheet_id=reference)
    staff_inserted, staff_updated = import_workers_from_google_sheet(
        conn,
        sheet_id_or_url=reference,
        sheet_name=staff_sheet_name,
    )
    suppliers_imported = import_suppliers_from_google_sheet(
        conn,
        sheet_id=reference,
        sheet_name=suppliers_sheet_name,
    )

    return {
        "fleetImported": fleet_imported,
        "staffInserted": staff_inserted,
        "staffUpdated": staff_updated,
        "suppliersImported": suppliers_imported,
        "staffSheetName": staff_sheet_name,
        "suppliersSheetName": suppliers_sheet_name,
    }


__all__ = ["sync_operations_workbook"]
