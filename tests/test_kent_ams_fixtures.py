from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from analytics.db import ensure_dashboard_tables
from analytics.kent_ams_import import import_kent_ams_records, list_prioritized_tenders


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "kent_ams"


def test_kent_tender_fixture_import_smoke():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)
    conn.execute(
        """
        INSERT INTO trucks (truck_id, name, active, updated_at)
        VALUES ('TRK-FX-1', 'Fixture Truck', 1, '2026-03-12T00:00:00+00:00')
        """
    )
    conn.execute(
        """
        INSERT INTO workers (name, active, updated_at)
        VALUES ('Fixture Worker 1', 1, '2026-03-12T00:00:00+00:00')
        """
    )
    conn.execute(
        """
        INSERT INTO workers (name, active, updated_at)
        VALUES ('Fixture Worker 2', 1, '2026-03-12T00:00:00+00:00')
        """
    )
    conn.commit()

    payload = json.loads((FIXTURE_DIR / "tenders_sample.json").read_text())
    imported, updated = import_kent_ams_records(
        conn, "tenders", payload, dry_run=False
    )
    assert imported == 1
    assert updated == 0

    rows = list_prioritized_tenders(conn, status="open", limit=10)
    assert len(rows) == 1
    row = rows[0]
    assert row["tenderExternalId"] == "FIXTURE-T-1001"
    assert row["profitRuleMode"] == "EITHER"
    assert "beyond_transfer_rule" in row["overrideableFlags"]
    assert row["policyMatched"] is True
