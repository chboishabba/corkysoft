from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.db import ensure_dashboard_tables, upsert_truck
from analytics.db.inventory import allocate_inventory_to_segment, upsert_inventory_item, upsert_inventory_requirement
from analytics.operational_signals import upsert_job_operational_signal
from analytics.operations_assignment import ensure_segment
from analytics.operations_diary import upsert_customer_invoice_review, upsert_operations_diary_task
from analytics.price_distribution import import_historical_jobs_from_dataframe
from corkysoft.mcp import __main__ as mcp_main
from corkysoft.mcp.bridge import _call_tool, _registry_envelope
from corkysoft.mcp.registry import build_default_registry
from corkysoft.mcp.server import build_fastmcp_server


def _job(conn: sqlite3.Connection, client: str, origin: str, destination: str) -> int:
    cursor = conn.execute(
        """
        INSERT INTO jobs (
            client,
            origin,
            destination,
            origin_resolved,
            destination_resolved,
            job_date,
            distance_km,
            volume_m3,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            client,
            origin,
            destination,
            origin,
            destination,
            "2026-03-22",
            45.0,
            8.0,
            "2026-03-22T00:00:00+00:00",
        ),
    )
    return int(cursor.lastrowid)


def test_registry_lists_four_read_only_tools() -> None:
    registry = build_default_registry()
    tools = registry.list_tools()

    assert len(tools) == 4
    assert {tool.name for tool in tools} == {
        "corkysoft.profitability_summary",
        "corkysoft.dispatch_recommendations",
        "corkysoft.operations_diary_summary",
        "corkysoft.quote_guidance_preview",
    }
    assert all(tool.read_only for tool in tools)

    envelope = _registry_envelope()
    assert envelope["ok"] is True
    assert len(envelope["tools"]) == 4


def test_call_tool_rejects_blank_name() -> None:
    response = _call_tool("", {})

    assert response["ok"] is False
    assert response["error"]["code"] == "input_error"
    assert "tool name is required" in response["error"]["message"]


def test_call_tool_reports_unknown_tool() -> None:
    response = _call_tool("corkysoft.unknown_tool", {})

    assert response["ok"] is False
    assert response["error"]["code"] == "tool_error"
    assert "Unknown tool" in response["error"]["message"]


def test_mcp_main_defaults_to_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_bridge() -> int:
        calls.append("bridge")
        return 7

    def fake_server() -> None:
        calls.append("server")

    monkeypatch.setattr(mcp_main, "run_bridge", fake_bridge)
    monkeypatch.setattr(mcp_main, "run_fastmcp", fake_server)

    result = mcp_main.main([])

    assert result == 7
    assert calls == ["bridge"]


def test_mcp_main_rejects_conflicting_transport_flags(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(mcp_main, "run_bridge", lambda: 0)
    monkeypatch.setattr(mcp_main, "run_fastmcp", lambda: None)

    result = mcp_main.main(["--bridge", "--server"])

    assert result == 2
    captured = capsys.readouterr()
    assert "Choose only one transport" in captured.err


def test_build_fastmcp_server_raises_clear_error_when_sdk_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_modules = dict(sys.modules)
    fake_mcp = ModuleType("mcp")
    fake_server_module = ModuleType("mcp.server")
    monkeypatch.setitem(sys.modules, "mcp", fake_mcp)
    monkeypatch.setitem(sys.modules, "mcp.server", fake_server_module)
    sys.modules.pop("mcp.server.fastmcp", None)

    try:
        with pytest.raises(RuntimeError, match="Optional MCP transport dependency missing"):
            build_fastmcp_server()
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)


def test_quote_guidance_preview_tool_returns_benchmark_overlay(tmp_path: Path) -> None:
    db_path = tmp_path / "quote.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE historical_jobs (
                id INTEGER PRIMARY KEY,
                price_per_m3 REAL,
                revenue_total REAL,
                volume_m3 REAL,
                final_cost REAL,
                origin TEXT,
                destination TEXT,
                origin_postcode TEXT,
                destination_postcode TEXT
            );
            """
        )
        conn.executemany(
            """
            INSERT INTO historical_jobs (
                price_per_m3,
                revenue_total,
                volume_m3,
                final_cost,
                origin,
                destination,
                origin_postcode,
                destination_postcode
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (240.0, 2400.0, 10.0, 1800.0, "Brisbane", "Gold Coast", "4000", "4217"),
                (255.0, 2550.0, 10.0, 1900.0, "Gold Coast", "Brisbane", "4217", "4000"),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    response = _call_tool(
        "corkysoft.quote_guidance_preview",
        {
            "db_path": str(db_path),
            "origin_resolved": "Brisbane",
            "destination_resolved": "Gold Coast",
            "origin_postcode": "4000",
            "destination_postcode": "4217",
            "cubic_m": 10.0,
            "current_quote_total": 2500.0,
        },
    )

    assert response["ok"] is True
    result = response["result"]
    assert result["benchmark_available"] is True
    assert result["benchmark_job_count"] == 2
    assert result["recommended_quote_total"] is not None


def test_dispatch_and_diary_tools_return_read_only_summaries(tmp_path: Path) -> None:
    db_path = tmp_path / "ops.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        ensure_dashboard_tables(conn)
        conn.execute("ALTER TABLE jobs ADD COLUMN job_number TEXT")
        upsert_truck(conn, truck_id="TRK-BACKHAUL", name="Backhaul Truck", capacity_m3=50.0)

        signal_job_id = _job(conn, "Signal Load", "Alpha", "Beta")
        signal_segment = ensure_segment(
            conn,
            job_id=signal_job_id,
            segment_sequence=1,
            from_location="Alpha",
            to_location="Beta",
            planned_start="2026-03-21T08:00:00+00:00",
            planned_end="2026-03-21T12:00:00+00:00",
        )
        allocate_inventory_to_segment(
            conn,
            segment_id=int(signal_segment["id"]),
            inventory_item_id=int(
                upsert_inventory_item(conn, name="Signal Item", quantity=5, architecture="general")["id"]
            ),
            quantity=5,
            status="assigned",
        )
        conn.execute(
            "UPDATE shipments SET truck_id = ?, from_location = ?, to_location = ? WHERE segment_id = ?",
            ("TRK-BACKHAUL", "Alpha", "Beta", int(signal_segment["id"])),
        )

        item = upsert_inventory_item(conn, name="Blankets", quantity=10, architecture="general")
        job_id = _job(conn, "Backhaul Client", "Alpha", "Beta")
        conn.execute("UPDATE jobs SET job_number = ? WHERE id = ?", ("BACKHAUL-1", job_id))
        segment = ensure_segment(
            conn,
            job_id=job_id,
            segment_sequence=1,
            planned_start="2026-03-22T08:00:00+00:00",
            planned_end="2026-03-22T12:00:00+00:00",
        )
        upsert_inventory_requirement(
            conn,
            job_id=job_id,
            segment_id=int(segment["id"]),
            inventory_item_id=int(item["id"]),
            requirement_name="Blankets",
            required_quantity=2,
            substitution_allowed=False,
            architecture="general",
        )
        allocate_inventory_to_segment(
            conn,
            segment_id=int(segment["id"]),
            inventory_item_id=int(item["id"]),
            quantity=2,
        )
        upsert_job_operational_signal(
            conn,
            job_number="BACKHAUL-1",
            origin="Alpha",
            destination="Beta",
            estimated_volume_m3=8.0,
            source="planning",
        )
        upsert_operations_diary_task(
            conn,
            job_id=job_id,
            task_date="2026-03-22",
            title="Call customer before invoicing",
        )
        upsert_customer_invoice_review(
            conn,
            job_id=job_id,
            invoice_status="ready_to_invoice",
            invoice_reference="INV-1001",
            invoice_amount=2400.0,
        )
        conn.commit()
    finally:
        conn.close()

    dispatch_response = _call_tool(
        "corkysoft.dispatch_recommendations",
        {"db_path": str(db_path), "job_id": job_id},
    )
    assert dispatch_response["ok"] is True
    assert dispatch_response["result"]["totalCount"] == 1
    assert dispatch_response["result"]["opportunities"][0]["opportunityType"] == "backhaul_share_candidate"

    diary_response = _call_tool(
        "corkysoft.operations_diary_summary",
        {
            "db_path": str(db_path),
            "anchor_date": "2026-03-22",
            "view_mode": "day",
            "focus_job_id": job_id,
        },
    )
    assert diary_response["ok"] is True
    assert diary_response["result"]["summary"]["jobCount"] == 1
    assert diary_response["result"]["jobs"][0]["invoiceStatus"] == "ready_to_invoice"


def test_profitability_summary_tool_returns_model_and_validation(tmp_path: Path) -> None:
    db_path = tmp_path / "profit.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE historical_jobs (
                id INTEGER PRIMARY KEY,
                job_date TEXT,
                client TEXT,
                corridor_display TEXT,
                price_per_m3 REAL,
                revenue_total REAL,
                revenue REAL,
                volume_m3 REAL,
                volume REAL,
                distance_km REAL,
                final_cost REAL,
                origin TEXT,
                destination TEXT,
                origin_postcode TEXT,
                destination_postcode TEXT,
                created_at TEXT,
                updated_at TEXT
            );
            """
        )
        df = pd.DataFrame(
            {
                "date": [
                    "2025-01-15",
                    "2025-02-20",
                    "2025-03-12",
                    "2025-04-01",
                    "2025-05-18",
                    "2025-06-07",
                    "2025-07-22",
                    "2025-08-11",
                ],
                "origin": ["Brisbane"] * 4 + ["Gold Coast"] * 4,
                "destination": ["Gold Coast"] * 4 + ["Brisbane"] * 4,
                "volume_m3": [10, 11, 9, 10, 10, 11, 9, 10],
                "revenue_total": [2600, 2750, 2500, 2680, 2550, 2670, 2490, 2610],
                "distance_km": [78, 78, 78, 78, 78, 78, 78, 78],
                "final_cost": [1800, 1820, 1785, 1810, 1775, 1795, 1760, 1780],
                "client": ["Client A"] * 8,
                "corridor_display": ["Brisbane -> Gold Coast"] * 8,
            }
        )
        inserted, skipped = import_historical_jobs_from_dataframe(conn, df)
        assert inserted == 8
        assert skipped == 0
        conn.commit()
    finally:
        conn.close()

    response = _call_tool(
        "corkysoft.profitability_summary",
        {"db_path": str(db_path), "preview_max_corridors": 2},
    )

    assert response["ok"] is True
    result = response["result"]
    assert result["jobCount"] == 8
    assert result["model"]["fittedJobCount"] == 8
    assert result["validation"]["trustLabel"] in {"reviewable", "caution", "low_support", "insufficient_data"}
    assert isinstance(result["preview"], list)
