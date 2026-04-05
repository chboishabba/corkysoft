from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pandas as pd
from analytics.db import ensure_dashboard_tables, upsert_truck, upsert_worker
from analytics.adaptive_policy import AdaptivePolicyProposalItem, create_adaptive_policy_proposal
from analytics.auth import upsert_dashboard_user
from analytics.operations_assignment import assign_segment_resources, ensure_segment
from analytics.operations_diary import upsert_operations_diary_task
from corkysoft.pricing import PRICING_MODELS
from corkysoft.repo import ensure_schema
from dashboard.shell_signals import (
    build_admin_shell_signal_bundle,
    build_network_shell_signal_bundle,
    build_operations_shell_signal_bundle,
    build_pricing_shell_signal_bundle,
    build_quote_shell_signal_bundle,
    build_shell_signal_bundle,
)


def _live_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE truck_positions (
            truck_id TEXT PRIMARY KEY,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            status TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            heading REAL,
            speed_kph REAL,
            notes TEXT
        );
        CREATE TABLE active_routes (
            route_id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER,
            truck_id TEXT NOT NULL UNIQUE,
            origin_lat REAL NOT NULL,
            origin_lon REAL NOT NULL,
            dest_lat REAL NOT NULL,
            dest_lon REAL NOT NULL,
            progress REAL NOT NULL,
            eta TEXT,
            status TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            notes TEXT,
            route_geometry TEXT,
            started_at TEXT,
            travel_seconds REAL
        );
        """
    )
    return conn


def _operations_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    ensure_dashboard_tables(conn)
    return conn


def _quote_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    ensure_dashboard_tables(conn)
    return conn


def test_build_shell_signal_bundle_returns_scaffold_contract_for_operations() -> None:
    bundle = build_shell_signal_bundle("Operations")

    assert bundle.scope_label == "Operations"
    assert bundle.freshness_state == "scaffold"
    assert bundle.decision_grade == "placeholder"
    assert bundle.owner == "Operations control"
    assert len(bundle.metrics) == 3
    assert bundle.alert.title == "Operations Alert"
    assert bundle.metrics[0].signal_id == "ops_dispatch_fulfillment"


def test_build_shell_signal_bundle_returns_unknown_contract_for_unregistered_scope() -> None:
    bundle = build_shell_signal_bundle("Unregistered")

    assert bundle.scope_label == "Unregistered"
    assert bundle.freshness_state == "scaffold"
    assert bundle.alert.title == "Signal State Unknown"


def test_build_network_shell_signal_bundle_reports_fresh_live_telemetry() -> None:
    conn = _live_conn()
    now = datetime.now(UTC)
    conn.execute(
        "INSERT INTO truck_positions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("TRK-1", -27.47, 153.02, "en_route", now.isoformat(), None, 72.0, None),
    )
    conn.execute(
        """
        INSERT INTO active_routes (
            job_id, truck_id, origin_lat, origin_lon, dest_lat, dest_lon, progress,
            eta, status, updated_at, notes, route_geometry, started_at, travel_seconds
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            101,
            "TRK-1",
            -27.47,
            153.02,
            -33.86,
            151.20,
            0.4,
            (now + timedelta(hours=6)).isoformat(),
            "en_route",
            now.isoformat(),
            None,
            None,
            now.isoformat(),
            3600.0,
        ),
    )

    bundle = build_network_shell_signal_bundle(conn)

    assert bundle.freshness_state == "fresh"
    assert bundle.decision_grade == "advisory"
    assert bundle.source == "analytics.live_data"
    assert bundle.metrics[0].label == "Active Nodes"
    assert bundle.metrics[0].value == "2"
    assert bundle.metrics[1].value == "1"
    assert bundle.alert.title == "Network Status"


def test_build_network_shell_signal_bundle_reports_stale_telemetry() -> None:
    conn = _live_conn()
    stale = datetime.now(UTC) - timedelta(minutes=30)
    conn.execute(
        "INSERT INTO truck_positions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("TRK-2", -27.47, 153.02, "delayed", stale.isoformat(), None, 0.0, None),
    )
    conn.execute(
        """
        INSERT INTO active_routes (
            job_id, truck_id, origin_lat, origin_lon, dest_lat, dest_lon, progress,
            eta, status, updated_at, notes, route_geometry, started_at, travel_seconds
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            202,
            "TRK-2",
            -27.47,
            153.02,
            -33.86,
            151.20,
            0.8,
            stale.isoformat(),
            "delayed",
            stale.isoformat(),
            None,
            None,
            stale.isoformat(),
            3600.0,
        ),
    )

    bundle = build_network_shell_signal_bundle(conn)

    assert bundle.freshness_state == "stale"
    assert bundle.alert.title == "Telemetry Stale"
    assert bundle.metrics[1].delta == "Telemetry stale"


def test_build_network_shell_signal_bundle_reports_unknown_when_empty() -> None:
    conn = _live_conn()

    bundle = build_network_shell_signal_bundle(conn)

    assert bundle.freshness_state == "unknown"
    assert bundle.alert.title == "Telemetry Unavailable"
    assert bundle.metrics[0].value == "0"
    assert bundle.metrics[2].value == "Unknown"


def test_build_operations_shell_signal_bundle_reports_fresh_summary_state() -> None:
    conn = _operations_conn()
    upsert_truck(conn, truck_id="TRK-OPS", name="Truck Ops", capacity_m3=40.0)
    worker = upsert_worker(conn, name="Ops Worker")
    job_id = conn.execute(
        """
        INSERT INTO jobs (
            client, origin, destination, origin_resolved, destination_resolved, job_date,
            distance_km, volume_m3, origin_lat, origin_lon, dest_lat, dest_lon, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "Ops Client",
            "Brisbane",
            "Cairns",
            "Brisbane",
            "Cairns",
            "2026-03-20",
            1700.0,
            36.0,
            -27.47,
            153.02,
            -16.92,
            145.77,
            "2026-03-12T00:00:00+00:00",
        ),
    ).lastrowid
    segment = ensure_segment(
        conn,
        job_id=int(job_id),
        segment_sequence=1,
        planned_start="2026-03-20T08:00:00+00:00",
        planned_end="2026-03-20T12:00:00+00:00",
    )
    assign_segment_resources(
        conn,
        segment_id=int(segment["id"]),
        truck_ids=["TRK-OPS"],
        worker_assignments=[{"workerId": int(worker["id"])}],
    )
    upsert_operations_diary_task(
        conn,
        job_id=int(job_id),
        task_date="2026-03-20",
        title="Review site handoff",
    )

    bundle = build_operations_shell_signal_bundle(conn)

    assert bundle.freshness_state == "fresh"
    assert bundle.decision_grade == "advisory"
    assert bundle.metrics[0].label == "Dispatch Fulfillment"
    assert bundle.metrics[0].value == "100%"
    assert bundle.metrics[1].value == "1"
    assert bundle.metrics[2].value == "1"
    assert bundle.alert.title == "Operational Follow-Through Pending"


def test_build_operations_shell_signal_bundle_reports_unknown_on_failure() -> None:
    conn = sqlite3.connect(":memory:")
    bundle = build_operations_shell_signal_bundle(conn)

    assert bundle.freshness_state == "fresh"
    assert bundle.alert.title == "Operations Status"
    assert bundle.metrics[0].value == "100%"
    assert bundle.metrics[1].value == "0"


def test_build_quote_shell_signal_bundle_reports_saved_quote_history() -> None:
    conn = _quote_conn()
    conn.execute(
        """
        INSERT INTO quotes (
            created_at, quote_date, origin_input, destination_input,
            origin_resolved, destination_resolved, origin_lon, origin_lat, dest_lon, dest_lat,
            distance_km, duration_hr, cubic_m, pricing_model, base_subtotal, base_components,
            modifiers_applied, modifiers_total, seasonal_multiplier, seasonal_label,
            total_before_margin, margin_percent, client_id, client_display,
            manual_quote, final_quote, summary
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "2026-04-02T00:00:00+00:00",
            "2026-04-02",
            "Brisbane",
            "Sydney",
            "Brisbane",
            "Sydney",
            153.02,
            -27.47,
            151.21,
            -33.87,
            900.0,
            10.0,
            30.0,
            PRICING_MODELS[0].id,
            800.0,
            "{}",
            "[]",
            0.0,
            1.0,
            "Base season",
            1000.0,
            20.0,
            None,
            "Acme",
            None,
            1200.0,
            "summary",
        ),
    )

    bundle = build_quote_shell_signal_bundle(conn)

    assert bundle.freshness_state == "fresh"
    assert bundle.metrics[0].value == "1"
    assert bundle.metrics[1].value == "20.0%"
    assert bundle.metrics[2].value == "1"
    assert bundle.alert.title == "Commercial Snapshot"


def test_build_quote_shell_signal_bundle_reports_empty_history() -> None:
    conn = _quote_conn()

    bundle = build_quote_shell_signal_bundle(conn)

    assert bundle.freshness_state == "fresh"
    assert bundle.metrics[0].value == "0"
    assert bundle.alert.title == "No Saved Quotes Yet"


def test_build_pricing_shell_signal_bundle_reports_filtered_margin_state() -> None:
    filtered_df = pd.DataFrame(
        {
            "price_per_m3": [210.0, 180.0, 140.0],
            "break_even_per_m3": [150.0, 150.0, 150.0],
            "margin_per_m3": [60.0, 30.0, -10.0],
        }
    )

    bundle = build_pricing_shell_signal_bundle(filtered_df, break_even_value=150.0)

    assert bundle.freshness_state == "fresh"
    assert bundle.metrics[0].value == "$30"
    assert bundle.metrics[1].value == "1"
    assert bundle.metrics[1].delta == "33% of priced jobs"
    assert bundle.metrics[2].value == "$180"
    assert bundle.alert.title == "Break-Even Pressure Detected"
    assert bundle.alert.severity == "critical"


def test_build_pricing_shell_signal_bundle_reports_empty_scope() -> None:
    filtered_df = pd.DataFrame(columns=["price_per_m3", "margin_per_m3"])

    bundle = build_pricing_shell_signal_bundle(filtered_df, break_even_value=150.0)

    assert bundle.freshness_state == "fresh"
    assert bundle.metrics[0].value == "Unknown"
    assert bundle.metrics[1].value == "0"
    assert bundle.metrics[2].value == "Unknown"
    assert bundle.alert.title == "No Priced Jobs In Scope"


def test_build_pricing_shell_signal_bundle_reports_unknown_on_dataset_error() -> None:
    filtered_df = pd.DataFrame()

    bundle = build_pricing_shell_signal_bundle(
        filtered_df,
        break_even_value=150.0,
        dataset_error="missing upstream selection",
    )

    assert bundle.freshness_state == "unknown"
    assert bundle.metrics[0].value == "Unknown"
    assert bundle.metrics[1].value == "Unknown"
    assert bundle.alert.title == "Pricing Dataset Unavailable"


def test_build_pricing_shell_signal_bundle_fails_closed_on_malformed_scope() -> None:
    filtered_df = pd.DataFrame({"margin_per_m3": [20.0, 15.0]})

    bundle = build_pricing_shell_signal_bundle(filtered_df, break_even_value=150.0)

    assert bundle.freshness_state == "unknown"
    assert bundle.metrics[0].value == "Unknown"
    assert bundle.metrics[1].value == "Unknown"
    assert bundle.alert.title == "Pricing Dataset Unavailable"


def test_build_admin_shell_signal_bundle_reports_missing_admin_coverage() -> None:
    conn = _quote_conn()

    bundle = build_admin_shell_signal_bundle(conn)

    assert bundle.freshness_state == "unknown"
    assert bundle.metrics[0].value == "0"
    assert bundle.metrics[1].value == "0"
    assert bundle.metrics[2].value == "Unknown"
    assert bundle.alert.title == "No Active Admin Coverage"
    assert bundle.alert.severity == "critical"


def test_build_admin_shell_signal_bundle_does_not_bootstrap_governance_tables_on_read() -> None:
    conn = _quote_conn()
    before_tables = {
        row["name"]
        for row in conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name IN (
                'kent_job_tenders',
                'kent_tender_override_reason_codes',
                'adaptive_policy_proposals'
            )
            """
        ).fetchall()
    }

    build_admin_shell_signal_bundle(conn)

    after_tables = {
        row["name"]
        for row in conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name IN (
                'kent_job_tenders',
                'kent_tender_override_reason_codes',
                'adaptive_policy_proposals'
            )
            """
        ).fetchall()
    }

    assert before_tables == set()
    assert after_tables == set()


def test_build_admin_shell_signal_bundle_reports_governance_review_pressure() -> None:
    conn = _quote_conn()
    upsert_dashboard_user(
        conn,
        email="admin@example.com",
        display_name="Admin",
        role_key="system_rollout_admin",
        active=True,
        allowed_role_keys=("dispatcher", "system_rollout_admin"),
    )
    create_adaptive_policy_proposal(
        conn,
        proposal_type="manual",
        actor="ops_manager",
        items=[
            AdaptivePolicyProposalItem(
                key="adaptive.weather_risk_multiplier",
                current_value=1.0,
                proposed_value=1.05,
                target_value=1.05,
                max_delta=0.05,
            )
        ],
    )

    bundle = build_admin_shell_signal_bundle(conn)

    assert bundle.freshness_state == "unknown"
    assert bundle.metrics[0].value == "1"
    assert bundle.metrics[1].value == "1"
    assert bundle.metrics[2].value == "1"
    assert bundle.alert.title == "Governance Review Pending"
    assert bundle.alert.severity == "warning"
