from __future__ import annotations

import sqlite3

from analytics.dashboard_layouts import (
    get_dashboard_role_layouts,
    missing_recommended_primary_tabs,
    resolve_dashboard_layout,
    upsert_dashboard_role_layout,
)
from analytics.db import ensure_global_parameters_table


TABS = [
    "Histogram",
    "Dispatch",
    "Planner",
    "Operations",
    "Calls",
    "Fleet",
    "Quote builder",
]


def test_role_layout_defaults_bootstrap() -> None:
    conn = sqlite3.connect(":memory:")
    ensure_global_parameters_table(conn)

    layouts = get_dashboard_role_layouts(conn, available_tabs=TABS)
    dispatcher = next(item for item in layouts if item["roleKey"] == "dispatcher")
    assert dispatcher["defaultLandingTab"] == "Dispatch"
    assert "Planner" in dispatcher["primaryTabs"]


def test_resolve_dashboard_layout_respects_query_param_and_hidden_tabs() -> None:
    conn = sqlite3.connect(":memory:")
    ensure_global_parameters_table(conn)
    layout = next(
        item for item in get_dashboard_role_layouts(conn, available_tabs=TABS) if item["roleKey"] == "dispatcher"
    )
    resolved = resolve_dashboard_layout(
        available_tabs=TABS,
        layout=layout,
        requested_tab="Quote builder",
    )
    assert resolved["landingTab"] == "Quote builder"
    assert "Quote builder" in resolved["tabOrder"]


def test_upsert_dashboard_role_layout_updates_defaults() -> None:
    conn = sqlite3.connect(":memory:")
    ensure_global_parameters_table(conn)
    updated = upsert_dashboard_role_layout(
        conn,
        role_key="dispatcher",
        default_landing_tab="Planner",
        primary_tabs=["Planner", "Dispatch"],
        hidden_tabs=["Histogram"],
        available_tabs=TABS,
    )
    assert updated["defaultLandingTab"] == "Planner"
    assert updated["primaryTabs"] == ["Planner", "Dispatch"]


def test_missing_recommended_primary_tabs_detects_stale_dispatcher_layout() -> None:
    conn = sqlite3.connect(":memory:")
    ensure_global_parameters_table(conn)
    updated = upsert_dashboard_role_layout(
        conn,
        role_key="dispatcher",
        default_landing_tab="Dispatch",
        primary_tabs=["Dispatch", "Planner", "Operations"],
        hidden_tabs=[],
        available_tabs=TABS,
    )
    missing = missing_recommended_primary_tabs(
        role_key="dispatcher",
        layout=updated,
        available_tabs=TABS,
    )
    assert "Calls" in missing
