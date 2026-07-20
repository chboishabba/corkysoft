from __future__ import annotations

import sqlite3

from analytics.dashboard_layouts import (
    find_layout_by_tab,
    get_dashboard_role_layouts,
    missing_recommended_primary_tabs,
    resolve_dashboard_layout,
    upsert_dashboard_role_layout,
)
from analytics.db import ensure_global_parameters_table


TABS = [
    "Quote",
    "Pricing Intelligence",
    "Network",
    "Operations",
    "Admin",
]


def test_role_layout_defaults_bootstrap() -> None:
    conn = sqlite3.connect(":memory:")
    ensure_global_parameters_table(conn)

    layouts = get_dashboard_role_layouts(conn, available_tabs=TABS)
    dispatcher = next(item for item in layouts if item["roleKey"] == "dispatcher")
    assert dispatcher["defaultLandingTab"] == "Operations"
    assert "Network" in dispatcher["primaryTabs"]
    assert "Admin" in dispatcher["hiddenTabs"]
    assert "Pricing Intelligence" in dispatcher["hiddenTabs"]


def test_resolve_dashboard_layout_does_not_reveal_hidden_tab_from_query_param() -> None:
    conn = sqlite3.connect(":memory:")
    ensure_global_parameters_table(conn)
    layout = next(
        item for item in get_dashboard_role_layouts(conn, available_tabs=TABS) if item["roleKey"] == "dispatcher"
    )
    resolved = resolve_dashboard_layout(
        available_tabs=TABS,
        layout=layout,
        requested_tab="Admin",
    )
    assert resolved["landingTab"] == "Operations"
    assert "Admin" not in resolved["tabOrder"]


def test_resolve_dashboard_layout_still_allows_visible_requested_tab() -> None:
    conn = sqlite3.connect(":memory:")
    ensure_global_parameters_table(conn)
    layout = next(
        item for item in get_dashboard_role_layouts(conn, available_tabs=TABS) if item["roleKey"] == "dispatcher"
    )
    resolved = resolve_dashboard_layout(
        available_tabs=TABS,
        layout=layout,
        requested_tab="Network",
    )
    assert resolved["landingTab"] == "Network"
    assert "Network" in resolved["tabOrder"]


def test_resolve_dashboard_layout_for_primary_only_role_shows_primary_tabs_by_default() -> None:
    conn = sqlite3.connect(":memory:")
    ensure_global_parameters_table(conn)
    layout = next(
        item for item in get_dashboard_role_layouts(conn, available_tabs=TABS) if item["roleKey"] == "dispatcher"
    )
    resolved = resolve_dashboard_layout(
        available_tabs=TABS,
        layout=layout,
    )
    assert resolved["tabOrder"] == ["Operations", "Network", "Quote"]


def test_resolve_dashboard_layout_applies_primary_only_hiding_even_with_stale_session_hidden_tabs() -> None:
    conn = sqlite3.connect(":memory:")
    ensure_global_parameters_table(conn)
    layout = next(
        item for item in get_dashboard_role_layouts(conn, available_tabs=TABS) if item["roleKey"] == "dispatcher"
    )
    resolved = resolve_dashboard_layout(
        available_tabs=TABS,
        layout=layout,
        session_hidden_tabs=["Admin"],
    )
    assert resolved["tabOrder"] == ["Operations", "Network", "Quote"]


def test_upsert_dashboard_role_layout_updates_defaults() -> None:
    conn = sqlite3.connect(":memory:")
    ensure_global_parameters_table(conn)
    updated = upsert_dashboard_role_layout(
        conn,
        role_key="dispatcher",
        default_landing_tab="Network",
        primary_tabs=["Network", "Operations"],
        hidden_tabs=["Pricing Intelligence"],
        available_tabs=TABS,
    )
    assert updated["defaultLandingTab"] == "Network"
    assert updated["primaryTabs"] == ["Network", "Operations"]


def test_missing_recommended_primary_tabs_detects_stale_dispatcher_layout() -> None:
    conn = sqlite3.connect(":memory:")
    ensure_global_parameters_table(conn)
    updated = upsert_dashboard_role_layout(
        conn,
        role_key="dispatcher",
        default_landing_tab="Operations",
        primary_tabs=["Operations", "Network"],
        hidden_tabs=[],
        available_tabs=TABS,
    )
    missing = missing_recommended_primary_tabs(
        role_key="dispatcher",
        layout=updated,
        available_tabs=TABS,
    )
    assert "Quote" in missing


def test_find_layout_by_tab_does_not_infer_role_from_shared_surfaces() -> None:
    conn = sqlite3.connect(":memory:")
    ensure_global_parameters_table(conn)
    layouts = get_dashboard_role_layouts(conn, available_tabs=TABS)
    assert find_layout_by_tab(layouts, "Quote") is None
    assert find_layout_by_tab(layouts, "Operations") is None
    assert find_layout_by_tab(layouts, "Network") is None
    assert find_layout_by_tab(layouts, "Nonexistent tab") is None


def test_find_layout_by_tab_returns_unique_primary_match() -> None:
    layouts = [
        {
            "roleKey": "dispatcher",
            "label": "Dispatcher",
            "primaryTabs": ["Operations"],
            "hiddenTabs": ["Admin"],
        },
        {
            "roleKey": "labor_planner",
            "label": "Labor Planner / Staff Coordinator",
            "primaryTabs": ["Network"],
            "hiddenTabs": [],
        },
    ]
    selected = find_layout_by_tab(layouts, "Operations")
    assert selected is not None
    assert selected["roleKey"] == "dispatcher"


def test_find_layout_by_tab_rejects_ambiguous_primary_matches() -> None:
    layouts = [
        {
            "roleKey": "dispatcher",
            "label": "Dispatcher",
            "primaryTabs": ["Operations"],
            "hiddenTabs": ["Admin"],
        },
        {
            "roleKey": "labor_planner",
            "label": "Labor Planner / Staff Coordinator",
            "primaryTabs": ["Operations"],
            "hiddenTabs": [],
        },
    ]
    assert find_layout_by_tab(layouts, "Operations") is None
