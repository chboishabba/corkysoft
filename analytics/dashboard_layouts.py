"""Role-based dashboard layout helpers."""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Iterable, Sequence

from analytics.db.parameters import ensure_global_parameters_table, get_parameter_text, set_parameter_text

DASHBOARD_ROLE_LAYOUTS_KEY = "dashboard.role_layouts.v1"

ROLE_LAYOUT_DEFAULTS: dict[str, dict[str, Any]] = {
    "estimator": {
        "label": "Estimator",
        "defaultLandingTab": "Quote",
        "primaryTabs": ["Quote", "Pricing Intelligence", "Network"],
        "hiddenTabs": ["Admin"],
    },
    "dispatcher": {
        "label": "Dispatcher",
        "defaultLandingTab": "Operations",
        "primaryTabs": ["Operations", "Network", "Quote"],
        "hiddenTabs": ["Admin", "Pricing Intelligence"],
    },
    "fleet_operations_manager": {
        "label": "Fleet / Operations Manager",
        "defaultLandingTab": "Operations",
        "primaryTabs": ["Operations", "Network", "Pricing Intelligence"],
        "hiddenTabs": ["Admin"],
    },
    "labor_planner": {
        "label": "Labor Planner / Staff Coordinator",
        "defaultLandingTab": "Operations",
        "primaryTabs": ["Operations", "Network"],
        "hiddenTabs": ["Admin", "Pricing Intelligence", "Quote"],
    },
    "maintenance_compliance": {
        "label": "Maintenance / Compliance Coordinator",
        "defaultLandingTab": "Operations",
        "primaryTabs": ["Operations", "Network"],
        "hiddenTabs": ["Admin", "Pricing Intelligence", "Quote"],
    },
    "inventory_supplier": {
        "label": "Inventory / Supplier Coordinator",
        "defaultLandingTab": "Operations",
        "primaryTabs": ["Operations"],
        "hiddenTabs": ["Admin", "Pricing Intelligence", "Quote", "Network"],
    },
    "commercial_owner": {
        "label": "Commercial Owner",
        "defaultLandingTab": "Pricing Intelligence",
        "primaryTabs": ["Quote", "Pricing Intelligence", "Operations", "Admin"],
        "hiddenTabs": ["Network"],
    },
    "system_rollout_admin": {
        "label": "System / Rollout Admin",
        "defaultLandingTab": "Admin",
        "primaryTabs": ["Admin", "Operations", "Network", "Pricing Intelligence", "Quote"],
        "hiddenTabs": [],
    },
}
PRIMARY_ONLY_ROLE_KEYS = {
    "dispatcher",
    "fleet_operations_manager",
    "labor_planner",
    "maintenance_compliance",
    "inventory_supplier",
}


def _normalise_tabs(values: Iterable[str], available_tabs: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    allowed = set(available_tabs)
    result: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item in seen or item not in allowed:
            continue
        seen.add(item)
        result.append(item)
    return result


def _recommended_hidden_tabs(
    role_key: str,
    *,
    primary_tabs: Sequence[str],
    hidden_tabs: Sequence[str],
    available_tabs: Sequence[str],
) -> list[str]:
    combined = _normalise_tabs(hidden_tabs, available_tabs)
    if role_key not in PRIMARY_ONLY_ROLE_KEYS:
        return combined

    primary_set = set(_normalise_tabs(primary_tabs, available_tabs))
    recommended = [tab for tab in available_tabs if tab not in primary_set]
    return _normalise_tabs([*combined, *recommended], available_tabs)


def _sanitise_layout(role_key: str, payload: dict[str, Any], available_tabs: Sequence[str]) -> dict[str, Any]:
    base = ROLE_LAYOUT_DEFAULTS[role_key]
    default_tab = str(payload.get("defaultLandingTab") or base["defaultLandingTab"])
    if default_tab not in available_tabs:
        default_tab = str(base["defaultLandingTab"])
    primary_tabs = _normalise_tabs(payload.get("primaryTabs", base["primaryTabs"]), available_tabs)
    hidden_tabs = _recommended_hidden_tabs(
        role_key,
        primary_tabs=primary_tabs,
        hidden_tabs=payload.get("hiddenTabs", base["hiddenTabs"]),
        available_tabs=available_tabs,
    )
    if default_tab in hidden_tabs:
        hidden_tabs = [tab for tab in hidden_tabs if tab != default_tab]
    return {
        "roleKey": role_key,
        "label": str(base["label"]),
        "defaultLandingTab": default_tab,
        "primaryTabs": primary_tabs,
        "hiddenTabs": hidden_tabs,
    }


def missing_recommended_primary_tabs(
    *,
    role_key: str,
    layout: dict[str, Any],
    available_tabs: Sequence[str],
) -> list[str]:
    if role_key not in ROLE_LAYOUT_DEFAULTS:
        return []
    recommended = _normalise_tabs(ROLE_LAYOUT_DEFAULTS[role_key]["primaryTabs"], available_tabs)
    actual = _normalise_tabs(layout.get("primaryTabs", []), available_tabs)
    actual_set = set(actual)
    return [tab for tab in recommended if tab not in actual_set]


def get_dashboard_role_layouts(
    conn: sqlite3.Connection,
    *,
    available_tabs: Sequence[str],
) -> list[dict[str, Any]]:
    ensure_global_parameters_table(conn)
    raw = get_parameter_text(conn, DASHBOARD_ROLE_LAYOUTS_KEY)
    stored: dict[str, Any] = {}
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                stored = parsed
        except Exception:
            stored = {}
    layouts: list[dict[str, Any]] = []
    to_store: dict[str, Any] = {}
    for role_key in ROLE_LAYOUT_DEFAULTS:
        layout = _sanitise_layout(role_key, stored.get(role_key, {}), available_tabs)
        layouts.append(layout)
        to_store[role_key] = {
            "defaultLandingTab": layout["defaultLandingTab"],
            "primaryTabs": layout["primaryTabs"],
            "hiddenTabs": layout["hiddenTabs"],
        }
    if raw is None:
        set_parameter_text(
            conn,
            DASHBOARD_ROLE_LAYOUTS_KEY,
            json.dumps(to_store),
            description="Dashboard role layout defaults.",
        )
    return layouts


def find_layout_by_tab(
    layouts: Sequence[dict[str, Any]],
    tab: str | None,
) -> dict[str, Any] | None:
    if not tab:
        return None
    for layout in layouts:
        primary_tabs = set(layout.get("primaryTabs", []))
        if tab in primary_tabs:
            return layout
    for layout in layouts:
        hidden_tabs = set(layout.get("hiddenTabs", []))
        if tab in hidden_tabs:
            return layout
    return None


def upsert_dashboard_role_layout(
    conn: sqlite3.Connection,
    *,
    role_key: str,
    default_landing_tab: str,
    primary_tabs: Sequence[str],
    hidden_tabs: Sequence[str],
    available_tabs: Sequence[str],
) -> dict[str, Any]:
    if role_key not in ROLE_LAYOUT_DEFAULTS:
        raise ValueError(f"Unknown dashboard role: {role_key}")
    ensure_global_parameters_table(conn)
    raw = get_parameter_text(conn, DASHBOARD_ROLE_LAYOUTS_KEY)
    stored: dict[str, Any] = {}
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                stored = parsed
        except Exception:
            stored = {}
    stored[role_key] = {
        "defaultLandingTab": default_landing_tab,
        "primaryTabs": list(primary_tabs),
        "hiddenTabs": list(hidden_tabs),
    }
    set_parameter_text(
        conn,
        DASHBOARD_ROLE_LAYOUTS_KEY,
        json.dumps(stored),
        description="Dashboard role layout defaults.",
    )
    return _sanitise_layout(role_key, stored[role_key], available_tabs)


def resolve_dashboard_layout(
    *,
    available_tabs: Sequence[str],
    layout: dict[str, Any],
    requested_tab: str | None = None,
    session_primary_tabs: Sequence[str] | None = None,
    session_hidden_tabs: Sequence[str] | None = None,
    session_landing_tab: str | None = None,
    show_all_tabs: bool = False,
) -> dict[str, Any]:
    base_order = list(available_tabs)
    primary_tabs = _normalise_tabs(session_primary_tabs or layout.get("primaryTabs", []), base_order)
    hidden_tabs = [] if show_all_tabs else _recommended_hidden_tabs(
        str(layout.get("roleKey") or ""),
        primary_tabs=primary_tabs,
        hidden_tabs=session_hidden_tabs or layout.get("hiddenTabs", []),
        available_tabs=base_order,
    )
    visible_tabs = [tab for tab in base_order if tab not in hidden_tabs]
    ordered: list[str] = []
    for tab in primary_tabs:
        if tab in visible_tabs and tab not in ordered:
            ordered.append(tab)
    for tab in visible_tabs:
        if tab not in ordered:
            ordered.append(tab)
    landing_tab = session_landing_tab or str(layout.get("defaultLandingTab") or ordered[0])
    if requested_tab and requested_tab in ordered:
        landing_tab = requested_tab
    elif landing_tab not in ordered:
        landing_tab = ordered[0]
    return {
        "roleKey": layout["roleKey"],
        "label": layout["label"],
        "tabOrder": ordered,
        "landingTab": landing_tab,
        "primaryTabs": primary_tabs,
        "hiddenTabs": hidden_tabs,
    }


__all__ = [
    "DASHBOARD_ROLE_LAYOUTS_KEY",
    "ROLE_LAYOUT_DEFAULTS",
    "get_dashboard_role_layouts",
    "missing_recommended_primary_tabs",
    "find_layout_by_tab",
    "resolve_dashboard_layout",
    "upsert_dashboard_role_layout",
]
