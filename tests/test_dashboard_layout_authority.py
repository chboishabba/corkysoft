from __future__ import annotations

from analytics.dashboard_layouts import ROLE_LAYOUT_DEFAULTS, resolve_dashboard_layout


AVAILABLE_TABS = ["Quote", "Pricing Intelligence", "Network", "Operations", "Admin"]


def _layout(role_key: str) -> dict[str, object]:
    values = ROLE_LAYOUT_DEFAULTS[role_key]
    return {
        "roleKey": role_key,
        "label": values["label"],
        "defaultLandingTab": values["defaultLandingTab"],
        "primaryTabs": list(values["primaryTabs"]),
        "hiddenTabs": list(values["hiddenTabs"]),
    }


def test_primary_only_role_cannot_reveal_hidden_tabs_with_show_all() -> None:
    resolved = resolve_dashboard_layout(
        available_tabs=AVAILABLE_TABS,
        layout=_layout("dispatcher"),
        requested_tab="Admin",
        session_primary_tabs=AVAILABLE_TABS,
        session_hidden_tabs=[],
        session_landing_tab="Admin",
        show_all_tabs=True,
    )

    assert "Admin" not in resolved["tabOrder"]
    assert "Pricing Intelligence" not in resolved["tabOrder"]
    assert resolved["landingTab"] == "Operations"
    assert set(resolved["hiddenTabs"]) == {"Admin", "Pricing Intelligence"}


def test_restricted_role_stale_session_cannot_expand_beyond_primary_tabs() -> None:
    resolved = resolve_dashboard_layout(
        available_tabs=AVAILABLE_TABS,
        layout=_layout("inventory_supplier"),
        requested_tab="Network",
        session_primary_tabs=AVAILABLE_TABS,
        session_hidden_tabs=[],
        session_landing_tab="Network",
        show_all_tabs=True,
    )

    assert resolved["tabOrder"] == ["Operations"]
    assert resolved["landingTab"] == "Operations"


def test_system_admin_may_explicitly_show_all_tabs() -> None:
    resolved = resolve_dashboard_layout(
        available_tabs=AVAILABLE_TABS,
        layout=_layout("system_rollout_admin"),
        requested_tab="Quote",
        session_hidden_tabs=["Quote"],
        show_all_tabs=True,
    )

    assert set(resolved["tabOrder"]) == set(AVAILABLE_TABS)
    assert resolved["landingTab"] == "Quote"
