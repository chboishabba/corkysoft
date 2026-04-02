from __future__ import annotations

from typing import Any

ANALYTICS_SHELL_TABS = {
    "Pricing Intelligence",
}
LABOR_SHELL_TABS = set()
COMMERCIAL_SHELL_TABS = {
    "Quote",
}
FLEET_SHELL_TABS = set()
OPERATIONS_SHELL_TABS = {
    "Network",
    "Operations",
}
ADMIN_SHELL_TABS = {
    "Admin",
}

def resolve_dashboard_shell(active_tab: str | None) -> dict[str, Any]:
    if active_tab in ANALYTICS_SHELL_TABS:
        return {
            "title": "Pricing Intelligence",
            "caption": "Analyse yield, spot loss-leading corridors, and run optimization routines.",
            "sidebar_heading": "Filters",
            "sidebar_caption": None,
            "collapse_analytics_sidebar": False,
        }

    if active_tab in OPERATIONS_SHELL_TABS:
        return {
            "title": "Operations & Network Control",
            "caption": "Guided network overview and operations fulfillment workflow.",
            "sidebar_heading": "Workflow support",
            "sidebar_caption": "Filters apply where relevant to network segments.",
            "collapse_analytics_sidebar": True,
        }

    if active_tab in COMMERCIAL_SHELL_TABS:
        return {
            "title": "Quote Workspace",
            "caption": "Build and manage quotes with margin insights.",
            "sidebar_heading": "Quoting support",
            "sidebar_caption": "Analytics filters context is available minimally.",
            "collapse_analytics_sidebar": True,
        }

    if active_tab in ADMIN_SHELL_TABS:
        return {
            "title": "Administration",
            "caption": "Role and system management.",
            "sidebar_heading": "Admin",
            "sidebar_caption": "System level configurations",
            "collapse_analytics_sidebar": True,
        }

    return {
        "title": "Decision System",
        "caption": "Guided workflow workspace.",
        "sidebar_heading": "Workflow support",
        "sidebar_caption": "",
        "collapse_analytics_sidebar": True,
    }
