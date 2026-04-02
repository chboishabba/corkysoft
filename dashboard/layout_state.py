from __future__ import annotations

from typing import Any

import streamlit as st

LAYOUT_PENDING_KEY = "dashboard_layout_pending_reset"


def layout_defaults_from_layout(layout: dict[str, Any]) -> dict[str, Any]:
    return {
        "primaryTabs": list(layout.get("primaryTabs", [])),
        "hiddenTabs": list(layout.get("hiddenTabs", [])),
        "landingTab": layout.get("defaultLandingTab"),
        "showAll": False,
    }


def hydrate_role_layout_session(
    selected_role_layout: dict[str, Any],
    *,
    force_reset: bool = False,
) -> None:
    label = selected_role_layout["label"]
    pending = st.session_state.pop(LAYOUT_PENDING_KEY, None)

    def _apply(values: dict[str, Any]) -> None:
        st.session_state["dashboard_session_primary_tabs"] = list(values["primaryTabs"])
        st.session_state["dashboard_session_hidden_tabs"] = list(values["hiddenTabs"])
        st.session_state["dashboard_session_landing_tab"] = values.get("landingTab")
        st.session_state["dashboard_show_all_tabs"] = bool(values.get("showAll", False))
        st.session_state["dashboard_active_role_last"] = label

    if pending:
        _apply(pending)
        return

    if force_reset:
        _apply(layout_defaults_from_layout(selected_role_layout))
        return

    last_label = st.session_state.get("dashboard_active_role_last")
    if last_label != label:
        _apply(layout_defaults_from_layout(selected_role_layout))
