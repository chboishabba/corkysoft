from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import inspect
import streamlit as st


@dataclass
class TabRegistryResult:
    tab_map: Dict[str, Any]
    requested_tab: str
    requested_tab_index: int
    tab_order: List[str]


def build_tab_map(
    tab_labels: List[str],
    requested_tab: str,
    params: Dict[str, List[str]],
    tabs_placeholder: "st.delta_generator.DeltaGenerator",
) -> TabRegistryResult:
    requested_tab_index = tab_labels.index(requested_tab)
    can_assign_tab_key = False
    try:
        can_assign_tab_key = "key" in inspect.signature(st.tabs).parameters
    except (TypeError, ValueError):
        can_assign_tab_key = False

    tab_order = tab_labels
    if can_assign_tab_key:
        tabs_key = "dashboard_active_tab"
        view_param_requested = "view" in params
        if tabs_key not in st.session_state or (
            view_param_requested and st.session_state.get(tabs_key) != requested_tab_index
        ):
            st.session_state[tabs_key] = requested_tab_index
        with tabs_placeholder:
            streamlit_tabs = st.tabs(tab_order, key=tabs_key)
    else:
        with tabs_placeholder:
            streamlit_tabs = st.tabs(tab_order)

    tab_map = {label: tab for label, tab in zip(tab_order, streamlit_tabs)}
    return TabRegistryResult(
        tab_map=tab_map,
        requested_tab=requested_tab,
        requested_tab_index=requested_tab_index,
        tab_order=tab_order,
    )
