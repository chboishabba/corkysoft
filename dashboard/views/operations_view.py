import streamlit as st
import pandas as pd
import sqlite3
import inspect
from typing import Any

from dashboard.components.kpi_strip import render_kpi_strip
from dashboard.components.alert_banner import render_alert_banner
from dashboard.components.data_provenance import render_signal_contract_notice
from dashboard.components.dispatch import render_dispatch_tab
from dashboard.components.dispatch_calendar import render_dispatch_calendar
from dashboard.components.crew_workflow import render_crew_workflow
from dashboard.components.customer_communications import render_customer_communications
from dashboard.components.operations_diary import render_operations_diary_tab
from dashboard.components.planner import render_planner_tab
from dashboard.components.operations import render_operations_tab
from dashboard.components.maintenance import render_fleet_tab, render_vehicle_maintenance_tab
from dashboard.components.inventory import render_inventory_tab
from dashboard.components.staff import render_staff_tab
from dashboard.components.worker_time import render_driver_shifts_tab
from dashboard.components.payroll_labor_analytics import render_payroll_labor_analytics_tab
from dashboard.shell_signals import build_operations_shell_signal_bundle
from dashboard.theme import tier_separator, hero_section
from corkysoft.operations_platform import ensure_operations_platform_schema


def _display_operations_primary_tabs(
    labels: list[str],
    requested_label: str,
    *,
    can_assign_tab_key: bool,
) -> list[str]:
    if can_assign_tab_key or requested_label not in labels or requested_label == labels[0]:
        return labels
    return [requested_label, *[label for label in labels if label != requested_label]]


def render_operations_view(
    conn: sqlite3.Connection,
    filtered_df: pd.DataFrame,
    rerun_app: Any,
    workspace_state: dict[str, Any] | None = None,
):
    ensure_operations_platform_schema(conn)

    # ── Tier 1: KPI Strip ──────────────────────────────────────────
    signal_bundle = build_operations_shell_signal_bundle(conn)
    render_signal_contract_notice(signal_bundle)
    render_kpi_strip(signal_bundle.metric_cards())

    # ── Tier 2: Alerts / Risks ─────────────────────────────────────
    render_alert_banner(
        signal_bundle.alert.message,
        severity=signal_bundle.alert.severity,
        title=signal_bundle.alert.title,
    )

    tier_separator()

    # ── Tier 3: Primary Decision View (Hero) ───────────────────────
    hero_section(
        "Operations Control",
        "Dispatch, calendar, crew execution, customer communication, planning and closure.",
    )

    primary_tab_labels = [
        "Dispatch",
        "Calendar",
        "Crew",
        "Customer Comms",
        "Planner",
        "Operations Diary",
    ]
    requested_primary_tab = str((workspace_state or {}).get("operations_tab") or primary_tab_labels[0])
    if requested_primary_tab not in primary_tab_labels:
        requested_primary_tab = primary_tab_labels[0]
    can_assign_tab_key = False
    try:
        can_assign_tab_key = "key" in inspect.signature(st.tabs).parameters
    except (TypeError, ValueError):
        can_assign_tab_key = False
    if can_assign_tab_key:
        tab_key = "operations_primary_tab"
        requested_index = primary_tab_labels.index(requested_primary_tab)
        if st.session_state.get(tab_key) != requested_index:
            st.session_state[tab_key] = requested_index
        display_tab_labels = primary_tab_labels
        primary_tabs = st.tabs(display_tab_labels, key=tab_key)
    else:
        display_tab_labels = _display_operations_primary_tabs(
            primary_tab_labels,
            requested_primary_tab,
            can_assign_tab_key=can_assign_tab_key,
        )
        primary_tabs = st.tabs(display_tab_labels)
    primary_tab_map = dict(zip(display_tab_labels, primary_tabs, strict=False))
    with primary_tab_map["Dispatch"]:
        render_dispatch_tab(conn)
    with primary_tab_map["Calendar"]:
        render_dispatch_calendar(conn)
    with primary_tab_map["Crew"]:
        render_crew_workflow(conn)
    with primary_tab_map["Customer Comms"]:
        render_customer_communications(conn)
    with primary_tab_map["Planner"]:
        render_planner_tab(filtered_df=filtered_df, conn=conn)
    with primary_tab_map["Operations Diary"]:
        render_operations_diary_tab(conn)

    tier_separator()

    # ── Tier 4: Actions (above secondary data) ─────────────────────
    st.markdown(
        '<div class="ck-action-bar">'
        '<span class="ck-action-label">Operations Actions</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    action_cols = st.columns([1, 1, 4])
    with action_cols[0]:
        st.button("Auto-Assign Drivers", type="primary", key="ops_auto_assign")
    with action_cols[1]:
        st.button("Schedule Maintenance", key="ops_schedule_maintenance")

    tier_separator()

    # ── Tier 5: Secondary Data (collapsed) ─────────────────────────
    with st.expander("🔧 Fleet, Inventory & Staff Details", expanded=False):
        sec_tabs = st.tabs([
            "Operations",
            "Fleet",
            "Vehicle Maintenance",
            "Inventory",
            "Staff",
            "Driver Shifts",
            "Payroll / Labor",
        ])
        with sec_tabs[0]:
            render_operations_tab(conn)
        with sec_tabs[1]:
            render_fleet_tab(conn)
        with sec_tabs[2]:
            render_vehicle_maintenance_tab(conn)
        with sec_tabs[3]:
            render_inventory_tab(conn, rerun_app=rerun_app)
        with sec_tabs[4]:
            render_staff_tab(conn, rerun_app=rerun_app)
        with sec_tabs[5]:
            render_driver_shifts_tab(conn, rerun_app=rerun_app)
        with sec_tabs[6]:
            render_payroll_labor_analytics_tab(conn, rerun_app=rerun_app)
