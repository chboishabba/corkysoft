import streamlit as st
import sqlite3
from typing import Any

from dashboard.components.kpi_strip import render_kpi_strip
from dashboard.components.alert_banner import render_alert_banner
from dashboard.components.data_provenance import render_signal_contract_notice
from dashboard.components.kent import render_kent_admin_tab
from dashboard.shell_signals import build_admin_shell_signal_bundle
from dashboard.theme import tier_separator, hero_section

def render_admin_view(
    conn: sqlite3.Connection,
    auth_role_key: str | None,
    rerun_app: Any,
    render_dashboard_user_admin: Any
):
    # ── Tier 1: KPI Strip ──────────────────────────────────────────
    signal_bundle = build_admin_shell_signal_bundle(conn)
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
    hero_section("User & Integration Management", "Manage dashboard users, policies, and integration settings.")

    render_kent_admin_tab(
        conn,
        current_role_key=auth_role_key,
        rerun_app=rerun_app,
        render_dashboard_user_admin=render_dashboard_user_admin
    )

    tier_separator()

    # ── Tier 4: Actions (above secondary data) ─────────────────────
    st.markdown(
        '<div class="ck-action-bar">'
        '<span class="ck-action-label">Admin Actions</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    action_cols = st.columns([1, 1, 4])
    with action_cols[0]:
        st.button("Force Sync Users", type="primary", key="admin_force_sync")
    with action_cols[1]:
        st.button("Export System Logs", key="admin_export_logs")

    tier_separator()

    # ── Tier 5: Secondary Data (collapsed) ─────────────────────────
    with st.expander("📄 System Logs", expanded=False):
        st.write("System log snapshot:")
        st.code("[2026-04-02 10:00:01] System OK.\n[2026-04-02 10:05:22] Role change synchronized.")
