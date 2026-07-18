import streamlit as st
import pandas as pd
import sqlite3
from typing import Any, Dict

from dashboard.components.kpi_strip import render_kpi_strip
from dashboard.components.alert_banner import render_alert_banner
from dashboard.components.data_provenance import render_signal_contract_notice
from dashboard.components.quote_builder import render_quote_builder
from dashboard.components.accepted_work import render_quote_acceptance_panel
from dashboard.components.calls import render_calls_tab
from dashboard.components.kent import render_kent_tenders_tab
from dashboard.shell_signals import build_quote_shell_signal_bundle
from dashboard.theme import tier_separator, hero_section


def render_quote_view(
    conn: sqlite3.Connection,
    filtered_df: pd.DataFrame,
    mapping: Dict[str, Any],
    state: Any,
    rerun_app: Any,
):
    # ── Tier 1: KPI Strip ──────────────────────────────────────────
    signal_bundle = build_quote_shell_signal_bundle(conn)
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
    hero_section("Quote Builder", "Build and price a new quote or select from historical routes.")
    render_quote_builder(
        filtered_df=filtered_df,
        mapping=mapping,
        conn=conn,
        state=state,
    )

    tier_separator()

    # ── Tier 4: Quote-to-job action ────────────────────────────────
    st.markdown(
        '<div class="ck-action-bar">'
        '<span class="ck-action-label">Accepted Work</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    render_quote_acceptance_panel(conn)

    tier_separator()

    # ── Tier 5: Secondary Data (collapsed) ─────────────────────────
    with st.expander("📋 Calls & Tenders", expanded=False):
        tabs = st.tabs(["Calls", "Kent Tenders"])
        with tabs[0]:
            render_calls_tab(conn)
        with tabs[1]:
            render_kent_tenders_tab(conn, rerun_app=rerun_app)
