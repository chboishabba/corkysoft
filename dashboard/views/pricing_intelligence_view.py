import streamlit as st
import pandas as pd
import sqlite3
from typing import Any, Dict

from dashboard.components.kpi_strip import render_kpi_strip
from dashboard.components.alert_banner import render_alert_banner
from dashboard.components.data_provenance import render_signal_contract_notice
from dashboard.components.optimizer import render_optimizer
from dashboard.components.price_history import render_price_history_tab
from dashboard.components.distribution_overview import render_distribution_analytics_surface
from dashboard.shell_signals import build_pricing_shell_signal_bundle
from dashboard.theme import tier_separator, hero_section

def render_pricing_intelligence_view(
    conn: sqlite3.Connection,
    filtered_df: pd.DataFrame,
    mapping: Dict[str, Any],
    break_even_value: float,
    start_date: Any,
    end_date: Any,
    dataset_error: str | None
):
    # ── Tier 1: KPI Strip ──────────────────────────────────────────
    signal_bundle = build_pricing_shell_signal_bundle(
        filtered_df=filtered_df,
        break_even_value=break_even_value,
        dataset_error=dataset_error,
    )
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
    hero_section("Profitability Optimizer", "Generate corridor-level price uplift suggestions.")
    render_optimizer(filtered_df)

    tier_separator()

    # ── Tier 4: Actions (above secondary data) ─────────────────────
    st.markdown(
        '<div class="ck-action-bar">'
        '<span class="ck-action-label">Pricing Actions</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    action_cols = st.columns([1, 1, 4])
    with action_cols[0]:
        st.button("Apply Suggested Uplift", type="primary", key="pricing_apply_uplift")
    with action_cols[1]:
        st.button("Export Margin Report", key="pricing_export_report")

    tier_separator()

    # ── Tier 5: Secondary Data (collapsed) ─────────────────────────
    with st.expander("📊 Historic Analytics & Distribution", expanded=False):
        fake_tab_map = {
            "Histogram": st.container(),
            "Price history": st.container(),
            "Profitability insights": st.container()
        }
        tabs = st.tabs(["Histogram", "Price history", "Profitability insights"])

        fake_tab_map["Histogram"] = tabs[0]
        fake_tab_map["Price history"] = tabs[1]
        fake_tab_map["Profitability insights"] = tabs[2]

        render_distribution_analytics_surface(
            tab_map=fake_tab_map,
            filtered_df=filtered_df,
            filtered_mapping=mapping,
            break_even_value=break_even_value,
            dataset_error=dataset_error,
            conn=conn,
            start_date=start_date,
            end_date=end_date,
            show_overview=False
        )
