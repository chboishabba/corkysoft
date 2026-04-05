import streamlit as st
import pandas as pd
import sqlite3
from typing import Any, Dict

from dashboard.components.kpi_strip import render_kpi_strip
from dashboard.components.alert_banner import render_alert_banner
from dashboard.components.data_provenance import render_signal_contract_notice
from dashboard.components.distribution_overview import render_distribution_analytics_surface
from dashboard.components.route_maps import render_route_maps_tab
from dashboard.shell_signals import build_network_shell_signal_bundle
from dashboard.theme import tier_separator, hero_section


def render_network_view(
    conn: sqlite3.Connection,
    filtered_df: pd.DataFrame,
    mapping: Dict[str, Any],
    break_even_value: float,
    dataset_key: str,
    dataset_error: str | None
):
    # ── Tier 1: KPI Strip ──────────────────────────────────────────
    signal_bundle = build_network_shell_signal_bundle(conn)
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
    hero_section("Live Network Overview", "Real-time network map with telemetry and heatmap overlays.")

    surface_mode = st.radio(
        "Network map mode",
        ("Live network", "Routes/points", "Heatmap", "Isochrones"),
        horizontal=True,
        key="network_primary_map_mode",
        help=(
            "Render exactly one top-of-page map surface: the live network overlay, "
            "corridor routes/points, a corridor heatmap, or provider-backed isochrones."
        ),
    )

    if surface_mode == "Live network":
        fake_tab_map = {"Live network overview": st.container()}
        render_distribution_analytics_surface(
            tab_map=fake_tab_map,
            filtered_df=filtered_df,
            filtered_mapping=mapping,
            break_even_value=break_even_value,
            dataset_error=dataset_error,
            conn=conn,
            start_date=None,
            end_date=None,
            show_overview=False
        )
    else:
        render_route_maps_tab(
            filtered_df=filtered_df,
            mapping=mapping,
            conn=conn,
            dataset_key=dataset_key,
            metro_distance_km=100.0,
            show_title=False,
            forced_mode=surface_mode,
            network_host=True,
        )

    tier_separator()

    # ── Tier 4: Actions (above secondary data) ─────────────────────
    st.markdown(
        '<div class="ck-action-bar">'
        '<span class="ck-action-label">Network Actions</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    action_cols = st.columns([1, 1, 4])
    with action_cols[0]:
        st.button("Refresh Telemetry", type="primary", key="network_refresh_telemetry")
    with action_cols[1]:
        st.button("Override Route", key="network_override_route")
