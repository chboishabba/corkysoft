import streamlit as st
import pandas as pd
import sqlite3
from typing import Any, Dict

from dashboard.components.distribution_overview import render_distribution_analytics_surface
from dashboard.components.route_maps import render_route_maps_tab

def render_network_view(
    conn: sqlite3.Connection,
    filtered_df: pd.DataFrame,
    mapping: Dict[str, Any],
    break_even_value: float,
    dataset_key: str,
    dataset_error: str | None
):
    st.subheader("Network KPI")
    kpi_cols = st.columns(3)
    kpi_cols[0].metric("Active Nodes", "24", "+1")
    kpi_cols[1].metric("Live Trucks", "18", "Optimal")
    kpi_cols[2].metric("Congestion Level", "Low")
    
    st.info("Alert: No significant network disruptions. SYD terminal operating at 85% capacity.")
    
    st.markdown("---")
    st.subheader("Live Network Overview")
    
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
    
    st.markdown("---")
    with st.expander("Secondary Data: Historic Route Maps"):
        render_route_maps_tab(
            filtered_df=filtered_df,
            mapping=mapping,
            conn=conn,
            dataset_key=dataset_key,
            metro_distance_km=100.0
        )
    
    st.markdown("---")
    st.subheader("Actions")
    col1, col2 = st.columns([1, 6])
    with col1:
        st.button("Refresh Telemetry", type="primary")
    with col2:
        st.button("Override Route")
