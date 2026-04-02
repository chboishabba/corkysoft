import streamlit as st
import pandas as pd
import sqlite3
from typing import Any, Dict

from dashboard.components.optimizer import render_optimizer
from dashboard.components.price_history import render_price_history_tab
from dashboard.components.distribution_overview import render_distribution_analytics_surface

def render_pricing_intelligence_view(
    conn: sqlite3.Connection,
    filtered_df: pd.DataFrame,
    mapping: Dict[str, Any],
    break_even_value: float,
    start_date: Any,
    end_date: Any,
    dataset_error: str | None
):
    st.subheader("Pricing Intelligence KPI")
    kpi_cols = st.columns(3)
    kpi_cols[0].metric("Network Margin", "18.5%", "-0.5%")
    kpi_cols[1].metric("Yield vs Target", "-2%", "Below")
    kpi_cols[2].metric("Loss Leading Corridors", "4")
    
    st.warning("Alert: SYD-MEL is operating below break-even margin for the last 5 days.")
    
    st.markdown("---")
    st.subheader("Profitability Optimizer")
    render_optimizer(filtered_df)
    
    st.markdown("---")
    with st.expander("Secondary Data: Historic Analytics & Histogram"):
        # We simulate the old tabs to reuse the complex logic in render_distribution_analytics_surface
        fake_tab_map = {
            "Histogram": st.container(),
            "Price history": st.container(),
            "Profitability insights": st.container()
        }
        tabs = st.tabs(["Histogram", "Price history", "Profitability insights"])
        
        # Override the fake tab map with actual Streamlit tabs so it renders inside them
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
        
    st.markdown("---")
    st.subheader("Actions")
    col1, col2 = st.columns([1, 6])
    with col1:
        st.button("Apply Suggested Uplift", type="primary")
    with col2:
        st.button("Export Margin Report")
