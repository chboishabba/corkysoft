import streamlit as st
import pandas as pd
import sqlite3
from typing import Any, Dict

from dashboard.components.quote_builder import render_quote_builder
from dashboard.components.calls import render_calls_tab
from dashboard.components.kent import render_kent_tenders_tab

def render_quote_view(
    conn: sqlite3.Connection,
    filtered_df: pd.DataFrame,
    mapping: Dict[str, Any],
    state: Any,
    rerun_app: Any
):
    st.subheader("Quote KPI")
    kpi_cols = st.columns(3)
    kpi_cols[0].metric("Quote Win Rate", "34%", "+2%")
    kpi_cols[1].metric("Avg Margin Built", "21%", "-1%")
    kpi_cols[2].metric("Active Pending", "12")
    
    st.warning("Alert: 3 quotes are awaiting urgent review. Client Acme Inc credit check delayed.")
    
    st.markdown("---")
    st.subheader("Quote Builder")
    render_quote_builder(
        filtered_df=filtered_df,
        mapping=mapping,
        conn=conn,
        state=state,
    )
    
    st.markdown("---")
    with st.expander("Secondary Data: Calls & Tenders"):
        tabs = st.tabs(["Calls", "Kent Tenders"])
        with tabs[0]:
            render_calls_tab(conn)
        with tabs[1]:
            render_kent_tenders_tab(conn, rerun_app=rerun_app)
    
    st.markdown("---")
    st.subheader("Actions")
    col1, col2 = st.columns([1, 6])
    with col1:
        st.button("Save Draft Quote", type="primary")
    with col2:
        st.button("Request Approval")
