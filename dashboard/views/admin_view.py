import streamlit as st
import sqlite3
from typing import Any

from dashboard.components.kent import render_kent_admin_tab

def render_admin_view(
    conn: sqlite3.Connection,
    auth_role_key: str | None,
    rerun_app: Any,
    render_dashboard_user_admin: Any
):
    st.subheader("Admin KPI")
    kpi_cols = st.columns(3)
    kpi_cols[0].metric("System Health", "99.9%", "+0.01%")
    kpi_cols[1].metric("Active Users", "14")
    kpi_cols[2].metric("Pending Roles", "0")
    
    st.info("Alert: Kent AMS integration running smoothly.")
    
    st.markdown("---")
    st.subheader("User & Integration Management")
    
    render_kent_admin_tab(
        conn,
        current_role_key=auth_role_key,
        rerun_app=rerun_app,
        render_dashboard_user_admin=render_dashboard_user_admin
    )
    
    st.markdown("---")
    with st.expander("Secondary Data: System Logs"):
        st.write("System log snapshot:")
        st.code("[2026-04-02 10:00:01] System OK.\n[2026-04-02 10:05:22] Role change synchronized.")
        
    st.markdown("---")
    st.subheader("Actions")
    col1, col2 = st.columns([1, 6])
    with col1:
        st.button("Force Sync Users", type="primary")
    with col2:
        st.button("Export System Logs")
