import streamlit as st
import pandas as pd
import sqlite3
from typing import Any

from dashboard.components.dispatch import render_dispatch_tab
from dashboard.components.operations_diary import render_operations_diary_tab
from dashboard.components.planner import render_planner_tab
from dashboard.components.operations import render_operations_tab
from dashboard.components.maintenance import render_fleet_tab, render_vehicle_maintenance_tab
from dashboard.components.inventory import render_inventory_tab
from dashboard.components.staff import render_staff_tab
from dashboard.components.worker_time import render_driver_shifts_tab
from dashboard.components.payroll_labor_analytics import render_payroll_labor_analytics_tab

def render_operations_view(
    conn: sqlite3.Connection,
    filtered_df: pd.DataFrame,
    rerun_app: Any
):
    st.subheader("Operations KPI")
    kpi_cols = st.columns(3)
    kpi_cols[0].metric("Dispatch Fulfillment", "98%", "+1%")
    kpi_cols[1].metric("Active Driver Shifts", "42")
    kpi_cols[2].metric("Fleet Availability", "92%", "-2%")
    
    st.error("Alert: Maintenance due on 2 long-haul vehicles. 1 Shift gap in Metro deliveries.")
    
    st.markdown("---")
    st.subheader("Operations Control (Dispatch & Planner)")
    primary_tabs = st.tabs(["Dispatch", "Planner", "Operations Diary"])
    with primary_tabs[0]:
        render_dispatch_tab(conn)
    with primary_tabs[1]:
        render_planner_tab(filtered_df=filtered_df, conn=conn)
    with primary_tabs[2]:
        render_operations_diary_tab(conn)
        
    st.markdown("---")
    with st.expander("Secondary Data: Fleet, Inventory & Staff"):
        sec_tabs = st.tabs([
            "Operations",
            "Fleet",
            "Vehicle maintenance",
            "Inventory",
            "Staff",
            "Driver shifts",
            "Payroll / Labor analytics"
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
            
    st.markdown("---")
    st.subheader("Actions")
    col1, col2 = st.columns([1, 6])
    with col1:
        st.button("Auto-Assign Drivers", type="primary")
    with col2:
        st.button("Schedule Maintenance")
