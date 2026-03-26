from __future__ import annotations

import io
import sqlite3
from datetime import date
from typing import Callable

import pandas as pd
import plotly.express as px
import streamlit as st

from analytics.db import (
    ABSENCE_RECORD_STATUSES,
    ABSENCE_RECORD_TYPES,
    create_worker_absence_record,
)
from analytics.labor_analytics import (
    OVERTIME_DAILY_HOURS_DEFAULT,
    build_payroll_labor_analytics,
)


def render_payroll_labor_analytics_tab(
    conn: sqlite3.Connection,
    *,
    rerun_app: Callable[[], None],
) -> None:
    st.subheader("Payroll preparation and labor analytics")
    st.caption(
        "Aggregate-first labor forecasting and workforce insight built from planned assignments, imported shifts, and reviewed worker-time events. Corkysoft prepares payroll truth here; it does not execute payroll."
    )

    baseline = build_payroll_labor_analytics(conn)
    known_dates: list[date] = []
    for row in baseline.get("hoursCostDistributionRows", []):
        parsed = pd.to_datetime(row.get("date"), errors="coerce")
        if not pd.isna(parsed):
            known_dates.append(parsed.date())
    for row in baseline.get("payForecastRows", []):
        parsed = pd.to_datetime(row.get("date"), errors="coerce")
        if not pd.isna(parsed):
            known_dates.append(parsed.date())

    if known_dates:
        min_date = min(known_dates)
        max_date = max(known_dates)
    else:
        today = date.today()
        min_date = today
        max_date = today

    date_range = st.date_input(
        "Payroll / labor date range",
        value=(min_date, max_date),
        key="payroll_labor_analytics_date_range",
    )
    overtime_threshold = st.number_input(
        "Daily overtime threshold (hours)",
        min_value=0.0,
        max_value=24.0,
        value=float(OVERTIME_DAILY_HOURS_DEFAULT),
        step=0.5,
        key="payroll_labor_analytics_overtime_threshold",
    )

    start_date = None
    end_date = None
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date = date_range[0].isoformat() if date_range[0] else None
        end_date = date_range[1].isoformat() if date_range[1] else None

    analytics_payload = build_payroll_labor_analytics(
        conn,
        start_date=start_date,
        end_date=end_date,
        overtime_daily_hours=float(overtime_threshold),
    )

    summary = analytics_payload["summary"]
    pay_forecast_rows = analytics_payload["payForecastRows"]
    export_ready_rows = analytics_payload["exportReadyLaborSummaries"]
    distribution_rows = analytics_payload["hoursCostDistributionRows"]
    overtime_rows = analytics_payload["overtimeRows"]
    plan_vs_actual = analytics_payload["planVsActual"]
    confidence = analytics_payload["confidence"]
    absence_summary = analytics_payload["absenceSummary"]
    absence_rows = analytics_payload["absenceRows"]
    labor_cost_drivers = analytics_payload["laborCostDrivers"]

    summary_cols = st.columns(4)
    summary_cols[0].metric("Planned exposure", f"${summary['plannedExposure']:,.2f}")
    summary_cols[1].metric("Imported labor cost", f"${summary['importedCost']:,.2f}")
    summary_cols[2].metric("Reviewed actual cost", f"${summary['reviewedActualCost']:,.2f}")
    summary_cols[3].metric(
        "Payroll-prep confidence",
        f"{int(summary['confidenceScore'])} ({summary['confidenceLabel']})",
    )

    st.markdown("#### Pay Forecast")
    pay_forecast_df = pd.DataFrame(pay_forecast_rows)
    if pay_forecast_df.empty:
        st.caption("No planned or imported labor data is available for the selected range.")
    else:
        pay_forecast_df = pay_forecast_df.sort_values(
            by=["importedCost", "plannedExposure", "reviewedActualCost"],
            ascending=[False, False, False],
            kind="stable",
        )
        st.dataframe(
            pay_forecast_df.rename(
                columns={
                    "workerName": "Worker",
                    "plannedHours": "Planned hours",
                    "plannedExposure": "Planned exposure",
                    "importedHours": "Imported hours",
                    "importedCost": "Imported cost",
                    "reviewedActualCost": "Reviewed actual cost",
                    "acceptedEventCount": "Accepted events",
                    "hourlyRateBasis": "Hourly-rate basis",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        top_pay_df = pay_forecast_df.head(10).copy()
        pay_chart = px.bar(
            top_pay_df,
            x="workerName",
            y=["plannedExposure", "importedCost", "reviewedActualCost"],
            barmode="group",
            title="Top workers by planned/imported labor cost",
            labels={"workerName": "Worker", "value": "Amount", "variable": "Series"},
        )
        st.plotly_chart(pay_chart, width="stretch")

    st.markdown("#### Export-ready Labor Summary")
    export_df = pd.DataFrame(export_ready_rows)
    if export_df.empty:
        st.caption("No export-ready labor summary rows are available for the selected range.")
    else:
        st.dataframe(
            export_df.rename(
                columns={
                    "workerName": "Worker",
                    "dateRangeStart": "Range start",
                    "dateRangeEnd": "Range end",
                    "plannedExposure": "Planned exposure",
                    "importedCost": "Imported cost",
                    "reviewedActualCost": "Reviewed actual cost",
                    "importedHours": "Imported hours",
                    "overtimeHours": "Overtime hours",
                    "absenceDays": "Absence days",
                    "absenceHours": "Absence hours",
                    "acceptedEventCount": "Accepted events",
                    "pendingReviewCount": "Pending reviews",
                    "hourlyRateBasis": "Hourly-rate basis",
                    "exportReady": "Export ready",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        export_buffer = io.StringIO()
        export_df.to_csv(export_buffer, index=False)
        st.download_button(
            "Download payroll-prep summary CSV",
            export_buffer.getvalue(),
            file_name="payroll_labor_export_summary.csv",
            mime="text/csv",
            key="payroll_labor_export_summary_csv",
        )

    st.markdown("#### Hours / Cost Distribution")
    distribution_df = pd.DataFrame(distribution_rows)
    if distribution_df.empty:
        st.caption("No imported labor-cost rows are available for the selected range.")
    else:
        dist_cols = st.columns(2)
        hours_fig = px.histogram(
            distribution_df,
            x="hours",
            nbins=20,
            title="Imported shift hours distribution",
            labels={"hours": "Hours"},
        )
        cost_fig = px.histogram(
            distribution_df,
            x="costTotal",
            nbins=20,
            title="Imported labor cost distribution",
            labels={"costTotal": "Cost"},
        )
        dist_cols[0].plotly_chart(hours_fig, width="stretch")
        dist_cols[1].plotly_chart(cost_fig, width="stretch")

    st.markdown("#### Overtime Distribution")
    overtime_df = pd.DataFrame(overtime_rows)
    if overtime_df.empty:
        st.caption("No imported shift rows are available to evaluate overtime in the selected range.")
    else:
        overtime_worker_df = (
            overtime_df.groupby("workerName", dropna=False)[["overtimeHours", "totalHours", "totalCost"]]
            .sum()
            .reset_index()
            .sort_values(["overtimeHours", "totalHours"], ascending=[False, False], kind="stable")
        )
        st.caption(
            f"V1 overtime uses a simple daily-hours-above-threshold heuristic ({float(overtime_threshold):.1f} h/day), not award interpretation."
        )
        st.dataframe(
            overtime_worker_df.rename(
                columns={
                    "workerName": "Worker",
                    "overtimeHours": "Overtime hours",
                    "totalHours": "Total hours",
                    "totalCost": "Total cost",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        overtime_chart = px.bar(
            overtime_worker_df.head(10),
            x="workerName",
            y="overtimeHours",
            title="Overtime hours by worker",
            labels={"workerName": "Worker", "overtimeHours": "Overtime hours"},
        )
        st.plotly_chart(overtime_chart, width="stretch")

    st.markdown("#### Plan vs Actual")
    variance_cols = st.columns(5)
    variance_cols[0].metric("Planned only", int(plan_vs_actual["plannedOnlyCount"]))
    variance_cols[1].metric("Imported only", int(plan_vs_actual["importedOnlyCount"]))
    variance_cols[2].metric("Matched", int(plan_vs_actual["matchedCount"]))
    variance_cols[3].metric("Accepted matched shifts", int(plan_vs_actual["acceptedMatchedShiftCount"]))
    variance_cols[4].metric("Accepted unmatched", int(plan_vs_actual["acceptedUnmatchedCount"]))

    st.markdown("#### Confidence / Anomalies")
    confidence_cols = st.columns(4)
    confidence_cols[0].metric("Pending review", int(confidence["pendingReviewCount"]))
    confidence_cols[1].metric("Duplicate events", int(confidence["duplicateEventCount"]))
    confidence_cols[2].metric("Missing prior clock-on", int(confidence["missingPriorClockOnCount"]))
    confidence_cols[3].metric("Accepted unmatched events", int(confidence["acceptedUnmatchedCount"]))
    st.caption(
        "Confidence reflects worker-time review/anomaly health. Absence is now based on explicit recorded leave/absence rows rather than inferred missing shifts."
    )

    st.markdown("#### Absence / Leave")
    absence_cols = st.columns(4)
    absence_cols[0].metric("Recorded rows", int(absence_summary["recordCount"]))
    absence_cols[1].metric("Confirmed", int(absence_summary["confirmedCount"]))
    absence_cols[2].metric("Planned", int(absence_summary["plannedCount"]))
    absence_cols[3].metric("Sick days", f"{absence_summary['sickDays']:.1f}")
    secondary_absence_cols = st.columns(4)
    secondary_absence_cols[0].metric("Annual leave days", f"{absence_summary['annualLeaveDays']:.1f}")
    secondary_absence_cols[1].metric("Personal leave days", f"{absence_summary['personalLeaveDays']:.1f}")
    secondary_absence_cols[2].metric("Unpaid leave days", f"{absence_summary['unpaidLeaveDays']:.1f}")
    secondary_absence_cols[3].metric("Carer's leave days", f"{absence_summary['carersLeaveDays']:.1f}")

    worker_options = conn.execute("SELECT id, name FROM workers ORDER BY name").fetchall()
    if worker_options:
        with st.expander("Record absence / leave", expanded=False):
            with st.form("payroll_absence_record_form"):
                worker_label_map = {
                    f"{row['name']} ({int(row['id'])})": int(row["id"])
                    for row in worker_options
                }
                selected_worker_label = st.selectbox(
                    "Worker",
                    options=list(worker_label_map.keys()),
                    key="payroll_absence_worker",
                )
                absence_form_cols = st.columns(3)
                absence_start_date = absence_form_cols[0].date_input(
                    "Start date",
                    value=min_date,
                    key="payroll_absence_start_date",
                )
                absence_end_date = absence_form_cols[1].date_input(
                    "End date",
                    value=min_date,
                    key="payroll_absence_end_date",
                )
                absence_type = absence_form_cols[2].selectbox(
                    "Type",
                    options=list(ABSENCE_RECORD_TYPES),
                    key="payroll_absence_type",
                )
                absence_meta_cols = st.columns(4)
                absence_status = absence_meta_cols[0].selectbox(
                    "Status",
                    options=list(ABSENCE_RECORD_STATUSES),
                    key="payroll_absence_status",
                )
                absence_hours = float(
                    absence_meta_cols[1].number_input(
                        "Hours per day",
                        min_value=0.0,
                        max_value=24.0,
                        value=8.0,
                        step=0.5,
                        key="payroll_absence_hours_per_day",
                    )
                )
                absence_source = absence_meta_cols[2].text_input(
                    "Source",
                    value="manual_manager",
                    key="payroll_absence_source",
                )
                absence_recorded_by = absence_meta_cols[3].text_input(
                    "Recorded by",
                    value="manager",
                    key="payroll_absence_recorded_by",
                )
                absence_note = st.text_area("Note", key="payroll_absence_note")
                if st.form_submit_button("Record absence / leave"):
                    try:
                        create_worker_absence_record(
                            conn,
                            worker_id=worker_label_map[selected_worker_label],
                            start_date=absence_start_date.isoformat(),
                            end_date=absence_end_date.isoformat(),
                            absence_type=absence_type,
                            status=absence_status,
                            hours_per_day=absence_hours,
                            note=absence_note.strip() or None,
                            source=absence_source.strip() or None,
                            recorded_by=absence_recorded_by.strip() or None,
                        )
                    except Exception as exc:
                        st.error(f"Failed to record absence / leave: {exc}")
                    else:
                        st.success("Absence / leave record saved.")
                        rerun_app()

    absence_df = pd.DataFrame(absence_rows)
    if absence_df.empty:
        st.caption("No absence / leave rows are recorded for the selected range.")
    else:
        st.dataframe(
            absence_df.rename(
                columns={
                    "workerName": "Worker",
                    "startDate": "Start date",
                    "endDate": "End date",
                    "absenceType": "Type",
                    "status": "Status",
                    "hoursPerDay": "Hours / day",
                    "note": "Note",
                    "source": "Source",
                    "recordedBy": "Recorded by",
                }
            )[
                [
                    "Worker",
                    "Start date",
                    "End date",
                    "Type",
                    "Status",
                    "Hours / day",
                    "Source",
                    "Recorded by",
                    "Note",
                ]
            ],
            width="stretch",
            hide_index=True,
        )

    st.markdown("#### Labor Cost Drivers")
    driver_dimension = st.radio(
        "Cost-driver grouping",
        options=["worker", "client", "corridor", "truck", "job"],
        horizontal=True,
        key="payroll_labor_cost_driver_dimension",
    )
    driver_df = pd.DataFrame(labor_cost_drivers.get(driver_dimension, []))
    if driver_df.empty:
        st.caption("No labor cost-driver rows are available for the selected range.")
    else:
        st.dataframe(
            driver_df.rename(
                columns={
                    "dimensionValue": "Value",
                    "totalHours": "Total hours",
                    "totalCost": "Total cost",
                    "shiftCount": "Shift count",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        driver_chart = px.bar(
            driver_df.head(10),
            x="dimensionValue",
            y="totalCost",
            title=f"Top {driver_dimension} labor cost drivers",
            labels={"dimensionValue": driver_dimension.title(), "totalCost": "Total cost"},
        )
        st.plotly_chart(driver_chart, width="stretch")
