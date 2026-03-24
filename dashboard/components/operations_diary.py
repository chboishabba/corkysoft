from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from typing import Any

import pandas as pd
import streamlit as st

from analytics.operations_diary import (
    CUSTOMER_INVOICE_STATUSES,
    DIARY_TASK_SCOPES,
    DIARY_TASK_STATUSES,
    SUBCONTRACTOR_BILL_STATUSES,
    build_job_usage_details,
    build_operations_diary,
    delete_operations_diary_task,
    upsert_customer_invoice_review,
    upsert_operations_diary_task,
    upsert_subcontractor_bill_review,
)


def _set_query_params(**params: str) -> None:
    query_params = getattr(st, "query_params", None)
    if query_params is not None:
        query_params.from_dict(params)
        return
    st.experimental_set_query_params(**params)


def _get_query_param(key: str, default: str) -> str:
    query_params = getattr(st, "query_params", None)
    if query_params is not None:
        values = query_params.get_all(key)
        return values[0] if values else default
    return st.experimental_get_query_params().get(key, [default])[0]


def render_operations_diary_tab(conn: sqlite3.Connection) -> None:
    st.subheader("Operations diary")
    st.caption(
        "Manager-facing day/week cockpit linking planning, assignments, usage review, diary tasks, customer invoicing, and subcontractor-bill reconciliation."
    )

    requested_mode = _get_query_param("diary_view", "day")
    requested_date = _get_query_param("diary_date", date.today().isoformat())
    requested_job = _get_query_param("diary_job", "")
    focus_job_id = int(requested_job) if requested_job.isdigit() else None

    controls = st.columns(3)
    view_mode = controls[0].radio(
        "Diary view",
        options=["day", "week"],
        horizontal=True,
        index=0 if requested_mode != "week" else 1,
        format_func=lambda value: "Day" if value == "day" else "Week",
        key="operations_diary_view_mode",
    )
    anchor_date = controls[1].date_input(
        "Anchor date",
        value=_safe_date(requested_date),
        key="operations_diary_anchor_date",
    )
    if controls[2].button("Refresh diary", key="operations_diary_refresh"):
        _set_query_params(
            view="Operations diary",
            diary_view=view_mode,
            diary_date=anchor_date.isoformat(),
            diary_job=str(focus_job_id or ""),
        )

    diary = build_operations_diary(
        conn,
        anchor_date=anchor_date.isoformat(),
        view_mode=view_mode,
        focus_job_id=focus_job_id,
    )

    summary = diary["summary"]
    summary_cols = st.columns(6)
    summary_cols[0].metric("Jobs", int(summary["jobCount"]))
    summary_cols[1].metric("Tasks", int(summary["taskCount"]))
    summary_cols[2].metric("Open tasks", int(summary["openTaskCount"]))
    summary_cols[3].metric("Invoice issues", int(summary["invoiceExceptionCount"]))
    summary_cols[4].metric("Bill issues", int(summary["billExceptionCount"]))
    summary_cols[5].metric("Planned labor", int(summary["plannedLaborCount"]))

    exposure = diary["reconciliationExposure"]
    exposure_cols = st.columns(4)
    exposure_cols[0].metric("Supplier unresolved", float(summary["supplierUnresolvedTotal"]))
    exposure_cols[1].metric("Customer open", float(summary["customerOpenTotal"]))
    exposure_cols[2].metric(
        "Oldest supplier age",
        int(exposure["oldestSupplierAgeDays"]) if exposure["oldestSupplierAgeDays"] is not None else "n/a",
    )
    exposure_cols[3].metric(
        "Longest supplier latency",
        int(exposure["longestSupplierLatencyDays"]) if exposure["longestSupplierLatencyDays"] is not None else "n/a",
    )

    if exposure["activeSupplierRows"]:
        with st.expander("Top unresolved supplier exposure", expanded=False):
            st.caption(
                "Active aging uses the bill-received date. Latency compares bill receipt against job execution so delayed liabilities stay visible."
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Job": row["jobId"],
                            "Client": row["jobClient"],
                            "Supplier": row.get("supplierName"),
                            "Reference": row.get("reference"),
                            "Amount": row["amount"],
                            "Status": row["status"],
                            "Job executed": row["jobExecutionDate"],
                            "Bill received": row["receivedDate"],
                            "Latency days": row["latencyDays"],
                            "Unresolved age": row["unresolvedAgeDays"],
                        }
                        for row in exposure["activeSupplierRows"][:10]
                    ]
                ),
                width="stretch",
                hide_index=True,
            )

    if view_mode == "week":
        st.caption(f"Week window: {diary['startDate']} to {diary['endDate']}")

    job_rows = diary["jobs"]
    if not job_rows:
        st.info("No diary jobs or tasks are present for the selected period yet.")
        _render_global_task_form(conn, anchor_date=anchor_date.isoformat())
        return

    st.markdown("#### Jobs in scope")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Focus": "Current" if row["isFocusJob"] else "",
                    "Job": row["jobId"],
                    "Client": row["jobClient"] or "Unknown client",
                    "Origin": row["jobOrigin"],
                    "Destination": row["jobDestination"],
                    "Status": row["jobStatus"],
                    "Segments": row["segmentCount"],
                    "Tasks": row["taskCount"],
                    "Trucks": ", ".join(row["truckIds"]),
                    "Workers": ", ".join(row["workerNames"]),
                    "Suppliers": ", ".join(row["supplierNames"]),
                    "Invoice": row["invoiceStatus"],
                    "Subcontractor bill": row["billStatus"],
                    "Imported shifts": row["importedShiftCount"],
                    "Imported cost": row["importedShiftCost"],
                    "Planned start": row["plannedStart"],
                    "Planned end": row["plannedEnd"],
                }
                for row in job_rows
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    options = {
        _job_label(row): int(row["jobId"])
        for row in job_rows
    }
    selected_label = st.selectbox(
        "Inspect diary job",
        options=list(options.keys()),
        index=_default_job_index(job_rows, focus_job_id),
        key="operations_diary_selected_job",
    )
    selected_job_id = int(options[selected_label])
    _set_query_params(
        view="Operations diary",
        diary_view=view_mode,
        diary_date=anchor_date.isoformat(),
        diary_job=str(selected_job_id),
    )

    details = build_job_usage_details(conn, job_id=selected_job_id)
    job = details["job"]
    header_cols = st.columns(3)
    if header_cols[0].button("Open in Planner", key=f"operations_diary_to_planner_{selected_job_id}"):
        _set_query_params(
            view="Planner",
            diary_view=view_mode,
            diary_date=anchor_date.isoformat(),
            diary_job=str(selected_job_id),
        )
    if header_cols[1].button("Open in Dispatch", key=f"operations_diary_to_dispatch_{selected_job_id}"):
        _set_query_params(
            view="Dispatch",
            diary_view=view_mode,
            diary_date=anchor_date.isoformat(),
            diary_job=str(selected_job_id),
        )
    header_cols[2].caption(
        f"Review window: {diary['startDate']} to {diary['endDate']}"
    )

    st.markdown("#### Job usage summary")
    usage_cols = st.columns(6)
    usage_cols[0].metric("Segments", int(job["segmentCount"]))
    usage_cols[1].metric("Warnings", int(job["warningCount"]))
    usage_cols[2].metric("Blocks", int(job["blockingCount"]))
    usage_cols[3].metric("Required qty", float(job["requiredQuantity"]))
    usage_cols[4].metric("Allocated qty", float(job["allocatedQuantity"]))
    usage_cols[5].metric("Shortage qty", float(job["shortageQuantity"]))

    st.markdown("#### Segment detail")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Segment": row["segmentSequence"],
                    "From": row["fromLocation"],
                    "To": row["toLocation"],
                    "Planned start": row["plannedStart"],
                    "Planned end": row["plannedEnd"],
                    "Status": row["assignmentStatus"],
                    "Trucks": ", ".join(row["truckIds"]),
                    "Workers": ", ".join(row["workerNames"]),
                    "Suppliers": ", ".join(row["supplierNames"]),
                    "Warnings": row["warningCount"],
                    "Blocks": row["blockingCount"],
                }
                for row in job["segments"]
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    split_cols = st.columns(2)
    with split_cols[0]:
        st.markdown("#### Vehicle usage")
        vehicle_df = pd.DataFrame(details["vehicleUsage"])
        if vehicle_df.empty:
            st.caption("No vehicle usage rows are linked to this job yet.")
        else:
            st.dataframe(vehicle_df, width="stretch", hide_index=True)
    with split_cols[1]:
        st.markdown("#### Staff usage")
        staff_df = pd.DataFrame(details["staffUsage"])
        if staff_df.empty:
            st.caption("No staff usage rows are linked to this job yet.")
        else:
            if "plannedTruckIds" in staff_df.columns:
                staff_df["plannedTruckIds"] = staff_df["plannedTruckIds"].apply(lambda values: ", ".join(values))
            st.dataframe(staff_df, width="stretch", hide_index=True)

    fin_cols = st.columns(2)
    with fin_cols[0]:
        st.markdown("#### Customer invoice review")
        _render_customer_invoice_form(conn, job_id=selected_job_id, current=details["invoiceReview"])
    with fin_cols[1]:
        st.markdown("#### Subcontractor bill review")
        _render_subcontractor_bill_form(conn, job_id=selected_job_id, current_rows=details["billReviews"])

    exposure_rows = details["reconciliationExposure"]["activeSupplierRows"]
    if exposure_rows:
        st.markdown("#### Reconciliation exposure markers")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Supplier": row.get("supplierName"),
                        "Reference": row.get("reference"),
                        "Amount": row["amount"],
                        "Status": row["status"],
                        "Job executed": row["jobExecutionDate"],
                        "Bill received": row["receivedDate"],
                        "Latency days": row["latencyDays"],
                        "Unresolved age": row["unresolvedAgeDays"],
                    }
                    for row in exposure_rows
                ]
            ),
            width="stretch",
            hide_index=True,
        )

    st.markdown("#### Diary tasks")
    _render_job_task_editor(
        conn,
        job_id=selected_job_id,
        anchor_date=anchor_date.isoformat(),
        current_rows=details["tasks"],
    )

    if diary["tasks"]:
        with st.expander("All diary tasks in current window", expanded=False):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Date": row["task_date"],
                            "Job": row["job_id"],
                            "Scope": row["task_scope"],
                            "Type": row["task_type"],
                            "Title": row["title"],
                            "Status": row["status"],
                            "Worker": row.get("worker_name"),
                            "Truck": row.get("assigned_truck_id"),
                        }
                        for row in diary["tasks"]
                    ]
                ),
                width="stretch",
                hide_index=True,
            )


def _safe_date(value: str) -> date:
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return date.today()


def _job_label(row: dict[str, Any]) -> str:
    return f"#{row['jobId']} · {row.get('jobClient') or 'Unknown client'} · {row.get('jobStatus')}"


def _default_job_index(rows: list[dict[str, Any]], focus_job_id: int | None) -> int:
    if focus_job_id is None:
        return 0
    for index, row in enumerate(rows):
        if int(row["jobId"]) == int(focus_job_id):
            return index
    return 0


def _render_customer_invoice_form(
    conn: sqlite3.Connection,
    *,
    job_id: int,
    current: dict[str, Any] | None,
) -> None:
    with st.form(f"operations_diary_invoice_form_{job_id}"):
        status = st.selectbox(
            "Invoice status",
            options=list(CUSTOMER_INVOICE_STATUSES),
            index=list(CUSTOMER_INVOICE_STATUSES).index(
                str((current or {}).get("invoice_status") or "not_ready")
            ),
        )
        reference = st.text_input("Invoice reference", value=str((current or {}).get("invoice_reference") or ""))
        invoice_date = st.text_input("Invoice date", value=str((current or {}).get("invoice_date") or ""))
        amount = st.number_input(
            "Invoice amount",
            min_value=0.0,
            value=float((current or {}).get("invoice_amount") or 0.0),
            step=100.0,
        )
        reviewed_by = st.text_input("Reviewed by", value=str((current or {}).get("reviewed_by") or "manager"))
        note = st.text_area("Invoice note", value=str((current or {}).get("note") or ""))
        submitted = st.form_submit_button("Save invoice review")
        if submitted:
            upsert_customer_invoice_review(
                conn,
                job_id=job_id,
                invoice_status=status,
                invoice_reference=reference or None,
                invoice_date=invoice_date or None,
                invoice_amount=amount or None,
                note=note or None,
                reviewed_by=reviewed_by or None,
            )
            st.success("Invoice review saved.")
            _rerun()


def _render_subcontractor_bill_form(
    conn: sqlite3.Connection,
    *,
    job_id: int,
    current_rows: list[dict[str, Any]],
) -> None:
    if current_rows:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Status": row["bill_status"],
                        "Supplier": row.get("supplier_name"),
                        "Reference": row.get("bill_reference"),
                        "Date": row.get("bill_date"),
                        "Amount": row.get("amount"),
                    }
                    for row in current_rows
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    suppliers = conn.execute(
        "SELECT id, company_name FROM suppliers ORDER BY company_name"
    ).fetchall()
    supplier_options = {"<none>": None, **{str(row["company_name"]): int(row["id"]) for row in suppliers}}
    with st.form(f"operations_diary_bill_form_{job_id}"):
        status = st.selectbox("Bill status", options=list(SUBCONTRACTOR_BILL_STATUSES), index=1)
        supplier_label = st.selectbox("Supplier", options=list(supplier_options.keys()))
        bill_reference = st.text_input("Bill reference")
        bill_date = st.text_input("Bill date")
        amount = st.number_input("Bill amount", min_value=0.0, value=0.0, step=100.0)
        reviewed_by = st.text_input("Reviewed by", value="manager")
        note = st.text_area("Bill note")
        submitted = st.form_submit_button("Add subcontractor bill review")
        if submitted:
            upsert_subcontractor_bill_review(
                conn,
                job_id=job_id,
                supplier_id=supplier_options[supplier_label],
                bill_status=status,
                bill_reference=bill_reference or None,
                bill_date=bill_date or None,
                amount=amount or None,
                note=note or None,
                reviewed_by=reviewed_by or None,
            )
            st.success("Subcontractor bill review saved.")
            _rerun()


def _render_job_task_editor(
    conn: sqlite3.Connection,
    *,
    job_id: int,
    anchor_date: str,
    current_rows: list[dict[str, Any]],
) -> None:
    workers = conn.execute("SELECT id, name FROM workers ORDER BY name").fetchall()
    trucks = conn.execute("SELECT truck_id, name FROM trucks ORDER BY truck_id").fetchall()
    worker_options = {"<none>": None, **{str(row["name"]): int(row["id"]) for row in workers}}
    truck_options = {"<none>": None, **{str(row["truck_id"]): str(row["truck_id"]) for row in trucks}}

    edit_options = {"<new>": None, **{f"{row['task_date']} · {row['title']}": int(row["id"]) for row in current_rows}}
    selected_label = st.selectbox(
        "Task to edit",
        options=list(edit_options.keys()),
        key=f"operations_diary_task_editor_select_{job_id}",
    )
    selected_row = next((row for row in current_rows if int(row["id"]) == int(edit_options[selected_label])), None) if edit_options[selected_label] is not None else None

    with st.form(f"operations_diary_task_form_{job_id}"):
        task_date = st.text_input("Task date", value=str((selected_row or {}).get("task_date") or anchor_date))
        task_scope = st.selectbox(
            "Task scope",
            options=list(DIARY_TASK_SCOPES),
            index=list(DIARY_TASK_SCOPES).index(str((selected_row or {}).get("task_scope") or "day")),
        )
        task_type = st.text_input("Task type", value=str((selected_row or {}).get("task_type") or "follow_up"))
        title = st.text_input("Title", value=str((selected_row or {}).get("title") or ""))
        status = st.selectbox(
            "Status",
            options=list(DIARY_TASK_STATUSES),
            index=list(DIARY_TASK_STATUSES).index(str((selected_row or {}).get("status") or "open")),
        )
        worker_label = st.selectbox(
            "Assigned worker",
            options=list(worker_options.keys()),
            index=list(worker_options.values()).index(selected_row["assigned_worker_id"]) if selected_row and selected_row.get("assigned_worker_id") in worker_options.values() else 0,
        )
        truck_label = st.selectbox(
            "Assigned truck",
            options=list(truck_options.keys()),
            index=list(truck_options.values()).index(selected_row["assigned_truck_id"]) if selected_row and selected_row.get("assigned_truck_id") in truck_options.values() else 0,
        )
        planned_start = st.text_input("Planned start", value=str((selected_row or {}).get("planned_start") or ""))
        planned_end = st.text_input("Planned end", value=str((selected_row or {}).get("planned_end") or ""))
        notes = st.text_area("Notes", value=str((selected_row or {}).get("notes") or ""))
        save = st.form_submit_button("Save task")
        if save:
            upsert_operations_diary_task(
                conn,
                task_id=int(selected_row["id"]) if selected_row else None,
                job_id=job_id,
                task_date=task_date,
                task_scope=task_scope,
                task_type=task_type,
                title=title,
                status=status,
                assigned_worker_id=worker_options[worker_label],
                assigned_truck_id=truck_options[truck_label],
                planned_start=planned_start or None,
                planned_end=planned_end or None,
                notes=notes or None,
            )
            st.success("Diary task saved.")
            _rerun()

    if selected_row is not None and st.button("Delete selected task", key=f"operations_diary_task_delete_{job_id}"):
        delete_operations_diary_task(conn, task_id=int(selected_row["id"]))
        st.success("Diary task deleted.")
        _rerun()


def _render_global_task_form(conn: sqlite3.Connection, *, anchor_date: str) -> None:
    with st.form("operations_diary_global_task_form"):
        title = st.text_input("Create a day task")
        submitted = st.form_submit_button("Add day task")
        if submitted and title.strip():
            upsert_operations_diary_task(
                conn,
                task_date=anchor_date,
                title=title.strip(),
                task_scope="day",
            )
            st.success("Day task added.")
            _rerun()


def _rerun() -> None:
    rerun = getattr(st, "rerun", None)
    if callable(rerun):
        rerun()
    else:
        st.experimental_rerun()
