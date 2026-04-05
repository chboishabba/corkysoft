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
    list_observer_outbox_events,
    upsert_customer_invoice_review,
    upsert_operations_diary_task,
    upsert_subcontractor_bill_review,
)
from dashboard.query_params import _get_query_params, _set_workspace_query_params
from dashboard.state import _rerun_app


def _get_query_param(key: str, default: str) -> str:
    return _get_query_params().get(key, [default])[0]


def _set_operations_diary_workspace_params(
    *,
    view_mode: str,
    anchor_date: date,
    focus_job_id: int | None,
) -> None:
    _set_workspace_query_params(
        available_tabs=["Quote", "Pricing Intelligence", "Network", "Operations", "Admin"],
        view="Operations",
        workflow="operations_diary",
        diary_view=view_mode,
        diary_date=anchor_date.isoformat(),
        diary_job=str(focus_job_id or ""),
    )


def _labor_reconciliation_table_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return compact, diary-friendly table rows for labor reconciliation."""

    return [
        {
            "Status": row.get("status"),
            "Shift date": row.get("shiftDate"),
            "Worker": row.get("workerName"),
            "Trucks": _join_non_empty(row.get("truckIds", [])),
            "Planned start": row.get("plannedStart"),
            "Planned end": row.get("plannedEnd"),
            "Segment": row.get("segmentId"),
            "Source": row.get("source"),
        }
        for row in rows
    ]


def _labor_reconciliation_detail_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a key/value detail view for one labor reconciliation row."""

    return [
        {"Field": "Status", "Value": row.get("status")},
        {"Field": "Shift date", "Value": row.get("shiftDate")},
        {"Field": "Worker", "Value": row.get("workerName")},
        {"Field": "Worker ID", "Value": row.get("workerId")},
        {"Field": "Job", "Value": row.get("jobId")},
        {"Field": "Segment", "Value": row.get("segmentId")},
        {"Field": "Planned start", "Value": row.get("plannedStart")},
        {"Field": "Planned end", "Value": row.get("plannedEnd")},
        {"Field": "Trucks", "Value": _join_non_empty(row.get("truckIds", []))},
        {"Field": "Source", "Value": row.get("source")},
    ]


def _join_non_empty(values: Any) -> str:
    if not isinstance(values, list):
        return ""
    return ", ".join(str(value) for value in values if value not in (None, ""))


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
        _set_operations_diary_workspace_params(
            view_mode=view_mode,
            anchor_date=anchor_date,
            focus_job_id=focus_job_id,
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
    focus_scope = st.radio(
        "Diary focus",
        options=["all_jobs", "financial_exceptions", "operational_attention"],
        horizontal=True,
        format_func=lambda value: {
            "all_jobs": "All jobs",
            "financial_exceptions": "Financial exceptions",
            "operational_attention": "Operational attention",
        }[value],
        key="operations_diary_focus_scope",
    )
    if focus_scope == "financial_exceptions":
        job_rows = [
            row
            for row in job_rows
            if row["invoiceStatus"] in {"reconciliation_warning", "not_ready", "partially_invoiced"}
            or row["billStatus"] in {"pending_review", "received_unreconciled", "reconciliation_warning"}
        ]
    elif focus_scope == "operational_attention":
        job_rows = [
            row
            for row in job_rows
            if int(row["taskCount"]) > 0
            or int(row["segmentCount"]) == 0
            or row["status"] in {"review", "blocked", "warning"}
            or row["invoiceStatus"] == "reconciliation_warning"
            or row["billStatus"] == "reconciliation_warning"
        ]
    reconciliation_bucket = st.selectbox(
        "Reconciliation bucket",
        options=[
            "all",
            "invoice_exceptions",
            "supplier_bill_exceptions",
            "financial_exposure",
        ],
        format_func=lambda value: {
            "all": "All diary jobs",
            "invoice_exceptions": "Invoice exceptions",
            "supplier_bill_exceptions": "Supplier bill exceptions",
            "financial_exposure": "Any financial exposure",
        }[value],
        key="operations_diary_reconciliation_bucket",
    )
    if reconciliation_bucket == "invoice_exceptions":
        job_rows = [
            row
            for row in job_rows
            if row["invoiceStatus"] in {"not_ready", "ready_to_invoice", "partially_invoiced", "reconciliation_warning"}
        ]
    elif reconciliation_bucket == "supplier_bill_exceptions":
        job_rows = [
            row
            for row in job_rows
            if row["billStatus"] in {"awaiting_bill", "bill_received", "bill_exception"}
        ]
    elif reconciliation_bucket == "financial_exposure":
        job_rows = [
            row
            for row in job_rows
            if row["invoiceStatus"] in {"not_ready", "ready_to_invoice", "partially_invoiced", "reconciliation_warning"}
            or row["billStatus"] in {"awaiting_bill", "bill_received", "bill_exception"}
        ]
    if not job_rows:
        st.info("No diary jobs or tasks are present for the selected period yet.")
        _render_global_task_form(conn, anchor_date=anchor_date.isoformat())
        return

    action_rows: list[dict[str, Any]] = []
    for row in job_rows:
        if row["invoiceStatus"] in {"not_ready", "ready_to_invoice", "partially_invoiced", "reconciliation_warning"}:
            action_rows.append(
                {
                    "Priority": "financial",
                    "Action": "Invoice review",
                    "Job": row["jobId"],
                    "Client": row["jobClient"] or "Unknown client",
                    "Status": row["invoiceStatus"],
                    "Detail": "Customer invoice requires follow-through.",
                }
            )
        if row["billStatus"] in {"awaiting_bill", "bill_received", "bill_exception"}:
            action_rows.append(
                {
                    "Priority": "financial",
                    "Action": "Supplier bill review",
                    "Job": row["jobId"],
                    "Client": row["jobClient"] or "Unknown client",
                    "Status": row["billStatus"],
                    "Detail": "Subcontractor bill needs review or reconciliation.",
                }
            )
        if int(row["taskCount"]) > 0:
            action_rows.append(
                {
                    "Priority": "operations",
                    "Action": "Diary tasks open",
                    "Job": row["jobId"],
                    "Client": row["jobClient"] or "Unknown client",
                    "Status": f"{int(row['taskCount'])} task(s)",
                    "Detail": "Outstanding diary tasks in current window.",
                }
            )
    for exposure_row in exposure["activeSupplierRows"][:10]:
        action_rows.append(
            {
                "Priority": "financial",
                "Action": "Supplier exposure",
                "Job": exposure_row["jobId"],
                "Client": exposure_row["jobClient"] or "Unknown client",
                "Status": exposure_row["status"],
                "Detail": (
                    f"Unresolved ${float(exposure_row['amount'] or 0.0):,.0f} "
                    f"aged {int(exposure_row['unresolvedAgeDays'] or 0)} day(s)."
                ),
            }
        )
    if action_rows:
        st.markdown("#### Action queue")
        st.dataframe(
            pd.DataFrame(action_rows),
            width="stretch",
            hide_index=True,
        )

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
    _set_operations_diary_workspace_params(
        view_mode=view_mode,
        anchor_date=anchor_date,
        focus_job_id=selected_job_id,
    )

    details = build_job_usage_details(conn, job_id=selected_job_id)
    job = details["job"]
    header_cols = st.columns(3)
    if header_cols[0].button("Open in Planner", key=f"operations_diary_to_planner_{selected_job_id}"):
        _set_workspace_query_params(
            available_tabs=["Quote", "Pricing Intelligence", "Network", "Operations", "Admin"],
            view="Operations",
            workflow="planner",
        )
    if header_cols[1].button("Open in Dispatch", key=f"operations_diary_to_dispatch_{selected_job_id}"):
        _set_workspace_query_params(
            available_tabs=["Quote", "Pricing Intelligence", "Network", "Operations", "Admin"],
            view="Operations",
            workflow="dispatch",
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

    st.markdown("#### Labor reconciliation")
    labor_df = pd.DataFrame(details["laborReconciliation"])
    if labor_df.empty:
        st.caption("No labor reconciliation rows are linked to this job yet.")
    else:
        recon_cols = st.columns(3)
        recon_cols[0].metric("Planned only", int((labor_df["status"] == "planned_only").sum()))
        recon_cols[1].metric("Imported only", int((labor_df["status"] == "imported_only").sum()))
        recon_cols[2].metric("Matched", int((labor_df["status"] == "matched").sum()))
        display_df = labor_df.copy()
        display_df["truckIds"] = display_df["truckIds"].apply(
            lambda values: ", ".join(values) if isinstance(values, list) else ""
        )
        st.dataframe(
            display_df[
                [
                    "status",
                    "shiftDate",
                    "workerName",
                    "truckIds",
                    "segmentId",
                    "plannedStart",
                    "plannedEnd",
                    "source",
                ]
            ].rename(
                columns={
                    "status": "Status",
                    "shiftDate": "Date",
                    "workerName": "Worker",
                    "truckIds": "Trucks",
                    "segmentId": "Segment",
                    "plannedStart": "Planned start",
                    "plannedEnd": "Planned end",
                    "source": "Source",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    fin_cols = st.columns(2)
    with fin_cols[0]:
        st.markdown("#### Customer invoice review")
        _render_customer_invoice_form(conn, job_id=selected_job_id, current=details["invoiceReview"])
    with fin_cols[1]:
        st.markdown("#### Subcontractor bill review")
        _render_subcontractor_bill_form(conn, job_id=selected_job_id, current_rows=details["billReviews"])

    next_action_cols = st.columns(3)
    if details["invoiceReview"] and details["invoiceReview"]["invoice_status"] in {
        "not_ready",
        "ready_to_invoice",
        "partially_invoiced",
        "reconciliation_warning",
    }:
        if next_action_cols[0].button(
            "Create invoice follow-up task",
            key=f"operations_diary_invoice_follow_up_{selected_job_id}",
        ):
            upsert_operations_diary_task(
                conn,
                job_id=selected_job_id,
                task_date=anchor_date.isoformat(),
                task_scope="job",
                task_type="invoice_review",
                title=f"Follow up customer invoice for job #{selected_job_id}",
                notes=f"Invoice status: {details['invoiceReview']['invoice_status']}",
            )
            st.success("Invoice follow-up task created.")
            _rerun_app()
    if any(
        row["bill_status"] in {"awaiting_bill", "bill_received", "bill_exception"}
        for row in details["billReviews"]
    ):
        if next_action_cols[1].button(
            "Create bill follow-up task",
            key=f"operations_diary_bill_follow_up_{selected_job_id}",
        ):
            upsert_operations_diary_task(
                conn,
                job_id=selected_job_id,
                task_date=anchor_date.isoformat(),
                task_scope="job",
                task_type="bill_review",
                title=f"Follow up subcontractor bill for job #{selected_job_id}",
                notes="Created from operations diary bill exception review.",
            )
            st.success("Bill follow-up task created.")
            _rerun_app()

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
        severity = details["reconciliationExposure"].get("exposureSeverity")
        if severity and severity != "none":
            st.caption(
                f"Overall exposure severity: {severity.replace('_', ' ').capitalize()}."
            )

    _render_observer_outbox_section(conn, selected_job_id=selected_job_id)

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


def _render_observer_outbox_section(
    conn: sqlite3.Connection,
    *,
    selected_job_id: int,
) -> None:
    with st.expander("Observer outbox", expanded=False):
        st.caption(
            "Inspect emitted planning snapshots, reconciliation exceptions, and review/task envelopes without leaving the manager workflow."
        )
        scope_col, family_col, limit_col = st.columns(3)
        selected_job_only = scope_col.checkbox(
            "Only selected job",
            value=True,
            key=f"operations_diary_observer_selected_job_{selected_job_id}",
        )
        discovery_rows = list_observer_outbox_events(
            conn,
            limit=200,
            job_id=selected_job_id if selected_job_only else None,
        )
        family_options = ["all"] + sorted({str(row["eventFamily"]) for row in discovery_rows})
        selected_family = family_col.selectbox(
            "Event family",
            options=family_options,
            key=f"operations_diary_observer_family_{selected_job_id}",
        )
        limit = int(
            limit_col.selectbox(
                "Rows",
                options=[10, 25, 50, 100],
                index=1,
                key=f"operations_diary_observer_limit_{selected_job_id}",
            )
        )
        rows = list_observer_outbox_events(
            conn,
            limit=limit,
            event_family=None if selected_family == "all" else selected_family,
            job_id=selected_job_id if selected_job_only else None,
        )
        if not rows:
            st.caption("No observer events match the current filters.")
            return

        family_counts: dict[str, int] = {}
        for row in rows:
            family = str(row["eventFamily"])
            family_counts[family] = family_counts.get(family, 0) + 1

        metric_cols = st.columns(4)
        metric_cols[0].metric("Events", len(rows))
        metric_cols[1].metric("Families", len(family_counts))
        metric_cols[2].metric("Latest family", str(rows[0]["eventFamily"]))
        metric_cols[3].metric("Latest status", str(rows[0].get("status") or "n/a"))

        st.dataframe(
            pd.DataFrame([_observer_summary_row(row) for row in rows]),
            width="stretch",
            hide_index=True,
        )

        for row in rows[: min(len(rows), 10)]:
            header = (
                f"{row['eventFamily']} · {row['summary']} · "
                f"{row.get('recordedAt') or row.get('eventTime')}"
            )
            with st.expander(header, expanded=False):
                detail_cols = st.columns(4)
                detail_cols[0].metric("Authority", str(row.get("authorityClass") or "n/a"))
                detail_cols[1].metric("Actor", str(row.get("actorRef") or "n/a"))
                detail_cols[2].metric("Job", _observer_job_label(row))
                detail_cols[3].metric("Status", str(row.get("status") or "n/a"))
                st.markdown("**Object refs**")
                st.json(row.get("objectRefs") or {})
                st.markdown("**Payload**")
                st.json(row.get("payload") or {})
                provenance = row.get("provenanceRefs") or []
                evidence = row.get("evidenceRefs") or []
                if provenance:
                    st.markdown("**Provenance refs**")
                    st.json(provenance)
                if evidence:
                    st.markdown("**Evidence refs**")
                    st.json(evidence)


def _observer_job_label(row: dict[str, Any]) -> str:
    object_refs = row.get("objectRefs") or {}
    payload = row.get("payload") or {}
    job_id = object_refs.get("job_id") or payload.get("jobId")
    return str(job_id) if job_id is not None else "n/a"


def _observer_summary_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "Recorded at": row.get("recordedAt"),
        "Family": row.get("eventFamily"),
        "Type": row.get("eventType"),
        "Job": _observer_job_label(row),
        "Summary": row.get("summary"),
        "Status": row.get("status"),
        "Actor": row.get("actorRef"),
        "Authority": row.get("authorityClass"),
        "Source entity": row.get("sourceEntityId"),
    }


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
            _rerun_app()


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
            _rerun_app()


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
            _rerun_app()

    if selected_row is not None and st.button("Delete selected task", key=f"operations_diary_task_delete_{job_id}"):
        delete_operations_diary_task(conn, task_id=int(selected_row["id"]))
        st.success("Diary task deleted.")
        _rerun_app()


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
            _rerun_app()
