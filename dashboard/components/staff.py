from __future__ import annotations

import io
import math
import os
import sqlite3
from typing import Callable, Optional

import pandas as pd
import streamlit as st

from analytics.db import (
    ensure_dashboard_tables,
    import_workers_from_google_sheet,
    import_workers_from_staff_sheet,
    upsert_worker,
)
from analytics.driver_shifts import load_driver_shifts_dataframe
from analytics.operations_assignment import (
    assign_worker_compliance,
    assign_worker_role,
    ensure_worker_compliance,
    ensure_worker_role,
    list_operational_readiness_items,
    list_segments_for_worker,
    list_worker_assignment_summary,
)
from dashboard.components.worker_time import (
    render_worker_time_review_controls,
    worker_time_events_df,
)


def _split_worker_name(name: str) -> tuple[str, str]:
    parts = name.strip().split(" ", 1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def _format_truck_list(truck_string: str | float | None) -> str:
    if truck_string is None or (isinstance(truck_string, float) and pd.isna(truck_string)):
        return ""
    trucks = {truck.strip() for truck in str(truck_string).split(",") if truck.strip()}
    return ", ".join(sorted(trucks))


def _default_operations_sheet_reference() -> str:
    return (
        os.environ.get("OPERATIONS_WORKBOOK_URL")
        or os.environ.get("OPERATIONS_WORKBOOK_SHEET_ID")
        or ""
    )


def _prepare_staff_export(workers_df: pd.DataFrame) -> bytes:
    export_df = workers_df.copy()
    first_names: list[str] = []
    last_names: list[str] = []
    for _, row in export_df.iterrows():
        first, last = _split_worker_name(str(row.get("name", "")))
        first_names.append(first)
        last_names.append(last)

    export_df.insert(0, "FIRST NAME", first_names)
    export_df.insert(1, "LAST NAME", last_names)
    export_df = export_df[
        [
            "FIRST NAME",
            "LAST NAME",
            "role",
            "rate",
            "tickets",
            "phone",
            "active",
        ]
    ].rename(columns={"role": "ROLE", "rate": "RATE", "tickets": "TICKETS"})

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        export_df.to_excel(writer, sheet_name="STAFF", index=False)
    buffer.seek(0)
    return buffer.getvalue()


def render_staff_tab(
    conn: sqlite3.Connection,
    *,
    rerun_app: Callable[[], None],
) -> None:
    st.subheader("Staff roster (STAFF)")
    st.caption(
        "Import, audit, and edit the STAFF worksheet. Link workers to driver shifts and vehicles."
    )

    ensure_dashboard_tables(conn)
    assignment_summary = list_worker_assignment_summary(conn)
    import_feedback: Optional[tuple[str, str]] = None

    st.markdown("### Operational overview")
    st.caption("Review and manage the live roster before touching import/export or governance controls.")

    workers_df = pd.read_sql_query(
        """
        SELECT
            w.id,
            w.name,
            w.role,
            w.phone,
            w.rate,
            w.tickets,
            w.active,
            w.hired_at,
            w.updated_at,
            COUNT(ds.id) AS shift_count,
            MAX(ds.shift_date) AS last_shift_date,
            GROUP_CONCAT(DISTINCT ds.truck_id) AS shift_trucks
        FROM workers AS w
        LEFT JOIN driver_shifts AS ds ON ds.worker_id = w.id
        GROUP BY w.id
        ORDER BY w.name
        """,
        conn,
    )

    vehicle_df = pd.read_sql_query(
        """
        SELECT truck_id, present_driver
        FROM vehicle_details
        WHERE present_driver IS NOT NULL AND TRIM(present_driver) != ''
        """,
        conn,
    )
    vehicle_assignments = (
        vehicle_df.groupby("present_driver")["truck_id"]
        .apply(lambda series: ", ".join(sorted({str(val).strip() for val in series if str(val).strip()})))
        .to_dict()
    )

    if not workers_df.empty:
        workers_df["active"] = workers_df["active"].astype(bool)
        workers_df["last_shift_date"] = pd.to_datetime(
            workers_df["last_shift_date"], errors="coerce"
        ).dt.date
        workers_df["shift_trucks"] = workers_df["shift_trucks"].apply(_format_truck_list)
        workers_df["imported_trucks"] = workers_df["name"].map(vehicle_assignments).fillna("")
        workers_df["planned_segment_count"] = workers_df["id"].map(
            lambda worker_id: assignment_summary.get(int(worker_id), {}).get("plannedSegmentCount", 0)
        )
        workers_df["planned_job_count"] = workers_df["id"].map(
            lambda worker_id: assignment_summary.get(int(worker_id), {}).get("plannedJobCount", 0)
        )
        workers_df["planned_trucks"] = workers_df["id"].map(
            lambda worker_id: ", ".join(
                assignment_summary.get(int(worker_id), {}).get("plannedTrucks", [])
            )
        )
        workers_df["next_planned_start"] = pd.to_datetime(
            workers_df["id"].map(
                lambda worker_id: assignment_summary.get(int(worker_id), {}).get("nextPlannedStart")
            ),
            errors="coerce",
        ).dt.date
        workers_df["shift_count"] = workers_df["shift_count"].fillna(0).astype(int)

    summary_cols = st.columns(3)
    summary_cols[0].metric("Total workers", int(len(workers_df)))
    active_count = int(workers_df[workers_df["active"]].shape[0]) if not workers_df.empty else 0
    summary_cols[1].metric("Active workers", active_count)
    summary_cols[2].metric(
        "Workers on planned segments",
        int((workers_df["planned_segment_count"] > 0).sum()) if not workers_df.empty else 0,
    )

    st.markdown("#### Roster filters")
    filter_cols = st.columns(3)
    name_filter = filter_cols[0].text_input("Search by name", key="staff_name_filter")
    role_options = (
        sorted(
            filter(
                lambda r: bool(r),
                workers_df.get("role").dropna().unique().tolist(),
            )
        )
        if not workers_df.empty and "role" in workers_df
        else []
    )
    role_filter = filter_cols[1].multiselect("Roles", role_options, key="staff_role_filter")
    status_filter = filter_cols[2].selectbox(
        "Active status",
        ["All", "Active", "Inactive"],
        key="staff_status_filter",
    )

    filtered_df = workers_df.copy()
    if name_filter:
        filtered_df = filtered_df[
            filtered_df["name"].str.contains(name_filter, case=False, na=False)
        ]
    if role_filter:
        filtered_df = filtered_df[filtered_df["role"].isin(role_filter)]
    if status_filter == "Active":
        filtered_df = filtered_df[filtered_df["active"]]
    elif status_filter == "Inactive":
        filtered_df = filtered_df[~filtered_df["active"]]

    st.markdown("#### Live roster editor")
    display_columns = [
        "id",
        "name",
        "role",
        "phone",
        "rate",
        "tickets",
        "active",
        "last_shift_date",
        "shift_count",
        "planned_segment_count",
        "planned_job_count",
        "next_planned_start",
        "planned_trucks",
        "shift_trucks",
        "imported_trucks",
    ]
    present_columns = [col for col in display_columns if col in filtered_df.columns]
    edited_df = st.data_editor(
        filtered_df[present_columns],
        hide_index=True,
        width="stretch",
        num_rows="dynamic",
        column_config={
            "id": st.column_config.Column("ID", disabled=True, width="small"),
            "name": st.column_config.Column("Name"),
            "role": st.column_config.Column("Role"),
            "phone": st.column_config.Column("Phone"),
            "rate": st.column_config.NumberColumn("Rate", format="%.2f"),
            "tickets": st.column_config.NumberColumn("Tickets", format="%d"),
            "active": st.column_config.CheckboxColumn("Active"),
            "last_shift_date": st.column_config.DateColumn("Last shift", disabled=True),
            "shift_count": st.column_config.NumberColumn("Shift count", disabled=True),
            "planned_segment_count": st.column_config.NumberColumn("Planned segments", disabled=True),
            "planned_job_count": st.column_config.NumberColumn("Planned jobs", disabled=True),
            "next_planned_start": st.column_config.DateColumn("Next planned start", disabled=True),
            "planned_trucks": st.column_config.Column("Planned trucks", disabled=True),
            "shift_trucks": st.column_config.Column("Recent trucks", disabled=True),
            "imported_trucks": st.column_config.Column("Imported sheet trucks", disabled=True),
        },
    )

    if st.button("Save staff changes", type="primary", key="staff_save_button"):
        if edited_df.empty:
            st.info("No staff rows to save.")
        else:
            errors: list[str] = []
            saved = 0
            for idx, row in edited_df.iterrows():
                name = str(row.get("name") or "").strip()
                if not name:
                    errors.append(f"Row {idx + 1}: name is required.")
                    continue

                rate_raw = row.get("rate")
                rate_value: float | None
                if rate_raw in ("", None) or (isinstance(rate_raw, float) and math.isnan(rate_raw)):
                    rate_value = None
                else:
                    try:
                        rate_value = float(rate_raw)
                        if rate_value < 0:
                            raise ValueError("Rate cannot be negative")
                    except Exception as exc:
                        errors.append(f"{name}: invalid rate ({exc}).")
                        continue

                tickets_raw = row.get("tickets")
                tickets_value: int | None
                if tickets_raw in ("", None) or (isinstance(tickets_raw, float) and math.isnan(tickets_raw)):
                    tickets_value = None
                else:
                    try:
                        tickets_value = int(tickets_raw)
                        if tickets_value < 0:
                            raise ValueError("Tickets cannot be negative")
                    except Exception as exc:
                        errors.append(f"{name}: invalid tickets ({exc}).")
                        continue

                try:
                    upsert_worker(
                        conn,
                        name=name,
                        role=str(row.get("role") or ""),
                        phone=str(row.get("phone") or ""),
                        rate=rate_value,
                        tickets=tickets_value,
                        active=bool(row.get("active")),
                    )
                except Exception as exc:  # pragma: no cover - surfaced in UI
                    errors.append(f"{name}: failed to save ({exc}).")
                else:
                    saved += 1

            if errors:
                st.error("\n".join(errors))
            if saved and not errors:
                st.success(f"Saved {saved} staff record{'s' if saved != 1 else ''}.")
                rerun_app()
            elif saved:
                st.info(
                    f"Saved {saved} staff record{'s' if saved != 1 else ''}. Fix the remaining issues and try again."
                )

    st.divider()
    st.markdown("### Worker review and linked shifts")
    st.caption("Inspect a worker to see recent shifts, planned segments, and worker-time events.")
    st.subheader("Linked shifts and vehicle assignments")
    if workers_df.empty:
        st.info("No staff available to display shift links.")
        return

    worker_choice = st.selectbox(
        "Choose a worker to review recent shifts and vehicles",
        sorted(workers_df["name"].tolist()),
        key="staff_worker_review",
    )
    if worker_choice:
        worker_row = workers_df.loc[workers_df["name"] == worker_choice].iloc[0]
        worker_time_df = worker_time_events_df(conn, limit=500)
        if not worker_time_df.empty:
            worker_time_df = worker_time_df[
                worker_time_df["workerId"].fillna(-1).astype(int) == int(worker_row["id"])
            ].copy()
        planned_segments = list_segments_for_worker(conn, worker_id=int(worker_row["id"]))
        if planned_segments:
            planned_df = pd.DataFrame(
                [
                    {
                        "Job": row["jobId"],
                        "Segment": row["segmentSequence"],
                        "From": row["fromLocation"] or row["jobOrigin"],
                        "To": row["toLocation"] or row["jobDestination"],
                        "Planned start": row["plannedStart"],
                        "Planned end": row["plannedEnd"],
                        "Status": row["assignmentStatus"],
                        "Trucks": ", ".join(
                            assignment["truckId"]
                            for assignment in row["truckAssignments"]
                            if assignment.get("truckId")
                        ),
                    }
                    for row in planned_segments
                ]
            )
            st.markdown("#### Planned segment assignments")
            st.dataframe(planned_df, width="stretch", hide_index=True)

        shift_df = load_driver_shifts_dataframe(conn, workers=[worker_choice])
        if not shift_df.empty:
            shift_df = shift_df.copy()
            shift_df["shift_date"] = pd.to_datetime(shift_df["shift_date"], errors="coerce").dt.date
            columns = [
                "shift_date",
                "truck_id",
                "truck_name",
                "shift_window_start",
                "shift_window_end",
                "shift_start",
                "shift_end",
                "hours",
                "hourly_rate",
                "cost_total",
                "source",
            ]
            present_shift_cols = [col for col in columns if col in shift_df.columns]
            st.dataframe(
                shift_df.sort_values(by="shift_date", ascending=False)[present_shift_cols],
                width="stretch",
            )
        else:
            st.caption("No driver shifts linked to this worker yet.")

        imported_trucks = vehicle_assignments.get(worker_choice)
        if imported_trucks:
            st.info(f"Imported sheet truck context: {imported_trucks}")
        elif not shift_df.empty:
            st.caption("No imported sheet truck assignment; trucks only appear in recorded shifts.")
        else:
            st.caption("No imported sheet truck assignment recorded for this worker.")

        st.markdown("#### Reviewed worker-time events")
        if worker_time_df.empty:
            st.caption("No worker-time capture events are linked to this worker yet.")
        else:
            review_cols = st.columns(4)
            review_cols[0].metric(
                "Pending review",
                int((worker_time_df["reviewStatus"] == "pending_review").sum()),
            )
            review_cols[1].metric(
                "Accepted",
                int((worker_time_df["reviewStatus"] == "accepted").sum()),
            )
            review_cols[2].metric(
                "Rejected",
                int((worker_time_df["reviewStatus"] == "rejected").sum()),
            )
            latest_reviewed = worker_time_df["reviewedAt"].dropna()
            review_cols[3].metric(
                "Latest reviewed",
                latest_reviewed.max().date().isoformat() if not latest_reviewed.empty else "n/a",
            )
            st.dataframe(
                worker_time_df[
                    [
                        "id",
                        "eventType",
                        "channel",
                        "effectiveTimestamp",
                        "confidence",
                        "reviewStatus",
                        "reviewer",
                        "reviewedAt",
                        "jobId",
                        "segmentId",
                        "truckId",
                        "anomalyFlags",
                    ]
                ].rename(
                    columns={
                        "id": "Event",
                        "eventType": "Event type",
                        "channel": "Channel",
                        "effectiveTimestamp": "Effective time",
                        "confidence": "Confidence",
                        "reviewStatus": "Review",
                        "reviewer": "Reviewer",
                        "reviewedAt": "Reviewed at",
                        "jobId": "Job",
                        "segmentId": "Segment",
                        "truckId": "Truck",
                        "anomalyFlags": "Anomalies",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
            with st.expander("Review pending worker-time events", expanded=False):
                pending_worker_time = worker_time_df[
                    worker_time_df["reviewStatus"] == "pending_review"
                ].copy()
                render_worker_time_review_controls(
                    conn,
                    pending_events=pending_worker_time,
                    key_prefix=f"staff_worker_time_{int(worker_row['id'])}",
                    rerun_app=rerun_app,
                )

    st.divider()
    st.subheader("Roles and compliances")
    if workers_df.empty:
        st.caption("Add staff before managing roles and compliances.")
        return

    admin_cols = st.columns(2)
    with admin_cols[0]:
        selected_admin_worker = st.selectbox(
            "Worker for role/compliance admin",
            sorted(workers_df["name"].tolist()),
            key="staff_worker_admin",
        )
    worker_admin_row = workers_df.loc[workers_df["name"] == selected_admin_worker].iloc[0]
    worker_admin_id = int(worker_admin_row["id"])

    role_rows = conn.execute(
        """
        SELECT wr.id, wr.name
        FROM worker_role_assignments AS wra
        JOIN worker_roles AS wr ON wr.id = wra.role_id
        WHERE wra.worker_id = ?
        ORDER BY wr.name
        """,
        (worker_admin_id,),
    ).fetchall()
    compliance_rows = conn.execute(
        """
        SELECT wc.id, wc.name, wca.expiry_date
        FROM worker_compliance_assignments AS wca
        JOIN worker_compliances AS wc ON wc.id = wca.compliance_id
        WHERE wca.worker_id = ?
        ORDER BY wc.name
        """,
        (worker_admin_id,),
    ).fetchall()

    role_col, compliance_col = st.columns(2)
    with role_col:
        st.markdown("#### Role assignments")
        if role_rows:
            st.dataframe(
                pd.DataFrame([{"Role": row["name"]} for row in role_rows]),
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("No role assignments recorded.")
        available_roles = conn.execute(
            "SELECT id, name FROM worker_roles ORDER BY name"
        ).fetchall()
        role_options = {row["name"]: int(row["id"]) for row in available_roles}
        selected_role_name = st.selectbox(
            "Existing role",
            options=["<new role>", *role_options.keys()],
            key="staff_role_existing_select",
        )
        new_role_name = st.text_input("New role name", value="", key="staff_new_role_name")
        if st.button("Assign role", key="staff_assign_role_button"):
            try:
                role_id = (
                    ensure_worker_role(conn, name=new_role_name.strip())
                    if selected_role_name == "<new role>"
                    else role_options[selected_role_name]
                )
                assign_worker_role(conn, worker_id=worker_admin_id, role_id=role_id)
            except Exception as exc:  # pragma: no cover
                st.error(f"Failed to assign role: {exc}")
            else:
                st.success("Role assignment saved.")
                rerun_app()

    with compliance_col:
        st.markdown("#### Compliance assignments")
        if compliance_rows:
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Compliance": row["name"], "Expiry": row["expiry_date"]}
                        for row in compliance_rows
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("No compliance assignments recorded.")
        available_compliances = conn.execute(
            "SELECT id, name FROM worker_compliances ORDER BY name"
        ).fetchall()
        compliance_options = {row["name"]: int(row["id"]) for row in available_compliances}
        selected_compliance_name = st.selectbox(
            "Existing compliance",
            options=["<new compliance>", *compliance_options.keys()],
            key="staff_compliance_existing_select",
        )
        new_compliance_name = st.text_input(
            "New compliance name", value="", key="staff_new_compliance_name"
        )
        expiry_value = st.text_input(
            "Compliance expiry (ISO date)",
            value="",
            placeholder="2026-12-31",
            key="staff_compliance_expiry",
        )
        if st.button("Assign compliance", key="staff_assign_compliance_button"):
            try:
                compliance_id = (
                    ensure_worker_compliance(conn, name=new_compliance_name.strip())
                    if selected_compliance_name == "<new compliance>"
                    else compliance_options[selected_compliance_name]
                )
                assign_worker_compliance(
                    conn,
                    worker_id=worker_admin_id,
                    compliance_id=compliance_id,
                    expiry_date=expiry_value.strip() or None,
                )
            except Exception as exc:  # pragma: no cover
                st.error(f"Failed to assign compliance: {exc}")
            else:
                st.success("Compliance assignment saved.")
                rerun_app()

    worker_readiness_items = [
        item
        for item in list_operational_readiness_items(conn, resource_type="worker")
        if item["resourceId"] == str(worker_admin_id)
    ]
    if worker_readiness_items:
        st.markdown("#### Worker readiness alerts")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Status": item["status"],
                        "Rule": item["ruleType"],
                        "Due": item["dueAt"],
                        "Details": item["details"],
                    }
                    for item in worker_readiness_items
                ]
            ),
            width="stretch",
            hide_index=True,
        )

    st.divider()
    st.markdown("### Import/export STAFF worksheet")
    st.caption("Import STAFF data after verifying the roster above, or download the current sheet.")
    with st.expander("Import/export STAFF worksheet", expanded=False):
        google_col, import_col, export_col = st.columns(3)
        with google_col:
            staff_sheet_reference = st.text_input(
                "Google Sheets ID or URL",
                value=_default_operations_sheet_reference(),
                help="Shared operations workbook containing the STAFF tab.",
                key="staff_sheet_reference",
            )
            if st.button("Import STAFF from Google Sheet", key="staff_google_import_button"):
                try:
                    inserted, updated = import_workers_from_google_sheet(
                        conn,
                        sheet_id_or_url=staff_sheet_reference.strip() or None,
                    )
                except Exception as exc:  # pragma: no cover - surfaced in UI
                    import_feedback = (
                        "error",
                        f"Failed to import staff from Google Sheets: {exc}",
                    )
                else:
                    import_feedback = (
                        "success",
                        f"Imported {inserted} new staff and updated {updated} existing records from Google Sheets.",
                    )
        with import_col:
            staff_upload = st.file_uploader(
                "Upload STAFF workbook (.xlsx)",
                type=["xlsx"],
                help="Re-use the STAFF worksheet downloaded from Google Sheets.",
                key="staff_upload_widget",
            )
            if st.button("Import STAFF", key="staff_import_button"):
                if staff_upload is None:
                    import_feedback = (
                        "warning",
                        "Choose a STAFF workbook before importing.",
                    )
                else:
                    try:
                        inserted, updated = import_workers_from_staff_sheet(
                            conn, staff_upload
                        )
                    except Exception as exc:  # pragma: no cover - surfaced in UI
                        import_feedback = (
                            "error",
                            f"Failed to import staff: {exc}",
                        )
                    else:
                        import_feedback = (
                            "success",
                            f"Imported {inserted} new staff and updated {updated} existing records.",
                        )

        with export_col:
            workers_for_export = pd.read_sql_query(
                "SELECT name, role, rate, tickets, phone, active FROM workers ORDER BY name",
                conn,
            )
            if workers_for_export.empty:
                st.caption(
                    "Add staff before exporting a workbook compatible with the STAFF sheet."
                )
            else:
                export_bytes = _prepare_staff_export(workers_for_export)
                st.download_button(
                    "Download STAFF workbook",
                    export_bytes,
                    file_name="STAFF.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="staff_export_button",
                )

    if import_feedback:
        level, message = import_feedback
        if level == "success":
            st.success(message)
        elif level == "warning":
            st.info(message)
        else:
            st.error(message)
