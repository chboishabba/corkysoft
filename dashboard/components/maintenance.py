from __future__ import annotations

import os
from typing import Iterable, Optional

import pandas as pd
import streamlit as st

from analytics.adaptive_policy import (
    approve_adaptive_policy_proposal,
    apply_adaptive_policy_proposal,
    list_adaptive_policy_proposals,
    load_adaptive_policy_snapshot,
    reject_adaptive_policy_proposal,
)
from analytics.dashboard_layouts import get_dashboard_role_layouts, upsert_dashboard_role_layout
from analytics.db import ensure_dashboard_tables, upsert_truck, upsert_vehicle_details
from analytics.lane_assignment import (
    LANE_PROPOSAL_STATUS_APPROVED,
    LANE_PROPOSAL_STATUS_PENDING_REVIEW,
    apply_lane_promotion_proposal,
    approve_lane_promotion_proposal,
    create_lane_promotion_proposal,
    create_lane_promotion_proposal_for_clusters,
    ensure_lane_assignment_schema,
    list_lane_promotion_proposals,
    reject_lane_promotion_proposal,
)
from analytics.price_distribution import latest_historical_ingest_summary
from analytics.situational_awareness import update_adaptive_policy_from_disruptions
from analytics.vehicle_repairs import (
    import_vehicle_repairs_from_sheet,
    load_vehicle_repairs,
)
from analytics.operations_workbook import sync_operations_workbook
from analytics.operations_assignment import (
    approve_operations_cutover_promotion,
    apply_operations_cutover_recommendation,
    get_operations_policy,
    list_operational_readiness_items,
    list_operations_cutover_events,
    list_operations_cutover_rollout,
    list_operations_cutover_workflows,
    list_segments_for_truck,
    list_truck_assignment_summary,
    reject_operations_cutover_promotion,
    record_operations_cutover_event,
    request_operations_cutover_promotion,
    upsert_operations_cutover_workflow,
    update_operations_policy,
)
from analytics.vehicle_workbook import (
    import_vehicle_details_from_dataframe,
    import_vehicle_details_from_workbook,
    import_vehicle_details_from_google_sheet,
)

from dashboard.state import _rerun_app


def _format_currency(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "$0"
    return f"${value:,.0f}"


def load_vehicle_overview(conn) -> pd.DataFrame:
    ensure_dashboard_tables(conn)
    query = """
        SELECT
            t.truck_id,
            t.name,
            t.capacity_m3,
            t.active,
            t.notes,
            vd.state,
            vd.rego,
            vd.rego_expiry,
            vd.make,
            vd.model,
            vd.year,
            vd.body_type,
            vd.description,
            vd.nhv_code,
            vd.insurance,
            vd.odometer,
            vd.last_service,
            vd.next_service,
            vd.coi_number,
            vd.coi_due,
            vd.present_driver,
            vd.daily_check_complete
        FROM trucks AS t
        LEFT JOIN vehicle_details AS vd ON vd.truck_id = t.truck_id
        ORDER BY t.truck_id
    """
    return pd.read_sql_query(query, conn)


def _lane_assignment_health_summary(conn) -> pd.DataFrame:
    ensure_lane_assignment_schema(conn)
    return pd.read_sql_query(
        """
        SELECT
            dataset,
            lane_assignment_status,
            row_count
        FROM (
            SELECT
                'historical' AS dataset,
                COALESCE(NULLIF(TRIM(lane_assignment_status), ''), 'unassigned') AS lane_assignment_status,
                COUNT(*) AS row_count
            FROM historical_jobs
            GROUP BY 1, 2

            UNION ALL

            SELECT
                'live' AS dataset,
                COALESCE(NULLIF(TRIM(lane_assignment_status), ''), 'unassigned') AS lane_assignment_status,
                COUNT(*) AS row_count
            FROM jobs
            GROUP BY 1, 2
        )
        ORDER BY dataset, lane_assignment_status
        """,
        conn,
    )


def _recent_lane_assignment_gaps(conn, *, limit: int = 20) -> pd.DataFrame:
    ensure_lane_assignment_schema(conn)
    historical_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(historical_jobs)").fetchall()
    }
    live_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
    }
    historical_origin_cluster = (
        "origin_cluster_key" if "origin_cluster_key" in historical_columns else "NULL"
    )
    historical_destination_cluster = (
        "destination_cluster_key" if "destination_cluster_key" in historical_columns else "NULL"
    )
    live_origin_cluster = (
        "origin_cluster_key" if "origin_cluster_key" in live_columns else "NULL"
    )
    live_destination_cluster = (
        "destination_cluster_key" if "destination_cluster_key" in live_columns else "NULL"
    )
    return pd.read_sql_query(
        f"""
        SELECT
            dataset,
            row_id,
            reference,
            corridor_display,
            origin_cluster_key,
            destination_cluster_key,
            lane_assignment_status,
            lane_assignment_source,
            lane_assignment_note,
            updated_at
        FROM (
            SELECT
                'historical' AS dataset,
                id AS row_id,
                COALESCE(client, CAST(id AS TEXT)) AS reference,
                corridor_display,
                {historical_origin_cluster} AS origin_cluster_key,
                {historical_destination_cluster} AS destination_cluster_key,
                COALESCE(NULLIF(TRIM(lane_assignment_status), ''), 'unassigned') AS lane_assignment_status,
                lane_assignment_source,
                lane_assignment_note,
                updated_at
            FROM historical_jobs
            WHERE COALESCE(NULLIF(TRIM(lane_assignment_status), ''), 'unassigned') != 'assigned'

            UNION ALL

            SELECT
                'live' AS dataset,
                id AS row_id,
                COALESCE(client, CAST(id AS TEXT)) AS reference,
                COALESCE(origin, '?') || ' → ' || COALESCE(destination, '?') AS corridor_display,
                {live_origin_cluster} AS origin_cluster_key,
                {live_destination_cluster} AS destination_cluster_key,
                COALESCE(NULLIF(TRIM(lane_assignment_status), ''), 'unassigned') AS lane_assignment_status,
                lane_assignment_source,
                lane_assignment_note,
                updated_at
            FROM jobs
            WHERE COALESCE(NULLIF(TRIM(lane_assignment_status), ''), 'unassigned') != 'assigned'
        )
        ORDER BY updated_at DESC, row_id DESC
        LIMIT ?
        """,
        conn,
        params=(int(limit),),
    )


def render_vehicle_maintenance_tab(conn) -> None:
    """Render the maintenance history tab."""

    st.markdown("### Vehicle maintenance history")
    st.caption("Import workshop visits from Google Sheets and keep a running log of spend per truck.")

    vehicle_overview = load_vehicle_overview(conn)
    if vehicle_overview.empty:
        st.info("No vehicles found. Use the Fleet tab to add trucks or import VEHICLE_DETAILS data.")
    else:
        metadata_columns: Iterable[str] = (
            "truck_id",
            "rego",
            "rego_expiry",
            "insurance",
            "odometer",
            "last_service",
            "next_service",
            "present_driver",
            "daily_check_complete",
        )
        st.markdown("#### Vehicle register")
        vehicle_preview = vehicle_overview.loc[:, metadata_columns].copy()
        vehicle_preview = vehicle_preview.rename(
            columns={
                "truck_id": "Vehicle",
                "rego": "Rego",
                "rego_expiry": "Rego expiry",
                "insurance": "Insurance",
                "odometer": "Odometer",
                "last_service": "Last service",
                "next_service": "Next service",
                "present_driver": "Assigned driver",
                "daily_check_complete": "Daily check complete",
            }
        )
        vehicle_preview["Daily check complete"] = vehicle_preview["Daily check complete"].map(
            {1: "Yes", 0: "No"}
        )
        st.dataframe(vehicle_preview, width='stretch')

    default_sheet = os.environ.get("VEHICLE_REPAIRS_SHEET_URL") or os.environ.get(
        "VEHICLE_REPAIRS_SHEET"
    )

    with st.expander("Import VEHICLE_REPAIRS sheet", expanded=False):
        sheet_url = st.text_input(
            "Google Sheets CSV/Excel URL",
            value=default_sheet or "",
            help=(
                "Paste the CSV export link for the VEHICLE_REPAIRS sheet. "
                "Both CSV and XLSX feeds are supported."
            ),
            key="vehicle_repairs_sheet_url",
        )
        if st.button("Import vehicle repairs", key="vehicle_repairs_import_button"):
            if not sheet_url:
                st.error("Provide a Google Sheets URL before importing.")
            else:
                try:
                    inserted, updated = import_vehicle_repairs_from_sheet(
                        conn, sheet_url=sheet_url
                    )
                except Exception as exc:  # pragma: no cover - surfaced in UI only
                    st.error(f"Failed to import repairs: {exc}")
                else:
                    st.success(
                        f"Imported {inserted} new record{'s' if inserted != 1 else ''}. "
                        f"Updated {updated} existing row{'s' if updated != 1 else ''}."
                    )

    repairs_df = load_vehicle_repairs(conn)
    if repairs_df.empty:
        st.info(
            "No vehicle repairs captured yet. Import the VEHICLE_REPAIRS sheet to populate this view."
        )
        return

    display_df = repairs_df.copy()
    for column in ("service_date", "next_service_date", "created_at", "updated_at"):
        if column in display_df.columns:
            display_df[column] = pd.to_datetime(display_df[column], errors="coerce")

    display_df = display_df.sort_values(
        by=["service_date", "created_at"], ascending=[False, False], na_position="last"
    )

    st.markdown("#### Spend by vehicle")
    spend_summary = (
        display_df.assign(price_numeric=pd.to_numeric(display_df["price"], errors="coerce"))
        .groupby("truck_id")
        .agg(
            total_spend=pd.NamedAgg(column="price_numeric", aggfunc="sum"),
            jobs=pd.NamedAgg(column="job_item", aggfunc="count"),
            last_service=pd.NamedAgg(column="service_date", aggfunc="max"),
        )
        .reset_index()
        .sort_values("total_spend", ascending=False)
    )

    summary_cols = st.columns(max(1, min(3, len(spend_summary))))
    for idx, row in spend_summary.iterrows():
        column = summary_cols[idx % len(summary_cols)]
        column.metric(
            f"{row['truck_id']} spend",
            _format_currency(row["total_spend"]),
            help="Total repair spend captured for this vehicle.",
        )
        column.metric(
            f"{row['truck_id']} jobs",
            f"{int(row['jobs'])}",
            help="Count of repair log entries.",
        )
        if pd.notna(row["last_service"]):
            column.caption(f"Last service: {pd.to_datetime(row['last_service']).date()}")

    st.dataframe(
        spend_summary.rename(
            columns={
                "truck_id": "Vehicle",
                "total_spend": "Spend",
                "jobs": "Repairs logged",
                "last_service": "Most recent service",
            }
        ),
        width='stretch',
    )

    st.markdown("#### Repair log")
    if not vehicle_overview.empty:
        metadata_fields = [
            "rego",
            "insurance",
            "odometer",
            "last_service",
            "next_service",
            "present_driver",
            "daily_check_complete",
        ]
        merged = display_df.merge(
            vehicle_overview[["truck_id", *metadata_fields]],
            on="truck_id",
            how="left",
        )
        merged = merged.rename(
            columns={
                "rego": "Rego",
                "insurance": "Insurance",
                "odometer": "Odometer",
                "last_service": "Last service marker",
                "next_service": "Next service marker",
                "present_driver": "Assigned driver",
                "daily_check_complete": "Daily check complete",
            }
        )
        merged["Daily check complete"] = merged["Daily check complete"].map({1: "Yes", 0: "No"})
        st.dataframe(merged, width='stretch')
    else:
        st.dataframe(display_df, width='stretch')


def _parse_sheet_id(sheet_reference: str) -> str:
    sheet_reference = sheet_reference.strip()
    if "/d/" in sheet_reference:
        parts = sheet_reference.split("/d/")
        if len(parts) > 1:
            remainder = parts[1]
            return remainder.split("/")[0]
    return sheet_reference


def render_fleet_tab(conn) -> None:
    st.markdown("### Fleet register")
    st.caption("Manage trucks, vehicle metadata, and VEHICLE_DETAILS imports.")

    ensure_dashboard_tables(conn)
    ensure_lane_assignment_schema(conn)
    vehicle_df = load_vehicle_overview(conn)
    assignment_summary = list_truck_assignment_summary(conn)
    if not vehicle_df.empty:
        vehicle_df = vehicle_df.copy()
        vehicle_df["planned_segment_count"] = vehicle_df["truck_id"].map(
            lambda truck_id: assignment_summary.get(str(truck_id), {}).get("plannedSegmentCount", 0)
        )
        vehicle_df["planned_job_count"] = vehicle_df["truck_id"].map(
            lambda truck_id: assignment_summary.get(str(truck_id), {}).get("plannedJobCount", 0)
        )
        vehicle_df["next_planned_start"] = vehicle_df["truck_id"].map(
            lambda truck_id: assignment_summary.get(str(truck_id), {}).get("nextPlannedStart")
        )
        vehicle_df["planned_workers"] = vehicle_df["truck_id"].map(
            lambda truck_id: ", ".join(
                assignment_summary.get(str(truck_id), {}).get("plannedWorkers", [])
            )
        )

    with st.expander("Sync shared operations workbook", expanded=False):
        shared_reference = st.text_input(
            "Operations workbook ID or URL",
            value=(
                os.environ.get("OPERATIONS_WORKBOOK_URL")
                or os.environ.get("OPERATIONS_WORKBOOK_SHEET_ID")
                or ""
            ),
            help="Refresh FLEET, STAFF, and SUPPLIERS from the shared operations workbook.",
            key="operations_workbook_reference",
        )
        if st.button(
            "Sync operations workbook",
            key="operations_workbook_sync_button",
            disabled=not shared_reference.strip(),
        ):
            try:
                summary = sync_operations_workbook(
                    conn,
                    sheet_id_or_url=shared_reference.strip(),
                )
            except Exception as exc:  # pragma: no cover - UI feedback only
                st.error(f"Failed to sync operations workbook: {exc}")
            else:
                st.success(
                    "Synced operations workbook: "
                    f"{summary['fleetImported']} fleet rows, "
                    f"{summary['staffInserted']} staff inserted, "
                    f"{summary['staffUpdated']} staff updated, "
                    f"{summary['suppliersImported']} suppliers."
                )
                _rerun_app()

    with st.expander("Assignment readiness policy", expanded=False):
        policy = get_operations_policy(conn)
        policy_cols_1 = st.columns(4)
        rego_warning_days = int(
            policy_cols_1[0].number_input("Rego warning days", min_value=0, value=policy["regoWarningDays"])
        )
        coi_warning_days = int(
            policy_cols_1[1].number_input("COI warning days", min_value=0, value=policy["coiWarningDays"])
        )
        service_warning_days = int(
            policy_cols_1[2].number_input("Service warning days", min_value=0, value=policy["serviceWarningDays"])
        )
        compliance_warning_days = int(
            policy_cols_1[3].number_input("Compliance warning days", min_value=0, value=policy["complianceWarningDays"])
        )
        policy_cols_2 = st.columns(4)
        service_overdue_blocks = policy_cols_2[0].checkbox(
            "Service overdue blocks",
            value=policy["serviceOverdueBlocks"],
        )
        conflict_blocks = policy_cols_2[1].checkbox(
            "Conflicts block assignment",
            value=policy["conflictBlocks"],
        )
        service_override_allowed = policy_cols_2[2].checkbox(
            "Allow service override",
            value=policy["serviceOverrideAllowed"],
        )
        conflict_override_allowed = policy_cols_2[3].checkbox(
            "Allow conflict override",
            value=policy["conflictOverrideAllowed"],
        )
        if st.button("Save readiness policy", key="operations_policy_save_button"):
            update_operations_policy(
                conn,
                rego_warning_days=rego_warning_days,
                coi_warning_days=coi_warning_days,
                service_warning_days=service_warning_days,
                compliance_warning_days=compliance_warning_days,
                service_overdue_blocks=service_overdue_blocks,
                conflict_blocks=conflict_blocks,
                service_override_allowed=service_override_allowed,
                conflict_override_allowed=conflict_override_allowed,
            )
            st.success("Readiness policy updated.")
            _rerun_app()

    with st.expander("Role layout defaults", expanded=False):
        available_tabs = [
            "Quote",
            "Pricing Intelligence",
            "Network",
            "Operations",
            "Admin",
        ]
        role_layouts = get_dashboard_role_layouts(conn, available_tabs=available_tabs)
        role_options = {row["label"]: row for row in role_layouts}
        selected_role_label = st.selectbox(
            "Role",
            options=list(role_options.keys()),
            key="dashboard_role_layout_selected_role",
        )
        selected_role = role_options[selected_role_label]
        layout_cols = st.columns(3)
        default_landing = layout_cols[0].selectbox(
            "Default landing tab",
            options=available_tabs,
            index=available_tabs.index(selected_role["defaultLandingTab"]),
            key="dashboard_role_layout_default_landing",
        )
        primary_tabs = layout_cols[1].multiselect(
            "Primary tabs",
            options=available_tabs,
            default=selected_role["primaryTabs"],
            key="dashboard_role_layout_primary_tabs",
        )
        hidden_tabs = layout_cols[2].multiselect(
            "Hidden tabs",
            options=[tab for tab in available_tabs if tab != default_landing],
            default=[tab for tab in selected_role["hiddenTabs"] if tab != default_landing],
            key="dashboard_role_layout_hidden_tabs",
        )
        if st.button("Save role layout defaults", key="dashboard_role_layout_save_button"):
            try:
                upsert_dashboard_role_layout(
                    conn,
                    role_key=selected_role["roleKey"],
                    default_landing_tab=default_landing,
                    primary_tabs=primary_tabs,
                    hidden_tabs=hidden_tabs,
                    available_tabs=available_tabs,
                )
            except Exception as exc:  # pragma: no cover
                st.error(f"Failed to save role layout defaults: {exc}")
            else:
                st.success("Role layout defaults updated.")
                _rerun_app()

    with st.expander("Spreadsheet cutover admin", expanded=False):
        cutover_rows = list_operations_cutover_rollout(conn)
        if not cutover_rows:
            st.info("No cutover workflows configured yet.")
        else:
            workflow_options = {
                f"{row['label']} · {row['cutoverStatus']}": row for row in cutover_rows
            }
            selected_label = st.selectbox(
                "Workflow",
                options=list(workflow_options.keys()),
                key="operations_cutover_selected_workflow",
            )
            selected = workflow_options[selected_label]
            admin_cols = st.columns(4)
            cutover_status = admin_cols[0].selectbox(
                "Cutover status",
                options=["sheet_primary", "dual_run", "native_primary", "fallback_only"],
                index=["sheet_primary", "dual_run", "native_primary", "fallback_only"].index(
                    selected["cutoverStatus"]
                ),
                key="operations_cutover_status",
            )
            owner_role = admin_cols[1].text_input(
                "Owner role",
                value=selected.get("ownerRole") or "",
                key="operations_cutover_owner_role",
            )
            snapshot_mode = admin_cols[2].selectbox(
                "Snapshot mode",
                options=["none", "on_demand", "daily"],
                index=["none", "on_demand", "daily"].index(selected["snapshotMode"]),
                key="operations_cutover_snapshot_mode",
            )
            fallback_mode = admin_cols[3].selectbox(
                "Fallback mode",
                options=["import_only", "read_only_sheet", "manual_csv"],
                index=["import_only", "read_only_sheet", "manual_csv"].index(
                    selected["fallbackMode"]
                ),
                key="operations_cutover_fallback_mode",
            )
            metric_cols = st.columns(5)
            cutover_target_percent = metric_cols[0].number_input(
                "Target native usage %",
                min_value=0.0,
                max_value=100.0,
                value=float(selected["metrics"]["cutoverTargetPercent"]),
                key="operations_cutover_target_percent",
            )
            metric_cols[1].metric(
                "Current native usage %",
                f"{float(selected['metrics']['nativeUsagePercent']):.1f}",
            )
            metric_cols[2].metric(
                "Fallback uses",
                int(selected["metrics"]["fallbackUsageCount"]),
            )
            metric_cols[3].metric(
                "Open issues",
                int(selected["metrics"]["openIssueCount"]),
            )
            metric_cols[4].metric(
                "Snapshot consumers",
                int(selected["metrics"]["snapshotConsumerCount"]),
            )
            checklist_cols = st.columns(4)
            native_ready = checklist_cols[0].checkbox(
                "Native ready",
                value=selected["checklist"]["nativeReady"],
                key="operations_cutover_native_ready",
            )
            dual_run_complete = checklist_cols[1].checkbox(
                "Dual-run complete",
                value=selected["checklist"]["dualRunComplete"],
                key="operations_cutover_dual_run_complete",
            )
            fallback_drill_complete = checklist_cols[2].checkbox(
                "Fallback drill complete",
                value=selected["checklist"]["fallbackDrillComplete"],
                key="operations_cutover_fallback_drill_complete",
            )
            operator_trained = checklist_cols[3].checkbox(
                "Operator trained",
                value=selected["checklist"]["operatorTrained"],
                key="operations_cutover_operator_trained",
            )
            snapshot_fields = st.text_input(
                "Snapshot fields (comma-separated)",
                value=", ".join(selected.get("snapshotFields", [])),
                key="operations_cutover_snapshot_fields",
            )
            st.caption(
                "Last review: "
                + str(selected["metrics"].get("lastReviewAt") or "not recorded")
                + " | Last fallback drill: "
                + str(selected.get("lastDrillAt") or "not recorded")
            )
            recommendation = selected.get("recommendation", {})
            approval = selected.get("approval", {})
            st.info(
                "Recommended transition: "
                + str(recommendation.get("recommendedStatus") or selected["cutoverStatus"])
                + " | "
                + str(recommendation.get("reason") or "No recommendation")
            )
            approval_cols = st.columns(4)
            approval_cols[0].metric("Approval status", str(approval.get("status") or "not_required"))
            approval_cols[1].metric("Requested by", str(approval.get("requestedBy") or "-"))
            approval_cols[2].metric("Approved by", str(approval.get("approvedBy") or "-"))
            approval_cols[3].metric("Rejected by", str(approval.get("rejectedBy") or "-"))
            if approval.get("requestNote"):
                st.caption("Request note: " + str(approval["requestNote"]))
            if approval.get("approvalNote"):
                st.caption("Approval note: " + str(approval["approvalNote"]))
            if approval.get("rejectionNote"):
                st.caption("Rejection note: " + str(approval["rejectionNote"]))
            if recommendation.get("blockedByApproval"):
                st.warning(str(recommendation.get("reason") or "Approval chain is incomplete."))
            rollback_instructions = st.text_area(
                "Rollback instructions",
                value=selected.get("rollbackInstructions") or "",
                key="operations_cutover_rollback_instructions",
            )
            notes = st.text_area(
                "Notes",
                value=selected.get("notes") or "",
                key="operations_cutover_notes",
            )
            if st.button("Save cutover workflow", key="operations_cutover_save_button"):
                upsert_operations_cutover_workflow(
                    conn,
                    workflow_key=selected["workflowKey"],
                    cutover_status=cutover_status,
                    owner_role=owner_role.strip() or None,
                    snapshot_mode=snapshot_mode,
                    snapshot_fields=[field.strip() for field in snapshot_fields.split(",") if field.strip()],
                    fallback_mode=fallback_mode,
                    cutover_target_percent=float(cutover_target_percent),
                    native_ready=native_ready,
                    dual_run_complete=dual_run_complete,
                    fallback_drill_complete=fallback_drill_complete,
                    operator_trained=operator_trained,
                    rollback_instructions=rollback_instructions.strip() or None,
                    notes=notes.strip() or None,
                )
                st.success("Cutover workflow updated.")
                _rerun_app()
            promotion_actor = st.text_input(
                "Promotion actor",
                value="",
                help="Ops manager requests; commercial owner approves or rejects.",
                key="operations_cutover_promotion_actor",
            )
            promotion_note = st.text_input(
                "Promotion note",
                value="",
                key="operations_cutover_promotion_note",
            )
            promotion_cols = st.columns(4)
            if promotion_cols[0].button("Request promotion", key="operations_cutover_request_promotion_button"):
                try:
                    request_operations_cutover_promotion(
                        conn,
                        workflow_key=selected["workflowKey"],
                        actor=promotion_actor.strip(),
                        note=promotion_note.strip() or None,
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                        st.success("Promotion request recorded.")
                        _rerun_app()
            if promotion_cols[1].button("Approve promotion", key="operations_cutover_approve_promotion_button"):
                try:
                    approve_operations_cutover_promotion(
                        conn,
                        workflow_key=selected["workflowKey"],
                        actor=promotion_actor.strip(),
                        note=promotion_note.strip() or None,
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                        st.success("Promotion approval recorded.")
                        _rerun_app()
            if promotion_cols[2].button("Reject promotion", key="operations_cutover_reject_promotion_button"):
                try:
                    reject_operations_cutover_promotion(
                        conn,
                        workflow_key=selected["workflowKey"],
                        actor=promotion_actor.strip(),
                        note=promotion_note.strip(),
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                        st.success("Promotion rejection recorded.")
                        _rerun_app()
            if recommendation.get("actionable"):
                if promotion_cols[3].button(
                    "Apply recommended transition",
                    key="operations_cutover_apply_recommendation_button",
                ):
                    try:
                        apply_operations_cutover_recommendation(
                            conn,
                            workflow_key=selected["workflowKey"],
                            actor=promotion_actor.strip() or owner_role.strip() or None,
                            note="Applied from Fleet cutover admin.",
                        )
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        st.success("Recommended transition applied.")
                        _rerun_app()
            st.markdown("##### Cutover actions")
            action_actor = st.text_input(
                "Action actor",
                value="",
                help="Operator/manager identifier for review, drill, and fallback-use logs.",
                key="operations_cutover_action_actor",
            )
            action_note = st.text_input(
                "Action note",
                value="",
                key="operations_cutover_action_note",
            )
            snapshot_consumer = st.text_input(
                "Snapshot consumer/team",
                value="",
                key="operations_cutover_snapshot_consumer",
            )
            action_cols = st.columns(4)
            if action_cols[0].button("Record review", key="operations_cutover_record_review"):
                record_operations_cutover_event(
                    conn,
                    workflow_key=selected["workflowKey"],
                    event_type="review",
                    actor=action_actor.strip() or None,
                    note=action_note.strip() or None,
                )
                st.success("Review recorded.")
                _rerun_app()
            if action_cols[1].button("Record fallback drill", key="operations_cutover_record_drill"):
                record_operations_cutover_event(
                    conn,
                    workflow_key=selected["workflowKey"],
                    event_type="fallback_drill",
                    actor=action_actor.strip() or None,
                    note=action_note.strip() or None,
                )
                st.success("Fallback drill recorded.")
                _rerun_app()
            if action_cols[2].button("Record fallback use", key="operations_cutover_record_fallback"):
                record_operations_cutover_event(
                    conn,
                    workflow_key=selected["workflowKey"],
                    event_type="fallback_use",
                    actor=action_actor.strip() or None,
                    note=action_note.strip() or None,
                )
                st.success("Fallback use recorded.")
                _rerun_app()
            if action_cols[3].button("Record snapshot issued", key="operations_cutover_record_snapshot"):
                record_operations_cutover_event(
                    conn,
                    workflow_key=selected["workflowKey"],
                    event_type="snapshot_issued",
                    actor=action_actor.strip() or None,
                    note=action_note.strip() or None,
                    event_value=snapshot_consumer.strip() or None,
                )
                st.success("Snapshot issuance recorded.")
                _rerun_app()
            recent_events = list_operations_cutover_events(
                conn,
                workflow_key=selected["workflowKey"],
                limit=10,
            )
            if recent_events:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "When": row["createdAt"],
                                "Type": row["eventType"],
                                "Actor": row["actor"],
                                "Value": row["eventValue"],
                                "Note": row["note"],
                            }
                            for row in recent_events
                        ]
                    ),
                    width='stretch',
                    hide_index=True,
                )

    with st.expander("Historical ingest health", expanded=False):
        ingest_summary = latest_historical_ingest_summary(conn)
        if ingest_summary is None:
            st.info("No historical ingest runs recorded yet.")
        else:
            coverage = ingest_summary.get("coverage_summary") or {}
            ingest_cols = st.columns(5)
            ingest_cols[0].metric("Readiness", str(ingest_summary.get("readiness_status") or "unknown"))
            ingest_cols[1].metric("Total rows", int(ingest_summary.get("total_rows") or 0))
            ingest_cols[2].metric("Inserted", int(ingest_summary.get("inserted_rows") or 0))
            ingest_cols[3].metric("Skipped", int(ingest_summary.get("skipped_rows") or 0))
            ingest_cols[4].metric("Issues", int(ingest_summary.get("issue_count") or 0))
            st.caption(
                "Latest ingest source: "
                + str(ingest_summary.get("source_name") or "unknown")
                + " | completed: "
                + str(ingest_summary.get("completed_at") or "unknown")
            )
            st.caption(
                "Coverage ratio: "
                + f"{float(coverage.get('coverageRatio') or 0.0):.2f}"
                + " | inserted ratio: "
                + f"{float(coverage.get('insertedRatio') or 0.0):.2f}"
            )
            top_issues = coverage.get("topIssueCodes") or []
            if top_issues:
                st.dataframe(
                    pd.DataFrame(top_issues).rename(
                        columns={"issueCode": "Issue", "count": "Count"}
                    ),
                    width="stretch",
                    hide_index=True,
                )
            issues = ingest_summary.get("issues") or []
            if issues:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Row": issue["row_index"],
                                "Ref": issue["source_row_ref"],
                                "Severity": issue["severity"],
                                "Code": issue["issue_code"],
                                "Message": issue["message"],
                            }
                            for issue in issues[:20]
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )

    with st.expander("Lane assignment health", expanded=False):
        health_df = _lane_assignment_health_summary(conn)
        if health_df.empty:
            st.info("No lane-assignment data available yet.")
        else:
            summary_lookup = {
                (str(row["dataset"]), str(row["lane_assignment_status"])): int(row["row_count"])
                for _, row in health_df.iterrows()
            }
            metric_cols = st.columns(6)
            metric_cols[0].metric(
                "Historical assigned",
                summary_lookup.get(("historical", "assigned"), 0),
            )
            metric_cols[1].metric(
                "Historical ambiguous",
                summary_lookup.get(("historical", "ambiguous"), 0),
            )
            metric_cols[2].metric(
                "Historical unassigned",
                summary_lookup.get(("historical", "unassigned"), 0),
            )
            metric_cols[3].metric(
                "Live assigned",
                summary_lookup.get(("live", "assigned"), 0),
            )
            metric_cols[4].metric(
                "Live ambiguous",
                summary_lookup.get(("live", "ambiguous"), 0),
            )
            metric_cols[5].metric(
                "Live unassigned",
                summary_lookup.get(("live", "unassigned"), 0),
            )
            st.dataframe(
                health_df.rename(
                    columns={
                        "dataset": "Dataset",
                        "lane_assignment_status": "Status",
                        "row_count": "Rows",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

            gap_df = _recent_lane_assignment_gaps(conn, limit=25)
            if gap_df.empty:
                st.caption("No ambiguous or unassigned lane records.")
            else:
                grouped_gap_df = (
                    gap_df.assign(
                        proposed_lane_key=gap_df["origin_cluster_key"].fillna("").astype(str).str.strip()
                        + "->"
                        + gap_df["destination_cluster_key"].fillna("").astype(str).str.strip()
                    )
                    .groupby(
                        [
                            "dataset",
                            "lane_assignment_status",
                            "origin_cluster_key",
                            "destination_cluster_key",
                            "proposed_lane_key",
                        ],
                        dropna=False,
                    )
                    .agg(
                        candidate_count=pd.NamedAgg(column="row_id", aggfunc="count"),
                        sample_corridor=pd.NamedAgg(column="corridor_display", aggfunc="first"),
                        sample_source=pd.NamedAgg(column="lane_assignment_source", aggfunc="first"),
                        sample_note=pd.NamedAgg(column="lane_assignment_note", aggfunc="first"),
                        sample_row_id=pd.NamedAgg(column="row_id", aggfunc="first"),
                    )
                    .reset_index()
                    .sort_values(["candidate_count", "dataset", "proposed_lane_key"], ascending=[False, True, True])
                )
                st.dataframe(
                    grouped_gap_df.rename(
                        columns={
                            "dataset": "Dataset",
                            "lane_assignment_status": "Status",
                            "origin_cluster_key": "Origin cluster",
                            "destination_cluster_key": "Destination cluster",
                            "proposed_lane_key": "Proposed lane",
                            "candidate_count": "Rows",
                            "sample_corridor": "Example corridor",
                            "sample_source": "Example source",
                            "sample_note": "Example note",
                            "sample_row_id": "Representative row",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )
                st.dataframe(
                    gap_df.rename(
                        columns={
                            "dataset": "Dataset",
                            "row_id": "Row ID",
                            "reference": "Reference",
                            "corridor_display": "Corridor",
                            "lane_assignment_status": "Status",
                            "lane_assignment_source": "Source",
                            "lane_assignment_note": "Note",
                            "updated_at": "Updated",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )
                candidate_options = {
                    (
                        f"{row['dataset']} · {int(row['candidate_count'])} rows · "
                        f"{row['lane_assignment_status']} · "
                        f"{str(row['proposed_lane_key'])}"
                    ): row
                    for _, row in grouped_gap_df.iterrows()
                }
                proposal_cols = st.columns(3)
                selected_candidate_label = proposal_cols[0].selectbox(
                    "Promotion candidate",
                    options=list(candidate_options.keys()),
                    key="lane_promotion_candidate",
                )
                lane_actor = proposal_cols[1].text_input(
                    "Lane actor",
                    value="",
                    key="lane_promotion_actor",
                )
                lane_note = proposal_cols[2].text_input(
                    "Lane note",
                    value="",
                    key="lane_promotion_note",
                )
                if st.button("Create lane promotion proposal", key="lane_promotion_create"):
                    candidate = candidate_options[selected_candidate_label]
                    try:
                        create_lane_promotion_proposal(
                            conn,
                            dataset=str(candidate["dataset"]),
                            row_id=int(candidate["sample_row_id"]),
                            actor=lane_actor.strip(),
                            note=lane_note.strip() or None,
                        )
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        st.success("Lane promotion proposal created.")
                        _rerun_app()
                if st.button("Create grouped lane proposal", key="lane_promotion_create_grouped"):
                    candidate = candidate_options[selected_candidate_label]
                    try:
                        create_lane_promotion_proposal_for_clusters(
                            conn,
                            dataset=str(candidate["dataset"]),
                            origin_cluster_key=str(candidate["origin_cluster_key"]),
                            destination_cluster_key=str(candidate["destination_cluster_key"]),
                            actor=lane_actor.strip(),
                            note=lane_note.strip() or None,
                        )
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        st.success("Grouped lane promotion proposal created.")
                        _rerun_app()

            proposal_rows = list_lane_promotion_proposals(conn, limit=20)
            if proposal_rows:
                proposal_options = {
                    (
                        f"#{proposal['id']} · {proposal['status']} · "
                        f"{proposal['lane_display_name']}"
                    ): proposal
                    for proposal in proposal_rows
                }
                selected_proposal_label = st.selectbox(
                    "Lane proposal",
                    options=list(proposal_options.keys()),
                    key="lane_promotion_selected_proposal",
                )
                proposal = proposal_options[selected_proposal_label]
                proposal_meta = st.columns(4)
                proposal_meta[0].metric("Status", str(proposal["status"]))
                proposal_meta[1].metric("Requested by", str(proposal["requested_by"]))
                proposal_meta[2].metric("Approved by", str(proposal.get("approved_by") or "-"))
                proposal_meta[3].metric("Applied by", str(proposal.get("applied_by") or "-"))
                st.caption(
                    "Lane: "
                    + str(proposal["lane_key"])
                    + " | corridor: "
                    + str(proposal["corridor_group_key"])
                )
                if proposal.get("request_note"):
                    st.caption("Request note: " + str(proposal["request_note"]))
                source_summary = proposal.get("source_summary") or {}
                if source_summary:
                    st.dataframe(
                        pd.DataFrame([source_summary]).rename(
                            columns={
                                "rowId": "Row ID",
                                "corridorDisplay": "Corridor",
                                "laneAssignmentStatus": "Assignment status",
                                "laneAssignmentSource": "Assignment source",
                                "laneAssignmentNote": "Assignment note",
                            }
                        ),
                        width="stretch",
                        hide_index=True,
                    )
                review_cols = st.columns(3)
                review_actor = review_cols[0].text_input(
                    "Review actor",
                    value="",
                    key="lane_promotion_review_actor",
                )
                review_note = review_cols[1].text_input(
                    "Review note",
                    value="",
                    key="lane_promotion_review_note",
                )
                apply_note = review_cols[2].text_input(
                    "Apply note",
                    value="",
                    key="lane_promotion_apply_note",
                )
                action_cols = st.columns(3)
                if proposal["status"] == LANE_PROPOSAL_STATUS_PENDING_REVIEW:
                    if action_cols[0].button("Approve lane proposal", key="lane_promotion_approve"):
                        try:
                            approve_lane_promotion_proposal(
                                conn,
                                proposal_id=int(proposal["id"]),
                                actor=review_actor.strip(),
                                note=review_note.strip() or None,
                            )
                        except ValueError as exc:
                            st.error(str(exc))
                        else:
                            st.success("Lane promotion proposal approved.")
                            _rerun_app()
                    if action_cols[1].button("Reject lane proposal", key="lane_promotion_reject"):
                        try:
                            reject_lane_promotion_proposal(
                                conn,
                                proposal_id=int(proposal["id"]),
                                actor=review_actor.strip(),
                                note=review_note.strip(),
                            )
                        except ValueError as exc:
                            st.error(str(exc))
                        else:
                            st.success("Lane promotion proposal rejected.")
                            _rerun_app()
                if proposal["status"] == LANE_PROPOSAL_STATUS_APPROVED:
                    if action_cols[2].button("Apply lane proposal", key="lane_promotion_apply"):
                        try:
                            apply_lane_promotion_proposal(
                                conn,
                                proposal_id=int(proposal["id"]),
                                actor=review_actor.strip(),
                                note=apply_note.strip() or None,
                            )
                        except ValueError as exc:
                            st.error(str(exc))
                        else:
                            st.success("Lane promotion proposal applied.")
                            _rerun_app()

    with st.expander("Adaptive policy governance", expanded=False):
        snapshot = load_adaptive_policy_snapshot(conn)
        snapshot_cols = st.columns(4)
        snapshot_cols[0].metric("Lane ETA", f"{snapshot.lane_eta_multiplier:.3f}")
        snapshot_cols[1].metric("Weather risk", f"{snapshot.weather_risk_multiplier:.3f}")
        snapshot_cols[2].metric("Closure delay", f"{snapshot.closure_delay_factor:.3f}")
        snapshot_cols[3].metric("Seasonal uplift", f"{snapshot.seasonal_margin_uplift:.3f}")

        proposal_inputs = st.columns(4)
        proposal_actor = proposal_inputs[0].text_input(
            "Proposal actor",
            value="",
            key="adaptive_policy_proposal_actor",
        )
        lookback_hours = proposal_inputs[1].number_input(
            "Lookback hours",
            min_value=1,
            max_value=168,
            value=6,
            key="adaptive_policy_lookback_hours",
        )
        max_delta = proposal_inputs[2].number_input(
            "Max delta",
            min_value=0.01,
            max_value=1.0,
            value=0.1,
            step=0.01,
            key="adaptive_policy_max_delta",
        )
        proposal_note = proposal_inputs[3].text_input(
            "Proposal note",
            value="",
            key="adaptive_policy_proposal_note",
        )
        if st.button("Create disruption proposal", key="adaptive_policy_create_proposal"):
            try:
                update_adaptive_policy_from_disruptions(
                    conn,
                    actor=proposal_actor.strip(),
                    approval_mode="proposal",
                    lookback=pd.Timedelta(hours=int(lookback_hours)).to_pytimedelta(),
                    max_delta=float(max_delta),
                    note=proposal_note.strip() or None,
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.success("Adaptive-policy proposal created.")
                _rerun_app()

        proposal_rows = list_adaptive_policy_proposals(conn, limit=20)
        if not proposal_rows:
            st.info("No adaptive-policy proposals recorded yet.")
        else:
            proposal_options = {
                f"#{row['id']} · {row['proposal_type']} · {row['status']}": row
                for row in proposal_rows
            }
            selected_label = st.selectbox(
                "Proposal",
                options=list(proposal_options.keys()),
                key="adaptive_policy_selected_proposal",
            )
            selected = proposal_options[selected_label]
            governance_cols = st.columns(4)
            governance_cols[0].metric("Status", str(selected["status"]))
            governance_cols[1].metric("Requested by", str(selected.get("requested_by") or "-"))
            governance_cols[2].metric("Approved by", str(selected.get("approved_by") or "-"))
            governance_cols[3].metric("Applied by", str(selected.get("applied_by") or "-"))
            summary = selected.get("source_summary") or {}
            if summary:
                st.caption("Source summary: " + str(summary))
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Key": item["parameter_key"],
                            "Current": item["current_value"],
                            "Proposed": item["proposed_value"],
                            "Target": item["target_value"],
                            "Max delta": item["max_delta"],
                            "Description": item["description"],
                        }
                        for item in selected["items"]
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
            decision_actor = st.text_input(
                "Decision actor",
                value="",
                key="adaptive_policy_decision_actor",
            )
            decision_note = st.text_input(
                "Decision note",
                value="",
                key="adaptive_policy_decision_note",
            )
            decision_cols = st.columns(3)
            if decision_cols[0].button("Approve proposal", key="adaptive_policy_approve"):
                try:
                    approve_adaptive_policy_proposal(
                        conn,
                        proposal_id=int(selected["id"]),
                        actor=decision_actor.strip(),
                        note=decision_note.strip() or None,
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.success("Adaptive-policy proposal approved.")
                    _rerun_app()
            if decision_cols[1].button("Reject proposal", key="adaptive_policy_reject"):
                try:
                    reject_adaptive_policy_proposal(
                        conn,
                        proposal_id=int(selected["id"]),
                        actor=decision_actor.strip(),
                        note=decision_note.strip(),
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.success("Adaptive-policy proposal rejected.")
                    _rerun_app()
            if decision_cols[2].button("Apply approved proposal", key="adaptive_policy_apply"):
                try:
                    apply_adaptive_policy_proposal(
                        conn,
                        proposal_id=int(selected["id"]),
                        actor=decision_actor.strip(),
                        note=decision_note.strip() or None,
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.success("Adaptive-policy proposal applied.")
                    _rerun_app()

    readiness_items = list_operational_readiness_items(conn)
    blocked_items = [item for item in readiness_items if item["status"] == "blocked"]
    warning_items = [item for item in readiness_items if item["status"] == "warning"]

    st.markdown("#### Maintenance and compliance cockpit")
    cockpit_cols = st.columns(4)
    cockpit_cols[0].metric("Blocked items", len(blocked_items))
    cockpit_cols[1].metric("Due soon", len(warning_items))
    cockpit_cols[2].metric(
        "Blocked vehicles",
        len({item["resourceId"] for item in blocked_items if item["resourceType"] == "vehicle"}),
    )
    cockpit_cols[3].metric(
        "Workers due/blocked",
        len({item["resourceId"] for item in readiness_items if item["resourceType"] == "worker"}),
    )

    if readiness_items:
        readiness_df = pd.DataFrame(
            [
                {
                    "Status": item["status"],
                    "Type": item["resourceType"],
                    "Resource": item["resourceName"],
                    "Rule": item["ruleType"],
                    "Due": item["dueAt"],
                    "Overrideable": item["overrideable"],
                    "Details": item["details"],
                    "Imported": item["sourceImportedAt"],
                }
                for item in readiness_items
            ]
        )
        st.dataframe(readiness_df, width='stretch', hide_index=True)
    else:
        st.caption("No due-soon or blocked maintenance/compliance items detected.")

    with st.expander("Import/Export VEHICLE_DETAILS", expanded=False):
        sheet_input = st.text_input(
            "Google Sheets ID or URL",
            value=(
                os.environ.get("OPERATIONS_WORKBOOK_URL")
                or os.environ.get("OPERATIONS_WORKBOOK_SHEET_ID")
                or ""
            ),
            help="Paste the spreadsheet ID or full link for the VEHICLE_DETAILS workbook.",
        )
        upload = st.file_uploader(
            "Upload VEHICLE_DETAILS workbook (XLSX or CSV)",
            type=["xlsx", "xls", "csv"],
        )
        import_cols = st.columns(2)
        if import_cols[0].button("Import from Google Sheet"):
            try:
                sheet_id = _parse_sheet_id(sheet_input)
                imported = import_vehicle_details_from_google_sheet(conn, sheet_id=sheet_id)
            except Exception as exc:  # pragma: no cover - UI feedback only
                st.error(f"Failed to import VEHICLE_DETAILS: {exc}")
            else:
                st.success(f"Imported {imported} vehicle{'s' if imported != 1 else ''} from Google Sheets.")
                _rerun_app()

        if import_cols[1].button("Import uploaded workbook"):
            if not upload:
                st.error("Upload a workbook before importing.")
            else:
                try:
                    if upload.name.endswith(".csv"):
                        frame = pd.read_csv(upload)
                        imported = import_vehicle_details_from_dataframe(conn, frame)
                    else:
                        if hasattr(upload, "seek"):
                            upload.seek(0)
                        workbook = pd.ExcelFile(upload, engine="openpyxl")
                        imported = import_vehicle_details_from_workbook(conn, workbook)
                except Exception as exc:  # pragma: no cover - UI feedback only
                    st.error(f"Failed to import uploaded workbook: {exc}")
                else:
                    st.success(f"Imported {imported} vehicle{'s' if imported != 1 else ''} from the upload.")
                    _rerun_app()

        if not vehicle_df.empty:
            csv_data = vehicle_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download VEHICLE_DETAILS CSV",
                data=csv_data,
                file_name="vehicle_details.csv",
                mime="text/csv",
            )
        else:
            st.caption("Export will be available after vehicles are added.")

    filter_cols = st.columns(2)
    truck_filter = filter_cols[0].multiselect(
        "Filter by vehicle",
        options=sorted(vehicle_df["truck_id"].dropna().unique()) if not vehicle_df.empty else [],
    )
    active_filter = filter_cols[1].selectbox(
        "Active status",
        options=["All", "Active", "Inactive"],
        help="Quick filter to show only active or inactive trucks.",
    )

    filtered_df = vehicle_df
    if truck_filter:
        filtered_df = filtered_df[filtered_df["truck_id"].isin(truck_filter)]
    if active_filter != "All":
        desired = 1 if active_filter == "Active" else 0
        filtered_df = filtered_df[filtered_df["active"] == desired]

    st.markdown("#### Vehicles")
    if filtered_df.empty:
        st.info("No vehicles match the selected filters.")
    else:
        preview = filtered_df.copy()
        preview["active"] = preview["active"].map({1: "Yes", 0: "No"})
        st.dataframe(preview, width='stretch')

    st.markdown("#### Add or update vehicle")
    existing_ids = list(vehicle_df["truck_id"].dropna().unique()) if not vehicle_df.empty else []
    selection_label = "Select vehicle" if existing_ids else "New vehicle"
    selection_options = ["New vehicle", *existing_ids]
    selected_vehicle = st.selectbox(selection_label, options=selection_options)

    defaults: dict[str, object] = {}
    if selected_vehicle != "New vehicle" and not vehicle_df.empty:
        defaults = (
            vehicle_df.loc[vehicle_df["truck_id"] == selected_vehicle]
            .iloc[0]
            .to_dict()
        )

    with st.form("vehicle_editor"):
        truck_id_value = st.text_input("Truck ID (rego)", value=str(defaults.get("truck_id", "")))
        name_value = st.text_input("Name/label", value=str(defaults.get("name", "") or ""))
        capacity_value = st.number_input(
            "Capacity (m³)",
            min_value=0.0,
            value=float(defaults.get("capacity_m3")) if defaults.get("capacity_m3") is not None else 0.0,
            step=1.0,
        )
        active_value = st.checkbox("Active", value=bool(defaults.get("active", True)))
        notes_value = st.text_area("Notes", value=str(defaults.get("notes", "") or ""))

        st.markdown("##### Vehicle details")
        detail_cols1 = st.columns(3)
        state_value = detail_cols1[0].text_input("State", value=str(defaults.get("state", "") or ""))
        rego_expiry_default = pd.to_datetime(
            defaults.get("rego_expiry"), errors="coerce"
        )
        rego_expiry_value = detail_cols1[1].date_input(
            "Rego expiry",
            value=rego_expiry_default.date() if pd.notna(rego_expiry_default) else None,
        )
        insurance_value = detail_cols1[2].text_input("Insurance", value=str(defaults.get("insurance", "") or ""))

        detail_cols2 = st.columns(3)
        make_value = detail_cols2[0].text_input("Make", value=str(defaults.get("make", "") or ""))
        model_value = detail_cols2[1].text_input("Model", value=str(defaults.get("model", "") or ""))
        year_value = detail_cols2[2].number_input(
            "Year", min_value=0, max_value=9999, value=int(defaults.get("year")) if defaults.get("year") else 0, step=1
        )

        body_type_value = st.text_input("Body type", value=str(defaults.get("body_type", "") or ""))
        description_value = st.text_area("Description", value=str(defaults.get("description", "") or ""))
        nhv_code_value = st.text_input("NHV code", value=str(defaults.get("nhv_code", "") or ""))
        odometer_value = st.number_input(
            "Odometer", min_value=0, value=int(defaults.get("odometer")) if defaults.get("odometer") else 0, step=100
        )
        detail_cols3 = st.columns(2)
        last_service_default = pd.to_datetime(defaults.get("last_service"), errors="coerce")
        next_service_default = pd.to_datetime(defaults.get("next_service"), errors="coerce")
        last_service_value = detail_cols3[0].date_input(
            "Last service",
            value=last_service_default.date() if pd.notna(last_service_default) else None,
        )
        next_service_value = detail_cols3[1].date_input(
            "Next service",
            value=next_service_default.date() if pd.notna(next_service_default) else None,
        )
        detail_cols4 = st.columns(2)
        coi_number_value = detail_cols4[0].text_input("COI number", value=str(defaults.get("coi_number", "") or ""))
        coi_due_default = pd.to_datetime(defaults.get("coi_due"), errors="coerce")
        coi_due_value = detail_cols4[1].date_input(
            "COI due", value=coi_due_default.date() if pd.notna(coi_due_default) else None
        )
        present_driver_value = st.text_input("Assigned driver", value=str(defaults.get("present_driver", "") or ""))
        daily_check_value = st.checkbox(
            "Daily check complete", value=bool(defaults.get("daily_check_complete", False))
        )

        submitted = st.form_submit_button("Save vehicle")

    if submitted:
        if not truck_id_value.strip():
            st.error("Truck ID/rego is required.")
        else:
            upsert_truck(
                conn,
                truck_id=truck_id_value.strip(),
                name=name_value or None,
                capacity_m3=capacity_value if capacity_value else None,
                active=active_value,
                notes=notes_value or None,
            )
            upsert_vehicle_details(
                conn,
                truck_id=truck_id_value.strip(),
                state=state_value or None,
                rego=truck_id_value.strip(),
                rego_expiry=rego_expiry_value.isoformat() if rego_expiry_value else None,
                make=make_value or None,
                model=model_value or None,
                year=int(year_value) if year_value else None,
                body_type=body_type_value or None,
                description=description_value or None,
                nhv_code=nhv_code_value or None,
                insurance=insurance_value or None,
                odometer=int(odometer_value) if odometer_value else None,
                last_service=last_service_value.isoformat() if last_service_value else None,
                next_service=next_service_value.isoformat() if next_service_value else None,
                coi_number=coi_number_value or None,
                coi_due=coi_due_value.isoformat() if coi_due_value else None,
                present_driver=present_driver_value or None,
                daily_check_complete=daily_check_value,
            )
            st.success("Vehicle saved.")
            _rerun_app()

    if selected_vehicle != "New vehicle" and selected_vehicle in existing_ids:
        planned_segments = list_segments_for_truck(conn, truck_id=selected_vehicle)
        st.markdown("#### Planned segment assignments")
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
                        "Workers": ", ".join(
                            assignment["workerName"]
                            for assignment in row["workerAssignments"]
                            if assignment.get("workerName")
                        ),
                    }
                    for row in planned_segments
                ]
            )
            st.dataframe(planned_df, width='stretch', hide_index=True)
        else:
            st.caption("No planned job segments currently assign this vehicle.")

        if st.button("Delete vehicle", type="secondary"):
            conn.execute("DELETE FROM trucks WHERE truck_id = ?", (selected_vehicle,))
            conn.commit()
            st.success(f"Deleted vehicle {selected_vehicle}.")
            _rerun_app()
