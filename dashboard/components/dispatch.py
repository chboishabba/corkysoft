from __future__ import annotations

import sqlite3

import pandas as pd
import streamlit as st

from analytics.operations_assignment import (
    DISPATCH_SHARE_ACTION_STATUSES,
    list_job_operations_board,
    list_dispatch_share_actions,
    list_operations_cutover_events,
    list_operations_cutover_rollout,
    list_operational_share_opportunities,
    record_dispatch_share_action,
    record_operations_cutover_event,
)
from dashboard.query_params import _set_workspace_query_params
from dashboard.state import _rerun_app


def render_dispatch_tab(conn: sqlite3.Connection) -> None:
    st.subheader("Dispatch board")
    st.caption(
        "Run daily operations from jobs and segments directly. This is the native operational interface; sheet imports are fallback inputs."
    )

    cutover_rows = list_operations_cutover_rollout(conn)
    if cutover_rows:
        st.markdown("#### Cutover status")
        native_primary = sum(1 for row in cutover_rows if row["cutoverStatus"] == "native_primary")
        fallback_drilled = sum(1 for row in cutover_rows if row["checklist"]["fallbackDrillComplete"])
        target_met = sum(1 for row in cutover_rows if row["targetMet"])
        cutover_cols = st.columns(3)
        cutover_cols[0].metric("Native-primary workflows", native_primary)
        cutover_cols[1].metric("Fallback drills complete", fallback_drilled)
        cutover_cols[2].metric("Target-met workflows", target_met)
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Workflow": row["label"],
                        "Status": row["cutoverStatus"],
                        "Approval": row["approval"]["status"],
                        "Recommended": row["recommendation"]["recommendedStatus"],
                        "Owner": row["ownerRole"],
                        "Native surface": row["nativeSurface"],
                        "Spreadsheet source": row["spreadsheetSource"],
                        "Native usage %": row["metrics"]["nativeUsagePercent"],
                        "Target %": row["metrics"]["cutoverTargetPercent"],
                        "Open issues": row["metrics"]["openIssueCount"],
                        "Fallback uses": row["metrics"]["fallbackUsageCount"],
                        "Snapshot consumers": row["metrics"]["snapshotConsumerCount"],
                        "Target met": row["targetMet"],
                        "Snapshot mode": row["snapshotMode"],
                        "Fallback mode": row["fallbackMode"],
                        "All checks complete": row["allChecksComplete"],
                        "Last review": row["metrics"]["lastReviewAt"],
                        "Last drill": row["lastDrillAt"],
                    }
                    for row in cutover_rows
                ]
            ),
            width='stretch',
            hide_index=True,
        )
        dispatch_cutover = next(
            (row for row in cutover_rows if row["workflowKey"] == "dispatch_execution"),
            None,
        )
    else:
        dispatch_cutover = None

    board_rows = list_job_operations_board(conn)
    if not board_rows:
        st.info("No planned jobs/segments available yet. Use Operations to create and assign segments.")
        return
    opportunity_rows = list_operational_share_opportunities(conn)
    if opportunity_rows:
        st.markdown("#### Share / reallocation recommendations")
        st.caption(
            "These are operator recommendations based on spare-capacity and container-pressure signals, not automatic reassignments."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Job": row["jobId"],
                        "Job #": row.get("jobNumber") or "",
                        "Client": row.get("jobClient") or "",
                        "Route": " -> ".join(
                            part
                            for part in [row.get("jobOrigin") or "", row.get("jobDestination") or ""]
                            if part
                        ),
                        "Opportunity": row["opportunityType"],
                        "Utilisation": row["utilizationState"],
                        "Response": row["utilizationResponse"],
                        "Signal": row.get("spareCapacityLabel") or "untracked",
                        "Matching spare": row.get("matchingSpareTrucks", 0),
                        "Destination spare": row.get("destinationSpareTrucks", 0),
                        "Container shortage": row.get("containerShortageQuantity", 0.0),
                        "Shortage": row.get("shortageQuantity", 0.0),
                        "Latest action": row.get("latestActionType") or "",
                        "Action status": row.get("latestActionStatus") or "",
                        "Recommended action": row["recommendedAction"],
                    }
                    for row in opportunity_rows
                ]
            ),
            width="stretch",
            hide_index=True,
        )

    status_filter = st.multiselect(
        "Job status",
        options=["blocked", "override_required", "overridden", "planned", "draft"],
        default=["blocked", "override_required", "overridden", "planned", "draft"],
        key="dispatch_job_status_filter",
    )
    filter_cols = st.columns([2, 2])
    spare_capacity_filter = filter_cols[0].multiselect(
        "Spare-capacity signal",
        options=["favorable", "workable", "constrained", "untracked"],
        default=["favorable", "workable", "constrained", "untracked"],
        key="dispatch_spare_capacity_filter",
    )
    query = filter_cols[1].text_input(
        "Search client / route / worker / truck",
        value="",
        key="dispatch_search",
    )

    filtered_rows = board_rows
    if status_filter:
        filtered_rows = [row for row in filtered_rows if row["jobStatus"] in status_filter]
    if spare_capacity_filter:
        allowed_signals = set(spare_capacity_filter)
        filtered_rows = [
            row
            for row in filtered_rows
            if (row.get("spareCapacityLabel") or "untracked") in allowed_signals
        ]
    if query.strip():
        needle = query.strip().lower()
        filtered_rows = [
            row
            for row in filtered_rows
            if needle in " ".join(
                [
                    str(row.get("jobClient") or ""),
                    str(row.get("jobOrigin") or ""),
                    str(row.get("jobDestination") or ""),
                    " ".join(row.get("workerNames", [])),
                    " ".join(row.get("truckIds", [])),
                    " ".join(row.get("inventoryNames", [])),
                    " ".join(row.get("supplierNames", [])),
                ]
            ).lower()
        ]

    board_df = pd.DataFrame(
        [
            {
                "Job": row["jobId"],
                "Job #": row.get("jobNumber") or "",
                "Client": row["jobClient"],
                "Origin": row["jobOrigin"],
                "Destination": row["jobDestination"],
                "Status": row["jobStatus"],
                "Spare capacity": row.get("spareCapacityLabel") or "untracked",
                "Matching spare": row.get("matchingSpareTrucks", 0),
                "Destination spare": row.get("destinationSpareTrucks", 0),
                "Segments": row["segmentCount"],
                "Warnings": row["warningCount"],
                "Blocks": row["blockingCount"],
                "Override flags": row["overrideableCount"],
                "Trucks": ", ".join(row["truckIds"]),
                "Workers": ", ".join(row["workerNames"]),
                "Inventory": ", ".join(row["inventoryNames"]),
                "Suppliers": ", ".join(row["supplierNames"]),
                "Required qty": row["requiredQuantity"],
                "Allocated qty": row["allocatedQuantity"],
                "Approved substitution qty": row.get("approvedSubstitutionQuantity", 0.0),
                "Shortage qty": row["shortageQuantity"],
                "Shortages": row["inventoryShortageCount"],
                "Container reqs": row.get("containerRequirementCount", 0),
                "Container shortage": row.get("containerShortageQuantity", 0.0),
                "Execution stages": ", ".join(row.get("executionStages", [])),
                "Pending substitutions": row.get("pendingSubstitutionCount", 0),
                "Planned start": row["plannedStart"],
                "Planned end": row["plannedEnd"],
            }
            for row in filtered_rows
        ]
    )
    board_metric_cols = st.columns(4)
    board_metric_cols[0].metric(
        "Backhaul / spare-capacity opportunities",
        sum(
            1
            for row in filtered_rows
            if row.get("spareCapacityLabel") in {"favorable", "workable"}
        ),
    )
    board_metric_cols[1].metric(
        "Container-heavy jobs",
        sum(1 for row in filtered_rows if int(row.get("containerRequirementCount") or 0) > 0),
    )
    board_metric_cols[2].metric(
        "Container shortages",
        sum(1 for row in filtered_rows if float(row.get("containerShortageQuantity") or 0.0) > 0),
    )
    board_metric_cols[3].metric(
        "Untracked operational signals",
        sum(1 for row in filtered_rows if row.get("spareCapacityLabel") is None),
    )
    st.dataframe(board_df, width='stretch', hide_index=True)
    snapshot_recipient = st.text_input(
        "Snapshot recipient/team",
        value="",
        help="Optional label used when logging exported dispatch snapshots.",
        key="dispatch_snapshot_recipient",
    )
    snapshot_actor = st.text_input(
        "Snapshot operator",
        value="",
        help="Optional operator identifier for dispatch snapshot exports.",
        key="dispatch_snapshot_actor",
    )
    downloaded = st.download_button(
        "Download dispatch snapshot CSV",
        data=board_df.to_csv(index=False).encode("utf-8"),
        file_name="dispatch-board-snapshot.csv",
        mime="text/csv",
        key="dispatch_snapshot_download",
    )
    if downloaded and dispatch_cutover is not None:
        record_operations_cutover_event(
            conn,
            workflow_key="dispatch_execution",
            event_type="snapshot_issued",
            actor=snapshot_actor.strip() or None,
            note="Dispatch snapshot exported from native board.",
            event_value=snapshot_recipient.strip() or None,
        )
        st.success("Dispatch snapshot export logged.")

    if dispatch_cutover is not None:
        recent_events = list_operations_cutover_events(
            conn,
            workflow_key="dispatch_execution",
            limit=10,
        )
        if recent_events:
            st.markdown("#### Recent dispatch cutover events")
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

    if not filtered_rows:
        st.info("No dispatch jobs match the current filters. Adjust the filters to inspect a job.")
        return

    options = {
        f"Job {row['jobId']} · {row.get('jobClient') or 'Unassigned client'} · {row['jobStatus']}": row
        for row in filtered_rows
    }
    selected_label = st.selectbox(
        "Inspect job",
        options=list(options.keys()),
        key="dispatch_selected_job",
    )
    selected = options[selected_label]
    selected_opportunity = next(
        (row for row in opportunity_rows if int(row["jobId"]) == int(selected["jobId"])),
        None,
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Segments", selected["segmentCount"])
    metric_cols[1].metric("Warnings", selected["warningCount"])
    metric_cols[2].metric("Blocks", selected["blockingCount"])
    metric_cols[3].metric("Override flags", selected["overrideableCount"])
    fit_cols = st.columns(4)
    fit_cols[0].metric("Spare-capacity signal", selected.get("spareCapacityLabel") or "untracked")
    fit_cols[1].metric("Matching spare trucks", selected.get("matchingSpareTrucks", 0))
    fit_cols[2].metric("Destination spare trucks", selected.get("destinationSpareTrucks", 0))
    fit_cols[3].metric("Container shortage qty", selected.get("containerShortageQuantity", 0.0))
    if selected.get("spareCapacityScore") is not None:
        st.caption(
            "Operational fit is based on persisted route spare-capacity signals from ingest/planning. "
            f"Score {selected['spareCapacityScore']:.1f} from "
            f"{selected.get('operationalSignalSource') or 'unknown'} at "
            f"{selected.get('operationalSignalComputedAt') or 'unknown time'}."
        )
    elif int(selected.get("containerRequirementCount") or 0) > 0:
        st.caption(
            "This job is container-heavy, but no persisted spare-capacity signal is available yet."
        )
    if selected_opportunity is not None:
        st.markdown("#### Share / utilisation response")
        st.caption(selected_opportunity["recommendedAction"])
        response_cols = st.columns(4)
        response_cols[0].metric("Opportunity", selected_opportunity["opportunityType"])
        response_cols[1].metric("Utilisation response", selected_opportunity["utilizationResponse"])
        response_cols[2].metric("Latest action", selected_opportunity.get("latestActionType") or "none")
        response_cols[3].metric("Action status", selected_opportunity.get("latestActionStatus") or "none")
        with st.form(f"dispatch_share_action_{selected['jobId']}"):
            form_cols = st.columns(4)
            action_type = form_cols[0].selectbox(
                "Action",
                options=selected_opportunity["operatorActions"],
                index=0,
                format_func=lambda value: value.replace("_", " ").title(),
            )
            action_status = form_cols[1].selectbox(
                "Status",
                options=list(DISPATCH_SHARE_ACTION_STATUSES),
                index=0,
                format_func=lambda value: value.replace("_", " ").title(),
            )
            actor = form_cols[2].text_input("Actor")
            note = form_cols[3].text_input("Note")
            if st.form_submit_button("Record dispatch response", disabled=not actor.strip()):
                try:
                    record_dispatch_share_action(
                        conn,
                        job_id=int(selected["jobId"]),
                        opportunity_type=str(selected_opportunity["opportunityType"]),
                        utilization_state=str(selected_opportunity["utilizationState"]),
                        action_type=str(action_type),
                        action_status=str(action_status),
                        actor=actor,
                        note=note,
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.success("Dispatch response recorded.")
                    _rerun()
        recent_actions = list_dispatch_share_actions(conn, job_id=int(selected["jobId"]), limit=10)
        if recent_actions:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "When": row["createdAt"],
                            "Action": row["actionType"],
                            "Status": row["actionStatus"],
                            "Actor": row["actor"] or "",
                            "Opportunity": row["opportunityType"],
                            "Utilisation": row["utilizationState"],
                            "Note": row["note"] or "",
                        }
                        for row in recent_actions
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
    if st.button("Open in Operations diary", key="dispatch_open_operations_diary"):
        diary_date = str(selected.get("plannedStart") or "")[:10]
        _set_workspace_query_params(
            available_tabs=["Quote", "Pricing Intelligence", "Network", "Operations", "Admin"],
            view="Operations",
            workflow="operations_diary",
            diary_view="day",
            diary_date=diary_date or "",
            diary_job=str(selected["jobId"]),
        )
        _rerun_app()

    st.markdown("#### Segment detail")
    segment_df = pd.DataFrame(
        [
            {
                "Segment": row["segmentSequence"],
                "From": row["fromLocation"],
                "To": row["toLocation"],
                "Planned start": row["plannedStart"],
                "Planned end": row["plannedEnd"],
                "Status": row["assignmentStatus"],
                "Warnings": row["warningCount"],
                "Blocks": row["blockingCount"],
                "Override flags": row["overrideableCount"],
                "Trucks": ", ".join(row["truckIds"]),
                "Workers": ", ".join(row["workerNames"]),
                "Inventory": ", ".join(row["inventoryNames"]),
                "Suppliers": ", ".join(row["supplierNames"]),
                "Shipments": row["shipmentCount"],
                "Required qty": row["requiredQuantity"],
                "Allocated qty": row["allocatedQuantity"],
                "Approved substitution qty": row.get("approvedSubstitutionQuantity", 0.0),
                "Shortage qty": row["shortageQuantity"],
                "Container reqs": row.get("containerRequirementCount", 0),
                "Container shortage qty": row.get("containerShortageQuantity", 0.0),
                "Architectures": ", ".join(row["architectures"]),
                "Execution stages": ", ".join(row.get("executionStages", [])),
                "Pending substitutions": row.get("pendingSubstitutionCount", 0),
            }
            for row in selected["segments"]
        ]
    )
    st.dataframe(segment_df, width='stretch', hide_index=True)
