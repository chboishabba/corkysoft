from __future__ import annotations

import sqlite3

import pandas as pd
import streamlit as st

from analytics.operations_assignment import (
    list_job_operations_board,
    list_operations_cutover_events,
    list_operations_cutover_rollout,
    record_operations_cutover_event,
)


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

    status_filter = st.multiselect(
        "Job status",
        options=["blocked", "override_required", "overridden", "planned", "draft"],
        default=["blocked", "override_required", "overridden", "planned", "draft"],
        key="dispatch_job_status_filter",
    )
    query = st.text_input("Search client / route / worker / truck", value="", key="dispatch_search")

    filtered_rows = board_rows
    if status_filter:
        filtered_rows = [row for row in filtered_rows if row["jobStatus"] in status_filter]
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
                "Client": row["jobClient"],
                "Origin": row["jobOrigin"],
                "Destination": row["jobDestination"],
                "Status": row["jobStatus"],
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
                "Execution stages": ", ".join(row.get("executionStages", [])),
                "Pending substitutions": row.get("pendingSubstitutionCount", 0),
                "Planned start": row["plannedStart"],
                "Planned end": row["plannedEnd"],
            }
            for row in filtered_rows
        ]
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

    metric_cols = st.columns(4)
    metric_cols[0].metric("Segments", selected["segmentCount"])
    metric_cols[1].metric("Warnings", selected["warningCount"])
    metric_cols[2].metric("Blocks", selected["blockingCount"])
    metric_cols[3].metric("Override flags", selected["overrideableCount"])

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
                "Architectures": ", ".join(row["architectures"]),
                "Execution stages": ", ".join(row.get("executionStages", [])),
                "Pending substitutions": row.get("pendingSubstitutionCount", 0),
            }
            for row in selected["segments"]
        ]
    )
    st.dataframe(segment_df, width='stretch', hide_index=True)
