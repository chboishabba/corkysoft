from __future__ import annotations

import sqlite3

import pandas as pd
import streamlit as st

from analytics.operations_assignment import (
    assign_segment_resources,
    ensure_segment,
    get_operations_policy,
    list_operational_conflicts,
    list_segment_readiness,
)


def render_operations_tab(conn: sqlite3.Connection) -> None:
    st.subheader("Operations planning")
    st.caption(
        "Plan trucks and workers at the job-segment level. Corkysoft is the internal planning truth; spreadsheets are import inputs."
    )

    policy = get_operations_policy(conn)
    summary_cols = st.columns(4)
    summary_cols[0].metric("Rego warning", f"{policy['regoWarningDays']}d")
    summary_cols[1].metric("COI warning", f"{policy['coiWarningDays']}d")
    summary_cols[2].metric("Service warning", f"{policy['serviceWarningDays']}d")
    summary_cols[3].metric("Compliance warning", f"{policy['complianceWarningDays']}d")

    segments = list_segment_readiness(conn)
    filter_cols = st.columns(2)
    status_filter = filter_cols[0].selectbox(
        "Assignment status",
        options=["all", "draft", "planned", "blocked", "override_required", "overridden"],
        index=0,
        key="operations_assignment_status_filter",
    )
    job_filter_raw = filter_cols[1].text_input(
        "Job ID filter",
        value="",
        key="operations_job_id_filter",
    )

    if job_filter_raw.strip():
        try:
            target_job_id = int(job_filter_raw)
            segments = [row for row in segments if row["jobId"] == target_job_id]
        except ValueError:
            st.warning("Job ID filter must be numeric.")
    if status_filter != "all":
        segments = [row for row in segments if row["assignmentStatus"] == status_filter]

    if segments:
        overview = pd.DataFrame(
            [
                {
                    "Job": row["jobId"],
                    "Segment": row["segmentSequence"],
                    "From": row["fromLocation"] or row["jobOrigin"],
                    "To": row["toLocation"] or row["jobDestination"],
                    "Planned start": row["plannedStart"],
                    "Planned end": row["plannedEnd"],
                    "Status": row["assignmentStatus"],
                    "Warnings": len(row["warningFlags"]),
                    "Blocks": len(row["blockingFlags"]),
                    "Override flags": len(row["overrideableFlags"]),
                }
                for row in segments
            ]
        )
        st.dataframe(overview, width='stretch', hide_index=True)
    else:
        st.info("No job segments found for the current filters.")

    with st.expander("Advanced/manual segment editor", expanded=False):
        st.caption("Use this only for advanced/manual correction or edge cases. Normal planning should start in Planner.")
        create_cols = st.columns(3)
        job_id = int(create_cols[0].number_input("Job ID", min_value=1, step=1, value=1))
        segment_sequence = int(
            create_cols[1].number_input("Segment sequence", min_value=1, step=1, value=1)
        )
        from_location = create_cols[2].text_input("From", value="")
        create_cols_2 = st.columns(3)
        to_location = create_cols_2[0].text_input("To", value="")
        planned_start = create_cols_2[1].text_input(
            "Planned start (ISO)",
            value="",
            placeholder="2026-03-12T08:00:00+10:00",
        )
        planned_end = create_cols_2[2].text_input(
            "Planned end (ISO)",
            value="",
            placeholder="2026-03-12T12:00:00+10:00",
        )
        if st.button("Save segment", key="operations_save_segment_button"):
            try:
                ensure_segment(
                    conn,
                    job_id=job_id,
                    segment_sequence=segment_sequence,
                    from_location=from_location or None,
                    to_location=to_location or None,
                    planned_start=planned_start or None,
                    planned_end=planned_end or None,
                )
            except Exception as exc:  # pragma: no cover
                st.error(f"Failed to save segment: {exc}")
            else:
                st.success("Segment saved.")
                st.rerun()

    st.divider()
    st.markdown("### Assign resources")
    all_segments = list_segment_readiness(conn)
    if not all_segments:
        st.caption("Create a segment before assigning resources.")
        return

    segment_options = {
        f"Job {row['jobId']} / Segment {row['segmentSequence']} ({row['assignmentStatus']})": row
        for row in all_segments
    }
    selected_label = st.selectbox(
        "Target segment",
        options=list(segment_options.keys()),
        key="operations_target_segment",
    )
    selected_segment = segment_options[selected_label]

    truck_rows = conn.execute("SELECT truck_id, name FROM trucks WHERE active = 1 ORDER BY truck_id").fetchall()
    truck_labels = {
        f"{row['truck_id']} ({row['name'] or 'unnamed'})": row["truck_id"]
        for row in truck_rows
    }
    worker_rows = conn.execute("SELECT id, name FROM workers WHERE active = 1 ORDER BY name").fetchall()
    worker_labels = {f"{row['name']} (#{row['id']})": int(row["id"]) for row in worker_rows}
    role_rows = conn.execute("SELECT id, name FROM worker_roles ORDER BY name").fetchall()
    role_options = {"<none>": None, **{row["name"]: int(row["id"]) for row in role_rows}}
    compliance_rows = conn.execute("SELECT id, name FROM worker_compliances ORDER BY name").fetchall()
    compliance_options = {row["name"]: int(row["id"]) for row in compliance_rows}

    selected_truck_labels = st.multiselect(
        "Trucks",
        options=list(truck_labels.keys()),
        default=[
            next(
                (
                    label
                    for label, truck_id in truck_labels.items()
                    if truck_id == assignment["truckId"]
                ),
                None,
            )
            for assignment in selected_segment["truckAssignments"]
        ],
        key="operations_selected_trucks",
    )
    selected_worker_labels = st.multiselect(
        "Workers",
        options=list(worker_labels.keys()),
        default=[
            next(
                (
                    label
                    for label, worker_id in worker_labels.items()
                    if worker_id == assignment["workerId"]
                ),
                None,
            )
            for assignment in selected_segment["workerAssignments"]
        ],
        key="operations_selected_workers",
    )
    assignment_cols = st.columns(4)
    selected_role_name = assignment_cols[0].selectbox(
        "Required role",
        options=list(role_options.keys()),
        key="operations_selected_role",
    )
    selected_compliance_names = assignment_cols[1].multiselect(
        "Required compliances",
        options=list(compliance_options.keys()),
        key="operations_selected_compliances",
    )
    override = assignment_cols[2].checkbox("Override policy", value=False, key="operations_override")
    override_reason = assignment_cols[3].text_input(
        "Override reason code",
        value="manual_ops_override",
        disabled=not override,
        key="operations_override_reason",
    )
    override_note = st.text_area(
        "Override note",
        value="",
        disabled=not override,
        key="operations_override_note",
        height=80,
    )

    if st.button("Assign resources", type="primary", key="operations_assign_button"):
        try:
            readiness = assign_segment_resources(
                conn,
                segment_id=selected_segment["segmentId"],
                truck_ids=[truck_labels[label] for label in selected_truck_labels],
                worker_assignments=[
                    {
                        "workerId": worker_labels[label],
                        "roleId": role_options[selected_role_name],
                        "requiredComplianceIds": [
                            compliance_options[name] for name in selected_compliance_names
                        ],
                    }
                    for label in selected_worker_labels
                ],
                override=override,
                override_reason_code=override_reason or None,
                override_note=override_note or None,
            )
        except Exception as exc:  # pragma: no cover
            st.error(str(exc))
        else:
            st.success(f"Assignments saved. Segment status: {readiness['assignmentStatus']}.")
            st.rerun()

    st.divider()
    st.markdown("### Operational conflicts")
    conflicts = list_operational_conflicts(conn)
    if conflicts:
        st.dataframe(pd.DataFrame(conflicts), width='stretch', hide_index=True)
    else:
        st.caption("No active truck/worker conflicts detected.")
