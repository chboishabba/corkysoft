from __future__ import annotations

import sqlite3

import pandas as pd
import streamlit as st

from corkysoft.operations_platform import (
    JOB_STATE_TRANSITIONS,
    list_worker_assignments,
    record_crew_acknowledgement,
    record_crew_job_status,
)


def render_crew_workflow(conn: sqlite3.Connection) -> None:
    st.subheader("Crew workflow")
    st.caption("Lightweight role-scoped mobile-web workflow for assignments, acknowledgement, status and exceptions.")

    workers = conn.execute(
        "SELECT id, name, role FROM workers WHERE active = 1 ORDER BY name"
    ).fetchall()
    if not workers:
        st.info("No active workers are available.")
        return

    worker_options = {f"{row[1]} · {row[2] or 'worker'} · {row[0]}": int(row[0]) for row in workers}
    worker_label = st.selectbox("Worker", options=list(worker_options), key="crew_worker")
    worker_id = worker_options[worker_label]
    assignments = list_worker_assignments(conn, worker_id=worker_id)
    if not assignments:
        st.info("No active assignments for this worker.")
        return

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Job": row["job_number"] or row["job_id"],
                    "Client": row["client"],
                    "Route": f"{row['origin'] or ''} → {row['destination'] or ''}",
                    "State": row["job_status"],
                    "Start": row["planned_start"],
                    "End": row["planned_end"],
                    "Truck": ", ".join(row["truckIds"]),
                    "Instructions": row["internal_notes"] or "",
                }
                for row in assignments
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    options = {
        f"{row['job_number'] or row['job_id']} · {row['client']} · {row['job_status']}": row
        for row in assignments
    }
    selected_label = st.selectbox("Assigned job", options=list(options), key="crew_job")
    selected = options[selected_label]
    job_id = int(selected["job_id"])
    segment_id = int(selected["segment_id"])

    st.markdown(f"**{selected['origin']} → {selected['destination']}**")
    st.caption(selected["internal_notes"] or "No additional job instructions recorded.")
    note = st.text_area("Crew note", value="", key="crew_note")

    actions = st.columns(3)
    if actions[0].button("Acknowledge assignment", type="primary", key="crew_ack"):
        record_crew_acknowledgement(
            conn,
            job_id=job_id,
            segment_id=segment_id,
            worker_id=worker_id,
            note=note.strip() or None,
        )
        st.success("Assignment acknowledged.")
        st.rerun()

    current_state = str(selected["job_status"])
    next_states = sorted(JOB_STATE_TRANSITIONS.get(current_state, set()))
    next_operational = [state for state in next_states if state != "exception"]
    if next_operational:
        chosen_state = actions[1].selectbox(
            "Next state", options=next_operational, key="crew_next_state"
        )
        if actions[1].button("Record status", key="crew_status"):
            record_crew_job_status(
                conn,
                job_id=job_id,
                worker_id=worker_id,
                new_state=chosen_state,
                note=note.strip() or None,
            )
            st.success(f"Job moved to {chosen_state.replace('_', ' ')}.")
            st.rerun()
    else:
        actions[1].caption("No normal next state available.")

    if "exception" in next_states and actions[2].button("Flag exception", key="crew_exception"):
        record_crew_job_status(
            conn,
            job_id=job_id,
            worker_id=worker_id,
            new_state="exception",
            note=note.strip() or "Crew exception flagged.",
        )
        st.error("Exception recorded for dispatcher review.")
        st.rerun()


__all__ = ["render_crew_workflow"]
