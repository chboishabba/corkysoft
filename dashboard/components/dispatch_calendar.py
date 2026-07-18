from __future__ import annotations

import sqlite3
from datetime import date, datetime, time, timedelta

import pandas as pd
import streamlit as st

from corkysoft.operations_platform import JOB_STATES, list_dispatch_calendar


def _calendar_window(mode: str, anchor: date) -> tuple[datetime, datetime]:
    start = datetime.combine(anchor, time.min)
    if mode == "Week":
        start -= timedelta(days=start.weekday())
        return start, start + timedelta(days=7)
    return start, start + timedelta(days=1)


def render_dispatch_calendar(conn: sqlite3.Connection) -> None:
    st.subheader("Dispatch calendar")
    st.caption("Read-first daily/weekly schedule over persisted jobs, segments, assignments and readiness signals.")

    controls = st.columns([1, 1, 2])
    mode = controls[0].radio("View", options=["Day", "Week"], horizontal=True, key="dispatch_calendar_mode")
    anchor = controls[1].date_input("Date", value=date.today(), key="dispatch_calendar_date")
    statuses = controls[2].multiselect(
        "Job states", options=list(JOB_STATES), default=[state for state in JOB_STATES if state != "completed"], key="dispatch_calendar_states"
    )

    truck_rows = conn.execute("SELECT truck_id FROM trucks WHERE active = 1 ORDER BY truck_id").fetchall()
    worker_rows = conn.execute("SELECT id, name FROM workers WHERE active = 1 ORDER BY name").fetchall()
    filters = st.columns(3)
    truck_ids = filters[0].multiselect(
        "Truck", options=[str(row[0]) for row in truck_rows], key="dispatch_calendar_trucks"
    )
    worker_map = {f"{row[1]} · {row[0]}": int(row[0]) for row in worker_rows}
    selected_workers = filters[1].multiselect(
        "Worker", options=list(worker_map), key="dispatch_calendar_workers"
    )
    depot = filters[2].text_input("Depot/origin contains", value="", key="dispatch_calendar_depot")

    start, end = _calendar_window(mode, anchor)
    rows = list_dispatch_calendar(
        conn,
        start=start.isoformat(),
        end=end.isoformat(),
        statuses=statuses,
        truck_ids=truck_ids,
        worker_ids=[worker_map[label] for label in selected_workers],
        depot=depot.strip() or None,
    )
    if not rows:
        st.info("No scheduled segments match this calendar window and filter set.")
        return

    frame = pd.DataFrame(
        [
            {
                "Start": row["planned_start"],
                "End": row["planned_end"],
                "Job": row["job_number"] or row["job_id"],
                "Client": row["client"],
                "Route": f"{row['origin'] or ''} → {row['destination'] or ''}",
                "State": row["job_status"],
                "Trucks": ", ".join(row["truckIds"]),
                "Crew": ", ".join(row["workerNames"]),
                "Readiness": "ready" if row["ready"] else "attention",
                "Blocks": ", ".join(row["blockingFlags"]),
                "Warnings": ", ".join(row["warningFlags"]),
                "Job ID": row["job_id"],
                "Segment ID": row["segment_id"],
            }
            for row in rows
        ]
    )
    st.dataframe(frame, width="stretch", hide_index=True)

    job_options = {
        f"{row['job_number'] or row['job_id']} · {row['client']} · {row['planned_start']}": row
        for row in rows
    }
    selected_label = st.selectbox("Open scheduled job", options=list(job_options), key="dispatch_calendar_job")
    selected = job_options[selected_label]
    metric_cols = st.columns(4)
    metric_cols[0].metric("State", selected["job_status"])
    metric_cols[1].metric("Segment", selected["segment_sequence"])
    metric_cols[2].metric("Trucks", len(selected["truckIds"]))
    metric_cols[3].metric("Crew", len(selected["workerIds"]))
    st.caption(
        f"Job {selected['job_id']} / segment {selected['segment_id']} · "
        f"{selected['origin']} → {selected['destination']}"
    )


__all__ = ["render_dispatch_calendar"]
