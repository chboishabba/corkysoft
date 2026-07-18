from __future__ import annotations

import sqlite3
from datetime import date, datetime, time

import streamlit as st

from corkysoft.operations_platform import accept_quote_as_job, ensure_operations_platform_schema


def render_quote_acceptance_panel(conn: sqlite3.Connection) -> None:
    """Turn a saved quote into planned operational work without re-entry."""

    ensure_operations_platform_schema(conn)
    rows = conn.execute(
        """
        SELECT id, quote_date, client_display, origin_input, destination_input,
               cubic_m, final_quote, manual_quote, status, accepted_job_id
        FROM quotes
        ORDER BY id DESC
        LIMIT 100
        """
    ).fetchall()
    if not rows:
        st.info("Save a quote before accepting it as operational work.")
        return

    options = {
        f"Quote {row[0]} · {row[2] or 'Quote builder'} · {row[3]} → {row[4]}": row
        for row in rows
    }
    selected_label = st.selectbox("Saved quote", options=list(options), key="accepted_work_quote")
    selected = options[selected_label]
    accepted_job_id = selected[9]
    if accepted_job_id is not None:
        st.success(f"This quote is already accepted as job {accepted_job_id}.")
        return

    cols = st.columns(3)
    planned_date = cols[0].date_input(
        "Planned date",
        value=date.fromisoformat(str(selected[1])[:10]) if selected[1] else date.today(),
        key="accepted_work_date",
    )
    start_time = cols[1].time_input("Start", value=time(8, 0), key="accepted_work_start")
    duration_hours = cols[2].number_input(
        "Planned hours", min_value=0.5, max_value=48.0, value=8.0, step=0.5, key="accepted_work_hours"
    )
    actor = st.text_input("Accepting operator", value="dispatcher", key="accepted_work_actor")
    if st.button("Accept quote and create job", type="primary", key="accepted_work_submit"):
        start = datetime.combine(planned_date, start_time)
        end = start.timestamp() + float(duration_hours) * 3600
        result = accept_quote_as_job(
            conn,
            quote_id=int(selected[0]),
            actor=actor,
            planned_start=start.isoformat(),
            planned_end=datetime.fromtimestamp(end).isoformat(),
        )
        st.success(
            f"Created job {result['jobId']} with segment {result.get('segmentId')} and "
            f"{result.get('requirementCount', 0)} copied requirement(s)."
        )
        st.rerun()


__all__ = ["render_quote_acceptance_panel"]
