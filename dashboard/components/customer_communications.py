from __future__ import annotations

import json
import sqlite3

import pandas as pd
import streamlit as st

from corkysoft.operations_platform import (
    CUSTOMER_COMMUNICATION_EVENTS,
    propose_customer_communication,
)


def render_customer_communications(conn: sqlite3.Connection) -> None:
    st.subheader("Customer communication events")
    st.caption(
        "Create public-safe proposed messages before connecting SMS/email providers. "
        "Nothing is sent from this surface."
    )

    jobs = conn.execute(
        "SELECT id, job_number, client, origin, destination, status, planned_start FROM jobs ORDER BY id DESC LIMIT 200"
    ).fetchall()
    if not jobs:
        st.info("Create an operational job before proposing customer communications.")
        return

    job_options = {
        f"{row[1] or row[0]} · {row[2] or 'Unassigned'} · {row[5]}": row
        for row in jobs
    }
    selected_label = st.selectbox("Job", options=list(job_options), key="customer_comm_job")
    selected = job_options[selected_label]
    event_type = st.selectbox(
        "Event", options=list(CUSTOMER_COMMUNICATION_EVENTS), key="customer_comm_event"
    )
    cols = st.columns(3)
    recipient = cols[0].text_input("Recipient", value="", key="customer_comm_recipient")
    channel = cols[1].selectbox(
        "Intended channel", options=["internal", "email", "sms", "portal"], key="customer_comm_channel"
    )
    proposed_by = cols[2].text_input("Proposed by", value="dispatcher", key="customer_comm_actor")
    message = st.text_area(
        "Public-safe message",
        value=f"Update for job {selected[1] or selected[0]}: status {selected[5]}.",
        key="customer_comm_message",
    )
    payload = {
        "customer_name": selected[2] or "Customer",
        "job_number": selected[1] or str(selected[0]),
        "origin": selected[3] or "",
        "destination": selected[4] or "",
        "planned_start": selected[6],
        "status": selected[5],
        "message": message,
    }
    st.code(json.dumps(payload, indent=2, default=str), language="json")
    if st.button("Persist proposed message", type="primary", key="customer_comm_submit"):
        event_id = propose_customer_communication(
            conn,
            event_type=event_type,
            proposed_by=proposed_by,
            public_payload=payload,
            job_id=int(selected[0]),
            recipient=recipient.strip() or None,
            channel=channel,
            template_key=event_type,
        )
        st.success(f"Proposed customer communication event {event_id} saved. No message was sent.")

    rows = conn.execute(
        """
        SELECT id, event_type, channel, status, recipient, proposed_by, created_at
        FROM customer_communication_events
        WHERE job_id = ?
        ORDER BY id DESC
        LIMIT 25
        """,
        (int(selected[0]),),
    ).fetchall()
    if rows:
        st.dataframe(
            pd.DataFrame(
                rows,
                columns=["ID", "Event", "Channel", "Status", "Recipient", "Proposed by", "Created"],
            ),
            width="stretch",
            hide_index=True,
        )


__all__ = ["render_customer_communications"]
