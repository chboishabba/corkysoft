from __future__ import annotations

import sqlite3
from typing import Any

import pandas as pd
import streamlit as st

from dashboard.state import _rerun_app

from corkysoft.call_ops import (
    AMBIENT_SESSION_STATUSES,
    CALL_DIRECTIONS,
    CALL_EVENT_KINDS,
    CALL_LEG_KINDS,
    CALL_SOURCE_CHANNELS,
    CALL_STATUSES,
    WORKER_TIME_CHANNELS,
    WORKER_TIME_EVENT_TYPES,
    add_call_note,
    add_extracted_action,
    create_ambient_session,
    create_call_event,
    create_call_leg,
    create_call_session,
    decide_extracted_action,
    decide_worker_time_capture_event,
    generate_fake_ambient_transcript_artifact,
    generate_fake_transcript_artifact,
    get_call_session,
    list_ambient_sessions,
    list_ambient_transcript_artifacts,
    list_call_events,
    list_call_legs,
    list_call_notes,
    list_call_routing_events,
    list_call_sessions,
    list_extracted_actions,
    list_state_egress_events,
    list_transcript_artifacts,
    list_worker_time_capture_events,
    poll_transcript_artifact,
    record_transcript_artifact,
    record_worker_time_capture_event,
    resolve_call_links,
    submit_call_audio_for_transcription,
)


def render_calls_tab(conn: sqlite3.Connection) -> None:
    st.subheader("Calls and transcripts")
    st.caption(
        "Operational call console for routed call sessions, ambient transcript sessions, accepted actions, and worker time capture review. Raw transcripts are advisory until notes or actions are accepted."
    )

    with st.expander("Start call session", expanded=False):
        cols = st.columns(5)
        event_kind = cols[0].selectbox("Event kind", options=list(CALL_EVENT_KINDS), key="calls_event_kind")
        direction = cols[1].selectbox("Direction", options=list(CALL_DIRECTIONS), key="calls_direction")
        status = cols[2].selectbox("Status", options=list(CALL_STATUSES), index=list(CALL_STATUSES).index("completed"), key="calls_status")
        source_channel = cols[3].selectbox("Source channel", options=list(CALL_SOURCE_CHANNELS), key="calls_source_channel")
        destination_kind = cols[4].text_input("Initial destination kind", value="operator", key="calls_destination_kind")
        detail_cols = st.columns(5)
        title = detail_cols[0].text_input("Title", key="calls_title")
        caller_phone = detail_cols[1].text_input("Caller phone", key="calls_caller_phone")
        callee_phone = detail_cols[2].text_input("Callee phone", key="calls_callee_phone")
        operator_id = detail_cols[3].text_input("Operator id", key="calls_operator_id")
        destination_label = detail_cols[4].text_input("Initial destination label", key="calls_destination_label")
        link_cols = st.columns(4)
        job_id_raw = link_cols[0].text_input("Job id (optional)", key="calls_job_id")
        segment_id_raw = link_cols[1].text_input("Segment id (optional)", key="calls_segment_id")
        worker_id_raw = link_cols[2].text_input("Worker id (optional)", key="calls_worker_id")
        quote_id_raw = link_cols[3].text_input("Quote id (optional)", key="calls_quote_id")
        if st.button("Create call session", type="primary", key="calls_create_button"):
            try:
                create_call_session(
                    conn,
                    event_kind=event_kind,
                    direction=direction,
                    status=status,
                    source_channel=source_channel,
                    title=title or None,
                    caller_phone=caller_phone or None,
                    callee_phone=callee_phone or None,
                    operator_id=operator_id or None,
                    job_id=_int_or_none(job_id_raw),
                    segment_id=_int_or_none(segment_id_raw),
                    worker_id=_int_or_none(worker_id_raw),
                    quote_id=_int_or_none(quote_id_raw),
                    initial_destination_kind=destination_kind or None,
                    initial_destination_label=destination_label or None,
                )
            except Exception as exc:  # pragma: no cover
                st.error(f"Failed to create call session: {exc}")
            else:
                st.success("Call session created.")
                _rerun_app()

    session_rows = list_call_sessions(conn, limit=100)
    if not session_rows:
        st.info("No call sessions recorded yet.")
    else:
        st.dataframe(
            pd.DataFrame(session_rows)[
                [
                    "id",
                    "rootCallEventId",
                    "eventKind",
                    "direction",
                    "status",
                    "sourceChannel",
                    "callerPhone",
                    "clientName",
                    "jobId",
                    "workerName",
                    "legCount",
                    "pendingActionCount",
                    "createdAt",
                ]
            ].rename(
                columns={
                    "id": "Session",
                    "rootCallEventId": "Root event",
                    "eventKind": "Kind",
                    "direction": "Direction",
                    "status": "Status",
                    "sourceChannel": "Channel",
                    "callerPhone": "Caller",
                    "clientName": "Client",
                    "jobId": "Job",
                    "workerName": "Worker",
                    "legCount": "Legs",
                    "pendingActionCount": "Pending actions",
                    "createdAt": "Created",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        call_option_map = {
            f"#{row['id']} · {row.get('title') or row['eventKind']} · {row.get('callerPhone') or 'no caller'}": row
            for row in session_rows
        }
        selected_label = st.selectbox("Active call session", options=list(call_option_map.keys()), key="calls_selected_event")
        selected = call_option_map[selected_label]
        root_event_id = int(selected["rootCallEventId"]) if selected.get("rootCallEventId") is not None else None

        summary_cols = st.columns(4)
        summary_cols[0].metric("Client", selected.get("clientName") or "Unlinked")
        summary_cols[1].metric("Job", selected.get("jobId") or "Unlinked")
        summary_cols[2].metric("Worker", selected.get("workerName") or "Unlinked")
        summary_cols[3].metric("Pending actions", int(selected.get("pendingActionCount") or 0))

        with st.expander("Session legs and routing", expanded=True):
            leg_rows = list_call_legs(conn, call_session_id=int(selected["id"]))
            route_rows = list_call_routing_events(conn, call_session_id=int(selected["id"]))
            if leg_rows:
                st.dataframe(
                    pd.DataFrame(leg_rows)[["id", "legKind", "status", "destinationKind", "destinationLabel", "operatorId", "answeredAt", "latestTranscriptStatus"]].rename(columns={"id": "Leg"}),
                    width="stretch",
                    hide_index=True,
                )
                leg_map = {
                    f"#{row['id']} · {row['legKind']} · {row.get('destinationLabel') or row.get('destinationKind') or 'destination'}": row
                    for row in leg_rows
                }
                selected_leg_label = st.selectbox("Active leg", options=list(leg_map.keys()), key="calls_selected_leg")
                selected_leg = leg_map[selected_leg_label]
            else:
                st.caption("No legs recorded yet.")
                selected_leg = None
            if route_rows:
                st.dataframe(
                    pd.DataFrame(route_rows)[["eventType", "fromDestination", "toDestination", "actor", "createdAt"]],
                    width="stretch",
                    hide_index=True,
                )
            new_leg_cols = st.columns(5)
            leg_kind = new_leg_cols[0].selectbox("Leg kind", options=list(CALL_LEG_KINDS), key="calls_leg_kind")
            leg_status = new_leg_cols[1].selectbox("Leg status", options=list(CALL_STATUSES), key="calls_leg_status")
            leg_destination_kind = new_leg_cols[2].text_input("Destination kind", key="calls_leg_destination_kind")
            leg_destination_label = new_leg_cols[3].text_input("Destination label", key="calls_leg_destination_label")
            leg_operator_id = new_leg_cols[4].text_input("Leg operator", key="calls_leg_operator_id")
            if st.button("Add call leg", key="calls_add_leg"):
                try:
                    create_call_leg(
                        conn,
                        call_session_id=int(selected["id"]),
                        leg_kind=leg_kind,
                        direction=selected["direction"],
                        status=leg_status,
                        source_channel=selected["sourceChannel"],
                        destination_kind=leg_destination_kind or None,
                        destination_label=leg_destination_label or None,
                        operator_id=leg_operator_id or None,
                        caller_phone=selected.get("callerPhone"),
                        callee_phone=selected.get("calleePhone"),
                    )
                except Exception as exc:  # pragma: no cover
                    st.error(f"Failed to add leg: {exc}")
                else:
                    st.success("Call leg added.")
                    _rerun_app()

        with st.expander("Resolve links", expanded=False):
            link_cols = st.columns(5)
            client_id_raw = link_cols[0].text_input("Client id", value=str(selected.get("clientId") or ""), key="calls_resolve_client")
            quote_id_raw = link_cols[1].text_input("Quote id", value=str(selected.get("quoteId") or ""), key="calls_resolve_quote")
            job_id_raw = link_cols[2].text_input("Job id", value=str(selected.get("jobId") or ""), key="calls_resolve_job")
            segment_id_raw = link_cols[3].text_input("Segment id", value=str(selected.get("segmentId") or ""), key="calls_resolve_segment")
            worker_id_raw = link_cols[4].text_input("Worker id", value=str(selected.get("workerId") or ""), key="calls_resolve_worker")
            resolution_actor = st.text_input("Actor", key="calls_resolve_actor")
            resolution_note = st.text_input("Resolution note", key="calls_resolve_note")
            if st.button("Apply link resolution", key="calls_resolve_button"):
                try:
                    resolve_call_links(
                        conn,
                        call_event_id=root_event_id,
                        actor=resolution_actor or None,
                        client_id=_int_or_none(client_id_raw),
                        quote_id=_int_or_none(quote_id_raw),
                        job_id=_int_or_none(job_id_raw),
                        segment_id=_int_or_none(segment_id_raw),
                        worker_id=_int_or_none(worker_id_raw),
                        resolution_note=resolution_note or None,
                    )
                except Exception as exc:  # pragma: no cover
                    st.error(f"Failed to resolve call links: {exc}")
                else:
                    st.success("Call links updated.")
                    _rerun_app()

        transcript_rows = (
            list_transcript_artifacts(conn, call_leg_id=int(selected_leg["id"]))
            if selected_leg is not None
            else list_transcript_artifacts(conn, call_session_id=int(selected["id"]))
        )
        with st.expander("Transcript artifacts", expanded=True):
            upload_cols = st.columns(4)
            service_key = upload_cols[0].selectbox("Transcriber", options=["ops", "worker_time"], key="calls_transcriber")
            language = upload_cols[1].text_input("Language (optional)", key="calls_lang")
            diarize = upload_cols[2].checkbox("Diarize", value=True, key="calls_diarize")
            manual_status = upload_cols[3].selectbox("Manual artifact status", options=["queued", "completed", "failed"], key="calls_manual_status")
            uploaded = st.file_uploader("Audio for WhisperX upload", type=["wav", "mp3", "m4a", "ogg", "mp4"], key="calls_audio_file")
            transcript_text = st.text_area("Manual transcript text", key="calls_manual_transcript")
            fake_cols = st.columns(2)
            fake_scenario = fake_cols[0].text_input("Fake transcript scenario", key="calls_fake_scenario")
            fake_goal = fake_cols[1].text_input("Fake transcript desired outcome", key="calls_fake_goal")
            if st.button("Submit audio to WhisperX", key="calls_submit_audio", disabled=uploaded is None):
                if uploaded is None:
                    st.error("Choose an audio file first.")
                else:
                    try:
                        submit_call_audio_for_transcription(
                            conn,
                            call_leg_id=int(selected_leg["id"]) if selected_leg is not None else None,
                            call_event_id=root_event_id if selected_leg is None else None,
                            service_key=service_key,
                            file_bytes=uploaded.getvalue(),
                            filename=uploaded.name,
                            language=language or None,
                            diarize=bool(diarize),
                        )
                    except Exception as exc:  # pragma: no cover
                        st.error(f"Failed to submit audio: {exc}")
                    else:
                        st.success("Audio submitted to WhisperX.")
                        _rerun_app()
            if st.button("Add manual transcript artifact", key="calls_manual_artifact"):
                try:
                    record_transcript_artifact(
                        conn,
                        call_leg_id=int(selected_leg["id"]) if selected_leg is not None else None,
                        call_event_id=root_event_id if selected_leg is None else None,
                        service_key=service_key,
                        status=manual_status,
                        transcript_text=transcript_text or None,
                        is_final=manual_status == "completed",
                    )
                except Exception as exc:  # pragma: no cover
                    st.error(f"Failed to add transcript artifact: {exc}")
                else:
                    st.success("Transcript artifact added.")
                    _rerun_app()
            if st.button("Generate fake transcript", key="calls_fake_transcript"):
                try:
                    generate_fake_transcript_artifact(
                        conn,
                        call_leg_id=int(selected_leg["id"]) if selected_leg is not None else None,
                        call_event_id=root_event_id if selected_leg is None else None,
                        scenario=fake_scenario or None,
                        operator_goal=fake_goal or None,
                        service_key=service_key,
                    )
                except Exception as exc:  # pragma: no cover
                    st.error(f"Failed to generate fake transcript: {exc}")
                else:
                    st.success("Fake transcript generated.")
                    _rerun_app()
            if transcript_rows:
                st.dataframe(
                    pd.DataFrame(transcript_rows)[
                        ["id", "serviceKey", "externalTaskId", "status", "confidence", "isFinal", "createdAt"]
                    ].rename(
                        columns={
                            "id": "Artifact",
                            "serviceKey": "Service",
                            "externalTaskId": "Task",
                            "status": "Status",
                            "confidence": "Confidence",
                            "isFinal": "Final",
                            "createdAt": "Created",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )
                pollable = [row for row in transcript_rows if row.get("externalTaskId") and row.get("status") in {"queued", "in_progress"}]
                if pollable:
                    poll_options = {f"#{row['id']} · {row['status']} · {row.get('externalTaskId')}": row for row in pollable}
                    poll_label = st.selectbox("Poll artifact", options=list(poll_options.keys()), key="calls_poll_artifact")
                    if st.button("Poll WhisperX task", key="calls_poll_button"):
                        try:
                            poll_transcript_artifact(conn, artifact_id=int(poll_options[poll_label]["id"]))
                        except Exception as exc:  # pragma: no cover
                            st.error(f"Failed to poll transcript task: {exc}")
                        else:
                            st.success("Transcript task polled.")
                            _rerun_app()
                latest_text = next((row.get("transcriptText") for row in transcript_rows if row.get("transcriptText")), None)
                if latest_text:
                    st.text_area("Latest transcript", value=str(latest_text), height=180, key="calls_latest_transcript", disabled=True)
            else:
                st.caption("No transcript artifacts recorded yet.")

        with st.expander("Operator notes", expanded=True):
            note_author = st.text_input("Author", key="calls_note_author")
            note_text = st.text_area("Note", key="calls_note_text")
            if st.button("Add operator note", key="calls_add_note"):
                try:
                    add_call_note(
                        conn,
                        call_event_id=root_event_id,
                        author=note_author or None,
                        note_text=note_text,
                    )
                except Exception as exc:  # pragma: no cover
                    st.error(f"Failed to add call note: {exc}")
                else:
                    st.success("Call note added.")
                    _rerun_app()
            notes = list_call_notes(conn, call_event_id=root_event_id)
            if notes:
                st.dataframe(
                    pd.DataFrame(notes)[["author", "noteKind", "noteText", "authoritative", "createdAt"]],
                    width="stretch",
                    hide_index=True,
                )

        with st.expander("Extracted actions", expanded=True):
            action_text = st.text_input("Action text", key="calls_action_text")
            action_engine = st.text_input("Source engine", value="statibaker", key="calls_action_engine")
            if st.button("Add extracted action", key="calls_add_action"):
                try:
                    add_extracted_action(
                        conn,
                        call_event_id=root_event_id,
                        action_text=action_text,
                        source_engine=action_engine or None,
                    )
                except Exception as exc:  # pragma: no cover
                    st.error(f"Failed to add extracted action: {exc}")
                else:
                    st.success("Extracted action added.")
                    _rerun_app()
            actions = list_extracted_actions(conn, call_event_id=root_event_id)
            if actions:
                st.dataframe(
                    pd.DataFrame(actions)[["id", "actionText", "status", "decidedBy", "createdAt", "decidedAt"]].rename(
                        columns={"id": "Action"}
                    ),
                    width="stretch",
                    hide_index=True,
                )
                pending = [row for row in actions if row["status"] == "pending"]
                if pending:
                    pending_options = {f"#{row['id']} · {row['actionText']}": row for row in pending}
                    pending_label = st.selectbox("Pending action", options=list(pending_options.keys()), key="calls_pending_action")
                    decision_cols = st.columns(3)
                    decision = decision_cols[0].selectbox("Decision", options=["accepted", "rejected"], key="calls_action_decision")
                    decided_by = decision_cols[1].text_input("Decided by", key="calls_action_decided_by")
                    decision_note = decision_cols[2].text_input("Decision note", key="calls_action_decision_note")
                    if st.button("Apply action decision", key="calls_decide_action"):
                        try:
                            decide_extracted_action(
                                conn,
                                action_id=int(pending_options[pending_label]["id"]),
                                status=decision,
                                decided_by=decided_by or None,
                                decision_note=decision_note or None,
                            )
                        except Exception as exc:  # pragma: no cover
                            st.error(f"Failed to apply action decision: {exc}")
                        else:
                            st.success("Action decision recorded.")
                            _rerun_app()
            else:
                st.caption("No extracted actions recorded yet.")

    with st.expander("Ambient office transcript sessions", expanded=False):
        ambient_create_cols = st.columns(5)
        ambient_title = ambient_create_cols[0].text_input("Title", key="calls_ambient_title")
        ambient_location = ambient_create_cols[1].text_input("Source location", key="calls_ambient_location")
        ambient_device = ambient_create_cols[2].text_input("Source device", key="calls_ambient_device")
        ambient_team = ambient_create_cols[3].text_input("Team label", key="calls_ambient_team")
        ambient_status = ambient_create_cols[4].selectbox("Status", options=list(AMBIENT_SESSION_STATUSES), key="calls_ambient_status")
        ambient_job_id = st.text_input("Ambient linked job id (optional)", key="calls_ambient_job")
        if st.button("Create ambient session", key="calls_ambient_create"):
            try:
                create_ambient_session(
                    conn,
                    title=ambient_title or None,
                    source_location=ambient_location or None,
                    source_device=ambient_device or None,
                    team_label=ambient_team or None,
                    status=ambient_status,
                    job_id=_int_or_none(ambient_job_id),
                )
            except Exception as exc:  # pragma: no cover
                st.error(f"Failed to create ambient session: {exc}")
            else:
                st.success("Ambient session created.")
                _rerun_app()
        ambient_rows = list_ambient_sessions(conn, limit=25)
        if ambient_rows:
            st.dataframe(
                pd.DataFrame(ambient_rows)[["id", "title", "sourceLocation", "teamLabel", "status", "jobId", "createdAt"]].rename(columns={"id": "Ambient"}),
                width="stretch",
                hide_index=True,
            )
            ambient_map = {
                f"#{row['id']} · {row.get('title') or row.get('sourceLocation') or 'ambient'}": row
                for row in ambient_rows
            }
            ambient_label = st.selectbox("Ambient session", options=list(ambient_map.keys()), key="calls_ambient_selected")
            ambient_selected = ambient_map[ambient_label]
            ambient_transcripts = list_ambient_transcript_artifacts(conn, ambient_session_id=int(ambient_selected["id"]))
            ambient_scenario_cols = st.columns(2)
            ambient_scenario = ambient_scenario_cols[0].text_input("Ambient fake scenario", key="calls_ambient_scenario")
            ambient_goal = ambient_scenario_cols[1].text_input("Ambient desired outcome", key="calls_ambient_goal")
            if st.button("Generate ambient fake transcript", key="calls_ambient_fake"):
                try:
                    generate_fake_ambient_transcript_artifact(
                        conn,
                        ambient_session_id=int(ambient_selected["id"]),
                        scenario=ambient_scenario or None,
                        operator_goal=ambient_goal or None,
                    )
                except Exception as exc:  # pragma: no cover
                    st.error(f"Failed to generate ambient transcript: {exc}")
                else:
                    st.success("Ambient transcript generated.")
                    _rerun_app()
            if ambient_transcripts:
                st.dataframe(
                    pd.DataFrame(ambient_transcripts)[["id", "status", "confidence", "isFinal", "createdAt"]].rename(columns={"id": "Artifact"}),
                    width="stretch",
                    hide_index=True,
                )
                latest_ambient_text = next((row.get("transcriptText") for row in ambient_transcripts if row.get("transcriptText")), None)
                if latest_ambient_text:
                    st.text_area("Latest ambient transcript", value=str(latest_ambient_text), height=160, key="calls_ambient_text", disabled=True)
        else:
            st.caption("No ambient sessions recorded yet.")

    with st.expander("Worker time capture review", expanded=False):
        time_cols = st.columns(6)
        time_event_type = time_cols[0].selectbox("Event type", options=list(WORKER_TIME_EVENT_TYPES), key="calls_time_event_type")
        time_channel = time_cols[1].selectbox("Channel", options=list(WORKER_TIME_CHANNELS), index=list(WORKER_TIME_CHANNELS).index("voice_call"), key="calls_time_channel")
        time_worker_id = time_cols[2].text_input("Worker id", key="calls_time_worker_id")
        time_phone = time_cols[3].text_input("Caller phone", key="calls_time_phone")
        time_effective = time_cols[4].text_input("Effective timestamp", key="calls_time_effective")
        time_confidence = time_cols[5].number_input("Confidence", min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="calls_time_confidence")
        aux_cols = st.columns(6)
        time_call_id = aux_cols[0].text_input("Call event id", key="calls_time_call_id")
        time_call_session_id = aux_cols[1].text_input("Call session id", key="calls_time_call_session_id")
        time_call_leg_id = aux_cols[2].text_input("Call leg id", key="calls_time_call_leg_id")
        time_job_id = aux_cols[3].text_input("Job id", key="calls_time_job_id")
        time_segment_id = aux_cols[4].text_input("Segment id", key="calls_time_segment_id")
        time_truck_id = aux_cols[5].text_input("Truck id", key="calls_time_truck_id")
        if st.button("Record worker time capture", key="calls_time_record"):
            try:
                record_worker_time_capture_event(
                    conn,
                    event_type=time_event_type,
                    channel=time_channel,
                    effective_timestamp=time_effective or None,
                    worker_id=_int_or_none(time_worker_id),
                    caller_phone=time_phone or None,
                    call_event_id=_int_or_none(time_call_id),
                    call_session_id=_int_or_none(time_call_session_id),
                    call_leg_id=_int_or_none(time_call_leg_id),
                    job_id=_int_or_none(time_job_id),
                    segment_id=_int_or_none(time_segment_id),
                    truck_id=time_truck_id or None,
                    confidence=float(time_confidence),
                )
            except Exception as exc:  # pragma: no cover
                st.error(f"Failed to record worker time capture: {exc}")
            else:
                st.success("Worker time capture recorded.")
                _rerun_app()
        worker_time_rows = list_worker_time_capture_events(conn, limit=50)
        if worker_time_rows:
            st.dataframe(
                pd.DataFrame(worker_time_rows)[
                    ["id", "callSessionId", "callLegId", "workerName", "workerNameRaw", "eventType", "channel", "effectiveTimestamp", "confidence", "reviewStatus", "jobId", "truckId"]
                ].rename(columns={"id": "Event"}),
                width="stretch",
                hide_index=True,
            )
            pending = [row for row in worker_time_rows if row["reviewStatus"] == "pending_review"]
            if pending:
                pending_options = {f"#{row['id']} · {row['eventType']} · {row.get('workerName') or row.get('workerNameRaw') or 'unknown'}": row for row in pending}
                pending_label = st.selectbox("Pending time event", options=list(pending_options.keys()), key="calls_time_pending")
                decision_cols = st.columns(5)
                time_review_status = decision_cols[0].selectbox("Review decision", options=["accepted", "rejected"], key="calls_time_review_status")
                time_reviewer = decision_cols[1].text_input("Reviewer", key="calls_time_reviewer")
                time_resolved_worker_id = decision_cols[2].text_input("Resolved worker id", key="calls_time_resolved_worker")
                time_review_job_id = decision_cols[3].text_input("Resolved job id", key="calls_time_review_job")
                time_review_segment_id = decision_cols[4].text_input("Resolved segment id", key="calls_time_review_segment")
                time_review_note = st.text_input("Review note", key="calls_time_review_note")
                if st.button("Apply time review", key="calls_time_apply_review"):
                    try:
                        decide_worker_time_capture_event(
                            conn,
                            event_id=int(pending_options[pending_label]["id"]),
                            review_status=time_review_status,
                            reviewer=time_reviewer or None,
                            review_note=time_review_note or None,
                            worker_id=_int_or_none(time_resolved_worker_id),
                            job_id=_int_or_none(time_review_job_id),
                            segment_id=_int_or_none(time_review_segment_id),
                        )
                    except Exception as exc:  # pragma: no cover
                        st.error(f"Failed to review worker time capture: {exc}")
                    else:
                        st.success("Worker time capture reviewed.")
                        _rerun_app()

    with st.expander("Recent append-only egress", expanded=False):
        egress = list_state_egress_events(conn, limit=25)
        if egress:
            st.dataframe(
                pd.DataFrame(egress)[["eventType", "sourceEntityId", "authorityClass", "occurredAt", "ingestedAt"]],
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("No append-only egress events recorded yet.")


def _int_or_none(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None
