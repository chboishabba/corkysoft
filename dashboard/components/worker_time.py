from __future__ import annotations

import os
import sqlite3
from typing import Any, Callable

import pandas as pd
import streamlit as st

from analytics.driver_shifts import (
    DEFAULT_DRIVER_SHEET_NAME,
    import_driver_shifts_from_sheet,
    load_driver_shifts_dataframe,
)
from analytics.operations_assignment import (
    list_labor_reconciliation,
    list_planned_labor_assignments,
)
from corkysoft.call_ops import (
    decide_worker_time_capture_event,
    list_worker_time_capture_events,
)


def worker_time_events_df(
    conn: sqlite3.Connection,
    *,
    limit: int = 500,
) -> pd.DataFrame:
    rows = list_worker_time_capture_events(conn, limit=limit)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "createdAt" in df.columns:
        df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce")
    if "reviewedAt" in df.columns:
        df["reviewedAt"] = pd.to_datetime(df["reviewedAt"], errors="coerce")
    if "effectiveTimestamp" in df.columns:
        df["effectiveTimestamp"] = pd.to_datetime(df["effectiveTimestamp"], errors="coerce")
    if "rawPayload" in df.columns:
        df["anomalyFlags"] = df["rawPayload"].apply(
            lambda payload: ", ".join((payload or {}).get("anomalyFlags", []))
            if isinstance(payload, dict)
            else ""
        )
    else:
        df["anomalyFlags"] = ""
    return df


def _build_worker_time_shift_comparison(
    *,
    imported_shifts: pd.DataFrame,
    worker_time_events: pd.DataFrame,
) -> pd.DataFrame:
    def _norm(value: Any) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        if isinstance(value, int):
            return str(value)
        text = str(value).strip()
        if text.endswith(".0"):
            try:
                return str(int(float(text)))
            except ValueError:
                return text
        return text

    def _event_within_shift_window(
        shift_date_value: Any,
        shift_window_start: Any,
        shift_window_end: Any,
        effective_timestamp: Any,
    ) -> bool | None:
        if pd.isna(shift_date_value) or pd.isna(effective_timestamp):
            return None
        if not _norm(shift_window_start) or not _norm(shift_window_end):
            return None
        event_ts = pd.to_datetime(effective_timestamp, errors="coerce")
        if pd.isna(event_ts):
            return None
        if getattr(event_ts, "tzinfo", None) is not None:
            event_ts = event_ts.tz_localize(None)
        window_start = pd.to_datetime(
            f"{shift_date_value} {_norm(shift_window_start)}",
            errors="coerce",
        )
        window_end = pd.to_datetime(
            f"{shift_date_value} {_norm(shift_window_end)}",
            errors="coerce",
        )
        if pd.isna(window_start) or pd.isna(window_end):
            return None
        if window_end < window_start:
            window_end = window_end + pd.Timedelta(days=1)
            if event_ts < window_start:
                event_ts = event_ts + pd.Timedelta(days=1)
        return bool(window_start <= event_ts <= window_end)

    comparison_rows: list[dict[str, Any]] = []
    accepted_events = worker_time_events[
        worker_time_events["reviewStatus"] == "accepted"
    ].copy()
    accepted_events = accepted_events.reset_index(drop=True)
    matched_event_indexes: set[int] = set()

    for _, imported_row in imported_shifts.iterrows():
        shift_date = imported_row.get("shift_date")
        if pd.isna(shift_date):
            continue
        shift_date_str = str(shift_date)
        worker_name = _norm(imported_row.get("worker_name"))
        truck_id = _norm(imported_row.get("truck_id"))
        linked_job_id = _norm(imported_row.get("linked_job_id"))
        imported_window = " - ".join(
            part
            for part in [
                _norm(imported_row.get("shift_window_start")),
                _norm(imported_row.get("shift_window_end")),
            ]
            if part
        ) or "n/a"

        candidate_mask = (
            accepted_events["effective_date"].astype(str) == shift_date_str
        ) & (accepted_events["workerName"].fillna("").astype(str).str.strip() == worker_name)
        candidates = accepted_events[candidate_mask].copy()

        status = "imported_only"
        call_truck = ""
        call_job = ""
        call_time = ""
        matched_event_index: int | None = None

        if not candidates.empty:
            in_window_candidates: list[tuple[int, pd.Series]] = []
            fallback_candidates: list[tuple[int, pd.Series]] = []
            for candidate_index, candidate in candidates.iterrows():
                within_window = _event_within_shift_window(
                    shift_date,
                    imported_row.get("shift_window_start"),
                    imported_row.get("shift_window_end"),
                    candidate.get("effectiveTimestamp"),
                )
                if within_window is True or within_window is None:
                    in_window_candidates.append((candidate_index, candidate))
                else:
                    fallback_candidates.append((candidate_index, candidate))

            candidate_groups = [in_window_candidates, fallback_candidates]
            for candidate_group in candidate_groups:
                for candidate_index, candidate in candidate_group:
                    candidate_truck = _norm(candidate.get("truckId"))
                    candidate_job = _norm(candidate.get("jobId"))
                    candidate_time = _norm(candidate.get("effectiveTimestamp"))
                    same_truck = candidate_truck == truck_id
                    same_job = candidate_job == linked_job_id
                    if same_truck and same_job:
                        status = "matched" if candidate_group is in_window_candidates else "time_mismatch"
                        call_truck = candidate_truck
                        call_job = candidate_job
                        call_time = candidate_time
                        matched_event_index = candidate_index
                        break
                if matched_event_index is not None:
                    break

            if matched_event_index is None and in_window_candidates:
                for candidate_index, candidate in in_window_candidates:
                    candidate_truck = _norm(candidate.get("truckId"))
                    candidate_job = _norm(candidate.get("jobId"))
                    candidate_time = _norm(candidate.get("effectiveTimestamp"))
                    same_truck = candidate_truck == truck_id
                    same_job = candidate_job == linked_job_id
                    if same_truck and not same_job:
                        status = "job_mismatch"
                    elif same_job and not same_truck:
                        status = "truck_mismatch"
                    else:
                        status = "assignment_mismatch"
                    call_truck = candidate_truck
                    call_job = candidate_job
                    call_time = candidate_time
                    matched_event_index = candidate_index
                    break

            if matched_event_index is None and fallback_candidates:
                candidate_index, candidate = fallback_candidates[0]
                candidate_truck = _norm(candidate.get("truckId"))
                candidate_job = _norm(candidate.get("jobId"))
                call_time = _norm(candidate.get("effectiveTimestamp"))
                same_truck = candidate_truck == truck_id
                same_job = candidate_job == linked_job_id
                if same_truck and same_job:
                    status = "time_mismatch"
                elif same_truck and not same_job:
                    status = "job_mismatch"
                elif same_job and not same_truck:
                    status = "truck_mismatch"
                else:
                    status = "assignment_mismatch"
                call_truck = candidate_truck
                call_job = candidate_job
                matched_event_index = candidate_index

        if matched_event_index is not None:
            matched_event_indexes.add(matched_event_index)

        comparison_rows.append(
            {
                "Status": status,
                "Date": shift_date_str,
                "Worker": worker_name,
                "Imported window": imported_window,
                "Call time": call_time or "n/a",
                "Imported truck": truck_id or "n/a",
                "Call truck": call_truck or "n/a",
                "Imported job": linked_job_id or "n/a",
                "Call job": call_job or "n/a",
            }
        )

    unmatched_events = accepted_events.drop(index=list(matched_event_indexes), errors="ignore")
    for _, event_row in unmatched_events.iterrows():
        effective_date = event_row.get("effective_date")
        if pd.isna(effective_date):
            continue
        comparison_rows.append(
            {
                "Status": "call_only",
                "Date": str(effective_date),
                "Worker": _norm(event_row.get("workerName")),
                "Imported window": "n/a",
                "Call time": _norm(event_row.get("effectiveTimestamp")) or "n/a",
                "Imported truck": "n/a",
                "Call truck": _norm(event_row.get("truckId")) or "n/a",
                "Imported job": "n/a",
                "Call job": _norm(event_row.get("jobId")) or "n/a",
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    if comparison_df.empty:
        return comparison_df
    return comparison_df.sort_values(
        by=["Date", "Worker", "Status", "Imported truck", "Call truck"],
        ascending=[True, True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)


def _display_worker_time_shift_comparison(
    comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    if comparison_df.empty:
        return comparison_df
    status_labels = {
        "matched": "Matched",
        "time_mismatch": "Timing drift",
        "truck_mismatch": "Truck mismatch",
        "job_mismatch": "Job mismatch",
        "assignment_mismatch": "Truck + job mismatch",
        "imported_only": "Imported only",
        "call_only": "Call-derived only",
    }
    status_explanations = {
        "matched": "Imported shift and accepted worker-time event align.",
        "time_mismatch": "Worker/job/truck align, but the accepted event falls outside the imported shift window.",
        "truck_mismatch": "Worker and job align, but truck assignment differs.",
        "job_mismatch": "Worker and truck align, but linked job differs.",
        "assignment_mismatch": "Worker matches, but both truck and job differ.",
        "imported_only": "Imported shift has no accepted call-derived worker-time match.",
        "call_only": "Accepted worker-time event has no imported shift match.",
    }
    display_df = comparison_df.copy()
    display_df["Reconciliation"] = display_df["Status"].map(status_labels).fillna(
        display_df["Status"]
    )
    display_df["Why"] = display_df["Status"].map(status_explanations).fillna("")
    ordered_columns = [
        "Reconciliation",
        "Why",
        "Date",
        "Worker",
        "Imported window",
        "Call time",
        "Imported truck",
        "Call truck",
        "Imported job",
        "Call job",
        "Status",
    ]
    present_columns = [column for column in ordered_columns if column in display_df.columns]
    return display_df[present_columns]


def _int_or_none(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def render_worker_time_review_controls(
    conn: sqlite3.Connection,
    *,
    pending_events: pd.DataFrame,
    key_prefix: str,
    rerun_app: Callable[[], None],
) -> None:
    if pending_events.empty:
        st.caption("No pending worker-time events for the current selection.")
        return

    option_map = {
        (
            f"#{int(row['id'])} · {row.get('eventType') or 'event'} · "
            f"{row.get('workerName') or row.get('workerNameRaw') or 'unknown'}"
        ): row
        for _, row in pending_events.iterrows()
    }
    selected_label = st.selectbox(
        "Pending worker-time event",
        options=list(option_map.keys()),
        key=f"{key_prefix}_pending_worker_time_event",
    )
    selected = option_map[selected_label]
    decision_cols = st.columns(5)
    review_status = decision_cols[0].selectbox(
        "Review decision",
        options=["accepted", "rejected"],
        key=f"{key_prefix}_worker_time_review_status",
    )
    reviewer = decision_cols[1].text_input(
        "Reviewer",
        value="",
        key=f"{key_prefix}_worker_time_reviewer",
    )
    resolved_worker_id = decision_cols[2].text_input(
        "Resolved worker id",
        value=str(selected.get("workerId") or ""),
        key=f"{key_prefix}_worker_time_worker_id",
    )
    resolved_job_id = decision_cols[3].text_input(
        "Resolved job id",
        value=str(selected.get("jobId") or ""),
        key=f"{key_prefix}_worker_time_job_id",
    )
    resolved_segment_id = decision_cols[4].text_input(
        "Resolved segment id",
        value=str(selected.get("segmentId") or ""),
        key=f"{key_prefix}_worker_time_segment_id",
    )
    follow_cols = st.columns(2)
    resolved_truck_id = follow_cols[0].text_input(
        "Resolved truck id",
        value=str(selected.get("truckId") or ""),
        key=f"{key_prefix}_worker_time_truck_id",
    )
    review_note = follow_cols[1].text_input(
        "Review note",
        value="",
        key=f"{key_prefix}_worker_time_review_note",
    )
    if st.button("Apply worker-time review", key=f"{key_prefix}_apply_worker_time_review"):
        try:
            decide_worker_time_capture_event(
                conn,
                event_id=int(selected["id"]),
                review_status=review_status,
                reviewer=reviewer or None,
                review_note=review_note or None,
                worker_id=_int_or_none(resolved_worker_id),
                job_id=_int_or_none(resolved_job_id),
                segment_id=_int_or_none(resolved_segment_id),
                truck_id=(resolved_truck_id or None),
            )
        except Exception as exc:  # pragma: no cover
            st.error(f"Failed to review worker-time event: {exc}")
        else:
            st.success("Worker-time review recorded.")
            rerun_app()


def render_driver_shifts_tab(
    conn: sqlite3.Connection,
    *,
    rerun_app: Callable[[], None],
) -> None:
    st.subheader("Labor planning and shift reconciliation")
    st.caption(
        "Use native planned labor from job segments as the planning surface. VEHICLE_DRIVER remains an imported reconciliation feed."
    )

    roster_df = pd.DataFrame(list_planned_labor_assignments(conn))
    st.markdown("#### Native planned labor roster")
    if roster_df.empty:
        st.caption("No planned labor assignments exist yet. Assign workers and trucks to job segments in Operations.")
    else:
        roster_display = roster_df.copy()
        roster_display["truckIds"] = roster_display["truckIds"].apply(
            lambda values: ", ".join(values) if isinstance(values, list) else ""
        )
        roster_display["truckNames"] = roster_display["truckNames"].apply(
            lambda values: ", ".join(values) if isinstance(values, list) else ""
        )
        st.dataframe(
            roster_display[
                [
                    "jobId",
                    "segmentSequence",
                    "workerName",
                    "truckIds",
                    "plannedStart",
                    "plannedEnd",
                    "assignmentStatus",
                ]
            ].rename(
                columns={
                    "jobId": "Job",
                    "segmentSequence": "Segment",
                    "workerName": "Worker",
                    "truckIds": "Trucks",
                    "plannedStart": "Planned start",
                    "plannedEnd": "Planned end",
                    "assignmentStatus": "Status",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    reconciliation = pd.DataFrame(list_labor_reconciliation(conn))
    st.markdown("#### Plan vs imported shift reconciliation")
    if reconciliation.empty:
        st.caption("No planned/imported labor reconciliation items available yet.")
    else:
        recon_cols = st.columns(3)
        recon_cols[0].metric(
            "Planned only",
            int((reconciliation["status"] == "planned_only").sum()),
        )
        recon_cols[1].metric(
            "Imported only",
            int((reconciliation["status"] == "imported_only").sum()),
        )
        recon_cols[2].metric(
            "Matched",
            int((reconciliation["status"] == "matched").sum()),
        )
        recon_display = reconciliation.copy()
        recon_display["truckIds"] = recon_display["truckIds"].apply(
            lambda values: ", ".join(values) if isinstance(values, list) else ""
        )
        st.dataframe(
            recon_display[
                [
                    "status",
                    "shiftDate",
                    "workerName",
                    "truckIds",
                    "jobId",
                    "segmentId",
                    "source",
                ]
            ].rename(
                columns={
                    "status": "Status",
                    "shiftDate": "Date",
                    "workerName": "Worker",
                    "truckIds": "Trucks",
                    "jobId": "Job",
                    "segmentId": "Segment",
                    "source": "Source",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    worker_time_df = worker_time_events_df(conn, limit=500)
    st.markdown("#### Reviewed worker-time capture")
    if worker_time_df.empty:
        st.caption("No worker-time capture events recorded yet.")
    else:
        worker_time_metric_cols = st.columns(4)
        worker_time_metric_cols[0].metric(
            "Pending review",
            int((worker_time_df["reviewStatus"] == "pending_review").sum()),
        )
        worker_time_metric_cols[1].metric(
            "Accepted",
            int((worker_time_df["reviewStatus"] == "accepted").sum()),
        )
        worker_time_metric_cols[2].metric(
            "Rejected",
            int((worker_time_df["reviewStatus"] == "rejected").sum()),
        )
        accepted_hours_proxy = worker_time_df[
            worker_time_df["reviewStatus"] == "accepted"
        ].shape[0]
        worker_time_metric_cols[3].metric("Accepted events", int(accepted_hours_proxy))
        st.dataframe(
            worker_time_df[
                [
                    "id",
                    "workerName",
                    "eventType",
                    "channel",
                    "effectiveTimestamp",
                    "reviewStatus",
                    "jobId",
                    "segmentId",
                    "truckId",
                    "anomalyFlags",
                ]
            ].rename(
                columns={
                    "id": "Event",
                    "workerName": "Worker",
                    "eventType": "Event type",
                    "channel": "Channel",
                    "effectiveTimestamp": "Effective time",
                    "reviewStatus": "Review",
                    "jobId": "Job",
                    "segmentId": "Segment",
                    "truckId": "Truck",
                    "anomalyFlags": "Anomalies",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    st.divider()
    st.markdown("#### Imported VEHICLE_DRIVER feed")

    with st.expander("Import from Google Sheet", expanded=False):
        default_sheet_id = os.environ.get("VEHICLE_DRIVER_SHEET_ID", "")
        sheet_id = st.text_input(
            "Sheet ID or full URL",
            value=default_sheet_id,
            help="Paste the Google Sheet ID or sharing URL for the VEHICLE_DRIVER tab.",
            key="driver_shift_sheet_id",
        )
        sheet_name = st.text_input(
            "Sheet tab name",
            value=DEFAULT_DRIVER_SHEET_NAME,
            help="Defaults to the VEHICLE_DRIVER tab name.",
            key="driver_shift_sheet_name",
        )
        if st.button(
            "Import driver shifts",
            type="primary",
            key="driver_shift_import_button",
            disabled=not sheet_id.strip(),
        ):
            try:
                inserted, updated = import_driver_shifts_from_sheet(
                    conn,
                    sheet_id=sheet_id.strip(),
                    sheet_name=sheet_name.strip() or DEFAULT_DRIVER_SHEET_NAME,
                )
            except Exception as exc:  # pragma: no cover - surfaced in UI
                st.error(f"Failed to import driver shifts: {exc}")
            else:
                st.success(
                    f"Imported {inserted} new shift entries and refreshed {updated} existing rows."
                )
                rerun_app()

    df = load_driver_shifts_dataframe(conn)
    if df.empty:
        st.info(
            "No driver shifts available. Import the VEHICLE_DRIVER sheet to populate this view."
        )
        return

    df = df.copy()
    df["shift_date"] = pd.to_datetime(df["shift_date"], errors="coerce")
    df = df.dropna(subset=["shift_date"])
    if df.empty:
        st.info("Driver shift dates could not be parsed from the data.")
        return

    min_date = df["shift_date"].min().date()
    max_date = df["shift_date"].max().date()
    date_range = st.date_input(
        "Shift date range",
        value=(min_date, max_date),
    )
    selected_workers = st.multiselect(
        "Drivers/workers",
        sorted(df["worker_name"].dropna().unique().tolist()),
    )
    selected_trucks = st.multiselect(
        "Trucks",
        sorted(df["truck_id"].dropna().unique().tolist()),
    )

    filtered = df
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        if start_date:
            filtered = filtered[filtered["shift_date"] >= pd.to_datetime(start_date)]
        if end_date:
            filtered = filtered[filtered["shift_date"] <= pd.to_datetime(end_date)]
    if selected_workers:
        filtered = filtered[filtered["worker_name"].isin(selected_workers)]
    if selected_trucks:
        filtered = filtered[filtered["truck_id"].isin(selected_trucks)]

    worker_time_filtered = worker_time_df.copy()
    if not worker_time_filtered.empty:
        effective_dates = pd.to_datetime(
            worker_time_filtered["effectiveTimestamp"], errors="coerce"
        )
        worker_time_filtered = worker_time_filtered.assign(
            effective_date=effective_dates.dt.date
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            if start_date:
                worker_time_filtered = worker_time_filtered[
                    worker_time_filtered["effective_date"] >= start_date
                ]
            if end_date:
                worker_time_filtered = worker_time_filtered[
                    worker_time_filtered["effective_date"] <= end_date
                ]
        if selected_workers:
            worker_time_filtered = worker_time_filtered[
                worker_time_filtered["workerName"].isin(selected_workers)
            ]
        if selected_trucks:
            worker_time_filtered = worker_time_filtered[
                worker_time_filtered["truckId"].isin(selected_trucks)
            ]

    filtered = filtered.sort_values(
        by=["shift_date", "shift_start", "truck_id", "worker_name"],
        ascending=[False, True, True, True],
    )
    filtered = filtered.assign(shift_date=filtered["shift_date"].dt.date)

    shift_vs_call_df = _build_worker_time_shift_comparison(
        imported_shifts=filtered,
        worker_time_events=worker_time_filtered,
    )

    total_hours = filtered["hours"].sum(skipna=True) if "hours" in filtered else 0
    total_cost = (
        filtered["cost_total"].sum(skipna=True) if "cost_total" in filtered else 0
    )
    metric_cols = st.columns(2)
    metric_cols[0].metric("Total hours", f"{total_hours:,.2f}")
    metric_cols[1].metric("Total cost", f"${total_cost:,.2f}")

    display_columns = [
        "shift_date",
        "truck_id",
        "truck_name",
        "worker_name",
        "linked_job_id",
        "shipment_id",
        "role",
        "shift_window_start",
        "shift_window_end",
        "ticket_numbers",
        "shift_start",
        "shift_end",
        "hours",
        "hourly_rate",
        "cost_total",
        "source",
    ]
    present_columns = [col for col in display_columns if col in filtered.columns]
    st.dataframe(filtered[present_columns], width="stretch")

    if not worker_time_filtered.empty:
        st.markdown("#### Worker-time events in selected range")
        st.dataframe(
            worker_time_filtered[
                [
                    "effective_date",
                    "workerName",
                    "eventType",
                    "channel",
                    "reviewStatus",
                    "jobId",
                    "segmentId",
                    "truckId",
                    "anomalyFlags",
                ]
            ].rename(
                columns={
                    "effective_date": "Date",
                    "workerName": "Worker",
                    "eventType": "Event type",
                    "channel": "Channel",
                    "reviewStatus": "Review",
                    "jobId": "Job",
                    "segmentId": "Segment",
                    "truckId": "Truck",
                    "anomalyFlags": "Anomalies",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        with st.expander("Review pending worker-time events in this range", expanded=False):
            pending_worker_time = worker_time_filtered[
                worker_time_filtered["reviewStatus"] == "pending_review"
            ].copy()
            render_worker_time_review_controls(
                conn,
                pending_events=pending_worker_time,
                key_prefix="driver_shifts_worker_time",
                rerun_app=rerun_app,
            )

    st.markdown("#### Imported shifts vs accepted call-derived worker time")
    if shift_vs_call_df.empty:
        st.caption("No imported/accepted comparison rows are available for the current selection.")
    else:
        display_shift_vs_call_df = _display_worker_time_shift_comparison(shift_vs_call_df)
        compare_cols = st.columns(4)
        compare_cols[0].metric(
            "Matched",
            int((shift_vs_call_df["Status"] == "matched").sum()),
        )
        compare_cols[1].metric(
            "Mismatch / timing drift",
            int(
                shift_vs_call_df["Status"].isin(
                    ["truck_mismatch", "job_mismatch", "assignment_mismatch", "time_mismatch"]
                ).sum()
            ),
        )
        compare_cols[2].metric(
            "Imported only",
            int((shift_vs_call_df["Status"] == "imported_only").sum()),
        )
        compare_cols[3].metric(
            "Call-derived only",
            int((shift_vs_call_df["Status"] == "call_only").sum()),
        )
        st.caption(
            "Rows below show the exact reconciliation class for each imported shift or accepted call-derived event."
        )
        st.dataframe(display_shift_vs_call_df, width="stretch", hide_index=True)
