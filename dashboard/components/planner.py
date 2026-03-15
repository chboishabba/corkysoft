from __future__ import annotations

import sqlite3
import math
import json
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.db import list_inventory_requirements
from analytics.db.site_media import (
    MEDIA_INFERENCE_TYPES,
    SITE_ASSESSMENT_RISK_LEVELS,
    SITE_KIND_VALUES,
    SITE_MEDIA_TYPES,
    SITE_TRUCK_SUITABILITY,
    create_media_inference_result,
    create_site_media_asset,
    list_media_inference_results,
    list_site_media_assets,
    persist_uploaded_site_media,
    review_media_inference_result,
    upsert_site_assessment,
)
from analytics.live_data import extract_route_path
from analytics.operations_assignment import ensure_segment
from analytics.planner import (
    build_planner_proposal,
    infer_planner_corridor_for_job,
    list_planner_corridor_candidates,
)
from dashboard.map_provider import plotly_map_layout
from dashboard.map_provider import (
    google_street_view_360_url,
    google_street_view_static_url,
    street_view_available,
)


def render_planner_tab(filtered_df: pd.DataFrame, conn: sqlite3.Connection) -> None:
    st.subheader("Planner")
    st.caption(
        "Hybrid planning surface for pre-award route shaping and post-award operational leg planning. Planner confirms into internal job_segments only after review."
    )

    corridor_candidates = list_planner_corridor_candidates(filtered_df)
    if not corridor_candidates:
        st.info(
            "No corridor history is available in the current dataset. Load historical or live route data to use the planner."
        )
        return

    jobs = _load_jobs(conn)
    planning_mode = st.radio(
        "Planning mode",
        options=["job_first", "corridor_first"],
        index=0,
        horizontal=True,
        format_func=lambda value: "Job-first" if value == "job_first" else "Map/corridor-first",
        key="planner_selection_mode",
    )

    st.markdown("#### Corridor candidates")
    candidates_df = pd.DataFrame(corridor_candidates)
    st.dataframe(
        candidates_df.rename(
            columns={
                "corridor": "Corridor",
                "jobCount": "Jobs",
                "overlapScore": "Overlap",
                "familiarityScore": "Familiarity",
                "avgMarginPct": "Avg margin %",
                "profitableShare": "Profitable share",
                "avgDistanceKm": "Avg km",
                "avgPricePerM3": "Avg $/m3",
                "avgVolumeM3": "Avg m3",
                "candidateScore": "Score",
                "origin": "Origin",
                "destination": "Destination",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    selected_job: dict[str, Any] | None = None
    selected_job_id: int | None = None
    target_job_id: int | None = None
    selected_corridor = corridor_candidates[0]["corridor"]

    control_cols = st.columns(3)
    if planning_mode == "job_first":
        if not jobs:
            st.warning("No jobs are available yet. Use corridor-first mode until jobs exist.")
            return
        job_labels = {
            _format_job_label(job): int(job["id"])
            for job in jobs
        }
        selected_job_label = control_cols[0].selectbox(
            "Job",
            options=list(job_labels.keys()),
            key="planner_job_first_selected_job",
        )
        selected_job_id = int(job_labels[selected_job_label])
        selected_job = next(job for job in jobs if int(job["id"]) == selected_job_id)
        inferred_corridor = infer_planner_corridor_for_job(
            filtered_df,
            origin=selected_job.get("origin") or selected_job.get("origin_resolved"),
            destination=selected_job.get("destination") or selected_job.get("destination_resolved"),
        )
        corridor_options = [row["corridor"] for row in corridor_candidates[:25]]
        if inferred_corridor and inferred_corridor not in corridor_options:
            corridor_options = [inferred_corridor, *corridor_options]
        selected_corridor = control_cols[1].selectbox(
            "Suggested corridor",
            options=corridor_options,
            index=corridor_options.index(inferred_corridor) if inferred_corridor in corridor_options else 0,
            key="planner_job_first_corridor",
        )
        target_job_id = selected_job_id
    else:
        corridor_options = [row["corridor"] for row in corridor_candidates[:25]]
        selected_corridor = control_cols[0].selectbox(
            "Target corridor",
            options=corridor_options,
            key="planner_target_corridor",
        )
        job_option_labels = ["<attach later>"] + [_format_job_label(job) for job in jobs]
        selected_job_label = control_cols[1].selectbox(
            "Attach to job",
            options=job_option_labels,
            key="planner_corridor_first_target_job",
        )
        if selected_job_label != "<attach later>":
            selected_job_id = next(
                int(job["id"]) for job in jobs if _format_job_label(job) == selected_job_label
            )
            selected_job = next(job for job in jobs if int(job["id"]) == selected_job_id)
            target_job_id = selected_job_id

    selected_template = control_cols[2].selectbox(
        "Draft leg template",
        options=["single_leg", "pickup_linehaul_delivery", "staging_linehaul_delivery"],
        index=1,
        key="planner_template",
    )

    proposal = build_planner_proposal(
        conn,
        filtered_df,
        selection_mode=planning_mode,
        corridor=selected_corridor,
        preferred_template=selected_template,
        job_context=selected_job,
    )
    inventory_requirements = (
        list_inventory_requirements(conn, job_id=int(target_job_id))
        if target_job_id is not None
        else []
    )
    shortage_requirements = [
        item for item in inventory_requirements if float(item.get("shortageQuantity") or 0.0) > 0
    ]

    selection_cols = st.columns(2)
    with selection_cols[0]:
        st.markdown("#### Selection")
        selection_rows = [
            {"Field": "Mode", "Value": "Job-first" if planning_mode == "job_first" else "Map/corridor-first"},
            {"Field": "Corridor", "Value": proposal["selection"]["corridor"]},
            {"Field": "Origin", "Value": proposal["selection"]["origin"]},
            {"Field": "Destination", "Value": proposal["selection"]["destination"]},
        ]
        if proposal["jobContext"]:
            selection_rows.extend(
                [
                    {"Field": "Job", "Value": f"#{proposal['jobContext']['jobId']}"},
                    {"Field": "Client", "Value": proposal["jobContext"]["client"] or "Unknown client"},
                    {"Field": "Move date", "Value": proposal["jobContext"]["jobDate"] or "Unknown"},
                ]
            )
        st.dataframe(pd.DataFrame(selection_rows), width="stretch", hide_index=True)
    with selection_cols[1]:
        st.markdown("#### Routing preview")
        _render_planner_map(selected_job=selected_job, corridor_rows=proposal["candidateRows"], filtered_df=filtered_df)

    site_points = build_planner_site_points(
        selected_job=selected_job,
        corridor_rows=proposal["candidateRows"],
        filtered_df=filtered_df,
    )
    st.markdown("#### Site context")
    _render_planner_site_context(site_points=site_points)
    site_context = proposal.get(
        "siteContext",
        {
            "assessments": [],
            "mediaAssets": [],
            "acceptedVolumeEstimate": None,
            "acceptedSiteFeatures": [],
            "acceptedDetections": [],
        },
    )
    _render_planner_site_summary(site_context=site_context)
    if target_job_id is not None:
        _render_planner_site_management(
            conn,
            job_id=int(target_job_id),
            site_points=site_points,
        )

    context_cols = st.columns(3)
    with context_cols[0]:
        st.markdown("#### Commercial / history")
        top_candidate = proposal["candidateRows"][0] if proposal["candidateRows"] else {}
        st.metric("Historical jobs", int(top_candidate.get("jobCount") or 0))
        st.metric("Familiarity", float(top_candidate.get("familiarityScore") or 0.0))
        st.metric("Avg margin %", _format_float(top_candidate.get("avgMarginPct")))
        st.metric("Avg distance km", _format_float(proposal["routingContext"]["avgDistanceKm"]))
    with context_cols[1]:
        st.markdown("#### Routing / traffic context")
        st.metric("Routing provider", proposal["routingContext"]["routingProvider"])
        st.metric("Geometry coverage", f"{proposal['routingContext']['geometryCoveragePct']:.1f}%")
        st.metric("Route pattern", proposal["routingContext"]["routePattern"])
        if proposal["routingContext"]["selectedJobHasGeometry"]:
            st.caption("Selected job already has stored route geometry.")
        elif proposal["jobContext"]:
            st.caption("Selected job is missing stored route geometry.")
        else:
            st.caption("Traffic context is currently derived from history, distance, and geometry coverage.")
    with context_cols[2]:
        st.markdown("#### Resource-fit")
        st.metric("Spare-capacity score", proposal["resourceContext"]["spareCapacityScore"])
        st.metric("Matching spare trucks", proposal["resourceContext"]["matchingSpareTrucks"])
        st.metric(
            "Blocked resources",
            f"{proposal['resourceContext']['blockedVehicles']} vehicles / {proposal['resourceContext']['blockedWorkers']} workers",
        )
        st.caption(
            f"Planned labor on selected day: {proposal['resourceContext']['plannedLaborAssignments'] if proposal['resourceContext']['plannedLaborAssignments'] is not None else 'n/a'}"
        )

    if inventory_requirements:
        st.markdown("#### Inventory fit")
        inventory_cols = st.columns(3)
        inventory_cols[0].metric("Requirement lines", len(inventory_requirements))
        inventory_cols[1].metric(
            "Shortage qty",
            round(sum(float(item["shortageQuantity"]) for item in shortage_requirements), 2),
        )
        inventory_cols[2].metric("Shortage lines", len(shortage_requirements))
        if shortage_requirements:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Segment": item["segmentSequence"],
                            "Requirement": item["requirementName"],
                            "Architecture": item["architecture"],
                            "Required": item["requiredQuantity"],
                            "Allocated": item["allocatedQuantity"],
                            "Shortage": item["shortageQuantity"],
                            "Substitutable": item["substitutionAllowed"],
                        }
                        for item in shortage_requirements
                    ]
                ),
                width="stretch",
                hide_index=True,
            )

    if proposal["warnings"]:
        st.markdown("#### Warnings")
        for warning in proposal["warnings"]:
            st.warning(warning)

    st.markdown("#### Draft operational legs")
    legs_df = pd.DataFrame(proposal["draftLegs"])
    if legs_df.empty:
        st.caption("No draft legs could be generated for the current selection.")
    else:
        st.dataframe(
            legs_df.rename(
                columns={
                    "sequence": "Sequence",
                    "legType": "Leg type",
                    "from": "From",
                    "to": "To",
                    "note": "Planning note",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    st.markdown("#### Explainability")
    explain_cols = st.columns(4)
    for idx, key in enumerate(("history", "commercial", "routing", "resources")):
        with explain_cols[idx]:
            st.caption(key.capitalize())
            for line in proposal["explainability"].get(key, []):
                st.caption(f"- {line}")

    confirm_label = (
        "Confirm plan into selected job"
        if planning_mode == "job_first"
        else "Confirm plan into attached job"
    )
    if st.button(confirm_label, type="primary", key="planner_confirm_button"):
        if target_job_id is None:
            st.error("Choose a target job before confirming the draft plan.")
        elif not proposal["draftLegs"]:
            st.error("No draft legs are available to confirm.")
        else:
            try:
                for leg in proposal["draftLegs"]:
                    ensure_segment(
                        conn,
                        job_id=int(target_job_id),
                        segment_sequence=int(leg["sequence"]),
                        from_location=str(leg["from"]),
                        to_location=str(leg["to"]),
                    )
            except Exception as exc:  # pragma: no cover
                st.error(f"Failed to confirm planner proposal: {exc}")
            else:
                st.success(
                    "Draft operational plan confirmed into job segments. Continue in Operations to assign resources."
                )
                rerun = getattr(st, "rerun", None)
                if callable(rerun):
                    rerun()
                else:
                    _rerun_app()


def _load_jobs(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
            id,
            client,
            origin,
            destination,
            origin_resolved,
            destination_resolved,
            job_date,
            distance_km,
            volume_m3,
            volume,
            origin_lat,
            origin_lon,
            dest_lat,
            dest_lon,
            route_geojson
        FROM jobs
        ORDER BY id DESC
        LIMIT 100
        """
    ).fetchall()
    return [dict(row) for row in rows]


def _format_job_label(job: dict[str, Any]) -> str:
    return (
        f"#{job['id']} · {job.get('client') or 'Unknown client'} · "
        f"{job.get('origin_resolved') or job.get('origin') or '?'} → "
        f"{job.get('destination_resolved') or job.get('destination') or '?'}"
    )


def _render_planner_map(
    *,
    selected_job: dict[str, Any] | None,
    corridor_rows: list[dict[str, Any]],
    filtered_df: pd.DataFrame,
) -> None:
    figure = build_planner_preview_figure(
        selected_job=selected_job,
        corridor_rows=corridor_rows,
        filtered_df=filtered_df,
    )
    if figure is None:
        st.caption("Map preview is unavailable for this selection because route coordinates are missing.")
        return
    st.plotly_chart(figure, width="stretch")


def _render_planner_site_context(*, site_points: list[dict[str, Any]]) -> None:
    if not site_points:
        st.caption("Site context is unavailable for this selection because usable coordinates are missing.")
        return

    cols = st.columns(len(site_points))
    imagery_ready = street_view_available()
    for idx, point in enumerate(site_points):
        with cols[idx]:
            st.caption(point["label"])
            st.write(point["name"])
            st.caption(f"{point['lat']:.5f}, {point['lon']:.5f}")
            if imagery_ready and point.get("streetViewUrl"):
                st.image(point["streetViewUrl"], width="stretch")
                st.caption("Street-level context from the active Google routing provider.")
                if point.get("streetView360Url"):
                    st.link_button("Open 360 view", point["streetView360Url"], use_container_width=True)
            else:
                st.info(
                    "Street-level imagery is unavailable for the current provider selection. "
                    "Switch to Google Maps with a configured API key to enable it."
                )


def _render_planner_site_summary(*, site_context: dict[str, Any]) -> None:
    assessments = site_context.get("assessments", []) or []
    media_assets = site_context.get("mediaAssets", []) or []
    accepted_volume = site_context.get("acceptedVolumeEstimate") or {}
    feature_rows = site_context.get("acceptedSiteFeatures", []) or []
    derived_constraints = site_context.get("derivedConstraints") or {}

    metric_cols = st.columns(4)
    metric_cols[0].metric("Accepted assessments", len(assessments))
    metric_cols[1].metric("Linked site media", len(media_assets))
    metric_cols[2].metric("Accepted feature sets", len(feature_rows))
    metric_cols[3].metric(
        "Accepted volume",
        _format_volume_metric(accepted_volume.get("payload") if isinstance(accepted_volume, dict) else None),
    )

    if not assessments and not feature_rows and not media_assets:
        st.info(
            "No accepted site evidence yet. Save a manual site assessment to make truck fit, shuttle need, "
            "labor uplift, and access/load consequences visible in Planner."
        )

    if assessments:
        assessment_rows = [
            {
                "Site": item["siteKind"],
                "Loading": item["loadingAccessRisk"],
                "Parking": item["parkingRisk"],
                "Street": item["narrowStreetRisk"],
                "Stairs": item["stairsRisk"],
                "Clearance": item["clearanceRisk"],
                "Truck fit": item["largeVehicleSuitability"],
                "Uncertain": item["uncertaintyFlag"],
                "Note": item["note"] or "",
            }
            for item in assessments
        ]
        st.dataframe(pd.DataFrame(assessment_rows), width="stretch", hide_index=True)

    if feature_rows:
        st.caption("Accepted site-feature outputs")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Site / Source": row.get("source"),
                        "Features": ", ".join(f"{key}={value}" for key, value in sorted((row.get("payload") or {}).items())),
                        "Confidence": row.get("confidence"),
                    }
                    for row in feature_rows
                ]
            ),
            width="stretch",
            hide_index=True,
        )

    if derived_constraints:
        st.caption("Derived planning constraints from accepted site evidence")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Truck unsuitable": bool(derived_constraints.get("truckUnsuitable")),
                        "Shuttle recommended": bool(derived_constraints.get("shuttleRecommended")),
                        "Labor uplift %": int(derived_constraints.get("laborUpliftPct") or 0),
                        "Access/load uplift min": int(derived_constraints.get("accessTimeUpliftMinutes") or 0),
                        "Loading delay risk": str(derived_constraints.get("loadingDelayRisk") or "low"),
                        "Review needed": bool(derived_constraints.get("reviewNeeded")),
                        "Reasons": "; ".join(derived_constraints.get("reasons") or []),
                    }
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.caption(
            "Derived planning constraints will appear here once accepted site assessments or reviewed site outputs exist."
        )


def _render_planner_site_management(
    conn: sqlite3.Connection,
    *,
    job_id: int,
    site_points: list[dict[str, Any]],
) -> None:
    st.markdown("#### Site assessment and media")
    manage_cols = st.columns(2)
    site_options = [point["label"].lower().split()[0] for point in site_points] or list(SITE_KIND_VALUES)
    with manage_cols[0]:
        with st.form(f"planner_site_assessment_form_{job_id}"):
            site_kind = st.selectbox("Site", options=site_options, key=f"planner_site_kind_{job_id}")
            risk_cols = st.columns(3)
            loading_access_risk = risk_cols[0].selectbox("Loading access", options=list(SITE_ASSESSMENT_RISK_LEVELS), index=1, key=f"planner_loading_risk_{job_id}")
            parking_risk = risk_cols[1].selectbox("Parking", options=list(SITE_ASSESSMENT_RISK_LEVELS), index=1, key=f"planner_parking_risk_{job_id}")
            narrow_street_risk = risk_cols[2].selectbox("Narrow street", options=list(SITE_ASSESSMENT_RISK_LEVELS), index=1, key=f"planner_narrow_risk_{job_id}")
            risk_cols_2 = st.columns(3)
            stairs_risk = risk_cols_2[0].selectbox("Stairs", options=list(SITE_ASSESSMENT_RISK_LEVELS), index=1, key=f"planner_stairs_risk_{job_id}")
            clearance_risk = risk_cols_2[1].selectbox("Clearance", options=list(SITE_ASSESSMENT_RISK_LEVELS), index=1, key=f"planner_clearance_risk_{job_id}")
            truck_suitability = risk_cols_2[2].selectbox("Truck suitability", options=list(SITE_TRUCK_SUITABILITY), index=1, key=f"planner_truck_fit_{job_id}")
            uncertainty_flag = st.checkbox("Mark uncertain / review-needed", key=f"planner_site_uncertain_{job_id}")
            note = st.text_area("Assessment note", key=f"planner_site_note_{job_id}")
            reviewer = st.text_input("Reviewed by", value="planner", key=f"planner_site_reviewer_{job_id}")
            submitted = st.form_submit_button("Save site assessment")
        if submitted:
            upsert_site_assessment(
                conn,
                job_id=job_id,
                site_kind=site_kind,
                loading_access_risk=loading_access_risk,
                parking_risk=parking_risk,
                narrow_street_risk=narrow_street_risk,
                stairs_risk=stairs_risk,
                clearance_risk=clearance_risk,
                large_vehicle_suitability=truck_suitability,
                uncertainty_flag=uncertainty_flag,
                note=note,
                reviewed_by=reviewer,
                accepted=True,
            )
            st.success("Site assessment saved.")
            _rerun_app()

        save_google_site = st.selectbox(
            "Save Google site reference",
            options=["<none>"] + [point["label"] for point in site_points if point.get("streetViewUrl")],
            key=f"planner_google_ref_site_{job_id}",
        )
        if st.button("Save current Google site media", key=f"planner_save_google_media_{job_id}", disabled=save_google_site == "<none>"):
            point = next(item for item in site_points if item["label"] == save_google_site)
            create_site_media_asset(
                conn,
                job_id=job_id,
                site_kind=save_google_site.lower().split()[0],
                media_type="street_view_360" if point.get("streetView360Url") else "street_view_static",
                source="google",
                title=f"{point['label']} Google street view",
                media_url=point.get("streetView360Url") or point.get("streetViewUrl"),
                heading_degrees=None,
                metadata={
                    "streetViewStaticUrl": point.get("streetViewUrl"),
                    "streetView360Url": point.get("streetView360Url"),
                    "lat": point.get("lat"),
                    "lon": point.get("lon"),
                },
            )
            st.success("Google site media reference saved.")
            _rerun_app()

    with manage_cols[1]:
        upload_site_kind = st.selectbox("Upload site", options=site_options, key=f"planner_upload_site_kind_{job_id}")
        media_type = st.selectbox("Media type", options=list(SITE_MEDIA_TYPES), index=2, key=f"planner_media_type_{job_id}")
        uploaded = st.file_uploader(
            "Walkaround media",
            type=["jpg", "jpeg", "png", "mp4", "mov", "m4v", "webm"],
            key=f"planner_site_media_upload_{job_id}",
        )
        captured_by = st.text_input("Captured by", value="planner", key=f"planner_media_captured_by_{job_id}")
        if st.button("Upload walkaround media", key=f"planner_upload_site_media_btn_{job_id}", disabled=uploaded is None):
            if uploaded is None:
                st.error("Choose a media file first.")
            else:
                persist_uploaded_site_media(
                    conn,
                    job_id=job_id,
                    site_kind=upload_site_kind,
                    media_type=media_type,
                    source="uploaded",
                    uploaded_name=uploaded.name,
                    mime_type=getattr(uploaded, "type", None),
                    file_bytes=uploaded.getvalue(),
                    captured_by=captured_by,
                )
                st.success("Walkaround media uploaded.")
                _rerun_app()

    st.markdown("#### Advisory CV / volume outputs")
    pending_rows = list_media_inference_results(conn, job_id=job_id, statuses=("pending_review",))
    media_assets = list_site_media_assets(conn, job_id=job_id)
    asset_options = {"<none>": None}
    asset_options.update({
        f"#{asset['id']} · {asset['siteKind']} · {asset['mediaType']}": int(asset["id"])
        for asset in media_assets
    })
    with st.form(f"planner_inference_form_{job_id}"):
        infer_cols = st.columns(3)
        result_type = infer_cols[0].selectbox("Inference type", options=list(MEDIA_INFERENCE_TYPES), key=f"planner_inference_type_{job_id}")
        linked_asset = infer_cols[1].selectbox("Media asset", options=list(asset_options.keys()), key=f"planner_inference_asset_{job_id}")
        confidence = infer_cols[2].number_input("Confidence", min_value=0.0, max_value=1.0, value=0.75, step=0.05, key=f"planner_inference_conf_{job_id}")
        payload_text = st.text_area(
            "Inference payload (JSON)",
            value='{"estimated_m3": 36.0}' if result_type == "volume_estimate" else '{"narrow_access": true}',
            key=f"planner_inference_payload_{job_id}",
        )
        model_name = st.text_input("Model name", value="manual-sim", key=f"planner_inference_model_{job_id}")
        submit_inference = st.form_submit_button("Add advisory inference")
    if submit_inference:
        try:
            payload = json.loads(payload_text)
        except Exception as exc:
            st.error(f"Payload must be valid JSON: {exc}")
        else:
            if not isinstance(payload, dict):
                st.error("Inference payload must be a JSON object.")
            else:
                create_media_inference_result(
                    conn,
                    media_asset_id=asset_options[linked_asset],
                    job_id=job_id,
                    result_type=result_type,
                    payload=payload,
                    confidence=float(confidence),
                    source="manual",
                    model_name=model_name,
                    status="pending_review",
                )
                st.success("Advisory inference stored.")
                _rerun_app()

    if pending_rows:
        for row in pending_rows:
            with st.expander(f"Inference #{row['id']} · {row['resultType']} · {row['status']}"):
                st.json(row["payload"])
                action_cols = st.columns(3)
                corrected_text = st.text_area(
                    "Corrected payload (optional JSON)",
                    value="",
                    key=f"planner_inference_corrected_{job_id}_{row['id']}",
                )
                if action_cols[0].button("Accept", key=f"planner_accept_inference_{row['id']}"):
                    try:
                        corrected_payload = _parse_optional_json(corrected_text)
                    except Exception as exc:
                        st.error(f"Corrected payload must be valid JSON: {exc}")
                    else:
                        review_media_inference_result(
                            conn,
                            row["id"],
                            decision="corrected" if corrected_payload else "accepted",
                            reviewed_by="planner",
                            corrected_payload=corrected_payload,
                        )
                        st.success("Inference accepted.")
                        _rerun_app()
                if action_cols[1].button("Reject", key=f"planner_reject_inference_{row['id']}"):
                    review_media_inference_result(
                        conn,
                        row["id"],
                        decision="rejected",
                        reviewed_by="planner",
                    )
                    st.success("Inference rejected.")
                    _rerun_app()


def build_planner_preview_figure(
    *,
    selected_job: dict[str, Any] | None,
    corridor_rows: list[dict[str, Any]],
    filtered_df: pd.DataFrame,
) -> go.Figure | None:
    preview_points = pd.DataFrame()
    route_line: list[tuple[float, float]] = []

    if selected_job and all(
        selected_job.get(column) is not None
        for column in ("origin_lat", "origin_lon", "dest_lat", "dest_lon")
    ):
        preview_points = pd.DataFrame(
            [
                {
                    "label": "Origin",
                    "lat": float(selected_job["origin_lat"]),
                    "lon": float(selected_job["origin_lon"]),
                },
                {
                    "label": "Destination",
                    "lat": float(selected_job["dest_lat"]),
                    "lon": float(selected_job["dest_lon"]),
                },
            ]
        )
        route_geojson = selected_job.get("route_geojson")
        if isinstance(route_geojson, str) and route_geojson.strip():
            try:
                route_line = extract_route_path(route_geojson)
            except Exception:
                route_line = []
    elif corridor_rows:
        corridor_name = corridor_rows[0]["corridor"]
        scoped_df = filtered_df[
            filtered_df.get("corridor_display", pd.Series(dtype=str)).astype(str) == corridor_name
        ].copy()
        points: list[dict[str, Any]] = []
        for _, row in scoped_df.head(15).iterrows():
            if pd.notna(row.get("origin_lat")) and pd.notna(row.get("origin_lon")):
                points.append(
                    {
                        "label": f"{row.get('origin') or 'Origin'}",
                        "lat": float(row["origin_lat"]),
                        "lon": float(row["origin_lon"]),
                    }
                )
            if pd.notna(row.get("dest_lat")) and pd.notna(row.get("dest_lon")):
                points.append(
                    {
                        "label": f"{row.get('destination') or 'Destination'}",
                        "lat": float(row["dest_lat"]),
                        "lon": float(row["dest_lon"]),
                    }
                )
        if points:
            preview_points = pd.DataFrame(points)
            if "route_geojson" in scoped_df.columns:
                for _, candidate in scoped_df.iterrows():
                    route_geojson = candidate.get("route_geojson")
                    if isinstance(route_geojson, str) and route_geojson.strip():
                        try:
                            route_line = extract_route_path(route_geojson)
                        except Exception:
                            route_line = []
                        if route_line:
                            break

    if preview_points.empty:
        return None

    centre = {
        "lat": float(preview_points["lat"].mean()),
        "lon": float(preview_points["lon"].mean()),
    }
    figure = go.Figure()
    if route_line:
        figure.add_trace(
            go.Scattermap(
                lat=[point[0] for point in route_line],
                lon=[point[1] for point in route_line],
                mode="lines",
                line={"width": 4, "color": "#2f6fed"},
                name="Route preview",
                hovertemplate="Route preview<extra></extra>",
            )
        )
    figure.add_trace(
        go.Scattermap(
            lat=preview_points["lat"],
            lon=preview_points["lon"],
            mode="markers",
            marker={"size": 12, "color": "#c65d2e"},
            text=preview_points["label"],
            hovertemplate="%{text}<extra></extra>",
            name="Stops",
        )
    )
    figure.update_layout(
        **plotly_map_layout(centre, zoom=4, engine="map"),
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        height=360,
        showlegend=False,
    )
    return figure


def build_planner_site_points(
    *,
    selected_job: dict[str, Any] | None,
    corridor_rows: list[dict[str, Any]],
    filtered_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    if selected_job and all(
        selected_job.get(column) is not None
        for column in ("origin_lat", "origin_lon", "dest_lat", "dest_lon")
    ):
        origin_heading = _bearing_degrees(
            float(selected_job["origin_lat"]),
            float(selected_job["origin_lon"]),
            float(selected_job["dest_lat"]),
            float(selected_job["dest_lon"]),
        )
        destination_heading = _bearing_degrees(
            float(selected_job["dest_lat"]),
            float(selected_job["dest_lon"]),
            float(selected_job["origin_lat"]),
            float(selected_job["origin_lon"]),
        )
        points.extend(
            [
                {
                    "label": "Origin site",
                    "name": selected_job.get("origin_resolved") or selected_job.get("origin") or "Origin",
                    "lat": float(selected_job["origin_lat"]),
                    "lon": float(selected_job["origin_lon"]),
                    "streetViewUrl": google_street_view_static_url(
                        lat=float(selected_job["origin_lat"]),
                        lon=float(selected_job["origin_lon"]),
                        heading=origin_heading,
                    ),
                    "streetView360Url": google_street_view_360_url(
                        lat=float(selected_job["origin_lat"]),
                        lon=float(selected_job["origin_lon"]),
                        heading=origin_heading,
                    ),
                },
                {
                    "label": "Destination site",
                    "name": selected_job.get("destination_resolved") or selected_job.get("destination") or "Destination",
                    "lat": float(selected_job["dest_lat"]),
                    "lon": float(selected_job["dest_lon"]),
                    "streetViewUrl": google_street_view_static_url(
                        lat=float(selected_job["dest_lat"]),
                        lon=float(selected_job["dest_lon"]),
                        heading=destination_heading,
                    ),
                    "streetView360Url": google_street_view_360_url(
                        lat=float(selected_job["dest_lat"]),
                        lon=float(selected_job["dest_lon"]),
                        heading=destination_heading,
                    ),
                },
            ]
        )
        return points

    if corridor_rows:
        corridor_name = corridor_rows[0]["corridor"]
        scoped_df = filtered_df[
            filtered_df.get("corridor_display", pd.Series(dtype=str)).astype(str) == corridor_name
        ].copy()
        if not scoped_df.empty:
            row = scoped_df.iloc[0]
            if pd.notna(row.get("origin_lat")) and pd.notna(row.get("origin_lon")) and pd.notna(row.get("dest_lat")) and pd.notna(row.get("dest_lon")):
                origin_lat = float(row["origin_lat"])
                origin_lon = float(row["origin_lon"])
                dest_lat = float(row["dest_lat"])
                dest_lon = float(row["dest_lon"])
                points.extend(
                    [
                        {
                            "label": "Origin site",
                            "name": row.get("origin") or "Origin",
                            "lat": origin_lat,
                            "lon": origin_lon,
                            "streetViewUrl": google_street_view_static_url(
                                lat=origin_lat,
                                lon=origin_lon,
                                heading=_bearing_degrees(origin_lat, origin_lon, dest_lat, dest_lon),
                            ),
                            "streetView360Url": google_street_view_360_url(
                                lat=origin_lat,
                                lon=origin_lon,
                                heading=_bearing_degrees(origin_lat, origin_lon, dest_lat, dest_lon),
                            ),
                        },
                        {
                            "label": "Destination site",
                            "name": row.get("destination") or "Destination",
                            "lat": dest_lat,
                            "lon": dest_lon,
                            "streetViewUrl": google_street_view_static_url(
                                lat=dest_lat,
                                lon=dest_lon,
                                heading=_bearing_degrees(dest_lat, dest_lon, origin_lat, origin_lon),
                            ),
                            "streetView360Url": google_street_view_360_url(
                                lat=dest_lat,
                                lon=dest_lon,
                                heading=_bearing_degrees(dest_lat, dest_lon, origin_lat, origin_lon),
                            ),
                        },
                    ]
                )
    return points


def _bearing_degrees(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)
    x = math.sin(delta_lon) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lon)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360.0) % 360.0


def _format_float(value: Any) -> str:
    if value in (None, ""):
        return "n/a"
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return str(value)


def _format_volume_metric(payload: Any) -> str:
    if not isinstance(payload, dict):
        return "n/a"
    value = payload.get("estimated_m3")
    try:
        return f"{float(value):.1f} m3"
    except (TypeError, ValueError):
        return "n/a"


def _parse_optional_json(value: str) -> dict[str, Any] | None:
    text = value.strip()
    if not text:
        return None
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object.")
    return payload


def _rerun_app() -> None:
    rerun = getattr(st, "rerun", None)
    if callable(rerun):
        rerun()
        return

    experimental_rerun = getattr(st, "experimental_rerun", None)
    if callable(experimental_rerun):
        experimental_rerun()
        return

    raise RuntimeError("Streamlit rerun API is unavailable.")
