"""Heuristic planner scaffolding built on corridor, routing, and resource context."""
from __future__ import annotations

import os
import sqlite3
from collections import Counter
from typing import Any, Mapping, Sequence

import pandas as pd

from analytics.operational_signals import compute_route_spare_capacity_signal
from analytics.operations_assignment import (
    list_operational_readiness_items,
    list_planned_labor_assignments,
)
from analytics.db.site_media import accepted_site_context


def list_planner_corridor_candidates(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty or "corridor_display" not in df.columns:
        return []
    working = df.copy()
    numeric_columns = [
        "margin_per_m3_pct",
        "price_per_m3",
        "distance_km",
        "m3",
        "volume_m3",
    ]
    for column in numeric_columns:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
    payload: list[dict[str, Any]] = []
    grouped = working.groupby("corridor_display", dropna=False)
    for corridor, group in grouped:
        corridor_label = str(corridor or "Unknown corridor")
        job_count = int(len(group))
        overlap_score = float(job_count)
        familiarity = min(job_count / 10.0, 1.0)
        profitable_share = None
        if "margin_per_m3_pct" in group.columns:
            profitable_share = float((group["margin_per_m3_pct"] > 0).mean())
        avg_margin_pct = (
            float(group["margin_per_m3_pct"].mean())
            if "margin_per_m3_pct" in group.columns and group["margin_per_m3_pct"].notna().any()
            else None
        )
        avg_distance_km = (
            float(group["distance_km"].mean())
            if "distance_km" in group.columns and group["distance_km"].notna().any()
            else None
        )
        avg_price_per_m3 = (
            float(group["price_per_m3"].mean())
            if "price_per_m3" in group.columns and group["price_per_m3"].notna().any()
            else None
        )
        avg_volume_m3 = (
            float(group["volume_m3"].mean())
            if "volume_m3" in group.columns and group["volume_m3"].notna().any()
            else float(group["m3"].mean())
            if "m3" in group.columns and group["m3"].notna().any()
            else None
        )
        score = overlap_score * 0.5 + familiarity * 30.0
        if avg_margin_pct is not None:
            score += max(avg_margin_pct, -50.0) * 0.2
        payload.append(
            {
                "corridor": corridor_label,
                "jobCount": job_count,
                "overlapScore": round(overlap_score, 2),
                "familiarityScore": round(familiarity, 2),
                "avgMarginPct": round(avg_margin_pct, 2) if avg_margin_pct is not None else None,
                "profitableShare": round(profitable_share, 2) if profitable_share is not None else None,
                "avgDistanceKm": round(avg_distance_km, 1) if avg_distance_km is not None else None,
                "avgPricePerM3": round(avg_price_per_m3, 2) if avg_price_per_m3 is not None else None,
                "avgVolumeM3": round(avg_volume_m3, 1) if avg_volume_m3 is not None else None,
                "origin": _mode(group, ["origin", "origin_resolved", "origin_city"]),
                "destination": _mode(group, ["destination", "destination_resolved", "destination_city"]),
                "candidateScore": round(score, 2),
            }
        )
    return sorted(
        payload,
        key=lambda item: (-float(item["candidateScore"]), -int(item["jobCount"]), item["corridor"]),
    )


def infer_planner_corridor_for_job(
    df: pd.DataFrame,
    *,
    origin: str | None,
    destination: str | None,
) -> str | None:
    candidates = list_planner_corridor_candidates(df)
    if not candidates:
        return None
    origin_key = _norm(origin)
    destination_key = _norm(destination)
    if not origin_key and not destination_key:
        return candidates[0]["corridor"]

    best: tuple[float, str] | None = None
    for candidate in candidates:
        score = 0.0
        score += _match_score(origin_key, candidate.get("origin"))
        score += _match_score(destination_key, candidate.get("destination"))
        score += float(candidate.get("candidateScore") or 0.0) * 0.01
        if best is None or score > best[0]:
            best = (score, str(candidate["corridor"]))
    if best and best[0] > 0:
        return best[1]
    return candidates[0]["corridor"]


def build_planner_proposal(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    *,
    selection_mode: str,
    corridor: str | None = None,
    preferred_template: str | None = None,
    job_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selected_job = dict(job_context) if job_context else None
    resolved_corridor = corridor
    if selection_mode == "job_first" and selected_job:
        resolved_corridor = resolved_corridor or infer_planner_corridor_for_job(
            df,
            origin=selected_job.get("origin") or selected_job.get("origin_resolved"),
            destination=selected_job.get("destination") or selected_job.get("destination_resolved"),
        )
    candidates = list_planner_corridor_candidates(df)
    if not resolved_corridor and candidates:
        resolved_corridor = str(candidates[0]["corridor"])
    if not resolved_corridor:
        return {
            "selectionMode": selection_mode,
            "selection": {"corridor": corridor},
            "jobContext": _job_context_payload(selected_job),
            "candidateRows": [],
            "routingContext": _empty_routing_context(),
            "resourceContext": _empty_resource_context(),
            "siteContext": {"assessments": [], "mediaAssets": [], "acceptedVolumeEstimate": None, "acceptedSiteFeatures": [], "acceptedDetections": []},
            "draftLegs": [],
            "template": preferred_template or "single_leg",
            "warnings": ["No corridor history is available for this selection."],
            "explainability": {
                "history": ["No corridor history is available for this selection."],
                "commercial": [],
                "routing": ["Routing context is unavailable without corridor history."],
                "resources": ["Resource-fit context is unavailable without a resolved lane."],
                "site": ["No linked site assessments or reviewed media outputs are available."],
            },
        }

    scoped = df[df["corridor_display"].astype(str) == resolved_corridor].copy() if "corridor_display" in df.columns else df.iloc[0:0].copy()
    scoped_candidates = list_planner_corridor_candidates(scoped) if not scoped.empty else []
    best = scoped_candidates[0] if scoped_candidates else _fallback_candidate(resolved_corridor, selected_job)
    avg_distance = best.get("avgDistanceKm")
    template = preferred_template or (
        "pickup_linehaul_delivery" if avg_distance and avg_distance >= 80 else "single_leg"
    )
    origin_value = (
        _first_text(
            selected_job,
            ["origin_resolved", "origin"],
        )
        or str(best.get("origin") or "Origin site")
    )
    destination_value = (
        _first_text(
            selected_job,
            ["destination_resolved", "destination"],
        )
        or str(best.get("destination") or "Destination site")
    )
    routing_context = _build_routing_context(scoped, selected_job=selected_job, candidate=best)
    resource_context = _build_resource_context(
        conn,
        origin=origin_value,
        destination=destination_value,
        job_context=selected_job,
    )
    site_context = accepted_site_context(
        conn,
        job_id=int(selected_job["id"]) if selected_job and selected_job.get("id") is not None else None,
    )
    site_constraints = _derive_site_constraints(
        site_context=site_context,
        planned_volume_m3=_as_float((selected_job or {}).get("volume_m3") if selected_job else None),
    )
    site_context = {**site_context, "derivedConstraints": site_constraints}
    draft_legs = _draft_legs_for_template(
        template,
        origin=origin_value,
        destination=destination_value,
        avg_margin_pct=best.get("avgMarginPct"),
        route_label=routing_context["routePattern"],
    )
    warnings = _build_warnings(
        selection_mode=selection_mode,
        selected_job=selected_job,
        resolved_corridor=resolved_corridor,
        scoped=scoped,
        routing_context=routing_context,
        resource_context=resource_context,
        site_context=site_context,
    )
    explainability = {
        "history": [
            f"Selected corridor: {resolved_corridor}.",
            f"Historical overlap score: {best.get('overlapScore', 0)} from {best.get('jobCount', 0)} matching jobs.",
            f"Corridor familiarity score: {best.get('familiarityScore', 0)}.",
        ],
        "commercial": [],
        "routing": [],
        "resources": [],
    }
    if best.get("avgMarginPct") is not None:
        explainability["commercial"].append(
            f"Average historical margin % on this corridor: {best['avgMarginPct']}."
        )
    if best.get("avgPricePerM3") is not None:
        explainability["commercial"].append(
            f"Average historical price per m3: {best['avgPricePerM3']}."
        )
    explainability["routing"].append(
        f"Routing provider: {routing_context['routingProvider']}; geometry coverage: {routing_context['geometryCoveragePct']}%."
    )
    explainability["routing"].append(
        f"Route pattern classified as {routing_context['routePattern']} from {routing_context['avgDistanceKm']} km average distance."
    )
    explainability["routing"].append(
        "Live traffic weighting is not yet modeled; routing context is based on stored geometry and distance history."
    )
    explainability["resources"].append(
        f"Spare-capacity signal: {resource_context['spareCapacityLabel']} ({resource_context['spareCapacityScore']})."
    )
    explainability["resources"].append(
        f"Blocked resources: {resource_context['blockedVehicles']} vehicles, {resource_context['blockedWorkers']} workers."
    )
    if resource_context["plannedLaborAssignments"] is not None:
        explainability["resources"].append(
            f"Planned labor assignments on the selected day: {resource_context['plannedLaborAssignments']}."
        )
    explainability["site"] = _site_explainability(site_context)
    return {
        "selectionMode": selection_mode,
        "selection": {
            "corridor": resolved_corridor,
            "origin": origin_value,
            "destination": destination_value,
        },
        "jobContext": _job_context_payload(selected_job),
        "candidateRows": scoped_candidates or [best],
        "routingContext": routing_context,
        "resourceContext": resource_context,
        "siteContext": site_context,
        "draftLegs": draft_legs,
        "template": template,
        "warnings": warnings,
        "explainability": explainability,
    }


def _job_context_payload(job_context: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not job_context:
        return None
    return {
        "jobId": job_context.get("id"),
        "client": job_context.get("client"),
        "origin": _first_text(job_context, ["origin_resolved", "origin"]),
        "destination": _first_text(job_context, ["destination_resolved", "destination"]),
        "jobDate": job_context.get("job_date"),
        "distanceKm": _as_float(job_context.get("distance_km")),
        "hasRouteGeometry": _has_geometry(job_context.get("route_geojson")),
    }


def _build_routing_context(
    scoped: pd.DataFrame,
    *,
    selected_job: Mapping[str, Any] | None,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    geometry_column = _geometry_column(scoped)
    geometry_coverage_pct = 0.0
    if geometry_column and not scoped.empty:
        geometry_coverage_pct = round(float(scoped[geometry_column].apply(_has_geometry).mean() * 100.0), 1)
    avg_distance = _as_float(candidate.get("avgDistanceKm")) or _as_float(
        selected_job.get("distance_km") if selected_job else None
    ) or 0.0
    if avg_distance >= 800:
        route_pattern = "linehaul"
    elif avg_distance >= 120:
        route_pattern = "regional"
    else:
        route_pattern = "metro"
    return {
        "routingProvider": os.environ.get("ROUTING_PROVIDER", "ors"),
        "geometryCoveragePct": geometry_coverage_pct,
        "selectedJobHasGeometry": bool(selected_job and _has_geometry(selected_job.get("route_geojson"))),
        "selectedJobHasCoordinates": bool(
            selected_job
            and all(
                _as_float(selected_job.get(column)) is not None
                for column in ("origin_lat", "origin_lon", "dest_lat", "dest_lon")
            )
        ),
        "avgDistanceKm": round(avg_distance, 1),
        "routePattern": route_pattern,
    }


def _build_resource_context(
    conn: sqlite3.Connection,
    *,
    origin: str | None,
    destination: str | None,
    job_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    spare_signal = compute_route_spare_capacity_signal(
        conn,
        origin=origin,
        destination=destination,
        estimated_volume_m3=_as_float(
            (job_context or {}).get("volume_m3") or (job_context or {}).get("volume")
        ),
    )
    readiness_items = list_operational_readiness_items(conn)
    blocked_vehicles = len(
        {item["resourceId"] for item in readiness_items if item["resourceType"] == "vehicle" and item["status"] == "blocked"}
    )
    blocked_workers = len(
        {item["resourceId"] for item in readiness_items if item["resourceType"] == "worker" and item["status"] == "blocked"}
    )
    warning_vehicles = len(
        {item["resourceId"] for item in readiness_items if item["resourceType"] == "vehicle" and item["status"] == "warning"}
    )
    warning_workers = len(
        {item["resourceId"] for item in readiness_items if item["resourceType"] == "worker" and item["status"] == "warning"}
    )
    active_truck_row = conn.execute(
        "SELECT COUNT(*) AS count FROM trucks WHERE active = 1"
    ).fetchone()
    active_worker_row = conn.execute(
        "SELECT COUNT(*) AS count FROM workers WHERE active = 1"
    ).fetchone()
    planned_date = _job_date_only(job_context.get("job_date") if job_context else None)
    planned_assignments = (
        list_planned_labor_assignments(conn, start_date=planned_date, end_date=planned_date)
        if planned_date
        else []
    )
    return {
        "spareCapacityScore": float(spare_signal["score"]),
        "spareCapacityLabel": str(spare_signal["label"]),
        "matchingSpareTrucks": int(spare_signal["matchingSpareTrucks"]),
        "destinationSpareTrucks": int(spare_signal["destinationSpareTrucks"]),
        "activeSignalTrucks": int(spare_signal["activeTrucks"]),
        "activeFleetTrucks": int(active_truck_row["count"] if active_truck_row is not None else 0),
        "activeWorkers": int(active_worker_row["count"] if active_worker_row is not None else 0),
        "blockedVehicles": blocked_vehicles,
        "blockedWorkers": blocked_workers,
        "warningVehicles": warning_vehicles,
        "warningWorkers": warning_workers,
        "plannedLaborAssignments": len(planned_assignments) if planned_date else None,
        "plannedDate": planned_date,
    }


def _build_warnings(
    *,
    selection_mode: str,
    selected_job: Mapping[str, Any] | None,
    resolved_corridor: str,
    scoped: pd.DataFrame,
    routing_context: Mapping[str, Any],
    resource_context: Mapping[str, Any],
    site_context: Mapping[str, Any],
) -> list[str]:
    warnings: list[str] = []
    if selection_mode == "job_first" and not selected_job:
        warnings.append("No job is selected for job-first planning.")
    if scoped.empty:
        warnings.append(f"No corridor history matched {resolved_corridor}.")
    if float(routing_context.get("geometryCoveragePct") or 0.0) <= 0:
        warnings.append("Stored route geometry is missing for this corridor; routing context is distance-based only.")
    if not bool(routing_context.get("selectedJobHasCoordinates", True)) and selected_job:
        warnings.append("Selected job is missing route coordinates, so map context is limited.")
    if float(resource_context.get("spareCapacityScore") or 0.0) < 55.0:
        warnings.append("Resource-fit is constrained on this lane based on current spare-capacity signals.")
    if int(resource_context.get("blockedVehicles") or 0) > 0 or int(resource_context.get("blockedWorkers") or 0) > 0:
        warnings.append("There are blocked resources in the current operations pool.")
    for assessment in site_context.get("assessments", []):
        if assessment.get("largeVehicleSuitability") == "unsuitable":
            warnings.append(f"{assessment['siteKind'].capitalize()} site is marked unsuitable for large vehicles.")
        high_flags = [
            label
            for label, field in (
                ("loading access", "loadingAccessRisk"),
                ("parking", "parkingRisk"),
                ("narrow street", "narrowStreetRisk"),
                ("stairs", "stairsRisk"),
                ("clearance", "clearanceRisk"),
            )
            if assessment.get(field) == "high"
        ]
        if high_flags:
            warnings.append(
                f"{assessment['siteKind'].capitalize()} site has high-risk constraints: {', '.join(high_flags)}."
            )
        if assessment.get("uncertaintyFlag"):
            warnings.append(f"{assessment['siteKind'].capitalize()} site assessment is still marked uncertain.")
    accepted_volume = site_context.get("acceptedVolumeEstimate") or {}
    payload = accepted_volume.get("payload") if isinstance(accepted_volume, Mapping) else None
    estimated_volume = _as_float((payload or {}).get("estimated_m3") if isinstance(payload, Mapping) else None)
    planned_volume = _as_float((selected_job or {}).get("volume_m3") if selected_job else None)
    if estimated_volume is not None and planned_volume is not None and estimated_volume > planned_volume * 1.15:
        warnings.append(
            f"Accepted visual volume estimate ({estimated_volume:.1f} m3) materially exceeds planned job volume ({planned_volume:.1f} m3)."
        )
    derived_constraints = site_context.get("derivedConstraints") or {}
    if derived_constraints.get("truckUnsuitable"):
        warnings.append("Accepted site evidence indicates the standard truck plan is unsuitable.")
    if derived_constraints.get("shuttleRecommended"):
        warnings.append("Accepted site evidence suggests a shuttle or smaller-vehicle leg is required.")
    labor_uplift_pct = int(derived_constraints.get("laborUpliftPct") or 0)
    if labor_uplift_pct > 0:
        warnings.append(f"Accepted site evidence implies a labor uplift of about {labor_uplift_pct}%.")
    access_time_uplift = int(derived_constraints.get("accessTimeUpliftMinutes") or 0)
    if access_time_uplift > 0:
        warnings.append(f"Accepted site evidence implies roughly {access_time_uplift} extra access/load minutes.")
    loading_delay_risk = str(derived_constraints.get("loadingDelayRisk") or "")
    if loading_delay_risk in {"medium", "high"}:
        warnings.append(f"Accepted site evidence indicates {loading_delay_risk}-risk loading delays.")
    return warnings


def _site_explainability(site_context: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    assessments = site_context.get("assessments", []) or []
    if assessments:
        lines.append(f"Accepted site assessments available for {len(assessments)} site(s).")
    volume_row = site_context.get("acceptedVolumeEstimate") or {}
    payload = volume_row.get("payload") if isinstance(volume_row, Mapping) else None
    if isinstance(payload, Mapping) and payload.get("estimated_m3") is not None:
        lines.append(f"Accepted visual volume estimate: {float(payload['estimated_m3']):.1f} m3.")
    feature_rows = site_context.get("acceptedSiteFeatures", []) or []
    if feature_rows:
        lines.append(f"Accepted site-feature inference rows: {len(feature_rows)}.")
    media_assets = site_context.get("mediaAssets", []) or []
    if media_assets:
        lines.append(f"Linked site media assets: {len(media_assets)}.")
    derived_constraints = site_context.get("derivedConstraints") or {}
    if derived_constraints:
        if derived_constraints.get("truckUnsuitable"):
            lines.append("Derived constraint: current large-vehicle plan is unsuitable.")
        if derived_constraints.get("shuttleRecommended"):
            lines.append("Derived constraint: add a shuttle/smaller-vehicle leg.")
        if int(derived_constraints.get("laborUpliftPct") or 0) > 0:
            lines.append(f"Derived labor uplift: {int(derived_constraints['laborUpliftPct'])}%.")
        if int(derived_constraints.get("accessTimeUpliftMinutes") or 0) > 0:
            lines.append(
                f"Derived access/load time uplift: {int(derived_constraints['accessTimeUpliftMinutes'])} minutes."
            )
        if derived_constraints.get("loadingDelayRisk"):
            lines.append(f"Derived loading-delay risk: {derived_constraints['loadingDelayRisk']}.")
    if not lines:
        lines.append("No accepted site assessments or reviewed media outputs are linked yet.")
    return lines


def _derive_site_constraints(
    *,
    site_context: Mapping[str, Any],
    planned_volume_m3: float | None,
) -> dict[str, Any]:
    assessments = site_context.get("assessments", []) or []
    accepted_volume = site_context.get("acceptedVolumeEstimate") or {}
    payload = accepted_volume.get("payload") if isinstance(accepted_volume, Mapping) else None
    estimated_volume = _as_float((payload or {}).get("estimated_m3") if isinstance(payload, Mapping) else None)

    truck_unsuitable = False
    shuttle_recommended = False
    labor_uplift_pct = 0
    access_time_uplift_minutes = 0
    loading_delay_risk = "low"
    review_needed = False
    reasons: list[str] = []

    for assessment in assessments:
        site_kind = str(assessment.get("siteKind") or "site")
        if assessment.get("largeVehicleSuitability") == "unsuitable":
            truck_unsuitable = True
            shuttle_recommended = True
            access_time_uplift_minutes = max(access_time_uplift_minutes, 30)
            loading_delay_risk = "high"
            reasons.append(f"{site_kind} site marked unsuitable for large vehicles")
        elif assessment.get("largeVehicleSuitability") == "restricted":
            shuttle_recommended = True
            access_time_uplift_minutes = max(access_time_uplift_minutes, 20)
            loading_delay_risk = "high" if loading_delay_risk == "low" else loading_delay_risk
            reasons.append(f"{site_kind} site marked restricted for large vehicles")

        if assessment.get("loadingAccessRisk") == "high":
            labor_uplift_pct = max(labor_uplift_pct, 20)
            access_time_uplift_minutes = max(access_time_uplift_minutes, 20)
            loading_delay_risk = "high"
            reasons.append(f"{site_kind} loading access risk is high")
        elif assessment.get("loadingAccessRisk") == "medium":
            labor_uplift_pct = max(labor_uplift_pct, 10)
            access_time_uplift_minutes = max(access_time_uplift_minutes, 10)
            if loading_delay_risk == "low":
                loading_delay_risk = "medium"

        if assessment.get("parkingRisk") == "high":
            access_time_uplift_minutes = max(access_time_uplift_minutes, 20)
            loading_delay_risk = "high"
            reasons.append(f"{site_kind} parking risk is high")
        elif assessment.get("parkingRisk") == "medium" and loading_delay_risk == "low":
            loading_delay_risk = "medium"

        if assessment.get("narrowStreetRisk") == "high":
            shuttle_recommended = True
            access_time_uplift_minutes = max(access_time_uplift_minutes, 25)
            loading_delay_risk = "high"
            reasons.append(f"{site_kind} narrow-street risk is high")

        if assessment.get("stairsRisk") == "high":
            labor_uplift_pct = max(labor_uplift_pct, 25)
            access_time_uplift_minutes = max(access_time_uplift_minutes, 20)
            reasons.append(f"{site_kind} stairs risk is high")
        elif assessment.get("stairsRisk") == "medium":
            labor_uplift_pct = max(labor_uplift_pct, 10)

        if assessment.get("clearanceRisk") == "high":
            truck_unsuitable = True
            shuttle_recommended = True
            loading_delay_risk = "high"
            reasons.append(f"{site_kind} clearance risk is high")
        elif assessment.get("clearanceRisk") == "medium":
            shuttle_recommended = shuttle_recommended or False
            if loading_delay_risk == "low":
                loading_delay_risk = "medium"

        if assessment.get("uncertaintyFlag"):
            review_needed = True

    if estimated_volume is not None and planned_volume_m3 is not None:
        if estimated_volume > planned_volume_m3 * 1.25:
            labor_uplift_pct = max(labor_uplift_pct, 15)
            access_time_uplift_minutes = max(access_time_uplift_minutes, 15)
            reasons.append("accepted visual volume estimate materially exceeds planned volume")

    return {
        "truckUnsuitable": truck_unsuitable,
        "shuttleRecommended": shuttle_recommended,
        "laborUpliftPct": labor_uplift_pct,
        "accessTimeUpliftMinutes": access_time_uplift_minutes,
        "loadingDelayRisk": loading_delay_risk,
        "reviewNeeded": review_needed,
        "reasons": reasons,
    }


def _fallback_candidate(corridor: str, selected_job: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        "corridor": corridor,
        "origin": _first_text(selected_job, ["origin_resolved", "origin"]) if selected_job else None,
        "destination": _first_text(selected_job, ["destination_resolved", "destination"]) if selected_job else None,
        "avgDistanceKm": _as_float((selected_job or {}).get("distance_km")),
        "avgMarginPct": None,
        "avgPricePerM3": None,
        "jobCount": 0,
        "familiarityScore": 0.0,
        "overlapScore": 0.0,
    }


def _empty_routing_context() -> dict[str, Any]:
    return {
        "routingProvider": os.environ.get("ROUTING_PROVIDER", "ors"),
        "geometryCoveragePct": 0.0,
        "selectedJobHasGeometry": False,
        "selectedJobHasCoordinates": False,
        "avgDistanceKm": 0.0,
        "routePattern": "unknown",
    }


def _empty_resource_context() -> dict[str, Any]:
    return {
        "spareCapacityScore": 50.0,
        "spareCapacityLabel": "neutral",
        "matchingSpareTrucks": 0,
        "destinationSpareTrucks": 0,
        "activeSignalTrucks": 0,
        "activeFleetTrucks": 0,
        "activeWorkers": 0,
        "blockedVehicles": 0,
        "blockedWorkers": 0,
        "warningVehicles": 0,
        "warningWorkers": 0,
        "plannedLaborAssignments": None,
        "plannedDate": None,
    }


def _geometry_column(df: pd.DataFrame) -> str | None:
    for column in ("route_geojson", "route_geometry", "geojson"):
        if column in df.columns:
            return column
    return None


def _has_geometry(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text and text not in {"{}", "[]", "null", "None"})


def _norm(value: str | None) -> str:
    return (value or "").strip().lower()


def _match_score(job_value: str, candidate_value: Any) -> float:
    candidate_key = _norm(str(candidate_value or ""))
    if not job_value or not candidate_key:
        return 0.0
    if job_value == candidate_key:
        return 10.0
    if job_value in candidate_key or candidate_key in job_value:
        return 4.0
    return 0.0


def _first_text(payload: Mapping[str, Any] | None, keys: Sequence[str]) -> str | None:
    if not payload:
        return None
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _mode(df: pd.DataFrame, columns: Sequence[str]) -> str | None:
    for column in columns:
        if column in df.columns:
            values = [str(value).strip() for value in df[column].dropna().tolist() if str(value).strip()]
            if values:
                return Counter(values).most_common(1)[0][0]
    return None


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _job_date_only(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:10]


def _draft_legs_for_template(
    template: str,
    *,
    origin: str,
    destination: str,
    avg_margin_pct: float | None,
    route_label: str,
) -> list[dict[str, Any]]:
    commercial_note = (
        f"Estimated corridor margin context: {avg_margin_pct:.2f}%"
        if avg_margin_pct is not None
        else "Margin context unavailable"
    )
    routing_note = f"Routing context currently classifies this leg as {route_label}."
    if template == "pickup_linehaul_delivery":
        return [
            {
                "sequence": 1,
                "legType": "pickup",
                "from": origin,
                "to": origin,
                "note": f"Origin handling and load preparation. {commercial_note}",
            },
            {
                "sequence": 2,
                "legType": "linehaul",
                "from": origin,
                "to": destination,
                "note": f"Primary corridor movement shaped by historical overlap and route familiarity. {routing_note}",
            },
            {
                "sequence": 3,
                "legType": "delivery",
                "from": destination,
                "to": destination,
                "note": "Destination handling and unload.",
            },
        ]
    if template == "staging_linehaul_delivery":
        staging = f"{origin} staging"
        return [
            {
                "sequence": 1,
                "legType": "staging",
                "from": origin,
                "to": staging,
                "note": "Intermediate staging or cross-dock preparation.",
            },
            {
                "sequence": 2,
                "legType": "linehaul",
                "from": staging,
                "to": destination,
                "note": f"Primary corridor movement. {commercial_note} {routing_note}",
            },
            {
                "sequence": 3,
                "legType": "delivery",
                "from": destination,
                "to": destination,
                "note": "Final delivery and site completion.",
            },
        ]
    return [
        {
            "sequence": 1,
            "legType": "service_leg",
            "from": origin,
            "to": destination,
            "note": f"Single-leg operational plan. {commercial_note} {routing_note}",
        },
    ]


__all__ = [
    "build_planner_proposal",
    "infer_planner_corridor_for_job",
    "list_planner_corridor_candidates",
]
