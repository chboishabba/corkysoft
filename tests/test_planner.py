from __future__ import annotations

import sqlite3
import os

import pandas as pd

from analytics.db import ensure_dashboard_tables, upsert_truck, upsert_vehicle_details, upsert_worker
from analytics.db.site_media import create_media_inference_result, upsert_site_assessment
from analytics.operations_assignment import assign_segment_resources, ensure_segment
from analytics.planner import (
    build_planner_day_view,
    build_planner_proposal,
    infer_planner_corridor_for_job,
    list_planner_corridor_candidates,
)
from dashboard.components.planner import build_planner_preview_figure
from dashboard.components.planner import build_planner_site_points


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "corridor_display": "Brisbane → Cairns",
                "lane_assignment_status": "assigned",
                "lane_key": "postcode:4000->postcode:4870",
                "corridor_group_key": "postcode:4000<->postcode:4870",
                "origin": "Brisbane",
                "destination": "Cairns",
                "margin_per_m3_pct": 22.0,
                "distance_km": 1700.0,
                "price_per_m3": 210.0,
                "volume_m3": 38.0,
                "origin_lat": -27.47,
                "origin_lon": 153.02,
                "dest_lat": -16.92,
                "dest_lon": 145.77,
                "route_geojson": "{\"type\":\"FeatureCollection\",\"features\":[]}",
            },
            {
                "corridor_display": "Brisbane → Cairns",
                "lane_assignment_status": "assigned",
                "lane_key": "postcode:4000->postcode:4870",
                "corridor_group_key": "postcode:4000<->postcode:4870",
                "origin": "Brisbane",
                "destination": "Cairns",
                "margin_per_m3_pct": 18.0,
                "distance_km": 1680.0,
                "price_per_m3": 205.0,
                "volume_m3": 34.0,
                "origin_lat": -27.47,
                "origin_lon": 153.02,
                "dest_lat": -16.92,
                "dest_lon": 145.77,
                "route_geojson": "{\"type\":\"FeatureCollection\",\"features\":[]}",
            },
            {
                "corridor_display": "Brisbane → Sydney",
                "lane_assignment_status": "assigned",
                "lane_key": "postcode:4000->postcode:2000",
                "corridor_group_key": "postcode:2000<->postcode:4000",
                "origin": "Brisbane",
                "destination": "Sydney",
                "margin_per_m3_pct": 5.0,
                "distance_km": 900.0,
                "price_per_m3": 160.0,
                "volume_m3": 18.0,
                "origin_lat": -27.47,
                "origin_lon": 153.02,
                "dest_lat": -33.87,
                "dest_lon": 151.21,
                "route_geojson": None,
            },
        ]
    )


def _seed_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)
    upsert_truck(conn, truck_id="TRK-1", name="Truck 1", capacity_m3=50.0)
    upsert_worker(conn, name="Alex Planner")
    job_id = conn.execute(
        """
        INSERT INTO jobs (
            client,
            origin,
            destination,
            origin_resolved,
            destination_resolved,
            job_date,
            distance_km,
            volume_m3,
            origin_lat,
            origin_lon,
            dest_lat,
            dest_lon,
            route_geojson,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "Kent",
            "Brisbane",
            "Cairns",
            "Brisbane",
            "Cairns",
            "2026-03-20",
            1700.0,
            36.0,
            -27.47,
            153.02,
            -16.92,
            145.77,
            "{\"type\":\"FeatureCollection\",\"features\":[]}",
            "2026-03-12T00:00:00+00:00",
        ),
    ).lastrowid
    conn.execute(
        """
        INSERT INTO shipments (
            job_id,
            truck_id,
            quantity,
            from_location,
            to_location,
            status,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(job_id),
            "TRK-1",
            5.0,
            "Brisbane",
            "Cairns",
            "planned",
            "2026-03-12T00:00:00+00:00",
        ),
    )
    conn.commit()
    return conn


def test_list_planner_corridor_candidates_ranks_by_score() -> None:
    rows = list_planner_corridor_candidates(_frame())
    assert rows[0]["corridor"] == "Brisbane → Cairns"
    assert rows[0]["laneKey"] == "postcode:4000->postcode:4870"
    assert rows[0]["jobCount"] == 2


def test_infer_planner_corridor_for_job_prefers_matching_lane() -> None:
    corridor = infer_planner_corridor_for_job(
        _frame(),
        origin="Brisbane",
        destination="Cairns",
    )
    assert corridor == "Brisbane → Cairns"


def test_build_planner_proposal_job_first_includes_routing_and_resource_context() -> None:
    conn = _seed_conn()
    job = dict(
        conn.execute(
            """
            SELECT id, client, origin, destination, origin_resolved, destination_resolved,
                   job_date, distance_km, volume_m3, origin_lat, origin_lon, dest_lat, dest_lon, route_geojson
            FROM jobs
            LIMIT 1
            """
        ).fetchone()
    )

    proposal = build_planner_proposal(
        conn,
        _frame(),
        selection_mode="job_first",
        preferred_template="pickup_linehaul_delivery",
        job_context=job,
    )

    assert proposal["selectionMode"] == "job_first"
    assert proposal["jobContext"]["jobId"] == job["id"]
    assert proposal["selection"]["corridor"] == "Brisbane → Cairns"
    assert proposal["routingContext"]["selectedJobHasGeometry"] is True
    assert proposal["resourceContext"]["matchingSpareTrucks"] == 1
    assert proposal["draftLegs"]
    assert proposal["explainability"]["routing"]
    assert proposal["explainability"]["resources"]


def test_build_planner_proposal_corridor_first_warns_when_geometry_is_missing() -> None:
    conn = _seed_conn()

    proposal = build_planner_proposal(
        conn,
        _frame(),
        selection_mode="corridor_first",
        corridor="Brisbane → Sydney",
        preferred_template="single_leg",
    )

    assert proposal["selectionMode"] == "corridor_first"
    assert proposal["selection"]["corridor"] == "Brisbane → Sydney"
    assert any("geometry" in warning.lower() for warning in proposal["warnings"])


def test_build_planner_proposal_warns_when_resources_are_blocked() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)
    upsert_truck(conn, truck_id="TRK-BLOCKED", name="Truck blocked", capacity_m3=50.0)
    upsert_vehicle_details(
        conn,
        truck_id="TRK-BLOCKED",
        rego="TRK-BLOCKED",
        rego_expiry="2000-01-01",
        coi_due="2099-12-31",
        next_service="2099-12-31",
        daily_check_complete=True,
    )
    job_id = conn.execute(
        """
        INSERT INTO jobs (
            client, origin, destination, origin_resolved, destination_resolved, job_date,
            distance_km, volume_m3, origin_lat, origin_lon, dest_lat, dest_lon, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "Kent",
            "Brisbane",
            "Cairns",
            "Brisbane",
            "Cairns",
            "2026-03-20",
            1700.0,
            36.0,
            -27.47,
            153.02,
            -16.92,
            145.77,
            "2026-03-12T00:00:00+00:00",
        ),
    ).lastrowid
    job = dict(conn.execute("SELECT * FROM jobs WHERE id = ?", (int(job_id),)).fetchone())

    proposal = build_planner_proposal(
        conn,
        _frame(),
        selection_mode="job_first",
        preferred_template="pickup_linehaul_delivery",
        job_context=job,
    )

    assert proposal["resourceContext"]["blockedVehicles"] >= 1
    assert any("blocked" in warning.lower() or "resource" in warning.lower() for warning in proposal["warnings"])


def test_build_planner_day_view_filters_and_prioritises_focus_job() -> None:
    conn = _seed_conn()
    worker = conn.execute("SELECT id FROM workers LIMIT 1").fetchone()
    job = conn.execute("SELECT id FROM jobs LIMIT 1").fetchone()
    other_job_id = conn.execute(
        """
        INSERT INTO jobs (
            client, origin, destination, origin_resolved, destination_resolved, job_date,
            distance_km, volume_m3, origin_lat, origin_lon, dest_lat, dest_lon, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "Other Client",
            "Brisbane",
            "Sydney",
            "Brisbane",
            "Sydney",
            "2026-03-21",
            900.0,
            20.0,
            -27.47,
            153.02,
            -33.87,
            151.21,
            "2026-03-12T00:00:00+00:00",
        ),
    ).lastrowid
    focus_segment = ensure_segment(
        conn,
        job_id=int(job["id"]),
        segment_sequence=1,
        from_location="Brisbane",
        to_location="Cairns",
        planned_start="2026-03-20T08:00:00+00:00",
        planned_end="2026-03-20T12:00:00+00:00",
    )
    assign_segment_resources(
        conn,
        segment_id=int(focus_segment["id"]),
        truck_ids=["TRK-1"],
        worker_assignments=[{"workerId": int(worker["id"])}],
    )
    ensure_segment(
        conn,
        job_id=int(other_job_id),
        segment_sequence=1,
        from_location="Brisbane",
        to_location="Sydney",
        planned_start="2026-03-20T13:00:00+00:00",
        planned_end="2026-03-20T16:00:00+00:00",
    )
    ensure_segment(
        conn,
        job_id=int(other_job_id),
        segment_sequence=2,
        from_location="Sydney",
        to_location="Depot",
        planned_start="2026-03-21T08:00:00+00:00",
        planned_end="2026-03-21T10:00:00+00:00",
    )

    day_view = build_planner_day_view(
        conn,
        selected_date="2026-03-20",
        focus_job_id=int(job["id"]),
    )

    assert day_view["summary"]["segmentCount"] == 2
    assert day_view["summary"]["jobCount"] == 2
    assert day_view["summary"]["focusJobSegmentCount"] == 1
    assert day_view["summary"]["truckCount"] == 1
    assert day_view["summary"]["workerCount"] == 1
    assert day_view["segments"][0]["jobId"] == int(job["id"])
    assert day_view["segments"][0]["isFocusJob"] is True


def test_build_planner_preview_figure_uses_provider_aware_layout_and_route_line() -> None:
    previous_provider = os.environ.get("ROUTING_PROVIDER")
    previous_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    os.environ["ROUTING_PROVIDER"] = "google"
    os.environ["GOOGLE_MAPS_API_KEY"] = "test-key"
    try:
        figure = build_planner_preview_figure(
            selected_job={
                "origin_lat": -27.47,
                "origin_lon": 153.02,
                "dest_lat": -26.92,
                "dest_lon": 153.12,
                "route_geojson": (
                    '{"type":"Feature","geometry":{"type":"LineString","coordinates":'
                    '[[153.02,-27.47],[153.08,-27.20],[153.12,-26.92]]}}'
                ),
            },
            corridor_rows=[],
            filtered_df=pd.DataFrame(),
        )
    finally:
        if previous_provider is None:
            os.environ.pop("ROUTING_PROVIDER", None)
        else:
            os.environ["ROUTING_PROVIDER"] = previous_provider
        if previous_key is None:
            os.environ.pop("GOOGLE_MAPS_API_KEY", None)
        else:
            os.environ["GOOGLE_MAPS_API_KEY"] = previous_key

    assert figure is not None
    assert len(figure.data) == 2
    assert figure.data[0].mode == "lines"
    assert figure.data[1].mode == "markers"
    assert figure.layout.map.style == "white-bg"
    assert figure.layout.map.layers[0].source[0].startswith("https://mt1.google.com/vt/lyrs=m")


def test_build_planner_preview_figure_does_not_fallback_to_straight_line_without_geometry() -> None:
    figure = build_planner_preview_figure(
        selected_job={
            "origin_lat": -27.47,
            "origin_lon": 153.02,
            "dest_lat": -26.92,
            "dest_lon": 153.12,
            "route_geojson": None,
        },
        corridor_rows=[],
        filtered_df=pd.DataFrame(),
    )

    assert figure is not None
    assert len(figure.data) == 1
    assert figure.data[0].mode == "markers"


def test_build_planner_site_points_include_street_view_urls_for_google() -> None:
    previous_provider = os.environ.get("ROUTING_PROVIDER")
    previous_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    os.environ["ROUTING_PROVIDER"] = "google"
    os.environ["GOOGLE_MAPS_API_KEY"] = "test-key"
    try:
        points = build_planner_site_points(
            selected_job={
                "origin": "Brisbane Depot",
                "destination": "Sunshine Coast Site",
                "origin_lat": -27.47,
                "origin_lon": 153.02,
                "dest_lat": -26.92,
                "dest_lon": 153.12,
            },
            corridor_rows=[],
            filtered_df=pd.DataFrame(),
        )
    finally:
        if previous_provider is None:
            os.environ.pop("ROUTING_PROVIDER", None)
        else:
            os.environ["ROUTING_PROVIDER"] = previous_provider
        if previous_key is None:
            os.environ.pop("GOOGLE_MAPS_API_KEY", None)
        else:
            os.environ["GOOGLE_MAPS_API_KEY"] = previous_key

    assert len(points) == 2
    assert points[0]["label"] == "Origin site"
    assert points[1]["label"] == "Destination site"
    assert "maps.googleapis.com/maps/api/streetview" in points[0]["streetViewUrl"]
    assert "maps.googleapis.com/maps/api/streetview" in points[1]["streetViewUrl"]
    assert "google.com/maps/@" in points[0]["streetView360Url"]
    assert "google.com/maps/@" in points[1]["streetView360Url"]


def test_build_planner_proposal_includes_site_context_and_volume_warning() -> None:
    conn = _seed_conn()
    job = dict(conn.execute("SELECT * FROM jobs LIMIT 1").fetchone())
    upsert_site_assessment(
        conn,
        job_id=int(job["id"]),
        site_kind="origin",
        loading_access_risk="high",
        parking_risk="medium",
        narrow_street_risk="high",
        stairs_risk="low",
        clearance_risk="medium",
        large_vehicle_suitability="unsuitable",
        uncertainty_flag=True,
        note="Tight laneway access.",
        reviewed_by="planner",
    )
    create_media_inference_result(
        conn,
        job_id=int(job["id"]),
        result_type="volume_estimate",
        payload={"estimated_m3": 44.0},
        confidence=0.82,
        status="accepted",
        source="manual",
        model_name="manual-sim",
    )

    proposal = build_planner_proposal(
        conn,
        _frame(),
        selection_mode="job_first",
        preferred_template="pickup_linehaul_delivery",
        job_context=job,
    )

    assert proposal["siteContext"]["assessments"]
    assert proposal["siteContext"]["acceptedVolumeEstimate"] is not None
    assert proposal["siteContext"]["derivedConstraints"]["truckUnsuitable"] is True
    assert proposal["siteContext"]["derivedConstraints"]["shuttleRecommended"] is True
    assert proposal["siteContext"]["derivedConstraints"]["laborUpliftPct"] >= 15
    assert proposal["siteContext"]["derivedConstraints"]["accessTimeUpliftMinutes"] >= 15
    assert any("unsuitable" in warning.lower() for warning in proposal["warnings"])
    assert any("visual volume estimate" in warning.lower() for warning in proposal["warnings"])
    assert any("shuttle" in warning.lower() for warning in proposal["warnings"])
    assert proposal["explainability"]["site"]
