from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from analytics.db import ensure_historical_job_routes_table
from analytics.live_data import (
    MockTelemetryIngestor,
    TruckGpsSnapshot,
    TruckTelemetryHarness,
    build_live_heatmap_source,
    extract_route_path,
    load_active_routes,
    load_truck_positions,
)
from analytics.live_data import _position_along_route  # type: ignore[attr-defined]
from analytics.price_distribution import (
    PROFITABILITY_COLOURS,
    classify_profit_band,
    classify_profitability_status,
    prepare_profitability_route_data,
)
from corkysoft.schema import ensure_schema


def _build_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    ensure_historical_job_routes_table(conn)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS addresses (
            id INTEGER PRIMARY KEY,
            raw_input TEXT NOT NULL,
            normalized TEXT,
            city TEXT,
            state TEXT,
            postcode TEXT,
            country TEXT,
            lon REAL,
            lat REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS historical_jobs (
            id INTEGER PRIMARY KEY,
            job_date TEXT,
            client TEXT,
            origin_address_id INTEGER,
            destination_address_id INTEGER,
            volume_m3 REAL,
            revenue_total REAL,
            price_per_m3 REAL,
            distance_km REAL,
            duration_hr REAL,
            FOREIGN KEY(origin_address_id) REFERENCES addresses(id),
            FOREIGN KEY(destination_address_id) REFERENCES addresses(id)
        )
        """
    )
    addresses = [
        (1, "Brisbane", "BRISBANE", "Brisbane", "QLD", "4000", "Australia", 153.0260, -27.4705),
        (2, "Sydney", "SYDNEY", "Sydney", "NSW", "2000", "Australia", 151.2093, -33.8688),
        (3, "Melbourne", "MELBOURNE", "Melbourne", "VIC", "3000", "Australia", 144.9631, -37.8136),
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO addresses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        addresses,
    )
    jobs = [
        (1, "2024-01-01", "Acme", 1, 2, 40.0, 12000.0, 300.0, 900.0, 10.0),
        (2, "2024-01-02", "Bravo", 1, 3, 30.0, 9000.0, 280.0, 1600.0, 20.0),
    ]
    conn.executemany(
        """
        INSERT OR REPLACE INTO historical_jobs (
            id, job_date, client, origin_address_id, destination_address_id,
            volume_m3, revenue_total, price_per_m3, distance_km, duration_hr
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        jobs,
    )

    route_geojson = json.dumps(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [153.0260, -27.4705],
                            [153.2760, -27.4705],
                            [153.2760, -27.2205],
                        ],
                    },
                }
            ],
        }
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO historical_job_routes (
            historical_job_id, geojson, created_at, updated_at
        ) VALUES (?, ?, datetime('now'), datetime('now'))
        """,
        (1, route_geojson),
    )
    conn.commit()
    return conn


def test_mock_ingestor_populates_live_tables():
    conn = _build_conn()
    try:
        ingestor = MockTelemetryIngestor(conn, truck_ids=("TRK-1",))
        start = datetime(2024, 1, 1, 8, 0, tzinfo=UTC)
        midpoint = start + timedelta(hours=5)

        ingestor.run_cycle(now=start, jitter=0.0)
        ingestor.run_cycle(now=midpoint, jitter=0.0)

        trucks_df = load_truck_positions(conn)
        routes_df = load_active_routes(conn)

        assert not trucks_df.empty
        assert list(trucks_df["truck_id"]) == ["TRK-1"]

        route_row = routes_df.iloc[0]
        progress_value = float(route_row["progress"])
        travel_seconds = float(route_row["travel_seconds"])
        expected_progress = min(1.0, (midpoint - start).total_seconds() / travel_seconds)
        assert pytest.approx(progress_value, rel=1e-3) == pytest.approx(expected_progress, rel=1e-3)

        route_geometry = route_row["route_geometry"]
        assert route_geometry
        points = extract_route_path(route_geometry)
        expected_lat, expected_lon = _position_along_route(points, progress_value)

        truck_row = trucks_df.iloc[0]
        assert pytest.approx(float(truck_row["lat"]), rel=1e-4) == pytest.approx(expected_lat, rel=1e-4)
        assert pytest.approx(float(truck_row["lon"]), rel=1e-4) == pytest.approx(expected_lon, rel=1e-4)

        base_df = routes_df.copy()
        base_df["id"] = base_df["job_id"]
        base_df["price_per_m3"] = 300.0
        base_df["corridor_display"] = "Test Corridor"
        mapped = prepare_profitability_route_data(base_df, break_even=250.0)

        assert "colour" in mapped.columns
        assert all(isinstance(colour, list) for colour in mapped["colour"])
        assert "profitability_status" in mapped.columns
        assert set(mapped["profitability_status"]) <= {"Profitable", "Break-even"}
    finally:
        conn.close()


def test_harness_projects_real_gps_updates():
    conn = _build_conn()
    try:
        harness = TruckTelemetryHarness(conn)
        start = datetime(2024, 1, 1, 6, 0, tzinfo=UTC)

        harness.ingest(
            [
                TruckGpsSnapshot(
                    truck_id="LIVE-1",
                    lat=-27.4705,
                    lon=153.0260,
                    status="en_route",
                    recorded_at=start,
                    job_id=1,
                )
            ]
        )

        row = conn.execute(
            "SELECT progress, started_at, travel_seconds FROM active_routes WHERE truck_id=?",
            ("LIVE-1",),
        ).fetchone()
        assert row is not None
        assert pytest.approx(row["progress"], rel=1e-4) == 0.0
        assert row["travel_seconds"] == pytest.approx(10 * 3600.0)

        geometry = conn.execute(
            "SELECT route_geometry FROM active_routes WHERE truck_id=?",
            ("LIVE-1",),
        ).fetchone()[0]
        points = extract_route_path(geometry)
        mid_lat, mid_lon = _position_along_route(points, 0.5)

        harness.ingest(
            [
                TruckGpsSnapshot(
                    truck_id="LIVE-1",
                    lat=mid_lat,
                    lon=mid_lon,
                    status="en_route",
                    recorded_at=start + timedelta(hours=5),
                    job_id=1,
                )
            ]
        )

        updated = conn.execute(
            "SELECT progress, eta FROM active_routes WHERE truck_id=?",
            ("LIVE-1",),
        ).fetchone()
        assert updated is not None
        assert pytest.approx(updated["progress"], rel=1e-3) == 0.5
        assert updated["eta"].startswith("2024-01-01T16:00")

        trucks_df = load_truck_positions(conn)
        assert trucks_df.loc[trucks_df["truck_id"] == "LIVE-1", "lat"].notna().all()
    finally:
        conn.close()


def test_ingest_preserves_zero_coordinates():
    conn = _build_conn()
    try:
        harness = TruckTelemetryHarness(conn)
        recorded_at = datetime(2024, 1, 3, 9, 0, tzinfo=UTC)

        harness.ingest(
            [
                TruckGpsSnapshot(
                    truck_id="ZERO-1",
                    lat=-27.4705,
                    lon=153.0260,
                    status="en_route",
                    recorded_at=recorded_at,
                    job_id=1,
                    origin_lat=0.0,
                    origin_lon=153.0260,
                    dest_lat=None,
                    dest_lon=0.0,
                )
            ]
        )

        row = conn.execute(
            """
            SELECT origin_lat, origin_lon, dest_lat, dest_lon
            FROM active_routes
            WHERE truck_id=?
            """,
            ("ZERO-1",),
        ).fetchone()

        assert row is not None
        assert row["origin_lat"] == 0.0
        assert row["origin_lon"] == pytest.approx(153.0260)
        assert row["dest_lon"] == 0.0
        assert row["dest_lat"] == pytest.approx(-33.8688)
    finally:
        conn.close()


def test_build_live_heatmap_source_emphasises_live_points():
    historical_routes = pd.DataFrame(
        [
            {
                "origin_lat": -27.47,
                "origin_lon": 153.026,
                "dest_lat": -33.8688,
                "dest_lon": 151.2093,
            }
        ]
    )
    active_routes = pd.DataFrame(
        [
            {
                "origin_lat": -37.8136,
                "origin_lon": 144.9631,
                "dest_lat": -34.9285,
                "dest_lon": 138.6007,
            }
        ]
    )
    trucks = pd.DataFrame(
        [
            {
                "truck_id": "TRK-1",
                "lat": -35.308,
                "lon": 149.124,
            }
        ]
    )

    heatmap_df = build_live_heatmap_source(historical_routes, active_routes, trucks)

    assert set(heatmap_df["source"]) == {
        "Historical origin",
        "Historical destination",
        "Active origin",
        "Active destination",
        "Active truck",
    }

    def _weights_for(source: str) -> set[float]:
        return set(heatmap_df.loc[heatmap_df["source"] == source, "weight"])

    assert _weights_for("Historical origin") == {1.0}
    assert _weights_for("Historical destination") == {1.0}
    assert _weights_for("Active origin") == {3.0}
    assert _weights_for("Active destination") == {3.0}
    assert _weights_for("Active truck") == {5.0}

    numeric_coords = heatmap_df[["lat", "lon"]].apply(pd.to_numeric, errors="coerce")
    assert numeric_coords.notna().all().all()

    empty_heatmap = build_live_heatmap_source(
        pd.DataFrame(columns=["origin_lat", "origin_lon", "dest_lat", "dest_lon"]),
        pd.DataFrame(columns=["origin_lat", "origin_lon", "dest_lat", "dest_lon"]),
        pd.DataFrame(columns=["lat", "lon"]),
    )

    assert empty_heatmap.empty
    assert list(empty_heatmap.columns) == ["lat", "lon", "weight", "source"]


def test_classify_profit_band_edges():
    assert classify_profit_band(None, 250.0) == "Unknown"
    assert classify_profit_band(200.0, 250.0) == "Below break-even"
    assert classify_profit_band(260.0, 250.0) == "0-50 above break-even"
    assert classify_profit_band(320.0, 250.0) == "50-100 above break-even"
    assert classify_profit_band(370.0, 250.0) == "100+ above break-even"
    assert PROFITABILITY_COLOURS["Below break-even"]


def test_classify_profitability_status_thresholds():
    assert classify_profitability_status(None, 250.0) == "Unknown"
    assert classify_profitability_status("not-a-number", 250.0) == "Unknown"
    assert classify_profitability_status(240.0, 250.0) == "Loss-leading"
    assert classify_profitability_status(254.0, 250.0) == "Break-even"
    assert classify_profitability_status(300.0, 250.0) == "Profitable"
