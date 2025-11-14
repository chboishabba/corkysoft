import json
import sqlite3

import pytest

from analytics.db import ensure_dashboard_tables
from analytics.routing_provider import RouteGeometryResult
from analytics.routes_map import populate_route_geometry


ENCODED_ROUTE = "~~umEca|y[kt`f@kyaJ"


class DummyRouteProvider:
    """Route provider stub that can emit GeoJSON or encoded polylines."""

    def __init__(
        self,
        *,
        distance_km: float,
        duration_hr: float,
        encoded_polyline: str | None = None,
    ) -> None:
        self.distance_km = distance_km
        self.duration_hr = duration_hr
        self.encoded_polyline = encoded_polyline
        self.calls: list[tuple[tuple[float, float], tuple[float, float], str]] = []

    def route_geometry(
        self,
        *,
        origin: tuple[float, float],
        destination: tuple[float, float],
        profile: str = "driving-car",
    ) -> RouteGeometryResult:
        self.calls.append((origin, destination, profile))
        if self.encoded_polyline is not None:
            return RouteGeometryResult(
                distance_km=self.distance_km,
                duration_hr=self.duration_hr,
                encoded_polyline=self.encoded_polyline,
            )

        return RouteGeometryResult(
            distance_km=self.distance_km,
            duration_hr=self.duration_hr,
            feature_collection={
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [
                                [float(origin[0]), float(origin[1])],
                                [float(destination[0]), float(destination[1])],
                            ],
                        },
                    }
                ],
            },
        )


def make_provider(kind: str, *, distance_km: float, duration_hr: float) -> DummyRouteProvider:
    if kind == "encoded":
        return DummyRouteProvider(
            distance_km=distance_km,
            duration_hr=duration_hr,
            encoded_polyline=ENCODED_ROUTE,
        )
    return DummyRouteProvider(distance_km=distance_km, duration_hr=duration_hr)


@pytest.fixture()
def conn():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    ensure_dashboard_tables(connection)
    try:
        yield connection
    finally:
        connection.close()


@pytest.mark.parametrize("provider_kind", ["geojson", "encoded"])
def test_populate_route_geometry_historical_inserts_geojson(conn, provider_kind):
    origin_address_id = conn.execute(
        """
        INSERT INTO addresses (raw_input, normalized, country, lon, lat)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("Origin", "origin", "Australia", 151.2093, -33.8688),
    ).lastrowid
    dest_address_id = conn.execute(
        """
        INSERT INTO addresses (raw_input, normalized, country, lon, lat)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("Destination", "destination", "Australia", 153.0260, -27.4705),
    ).lastrowid

    conn.execute(
        """
        INSERT INTO historical_jobs (
            job_date,
            client,
            origin,
            destination,
            origin_address_id,
            destination_address_id
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "2024-01-01",
            "Test Client",
            "Origin",
            "Destination",
            origin_address_id,
            dest_address_id,
        ),
    )
    job_id = conn.execute("SELECT id FROM historical_jobs").fetchone()[0]

    provider = make_provider(provider_kind, distance_km=5.0, duration_hr=1.0)
    updated = populate_route_geometry(conn, [job_id], dataset="historical", provider=provider)

    assert updated == 1
    stored = conn.execute(
        "SELECT geojson FROM historical_job_routes WHERE historical_job_id = ?",
        (job_id,),
    ).fetchone()
    assert stored is not None
    parsed = json.loads(stored["geojson"])
    coords = parsed["features"][0]["geometry"]["coordinates"]
    assert coords[0][0] == pytest.approx(151.2093)
    assert coords[0][1] == pytest.approx(-33.8688)
    assert coords[-1][0] == pytest.approx(153.0260)
    assert coords[-1][1] == pytest.approx(-27.4705)

    job_row = conn.execute(
        "SELECT * FROM historical_jobs WHERE id = ?",
        (job_id,),
    ).fetchone()
    assert pytest.approx(job_row["distance_km"], rel=1e-3) == 5.0
    if "duration_hr" in job_row.keys():
        assert pytest.approx(job_row["duration_hr"], rel=1e-3) == 1.0


@pytest.mark.parametrize("provider_kind", ["geojson", "encoded"])
def test_populate_route_geometry_live_updates_job(conn, provider_kind):
    conn.execute("ALTER TABLE jobs ADD COLUMN duration_hr REAL")
    conn.execute(
        """
        INSERT INTO jobs (
            job_date,
            client,
            origin,
            destination,
            origin_lon,
            origin_lat,
            dest_lon,
            dest_lat
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "2024-01-01",
            "Test Client",
            "Origin",
            "Destination",
            151.2093,
            -33.8688,
            153.0260,
            -27.4705,
        ),
    )
    job_id = conn.execute("SELECT id FROM jobs").fetchone()[0]

    provider = make_provider(provider_kind, distance_km=10.0, duration_hr=2.0)
    updated = populate_route_geometry(conn, [job_id], dataset="live", provider=provider)

    assert updated == 1
    stored = conn.execute(
        "SELECT * FROM jobs WHERE id = ?",
        (job_id,),
    ).fetchone()
    assert stored is not None
    parsed = json.loads(stored["route_geojson"])
    coords = parsed["features"][0]["geometry"]["coordinates"]
    assert coords[0][0] == pytest.approx(151.2093)
    assert coords[0][1] == pytest.approx(-33.8688)
    assert coords[-1][0] == pytest.approx(153.0260)
    assert coords[-1][1] == pytest.approx(-27.4705)
    assert pytest.approx(stored["distance_km"], rel=1e-3) == 10.0
    if "duration_hr" in stored.keys():
        assert pytest.approx(stored["duration_hr"], rel=1e-3) == 2.0


def test_populate_route_geometry_historical_uses_inline_coordinates(conn):
    for column in ("origin_lon", "origin_lat", "dest_lon", "dest_lat", "duration_hr"):
        conn.execute(f"ALTER TABLE historical_jobs ADD COLUMN {column} REAL")

    conn.execute(
        """
        INSERT INTO historical_jobs (
            job_date,
            client,
            origin,
            destination,
            origin_lon,
            origin_lat,
            dest_lon,
            dest_lat
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "2024-02-02",
            "Inline Client",
            "Inline Origin",
            "Inline Destination",
            150.0,
            -35.0,
            151.0,
            -34.0,
        ),
    )
    job_id = conn.execute("SELECT id FROM historical_jobs ORDER BY id DESC").fetchone()[0]

    provider = make_provider("geojson", distance_km=8.0, duration_hr=1.5)
    updated = populate_route_geometry(conn, [job_id], dataset="historical", provider=provider)

    assert updated == 1
    stored = conn.execute(
        "SELECT geojson FROM historical_job_routes WHERE historical_job_id = ?",
        (job_id,),
    ).fetchone()
    assert stored is not None
    parsed = json.loads(stored["geojson"])
    coords = parsed["features"][0]["geometry"]["coordinates"]
    assert coords[0][0] == pytest.approx(150.0)
    assert coords[0][1] == pytest.approx(-35.0)
    assert coords[-1][0] == pytest.approx(151.0)
    assert coords[-1][1] == pytest.approx(-34.0)

    job_row = conn.execute(
        "SELECT * FROM historical_jobs WHERE id = ?",
        (job_id,),
    ).fetchone()
    assert pytest.approx(job_row["distance_km"], rel=1e-3) == 8.0
    if "duration_hr" in job_row.keys():
        assert pytest.approx(job_row["duration_hr"], rel=1e-3) == 1.5
    assert pytest.approx(job_row["origin_lon"], rel=1e-6) == 150.0
    assert pytest.approx(job_row["origin_lat"], rel=1e-6) == -35.0
    assert pytest.approx(job_row["dest_lon"], rel=1e-6) == 151.0
    assert pytest.approx(job_row["dest_lat"], rel=1e-6) == -34.0
