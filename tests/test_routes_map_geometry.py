import sqlite3

import pytest

from analytics.db import ensure_dashboard_tables
from analytics.routes_map import populate_route_geometry


class DummyORSClient:
    """Minimal OpenRouteService client stub used for tests."""

    def __init__(self, distance_m: float = 12345.6, duration_s: float = 7890.1):
        self.distance_m = distance_m
        self.duration_s = duration_s
        self.calls = []

    def directions(self, *, coordinates, profile, format):  # type: ignore[override]
        self.calls.append((coordinates, profile, format))
        if format != "geojson":
            raise ValueError("Expected geojson format")
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "summary": {
                            "distance": self.distance_m,
                            "duration": self.duration_s,
                        }
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coordinates,
                    },
                }
            ],
        }


@pytest.fixture()
def conn():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    ensure_dashboard_tables(connection)
    try:
        yield connection
    finally:
        connection.close()


def test_populate_route_geometry_historical_inserts_geojson(monkeypatch, conn):
    from analytics import routes_map

    monkeypatch.setattr(routes_map, "ROUTE_BACKOFF", 0.0)

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

    client = DummyORSClient(distance_m=5000.0, duration_s=3600.0)
    updated = populate_route_geometry(conn, [job_id], dataset="historical", client=client)

    assert updated == 1
    stored = conn.execute(
        "SELECT geojson FROM historical_job_routes WHERE historical_job_id = ?",
        (job_id,),
    ).fetchone()
    assert stored is not None
    assert "FeatureCollection" in stored["geojson"]

    job_row = conn.execute(
        "SELECT * FROM historical_jobs WHERE id = ?",
        (job_id,),
    ).fetchone()
    assert pytest.approx(job_row["distance_km"], rel=1e-3) == 5.0
    if "duration_hr" in job_row.keys():
        assert pytest.approx(job_row["duration_hr"], rel=1e-3) == 1.0


def test_populate_route_geometry_live_updates_job(monkeypatch, conn):
    from analytics import routes_map

    monkeypatch.setattr(routes_map, "ROUTE_BACKOFF", 0.0)

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

    client = DummyORSClient(distance_m=10000.0, duration_s=7200.0)
    updated = populate_route_geometry(conn, [job_id], dataset="live", client=client)

    assert updated == 1
    stored = conn.execute(
        "SELECT * FROM jobs WHERE id = ?",
        (job_id,),
    ).fetchone()
    assert stored is not None
    assert "FeatureCollection" in stored["route_geojson"]
    assert pytest.approx(stored["distance_km"], rel=1e-3) == 10.0
    if "duration_hr" in stored.keys():
        assert pytest.approx(stored["duration_hr"], rel=1e-3) == 2.0


def test_populate_route_geometry_historical_uses_inline_coordinates(monkeypatch, conn):
    from analytics import routes_map

    monkeypatch.setattr(routes_map, "ROUTE_BACKOFF", 0.0)

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

    client = DummyORSClient(distance_m=8000.0, duration_s=5400.0)
    updated = populate_route_geometry(conn, [job_id], dataset="historical", client=client)

    assert updated == 1
    stored = conn.execute(
        "SELECT geojson FROM historical_job_routes WHERE historical_job_id = ?",
        (job_id,),
    ).fetchone()
    assert stored is not None
    assert "FeatureCollection" in stored["geojson"]

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
