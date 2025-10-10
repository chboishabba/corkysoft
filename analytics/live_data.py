"""Utilities for managing live telemetry data for the dashboard."""
from __future__ import annotations

import math
import random
import sqlite3
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Optional, Sequence

import pandas as pd

from analytics.db import DEFAULT_DB_PATH, connection_scope

Status = str


TRUCK_STATUS_COLOURS = {
    "loading": [0, 123, 255],
    "en_route": [92, 184, 92],
    "delayed": [240, 173, 78],
    "arrived": [108, 117, 125],
}


def _ensure_live_tables(conn: sqlite3.Connection) -> None:
    """Ensure live telemetry tables exist.

    The Streamlit dashboard depends on the presence of the *truck_positions*
    and *active_routes* tables. These are usually created by
    :func:`corkysoft.schema.ensure_schema`, but the helper keeps the ingestion
    script resilient when pointed at a brand new database.
    """

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS truck_positions (
            truck_id TEXT PRIMARY KEY,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            status TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            heading REAL,
            speed_kph REAL,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS active_routes (
            route_id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER,
            truck_id TEXT NOT NULL UNIQUE,
            origin_lat REAL NOT NULL,
            origin_lon REAL NOT NULL,
            dest_lat REAL NOT NULL,
            dest_lon REAL NOT NULL,
            progress REAL NOT NULL DEFAULT 0 CHECK(progress BETWEEN 0 AND 1),
            eta TEXT,
            status TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            notes TEXT,
            FOREIGN KEY(truck_id) REFERENCES truck_positions(truck_id) ON DELETE CASCADE,
            FOREIGN KEY(job_id) REFERENCES historical_jobs(id) ON DELETE SET NULL
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_active_routes_job_id
            ON active_routes(job_id)
            WHERE job_id IS NOT NULL;
    """
    )


def load_truck_positions(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return a DataFrame containing the latest truck telemetry."""

    try:
        df = pd.read_sql_query("SELECT * FROM truck_positions", conn)
    except Exception:
        return pd.DataFrame(columns=["truck_id", "lat", "lon", "status", "updated_at"])
    return df


def load_active_routes(
    conn: sqlite3.Connection,
    *,
    job_ids: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Return a DataFrame of active routes optionally filtered by job IDs."""

    try:
        if job_ids:
            placeholders = ",".join(["?"] * len(job_ids))
            query = f"SELECT * FROM active_routes WHERE job_id IN ({placeholders})"
            df = pd.read_sql_query(query, conn, params=job_ids)
        else:
            df = pd.read_sql_query("SELECT * FROM active_routes", conn)
    except Exception:
        return pd.DataFrame(
            columns=[
                "job_id",
                "truck_id",
                "origin_lat",
                "origin_lon",
                "dest_lat",
                "dest_lon",
                "progress",
                "eta",
                "status",
                "updated_at",
            ]
        )
    return df


def _pick_candidate_routes(conn: sqlite3.Connection) -> list[dict[str, float]]:
    """Return historical jobs with geocoded endpoints suitable for routing."""

    rows = conn.execute(
        """
        SELECT
            hj.id,
            o.lat AS origin_lat,
            o.lon AS origin_lon,
            d.lat AS dest_lat,
            d.lon AS dest_lon
        FROM historical_jobs AS hj
        JOIN addresses AS o ON hj.origin_address_id = o.id
        JOIN addresses AS d ON hj.destination_address_id = d.id
        WHERE o.lat IS NOT NULL
          AND o.lon IS NOT NULL
          AND d.lat IS NOT NULL
          AND d.lon IS NOT NULL
    """
    ).fetchall()
    if rows:
        return [dict(row) for row in rows]

    # Fall back to depot zones so the mock telemetry still functions when
    # historical data has not yet been imported.
    from corkysoft.schema import DEPOT_METRO_ZONES

    fallback: list[dict[str, float]] = []
    for idx in range(len(DEPOT_METRO_ZONES) - 1):
        origin = DEPOT_METRO_ZONES[idx]
        destination = DEPOT_METRO_ZONES[idx + 1]
        fake_row = {
            "id": None,
            "origin_lat": origin[2],
            "origin_lon": origin[3],
            "dest_lat": destination[2],
            "dest_lon": destination[3],
        }
        fallback.append(fake_row)
    return fallback


def _interpolate(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    progress: float,
) -> tuple[float, float]:
    """Return a simple linear interpolation between start and end points."""

    lat = start_lat + (end_lat - start_lat) * progress
    lon = start_lon + (end_lon - start_lon) * progress
    return lat, lon


def _bearing(start_lat: float, start_lon: float, end_lat: float, end_lon: float) -> float:
    """Approximate bearing from the start point to the end point in degrees."""

    start_lat_rad = math.radians(start_lat)
    end_lat_rad = math.radians(end_lat)
    delta_lon = math.radians(end_lon - start_lon)

    y = math.sin(delta_lon) * math.cos(end_lat_rad)
    x = (
        math.cos(start_lat_rad) * math.sin(end_lat_rad)
        - math.sin(start_lat_rad) * math.cos(end_lat_rad) * math.cos(delta_lon)
    )
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360) % 360


@dataclass
class MockTelemetryIngestor:
    """Generate and persist mock telemetry data for the dashboard map."""

    conn: sqlite3.Connection
    truck_ids: Sequence[str]
    status_cycle: Sequence[Status] = ("loading", "en_route", "en_route", "en_route", "arrived")

    def __post_init__(self) -> None:
        _ensure_live_tables(self.conn)
        self.conn.row_factory = sqlite3.Row

    def _load_existing_progress(self) -> dict[str, tuple[Optional[int], float]]:
        rows = self.conn.execute(
            "SELECT truck_id, job_id, progress FROM active_routes"
        ).fetchall()
        return {
            row["truck_id"]: (row["job_id"], row["progress"])
            for row in rows
        }

    def run_cycle(self, *, jitter: float = 0.1) -> None:
        """Create or update telemetry records for each configured truck."""

        candidates = _pick_candidate_routes(self.conn)
        if not candidates:
            return

        existing_progress = self._load_existing_progress()
        routes_by_id = {
            route["id"]: route
            for route in candidates
            if route.get("id") is not None
        }
        assigned_job_ids: set[int] = set()
        now = datetime.now(UTC)

        active_payload: list[tuple] = []
        truck_payload: list[tuple] = []

        for idx, truck_id in enumerate(self.truck_ids):
            previous = existing_progress.get(truck_id)
            if previous:
                prev_job_id, prev_progress = previous
            else:
                prev_job_id = None
                prev_progress = None

            if prev_job_id is not None:
                route = routes_by_id.get(prev_job_id)
                if route is not None:
                    assigned_job_ids.add(prev_job_id)
            else:
                route = None

            if route is None:
                shuffled = candidates[:]
                random.shuffle(shuffled)
                for candidate in shuffled:
                    job_id = candidate.get("id")
                    if job_id is None or job_id not in assigned_job_ids:
                        route = candidate
                        if job_id is not None:
                            assigned_job_ids.add(job_id)
                        break
                if route is None:
                    route = random.choice(candidates)

            baseline_progress = prev_progress if prev_progress is not None else random.random() * 0.1
            route_progress = max(0.0, min(1.0, baseline_progress))
            increment = random.uniform(0.05, 0.15)
            progress = min(route_progress + increment, 1.0)

            status_idx = min(int(progress * len(self.status_cycle)), len(self.status_cycle) - 1)
            status = self.status_cycle[status_idx]

            lat, lon = _interpolate(
                route["origin_lat"],
                route["origin_lon"],
                route["dest_lat"],
                route["dest_lon"],
                progress,
            )

            bearing = _bearing(
                route["origin_lat"],
                route["origin_lon"],
                route["dest_lat"],
                route["dest_lon"],
            )
            speed = random.uniform(30, 90) * max(0.2, 1 - abs(0.5 - progress))
            eta = now + timedelta(minutes=max(5, (1 - progress) * random.uniform(20, 90)))

            truck_payload.append(
                (
                    truck_id,
                    lat,
                    lon,
                    status,
                    now.isoformat(),
                    bearing,
                    speed,
                    f"Progress {progress:.0%}",
                )
            )

            active_payload.append(
                (
                    route.get("id"),
                    truck_id,
                    route["origin_lat"],
                    route["origin_lon"],
                    route["dest_lat"],
                    route["dest_lon"],
                    progress,
                    eta.isoformat(),
                    status,
                    now.isoformat(),
                    None,
                )
            )

        self.conn.executemany(
            """
            INSERT INTO truck_positions (
                truck_id, lat, lon, status, updated_at, heading, speed_kph, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(truck_id) DO UPDATE SET
                lat = excluded.lat,
                lon = excluded.lon,
                status = excluded.status,
                updated_at = excluded.updated_at,
                heading = excluded.heading,
                speed_kph = excluded.speed_kph,
                notes = excluded.notes
            """,
            truck_payload,
        )

        self.conn.executemany(
            """
            INSERT INTO active_routes (
                job_id,
                truck_id,
                origin_lat,
                origin_lon,
                dest_lat,
                dest_lon,
                progress,
                eta,
                status,
                updated_at,
                notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(truck_id) DO UPDATE SET
                job_id = excluded.job_id,
                origin_lat = excluded.origin_lat,
                origin_lon = excluded.origin_lon,
                dest_lat = excluded.dest_lat,
                dest_lon = excluded.dest_lon,
                progress = excluded.progress,
                eta = excluded.eta,
                status = excluded.status,
                updated_at = excluded.updated_at,
                notes = excluded.notes
            """,
            active_payload,
        )

        # Clean up completed routes so trucks can be reassigned on the next cycle.
        self.conn.execute("DELETE FROM active_routes WHERE progress >= 0.999")
        self.conn.commit()


def run_mock_ingestor(
    *,
    db_path: str = DEFAULT_DB_PATH,
    truck_ids: Optional[Sequence[str]] = None,
    interval_seconds: float = 5.0,
    iterations: Optional[int] = None,
) -> None:
    """Run the mock telemetry ingestion loop."""

    trucks = truck_ids or ("BNE-01", "BNE-02", "BNE-03", "BNE-04")

    with connection_scope(db_path) as conn:
        ingestor = MockTelemetryIngestor(conn, truck_ids=trucks)
        count = 0
        while True:
            ingestor.run_cycle()
            count += 1
            if iterations is not None and count >= iterations:
                break
            time.sleep(interval_seconds)


__all__ = [
    "TRUCK_STATUS_COLOURS",
    "MockTelemetryIngestor",
    "load_truck_positions",
    "load_active_routes",
    "run_mock_ingestor",
]
