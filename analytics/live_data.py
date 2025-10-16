"""Utilities for managing live telemetry data for the dashboard."""
from __future__ import annotations

import json
import math
import random
import sqlite3
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Iterable, Optional, Sequence

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
            route_geometry TEXT,
            started_at TEXT,
            travel_seconds REAL,
            FOREIGN KEY(truck_id) REFERENCES truck_positions(truck_id) ON DELETE CASCADE,
            FOREIGN KEY(job_id) REFERENCES historical_jobs(id) ON DELETE SET NULL
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_active_routes_job_id
            ON active_routes(job_id)
            WHERE job_id IS NOT NULL;
    """
    )

    # Backfill new columns for existing deployments.
    column_rows = conn.execute("PRAGMA table_info(active_routes)").fetchall()
    existing_columns = {row[1] for row in column_rows}
    if "route_geometry" not in existing_columns:
        conn.execute("ALTER TABLE active_routes ADD COLUMN route_geometry TEXT")
    if "started_at" not in existing_columns:
        conn.execute("ALTER TABLE active_routes ADD COLUMN started_at TEXT")
    if "travel_seconds" not in existing_columns:
        conn.execute("ALTER TABLE active_routes ADD COLUMN travel_seconds REAL")


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


def build_live_heatmap_source(
    historical_routes: pd.DataFrame,
    active_routes: pd.DataFrame,
    trucks: pd.DataFrame,
) -> pd.DataFrame:
    """Return a point-based dataframe suitable for live network heatmaps.

    Historical corridor endpoints contribute a baseline weight, active routes
    are amplified so in-flight jobs stand out, and individual truck telemetry
    receives the highest weighting so the density view reflects real-time
    activity.
    """

    columns = ["lat", "lon", "weight", "source"]
    frames: list[pd.DataFrame] = []

    def _extract_points(
        df: pd.DataFrame,
        lat_column: str,
        lon_column: str,
        *,
        weight: float,
        source: str,
    ) -> None:
        if df.empty or lat_column not in df.columns or lon_column not in df.columns:
            return
        coords = df[[lat_column, lon_column]].apply(pd.to_numeric, errors="coerce").dropna()
        if coords.empty:
            return
        frame = coords.rename(columns={lat_column: "lat", lon_column: "lon"}).astype(float)
        frame["weight"] = float(weight)
        frame["source"] = source
        frames.append(frame)

    _extract_points(
        historical_routes,
        "origin_lat",
        "origin_lon",
        weight=1.0,
        source="Historical origin",
    )
    _extract_points(
        historical_routes,
        "dest_lat",
        "dest_lon",
        weight=1.0,
        source="Historical destination",
    )
    _extract_points(
        active_routes,
        "origin_lat",
        "origin_lon",
        weight=3.0,
        source="Active origin",
    )
    _extract_points(
        active_routes,
        "dest_lat",
        "dest_lon",
        weight=3.0,
        source="Active destination",
    )
    _extract_points(
        trucks,
        "lat",
        "lon",
        weight=5.0,
        source="Active truck",
    )

    if not frames:
        return pd.DataFrame(columns=columns)

    return pd.concat(frames, ignore_index=True)[columns]


def _pick_candidate_routes(conn: sqlite3.Connection) -> list[dict[str, float]]:
    """Return historical jobs with geocoded endpoints suitable for routing."""

    rows = conn.execute(
        """
        SELECT
            hj.id,
            o.lat AS origin_lat,
            o.lon AS origin_lon,
            d.lat AS dest_lat,
            d.lon AS dest_lon,
            hj.distance_km,
            hj.duration_hr,
            hr.geojson AS route_geometry
        FROM historical_jobs AS hj
        JOIN addresses AS o ON hj.origin_address_id = o.id
        JOIN addresses AS d ON hj.destination_address_id = d.id
        LEFT JOIN historical_job_routes AS hr ON hr.historical_job_id = hj.id
        WHERE o.lat IS NOT NULL
          AND o.lon IS NOT NULL
          AND d.lat IS NOT NULL
          AND d.lon IS NOT NULL
    """
    ).fetchall()
    if rows:
        candidates: list[dict[str, float]] = []
        for row in rows:
            candidate = dict(row)
            duration_hr = candidate.get("duration_hr")
            if duration_hr is not None:
                try:
                    candidate["travel_seconds"] = float(duration_hr) * 3600.0
                except (TypeError, ValueError):
                    candidate["travel_seconds"] = None
            distance_km = candidate.get("distance_km")
            if distance_km is not None:
                try:
                    candidate["distance_km"] = float(distance_km)
                except (TypeError, ValueError):
                    candidate["distance_km"] = None
            candidates.append(candidate)

        with_geometry = [candidate for candidate in candidates if candidate.get("route_geometry")]
        if with_geometry:
            return with_geometry
        return candidates

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


_EARTH_RADIUS_M = 6_371_000.0


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the distance in metres between two geographic coordinates."""

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = lat2_rad - lat1_rad
    delta_lon = math.radians(lon2 - lon1)

    sin_lat = math.sin(delta_lat / 2.0)
    sin_lon = math.sin(delta_lon / 2.0)
    a = sin_lat ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * sin_lon ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return _EARTH_RADIUS_M * c


def _route_points_from_geojson(geojson: str) -> list[tuple[float, float]]:
    """Return a ``[(lat, lon), ...]`` list extracted from GeoJSON *geojson*."""

    if not geojson:
        raise ValueError("Route geometry is empty")

    payload = json.loads(geojson)
    if not isinstance(payload, dict):
        raise ValueError("Invalid GeoJSON structure")

    geometry: Optional[dict]
    if payload.get("type") == "FeatureCollection":
        features = payload.get("features") or []
        if not features:
            raise ValueError("GeoJSON feature collection is empty")
        geometry = features[0].get("geometry")
    elif payload.get("type") == "Feature":
        geometry = payload.get("geometry")
    else:
        geometry = payload

    if not geometry or "type" not in geometry:
        raise ValueError("GeoJSON geometry missing")

    coords: list
    if geometry["type"] == "LineString":
        coords = geometry.get("coordinates") or []
    elif geometry["type"] == "MultiLineString":
        coords = [pt for segment in geometry.get("coordinates") or [] for pt in segment]
    else:
        raise ValueError(f"Unsupported geometry type: {geometry['type']}")

    if not coords:
        raise ValueError("GeoJSON contains no coordinates")

    return [(float(lat_lon[1]), float(lat_lon[0])) for lat_lon in coords]


def _route_path_length(points: Sequence[tuple[float, float]]) -> float:
    """Return the total length of *points* in metres."""

    if len(points) < 2:
        return 0.0
    length = 0.0
    for (lat1, lon1), (lat2, lon2) in zip(points[:-1], points[1:]):
        length += _haversine_m(lat1, lon1, lat2, lon2)
    return length


def _position_along_route(points: Sequence[tuple[float, float]], progress: float) -> tuple[float, float]:
    """Return the coordinate located at *progress* along the route defined by *points*."""

    if not points:
        raise ValueError("Route has no vertices")
    if len(points) == 1:
        return points[0]

    progress = max(0.0, min(1.0, float(progress)))
    target = _route_path_length(points) * progress
    if target <= 0:
        return points[0]

    traversed = 0.0
    for start, end in zip(points[:-1], points[1:]):
        segment = _haversine_m(start[0], start[1], end[0], end[1])
        if segment <= 0:
            continue
        if traversed + segment >= target:
            ratio = (target - traversed) / segment
            lat = start[0] + (end[0] - start[0]) * ratio
            lon = start[1] + (end[1] - start[1]) * ratio
            return lat, lon
        traversed += segment

    return points[-1]


def _progress_along_route(points: Sequence[tuple[float, float]], lat: float, lon: float) -> float:
    """Return the fractional progress of *(lat, lon)* along *points*.

    The function approximates the projection of the point onto the nearest
    segment of the route and expresses the distance travelled as a fraction of
    the total path length.
    """

    if len(points) < 2:
        return 0.0

    ref_lat, ref_lon = points[0]
    target_x, target_y = _to_xy(lat, lon, ref_lat, ref_lon)

    cumulative = [0.0]
    total = 0.0
    segments_xy: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for start, end in zip(points[:-1], points[1:]):
        start_xy = _to_xy(start[0], start[1], ref_lat, ref_lon)
        end_xy = _to_xy(end[0], end[1], ref_lat, ref_lon)
        segments_xy.append((start_xy, end_xy))
        segment_length = math.dist(start_xy, end_xy)
        total += segment_length
        cumulative.append(total)

    if total == 0:
        return 0.0

    best_distance = float("inf")
    best_fraction = 0.0
    for idx, (start_xy, end_xy) in enumerate(segments_xy):
        seg_vec = (end_xy[0] - start_xy[0], end_xy[1] - start_xy[1])
        seg_len_sq = seg_vec[0] ** 2 + seg_vec[1] ** 2
        if seg_len_sq == 0:
            continue
        point_vec = (target_x - start_xy[0], target_y - start_xy[1])
        t = max(0.0, min(1.0, (point_vec[0] * seg_vec[0] + point_vec[1] * seg_vec[1]) / seg_len_sq))
        proj_x = start_xy[0] + seg_vec[0] * t
        proj_y = start_xy[1] + seg_vec[1] * t
        distance = math.dist((proj_x, proj_y), (target_x, target_y))
        if distance < best_distance:
            best_distance = distance
            fraction = (cumulative[idx] + math.dist(start_xy, (proj_x, proj_y))) / total
            best_fraction = fraction

    return max(0.0, min(1.0, best_fraction))


def _to_xy(lat: float, lon: float, ref_lat: float, ref_lon: float) -> tuple[float, float]:
    """Return planar X/Y coordinates relative to *ref_lat*/*ref_lon*."""

    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    ref_lat_rad = math.radians(ref_lat)
    ref_lon_rad = math.radians(ref_lon)
    x = (lon_rad - ref_lon_rad) * math.cos((lat_rad + ref_lat_rad) / 2.0) * _EARTH_RADIUS_M
    y = (lat_rad - ref_lat_rad) * _EARTH_RADIUS_M
    return x, y


def extract_route_path(geojson: str) -> list[tuple[float, float]]:
    """Expose parsed route coordinates for external consumers."""

    return _route_points_from_geojson(geojson)


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


@dataclass
class MockTelemetryIngestor:
    """Generate and persist mock telemetry data for the dashboard map."""

    conn: sqlite3.Connection
    truck_ids: Sequence[str]
    status_cycle: Sequence[Status] = ("loading", "en_route", "en_route", "en_route", "arrived")

    def __post_init__(self) -> None:
        _ensure_live_tables(self.conn)
        self.conn.row_factory = sqlite3.Row

    def _load_existing_progress(self) -> dict[str, dict[str, Optional[float]]]:
        rows = self.conn.execute(
            "SELECT truck_id, job_id, progress, started_at, travel_seconds FROM active_routes"
        ).fetchall()
        state: dict[str, dict[str, Optional[float]]] = {}
        for row in rows:
            state[row["truck_id"]] = {
                "job_id": row["job_id"],
                "progress": row["progress"],
                "started_at": row["started_at"],
                "travel_seconds": row["travel_seconds"],
            }
        return state

    def run_cycle(self, *, jitter: float = 0.1, now: Optional[datetime] = None) -> None:
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
        current_time = now or datetime.now(UTC)

        active_payload: list[tuple] = []
        truck_payload: list[tuple] = []

        for idx, truck_id in enumerate(self.truck_ids):
            previous = existing_progress.get(truck_id, {})
            prev_job_id = previous.get("job_id")
            prev_progress = previous.get("progress")
            prev_started = previous.get("started_at")
            prev_travel_seconds = previous.get("travel_seconds")

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

            geometry_points: Optional[list[tuple[float, float]]] = None
            if route.get("route_geometry"):
                try:
                    geometry_points = _route_points_from_geojson(route["route_geometry"])
                except Exception:
                    geometry_points = None

            travel_seconds = route.get("travel_seconds")
            if travel_seconds is None and prev_travel_seconds:
                travel_seconds = prev_travel_seconds
            if not travel_seconds:
                distance_km = route.get("distance_km")
                if distance_km:
                    travel_seconds = float(distance_km) / 70.0 * 3600.0
                else:
                    travel_seconds = 3_600.0

            if prev_started:
                try:
                    started_at = datetime.fromisoformat(str(prev_started))
                except ValueError:
                    started_at = current_time
            else:
                jitter_ratio = max(0.0, float(jitter))
                jitter_seconds = (
                    random.uniform(0.0, jitter_ratio * travel_seconds)
                    if travel_seconds and jitter_ratio > 0
                    else 0.0
                )
                started_at = current_time - timedelta(seconds=jitter_seconds)

            elapsed = max(0.0, (current_time - started_at).total_seconds())
            progress = min(1.0, elapsed / travel_seconds) if travel_seconds else 1.0

            if prev_progress is not None and progress < prev_progress:
                progress = prev_progress

            status_idx = min(int(progress * len(self.status_cycle)), len(self.status_cycle) - 1)
            status = self.status_cycle[status_idx]

            if geometry_points:
                try:
                    lat, lon = _position_along_route(geometry_points, progress)
                except Exception:
                    lat, lon = (
                        route["origin_lat"] + (route["dest_lat"] - route["origin_lat"]) * progress,
                        route["origin_lon"] + (route["dest_lon"] - route["origin_lon"]) * progress,
                    )
                    geometry_points = None
            else:
                lat = route["origin_lat"] + (route["dest_lat"] - route["origin_lat"]) * progress
                lon = route["origin_lon"] + (route["dest_lon"] - route["origin_lon"]) * progress

            if geometry_points and len(geometry_points) > 1:
                heading_point = _position_along_route(geometry_points, min(progress + 0.01, 1.0))
                bearing = _bearing(lat, lon, heading_point[0], heading_point[1])
            else:
                bearing = _bearing(
                    route["origin_lat"],
                    route["origin_lon"],
                    route["dest_lat"],
                    route["dest_lon"],
                )

            distance_km = route.get("distance_km")
            if distance_km and travel_seconds:
                speed = (distance_km / (travel_seconds / 3600.0))
            else:
                speed = random.uniform(30, 90)
            eta = started_at + timedelta(seconds=travel_seconds) if travel_seconds else None

            truck_payload.append(
                (
                    truck_id,
                    lat,
                    lon,
                    status,
                    current_time.isoformat(),
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
                    eta.isoformat() if eta else None,
                    status,
                    current_time.isoformat(),
                    None,
                    route.get("route_geometry"),
                    started_at.isoformat(),
                    float(travel_seconds) if travel_seconds else None,
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
                notes,
                route_geometry,
                started_at,
                travel_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                notes = excluded.notes,
                route_geometry = COALESCE(excluded.route_geometry, active_routes.route_geometry),
                started_at = excluded.started_at,
                travel_seconds = excluded.travel_seconds
            """,
            active_payload,
        )

        # Clean up completed routes so trucks can be reassigned on the next cycle.
        self.conn.execute("DELETE FROM active_routes WHERE progress >= 0.999")
        self.conn.commit()


@dataclass
class TruckGpsSnapshot:
    """Represents a single GPS reading received from an external source."""

    truck_id: str
    lat: float
    lon: float
    status: Status
    recorded_at: datetime
    heading: Optional[float] = None
    speed_kph: Optional[float] = None
    job_id: Optional[int] = None
    eta: Optional[datetime] = None
    notes: Optional[str] = None
    progress: Optional[float] = None
    travel_seconds: Optional[float] = None
    origin_lat: Optional[float] = None
    origin_lon: Optional[float] = None
    dest_lat: Optional[float] = None
    dest_lon: Optional[float] = None
    route_geometry: Optional[str] = None
    started_at: Optional[datetime] = None


def _fetch_job_metadata(conn: sqlite3.Connection, job_id: int) -> Optional[dict]:
    row = conn.execute(
        """
        SELECT
            hj.id,
            hj.distance_km,
            hj.duration_hr,
            o.lat AS origin_lat,
            o.lon AS origin_lon,
            d.lat AS dest_lat,
            d.lon AS dest_lon,
            hr.geojson AS route_geometry
        FROM historical_jobs AS hj
        LEFT JOIN addresses AS o ON hj.origin_address_id = o.id
        LEFT JOIN addresses AS d ON hj.destination_address_id = d.id
        LEFT JOIN historical_job_routes AS hr ON hr.historical_job_id = hj.id
        WHERE hj.id = ?
        """,
        (job_id,),
    ).fetchone()
    return dict(row) if row else None


def _coalesce_route_points(meta: dict, snapshot: TruckGpsSnapshot) -> Optional[list[tuple[float, float]]]:
    if snapshot.route_geometry:
        geometry = snapshot.route_geometry
    elif meta:
        geometry = meta.get("route_geometry")
    else:
        geometry = None
    if geometry:
        try:
            return _route_points_from_geojson(geometry)
        except Exception:
            return None
    return None


class TruckTelemetryHarness:
    """Persist real GPS telemetry into the dashboard database."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        _ensure_live_tables(self.conn)
        self.conn.row_factory = sqlite3.Row

    def ingest(self, readings: Iterable[TruckGpsSnapshot]) -> None:
        trucks_payload: list[tuple] = []
        routes_payload: list[tuple] = []

        for reading in readings:
            recorded_at = _ensure_utc(reading.recorded_at)
            job_meta = None
            if reading.job_id is not None:
                job_meta = _fetch_job_metadata(self.conn, int(reading.job_id))

            if reading.origin_lat is not None:
                origin_lat = reading.origin_lat
            else:
                origin_lat = job_meta.get("origin_lat") if job_meta else None

            if reading.origin_lon is not None:
                origin_lon = reading.origin_lon
            else:
                origin_lon = job_meta.get("origin_lon") if job_meta else None

            if reading.dest_lat is not None:
                dest_lat = reading.dest_lat
            else:
                dest_lat = job_meta.get("dest_lat") if job_meta else None

            if reading.dest_lon is not None:
                dest_lon = reading.dest_lon
            else:
                dest_lon = job_meta.get("dest_lon") if job_meta else None

            travel_seconds = reading.travel_seconds
            if travel_seconds is None and job_meta and job_meta.get("duration_hr") is not None:
                try:
                    travel_seconds = float(job_meta["duration_hr"]) * 3600.0
                except (TypeError, ValueError):
                    travel_seconds = None

            points = _coalesce_route_points(job_meta or {}, reading)

            progress = reading.progress
            if progress is None and points:
                progress = _progress_along_route(points, reading.lat, reading.lon)

            if progress is None and origin_lat is not None and origin_lon is not None and dest_lat is not None and dest_lon is not None:
                # fallback to straight-line projection
                total = _haversine_m(origin_lat, origin_lon, dest_lat, dest_lon)
                if total:
                    travelled = _haversine_m(origin_lat, origin_lon, reading.lat, reading.lon)
                    progress = max(0.0, min(1.0, travelled / total))
                else:
                    progress = 0.0

            if progress is None:
                progress = 0.0

            try:
                progress = float(progress)
            except (TypeError, ValueError):
                progress = 0.0

            progress = max(0.0, min(1.0, progress))

            started_at = reading.started_at
            if started_at is None:
                if travel_seconds:
                    started_at = recorded_at - timedelta(seconds=progress * travel_seconds)
                else:
                    started_at = recorded_at
            else:
                started_at = _ensure_utc(started_at)

            eta = reading.eta
            if eta is None and travel_seconds is not None:
                eta = started_at + timedelta(seconds=travel_seconds)

            trucks_payload.append(
                (
                    reading.truck_id,
                    reading.lat,
                    reading.lon,
                    reading.status,
                    recorded_at.isoformat(),
                    reading.heading,
                    reading.speed_kph,
                    reading.notes,
                )
            )

            if reading.job_id is not None and None not in (origin_lat, origin_lon, dest_lat, dest_lon):
                routes_payload.append(
                    (
                        reading.job_id,
                        reading.truck_id,
                        origin_lat,
                        origin_lon,
                        dest_lat,
                        dest_lon,
                        float(max(0.0, min(1.0, progress))),
                        eta.isoformat() if eta else None,
                        reading.status,
                        recorded_at.isoformat(),
                    reading.notes,
                    reading.route_geometry or (job_meta.get("route_geometry") if job_meta else None),
                    started_at.isoformat(),
                    float(travel_seconds) if travel_seconds is not None else None,
                )
            )

        if trucks_payload:
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
                trucks_payload,
            )

        if routes_payload:
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
                    notes,
                    route_geometry,
                    started_at,
                    travel_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    notes = excluded.notes,
                    route_geometry = COALESCE(excluded.route_geometry, active_routes.route_geometry),
                    started_at = excluded.started_at,
                    travel_seconds = excluded.travel_seconds
                """,
                routes_payload,
            )

        if trucks_payload or routes_payload:
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
    "build_live_heatmap_source",
    "MockTelemetryIngestor",
    "TruckGpsSnapshot",
    "TruckTelemetryHarness",
    "extract_route_path",
    "load_truck_positions",
    "load_active_routes",
    "run_mock_ingestor",
]
