"""Helpers for loading and rendering saved job routes."""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from collections.abc import Iterable, Sequence
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence as SeqType, Tuple

import logging


from corkysoft.routing import ROUTE_BACKOFF, get_ors_client

logger = logging.getLogger(__name__)

def _iter_valid_coords(rows: Sequence[Mapping[str, Any]]) -> Iterable[tuple[float, float]]:
    """Yield ``(lat, lon)`` tuples for rows with coordinates."""

    for row in rows:
        for lat_key, lon_key in (("origin_lat", "origin_lon"), ("dest_lat", "dest_lon")):
            lat = row.get(lat_key)
            lon = row.get(lon_key)
            if lat is not None and lon is not None:
                yield float(lat), float(lon)


def compute_map_center(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    """Return ``[lat, lon]`` representing the average coordinate centre."""

    coords = list(_iter_valid_coords(rows))
    if not coords:
        # Rough centre of Australia as a sensible fallback for AU freight.
        return [-25.0, 135.0]

    sum_lat = sum(lat for lat, _ in coords)
    sum_lon = sum(lon for _, lon in coords)
    count = len(coords)
    return [sum_lat / count, sum_lon / count]


def combine_route_geojson(geojson_strings: Iterable[str]) -> Dict[str, Any]:
    """Merge individual GeoJSON feature collections into a single collection."""

    features: List[MutableMapping[str, Any]] = []

    for raw in geojson_strings:
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue

        if isinstance(payload, Mapping) and payload.get("type") == "FeatureCollection":
            for feat in payload.get("features", []) or []:
                if isinstance(feat, MutableMapping):
                    features.append(feat)
        elif isinstance(payload, Mapping) and payload.get("type") == "Feature":
            features.append(payload)  # pragma: no cover - unexpected but valid

    return {"type": "FeatureCollection", "features": features}


def build_job_route_map(rows: Sequence[Mapping[str, Any]], *, include_actual: bool) -> "folium.Map":
    """Return a Folium map visualising ``rows``.

    The map always plots straight-line routes between origin/destination points
    and optionally overlays saved route geometry when ``include_actual`` is true.
    """

    try:
        import folium
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise SystemExit("Folium not installed. Run: pip install folium") from exc

    center = compute_map_center(rows)
    fmap = folium.Map(location=center, zoom_start=5)

    markers_group = folium.FeatureGroup(name="Locations", show=True)
    crow_group = folium.FeatureGroup(name="Straight-line routes", show=not include_actual)
    actual_group = folium.FeatureGroup(name="Actual routed routes", show=include_actual)

    combined_geojson: List[str] = []

    for row in rows:
        o_lat = row.get("origin_lat")
        o_lon = row.get("origin_lon")
        d_lat = row.get("dest_lat")
        d_lon = row.get("dest_lon")

        origin_label = row.get("origin")
        dest_label = row.get("destination")
        origin_resolved = row.get("origin_resolved") or origin_label
        dest_resolved = row.get("destination_resolved") or dest_label

        if o_lat is not None and o_lon is not None:
            markers_group.add_child(
                folium.Marker(
                    [o_lat, o_lon],
                    popup=f"#{row['id']} Origin: {origin_resolved}",
                    icon=folium.Icon(color="blue"),
                )
            )
        if d_lat is not None and d_lon is not None:
            markers_group.add_child(
                folium.Marker(
                    [d_lat, d_lon],
                    popup=f"#{row['id']} Destination: {dest_resolved}",
                    icon=folium.Icon(color="red"),
                )
            )

        if None not in (o_lat, o_lon, d_lat, d_lon):
            crow_group.add_child(
                folium.PolyLine(
                    [[o_lat, o_lon], [d_lat, d_lon]],
                    color="#2ecc71",
                    weight=2.5,
                    opacity=0.7,
                    tooltip=f"#{row['id']} {origin_resolved} → {dest_resolved}",
                )
            )

        if include_actual:
            route_geojson = row.get("route_geojson")
            if isinstance(route_geojson, str):
                combined_geojson.append(route_geojson)

    markers_group.add_to(fmap)
    crow_group.add_to(fmap)

    if include_actual:
        combined = combine_route_geojson(combined_geojson)
        if combined["features"]:
            folium.GeoJson(
                combined,
                name="Actual routed routes",
                style_function=lambda _feature: {
                    "color": "#1b5e20",
                    "weight": 4,
                    "opacity": 0.75,
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=[],
                    aliases=[],
                    labels=False,
                    sticky=False,
                ),
            ).add_to(actual_group)
            actual_group.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)

    legend_html = """
    <div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999; background: white; padding: 8px 10px; border: 1px solid #bbb; border-radius: 6px; font-size: 12px;">
      <b>Legend</b><br>
      <span style="color:blue;">●</span> Origin<br>
      <span style="color:red;">●</span> Destination<br>
      <span style="color:#2ecc71;">▬</span> Straight-line route<br>
      <span style="color:#1b5e20;">▬</span> Actual routed path
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    return fmap


def fetch_job_route_rows(conn: sqlite3.Connection, *, include_actual: bool) -> list[Mapping[str, Any]]:
    """Return job rows with coordinates, tolerating legacy schemas.

    The query aliases display columns when ``origin_resolved`` or
    ``destination_resolved`` are absent and only includes ``route_geojson`` when
    the column exists. This keeps older databases compatible without requiring a
    manual migration step before rendering maps.
    """

    if conn.row_factory is None:
        conn.row_factory = sqlite3.Row

    table_info = conn.execute("PRAGMA table_info(jobs)").fetchall()
    available_columns = {row[1] for row in table_info}

    resolved_origin_expr = (
        "COALESCE(origin_resolved, origin) AS origin_resolved"
        if "origin_resolved" in available_columns
        else "origin AS origin_resolved"
    )
    resolved_destination_expr = (
        "COALESCE(destination_resolved, destination) AS destination_resolved"
        if "destination_resolved" in available_columns
        else "destination AS destination_resolved"
    )

    columns = [
        "id",
        "origin",
        "destination",
        "origin_lon",
        "origin_lat",
        "dest_lon",
        "dest_lat",
        resolved_origin_expr,
        resolved_destination_expr,
    ]
    if include_actual and "route_geojson" in available_columns:
        columns.append("route_geojson")

    query = f"""
        SELECT {', '.join(columns)}
        FROM jobs
        WHERE origin_lon IS NOT NULL AND dest_lon IS NOT NULL
    """

    cursor = conn.execute(query)
    return cursor.fetchall()


def _extract_coordinates(row: Mapping[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    """Return ``(origin_lon, origin_lat, dest_lon, dest_lat)`` when available."""

    try:
        origin_lon = float(row["origin_lon"])
        origin_lat = float(row["origin_lat"])
        dest_lon = float(row["dest_lon"])
        dest_lat = float(row["dest_lat"])
    except (KeyError, TypeError, ValueError):
        return None

    if any(value is None or (isinstance(value, float) and (value != value)) for value in (origin_lon, origin_lat, dest_lon, dest_lat)):
        return None

    return origin_lon, origin_lat, dest_lon, dest_lat


def _request_route_geojson(
    client: Any,
    origin_lon: float,
    origin_lat: float,
    dest_lon: float,
    dest_lat: float,
) -> Tuple[float, float, str]:
    """Return ``(distance_km, duration_hr, geojson)`` for the provided coordinates."""

    response = client.directions(
        coordinates=[[origin_lon, origin_lat], [dest_lon, dest_lat]],
        profile="driving-car",
        format="geojson",
    )
    features = response.get("features") if isinstance(response, Mapping) else None
    if not features:
        raise ValueError("ORS response missing features for route geometry request")
    first_feature = features[0]
    properties = first_feature.get("properties") if isinstance(first_feature, Mapping) else {}
    summary = properties.get("summary") if isinstance(properties, Mapping) else {}
    if not summary:
        raise ValueError("ORS response missing summary for route geometry request")

    distance_m = float(summary["distance"])
    duration_s = float(summary["duration"])
    geojson = json.dumps(response, separators=(",", ":"))
    time.sleep(ROUTE_BACKOFF)
    return distance_m / 1000.0, duration_s / 3600.0, geojson


def _store_historical_geometry(
    conn: sqlite3.Connection,
    job_id: int,
    *,
    distance_km: float,
    duration_hr: float,
    geojson: str,
    origin_lon: float,
    origin_lat: float,
    dest_lon: float,
    dest_lat: float,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO historical_job_routes (historical_job_id, geojson, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(historical_job_id) DO UPDATE SET
            geojson = excluded.geojson,
            updated_at = excluded.updated_at
        """,
        (job_id, geojson, timestamp, timestamp),
    )
    conn.execute(
        """
        UPDATE historical_jobs
        SET
            origin_lon = COALESCE(origin_lon, ?),
            origin_lat = COALESCE(origin_lat, ?),
            dest_lon = COALESCE(dest_lon, ?),
            dest_lat = COALESCE(dest_lat, ?),
            distance_km = ?,
            duration_hr = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (
            origin_lon,
            origin_lat,
            dest_lon,
            dest_lat,
            distance_km,
            duration_hr,
            timestamp,
            job_id,
        ),
    )


def _store_live_geometry(
    conn: sqlite3.Connection,
    job_id: int,
    *,
    distance_km: float,
    duration_hr: float,
    geojson: str,
    origin_lon: float,
    origin_lat: float,
    dest_lon: float,
    dest_lat: float,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        UPDATE jobs
        SET
            route_geojson = ?,
            distance_km = ?,
            duration_hr = ?,
            origin_lon = COALESCE(origin_lon, ?),
            origin_lat = COALESCE(origin_lat, ?),
            dest_lon = COALESCE(dest_lon, ?),
            dest_lat = COALESCE(dest_lat, ?),
            updated_at = ?
        WHERE id = ?
        """,
        (
            geojson,
            distance_km,
            duration_hr,
            origin_lon,
            origin_lat,
            dest_lon,
            dest_lat,
            timestamp,
            job_id,
        ),
    )


def populate_route_geometry(
    conn: sqlite3.Connection,
    job_ids: SeqType[int],
    *,
    dataset: str,
    client: Optional[Any] = None,
) -> int:
    """Populate ``route_geojson`` for the requested ``job_ids``.

    Parameters
    ----------
    conn:
        Active SQLite connection.
    job_ids:
        Iterable of job identifiers to update.
    dataset:
        Either ``"historical"`` or ``"live"`` designating the source table.
    client:
        Optional OpenRouteService client. When omitted ``get_ors_client`` is used.

    Returns
    -------
    int
        Count of routes whose geometry was updated.
    """

    identifiers = [int(jid) for jid in job_ids if jid is not None]
    if not identifiers:
        return 0

    placeholders = ",".join(["?"] * len(identifiers))
    logger.debug("Populating route geometry for %d %s job(s)", len(identifiers), dataset)

    if dataset not in {"historical", "live"}:
        raise ValueError("dataset must be either 'historical' or 'live'")

    if dataset == "historical":
        query = f"""
            SELECT
                hj.id,
                COALESCE(hj.origin_lon, o.lon) AS origin_lon,
                COALESCE(hj.origin_lat, o.lat) AS origin_lat,
                COALESCE(hj.dest_lon, d.lon) AS dest_lon,
                COALESCE(hj.dest_lat, d.lat) AS dest_lat,
                hr.geojson AS existing_geojson
            FROM historical_jobs AS hj
            LEFT JOIN addresses AS o ON hj.origin_address_id = o.id
            LEFT JOIN addresses AS d ON hj.destination_address_id = d.id
            LEFT JOIN historical_job_routes AS hr ON hr.historical_job_id = hj.id
            WHERE hj.id IN ({placeholders})
        """
    else:
        query = f"""
            SELECT
                id,
                origin_lon,
                origin_lat,
                dest_lon,
                dest_lat,
                route_geojson AS existing_geojson
            FROM jobs
            WHERE id IN ({placeholders})
        """

    rows = conn.execute(query, identifiers).fetchall()
    if not rows:
        return 0

    updated = 0
    ors_client = get_ors_client(client)

    for row in rows:
        job_id = int(row["id"])
        existing = row["existing_geojson"]
        if isinstance(existing, str) and existing.strip():
            continue

        coords = _extract_coordinates(row)
        if coords is None:
            logger.debug("Skipping job %s: missing coordinates", job_id)
            continue

        origin_lon, origin_lat, dest_lon, dest_lat = coords
        try:
            distance_km, duration_hr, geojson = _request_route_geojson(
                ors_client,
                origin_lon,
                origin_lat,
                dest_lon,
                dest_lat,
            )
        except Exception as exc:  # pragma: no cover - defensive logging only
            logger.warning("Failed to fetch route geometry for job %s: %s", job_id, exc)
            continue

        if dataset == "historical":
            _store_historical_geometry(
                conn,
                job_id,
                distance_km=distance_km,
                duration_hr=duration_hr,
                geojson=geojson,
                origin_lon=origin_lon,
                origin_lat=origin_lat,
                dest_lon=dest_lon,
                dest_lat=dest_lat,
            )
        else:
            _store_live_geometry(
                conn,
                job_id,
                distance_km=distance_km,
                duration_hr=duration_hr,
                geojson=geojson,
                origin_lon=origin_lon,
                origin_lat=origin_lat,
                dest_lon=dest_lon,
                dest_lat=dest_lat,
            )

        updated += 1

    conn.commit()
    return updated
