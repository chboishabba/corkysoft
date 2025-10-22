"""Helpers for loading and rendering saved job routes."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from typing import Any, Dict, List, Mapping, MutableMapping


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
