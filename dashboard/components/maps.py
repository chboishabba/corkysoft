"""Reusable map components for the dashboard interfaces."""
from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from analytics.live_data import (
    TRUCK_STATUS_COLOURS,
    build_live_heatmap_source,
    extract_route_path,
)
from analytics.price_distribution import (
    PROFITABILITY_COLOURS,
    compute_profitability_line_width,
    compute_tapered_route_polygon,
)
from dashboard.map_provider import (
    _resolved_provider,
    google_maps_requested_without_key,
    plotly_map_layout,
    pydeck_map_kwargs,
    using_google_maps,
)


logger = logging.getLogger(__name__)

__all__ = [
    "build_route_map",
    "render_network_map",
    "_initial_view_state",
    "_hex_to_rgb",
    "_geojson_to_path",
]


_AUS_LAT_LON = (-25.2744, 133.7751)


def _hex_to_rgb(value: str) -> Tuple[int, int, int]:
    """Convert ``value`` (hex or rgb string) into an ``(r, g, b)`` tuple."""

    colour = value.strip()
    if colour.startswith("#"):
        digits = colour.lstrip("#")
        if len(digits) == 3:
            digits = "".join(ch * 2 for ch in digits)
        if len(digits) != 6:
            raise ValueError(f"Unsupported hex colour format: {value}")
        return tuple(int(digits[idx : idx + 2], 16) for idx in (0, 2, 4))  # type: ignore[arg-type]

    if colour.startswith("rgb"):
        start = colour.find("(")
        end = colour.find(")")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Unsupported rgb colour format: {value}")
        components = colour[start + 1 : end].split(",")[:3]
        return tuple(int(float(component.strip())) for component in components)

    raise ValueError(f"Unsupported colour format: {value}")


def _geojson_to_path(value: Any) -> Optional[List[List[float]]]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        try:
            value = json.dumps(value)
        except (TypeError, ValueError):
            return None
    elif isinstance(value, (bytes, bytearray, memoryview)):
        try:
            value = bytes(value).decode("utf-8")
        except Exception:
            return None

    if not isinstance(value, str):
        return None

    payload = value.strip()
    if not payload or payload.startswith("<"):
        return None
    if '"type"' not in payload and '"features"' not in payload and '"geometry"' not in payload:
        return None
    try:
        coords = extract_route_path(payload)
    except Exception:
        return None
    return [[float(lon), float(lat)] for lat, lon in coords]


def _has_geometry_payload(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            value = bytes(value).decode("utf-8")
        except Exception:
            return False
    if isinstance(value, str):
        payload = value.strip()
        return bool(payload) and not payload.startswith("<")
    return False


def _sanitise_geojson_paths(
    series: pd.Series,
    *,
    label: str,
) -> pd.Series:
    paths: list[Optional[List[List[float]]]] = []
    skipped = 0
    for value in series.tolist():
        path = _geojson_to_path(value)
        if path is None and _has_geometry_payload(value):
            skipped += 1
        paths.append(path)

    if skipped:
        logger.warning(
            "Skipping %d malformed %s GeoJSON payload(s) before pydeck render.",
            skipped,
            label,
        )

    return pd.Series(paths, index=series.index, dtype=object)


def _coerce_position_sequence(
    value: object,
    *,
    minimum_points: int,
) -> Optional[List[List[float]]]:
    """Coerce a geometry payload into a list-of-coordinate-pairs.

    Accepts lists, tuples, or JSON-encoded payloads containing valid point
    sequences. Returns None when the payload is malformed or does not contain the
    required minimum number of points.
    """

    candidate: object = value

    if candidate is None:
        return None

    if isinstance(candidate, (bytes, bytearray, memoryview)):
        try:
            candidate = bytes(candidate).decode("utf-8")
        except Exception:
            return None

    if isinstance(candidate, str):
        payload = candidate.strip()
        if not payload or payload.startswith("<"):
            return None
        if (
            (payload.startswith("{") and payload.endswith("}"))
            or (payload.startswith("[") and payload.endswith("]"))
        ):
            try:
                candidate = json.loads(payload)
            except json.JSONDecodeError:
                return None
        else:
            if not payload:
                return None

    if isinstance(candidate, str):
        parts = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", candidate)
        if len(parts) < minimum_points * 2:
            return None
        numeric_points = [float(part) for part in parts[: minimum_points * 2]]
        pairs = [[numeric_points[i], numeric_points[i + 1]] for i in range(0, len(numeric_points), 2)]
        return pairs if len(pairs) >= minimum_points else None

    if isinstance(candidate, Mapping):
        geometry = candidate.get("geometry") if isinstance(candidate, Mapping) else None
        if isinstance(geometry, Mapping):
            candidate = geometry.get("coordinates") or []
        elif (
            "lat" in candidate or "longitude" in candidate or "lng" in candidate or "lon" in candidate
        ):
            candidate = [candidate]
        else:
            return None

    if not isinstance(candidate, (list, tuple)):
        return None
    if not candidate:
        return None

    while (
        len(candidate) == 1
        and isinstance(candidate[0], (list, tuple))
        and candidate[0]
        and isinstance(candidate[0][0], (list, tuple))
    ):
        candidate = candidate[0]  # type: ignore[assignment]

    coordinates: list[list[float]] = []
    for point in candidate:
        if isinstance(point, Mapping):
            lon = point.get("lon") or point.get("lng") or point.get("longitude")
            lat = point.get("lat") or point.get("latitude")
        elif (
            isinstance(point, (list, tuple))
            and len(point) >= 2
            and not isinstance(point[0], (list, tuple, dict))
        ):
            lon = point[0]
            lat = point[1]
        else:
            return None

        try:
            coordinates.append([float(lon), float(lat)])
        except (TypeError, ValueError):
            return None

    if not _is_valid_position_sequence(coordinates, minimum_points=minimum_points):
        return None
    return coordinates


def _build_colour_map(values: Sequence[object]) -> Dict[object, str]:
    palette = px.colors.qualitative.Bold or [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
    ]
    if not palette:
        palette = ["#636EFA"]
    if values:
        if len(values) > len(palette):
            repeats = (len(values) // len(palette)) + 1
            palette = (palette * repeats)[: len(values)]
        return {
            value: palette[idx % len(palette)]
            for idx, value in enumerate(values)
        }
    return {}


def build_route_map(
    df: pd.DataFrame,
    colour_label: str,
    *,
    show_routes: bool,
    show_points: bool,
    colour_mode: str = "categorical",
    colour_scale: Optional[Sequence[str]] = None,
    colorbar_tickformat: Optional[str] = None,
    use_route_geometry: bool = True,
    provider: Optional[str] = None,
) -> go.Figure:
    """Construct a Plotly map figure showing coloured routes and points."""

    logger.info(
        "ROUTE MAP DEBUG build_route_map start: shape=%s, columns=%s, show_routes=%s, "
        "show_points=%s, colour_mode=%s, use_route_geometry=%s, colour_scale_provided=%s, "
        "colorbar_tickformat=%s",
        df.shape,
        list(df.columns),
        show_routes,
        show_points,
        colour_mode,
        use_route_geometry,
        bool(colour_scale),
        colorbar_tickformat,
    )

    def _coerce_geojson(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, (bytes, bytearray, memoryview)):
            try:
                decoded = value.decode("utf-8")
            except Exception:
                return None
            stripped = decoded.strip()
            return stripped or None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        if isinstance(value, Mapping):
            try:
                return json.dumps(value)
            except (TypeError, ValueError):
                return None
        return None

    def _row_route_points(row: pd.Series) -> List[Tuple[float, float]]:
        """Return the ordered ``(lat, lon)`` points for ``row`` when available."""

        job_id = row.get("id", "n/a")
        if not use_route_geometry:
            logger.info(
                "ROUTE MAP DEBUG route geometry skipped: job_id=%s use_route_geometry disabled",
                job_id,
            )
            return []

        geojson_value: Optional[str] = None
        for column in ("route_geojson", "route_geometry", "geojson"):
            candidate = _coerce_geojson(row.get(column))
            if candidate:
                geojson_value = candidate
                break
        route_points: List[Tuple[float, float]] = []
        geometry_source = None
        if geojson_value:
            try:
                route_points = extract_route_path(geojson_value)
                geometry_source = "geojson"
            except Exception:
                route_points = []

        path_value = row.get("route_path")
        if isinstance(path_value, str):
            raw = path_value.strip()
            if raw:
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = None
                else:
                    path_value = parsed
            else:
                path_value = None

        if isinstance(path_value, dict) and not route_points:
            try:
                route_points = extract_route_path(json.dumps(path_value))
                geometry_source = geometry_source or "route_path_dict"
            except Exception:
                route_points = []

        if isinstance(path_value, (list, tuple)) and not route_points:
            coords: List[Tuple[float, float]] = []
            for point in path_value:
                lon: Optional[float]
                lat: Optional[float]
                if isinstance(point, dict):
                    lon = point.get("lon") or point.get("lng") or point.get("longitude")
                    lat = point.get("lat") or point.get("latitude")
                elif isinstance(point, (list, tuple)) and len(point) >= 2:
                    lon, lat = point[0], point[1]
                else:
                    continue

                try:
                    coords.append((float(lat), float(lon)))
                except (TypeError, ValueError):
                    continue

            if coords:
                route_points = coords
                geometry_source = geometry_source or "route_path_iterable"

        if route_points:
            sample_points = route_points[:3]
            logger.info(
                "ROUTE MAP DEBUG route geometry summary: job_id=%s source=%s points=%d sample=%s",
                job_id,
                geometry_source or "geojson",
                len(route_points),
                sample_points,
            )
        else:
            logger.info(
                "ROUTE MAP DEBUG route geometry missing: job_id=%s source=%s",
                job_id,
                geometry_source,
            )

        return route_points

    def _coerce_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _normalise_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        if isinstance(value, float) and math.isnan(value):
            return None
        return str(value)

    def _route_context(row: pd.Series, *, display_value: Any = None) -> Dict[str, str]:
        origin_label = (
            row.get("origin_city")
            or row.get("origin")
            or row.get("origin_raw")
            or "Origin"
        )
        destination_label = (
            row.get("destination_city")
            or row.get("destination")
            or row.get("destination_raw")
            or "Destination"
        )

        display_text = _normalise_text(display_value)
        if display_text is None:
            display_text = _normalise_text(row.get("map_colour_display"))
        if display_text is None:
            display_text = _normalise_text(row.get("map_colour_value"))

        job_id_text = _normalise_text(row.get("id")) or "n/a"

        origin_label = str(origin_label)
        destination_label = str(destination_label)
        route_title = f"Route: {origin_label} → {destination_label}"

        tooltip_parts = [route_title]
        if display_text:
            tooltip_parts.append(f"{colour_label}: {display_text}")
        tooltip_parts.append(f"Job ID: {job_id_text}")

        return {
            "origin_label": origin_label,
            "destination_label": destination_label,
            "route_title": route_title,
            "base_text": "<br>".join(tooltip_parts),
        }

    figure = go.Figure()
    plot_df = df.copy()
    colour_values = list(dict.fromkeys(plot_df["map_colour_value"].tolist()))
    colour_map = _build_colour_map(colour_values)

    def _route_heading(row: pd.Series) -> tuple[str, str]:
        origin_label = (
            row.get("origin_city")
            or row.get("origin")
            or row.get("origin_raw")
            or "Origin"
        )
        destination_label = (
            row.get("destination_city")
            or row.get("destination")
            or row.get("destination_raw")
            or "Destination"
        )
        return str(origin_label), str(destination_label)

    def _route_hover_text(row: pd.Series, display_value: object) -> tuple[str, str, str, str]:
        origin_label, destination_label = _route_heading(row)
        route_label = f"Route: {origin_label} → {destination_label}"
        job_id = row.get("id", "n/a")
        header = f"{colour_label}: {display_value}"
        base_text = f"{header}<br>{route_label}<br>Job ID: {job_id}"
        return origin_label, destination_label, route_label, base_text

    if colour_mode == "categorical":
        if show_routes:
            for value in colour_values:
                category_df = plot_df[plot_df["map_colour_value"] == value]
                if category_df.empty:
                    continue
                display_value = (
                    category_df.get("map_colour_display", pd.Series([value])).iloc[0]
                )
                lat_values: list[float | None] = []
                lon_values: list[float | None] = []
                text_values: list[str | None] = []
                for _, row in category_df.iterrows():
                    _, _, _, base_text = _route_hover_text(row, display_value)
                    route_points = _row_route_points(row)
                    context = _route_context(row, display_value=display_value)
                    base_text = context["base_text"]
                    if route_points:
                        for lat, lon in route_points:
                            lat_values.append(lat)
                            lon_values.append(lon)
                            text_values.append(base_text)
                        lat_values.append(None)
                        lon_values.append(None)
                        text_values.append(None)
                        continue

                    origin_lat = _coerce_float(row.get("origin_lat"))
                    origin_lon = _coerce_float(row.get("origin_lon"))
                    dest_lat = _coerce_float(row.get("dest_lat"))
                    dest_lon = _coerce_float(row.get("dest_lon"))
                    if (
                        origin_lat is not None
                        and origin_lon is not None
                        and dest_lat is not None
                        and dest_lon is not None
                    ):
                        lat_values.extend([origin_lat, dest_lat, None])
                        lon_values.extend([origin_lon, dest_lon, None])
                        text_values.extend([base_text, base_text, None])

                if lat_values and lon_values:
                    colour = colour_map.get(value, "#636EFA")
                    figure.add_trace(
                        go.Scattermap(
                            lat=lat_values,
                            lon=lon_values,
                            mode="lines",
                            line={"width": 2, "color": colour},
                            name=str(display_value),
                            legendgroup=str(value),
                            showlegend=False,
                            text=text_values,
                            hovertemplate="%{text}<extra></extra>",
                        )
                    )

        if show_points:
            for value in colour_values:
                category_df = plot_df[plot_df["map_colour_value"] == value]
                if category_df.empty:
                    continue
                display_value = (
                    category_df.get("map_colour_display", pd.Series([value])).iloc[0]
                )
                marker_lat: list[float] = []
                marker_lon: list[float] = []
                marker_text: list[str] = []
                for _, row in category_df.iterrows():
                    context = _route_context(row, display_value=display_value)
                    base_text = context["base_text"]
                    origin_label = context["origin_label"]
                    destination_label = context["destination_label"]

                    origin_lat = _coerce_float(row.get("origin_lat"))
                    origin_lon = _coerce_float(row.get("origin_lon"))
                    if origin_lat is not None and origin_lon is not None:
                        marker_lat.append(origin_lat)
                        marker_lon.append(origin_lon)
                        marker_text.append(
                            "<br>".join(
                                [
                                    base_text,
                                    f"Stop: Origin — {origin_label}",
                                ]
                            )
                        )

                    dest_lat = _coerce_float(row.get("dest_lat"))
                    dest_lon = _coerce_float(row.get("dest_lon"))
                    if dest_lat is not None and dest_lon is not None:
                        marker_lat.append(dest_lat)
                        marker_lon.append(dest_lon)
                        marker_text.append(
                            "<br>".join(
                                [
                                    base_text,
                                    f"Stop: Destination — {destination_label}",
                                ]
                            )
                        )

                if marker_lat and marker_lon:
                    colour = colour_map.get(value, "#636EFA")
                    figure.add_trace(
                        go.Scattermap(
                            lat=marker_lat,
                            lon=marker_lon,
                            mode="markers",
                            marker={"size": 10, "color": colour},
                            name=str(display_value),
                            legendgroup=str(value),
                            text=marker_text,
                            hovertemplate="%{text}<extra></extra>",
                        )
                    )

    elif colour_mode == "continuous":
        numeric_series = pd.to_numeric(plot_df["map_colour_value"], errors="coerce")
        numeric_values = numeric_series.dropna()
        if not numeric_values.empty:
            plot_df["map_colour_value"] = numeric_values
            colour_scale = colour_scale or px.colors.sequential.Viridis
            min_value = float(numeric_values.min())
            max_value = float(numeric_values.max())

            def _to_colour(value: float) -> str:
                if not colour_scale:
                    return "#636EFA"
                if math.isclose(min_value, max_value):
                    return colour_scale[-1]
                position = (value - min_value) / (max_value - min_value)
                return px.colors.sample_colorscale(colour_scale, [position])[0]

            if min_value < 0.0 < max_value:
                extent = max(abs(min_value), abs(max_value))
                cmin_value = -extent
                cmax_value = extent
                cmid_value: Optional[float] = 0.0
            else:
                cmin_value = min_value
                cmax_value = max_value
                cmid_value = None

            colorbar_dict = {"title": colour_label}
            if colorbar_tickformat:
                colorbar_dict["tickformat"] = colorbar_tickformat

            if show_routes:
                lat_values: list[float | None] = []
                lon_values: list[float | None] = []
                text_values: list[str | None] = []
                colour_values_for_lines: list[str] = []
                for _, row in plot_df.iterrows():
                    try:
                        value = float(row["map_colour_value"])
                    except (TypeError, ValueError):
                        continue
                    colour = _to_colour(value)
                    display_candidate = row.get("map_colour_display")
                    display_value = display_candidate if display_candidate is not None else value
                    context = _route_context(row, display_value=display_value)
                    base_text = context["base_text"]
                    route_points = _row_route_points(row)
                    if route_points:
                        for lat, lon in route_points:
                            lat_values.append(lat)
                            lon_values.append(lon)
                            text_values.append(base_text)
                            colour_values_for_lines.append(colour)
                        lat_values.append(None)
                        lon_values.append(None)
                        text_values.append(None)
                        colour_values_for_lines.append(colour)
                        continue

                    origin_lat = _coerce_float(row.get("origin_lat"))
                    origin_lon = _coerce_float(row.get("origin_lon"))
                    dest_lat = _coerce_float(row.get("dest_lat"))
                    dest_lon = _coerce_float(row.get("dest_lon"))
                    if (
                        origin_lat is not None
                        and origin_lon is not None
                        and dest_lat is not None
                        and dest_lon is not None
                    ):
                        lat_values.extend([origin_lat, dest_lat, None])
                        lon_values.extend([origin_lon, dest_lon, None])
                        text_values.extend([base_text, base_text, None])
                        colour_values_for_lines.extend([colour, colour, colour])

                if lat_values and lon_values:
                    line_colour = colour_values_for_lines[0] if colour_values_for_lines else colour_scale[0]
                    figure.add_trace(
                        go.Scattermap(
                            lat=lat_values,
                            lon=lon_values,
                            mode="lines",
                            line={"width": 3, "color": line_colour},
                            name=colour_label,
                            text=text_values,
                            hovertemplate="%{text}<extra></extra>",
                            showlegend=False,
                        )
                    )

            if show_points:
                marker_lat: list[float] = []
                marker_lon: list[float] = []
                marker_colour: list[float] = []
                marker_text: list[str] = []
                for _, row in plot_df.iterrows():
                    try:
                        value = float(row["map_colour_value"])
                    except (TypeError, ValueError):
                        continue
                    context = _route_context(row, display_value=value)
                    base_text = context["base_text"]
                    origin_label = context["origin_label"]
                    destination_label = context["destination_label"]
                    origin_lat = _coerce_float(row.get("origin_lat"))
                    origin_lon = _coerce_float(row.get("origin_lon"))
                    dest_lat = _coerce_float(row.get("dest_lat"))
                    dest_lon = _coerce_float(row.get("dest_lon"))
                    if origin_lat is not None and origin_lon is not None:
                        marker_lat.append(origin_lat)
                        marker_lon.append(origin_lon)
                        marker_colour.append(value)
                        marker_text.append(
                            "<br>".join([base_text, f"Stop: Origin — {origin_label}"])
                        )
                    if dest_lat is not None and dest_lon is not None:
                        marker_lat.append(dest_lat)
                        marker_lon.append(dest_lon)
                        marker_colour.append(value)
                        marker_text.append(
                            "<br>".join([base_text, f"Stop: Destination — {destination_label}"])
                        )

                if marker_lat and marker_lon and marker_colour:
                    marker_dict: dict[str, Any] = {
                        "size": 12,
                        "color": marker_colour,
                        "colorscale": colour_scale,
                        "colorbar": colorbar_dict,
                        "cmin": cmin_value,
                        "cmax": cmax_value,
                    }
                    if cmid_value is not None:
                        marker_dict["cmid"] = cmid_value
                    figure.add_trace(
                        go.Scattermap(
                            lat=marker_lat,
                            lon=marker_lon,
                            mode="markers",
                            marker=marker_dict,
                            text=marker_text,
                            hovertemplate="%{text}<extra></extra>",
                            name=colour_label,
                        )
                    )
    else:
        raise ValueError(f"Unsupported colour mode: {colour_mode}")

    figure.update_layout(
        **plotly_map_layout(
            {"lat": -25.0, "lon": 133.0},
            zoom=3,
            engine="map",
            provider=provider,
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        legend={"orientation": "h", "yanchor": "bottom", "y": 0.01},
    )
    trace_debug: List[Dict[str, Any]] = []
    for idx, trace in enumerate(figure.data):
        lat_values = list(getattr(trace, "lat", []) or [])
        lon_values = list(getattr(trace, "lon", []) or [])
        lat_count = sum(1 for value in lat_values if value is not None)
        lon_count = sum(1 for value in lon_values if value is not None)
        trace_debug.append(
            {
                "index": idx,
                "name": getattr(trace, "name", f"trace_{idx}"),
                "lat_count": lat_count,
                "lon_count": lon_count,
            }
        )
    logger.info(
        "ROUTE MAP DEBUG figure traces: total=%s details=%s",
        len(figure.data),
        trace_debug,
    )
    return figure


def _initial_view_state(df: pd.DataFrame) -> pdk.ViewState:
    if df.empty:
        return pdk.ViewState(latitude=_AUS_LAT_LON[0], longitude=_AUS_LAT_LON[1], zoom=4.0)
    lat_column = "lat" if "lat" in df.columns else "origin_lat"
    lon_column = "lon" if "lon" in df.columns else "origin_lon"
    lat = pd.to_numeric(df[lat_column], errors="coerce").dropna()
    lon = pd.to_numeric(df[lon_column], errors="coerce").dropna()
    if lat.empty or lon.empty:
        return pdk.ViewState(latitude=_AUS_LAT_LON[0], longitude=_AUS_LAT_LON[1], zoom=4.0)
    return pdk.ViewState(latitude=float(lat.mean()), longitude=float(lon.mean()), zoom=4.5)


def _pydeck_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or df.columns.is_unique:
        return df
    return df.loc[:, ~df.columns.duplicated()].copy()


def _base_map_layer_for_network(
    deck_map_kwargs: Mapping[str, Any],
    *,
    provider: Optional[str] = None,
) -> Optional[pdk.Layer]:
    """Return the explicit OSM base layer only for non-Google providers."""

    if provider is None:
        if using_google_maps() or google_maps_requested_without_key():
            return None
    elif provider == "google" or google_maps_requested_without_key(provider):
        return None
    if "map_provider" in deck_map_kwargs:
        return None
    return pdk.Layer(
        "TileLayer",
        data="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        min_zoom=0,
        max_zoom=19,
        tile_size=256,
        attribution="© OpenStreetMap contributors",
    )


def _is_valid_position_sequence(value: object, *, minimum_points: int) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) < minimum_points:
        return False
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return False
        try:
            lon = float(point[0])
            lat = float(point[1])
        except (TypeError, ValueError):
            return False
        if not (math.isfinite(lon) and math.isfinite(lat)):
            return False
    return True


def _valid_xy_row(row: pd.Series, *, lon_key: str, lat_key: str) -> bool:
    try:
        lon = float(row.get(lon_key))
        lat = float(row.get(lat_key))
    except (TypeError, ValueError):
        return False
    return math.isfinite(lon) and math.isfinite(lat)


def _filter_valid_geometry_rows(
    df: pd.DataFrame,
    *,
    column: str,
    minimum_points: int,
) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return df
    normalised = df.copy()
    normalised[column] = normalised[column].apply(
        lambda value: _coerce_position_sequence(value, minimum_points=minimum_points)
    )
    return _pydeck_frame(
        normalised[normalised[column].notna()].copy()
    )


def _clean_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _truck_label(row: pd.Series) -> str:
    rego = _clean_value(row.get("rego"))
    return rego or str(row.get("truck_id"))


def _format_truck_tooltip(row: pd.Series) -> str:
    label = _truck_label(row)
    status = _clean_value(row.get("status")) or "unknown"
    lines = [f"{label} ({status})"]

    make = _clean_value(row.get("make"))
    model = _clean_value(row.get("model"))
    year = _clean_value(row.get("year"))
    vehicle_bits = [part for part in (make, model) if part]
    if year:
        vehicle_bits.append(year)
    if vehicle_bits:
        lines.append("Vehicle: " + " ".join(vehicle_bits))

    body_type = _clean_value(row.get("body_type"))
    if body_type:
        lines.append(f"Body: {body_type}")

    nhv_code = _clean_value(row.get("nhv_code"))
    if nhv_code:
        lines.append(f"NHV code: {nhv_code}")

    odometer = _clean_value(row.get("odometer"))
    if odometer:
        lines.append(f"Odometer: {odometer}")

    next_service = _clean_value(row.get("next_service"))
    last_service = _clean_value(row.get("last_service"))
    if next_service:
        lines.append(f"Next service: {next_service}")
    if last_service:
        lines.append(f"Last service: {last_service}")

    coi_due = _clean_value(row.get("coi_due"))
    coi_number = _clean_value(row.get("coi_number"))
    if coi_due or coi_number:
        coi_parts = [part for part in (coi_due, coi_number) if part]
        lines.append(f"COI: {' / '.join(coi_parts)}")

    present_driver = _clean_value(row.get("present_driver"))
    if present_driver:
        lines.append(f"Driver: {present_driver}")

    daily_check = row.get("daily_check_complete")
    if daily_check is not None and not (isinstance(daily_check, float) and pd.isna(daily_check)):
        daily_label = "Yes" if bool(daily_check) else "No"
        lines.append(f"Daily check: {daily_label}")

    return "<br>".join(lines)


def render_network_map(
    historical_routes: pd.DataFrame,
    trucks: pd.DataFrame,
    active_routes: pd.DataFrame,
    *,
    toggle_key: str = "network_map_live_overlay_toggle",
) -> None:
    st.markdown("### Live network overview")
    effective_provider = _resolved_provider()

    show_live_overlay: bool = True
    toggle_help = (
        "Toggle the live overlay of active routes and truck telemetry without hiding the "
        "base map."
    )
    view_mode = st.radio(
        "Map view",
        ("Overlay", "Heatmap"),
        horizontal=True,
        key=f"{toggle_key}_view_mode",
        help="Switch between the layered network overlay and an aggregated density heatmap.",
    )
    if hasattr(st, "toggle"):
        show_live_overlay = st.toggle(
            "Show live network overlay",
            value=True,
            help=toggle_help,
            key=toggle_key,
        )
    else:
        # Older versions of Streamlit do not expose st.toggle; fall back to a checkbox.
        show_live_overlay = st.checkbox(
            "Show live network overlay",
            value=True,
            help=toggle_help,
            key=toggle_key,
        )

    show_actual_routes = False
    if show_live_overlay and (
        (
            not historical_routes.empty
            and "route_geojson" in historical_routes.columns
            and historical_routes["route_geojson"].notna().any()
        )
        or (
            not active_routes.empty
            and "route_geometry" in active_routes.columns
            and active_routes["route_geometry"].notna().any()
        )
    ):
        show_actual_routes = st.checkbox(
            "Show actual route traces",
            value=False,
            help="Overlay available polylines for historical lanes and live routes.",
            key=f"{toggle_key}_actual_routes",
        )

    deck_map_kwargs = pydeck_map_kwargs(None, provider=effective_provider)
    base_map_layer = _base_map_layer_for_network(
        deck_map_kwargs,
        provider=effective_provider,
    )

    truck_data = _pydeck_frame(trucks.copy())
    if not truck_data.empty:
        truck_data["truck_label"] = truck_data.apply(_truck_label, axis=1)
        truck_data["colour"] = truck_data["status"].map(TRUCK_STATUS_COLOURS)
        truck_data["colour"] = truck_data["colour"].apply(
            lambda value: value if isinstance(value, (list, tuple)) else [0, 122, 204]
        )
        truck_data["tooltip"] = truck_data.apply(_format_truck_tooltip, axis=1)

    historical_overlay = _pydeck_frame(historical_routes.copy())
    if not historical_overlay.empty and "route_geojson" in historical_overlay.columns:
        historical_overlay["route_path"] = _sanitise_geojson_paths(
            historical_overlay["route_geojson"],
            label="historical route",
        )
    else:
        historical_overlay["route_path"] = None

    if not historical_overlay.empty and "lane_key" in historical_overlay.columns:
        lane_overlay = _pydeck_frame(historical_overlay.copy())
        lane_overlay["_has_geojson"] = (
            lane_overlay["route_path"].notna() if "route_path" in lane_overlay.columns else False
        )
        lane_overlay["_job_weight"] = pd.to_numeric(
            lane_overlay.get("job_count", 0), errors="coerce"
        ).fillna(0.0)
        lane_overlay = (
            lane_overlay.sort_values(
                by=["_has_geojson", "_job_weight"], ascending=[False, False]
            )
            .drop(columns=["_has_geojson", "_job_weight"])
            .drop_duplicates(subset=["lane_key"])
        )
    else:
        lane_overlay = historical_overlay.copy()

    historical_has_paths = (
        not lane_overlay.empty
        and "route_path" in lane_overlay.columns
        and lane_overlay["route_path"].notna().any()
    )
    active_has_geometry = (
        not active_routes.empty
        and "route_geometry" in active_routes.columns
        and active_routes["route_geometry"].notna().any()
    )
    overlay_layers: list[pdk.Layer] = []

    if show_live_overlay and not lane_overlay.empty:
        lane_overlay = _filter_valid_geometry_rows(
            lane_overlay,
            column="route_polygon",
            minimum_points=3,
        )

    if view_mode == "Overlay":
        if show_live_overlay and not lane_overlay.empty:
            history_layer = pdk.Layer(
                "PolygonLayer",
                data=lane_overlay,
                get_polygon="route_polygon",
                get_fill_color="fill_colour",
                stroked=False,
                filled=True,
                pickable=True,
                extruded=False,
                parameters={"depthTest": False},
            )
            overlay_layers.append(history_layer)

        if show_live_overlay and not active_routes.empty:
            if "job_id" in active_routes.columns and not historical_routes.empty:
                enriched = _pydeck_frame(
                    active_routes.merge(
                        historical_routes[
                            [
                                "id",
                            "colour",
                            "fill_colour",
                            "line_width",
                            "route_polygon",
                            "profit_band",
                            "profitability_status",
                            "tooltip",
                        ]
                    ],
                    left_on="job_id",
                    right_on="id",
                    how="left",
                        suffixes=("", "_hist"),
                    )
                )
                if "colour" not in enriched.columns and "colour_hist" in enriched.columns:
                    enriched["colour"] = enriched["colour_hist"]
                if "profit_band" not in enriched.columns and "profit_band_hist" in enriched.columns:
                    enriched["profit_band"] = enriched["profit_band_hist"]
                if (
                    "profitability_status" not in enriched.columns
                    and "profitability_status_hist" in enriched.columns
                ):
                    enriched["profitability_status"] = enriched["profitability_status_hist"]
                if "tooltip" not in enriched.columns and "tooltip_hist" in enriched.columns:
                    enriched["tooltip"] = enriched["tooltip_hist"]
                if "fill_colour" not in enriched.columns and "fill_colour_hist" in enriched.columns:
                    enriched["fill_colour"] = enriched["fill_colour_hist"]
                if "line_width" not in enriched.columns and "line_width_hist" in enriched.columns:
                    enriched["line_width"] = enriched["line_width_hist"]
                if "route_polygon" not in enriched.columns and "route_polygon_hist" in enriched.columns:
                    enriched["route_polygon"] = enriched["route_polygon_hist"]
            else:
                enriched = _pydeck_frame(active_routes.copy())
                enriched["colour"] = [PROFITABILITY_COLOURS["Unknown"]] * len(enriched)
                enriched["profit_band"] = "Unknown"
                enriched["profitability_status"] = "Unknown"
                enriched["tooltip"] = "Active route"

            default_width = compute_profitability_line_width("Unknown")
            if "line_width" not in enriched.columns:
                enriched["line_width"] = default_width
            else:
                enriched["line_width"] = enriched["line_width"].fillna(default_width)

            def _ensure_fill_colour(value: object) -> list[int]:
                if isinstance(value, (list, tuple)):
                    rgba = list(value)
                else:
                    rgba = []
                if len(rgba) < 3:
                    rgba = [255, 255, 255]
                else:
                    rgba = [int(component) for component in rgba[:4]]
                if len(rgba) == 3:
                    rgba.append(180)
                if len(rgba) < 4:
                    rgba.extend([180] * (4 - len(rgba)))
                if len(rgba) > 4:
                    rgba = rgba[:4]
                return rgba

            if "fill_colour" not in enriched.columns:
                enriched["fill_colour"] = [
                    _ensure_fill_colour(PROFITABILITY_COLOURS["Unknown"]) for _ in range(len(enriched))
                ]
            else:
                enriched["fill_colour"] = enriched["fill_colour"].apply(_ensure_fill_colour)

            enriched["route_polygon"] = enriched.apply(
                lambda row: row.get("route_polygon")
                if isinstance(row.get("route_polygon"), list) and row.get("route_polygon")
                else compute_tapered_route_polygon(row),
                axis=1,
            )
            enriched = _filter_valid_geometry_rows(
                enriched,
                column="route_polygon",
                minimum_points=3,
            )

            enriched["colour"] = enriched["colour"].apply(
                lambda value: value if isinstance(value, (list, tuple)) else [255, 255, 255]
            )

            if not enriched.empty:
                active_layer = pdk.Layer(
                    "PolygonLayer",
                    data=enriched,
                    get_polygon="route_polygon",
                    get_fill_color="fill_colour",
                    stroked=False,
                    filled=True,
                    pickable=True,
                    extruded=False,
                    parameters={"depthTest": False},
                )
                overlay_layers.append(active_layer)

            if show_live_overlay and not lane_overlay.empty:
                if show_actual_routes and historical_has_paths:
                    history_paths = _filter_valid_geometry_rows(
                        _pydeck_frame(lane_overlay.dropna(subset=["route_path"])),
                        column="route_path",
                        minimum_points=2,
                    )
                    if not history_paths.empty:
                        history_layer = pdk.Layer(
                            "PathLayer",
                            data=history_paths,
                            get_path="route_path",
                            get_color="colour",
                            get_width="line_width",
                            width_min_pixels=1,
                            pickable=True,
                            opacity=0.4,
                            parameters={"depthTest": False},
                        )
                        overlay_layers.append(history_layer)
                else:
                    history_layer = pdk.Layer(
                        "LineLayer",
                        data=lane_overlay,
                        get_source_position="[origin_lon, origin_lat]",
                        get_target_position="[dest_lon, dest_lat]",
                        get_color="colour",
                        get_width="line_width",
                        pickable=True,
                        opacity=0.4,
                    )
                    overlay_layers.append(history_layer)

            if "job_id" in active_routes.columns and not historical_routes.empty:
                merge_columns = [
                    "id",
                    "colour",
                    "profit_band",
                    "profitability_status",
                    "tooltip",
                ]
                if "route_path" in historical_overlay.columns:
                    merge_columns.append("route_path")
                enriched = _pydeck_frame(
                    active_routes.merge(
                        historical_overlay[merge_columns],
                        left_on="job_id",
                        right_on="id",
                        how="left",
                        suffixes=("", "_hist"),
                    )
                )
                if "colour" not in enriched.columns and "colour_hist" in enriched.columns:
                    enriched["colour"] = enriched["colour_hist"]
                if (
                    "profit_band" not in enriched.columns
                    and "profit_band_hist" in enriched.columns
                ):
                    enriched["profit_band"] = enriched["profit_band_hist"]
                if (
                    "profitability_status" not in enriched.columns
                    and "profitability_status_hist" in enriched.columns
                ):
                    enriched["profitability_status"] = enriched["profitability_status_hist"]
                if "tooltip" not in enriched.columns and "tooltip_hist" in enriched.columns:
                    enriched["tooltip"] = enriched["tooltip_hist"]
                if "route_path" not in enriched.columns and "route_path_hist" in enriched.columns:
                    enriched["route_path"] = enriched["route_path_hist"]
            else:
                enriched = _pydeck_frame(active_routes.copy())
                enriched["colour"] = [PROFITABILITY_COLOURS["Unknown"]] * len(enriched)
                enriched["profit_band"] = "Unknown"
                enriched["profitability_status"] = "Unknown"
                enriched["tooltip"] = "Active route"

            enriched["colour"] = enriched["colour"].apply(
                lambda value: value if isinstance(value, (list, tuple)) else [255, 255, 255]
            )
            enriched["tooltip"] = enriched.apply(
                lambda row: "Truck {} ({})".format(
                    row["truck_id"],
                    row.get("profitability_status")
                    or row.get("profit_band", "Unknown"),
                ),
                axis=1,
            )

            if "route_geometry" in enriched.columns:
                enriched["route_path_geometry"] = _sanitise_geojson_paths(
                    enriched["route_geometry"],
                    label="active route",
                )
                if "route_path" in enriched.columns:
                    enriched["route_path"] = enriched["route_path_geometry"].combine_first(
                        enriched["route_path"]
                    )
                else:
                    enriched["route_path"] = enriched["route_path_geometry"]

            if "route_polygon" in enriched.columns:
                enriched = _filter_valid_geometry_rows(
                    enriched,
                    column="route_polygon",
                    minimum_points=3,
                )
            else:
                enriched["route_polygon"] = enriched.apply(
                    lambda row: compute_tapered_route_polygon(row),
                    axis=1,
                )
                enriched = _filter_valid_geometry_rows(
                    enriched,
                    column="route_polygon",
                    minimum_points=3,
                )

            if show_actual_routes:
                if "route_path" in enriched.columns:
                    active_path_data = _filter_valid_geometry_rows(
                        enriched,
                        column="route_path",
                        minimum_points=2,
                    )
                else:
                    active_path_data = pd.DataFrame()
                if not active_path_data.empty:
                    active_layer = pdk.Layer(
                        "PathLayer",
                        data=active_path_data,
                        get_path="route_path",
                        get_color="colour",
                        get_width=5,
                        width_min_pixels=2,
                        pickable=True,
                        opacity=0.9,
                        parameters={"depthTest": False},
                    )
                    overlay_layers.append(active_layer)
            else:
                active_layer = pdk.Layer(
                    "LineLayer",
                    data=enriched[
                        enriched.apply(
                            lambda row: _valid_xy_row(row, lon_key="origin_lon", lat_key="origin_lat")
                            and _valid_xy_row(row, lon_key="dest_lon", lat_key="dest_lat"),
                            axis=1,
                        )
                    ].copy(),
                    get_source_position="[origin_lon, origin_lat]",
                    get_target_position="[dest_lon, dest_lat]",
                    get_color="colour",
                    get_width=5,
                    pickable=True,
                    opacity=0.9,
                )
                overlay_layers.append(active_layer)

        if show_live_overlay and not truck_data.empty:
            trucks_layer = pdk.Layer(
                "ScatterplotLayer",
                data=truck_data,
                get_position="[lon, lat]",
                get_fill_color="colour",
                get_radius=800,
                pickable=True,
            )
            overlay_layers.append(trucks_layer)

            text_layer = pdk.Layer(
                "TextLayer",
                data=truck_data,
                get_position="[lon, lat]",
                get_text="truck_label",
                get_color="colour",
                get_size=12,
                size_units="meters",
                size_scale=16,
                get_alignment_baseline="bottom",
            )
            overlay_layers.append(text_layer)

        view_df_candidates: list[pd.DataFrame] = []
        if not historical_routes.empty:
            view_df_candidates.append(
                historical_routes.rename(columns={"origin_lat": "lat", "origin_lon": "lon"})[
                    ["lat", "lon"]
                ]
            )
        if not truck_data.empty:
            view_df_candidates.append(truck_data[["lat", "lon"]])
        if not active_routes.empty:
            view_df_candidates.append(
                active_routes.rename(columns={"origin_lat": "lat", "origin_lon": "lon"})[
                    ["lat", "lon"]
                ]
            )
        view_df = pd.concat(view_df_candidates) if view_df_candidates else pd.DataFrame()

        tooltip = None
        if show_live_overlay and overlay_layers:
            tooltip = {"html": "<b>{tooltip}</b>", "style": {"color": "white"}}

        layer_stack = [layer for layer in [base_map_layer, *overlay_layers] if layer is not None]
        deck_kwargs = {
            "layers": layer_stack,
            "initial_view_state": _initial_view_state(view_df),
            "tooltip": tooltip,
        }
        deck_kwargs.update(deck_map_kwargs)
        st.pydeck_chart(pdk.Deck(**deck_kwargs))

        if show_live_overlay and overlay_layers:
            legend_cols = st.columns(len(PROFITABILITY_COLOURS))
            for (band, colour), column in zip(PROFITABILITY_COLOURS.items(), legend_cols):
                colour_hex = "#" + "".join(f"{int(c):02x}" for c in colour)
                column.markdown(
                    f"<div style='color:{colour_hex}; font-weight:bold'>{band}</div>",
                    unsafe_allow_html=True,
                )
        elif show_live_overlay and not overlay_layers:
            st.info("No live overlays match the current filters yet.")
    else:
        if historical_routes.empty and trucks.empty and active_routes.empty:
            st.info("No geocoded historical jobs or live telemetry available to plot yet.")
            return

        radius_pixels = st.slider(
            "Heatmap radius",
            min_value=20,
            max_value=150,
            value=60,
            step=10,
            key=f"{toggle_key}_heatmap_radius",
            help="Adjust how far each point influence spreads across the heatmap.",
        )
        intensity = st.slider(
            "Heatmap intensity",
            min_value=0.2,
            max_value=4.0,
            value=1.0,
            step=0.2,
            key=f"{toggle_key}_heatmap_intensity",
            help="Increase to emphasise clusters of live activity.",
        )

        heatmap_source = _pydeck_frame(build_live_heatmap_source(historical_routes, active_routes, truck_data))

        if heatmap_source.empty:
            st.info(
                "Live heatmap requires geocoded historical routes, active routes, or truck telemetry."
            )
            return

        heatmap_layer = pdk.Layer(
            "HeatmapLayer",
            data=heatmap_source,
            get_position="[lon, lat]",
            aggregation="SUM",
            get_weight="weight",
            radiusPixels=radius_pixels,
            intensity=intensity,
        )

        layer_stack = [layer for layer in [base_map_layer, heatmap_layer] if layer is not None]
        deck_kwargs = {
            "layers": layer_stack,
            "initial_view_state": _initial_view_state(heatmap_source),
            "tooltip": None,
        }
        deck_kwargs.update(deck_map_kwargs)
        st.pydeck_chart(pdk.Deck(**deck_kwargs))

        st.caption(
            "Historical endpoints provide the base density, active routes carry more weight, "
            "and live trucks are boosted the most so current activity shines through."
        )
