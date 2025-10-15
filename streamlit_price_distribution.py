"""Streamlit dashboard for the price distribution analysis."""
from __future__ import annotations

import io
import json
import math
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

try:
    import folium
    from streamlit_folium import st_folium
except ModuleNotFoundError:  # pragma: no cover - optional dependency for pin UI
    folium = None  # type: ignore[assignment]
    st_folium = None  # type: ignore[assignment]

from analytics.db import connection_scope, ensure_dashboard_tables
from analytics.price_distribution import (
    DistributionSummary,
    ProfitabilitySummary,
    PROFITABILITY_COLOURS,
    ColumnMapping,
    compute_profitability_line_width,
    compute_tapered_route_polygon,
    available_heatmap_weightings,
    build_isochrone_polygons,
    filter_routes_by_country,
    build_heatmap_source,
    create_histogram,
    create_metro_profitability_figure,
    create_m3_margin_figure,
    create_m3_vs_km_figure,
    ensure_break_even_parameter,
    enrich_missing_route_coordinates,
    import_historical_jobs_from_dataframe,
    load_historical_jobs,
    load_quotes,
    load_live_jobs,
    prepare_metric_route_map_data,
    prepare_route_map_data,
    prepare_profitability_map_data,
    prepare_profitability_route_data,
    summarise_distribution,
    summarise_profitability,
    update_break_even,
)
from analytics.optimizer import (
    OptimizerParameters,
    OptimizerRun,
    can_run_optimizer,
    recommendations_to_frame,
    run_margin_optimizer,
)
from analytics.live_data import (
    TRUCK_STATUS_COLOURS,
    build_live_heatmap_source,
    extract_route_path,
    load_active_routes,
    load_truck_positions,
)
from corkysoft.pricing import DEFAULT_MODIFIERS
from corkysoft.quote_service import (
    COUNTRY_DEFAULT,
    QuoteInput,
    QuoteResult,
    build_summary,
    calculate_quote,
    format_currency,
)
from corkysoft.repo import (
    ClientDetails,
    ensure_schema as ensure_quote_schema,
    find_client_matches,
    format_client_display,
    persist_quote,
)
from corkysoft.routing import snap_coordinates_to_road
from corkysoft.schema import ensure_schema as ensure_core_schema


DEFAULT_TARGET_MARGIN_PERCENT = 20.0
_AUS_LAT_LON = (-25.2744, 133.7751)
_PIN_NOTE = "Manual pin override used for routing"
_HAVERSINE_MODAL_STATE_KEY = "quote_haversine_modal_ack"
_NULL_CLIENT_MODAL_STATE_KEY = "quote_null_client_modal_open"
_NULL_CLIENT_COMPANY_KEY = "quote_null_client_company"
_NULL_CLIENT_NOTES_KEY = "quote_null_client_notes"
_NULL_CLIENT_DEFAULT_COMPANY = "Null (filler) client"
_NULL_CLIENT_DEFAULT_NOTES = "Placeholder client captured via quote builder."
_ISOCHRONE_PALETTE = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]


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
_QUOTE_COUNTRY_STATE_KEY = "quote_builder_country"


def _geojson_to_path(value: Any) -> Optional[List[List[float]]]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        coords = extract_route_path(value)
    except Exception:
        return None
    return [[float(lon), float(lat)] for lat, lon in coords]


def _initial_pin_state(result: QuoteResult) -> Dict[str, Any]:
    return {
        "origin": {
            "lon": float(result.origin_lon),
            "lat": float(result.origin_lat),
        },
        "destination": {
            "lon": float(result.dest_lon),
            "lat": float(result.dest_lat),
        },
        "enabled": False,
    }


def _ensure_pin_state(result: QuoteResult) -> Dict[str, Any]:
    state: Dict[str, Any] = st.session_state.get("quote_pin_override", {})
    if not state or "origin" not in state or "destination" not in state:
        state = _initial_pin_state(result)
    else:
        state.setdefault("enabled", False)
        # When result coordinates change, refresh defaults so pins move with them
        origin_state = state.get("origin") or {}
        dest_state = state.get("destination") or {}
        if not origin_state:
            origin_state = {}
        if not dest_state:
            dest_state = {}
        origin_state.setdefault("lon", float(result.origin_lon))
        origin_state.setdefault("lat", float(result.origin_lat))
        dest_state.setdefault("lon", float(result.dest_lon))
        dest_state.setdefault("lat", float(result.dest_lat))
        state["origin"] = origin_state
        state["destination"] = dest_state
    st.session_state["quote_pin_override"] = state
    return state


def _pin_coordinates(entry: Dict[str, Any]) -> tuple[float, float]:
    lon = entry.get("lon")
    lat = entry.get("lat")
    if lon is None or lat is None:
        return (_AUS_LAT_LON[1], _AUS_LAT_LON[0])
    return (float(lon), float(lat))


def _pin_lon_key(map_key: str) -> str:
    return f"{map_key}_lon_input"


def _pin_lat_key(map_key: str) -> str:
    return f"{map_key}_lat_input"


def _render_pin_picker(
    label: str,
    *,
    map_key: str,
    entry: Dict[str, Any],
) -> tuple[float, float]:
    lon, lat = _pin_coordinates(entry)
    lon_key = _pin_lon_key(map_key)
    lat_key = _pin_lat_key(map_key)

    if lon_key not in st.session_state:
        st.session_state[lon_key] = float(lon)
    if lat_key not in st.session_state:
        st.session_state[lat_key] = float(lat)

    current_lon = float(st.session_state.get(lon_key, lon))
    current_lat = float(st.session_state.get(lat_key, lat))

    map_available = folium is not None and st_folium is not None
    if map_available:
        zoom = 12 if entry.get("lon") is not None and entry.get("lat") is not None else 4
        map_obj = folium.Map(location=[current_lat, current_lon], zoom_start=zoom)
        folium.Marker(
            [current_lat, current_lon],
            tooltip=f"{label} pin",
            icon=folium.Icon(color="blue" if label == "Origin" else "red"),
        ).add_to(map_obj)
        click_result = st_folium(map_obj, height=320, key=map_key, returned_objects=[])

        if isinstance(click_result, dict):
            last_clicked = click_result.get("last_clicked") or {}
            if "lat" in last_clicked and "lng" in last_clicked:
                current_lat = float(last_clicked["lat"])
                current_lon = float(last_clicked["lng"])
                st.session_state[lat_key] = current_lat
                st.session_state[lon_key] = current_lon
    else:
        st.warning(
            "Install 'folium' and 'streamlit-folium' for interactive pin dropping. The latitude/longitude inputs below remain available for manual edits."
        )

    lat_input = st.number_input(
        f"{label} latitude",
        value=float(st.session_state[lat_key]),
        format="%.6f",
        key=lat_key,
    )
    lon_input = st.number_input(
        f"{label} longitude",
        value=float(st.session_state[lon_key]),
        format="%.6f",
        key=lon_key,
    )

    current_lat = float(lat_input)
    current_lon = float(lon_input)

    entry["lon"] = current_lon
    entry["lat"] = current_lat
    st.session_state["quote_pin_override"] = st.session_state.get("quote_pin_override", {})
    return current_lon, current_lat

# -----------------------------------------------------------------------------
# Compatibility shim for metro-distance filtering across branches/modules
# -----------------------------------------------------------------------------
# Prefer the newer `filter_jobs_by_distance(df, metro_only=True/False, max_distance_km=...)`.
# If unavailable, fall back to `filter_metro_jobs(df, max_distance_km=...)`.
try:
    from inspect import signature

    from analytics.price_distribution import (  # type: ignore
        filter_jobs_by_distance as _filter_jobs_by_distance,
    )

    try:
        _FILTER_DISTANCE_PARAM = next(
            param
            for param in ("max_distance_km", "threshold_km")
            if param in signature(_filter_jobs_by_distance).parameters
        )
    except (StopIteration, ValueError, TypeError):
        _FILTER_DISTANCE_PARAM = None

    def _filter_by_distance(
        df: pd.DataFrame,
        *,
        metro_only: bool = False,
        max_distance_km: float = 100.0,
    ) -> pd.DataFrame:
        kwargs = {"metro_only": metro_only}
        if _FILTER_DISTANCE_PARAM is not None:
            kwargs[_FILTER_DISTANCE_PARAM] = max_distance_km
        return _filter_jobs_by_distance(df, **kwargs)

except Exception:
    try:
        from analytics.price_distribution import (  # type: ignore
            filter_metro_jobs as _filter_metro_jobs,
        )

        def _filter_by_distance(
            df: pd.DataFrame,
            *,
            metro_only: bool = False,
            max_distance_km: float = 100.0,
        ) -> pd.DataFrame:
            return _filter_metro_jobs(df, max_distance_km=max_distance_km) if metro_only else df

    except Exception:
        # Graceful no-op fallback if neither helper exists; show all rows.
        def _filter_by_distance(
            df: pd.DataFrame,
            *,
            metro_only: bool = False,
            max_distance_km: float = 100.0,
        ) -> pd.DataFrame:
            return df


st.set_page_config(
    page_title="Price distribution by corridor",
    layout="wide",
)

st.title("Price distribution (Airbnb-style)")
st.caption(
    "Visualise $ per m³ by corridor and client, with break-even bands to spot loss-leaders."
)

tabs_placeholder = st.container()


def _blank_column_mapping() -> ColumnMapping:
    return ColumnMapping(
        date=None,
        client=None,
        price=None,
        revenue=None,
        volume=None,
        origin=None,
        destination=None,
        corridor=None,
        distance=None,
        final_cost=None,
    )


def render_summary(
    summary: DistributionSummary,
    break_even: float,
    profitability_summary: ProfitabilitySummary,
    *,
    metro_summary: Optional[DistributionSummary] = None,
    metro_profitability: Optional[ProfitabilitySummary] = None,
    metro_distance_km: float = 100.0,
) -> None:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Jobs in filter", summary.job_count)
    valid_label = f"Valid $/m³ ({summary.priced_job_count})"
    col2.metric(
        valid_label,
        f"{summary.median:,.2f}" if summary.priced_job_count else "n/a",
    )
    col3.metric(
        "25th percentile",
        f"{summary.percentile_25:,.2f}" if summary.priced_job_count else "n/a",
    )
    col4.metric(
        "75th percentile",
        f"{summary.percentile_75:,.2f}" if summary.priced_job_count else "n/a",
    )
    below_pct = summary.below_break_even_ratio * 100 if summary.priced_job_count else 0.0
    col5.metric(
        "% below break-even",
        f"{below_pct:.1f}%",
        help=f"Break-even: ${break_even:,.2f} per m³",
    )

    def _format_value(
        value: Optional[float], *, currency: bool = False, percentage: bool = False
    ) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return "n/a"
        if currency:
            return f"${value:,.2f}"
        if percentage:
            return f"{value * 100:.1f}%"
        return f"{value:,.2f}"

    stats_cols = st.columns(4)
    stats = [
        ("Mean $/m³", summary.mean, True, False),
        ("Std dev $/m³", summary.std_dev, True, False),
        ("Kurtosis", summary.kurtosis, False, False),
        ("Skewness", summary.skewness, False, False),
    ]
    for column, (label, value, as_currency, as_percentage) in zip(stats_cols, stats):
        column.metric(
            label,
            _format_value(value, currency=as_currency, percentage=as_percentage),
        )

    profitability_cols = st.columns(4)
    profitability_metrics = [
        ("Median $/km", profitability_summary.revenue_per_km_median, True, False),
        ("Average $/km", profitability_summary.revenue_per_km_mean, True, False),
        (
            "Median margin $/m³",
            profitability_summary.margin_per_m3_median,
            True,
            False,
        ),
        (
            "Median margin %",
            profitability_summary.margin_per_m3_pct_median,
            False,
            True,
        ),
    ]
    for column, (label, value, as_currency, as_percentage) in zip(
        profitability_cols, profitability_metrics
    ):
        column.metric(
            label,
            _format_value(value, currency=as_currency, percentage=as_percentage),
        )

    if metro_summary and metro_profitability:
        st.markdown(
            f"**Metro subset (≤{metro_distance_km:,.0f} km)**"
        )
        share = 0.0
        if summary.job_count:
            share = metro_summary.job_count / summary.job_count
        st.caption(
            f"{metro_summary.job_count} jobs in metro scope "
            f"({share:.1%} of filtered jobs)."
        )

        metro_metrics = [
            ("Median $/km", "revenue_per_km_median", True, False),
            ("Average $/km", "revenue_per_km_mean", True, False),
            ("Median margin $/m³", "margin_per_m3_median", True, False),
            ("Median margin %", "margin_per_m3_pct_median", False, True),
        ]
        metro_cols = st.columns(len(metro_metrics))
        for column, (label, attr, as_currency, as_percentage) in zip(
            metro_cols, metro_metrics
        ):
            metro_value = getattr(metro_profitability, attr)
            overall_value = getattr(profitability_summary, attr)
            delta = None
            if (
                metro_value is not None
                and overall_value is not None
                and not any(
                    isinstance(val, float)
                    and (math.isnan(val) or math.isinf(val))
                    for val in (metro_value, overall_value)
                )
            ):
                diff = metro_value - overall_value
                if as_currency:
                    delta = f"{diff:+,.2f}"
                elif as_percentage:
                    delta = f"{diff * 100:+.1f}%"
                else:
                    delta = f"{diff:+.2f}"
            column.metric(
                label,
                _format_value(
                    metro_value, currency=as_currency, percentage=as_percentage
                ),
                delta=delta,
            )


def build_route_map(
    df: pd.DataFrame,
    colour_label: str,
    *,
    show_routes: bool,
    show_points: bool,
    colour_mode: str = "categorical",
    colour_scale: Optional[Sequence[str]] = None,
    colorbar_tickformat: Optional[str] = None,
) -> go.Figure:
    """Construct a Plotly Mapbox figure showing coloured routes and points."""

    def _row_route_points(row: pd.Series) -> List[Tuple[float, float]]:
        """Return the ordered ``(lat, lon)`` points for ``row`` when available."""

        geojson_value = row.get("route_geojson")
        if isinstance(geojson_value, str) and geojson_value.strip():
            try:
                return extract_route_path(geojson_value)
            except Exception:
                pass

        path_value = row.get("route_path")
        if isinstance(path_value, str):
            raw = path_value.strip()
            if not raw:
                return []
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
            else:
                path_value = parsed

        if isinstance(path_value, dict):
            try:
                return extract_route_path(json.dumps(path_value))
            except Exception:
                return []

        if isinstance(path_value, (list, tuple)):
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
                return coords

        return []

    palette = px.colors.qualitative.Bold or [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
    ]
    colour_values = list(dict.fromkeys(df["map_colour_value"].tolist()))
    if not palette:
        palette = ["#636EFA"]
    if len(colour_values) > len(palette):
        repeats = (len(colour_values) // len(palette)) + 1
        palette = (palette * repeats)[: len(colour_values)]
    colour_map = {
        value: palette[idx % len(palette)] for idx, value in enumerate(colour_values)
    }

    figure = go.Figure()
    plot_df = df.copy()

    if colour_mode == "categorical":
        palette = px.colors.qualitative.Bold or [
            "#636EFA",
            "#EF553B",
            "#00CC96",
            "#AB63FA",
        ]
        colour_values = list(dict.fromkeys(plot_df["map_colour_value"].tolist()))
        if not palette:
            palette = ["#636EFA"]
        if len(colour_values) > len(palette):
            repeats = (len(colour_values) // len(palette)) + 1
            palette = (palette * repeats)[: len(colour_values)]
        colour_map = {
            value: palette[idx % len(palette)] for idx, value in enumerate(colour_values)
        }

        if show_routes:
            for value in colour_values:
                category_df = plot_df[plot_df["map_colour_value"] == value]
                if category_df.empty:
                    continue
                display_value = (
                    category_df.get("map_colour_display", pd.Series([value])).iloc[0]
    if show_routes:
        for value in colour_values:
            category_df = df[df["map_colour_value"] == value]
            if category_df.empty:
                continue
            lat_values: list[float] = []
            lon_values: list[float] = []
            for _, row in category_df.iterrows():
                route_points = _row_route_points(row)
                if route_points:
                    for lat, lon in route_points:
                        lat_values.append(lat)
                        lon_values.append(lon)
                    lat_values.append(None)
                    lon_values.append(None)
                    continue

                try:
                    origin_lat = float(row["origin_lat"])
                    dest_lat = float(row["dest_lat"])
                    origin_lon = float(row["origin_lon"])
                    dest_lon = float(row["dest_lon"])
                except (TypeError, ValueError):
                    continue

                lat_values.extend([origin_lat, dest_lat, None])
                lon_values.extend([origin_lon, dest_lon, None])

            figure.add_trace(
                go.Scattermap(
                    lat=lat_values,
                    lon=lon_values,
                    mode="lines",
                    line={"width": 2, "color": colour_map[value]},
                    name=value,
                    legendgroup=value,
                    showlegend=False,
                    hoverinfo="skip",
                )
                lat_values: list[float] = []
                lon_values: list[float] = []
                for _, row in category_df.iterrows():
                    lat_values.extend([row["origin_lat"], row["dest_lat"], None])
                    lon_values.extend([row["origin_lon"], row["dest_lon"], None])
                figure.add_trace(
                    go.Scattermap(
                        lat=lat_values,
                        lon=lon_values,
                        mode="lines",
                        line={"width": 2, "color": colour_map[value]},
                        name=str(display_value),
                        legendgroup=str(value),
                        showlegend=False,
                        hoverinfo="skip",
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
                    job_id = row.get("id", "n/a")
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
                    marker_lat.append(row["origin_lat"])
                    marker_lon.append(row["origin_lon"])
                    marker_text.append(
                        f"{colour_label}: {display_value}<br>Origin: {origin_label}<br>Job ID: {job_id}"
                    )
                    marker_lat.append(row["dest_lat"])
                    marker_lon.append(row["dest_lon"])
                    marker_text.append(
                        f"{colour_label}: {display_value}<br>Destination: {destination_label}<br>Job ID: {job_id}"
                    )

                figure.add_trace(
                    go.Scattermap(
                        lat=marker_lat,
                        lon=marker_lon,
                        mode="markers",
                        marker={
                            "size": 9,
                            "color": colour_map[value],
                            "opacity": 0.85,
                        },
                        text=marker_text,
                        hovertemplate="%{text}<extra></extra>",
                        name=str(display_value),
                        legendgroup=str(value),
                    )
                )
                try:
                    origin_lat = float(row["origin_lat"])
                    origin_lon = float(row["origin_lon"])
                except (TypeError, ValueError):
                    pass
                else:
                    marker_lat.append(origin_lat)
                    marker_lon.append(origin_lon)
                    marker_text.append(
                        f"{colour_label}: {value}<br>Origin: {origin_label}<br>Job ID: {job_id}"
                    )

                try:
                    dest_lat = float(row["dest_lat"])
                    dest_lon = float(row["dest_lon"])
                except (TypeError, ValueError):
                    pass
                else:
                    marker_lat.append(dest_lat)
                    marker_lon.append(dest_lon)
                    marker_text.append(
                        f"{colour_label}: {value}<br>Destination: {destination_label}<br>Job ID: {job_id}"
                    )

            if not marker_lat or not marker_lon:
                continue

    elif colour_mode == "continuous":
        numeric_series = pd.to_numeric(plot_df["map_colour_value"], errors="coerce")
        numeric_series = numeric_series.replace([math.inf, -math.inf], pd.NA)
        valid_mask = numeric_series.notna()
        plot_df = plot_df.loc[valid_mask].copy()
        if not plot_df.empty:
            numeric_values = numeric_series.loc[valid_mask].astype(float)
            plot_df["map_colour_value"] = numeric_values
            colour_scale = colour_scale or px.colors.sequential.Viridis
            min_value = float(numeric_values.min())
            max_value = float(numeric_values.max())
            span = max_value - min_value

            def _to_colour(value: float) -> str:
                if not span or math.isclose(span, 0.0):
                    position = 0.5
                else:
                    position = (value - min_value) / span if span else 0.5
                position = min(max(position, 0.0), 1.0)
                return px.colors.sample_colorscale(colour_scale, [position])[0]

            colorbar_dict = {"title": colour_label}
            if colorbar_tickformat:
                colorbar_dict["tickformat"] = colorbar_tickformat

            if show_routes:
                for _, row in plot_df.iterrows():
                    value = float(row["map_colour_value"])
                    colour = _to_colour(value)
                    lat_values = [row["origin_lat"], row["dest_lat"], None]
                    lon_values = [row["origin_lon"], row["dest_lon"], None]
                    figure.add_trace(
                        go.Scattermap(
                            lat=lat_values,
                            lon=lon_values,
                            mode="lines",
                            line={"width": 3, "color": colour},
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

            if show_points:
                marker_lat: list[float] = []
                marker_lon: list[float] = []
                marker_text: list[str] = []
                marker_values: list[float] = []
                for _, row in plot_df.iterrows():
                    value = float(row["map_colour_value"])
                    display_value = row.get(
                        "map_colour_display",
                        f"{value:,.2f}",
                    )
                    job_id = row.get("id", "n/a")
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
                    marker_lat.extend([row["origin_lat"], row["dest_lat"]])
                    marker_lon.extend([row["origin_lon"], row["dest_lon"]])
                    marker_text.append(
                        f"{colour_label}: {display_value}<br>Origin: {origin_label}<br>Job ID: {job_id}"
                    )
                    marker_text.append(
                        f"{colour_label}: {display_value}<br>Destination: {destination_label}<br>Job ID: {job_id}"
                    )
                    marker_values.extend([value, value])

                figure.add_trace(
                    go.Scattermap(
                        lat=marker_lat,
                        lon=marker_lon,
                        mode="markers",
                        marker={
                            "size": 9,
                            "color": marker_values,
                            "colorscale": colour_scale,
                            "cmin": min_value,
                            "cmax": max_value,
                            "opacity": 0.85,
                            "colorbar": colorbar_dict,
                        },
                        text=marker_text,
                        hovertemplate="%{text}<extra></extra>",
                        showlegend=False,
                    )
                )
            else:
                figure.add_trace(
                    go.Scattermap(
                        lat=plot_df["origin_lat"].tolist(),
                        lon=plot_df["origin_lon"].tolist(),
                        mode="markers",
                        marker={
                            "size": 0.0001,
                            "color": numeric_values.tolist(),
                            "colorscale": colour_scale,
                            "cmin": min_value,
                            "cmax": max_value,
                            "colorbar": colorbar_dict,
                            "opacity": 0.0,
                        },
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

    all_lat = pd.concat([plot_df.get("origin_lat", pd.Series(dtype=float)), plot_df.get("dest_lat", pd.Series(dtype=float))])
    all_lon = pd.concat([plot_df.get("origin_lon", pd.Series(dtype=float)), plot_df.get("dest_lon", pd.Series(dtype=float))])
    center_lat = float(all_lat.mean()) if not all_lat.empty else 0.0
    center_lon = float(all_lon.mean()) if not all_lon.empty else 0.0

    figure.update_layout(
        mapbox={
            "style": "carto-positron",
            "center": {"lat": center_lat, "lon": center_lon},
            "zoom": 3,
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 0.01},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )
    return figure


def _initial_view_state(df: pd.DataFrame) -> pdk.ViewState:
    if df.empty:
        return pdk.ViewState(latitude=-25.2744, longitude=133.7751, zoom=4.0)
    lat_column = "lat" if "lat" in df.columns else "origin_lat"
    lon_column = "lon" if "lon" in df.columns else "origin_lon"
    lat = pd.to_numeric(df[lat_column], errors="coerce").dropna()
    lon = pd.to_numeric(df[lon_column], errors="coerce").dropna()
    if lat.empty or lon.empty:
        return pdk.ViewState(latitude=-25.2744, longitude=133.7751, zoom=4.0)
    return pdk.ViewState(latitude=float(lat.mean()), longitude=float(lon.mean()), zoom=4.5)


def render_network_map(
    historical_routes: pd.DataFrame,
    trucks: pd.DataFrame,
    active_routes: pd.DataFrame,
    *,
    toggle_key: str = "network_map_live_overlay_toggle",
) -> None:
    st.markdown("### Live network overview")

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

    base_map_layer = pdk.Layer(
        "TileLayer",
        data="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        min_zoom=0,
        max_zoom=19,
        tile_size=256,
        attribution="© OpenStreetMap contributors",
    )

    truck_data = trucks.copy()
    if not truck_data.empty:
        truck_data["colour"] = truck_data["status"].map(TRUCK_STATUS_COLOURS)
        truck_data["colour"] = truck_data["colour"].apply(
            lambda value: value if isinstance(value, (list, tuple)) else [0, 122, 204]
        )
        truck_data["tooltip"] = truck_data.apply(
            lambda row: f"{row['truck_id']} ({row['status']})", axis=1
        )

    historical_overlay = historical_routes.copy()
    if not historical_overlay.empty and "route_geojson" in historical_overlay.columns:
        historical_overlay["route_path"] = historical_overlay["route_geojson"].apply(
            _geojson_to_path
        )
    else:
        historical_overlay["route_path"] = None

    if not historical_overlay.empty and "lane_key" in historical_overlay.columns:
        lane_overlay = historical_overlay.copy()
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

    if view_mode == "Overlay":
        show_actual_routes = False
        if show_live_overlay and (historical_has_paths or active_has_geometry):
            show_actual_routes = st.checkbox(
                "Show actual route traces when available",
                value=historical_has_paths or active_has_geometry,
                help="Draw recorded telemetry paths instead of straight corridor lines when data is available.",
                key=f"{toggle_key}_show_actual_routes",
            )

        overlay_layers: list[pdk.Layer] = []

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
                enriched = active_routes.merge(
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
                enriched = active_routes.copy()
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

            enriched["colour"] = enriched["colour"].apply(
                lambda value: value if isinstance(value, (list, tuple)) else [255, 255, 255]
            )

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
                    history_paths = lane_overlay.dropna(subset=["route_path"])
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
                enriched = active_routes.merge(
                    historical_overlay[merge_columns],
                    left_on="job_id",
                    right_on="id",
                    how="left",
                    suffixes=("", "_hist"),
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
                enriched = active_routes.copy()
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
                enriched["route_path_geometry"] = enriched["route_geometry"].apply(_geojson_to_path)
                if "route_path" in enriched.columns:
                    enriched["route_path"] = enriched["route_path_geometry"].combine_first(
                        enriched["route_path"]
                    )
                else:
                    enriched["route_path"] = enriched["route_path_geometry"]

            if show_actual_routes:
                if "route_path" in enriched.columns:
                    active_path_data = enriched.dropna(subset=["route_path"])
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
                    )
                    overlay_layers.append(active_layer)
            else:
                active_layer = pdk.Layer(
                    "LineLayer",
                    data=enriched,
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
                get_text="truck_id",
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

        st.pydeck_chart(
            pdk.Deck(
                layers=[base_map_layer, *overlay_layers],
                initial_view_state=_initial_view_state(view_df),
                tooltip=tooltip,
                map_style=None,
            )
        )

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

        heatmap_source = build_live_heatmap_source(historical_routes, active_routes, truck_data)

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

        st.pydeck_chart(
            pdk.Deck(
                layers=[base_map_layer, heatmap_layer],
                initial_view_state=_initial_view_state(heatmap_source),
                tooltip=None,
                map_style=None,
            )
        )

        st.caption(
            "Historical endpoints provide the base density, active routes carry more weight, "
            "and live trucks are boosted the most so current activity shines through."
        )


def _set_query_params(**params: str) -> None:
    """Set Streamlit query parameters using the stable API when available."""
    query_params = getattr(st, "query_params", None)
    if query_params is not None:
        query_params.from_dict(params)
        return
    # Fallback for older Streamlit versions.
    st.experimental_set_query_params(**params)


def _get_query_params() -> Dict[str, List[str]]:
    """Return query parameters as a dictionary of lists."""
    query_params = getattr(st, "query_params", None)
    if query_params is not None:
        return {key: query_params.get_all(key) for key in query_params.keys()}
    return st.experimental_get_query_params()


def _rerun_app() -> None:
    """Trigger a Streamlit rerun using the available API."""
    rerun = getattr(st, "rerun", None)
    if rerun is not None:
        rerun()
        return
    st.experimental_rerun()


def _first_non_empty(route: pd.Series, columns: Sequence[str]) -> Optional[str]:
    for column in columns:
        if column in route and isinstance(route[column], str):
            value = route[column].strip()
            if value:
                return value
    return None


def _format_route_label(route: pd.Series) -> str:
    origin = _first_non_empty(
        route,
        [
            "corridor_display",
            "origin",
            "origin_city",
            "origin_normalized",
            "origin_raw",
        ],
    ) or "Origin"
    destination = _first_non_empty(
        route,
        [
            "destination",
            "destination_city",
            "destination_normalized",
            "destination_raw",
        ],
    ) or "Destination"
    distance_value: Optional[float] = None
    for column in ("distance_km", "distance", "km", "kms"):
        if column in route and pd.notna(route[column]):
            try:
                distance_value = float(route[column])
            except (TypeError, ValueError):
                continue
            break
    if distance_value is not None and not math.isnan(distance_value):
        return f"{origin} → {destination} ({distance_value:.1f} km)"
    return f"{origin} → {destination}"


def _extract_route_date(route: pd.Series) -> Optional[date]:
    for column in (
        "job_date",
        "move_date",
        "delivery_date",
        "created_at",
        "updated_at",
    ):
        if column in route and pd.notna(route[column]):
            try:
                return pd.to_datetime(route[column]).date()
            except Exception:
                continue
    return None


def _extract_route_volume(route: pd.Series, candidates: Sequence[str]) -> Optional[float]:
    for column in candidates:
        if not column:
            continue
        if column in route and pd.notna(route[column]):
            try:
                return float(route[column])
            except (TypeError, ValueError):
                continue
    return None


with connection_scope() as conn:
    break_even_value = ensure_break_even_parameter(conn)
    ensure_quote_schema(conn)

    df_all: pd.DataFrame = pd.DataFrame()
    mapping: ColumnMapping = _blank_column_mapping()
    dataset_loader = load_historical_jobs
    dataset_key = "historical"
    dataset_label = "Historical quotes"
    dataset_error: Optional[str] = None
    empty_dataset_message: Optional[str] = None

    start_date: Optional[date] = None
    end_date: Optional[date] = None
    selected_corridor: Optional[str] = None
    selected_clients: List[str] = []
    postcode_prefix: Optional[str] = None

    with st.sidebar:
        st.header("Filters")
        if st.button(
            "Initialise database tables",
            help=(
                "Create empty historical and live job tables so the dashboard can run "
                "before data imports."
            ),
        ):
            ensure_core_schema(conn)
            ensure_dashboard_tables(conn)
            ensure_quote_schema(conn)
            st.success(
                "Database tables initialised. Import data or start building quotes below."
            )

        dataset_options = {
            "Historical quotes": ("historical", load_historical_jobs),
            "Saved quick quotes": ("quotes", load_quotes),
            "Live jobs": ("live", load_live_jobs),
        }
        dataset_label = st.radio(
            "Dataset",
            options=list(dataset_options.keys()),
            format_func=lambda label: label,
        )
        dataset_key, dataset_loader = dataset_options[dataset_label]

        import_feedback: Optional[tuple[str, str]] = None
        if dataset_key == "historical":
            with st.expander("Import historical jobs from CSV", expanded=False):
                import_form = st.form(key="historical_import_form")
                uploaded_file = import_form.file_uploader(
                    "Select CSV file", type=["csv"], help="Requires headers such as date, origin, destination and m3."
                )
                submit_import = import_form.form_submit_button("Import jobs")
                if submit_import:
                    if uploaded_file is None:
                        import_feedback = (
                            "warning",
                            "Choose a CSV file before importing.",
                        )
                    else:
                        try:
                            imported_df = pd.read_csv(uploaded_file)
                        except Exception as exc:
                            import_feedback = (
                                "error",
                                f"Failed to read CSV: {exc}",
                            )
                        else:
                            try:
                                inserted, skipped_rows = import_historical_jobs_from_dataframe(
                                    conn, imported_df
                                )
                            except ValueError as exc:
                                import_feedback = ("error", str(exc))
                            except Exception as exc:
                                import_feedback = (
                                    "error",
                                    f"Failed to import historical jobs: {exc}",
                                )
                            else:
                                if inserted:
                                    message = (
                                        f"Imported {inserted} historical job"
                                        f"{'s' if inserted != 1 else ''}."
                                    )
                                    if skipped_rows:
                                        message += (
                                            f" Skipped {skipped_rows} row"
                                            f"{'s' if skipped_rows != 1 else ''} with missing or duplicate data."
                                        )
                                    import_feedback = ("success", message)
                                else:
                                    if skipped_rows:
                                        message = (
                                            "No new rows imported. Skipped "
                                            f"{skipped_rows} row{'s' if skipped_rows != 1 else ''} due to validation or duplicates."
                                        )
                                    else:
                                        message = "No rows imported from the provided file."
                                    import_feedback = ("warning", message)

        try:
            df_all, mapping = dataset_loader(conn)
        except RuntimeError as exc:
            dataset_error = str(exc)
        except Exception as exc:
            dataset_error = f"Failed to load {dataset_label.lower()} data: {exc}"

        if import_feedback:
            level, message = import_feedback
            if level == "success":
                st.success(message)
            elif level == "warning":
                st.info(message)
            else:
                st.error(message)

        data_available = dataset_error is None and not df_all.empty

        today_value = date.today()
        date_column = "job_date" if "job_date" in df_all.columns else mapping.date
        if data_available and date_column and date_column in df_all.columns:
            df_all[date_column] = pd.to_datetime(df_all[date_column], errors="coerce")
            min_date = df_all[date_column].min()
            max_date = df_all[date_column].max()
            default_start = (
                min_date.date() if isinstance(min_date, pd.Timestamp) else today_value
            )
            default_end = (
                max_date.date() if isinstance(max_date, pd.Timestamp) else today_value
            )
            date_range = st.date_input(
                "Date range",
                value=(default_start, default_end),
                min_value=default_start,
                max_value=default_end,
            )
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date = default_start
                end_date = default_end
        else:
            st.date_input(
                "Date range",
                value=(today_value, today_value),
                disabled=True,
            )
            start_date = None
            end_date = None

        corridor_options: List[str] = []
        if data_available:
            corridor_series = df_all.get("corridor_display")
            if corridor_series is not None:
                corridor_options = sorted(
                    pd.Series(corridor_series).dropna().astype(str).unique().tolist()
                )
        corridor_selection = st.selectbox(
            "Corridor",
            options=["All corridors"] + corridor_options,
            index=0,
            disabled=not data_available,
        )
        selected_corridor = None if corridor_selection == "All corridors" else corridor_selection

        client_options: List[str] = []
        if data_available:
            client_series = df_all.get("client_display")
            if client_series is not None:
                client_options = sorted(
                    pd.Series(client_series).dropna().astype(str).unique().tolist()
                )
        selected_clients = st.multiselect(
            "Client",
            options=client_options,
            default=client_options if client_options else [],
            disabled=not data_available,
        )

        postcode_prefix = st.text_input(
            "Corridor contains postcode prefix",
            value=postcode_prefix or "",
            disabled=not data_available,
            help="Match origin or destination postcode prefixes (e.g. 40 to match 4000-4099).",
        ) or None

        if dataset_error:
            st.error(dataset_error)
        elif not data_available:
            empty_messages = {
                "historical": (
                    "historical_jobs table has no rows yet. Import historical jobs to populate the view."
                ),
                "quotes": (
                    "quotes table has no rows yet. Save a quick quote to populate the view."
                ),
                "live": "jobs table has no rows yet. Add live jobs to populate the view.",
            }
            empty_dataset_message = empty_messages.get(
                dataset_key, "No rows available for the selected dataset."
            )
            st.info(empty_dataset_message)

        st.subheader("Break-even model")
        new_break_even = st.number_input(
            "Break-even $/m³",
            min_value=0.0,
            value=float(break_even_value),
            step=5.0,
            help="Used to draw break-even bands on the histogram.",
        )
        if st.button("Update break-even"):
            update_break_even(conn, new_break_even)
            st.success(f"Break-even updated to ${new_break_even:,.2f}")
            break_even_value = new_break_even

    data_available = dataset_error is None and not df_all.empty

    filtered_df = pd.DataFrame()
    filtered_mapping = mapping
    has_filtered_data = False
    if data_available:
        try:
            filtered_df, filtered_mapping = dataset_loader(
                conn,
                start_date=start_date,
                end_date=end_date,
                clients=selected_clients or None,
                corridor=selected_corridor,
                postcode_prefix=postcode_prefix,
            )
            has_filtered_data = not filtered_df.empty
        except RuntimeError as exc:
            dataset_error = str(exc)
        except Exception as exc:
            dataset_error = f"Failed to apply filters: {exc}"

    if dataset_error:
        st.error(dataset_error)
    elif not data_available:
        st.info(
            empty_dataset_message
            or "No rows available for the selected dataset. Use the initialise button to create empty tables."
        )
    elif not has_filtered_data:
        st.warning("No jobs match the selected filters. Quote builder remains available below.")

    tab_labels = [
        "Histogram",
        "Profitability insights",
        "Route maps",
        "Quote builder",
        "Optimizer",
    ]
    params = _get_query_params()
    requested_tab = params.get("view", [tab_labels[0]])[0]
    if requested_tab not in tab_labels:
        requested_tab = tab_labels[0]
    if requested_tab != tab_labels[0]:
        ordered_labels = [
            requested_tab,
            *[label for label in tab_labels if label != requested_tab],
        ]
    else:
        ordered_labels = tab_labels

    with tabs_placeholder:
        streamlit_tabs = st.tabs(ordered_labels)
    tab_map: Dict[str, Any] = {
        label: tab for label, tab in zip(ordered_labels, streamlit_tabs)
    }

    summary: Optional[DistributionSummary] = None
    profitability_summary: Optional[ProfitabilitySummary] = None
    metro_summary: Optional[DistributionSummary] = None
    metro_profitability: Optional[ProfitabilitySummary] = None
    metro_distance_km = 100.0
    if has_filtered_data:
        summary = summarise_distribution(filtered_df, break_even_value)
        profitability_summary = summarise_profitability(filtered_df)

        metro_df = _filter_by_distance(
            filtered_df, metro_only=True, max_distance_km=metro_distance_km
        )
        if not metro_df.empty:
            metro_summary = summarise_distribution(metro_df, break_even_value)
            metro_profitability = summarise_profitability(metro_df)

        render_summary(
            summary,
            break_even_value,
            profitability_summary,
            metro_summary=metro_summary,
            metro_profitability=metro_profitability,
            metro_distance_km=metro_distance_km,
        )

    truck_positions = load_truck_positions(conn)
    active_routes = load_active_routes(conn)
    map_routes = prepare_profitability_route_data(filtered_df, break_even_value)

    render_network_map(
        map_routes,
        truck_positions,
        active_routes,
        toggle_key="network_map_live_overlay_toggle_overview",
    )

    with tab_map["Histogram"]:
        if has_filtered_data:
            histogram = create_histogram(filtered_df, break_even_value)
            st.plotly_chart(histogram, use_container_width=True)
            st.caption(
                "Histogram overlays include the normal distribution fit plus kurtosis and dispersion markers for context."
            )
        elif dataset_error:
            st.error("Unable to load jobs — initialise the database and retry.")
        else:
            st.info("Import historical jobs to plot the price distribution histogram.")

    with tab_map["Profitability insights"]:
        if has_filtered_data:
            st.markdown("### Profitability insights")
            view_options = {
                "m³ vs km profitability": create_m3_vs_km_figure,
                "Quoted vs calculated $/m³": create_m3_margin_figure,
                "Metro profitability spotlight": lambda data: create_metro_profitability_figure(
                    data, max_distance_km=metro_distance_km
                ),
            }
            selected_view = st.radio(
                "Choose a view",
                list(view_options.keys()),
                horizontal=True,
                help="Switch between per-kilometre earnings and quoted-versus-cost comparisons.",
                key="profitability_view",
            )
            fig = view_options[selected_view](filtered_df)
            st.plotly_chart(fig, use_container_width=True)

            if selected_view == "Metro profitability spotlight":
                st.caption(
                    "Metro view highlights close-in routes with margin and cost sensitivity overlays."
                )

            if "margin_per_m3" in filtered_df.columns:
                st.markdown("#### Margin outliers")
                ranked = (
                    filtered_df.dropna(subset=["margin_per_m3"]).sort_values("margin_per_m3")
                )
                if not ranked.empty:
                    low_cols, high_cols = st.columns(2)
                    display_fields = [
                        col
                        for col in [
                            "job_date",
                            "client_display",
                            "corridor_display",
                            "price_per_m3",
                            "final_cost_per_m3",
                            "margin_per_m3",
                            "margin_per_m3_pct",
                        ]
                        if col in ranked.columns
                    ]
                    low_cols.write("Lowest margin jobs")
                    low_cols.dataframe(ranked.head(5)[display_fields])
                    high_cols.write("Highest margin jobs")
                    high_cols.dataframe(ranked.tail(5).iloc[::-1][display_fields])
                else:
                    st.info("No margin data available to highlight outliers yet.")
        elif dataset_error:
            st.error("Unable to calculate profitability without job data.")
        else:
            st.info("Import jobs with price and cost data to unlock profitability insights.")

    truck_positions = load_truck_positions(conn)
    active_routes = load_active_routes(conn)

    with tab_map["Route maps"]:
        st.markdown("### Corridor visualisation")
        map_mode = st.radio(
            "Visualisation mode",
            ("Routes/points", "Heatmap", "Isochrones"),
            horizontal=True,
            help=(
                "Switch between individual routes/points, an aggregate density heatmap, "
                "or travel-time isochrones around each corridor."
            ),
        )
        metro_only = st.checkbox(
            "Limit to metro jobs (≤100 km)",
            value=False,
            help="Apply a distance filter using distance_km ≤ 100 to focus on metro corridors.",
        )

        scoped_df = _filter_by_distance(
            filtered_df, metro_only=metro_only, max_distance_km=metro_distance_km
        )

        if map_mode == "Routes/points":
            required_columns = {"origin_lat", "origin_lon", "dest_lat", "dest_lon"}
            missing_coordinates = required_columns - set(scoped_df.columns)

            if scoped_df.empty:
                st.info("No jobs match the metro filter for the current selection.")
            elif missing_coordinates:
                st.info(
                    "Add geocoded origin and destination coordinates to visualise routes."
                )
            else:
                geocoded = scoped_df.dropna(subset=list(required_columns))
                if geocoded.empty:
                    st.info(
                        "No routes with coordinates are available for the current filters."
                    )
                else:
                    colour_mode_label = st.radio(
                        "Colour data by",
                        ("Categorical attribute", "Metric"),
                        horizontal=True,
                        help=(
                            "Switch between discrete attributes and continuous metrics "
                            "to colour the route and point layers."
                        ),
                    )
                    show_routes = st.checkbox("Show route lines", value=True)
                    show_points = st.checkbox("Show origin/destination points", value=True)

                    if not show_routes and not show_points:
                        st.info("Enable at least one layer to view the route map.")
                    elif colour_mode_label == "Categorical attribute":
                        colour_dimensions = {
                            "Job ID": "id",
                            "Client": "client_display",
                            "Destination city": "destination_city",
                            "Origin city": "origin_city",
                        }
                        available_colour_dimensions = {
                            label: column
                            for label, column in colour_dimensions.items()
                            if column in geocoded.columns
                        }

                        if not available_colour_dimensions:
                            st.info(
                                "No categorical columns available to colour the route map."
                            )
                        else:
                            colour_label = st.selectbox(
                                "Categorical attribute",
                                options=list(available_colour_dimensions.keys()),
                                help=(
                                    "Choose which attribute drives the route and point colouring."
                                ),
                            )
                            selected_column = available_colour_dimensions[colour_label]
                            try:
                                plotly_map_df = prepare_route_map_data(
                                    scoped_df, selected_column
                                )
                            except KeyError as exc:
                                st.warning(str(exc))
                                plotly_map_df = pd.DataFrame()

                            if plotly_map_df.empty:
                                st.info(
                                    "No routes with coordinates are available for the current filters."
                                )
                            else:
                                route_map = build_route_map(
                                    plotly_map_df,
                                    colour_label,
                                    show_routes=show_routes,
                                    show_points=show_points,
                                )
                                st.plotly_chart(route_map, use_container_width=True)
                    else:
                        metric_colour_options = {
                            "Margin $/m³": {
                                "column": "margin_per_m3",
                                "format": "currency_per_m3",
                                "scale": px.colors.diverging.RdYlGn,
                                "tickformat": "$.2f",
                            },
                            "Margin %": {
                                "column": "margin_total_pct",
                                "format": "percentage",
                                "scale": px.colors.diverging.BrBG,
                                "tickformat": ".1%",
                            },
                            "Total margin": {
                                "column": "margin_total",
                                "format": "currency",
                                "scale": px.colors.diverging.RdYlGn,
                                "tickformat": "$,.0f",
                            },
                            "Total revenue": {
                                "column": "revenue_total",
                                "format": "currency",
                                "scale": px.colors.sequential.PuBu,
                                "tickformat": "$,.0f",
                            },
                            "Quoted price $/m³": {
                                "column": "price_per_m3",
                                "format": "currency_per_m3",
                                "scale": px.colors.sequential.Plasma,
                                "tickformat": "$.2f",
                            },
                            "Volume (m³)": {
                                "column": "volume_m3",
                                "format": "volume",
                                "scale": px.colors.sequential.Blues,
                                "tickformat": ".1f",
                            },
                            "Distance (km)": {
                                "column": "distance_km",
                                "format": "distance",
                                "scale": px.colors.sequential.Oranges,
                                "tickformat": ".0f",
                            },
                            "Duration (hr)": {
                                "column": "duration_hr",
                                "format": "hours",
                                "scale": px.colors.sequential.Sunset,
                                "tickformat": ".1f",
                            },
                        }

                        available_metric_options: dict[str, dict[str, object]] = {}
                        for label, spec in metric_colour_options.items():
                            column = spec["column"]
                            if column not in geocoded.columns:
                                continue
                            numeric_series = pd.to_numeric(
                                geocoded[column], errors="coerce"
                            )
                            numeric_series = numeric_series.replace(
                                [math.inf, -math.inf], pd.NA
                            )
                            if numeric_series.notna().any():
                                available_metric_options[label] = spec

                        if not available_metric_options:
                            st.info(
                                "No numeric metrics are available to colour the route map."
                            )
                        else:
                            metric_label = st.selectbox(
                                "Metric",
                                options=list(available_metric_options.keys()),
                                help=(
                                    "Select a metric to drive the continuous colour scale."
                                ),
                            )
                            metric_spec = available_metric_options[metric_label]
                            metric_column = metric_spec["column"]
                            format_spec = metric_spec.get("format", "number")
                            try:
                                metric_map_df = prepare_metric_route_map_data(
                                    scoped_df,
                                    metric_column,
                                    format_spec=str(format_spec),
                                )
                            except KeyError as exc:
                                st.warning(str(exc))
                                metric_map_df = pd.DataFrame()

                            if metric_map_df.empty:
                                st.info(
                                    "No routes with the selected metric are available for the current filters."
                                )
                            else:
                                route_map = build_route_map(
                                    metric_map_df,
                                    metric_label,
                                    show_routes=show_routes,
                                    show_points=show_points,
                                    colour_mode="continuous",
                                    colour_scale=metric_spec.get("scale"),
                                    colorbar_tickformat=metric_spec.get("tickformat"),
                                )
                                st.plotly_chart(route_map, use_container_width=True)
        elif map_mode == "Heatmap":
            weight_options = available_heatmap_weightings(filtered_df)
            weight_label = st.selectbox(
                "Heatmap weighting",
                options=list(weight_options.keys()),
                help="Choose which metric influences the heatmap intensity.",
            )
            weight_column = weight_options[weight_label]

            if scoped_df.empty:
                st.info("No jobs match the metro filter for the current selection.")
            else:
                try:
                    heatmap_source = build_heatmap_source(
                        scoped_df,
                        weight_column=weight_column,
                    )
                except KeyError as exc:
                    st.warning(str(exc))
                    heatmap_source = pd.DataFrame(columns=["lat", "lon", "weight"])

                if heatmap_source.empty:
                    st.info("No geocoded points are available for the current filters.")
                else:
                    centre = {
                        "lat": float(heatmap_source["lat"].mean()),
                        "lon": float(heatmap_source["lon"].mean()),
                    }
                    colour_scales = {
                        None: px.colors.sequential.YlOrRd,
                        "volume_m3": px.colors.sequential.Blues,
                        "margin_total": px.colors.diverging.RdYlGn,
                        "margin_per_m3": px.colors.sequential.Magma,
                        "margin_total_pct": px.colors.diverging.BrBG,
                        "margin_per_m3_pct": px.colors.diverging.BrBG,
                    }
                    midpoint_columns = {
                        "margin_total",
                        "margin_per_m3",
                        "margin_total_pct",
                        "margin_per_m3_pct",
                    }
                    midpoint = 0.0 if weight_column in midpoint_columns else None
                    heatmap_fig = px.density_mapbox(
                        heatmap_source,
                        lat="lat",
                        lon="lon",
                        z="weight",
                        radius=45,
                        opacity=0.8,
                        color_continuous_scale=colour_scales.get(
                            weight_column, px.colors.sequential.YlOrRd
                        ),
                        color_continuous_midpoint=midpoint,
                    )
                    hover_templates = {
                        None: f"{weight_label}: %{{z:.0f}} jobs<extra></extra>",
                        "volume_m3": f"{weight_label}: %{{z:.1f}} m³<extra></extra>",
                        "margin_total": f"{weight_label}: $%{{z:,.0f}}<extra></extra>",
                        "margin_per_m3": f"{weight_label}: $%{{z:,.0f}}/m³<extra></extra>",
                        "margin_total_pct": f"{weight_label}: %{{z:.1%}}<extra></extra>",
                        "margin_per_m3_pct": f"{weight_label}: %{{z:.1%}}<extra></extra>",
                    }
                    hover_template = hover_templates.get(
                        weight_column, f"{weight_label}: %{{z:.2f}}<extra></extra>"
                    )
                    for trace in heatmap_fig.data:
                        trace.hovertemplate = hover_template

                    heatmap_fig.update_layout(
                        mapbox={
                            "style": "carto-positron",
                            "center": centre,
                            "zoom": 4,
                        },
                        margin={"l": 0, "r": 0, "t": 0, "b": 0},
                        coloraxis_colorbar={"title": weight_label},
                    )
                    st.plotly_chart(heatmap_fig, use_container_width=True)
        else:
            centre_label = st.radio(
                "Isochrone centre",
                ("Origin", "Destination"),
                horizontal=True,
                help="Choose whether to anchor isochrones at route origins or destinations.",
            )
            iso_hours = st.slider(
                "Travel time horizon (hours)",
                min_value=0.5,
                max_value=24.0,
                value=4.0,
                step=0.5,
                help=(
                    "Approximate reach based on the corridor's average speed multiplied by this time horizon."
                ),
            )
            max_iso_routes = st.slider(
                "Maximum corridors to display",
                min_value=5,
                max_value=80,
                value=25,
                step=5,
                help="Limit the number of polygons rendered to keep the map readable.",
            )

            iso_source = build_isochrone_polygons(
                scoped_df,
                centre="origin" if centre_label == "Origin" else "destination",
                horizon_hours=float(iso_hours),
                max_routes=int(max_iso_routes),
            )

            if iso_source.empty:
                st.info(
                    "No geocoded routes with distance data are available to build isochrones for the current filters."
                )
            else:
                figure = go.Figure()
                palette = _ISOCHRONE_PALETTE or ["#636EFA"]

                for idx, (_, row) in enumerate(iso_source.iterrows()):
                    colour_hex = palette[idx % len(palette)]
                    r, g, b = _hex_to_rgb(colour_hex)
                    fill_colour = f"rgba({r},{g},{b},0.18)"
                    line_colour = f"rgba({r},{g},{b},0.9)"

                    figure.add_trace(
                        go.Scattermapbox(
                            lat=row["latitudes"],
                            lon=row["longitudes"],
                            mode="lines",
                            fill="toself",
                            line={"width": 2.0, "color": line_colour},
                            fillcolor=fill_colour,
                            name=row["label"],
                            hovertemplate=f"{row['tooltip']}<extra></extra>",
                        )
                    )

                    figure.add_trace(
                        go.Scattermapbox(
                            lat=[row["centre_lat"]],
                            lon=[row["centre_lon"]],
                            mode="markers",
                            marker={"size": 7, "color": line_colour},
                            hovertemplate=f"{row['tooltip']}<extra></extra>",
                            showlegend=False,
                        )
                    )

                centre_lat = float(iso_source["centre_lat"].mean())
                centre_lon = float(iso_source["centre_lon"].mean())

                figure.update_layout(
                    mapbox={
                        "style": "carto-positron",
                        "center": {"lat": centre_lat, "lon": centre_lon},
                        "zoom": 4,
                    },
                    margin={"l": 0, "r": 0, "t": 0, "b": 0},
                    legend={"orientation": "h", "yanchor": "bottom", "y": 0.01},
                )
                st.plotly_chart(figure, use_container_width=True)

        network_routes = prepare_profitability_map_data(scoped_df, break_even_value)
        render_network_map(
            network_routes,
            truck_positions,
            active_routes,
            toggle_key="network_map_live_overlay_toggle_tab",
        )

    with tab_map["Quote builder"]:
        saved_rowid = st.session_state.pop("quote_saved_rowid", None)
        if saved_rowid is not None:
            st.success(f"Quote saved as record #{saved_rowid}.")

        st.markdown("### Quote builder")
        st.caption(
            "Use a historical route to pre-fill the quick quote form, calculate pricing and optionally persist the result."
        )
        session_inputs: Optional[QuoteInput] = st.session_state.get(  # type: ignore[assignment]
            "quote_inputs"
        )
        quote_result: Optional[QuoteResult] = st.session_state.get(  # type: ignore[assignment]
            "quote_result"
        )
        manual_option = "Manual entry"
        map_columns = {"origin_lon", "origin_lat", "dest_lon", "dest_lat"}
        selected_route: Optional[pd.Series] = None

        if _QUOTE_COUNTRY_STATE_KEY not in st.session_state:
            initial_country = (
                session_inputs.country
                if session_inputs and session_inputs.country
                else COUNTRY_DEFAULT
            )
            st.session_state[_QUOTE_COUNTRY_STATE_KEY] = initial_country

        active_country = st.session_state.get(_QUOTE_COUNTRY_STATE_KEY)
        normalized_country: Optional[str]
        if isinstance(active_country, str):
            normalized_country = active_country.strip() or None
        else:
            normalized_country = None

        quote_prefill_df = enrich_missing_route_coordinates(
            filtered_df,
            conn,
            country=normalized_country,
        )

        if map_columns.issubset(quote_prefill_df.columns):
            map_routes = quote_prefill_df.dropna(subset=list(map_columns)).copy()
            if isinstance(normalized_country, str) and normalized_country:
                map_routes = filter_routes_by_country(map_routes, normalized_country)
            if not map_routes.empty:
                map_routes = map_routes.reset_index(drop=True)
                map_routes["route_label"] = map_routes.apply(_format_route_label, axis=1)
                option_list = [manual_option] + map_routes["route_label"].tolist()
                default_label = st.session_state.get("quote_selected_route", manual_option)
                if default_label not in option_list:
                    default_label = manual_option
                selected_label = st.selectbox(
                    "Prefill from historical route",
                    options=option_list,
                    index=option_list.index(default_label),
                    key="quote_selected_route",
                    help="Pick a historical job to pull its origin and destination into the form.",
                )
                if selected_label != manual_option:
                    selected_route = map_routes.loc[
                        map_routes["route_label"] == selected_label
                    ].iloc[0]
                    midpoint_lat = (
                        float(selected_route["origin_lat"]) + float(selected_route["dest_lat"])
                    ) / 2
                    midpoint_lon = (
                        float(selected_route["origin_lon"]) + float(selected_route["dest_lon"])
                    ) / 2
                    line_data = [
                        {
                            "from": [
                                float(selected_route["origin_lon"]),
                                float(selected_route["origin_lat"]),
                            ],
                            "to": [
                                float(selected_route["dest_lon"]),
                                float(selected_route["dest_lat"]),
                            ],
                        }
                    ]
                    scatter_data = [
                        {
                            "position": [
                                float(selected_route["origin_lon"]),
                                float(selected_route["origin_lat"]),
                            ],
                            "label": _first_non_empty(
                                selected_route,
                                ["origin", "origin_city", "origin_normalized", "origin_raw"],
                            )
                            or "Origin",
                            "color": [33, 150, 243, 200],
                        },
                        {
                            "position": [
                                float(selected_route["dest_lon"]),
                                float(selected_route["dest_lat"]),
                            ],
                            "label": _first_non_empty(
                                selected_route,
                                [
                                    "destination",
                                    "destination_city",
                                    "destination_normalized",
                                    "destination_raw",
                                ],
                            )
                            or "Destination",
                            "color": [244, 67, 54, 200],
                        },
                    ]
                    deck = pdk.Deck(
                        map_style="mapbox://styles/mapbox/light-v9",
                        initial_view_state=pdk.ViewState(
                            latitude=midpoint_lat,
                            longitude=midpoint_lon,
                            zoom=5,
                            pitch=30,
                        ),
                        layers=[
                            pdk.Layer(
                                "LineLayer",
                                data=line_data,
                                get_source_position="from",
                                get_target_position="to",
                                get_color=[33, 150, 243, 160],
                                get_width=5,
                            ),
                            pdk.Layer(
                                "ScatterplotLayer",
                                data=scatter_data,
                                get_position="position",
                                get_fill_color="color",
                                get_radius=40000,
                            ),
                            pdk.Layer(
                                "TextLayer",
                                data=scatter_data,
                                get_position="position",
                                get_text="label",
                                get_size=12,
                                size_units="meters",
                                size_scale=16,
                                get_alignment_baseline="top",
                            ),
                        ],
                    )
                    st.pydeck_chart(deck)
                    st.caption("Selected route visualised on the map.")
            else:
                st.info("No geocoded routes are available for the current filters yet.")
        else:
            st.info("Longitude/latitude columns are required to plot routes for quoting.")

        base_candidates: List[str] = [
            "cubic_m",
            "volume_m3",
            "volume_cbm",
            "volume",
            "cbm",
        ]
        for candidate in (filtered_mapping.volume, mapping.volume):
            if candidate and candidate not in base_candidates:
                base_candidates.append(candidate)

        default_origin = session_inputs.origin if session_inputs else ""
        default_destination = session_inputs.destination if session_inputs else ""
        default_volume = session_inputs.cubic_m if session_inputs else 30.0
        default_date = session_inputs.quote_date if session_inputs else date.today()
        default_modifiers = list(session_inputs.modifiers) if session_inputs else []
        if session_inputs is None:
            default_margin_percent: Optional[float] = DEFAULT_TARGET_MARGIN_PERCENT
        else:
            default_margin_percent = session_inputs.target_margin_percent
        default_country = st.session_state.get(_QUOTE_COUNTRY_STATE_KEY, COUNTRY_DEFAULT)

        if selected_route is not None:
            default_origin = _first_non_empty(
                selected_route,
                [
                    "origin",
                    "origin_normalized",
                    "origin_city",
                    "origin_raw",
                ],
            ) or default_origin
            default_destination = _first_non_empty(
                selected_route,
                [
                    "destination",
                    "destination_normalized",
                    "destination_city",
                    "destination_raw",
                ],
            ) or default_destination
            route_volume = _extract_route_volume(selected_route, base_candidates)
            if route_volume is not None:
                default_volume = route_volume
            route_date = _extract_route_date(selected_route)
            if route_date is not None:
                default_date = route_date
            route_country = _first_non_empty(
                selected_route, ["origin_country", "destination_country"]
            )
            if route_country:
                st.session_state[_QUOTE_COUNTRY_STATE_KEY] = route_country
                default_country = route_country

        modifier_options = [mod.id for mod in DEFAULT_MODIFIERS]
        modifier_labels: Dict[str, str] = {mod.id: mod.label for mod in DEFAULT_MODIFIERS}

        client_rows = conn.execute(
            """
            SELECT id, first_name, last_name, company_name, email, phone,
                   address_line1, address_line2, city, state, postcode, country, notes
            FROM clients
            ORDER BY
                CASE WHEN company_name IS NOT NULL AND TRIM(company_name) <> '' THEN 0 ELSE 1 END,
                LOWER(COALESCE(company_name, '')),
                LOWER(COALESCE(first_name, '')),
                LOWER(COALESCE(last_name, ''))
            """
        ).fetchall()
        client_option_values: List[Optional[int]] = [None] + [int(row[0]) for row in client_rows]
        client_label_map: Dict[int, str] = {
            int(row[0]): format_client_display(row[1], row[2], row[3])
            for row in client_rows
        }
        default_client_id = session_inputs.client_id if session_inputs else None
        default_client_details = session_inputs.client_details if session_inputs else None
        client_match_choice_state = st.session_state.get("quote_client_match_choice", -1)
        client_form_should_expand = bool(
            (default_client_id and default_client_id in client_option_values)
            or (
                default_client_details
                and hasattr(default_client_details, "has_any_data")
                and default_client_details.has_any_data()
            )
        )
        selected_client_id_form: Optional[int] = (
            default_client_id if default_client_id in client_option_values else None
        )
        entered_client_details_form: Optional[ClientDetails] = default_client_details
        match_choice_form = client_match_choice_state

        with st.form("quote_builder_form"):
            origin_value = st.text_input("Origin", value=default_origin)
            destination_value = st.text_input(
                "Destination", value=default_destination
            )
            country_value = st.text_input(
                "Country",
                value=default_country or COUNTRY_DEFAULT,
                key=_QUOTE_COUNTRY_STATE_KEY,
            )
            cubic_m_value = st.number_input(
                "Volume (m³)",
                min_value=1.0,
                value=float(default_volume or 1.0),
                step=1.0,
            )
            quote_date_value = st.date_input("Move date", value=default_date)
            selected_modifier_ids = st.multiselect(
                "Modifiers",
                options=modifier_options,
                default=[mid for mid in default_modifiers if mid in modifier_options],
                format_func=lambda mod_id: modifier_labels.get(mod_id, mod_id),
            )
            margin_cols = st.columns(2)
            apply_margin = margin_cols[0].checkbox(
                "Apply margin",
                value=default_margin_percent is not None,
                help="Include a target margin percentage on top of calculated costs.",
            )
            margin_percent_value = margin_cols[1].number_input(
                "Target margin %",
                min_value=0.0,
                max_value=100.0,
                value=float(
                    default_margin_percent
                    if default_margin_percent is not None
                    else DEFAULT_TARGET_MARGIN_PERCENT
                ),
                step=1.0,
                help=(
                    "Enter the desired margin percentage. The value is only used when 'Apply margin'"
                    " is enabled."
                ),
            )
            with st.expander(
                "Client details (optional)", expanded=client_form_should_expand
            ):
                existing_index = 0
                if selected_client_id_form in client_option_values:
                    existing_index = client_option_values.index(selected_client_id_form)
                selected_client_id_form = st.selectbox(
                    "Link to existing client",
                    options=client_option_values,
                    index=existing_index,
                    format_func=lambda cid: (
                        "No client linked"
                        if cid is None
                        else client_label_map.get(cid, f"Client #{cid}")
                    ),
                )
                st.caption(
                    "Enter details below to create a client record if no existing client applies."
                )
                company_input = st.text_input(
                    "Company name",
                    value=(
                        default_client_details.company_name
                        if default_client_details and default_client_details.company_name
                        else ""
                    ),
                )
                first_name_input = st.text_input(
                    "First name",
                    value=(
                        default_client_details.first_name
                        if default_client_details and default_client_details.first_name
                        else ""
                    ),
                )
                last_name_input = st.text_input(
                    "Last name",
                    value=(
                        default_client_details.last_name
                        if default_client_details and default_client_details.last_name
                        else ""
                    ),
                )
                email_input = st.text_input(
                    "Email",
                    value=(
                        default_client_details.email
                        if default_client_details and default_client_details.email
                        else ""
                    ),
                )
                phone_input = st.text_input(
                    "Phone",
                    value=(
                        default_client_details.phone
                        if default_client_details and default_client_details.phone
                        else ""
                    ),
                )
                address_line1_input = st.text_input(
                    "Address line 1",
                    value=(
                        default_client_details.address_line1
                        if default_client_details and default_client_details.address_line1
                        else ""
                    ),
                )
                address_line2_input = st.text_input(
                    "Address line 2",
                    value=(
                        default_client_details.address_line2
                        if default_client_details and default_client_details.address_line2
                        else ""
                    ),
                )
                city_input = st.text_input(
                    "City / Suburb",
                    value=(
                        default_client_details.city
                        if default_client_details and default_client_details.city
                        else ""
                    ),
                )
                state_input = st.text_input(
                    "State / Territory",
                    value=(
                        default_client_details.state
                        if default_client_details and default_client_details.state
                        else ""
                    ),
                )
                postcode_input = st.text_input(
                    "Postcode",
                    value=(
                        default_client_details.postcode
                        if default_client_details and default_client_details.postcode
                        else ""
                    ),
                )
                client_country_default = (
                    default_client_details.country
                    if default_client_details and default_client_details.country
                    else country_value
                    if country_value
                    else COUNTRY_DEFAULT
                )
                client_country_input = st.text_input(
                    "Client country",
                    value=client_country_default,
                )
                notes_input = st.text_area(
                    "Notes",
                    value=(
                        default_client_details.notes
                        if default_client_details and default_client_details.notes
                        else ""
                    ),
                    height=80,
                )
                entered_client_details_form = ClientDetails(
                    company_name=company_input,
                    first_name=first_name_input,
                    last_name=last_name_input,
                    email=email_input,
                    phone=phone_input,
                    address_line1=address_line1_input,
                    address_line2=address_line2_input,
                    city=city_input,
                    state=state_input,
                    postcode=postcode_input,
                    country=client_country_input,
                    notes=notes_input,
                )
                match_choice_form = -1
                if (
                    selected_client_id_form is None
                    and entered_client_details_form.has_any_data()
                ):
                    matches = find_client_matches(conn, entered_client_details_form)
                    if matches:
                        match_labels = {
                            match.id: f"{match.display_name} ({match.reason})"
                            for match in matches
                        }
                        warning_lines = "\n".join(
                            f"- {label}" for label in match_labels.values()
                        )
                        st.warning(
                            "Potential existing clients found:\n" + warning_lines
                        )
                        match_options = [-1] + list(match_labels.keys())
                        default_choice = (
                            client_match_choice_state
                            if client_match_choice_state in match_options
                            else -1
                        )
                        match_choice_form = st.selectbox(
                            "Would you like to link one of these clients?",
                            options=match_options,
                            index=match_options.index(default_choice),
                            format_func=lambda value: (
                                "Create new client"
                                if value == -1
                                else match_labels.get(value, f"Client #{value}")
                            ),
                            key="quote_client_match_choice",
                        )
                    else:
                        st.session_state.pop("quote_client_match_choice", None)
                else:
                    st.session_state.pop("quote_client_match_choice", None)
            submitted = st.form_submit_button("Calculate quote")

        stored_inputs = session_inputs

        if submitted:
            if not origin_value or not destination_value:
                st.error("Origin and destination are required to calculate a quote.")
            else:
                margin_to_apply = float(margin_percent_value) if apply_margin else None
                selected_client_id_final = selected_client_id_form
                client_details_to_store: Optional[ClientDetails]
                if (
                    entered_client_details_form
                    and entered_client_details_form.has_any_data()
                ):
                    client_details_to_store = entered_client_details_form
                else:
                    client_details_to_store = None

                submission_valid = True
                if selected_client_id_final is None and client_details_to_store is not None:
                    if match_choice_form not in (-1, None):
                        selected_client_id_final = int(match_choice_form)

                if submission_valid:
                    quote_inputs = QuoteInput(
                        origin=origin_value,
                        destination=destination_value,
                        cubic_m=float(cubic_m_value),
                        quote_date=quote_date_value,
                        modifiers=list(selected_modifier_ids),
                        target_margin_percent=margin_to_apply,
                        country=country_value or COUNTRY_DEFAULT,
                        client_id=selected_client_id_final,
                        client_details=client_details_to_store,
                    )
                    try:
                        result = calculate_quote(conn, quote_inputs)
                    except RuntimeError as exc:
                        st.error(str(exc))
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        st.session_state["quote_inputs"] = quote_inputs
                        st.session_state["quote_result"] = result
                        st.session_state["quote_manual_override_enabled"] = False
                        st.session_state["quote_manual_override_amount"] = float(
                            result.final_quote
                        )
                        st.session_state["quote_pin_override"] = _initial_pin_state(result)
                        st.session_state.pop(_HAVERSINE_MODAL_STATE_KEY, None)
                        _set_query_params(view="Quote builder")
                        st.success("Quote calculated. Review the breakdown below.")
                        stored_inputs = quote_inputs
                        quote_result = result

        stored_inputs = st.session_state.get("quote_inputs")
        quote_result = st.session_state.get("quote_result")

        if quote_result and stored_inputs:
            st.markdown("#### Quote output")
            manual_enabled_key = "quote_manual_override_enabled"
            manual_amount_key = "quote_manual_override_amount"
            if manual_enabled_key not in st.session_state:
                st.session_state[manual_enabled_key] = (
                    quote_result.manual_quote is not None
                )
            if manual_amount_key not in st.session_state:
                st.session_state[manual_amount_key] = float(
                    quote_result.manual_quote
                    if quote_result.manual_quote is not None
                    else quote_result.final_quote
                )
            manual_override_enabled = bool(
                st.session_state.get(manual_enabled_key, False)
            )
            manual_override_amount = float(
                st.session_state.get(
                    manual_amount_key, quote_result.final_quote
                )
            )
            if manual_override_enabled:
                quote_result.manual_quote = manual_override_amount
            else:
                quote_result.manual_quote = None
            quote_result.summary_text = build_summary(stored_inputs, quote_result)
            st.session_state["quote_result"] = quote_result
            client_label: Optional[str] = None
            if stored_inputs.client_details and stored_inputs.client_details.display_name():
                client_label = stored_inputs.client_details.display_name()
            elif stored_inputs.client_id is not None:
                client_label = client_label_map.get(stored_inputs.client_id)
            if client_label:
                st.write(f"**Client:** {client_label}")
            st.write(
                f"**Route:** {quote_result.origin_resolved} → {quote_result.destination_resolved}"
            )
            st.write(
                f"**Distance:** {quote_result.distance_km:.1f} km ({quote_result.duration_hr:.1f} h)"
            )

            suggestion_cols = st.columns(2)

            def _render_address_feedback(
                col: "st.delta_generator.DeltaGenerator",
                label: str,
                candidates: Optional[List[str]],
                suggestions: Optional[List[str]],
                ambiguities: Optional[Dict[str, Sequence[str]]],
            ) -> None:
                clean_candidates = [c for c in candidates or [] if c]
                clean_suggestions = [s for s in suggestions or [] if s]
                clean_ambiguities = {
                    abbr: list(options)
                    for abbr, options in (ambiguities or {}).items()
                    if options
                }
                if not (
                    clean_candidates
                    or clean_suggestions
                    or clean_ambiguities
                ):
                    col.caption(f"No {label.lower()} corrections suggested.")
                    return

                col.markdown(f"**{label} corrections & suggestions**")
                if clean_candidates:
                    col.caption("Candidates considered during normalization:")
                    col.markdown(
                        "\n".join(f"- {candidate}" for candidate in clean_candidates)
                    )
                if clean_suggestions:
                    col.caption("Autocorrected place names from geocoding:")
                    col.markdown(
                        "\n".join(f"- {suggestion}" for suggestion in clean_suggestions)
                    )
                if clean_ambiguities:
                    col.caption("Ambiguous abbreviations detected:")
                    col.markdown(
                        "\n".join(
                            f"- **{abbr}** → {', '.join(options)}"
                            for abbr, options in clean_ambiguities.items()
                        )
                    )

            _render_address_feedback(
                suggestion_cols[0],
                "Origin",
                quote_result.origin_candidates,
                quote_result.origin_suggestions,
                quote_result.origin_ambiguities,
            )
            _render_address_feedback(
                suggestion_cols[1],
                "Destination",
                quote_result.destination_candidates,
                quote_result.destination_suggestions,
                quote_result.destination_ambiguities,
            )

            pin_state = _ensure_pin_state(quote_result)
            pin_related_notes: List[str] = []
            straight_line_detected = False
            for notes in (
                quote_result.origin_suggestions,
                quote_result.destination_suggestions,
            ):
                for note in notes or []:
                    if not note:
                        continue
                    lowered = note.lower()
                    if _PIN_NOTE.lower() in lowered or "straight-line" in lowered:
                        pin_related_notes.append(note)
                    if "straight-line" in lowered:
                        straight_line_detected = True

            if straight_line_detected and not st.session_state.get(
                _HAVERSINE_MODAL_STATE_KEY, False
            ):
                with st.modal(
                    "Routing fell back to a straight-line estimate",
                    key="quote_haversine_modal",
                ):
                    st.warning(
                        "OpenRouteService could not find a routable point within 350 m. "
                        "The quote currently relies on a straight-line distance estimate."
                    )
                    st.caption(
                        "Drop manual pins, click \"Snap pins to nearest road\", or edit the coordinates "
                        "below before recalculating to improve accuracy."
                    )
                    if st.button(
                        "Dismiss warning", key="quote_haversine_modal_dismiss"
                    ):
                        st.session_state[_HAVERSINE_MODAL_STATE_KEY] = True
                        _rerun_app()

            st.markdown("#### Manual pins for routing")
            if pin_related_notes and not pin_state.get("enabled", False):
                st.warning(
                    "Routing relied on snapping or a straight-line fallback. Drop pins or use "
                    '"Snap pins to nearest road" to improve accuracy before recalculating.'
                )
            else:
                st.caption(
                    "Drop a pin for each address when ORS cannot find a routable point within 350 m."
                )
            st.caption(
                "Click the maps or edit the latitude/longitude values to fine-tune the override pins."
            )

            control_cols = st.columns([3, 2])
            with control_cols[1]:
                snap_feedback = st.empty()
                snap_clicked = st.button(
                    "Snap pins to nearest road",
                    type="secondary",
                    key="quote_snap_to_nearest_road",
                    help=(
                        "Use OpenRouteService's nearest endpoint to move each pin onto the closest "
                        "routable road before recalculating."
                    ),
                )
            if snap_clicked:
                origin_lon_default, origin_lat_default = _pin_coordinates(
                    pin_state["origin"]
                )
                dest_lon_default, dest_lat_default = _pin_coordinates(
                    pin_state["destination"]
                )
                try:
                    snap_result = snap_coordinates_to_road(
                        (origin_lon_default, origin_lat_default),
                        (dest_lon_default, dest_lat_default),
                    )
                except RuntimeError as exc:
                    snap_feedback.error(f"Unable to snap pins: {exc}")
                else:
                    pin_state["origin"] = {
                        "lon": snap_result.origin[0],
                        "lat": snap_result.origin[1],
                    }
                    pin_state["destination"] = {
                        "lon": snap_result.destination[0],
                        "lat": snap_result.destination[1],
                    }
                    st.session_state[_pin_lon_key("quote_origin_pin_map")] = float(
                        snap_result.origin[0]
                    )
                    st.session_state[_pin_lat_key("quote_origin_pin_map")] = float(
                        snap_result.origin[1]
                    )
                    st.session_state[_pin_lon_key("quote_destination_pin_map")] = float(
                        snap_result.destination[0]
                    )
                    st.session_state[_pin_lat_key("quote_destination_pin_map")] = float(
                        snap_result.destination[1]
                    )
                    if snap_result.changed:
                        snap_feedback.success(
                            "Pins snapped to the nearest routable road."
                        )
                    else:
                        snap_feedback.info(
                            "Pins already align with the nearest routable road."
                        )

            pin_cols = st.columns(2)
            with pin_cols[0]:
                origin_lon, origin_lat = _render_pin_picker(
                    "Origin", map_key="quote_origin_pin_map", entry=pin_state["origin"]
                )
                st.caption(f"Origin pin: {origin_lat:.5f}, {origin_lon:.5f}")
            with pin_cols[1]:
                dest_lon, dest_lat = _render_pin_picker(
                    "Destination",
                    map_key="quote_destination_pin_map",
                    entry=pin_state["destination"],
                )
                st.caption(f"Destination pin: {dest_lat:.5f}, {dest_lon:.5f}")

            pin_state["origin"] = {"lon": origin_lon, "lat": origin_lat}
            pin_state["destination"] = {"lon": dest_lon, "lat": dest_lat}
            use_manual_pins = st.checkbox(
                "Use these pins for the next calculation",
                value=pin_state.get("enabled", False),
                key="quote_use_pin_overrides",
                help="Enable to re-run the quote using the pins above.",
            )
            pin_state["enabled"] = use_manual_pins
            st.session_state["quote_pin_override"] = pin_state

            if st.button(
                "Recalculate with manual pins",
                type="secondary",
                disabled=not use_manual_pins,
            ):
                manual_inputs = QuoteInput(
                    origin=stored_inputs.origin,
                    destination=stored_inputs.destination,
                    cubic_m=stored_inputs.cubic_m,
                    quote_date=stored_inputs.quote_date,
                    modifiers=list(stored_inputs.modifiers),
                    target_margin_percent=stored_inputs.target_margin_percent,
                    country=stored_inputs.country,
                    origin_coordinates=(origin_lon, origin_lat),
                    destination_coordinates=(dest_lon, dest_lat),
                )
                try:
                    manual_result = calculate_quote(conn, manual_inputs)
                except RuntimeError as exc:
                    st.error(str(exc))
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.session_state["quote_inputs"] = manual_inputs
                    st.session_state["quote_result"] = manual_result
                    st.session_state["quote_manual_override_enabled"] = False
                    st.session_state["quote_manual_override_amount"] = float(
                        manual_result.final_quote
                    )
                    pin_override_state = _initial_pin_state(manual_result)
                    pin_override_state["enabled"] = True
                    st.session_state["quote_pin_override"] = pin_override_state
                    st.session_state.pop(_HAVERSINE_MODAL_STATE_KEY, None)
                    st.success("Quote recalculated using manual pins.")
                    _set_query_params(view="Quote builder")
                    _rerun_app()

            metric_cols = st.columns(4)
            metric_cols[0].metric(
                "Final quote", format_currency(quote_result.final_quote)
            )
            metric_cols[1].metric(
                "Total before margin",
                format_currency(quote_result.total_before_margin),
            )
            metric_cols[2].metric(
                "Base subtotal", format_currency(quote_result.base_subtotal)
            )
            metric_cols[3].metric(
                "Distance (km)",
                f"{quote_result.distance_km:.1f}",
                f"{quote_result.duration_hr:.1f} h",
            )
            st.markdown(
                f"**Seasonal adjustment:** {quote_result.seasonal_label} ×{quote_result.seasonal_multiplier:.2f}"
            )
            if quote_result.margin_percent is not None:
                st.markdown(
                    f"**Margin:** {quote_result.margin_percent:.1f}% applied."
                )
            else:
                st.markdown("**Margin:** Not applied.")

            with st.expander("Base calculation details"):
                base_rows = [
                    {
                        "Component": "Base callout",
                        "Amount": format_currency(
                            quote_result.base_components.get("base_callout", 0.0)
                        ),
                    },
                    {
                        "Component": "Handling cost",
                        "Amount": format_currency(
                            quote_result.base_components.get("handling_cost", 0.0)
                        ),
                    },
                    {
                        "Component": "Linehaul cost",
                        "Amount": format_currency(
                            quote_result.base_components.get("linehaul_cost", 0.0)
                        ),
                    },
                    {
                        "Component": "Effective volume (m³)",
                        "Amount": f"{quote_result.base_components.get('effective_m3', stored_inputs.cubic_m):.1f}",
                    },
                    {
                        "Component": "Load factor",
                        "Amount": f"{quote_result.base_components.get('load_factor', 1.0):.2f}",
                    },
                ]
                st.table(pd.DataFrame(base_rows))

            with st.expander("Modifiers applied"):
                if quote_result.modifier_details:
                    modifier_rows = [
                        {
                            "Modifier": item["label"],
                            "Calculation": (
                                format_currency(item["value"])
                                if item["calc_type"] == "flat"
                                else f"{item['value'] * 100:.0f}% of base"
                            ),
                            "Amount": format_currency(item["amount"]),
                        }
                        for item in quote_result.modifier_details
                    ]
                    st.table(pd.DataFrame(modifier_rows))
                else:
                    st.write("No modifiers applied.")

            with st.expander("Copyable summary"):
                st.code(quote_result.summary_text)

            st.markdown("#### Submit quote")
            st.caption(
                "Optionally override the calculated quote amount before saving."
            )
            manual_override_enabled = st.checkbox(
                "Apply manual quote override",
                help=(
                    "Enable to store a different quote amount alongside the calculated value."
                ),
                key=manual_enabled_key,
            )
            manual_override_amount = st.number_input(
                "Manual quote amount",
                min_value=0.0,
                step=50.0,
                format="%.2f",
                key=manual_amount_key,
                disabled=not manual_override_enabled,
                help=(
                    "Enter the agreed quote to store in addition to the calculated amount."
                ),
            )
            action_cols = st.columns(2)
            if action_cols[0].button("Submit quote", type="primary"):
                manual_to_store: Optional[float]
                if manual_override_enabled:
                    manual_to_store = float(manual_override_amount)
                    if not math.isfinite(manual_to_store) or manual_to_store <= 0:
                        st.error("Manual quote must be a positive number.")
                        manual_to_store = None
                    else:
                        quote_result.manual_quote = manual_to_store
                else:
                    manual_to_store = None
                    quote_result.manual_quote = None
                quote_result.summary_text = build_summary(stored_inputs, quote_result)
                st.session_state["quote_result"] = quote_result
                should_persist = not (
                    manual_override_enabled and manual_to_store is None
                )
                trigger_null_client_modal = False
                if should_persist:
                    if not stored_inputs:
                        st.error("Calculate the quote before submitting it.")
                        should_persist = False
                    else:
                        client_details = stored_inputs.client_details
                        if stored_inputs.client_id is None:
                            if client_details and client_details.has_any_data():
                                if not client_details.has_identity():
                                    st.error(
                                        "Provide a company name or both first and last names when creating a client."
                                    )
                                    should_persist = False
                            else:
                                trigger_null_client_modal = True
                                should_persist = False
                if trigger_null_client_modal:
                    st.session_state[_NULL_CLIENT_MODAL_STATE_KEY] = True
                if should_persist:
                    try:
                        rowid = persist_quote(
                            conn,
                            stored_inputs,
                            quote_result,
                            manual_quote=manual_to_store,
                        )
                    except Exception as exc:  # pragma: no cover - UI feedback path
                        st.error(f"Failed to persist quote: {exc}")
                    else:
                        st.session_state["quote_saved_rowid"] = rowid
                        _set_query_params(view="Quote builder")
                        _rerun_app()
            if action_cols[1].button("Reset quote builder"):
                st.session_state.pop("quote_result", None)
                st.session_state.pop("quote_inputs", None)
                st.session_state.pop("quote_manual_override_enabled", None)
                st.session_state.pop("quote_manual_override_amount", None)
                st.session_state.pop("quote_pin_override", None)
                st.session_state.pop(_HAVERSINE_MODAL_STATE_KEY, None)
                _set_query_params(view="Quote builder")
                _rerun_app()

            if st.session_state.get(_NULL_CLIENT_MODAL_STATE_KEY):
                if _NULL_CLIENT_COMPANY_KEY not in st.session_state:
                    st.session_state[_NULL_CLIENT_COMPANY_KEY] = (
                        _NULL_CLIENT_DEFAULT_COMPANY
                    )
                if _NULL_CLIENT_NOTES_KEY not in st.session_state:
                    st.session_state[_NULL_CLIENT_NOTES_KEY] = (
                        _NULL_CLIENT_DEFAULT_NOTES
                    )
                with st.modal(
                    "Link this quote to a client",
                    key="quote_null_client_modal",
                ):
                    st.warning(
                        "A client must be linked before submitting a quote."
                        " Select an existing client in the form or use the placeholder"
                        " details below."
                    )
                    st.caption(
                        "Applying the filler details will populate the client fields in the"
                        " quote builder. You can then review and submit again."
                    )
                    st.text_input(
                        "Filler company name",
                        key=_NULL_CLIENT_COMPANY_KEY,
                    )
                    st.text_area(
                        "Notes (optional)",
                        key=_NULL_CLIENT_NOTES_KEY,
                        height=80,
                    )
                    modal_cols = st.columns(2)
                    if modal_cols[0].button(
                        "Use filler client", key="quote_null_client_apply"
                    ):
                        filler_details = ClientDetails(
                            company_name=(
                                st.session_state.get(_NULL_CLIENT_COMPANY_KEY)
                                or _NULL_CLIENT_DEFAULT_COMPANY
                            ),
                            notes=(
                                st.session_state.get(_NULL_CLIENT_NOTES_KEY)
                                or _NULL_CLIENT_DEFAULT_NOTES
                            ),
                        )
                        if stored_inputs:
                            stored_inputs.client_id = None
                            stored_inputs.client_details = filler_details
                            st.session_state["quote_inputs"] = stored_inputs
                        st.session_state[_NULL_CLIENT_MODAL_STATE_KEY] = False
                        _rerun_app()
                    if modal_cols[1].button(
                        "Cancel", key="quote_null_client_cancel"
                    ):
                        st.session_state[_NULL_CLIENT_MODAL_STATE_KEY] = False
                        st.session_state.pop(_NULL_CLIENT_COMPANY_KEY, None)
                        st.session_state.pop(_NULL_CLIENT_NOTES_KEY, None)
                        _rerun_app()

    with tab_map["Optimizer"]:
        st.markdown("### Margin optimizer")
        st.caption(
            "Generate corridor-level price uplift suggestions using the filtered job set."
        )

        optimizer_state: Dict[str, Any] = st.session_state.setdefault(
            "optimizer_state", {}
        )

        if not can_run_optimizer(filtered_df):
            st.info(
                "Optimizer requires price and cost per m³ columns. Import jobs with "
                "$ / m³ and cost data to enable recommendations."
            )
        else:
            defaults = optimizer_state.get(
                "defaults",
                {
                    "target_margin": 120.0,
                    "max_uplift": 25.0,
                    "min_job_count": 3,
                },
            )
            with st.form("optimizer_form"):
                target_margin = st.number_input(
                    "Target margin per m³",
                    min_value=0.0,
                    value=float(defaults.get("target_margin", 120.0)),
                    step=5.0,
                    help="Desired margin buffer applied to each corridor's historical median.",
                )
                max_uplift_pct = st.slider(
                    "Cap uplift %",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(defaults.get("max_uplift", 25.0)),
                    help="Limit how far the optimizer can move prices above the historical median.",
                )
                min_job_count = st.slider(
                    "Minimum jobs per corridor",
                    min_value=1,
                    max_value=10,
                    value=int(defaults.get("min_job_count", 3)),
                    help="Require a minimum number of jobs before trusting a recommendation.",
                )
                submitted = st.form_submit_button(
                    "Run optimizer", help="Recalculate uplifts for the current filters."
                )

            if submitted:
                params = OptimizerParameters(
                    target_margin_per_m3=target_margin,
                    max_uplift_pct=max_uplift_pct,
                    min_job_count=min_job_count,
                )
                run = run_margin_optimizer(filtered_df, params)
                optimizer_state["last_run"] = run
                optimizer_state["defaults"] = {
                    "target_margin": target_margin,
                    "max_uplift": max_uplift_pct,
                    "min_job_count": min_job_count,
                }
                st.session_state["optimizer_state"] = optimizer_state
                if run.recommendations:
                    st.success("Optimizer complete — review the suggested uplifts below.")
                else:
                    st.warning(
                        "Optimizer finished but no corridors met the criteria. Try lowering the minimum job count."
                    )

            last_run: Optional[OptimizerRun] = optimizer_state.get("last_run")
            if last_run:
                run_time = last_run.executed_at.strftime("%Y-%m-%d %H:%M UTC")
                st.caption(
                    f"Last run: {run_time} · Target margin ${last_run.parameters.target_margin_per_m3:,.0f}/m³ · "
                    f"Max uplift {last_run.parameters.max_uplift_pct:.0f}%"
                )
                recommendations_df = recommendations_to_frame(last_run.recommendations)
                if recommendations_df.empty:
                    st.info(
                        "No eligible corridors were found — adjust parameters or widen the dashboard filters."
                    )
                else:
                    metric_cols = st.columns(3)
                    metric_cols[0].metric(
                        "Corridors analysed", len(recommendations_df)
                    )
                    metric_cols[1].metric(
                        "Median uplift $/m³",
                        f"${recommendations_df['Uplift $/m³'].median():,.2f}",
                    )
                    metric_cols[2].metric(
                        "Highest uplift %",
                        f"{recommendations_df['Uplift %'].max():.1f}%",
                    )

                    chart = px.bar(
                        recommendations_df,
                        x="Corridor",
                        y="Uplift $/m³",
                        hover_data=["Recommended $/m³", "Uplift %", "Notes"],
                        title="Recommended uplift by corridor",
                    )
                    chart.update_layout(margin={"l": 0, "r": 0, "t": 40, "b": 0})
                    st.plotly_chart(chart, use_container_width=True)

                    st.dataframe(recommendations_df, use_container_width=True)

                    csv_data = recommendations_df.to_csv(index=False)
                    st.download_button(
                        "Download optimizer report",
                        csv_data,
                        file_name="optimizer_recommendations.csv",
                        mime="text/csv",
                    )

        st.info(
            "Optimizer works on the same filters applied across the dashboard, making it safe for non-technical teams to explore 'what if' pricing scenarios."
        )

    st.subheader("Filtered jobs")
    display_columns = [
        col
        for col in [
            "job_date",
            "corridor_display",
            "client_display",
            "price_per_m3",
        ]
        if col in filtered_df.columns
    ]
    remaining_columns = [
        col for col in filtered_df.columns if col not in display_columns
    ]
    st.dataframe(filtered_df[display_columns + remaining_columns])

    csv_buffer = io.StringIO()
    filtered_df.to_csv(csv_buffer, index=False)
    st.download_button(
        "Export filtered rows",
        csv_buffer.getvalue(),
        file_name="price_distribution_filtered.csv",
        mime="text/csv",
    )
