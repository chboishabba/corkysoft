"""Streamlit entrypoint for the price distribution dashboard."""
from __future__ import annotations

import io
import json
import math
from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from dashboard.app import render_price_distribution_dashboard
try:
    import folium
    from streamlit_folium import st_folium
except ModuleNotFoundError:  # pragma: no cover - optional dependency for pin UI
    folium = None  # type: ignore[assignment]
    st_folium = None  # type: ignore[assignment]

from dashboard.state import (
    _ensure_pin_state,
    _first_non_empty,
    _get_query_params,
    _initial_pin_state,
    _rerun_app,
    _set_query_params,
)
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
from dashboard.components.optimizer import render_optimizer
from dashboard.components.summary import render_summary
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
from corkysoft.repo import ensure_schema as ensure_quote_schema
from dashboard.components.maps import (
    _hex_to_rgb,
    build_route_map,
    render_network_map,
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
from dashboard.components.quote_builder import render_quote_builder


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
_QUOTE_COUNTRY_STATE_KEY = "quote_builder_country"
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


def _geojson_to_path(value: Any) -> Optional[List[List[float]]]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        coords = extract_route_path(value)
    except Exception:
        return None
    return [[float(lon), float(lat)] for lat, lon in coords]


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
        format="%.6f",
        key=lat_key,
    )
    lon_input = st.number_input(
        f"{label} longitude",
        format="%.6f",
        key=lon_key,
    )

    current_lat = float(lat_input)
    current_lon = float(lon_input)

    entry["lon"] = current_lon
    entry["lat"] = current_lat
    st.session_state["quote_pin_override"] = st.session_state.get("quote_pin_override", {})
    return current_lon, current_lat


def main() -> None:
    """Configure the Streamlit page and render the dashboard."""
    st.set_page_config(
        page_title="Price distribution by corridor",
        layout="wide",
    )
    render_price_distribution_dashboard()


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


if __name__ == "__main__":
    main()


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
    use_route_geometry: bool = True,
) -> go.Figure:
    """Construct a Plotly Mapbox figure showing coloured routes and points."""

    def _row_route_points(row: pd.Series) -> List[Tuple[float, float]]:
        """Return the ordered ``(lat, lon)`` points for ``row`` when available."""

        if not use_route_geometry:
            return []

        geojson_value = row.get("route_geojson")
        if isinstance(geojson_value, (bytes, bytearray, memoryview)):
            try:
                geojson_value = geojson_value.decode("utf-8")
            except Exception:
                geojson_value = None
        if isinstance(geojson_value, dict):
            try:
                geojson_value = json.dumps(geojson_value)
            except (TypeError, ValueError):
                geojson_value = None
        if isinstance(geojson_value, str):
            geojson_str = geojson_value.strip()
            if geojson_str:
                try:
                    return extract_route_path(geojson_str)
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
                        go.Scattermapbox(
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
                        go.Scattermapbox(
                            lat=marker_lat,
                            lon=marker_lon,
                            mode="markers",
                            marker={
                                "size": 9,
                                "color": colour,
                                "opacity": 0.85,
                            },
                            text=marker_text,
                            hovertemplate="%{text}<extra></extra>",
                            name=str(display_value),
                            legendgroup=str(value),
                        )
                    )

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
            cmid_value: Optional[float] = None
            if min_value < 0.0 < max_value:
                symmetric_bound = max(abs(min_value), abs(max_value))
                min_value = -symmetric_bound
                max_value = symmetric_bound
                cmid_value = 0.0
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
                    lat_values: list[float | None] = []
                    lon_values: list[float | None] = []
                    text_values: list[str | None] = []
                    display_candidate = row.get("map_colour_display")
                    display_value = (
                        display_candidate
                        if _normalise_text(display_candidate)
                        else f"{value:,.2f}"
                    )
                    context = _route_context(row, display_value=display_value)
                    base_text = context["base_text"]

                    route_points = _row_route_points(row)
                    if route_points:
                        for lat, lon in route_points:
                            lat_values.append(lat)
                            lon_values.append(lon)
                            text_values.append(base_text)
                        lat_values.append(None)
                        lon_values.append(None)
                        text_values.append(None)
                    else:
                        origin_lat = _coerce_float(row.get("origin_lat"))
                        origin_lon = _coerce_float(row.get("origin_lon"))
                        dest_lat = _coerce_float(row.get("dest_lat"))
                        dest_lon = _coerce_float(row.get("dest_lon"))
                        if (
                            origin_lat is None
                            or origin_lon is None
                            or dest_lat is None
                            or dest_lon is None
                        ):
                            continue
                        lat_values = [origin_lat, dest_lat, None]
                        lon_values = [origin_lon, dest_lon, None]
                        text_values = [base_text, base_text, None]

                    if not lat_values or not lon_values:
                        continue

                    figure.add_trace(
                        go.Scattermapbox(
                            lat=lat_values,
                            lon=lon_values,
                            mode="lines",
                            line={"width": 3, "color": colour},
                            showlegend=False,
                            text=text_values,
                            hovertemplate="%{text}<extra></extra>",
                        )
                    )

            if show_points:
                marker_lat: list[float] = []
                marker_lon: list[float] = []
                marker_text: list[str] = []
                marker_values: list[float] = []
                for _, row in plot_df.iterrows():
                    value = float(row["map_colour_value"])
                    display_candidate = row.get("map_colour_display")
                    display_value = _normalise_text(display_candidate) or f"{value:,.2f}"
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
                        marker_values.append(value)

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
                        marker_values.append(value)

                if marker_lat and marker_lon:
                    marker_config = {
                        "size": 9,
                        "color": marker_values,
                        "colorscale": colour_scale,
                        "cmin": min_value,
                        "cmax": max_value,
                        "opacity": 0.85,
                        "colorbar": colorbar_dict,
                    }
                    if cmid_value is not None:
                        marker_config["cmid"] = cmid_value
                    figure.add_trace(
                        go.Scattermapbox(
                            lat=marker_lat,
                            lon=marker_lon,
                            mode="markers",
                            marker=marker_config,
                            text=marker_text,
                            hovertemplate="%{text}<extra></extra>",
                            showlegend=False,
                        )
                    )
            else:
                coords_df = plot_df[["origin_lat", "origin_lon"]].apply(
                    pd.to_numeric, errors="coerce"
                )
                coords_df = coords_df.dropna()
                if not coords_df.empty:
                    marker_config = {
                        "size": 0.0001,
                        "color": numeric_values.loc[coords_df.index].tolist(),
                        "colorscale": colour_scale,
                        "cmin": min_value,
                        "cmax": max_value,
                        "colorbar": colorbar_dict,
                        "opacity": 0.0,
                    }
                    if cmid_value is not None:
                        marker_config["cmid"] = cmid_value
                    text_values = [
                        _route_context(plot_df.loc[idx])["base_text"]
                        for idx in coords_df.index
                    ]
                    figure.add_trace(
                        go.Scattermapbox(
                            lat=marker_lat,
                            lon=marker_lon,
                            mode="markers",
                            marker=marker_config,
                            hoverinfo="skip",
                            marker={
                                "size": 0.0001,
                                "color": marker_values,
                                "colorscale": colour_scale,
                                "cmin": min_value,
                                "cmax": max_value,
                                "colorbar": colorbar_dict,
                                "opacity": 0.0,
                            },
                            text=text_values,
                            hovertemplate="%{text}<extra></extra>",
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


def _get_query_params() -> Dict[str, List[str]]:
    """Return query parameters as a dictionary of lists."""
    query_params = getattr(st, "query_params", None)
    if query_params is not None:
        return {key: query_params.get_all(key) for key in query_params.keys()}
    return st.experimental_get_query_params()
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
        "Live network overview",
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

    with tab_map["Live network overview"]:
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

                    geometry_toggle_help = (
                        "Switch between straight-line haversine chords and the stored route geometry "
                        "when plotting route lines."
                    )
                    geometry_toggle_key = "route_map_use_route_geometry"
                    default_geometry_value = st.session_state.get(
                        geometry_toggle_key, True
                    )
                    if hasattr(st, "toggle"):
                        use_route_geometry = st.toggle(
                            "Use actual route geometry",
                            value=default_geometry_value,
                            help=geometry_toggle_help,
                            key=geometry_toggle_key,
                            disabled=not show_routes,
                        )
                    else:
                        use_route_geometry = st.checkbox(
                            "Use actual route geometry",
                            value=default_geometry_value,
                            help=geometry_toggle_help,
                            key=geometry_toggle_key,
                            disabled=not show_routes,
                        )

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
                                    use_route_geometry=use_route_geometry,
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
                                    use_route_geometry=use_route_geometry,
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


    with tab_map["Quote builder"]:
        render_quote_builder(filtered_df, filtered_mapping, conn, st.session_state)

    with tab_map["Optimizer"]:
        render_optimizer(filtered_df)

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
