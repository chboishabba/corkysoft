"""Route maps tab rendering helpers."""
from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlite3 import Connection

from analytics.price_distribution import (
    ColumnMapping,
    available_heatmap_weightings,
    build_heatmap_source,
    prepare_metric_route_map_data,
    prepare_route_map_data,
)
from analytics.live_data import extract_route_path
from analytics.route_map_prep import (
    build_isochrone_polygons,
    explain_isochrone_unavailability,
)
from analytics.routes_map import fetch_job_route_rows
from analytics.routing_provider import get_routing_provider
from dashboard.components.lane_scope import apply_lane_status_scope
from dashboard.components.maps import _hex_to_rgb, build_route_map
from dashboard.map_provider import (
    _resolved_provider,
    plotly_map_layout,
)
from dashboard.state import _rerun_app

__all__ = ["render_route_maps_tab"]

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


def _normalise_provider_labels(
    provider_names: List[object],
    *,
    routing_provider_env: object,
) -> List[str]:
    def _clean_provider_label(raw_label: object) -> str:
        raw = str(raw_label or "").strip()
        if not raw:
            return ""
        if raw.startswith("<class '") and raw.endswith("'>"):
            raw = raw[len("<class '") : -2]
        raw = raw.strip()
        return raw

    labels: List[str] = []
    fallback = str(routing_provider_env or "").strip().lower()

    for provider_name in provider_names:
        raw = _clean_provider_label(provider_name)
        if not raw:
            continue
        normalised = raw.lower()
        if normalised in {"google", "google maps", "google_maps"}:
            label = "Google Maps"
        elif normalised in {
            "ors",
            "openrouteservice",
            "open route service",
            "openrouteserviceprovider",
        }:
            label = "OpenRouteService"
        elif normalised in {"str", "type", "object", "unknown"}:
            if fallback == "google":
                label = "Google Maps"
            elif fallback == "ors":
                label = "OpenRouteService"
            else:
                continue
        else:
            label = raw
        if label not in labels:
            labels.append(label)

    if not labels:
        if fallback == "google":
            return ["Google Maps"]
        if fallback == "ors":
            return ["OpenRouteService"]
    return labels


# -----------------------------------------------------------------------------
# Compatibility shim for metro-distance filtering across branches/modules
# -----------------------------------------------------------------------------
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

except Exception:  # pragma: no cover - fallbacks exercised in integration
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

        def _filter_by_distance(
            df: pd.DataFrame,
            *,
            metro_only: bool = False,
            max_distance_km: float = 100.0,
        ) -> pd.DataFrame:
            return df



def render_route_maps_tab(
    filtered_df: pd.DataFrame,
    mapping: ColumnMapping,
    conn: Connection,
    dataset_key: str,
    metro_distance_km: float,
    *,
    show_title: bool = True,
    forced_mode: str | None = None,
    network_host: bool = False,
) -> None:
    """Render the Route maps tab contents."""
    effective_provider = _resolved_provider()
    os.environ["ROUTING_PROVIDER"] = effective_provider

    if show_title and not network_host:
        st.markdown("### Corridor visualisation")
    scoped_input_df = apply_lane_status_scope(
        filtered_df,
        scope_key="route_maps_lane_assignment_scope",
        label="Lane assignment scope",
        help_text=(
            "Route maps default to canonically assigned lane history. "
            "Include ambiguous or unassigned rows only when exploring unresolved records."
        ),
        caption_prefix="Route-map dataset rows after lane-status filter",
    )

    map_mode = forced_mode
    if map_mode is None:
        map_mode = st.radio(
            "Visualisation mode",
            ("Routes/points", "Heatmap", "Isochrones"),
            horizontal=True,
            help=(
                "Switch between individual routes/points, an aggregate density heatmap, "
                "or travel-time isochrones around each corridor."
            ),
            key="dashboard_route_map_mode",
        )
    metro_only = st.checkbox(
        "Limit to metro jobs (≤100 km)",
        value=False,
        help="Apply a distance filter using distance_km ≤ 100 to focus on metro corridors.",
        key="dashboard_route_map_metro_only",
    )

    scoped_df = _filter_by_distance(
        scoped_input_df, metro_only=metro_only, max_distance_km=metro_distance_km
    )
    map_df = scoped_df.copy()

    date_column: Optional[str] = None
    date_series: Optional[pd.Series] = None
    if not map_df.empty:
        candidate_columns: List[str] = []
        if "job_date" in map_df.columns:
            candidate_columns.append("job_date")
        if mapping.date and mapping.date in map_df.columns:
            candidate_columns.append(mapping.date)

        for candidate in candidate_columns:
            parsed = pd.to_datetime(map_df[candidate], errors="coerce")
            if parsed.notna().any():
                date_column = candidate
                date_series = parsed
                break

    if date_column and date_series is not None:
        map_df[date_column] = date_series
        valid_dates = date_series.dropna()
        if not valid_dates.empty:
            earliest = valid_dates.min().date()
            latest = valid_dates.max().date()
            date_mode = st.radio(
                "Route date selection",
                ("All dates", "Single day", "Date range"),
                horizontal=True,
                key="route_map_date_mode",
            )
            if date_mode == "Single day":
                selected_day = st.date_input(
                    "Select day",
                    value=latest,
                    min_value=earliest,
                    max_value=latest,
                    key="route_map_date_single",
                )
                mask = date_series.dt.date == selected_day
                map_df = map_df.loc[mask].copy()
                date_series = date_series.loc[mask]
            elif date_mode == "Date range":
                start_default = earliest
                end_default = latest
                selected_range = st.date_input(
                    "Select date range",
                    value=(start_default, end_default),
                    min_value=earliest,
                    max_value=latest,
                    key="route_map_date_range",
                )
                if isinstance(selected_range, tuple) and len(selected_range) == 2:
                    start_date = selected_range[0] or start_default
                    end_date = selected_range[1] or end_default
                else:
                    start_date, end_date = start_default, end_default
                mask = (date_series.dt.date >= start_date) & (date_series.dt.date <= end_date)
                map_df = map_df.loc[mask].copy()
                date_series = date_series.loc[mask]
            else:
                st.caption(
                    f"Displaying routes from {earliest.isoformat()} to {latest.isoformat()}."
                )

    if map_mode == "Routes/points":
        _render_route_lines_tab(map_df, dataset_key, conn)
    elif map_mode == "Heatmap":
        _render_heatmap_tab(scoped_input_df, map_df, provider=effective_provider)
    else:
        _render_isochrone_tab(map_df)


def _render_route_lines_tab(map_df: pd.DataFrame, dataset_key: str, conn: Connection) -> None:
    colour_mode_label = st.radio(
        "Colour data by",
        ("Categorical attribute", "Metric"),
        horizontal=True,
        help=(
            "Switch between discrete attributes and continuous metrics to colour the route and point layers."
        ),
        key="dashboard_route_colour_mode",
    )
    show_routes = st.checkbox(
        "Show route lines",
        value=True,
        key="dashboard_show_route_lines",
    )
    show_points = st.checkbox(
        "Show origin/destination points",
        value=True,
        key="dashboard_show_route_points",
    )
    include_saved_jobs = False
    if not network_host:
        include_saved_jobs = st.checkbox(
            "Include saved jobs overlay",
            value=False,
            key="dashboard_route_include_saved_jobs",
            help=(
                "Merge persisted job routes from the jobs table into this map. "
                "Jobs already present in the current selection are not duplicated."
            ),
        )

    geometry_toggle_help = (
        "Switch between straight-line haversine chords and the stored route geometry "
        "when plotting route lines."
    )
    geometry_toggle_key = "dashboard_route_use_route_geometry"
    default_geometry_value = st.session_state.get(geometry_toggle_key, True)
    if hasattr(st, "toggle"):
        use_route_geometry = st.toggle(
            "Use actual route geometry",
            value=bool(default_geometry_value),
            help=geometry_toggle_help,
            key=geometry_toggle_key,
            disabled=not show_routes,
        )
    else:
        use_route_geometry = st.checkbox(
            "Use actual route geometry",
            value=bool(default_geometry_value),
            help=geometry_toggle_help,
            key=geometry_toggle_key,
            disabled=not show_routes,
        )

    working_df = map_df.copy()
    if include_saved_jobs:
        working_df = _merge_saved_job_overlay_rows(working_df, conn)

    required_columns = {"origin_lat", "origin_lon", "dest_lat", "dest_lon"}
    missing_coordinates = required_columns - set(working_df.columns)
    if missing_coordinates:
        st.info("Add geocoded origin and destination coordinates to visualise routes.")
        return

    geocoded = working_df.dropna(subset=list(required_columns))
    if geocoded.empty:
        st.info("No routes with coordinates are available for the current filters.")
        return

    if use_route_geometry and show_routes:
        geometry_series = geocoded.get("route_geojson")
        if geometry_series is None:
            st.caption("Stored route geometry is not available for this dataset yet.")
        else:
            missing_mask = ~geometry_series.apply(_has_resolved_geometry)
            missing_count = int(missing_mask.sum())
            if missing_count > 0:
                st.info(
                    f"{missing_count} route{'s' if missing_count != 1 else ''} are missing stored geometry."
                )
                st.caption(
                    "Route geometry is enriched automatically during ingest/load paths. "
                    "Any remaining gaps indicate rows that could not be resolved by the configured routing provider."
                )
            else:
                st.caption("All displayed routes already have stored geometry.")

    if not show_routes and not show_points:
        st.info("Enable at least one layer to view the route map.")
        return

    if colour_mode_label == "Categorical attribute":
        _render_categorical_route_map(
            geocoded,
            show_routes,
            show_points,
            use_route_geometry,
            provider=effective_provider,
        )
    else:
        _render_metric_route_map(
            geocoded,
            show_routes,
            show_points,
            use_route_geometry,
            provider=effective_provider,
        )


def _has_geometry(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bool(value)
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _has_resolved_geometry(value: Any) -> bool:
    if not _has_geometry(value):
        return False
    try:
        return len(extract_route_path(str(value))) > 2
    except Exception:
        return False


def _render_categorical_route_map(
    geocoded: pd.DataFrame,
    show_routes: bool,
    show_points: bool,
    use_route_geometry: bool,
    *,
    provider: str,
) -> None:
    if use_route_geometry:
        geocoded = _enrich_route_geometry(geocoded)

    colour_dimensions = {
        "Data source": "route_dataset_source",
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
        st.info("No categorical columns available to colour the route map.")
        return

    colour_label = st.selectbox(
        "Categorical attribute",
        options=list(available_colour_dimensions.keys()),
        help=("Choose which attribute drives the route and point colouring."),
        key="dashboard_route_colour_dimension",
    )
    selected_column = available_colour_dimensions[colour_label]
    try:
        plotly_map_df = prepare_route_map_data(geocoded, selected_column)
    except KeyError as exc:
        st.warning(str(exc))
        plotly_map_df = pd.DataFrame()

    if plotly_map_df.empty:
        st.info("No routes with coordinates are available for the current filters.")
        return

    route_map = build_route_map(
        plotly_map_df,
        colour_label,
        show_routes=show_routes,
        show_points=show_points,
        use_route_geometry=use_route_geometry,
        provider=provider,
    )
    st.plotly_chart(route_map, width="stretch")


def _render_metric_route_map(
    geocoded: pd.DataFrame,
    show_routes: bool,
    show_points: bool,
    use_route_geometry: bool,
    *,
    provider: str,
) -> None:
    if use_route_geometry:
        geocoded = _enrich_route_geometry(geocoded)

    metric_colour_options: Dict[str, Dict[str, object]] = {
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
        "Cost vs Price (%)": {
            "column": "cost_vs_price_pct",
            "format": "percentage",
            "scale": px.colors.diverging.RdBu,
            "tickformat": ".0%",
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

    available_metric_options: Dict[str, Dict[str, object]] = {}
    for label, spec in metric_colour_options.items():
        column = spec["column"]
        if column not in geocoded.columns:
            continue
        numeric_series = pd.to_numeric(geocoded[column], errors="coerce")
        numeric_series = numeric_series.replace([math.inf, -math.inf], pd.NA)
        if numeric_series.notna().any():
            available_metric_options[label] = spec

    if not available_metric_options:
        st.info("No numeric metrics are available to colour the route map.")
        return

    metric_label = st.selectbox(
        "Metric",
        options=list(available_metric_options.keys()),
        help=("Select a metric to drive the continuous colour scale."),
        key="dashboard_route_metric_dimension",
    )
    metric_spec = available_metric_options[metric_label]
    metric_column = metric_spec["column"]
    format_spec = metric_spec.get("format", "number")
    try:
        metric_map_df = prepare_metric_route_map_data(
            geocoded,
            metric_column,
            format_spec=str(format_spec),
        )
    except KeyError as exc:
        st.warning(str(exc))
        metric_map_df = pd.DataFrame()

    if metric_map_df.empty:
        st.info("No routes with the selected metric are available for the current filters.")
        return

    route_map = build_route_map(
        metric_map_df,
        metric_label,
        show_routes=show_routes,
        show_points=show_points,
        colour_mode="continuous",
        colour_scale=metric_spec.get("scale"),
        colorbar_tickformat=metric_spec.get("tickformat"),
        use_route_geometry=use_route_geometry,
        provider=provider,
    )
    st.plotly_chart(route_map, width="stretch")


def _render_heatmap_tab(
    filtered_df: pd.DataFrame,
    map_df: pd.DataFrame,
    *,
    provider: str,
) -> None:
    weight_options = available_heatmap_weightings(filtered_df)
    weight_label = st.selectbox(
        "Heatmap weighting",
        options=list(weight_options.keys()),
        help="Choose which metric influences the heatmap intensity.",
        key="dashboard_heatmap_weighting",
    )
    weight_column = weight_options[weight_label]

    if map_df.empty:
        st.info("No jobs match the metro filter for the current selection.")
        return

    try:
        heatmap_source = build_heatmap_source(
            map_df,
            weight_column=weight_column,
        )
    except KeyError as exc:
        st.warning(str(exc))
        heatmap_source = pd.DataFrame(columns=["lat", "lon", "weight"])

    if heatmap_source.empty:
        st.info("No geocoded points are available for the current filters.")
        return

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
    heatmap_fig = px.density_map(
        heatmap_source,
        lat="lat",
        lon="lon",
        z="weight",
        radius=45,
        opacity=0.8,
        color_continuous_scale=colour_scales.get(weight_column, px.colors.sequential.YlOrRd),
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
    hover_template = hover_templates.get(weight_column, f"{weight_label}: %{{z:.2f}}<extra></extra>")
    for trace in heatmap_fig.data:
        trace.hovertemplate = hover_template

    heatmap_fig.update_layout(
        **plotly_map_layout(
            centre,
            zoom=4,
            engine="map",
            provider=provider,
        ),
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        coloraxis_colorbar={"title": weight_label},
    )
    st.plotly_chart(heatmap_fig, width="stretch")


def _render_isochrone_tab(map_df: pd.DataFrame) -> None:
    effective_provider = _resolved_provider()
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
            "Network-aware reachability horizon used when the configured routing stack can produce true isochrone polygons."
        ),
    )
    if hasattr(st, "toggle"):
        allow_approximate = st.toggle(
            "Allow synthetic reach circles",
            value=False,
            help=(
                "When provider-backed isochrones are unavailable, estimate reach using approximate circles "
                "based on route distance and speed."
            ),
        )
    else:
        allow_approximate = st.checkbox(
            "Allow synthetic reach circles",
            value=False,
            help=(
                "When provider-backed isochrones are unavailable, estimate reach using approximate circles "
                "based on route distance and speed."
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
        map_df,
        centre="origin" if centre_label == "Origin" else "destination",
        horizon_hours=float(iso_hours),
        max_routes=int(max_iso_routes),
        routing_provider=effective_provider,
        allow_approximate_fallback=allow_approximate,
    )

    if iso_source.empty:
        diagnostics = explain_isochrone_unavailability(
            map_df,
            centre="origin" if centre_label == "Origin" else "destination",
            horizon_hours=float(iso_hours),
            routing_provider=effective_provider,
        )
        st.info(
            "No network-aware isochrones are available for the current filters and routing configuration."
        )
        if diagnostics["reasons"]:
            st.caption("Why this can happen:")
            for reason in diagnostics["reasons"]:
                st.caption(f"- {reason}")
        if not allow_approximate:
            st.caption(
                "Synthetic circular reach estimates are currently disabled. Enable "
                "\"Allow synthetic reach circles\" to include fallback circles when provider "
                "polygons are not available."
            )
        if diagnostics["next_actions"]:
            st.caption("Next steps:")
            for action in diagnostics["next_actions"]:
                st.caption(f"- {action}")
        provider_labels = _normalise_provider_labels(
            diagnostics["provider_names"],
            routing_provider_env=diagnostics.get("routing_provider_env"),
        )
        if provider_labels:
            st.caption(
                "Attempted providers: " + ", ".join(provider_labels)
            )
        st.caption(
            f"Candidate routes with valid centre/distance: {diagnostics['candidate_rows']} / {diagnostics['input_rows']}"
        )
        return

    figure = go.Figure()
    palette = _ISOCHRONE_PALETTE or ["#636EFA"]

    for idx, (_, row) in enumerate(iso_source.iterrows()):
        colour_hex = palette[idx % len(palette)]
        r, g, b = _hex_to_rgb(colour_hex)
        fill_colour = f"rgba({r},{g},{b},0.18)"
        line_colour = f"rgba({r},{g},{b},0.9)"

        figure.add_trace(
            go.Scattermap(
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
            go.Scattermap(
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
        **plotly_map_layout(
            {"lat": centre_lat, "lon": centre_lon},
            zoom=4,
            engine="map",
        ),
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        legend={"orientation": "h", "yanchor": "bottom", "y": 0.01},
    )
    st.plotly_chart(figure, width="stretch")


def _merge_saved_job_overlay_rows(map_df: pd.DataFrame, conn: Connection) -> pd.DataFrame:
    def _coerce_float(value: Any) -> Optional[float]:
        try:
            return round(float(value), 5)
        except (TypeError, ValueError):
            return None

    def _route_signature(row: pd.Series) -> tuple[Any, ...]:
        geojson_value = row.get("route_geojson")
        geojson_signature = geojson_value.strip() if isinstance(geojson_value, str) else None
        return (
            _coerce_float(row.get("origin_lat")),
            _coerce_float(row.get("origin_lon")),
            _coerce_float(row.get("dest_lat")),
            _coerce_float(row.get("dest_lon")),
            geojson_signature,
        )

    base_df = map_df.copy()
    base_df["route_dataset_source"] = "Current selection"

    try:
        job_rows = fetch_job_route_rows(conn, include_actual=True)
    except Exception as exc:
        st.warning(f"Unable to load saved jobs overlay: {exc}")
        return base_df

    if not job_rows:
        st.caption("Saved jobs overlay is empty.")
        return base_df

    saved_df = pd.DataFrame(
        [dict(row) if hasattr(row, "keys") else dict(row) for row in job_rows]
    )
    if saved_df.empty:
        st.caption("Saved jobs overlay is empty.")
        return base_df

    saved_df["route_dataset_source"] = "Saved jobs"
    if "origin_resolved" in saved_df.columns and "origin_city" not in saved_df.columns:
        saved_df["origin_city"] = saved_df["origin_resolved"]
    if "destination_resolved" in saved_df.columns and "destination_city" not in saved_df.columns:
        saved_df["destination_city"] = saved_df["destination_resolved"]
    if "client_display" not in saved_df.columns:
        saved_df["client_display"] = "Saved jobs"

    duplicate_count = 0
    if "id" in base_df.columns and "id" in saved_df.columns:
        base_ids = pd.to_numeric(base_df["id"], errors="coerce").dropna()
        if not base_ids.empty:
            saved_ids = pd.to_numeric(saved_df["id"], errors="coerce")
            duplicate_mask = saved_ids.isin(set(base_ids.astype(int).tolist()))
            duplicate_count = int(duplicate_mask.sum())
            saved_df = saved_df.loc[~duplicate_mask].copy()

    if not saved_df.empty:
        base_signatures = {
            _route_signature(row)
            for _, row in base_df.iterrows()
        }
        signature_mask = saved_df.apply(
            lambda row: _route_signature(row) in base_signatures,
            axis=1,
        )
        signature_duplicates = int(signature_mask.sum())
        if signature_duplicates > 0:
            duplicate_count += signature_duplicates
            saved_df = saved_df.loc[~signature_mask].copy()

    if saved_df.empty:
        if duplicate_count > 0:
            st.caption("Saved jobs overlay is already covered by the current selection.")
        return base_df

    merged_df = pd.concat([base_df, saved_df], ignore_index=True, sort=False)
    if duplicate_count > 0:
        st.caption(
            f"Added {len(saved_df)} saved job route rows and skipped {duplicate_count} duplicates already in the current selection."
        )
    else:
        st.caption(f"Added {len(saved_df)} saved job route rows from the jobs table.")
    return merged_df


def _enrich_route_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing ``route_geojson`` values using the configured routing provider."""

    if df.empty or "route_geojson" in df.columns and df["route_geojson"].apply(_has_resolved_geometry).all():
        return df

    working = df.copy()
    provider_key = _resolved_provider()
    route_provider = get_routing_provider(provider=provider_key, client=None)

    def _coerce_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    missing_mask = ~working["route_geojson"].apply(_has_resolved_geometry) if "route_geojson" in working.columns else pd.Series(True, index=working.index)
    for idx, row in working.loc[missing_mask].iterrows():
        origin_lon = _coerce_float(row.get("origin_lon"))
        origin_lat = _coerce_float(row.get("origin_lat"))
        dest_lon = _coerce_float(row.get("dest_lon"))
        dest_lat = _coerce_float(row.get("dest_lat"))
        if None in (origin_lon, origin_lat, dest_lon, dest_lat):
            continue
        try:
            geometry = route_provider.route_geometry(
                origin=(origin_lon, origin_lat),
                destination=(dest_lon, dest_lat),
            )
            working.loc[idx, "route_geojson"] = geometry.dumps()
        except Exception:
            continue

    return working
