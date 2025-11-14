"""Streamlit dashboard for the price distribution analysis."""
from __future__ import annotations

import inspect
import io
import json
import math
import os
import sqlite3
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

if __package__ is None or __package__ == "":  # pragma: no cover - script execution support
    _MODULE_DIR = Path(__file__).resolve().parent
    _PROJECT_ROOT = _MODULE_DIR.parent
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

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
    build_price_history_series,
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
    compute_cost_vs_price_percentage,
    prepare_metric_route_map_data,
    prepare_route_map_data,
    prepare_profitability_map_data,
    prepare_profitability_route_data,
    summarise_distribution,
    summarise_last_year_distributions,
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
from analytics.routes_map import (
    build_job_route_map,
    fetch_job_route_rows,
    populate_route_geometry,
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
from dashboard.components.maps import (
    _geojson_to_path,
    _hex_to_rgb,
    _initial_view_state,
    build_route_map,
    render_network_map,
)
from dashboard.map_provider import (
    google_maps_api_key,
    plotly_map_layout,
    pydeck_map_kwargs,
)


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

PRICE_DASHBOARD_TABS = [
    "Histogram",
    "Price history",
    "Profitability insights",
    "Live network overview",
    "Route maps",
    "Quote builder",
    "Optimizer",
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
    if callable(rerun):
        rerun()
        return

    experimental_rerun = getattr(st, "experimental_rerun", None)
    if callable(experimental_rerun):
        experimental_rerun()
        return

    raise RuntimeError("Streamlit rerun API is unavailable.")


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

def render_price_distribution_dashboard():
    st.title("Price distribution (Airbnb-style)")
    st.caption(
        "Visualise $ per m³ by corridor and client, with break-even bands to spot loss-leaders."
    )

    tabs_placeholder = st.container()

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
                key="dashboard_sidebar_init_db",
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
                key="dashboard_dataset_selector",
            )
            dataset_key, dataset_loader = dataset_options[dataset_label]

            provider_options = {
                "OpenRouteService": "ors",
                "Google Maps": "google",
            }
            provider_labels = list(provider_options.keys())
            current_provider_env = os.environ.get("ROUTING_PROVIDER", "ors").strip().lower()
            default_provider_label = next(
                (label for label, value in provider_options.items() if value == current_provider_env),
                provider_labels[0],
            )
            provider_choice_label = st.radio(
                "Routing provider",
                options=provider_labels,
                index=provider_labels.index(default_provider_label),
                key="dashboard_routing_provider_selector",
                help=(
                    "Select which routing provider to use for map tiles and route geometry."
                ),
            )
            resolved_provider = provider_options[provider_choice_label]
            if resolved_provider != current_provider_env:
                os.environ["ROUTING_PROVIDER"] = resolved_provider
                _rerun_app()

            if resolved_provider == "google" and not google_maps_api_key():
                st.warning(
                    "Google Maps selected but GOOGLE_MAPS_API_KEY is not configured."
                )

            import_feedback: Optional[tuple[str, str]] = None
            if dataset_key == "historical":
                with st.expander("Import historical jobs from CSV", expanded=False):
                    import_form = st.form(key="dashboard_sidebar_historical_import_form")
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
                    key="date_range_active",
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
                    key="date_range_disabled",
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
                key="client_filter_multiselect",
            )

            postcode_prefix = st.text_input(
                "Corridor contains postcode prefix",
                value=postcode_prefix or "",
                disabled=not data_available,
                help="Match origin or destination postcode prefixes (e.g. 40 to match 4000-4099).",
                key="postcode_prefix_filter",
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
                key="break_even_input",
            )
            if st.button("Update break-even", key="break_even_update_button"):
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

        tab_labels = PRICE_DASHBOARD_TABS
        params = _get_query_params()
        requested_tab = params.get("view", [tab_labels[0]])[0]
        if requested_tab not in tab_labels:
            requested_tab = tab_labels[0]
        requested_tab_index = tab_labels.index(requested_tab)

        can_assign_tab_key = False
        try:
            can_assign_tab_key = "key" in inspect.signature(st.tabs).parameters
        except (TypeError, ValueError):
            can_assign_tab_key = False

        tab_order = tab_labels
        if can_assign_tab_key:
            tabs_key = "dashboard_active_tab"
            view_param_requested = "view" in params
            if tabs_key not in st.session_state or (
                view_param_requested
                and st.session_state.get(tabs_key) != requested_tab_index
            ):
                st.session_state[tabs_key] = requested_tab_index

            with tabs_placeholder:
                streamlit_tabs = st.tabs(tab_order, key=tabs_key)
        else:
            if requested_tab_index != 0:
                tab_order = [
                    requested_tab,
                    *[
                        label
                        for idx, label in enumerate(tab_labels)
                        if idx != requested_tab_index
                    ],
                ]
            with tabs_placeholder:
                streamlit_tabs = st.tabs(tab_order)

        tab_map: Dict[str, Any] = {
            label: tab for label, tab in zip(tab_order, streamlit_tabs)
        }

        summary: Optional[DistributionSummary] = None
        profitability_summary: Optional[ProfitabilitySummary] = None
        metro_summary: Optional[DistributionSummary] = None
        metro_profitability: Optional[ProfitabilitySummary] = None
        metro_distance_km = 100.0
        if has_filtered_data:
            filtered_df = filtered_df.copy()
            filtered_df["cost_vs_price_pct"] = compute_cost_vs_price_percentage(filtered_df)

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
                toggle_key="dashboard_network_map_toggle_overview",
            )

        with tab_map["Histogram"]:
            if has_filtered_data:
                with st.popover("❓ Histogram stats", use_container_width=True):
                    st.markdown(
                        """
                        **Break-even bands**  
                        Vertical markers highlight how far each band sits from the selected break-even $/m³. They help you quickly spot how much headroom or shortfall each cluster of jobs has.

                        **Normal fit overlay**  
                        The dark curve shows a normal distribution fitted to the selected jobs. It makes it easy to compare the real-world distribution with an ideal bell curve.

                        **Summary statistics**  
                        • **μ (mean)** — the average $/m³ across the visible jobs.  
                        • **σ (standard deviation)** — the typical spread around the mean.  
                        • **Kurtosis** — how heavy the tails are compared with a normal distribution (positive values mean more extreme outliers).
                        """
                    )
                histogram = create_histogram(filtered_df, break_even_value)
                st.plotly_chart(histogram, width="stretch")
                st.caption(
                    "Histogram overlays include the normal distribution fit plus kurtosis and dispersion markers for context."
                )
            elif dataset_error:
                st.error("Unable to load jobs — initialise the database and retry.")
            else:
                st.info("Import historical jobs to plot the price distribution histogram.")

        with tab_map["Price history"]:
            date_column = filtered_mapping.date or "job_date"
            if not has_filtered_data:
                st.info("Apply filters or import jobs to analyse price history trends.")
            elif date_column not in filtered_df.columns:
                st.warning("No date column available to build the price history view.")
            else:
                st.markdown("### Price history")
                frequency_options = {"Daily": "daily", "Weekly": "weekly", "Monthly": "monthly"}
                selected_frequency_label = st.radio(
                    "Aggregation frequency",
                    list(frequency_options.keys()),
                    horizontal=True,
                    key="price_history_frequency",
                )
                frequency_value = frequency_options[selected_frequency_label]

                dates = pd.to_datetime(filtered_df[date_column], errors="coerce")
                valid_dates = dates.dropna()
                default_start = (
                    start_date
                    or (valid_dates.min().date() if not valid_dates.empty else date.today())
                )
                default_end = (
                    end_date
                    or (valid_dates.max().date() if not valid_dates.empty else date.today())
                )

                history_range_kwargs: dict[str, Any] = {"key": "price_history_range"}
                if not valid_dates.empty:
                    history_range_kwargs["min_value"] = valid_dates.min().date()
                    history_range_kwargs["max_value"] = valid_dates.max().date()

                history_range = st.date_input(
                    "History range",
                    value=(default_start, default_end),
                    **history_range_kwargs,
                )

                if isinstance(history_range, tuple) and len(history_range) == 2:
                    history_start, history_end = history_range
                else:
                    history_start, history_end = default_start, default_end

                if history_start > history_end:
                    st.warning("Select a start date that precedes the end date to render the charts.")
                else:
                    history_series = build_price_history_series(
                        filtered_df,
                        frequency=frequency_value,
                        date_column=date_column,
                        start_date=history_start,
                        end_date=history_end,
                    )
                    previous_year_frames = summarise_last_year_distributions(
                        filtered_df,
                        date_column=date_column,
                        start_date=history_start,
                        end_date=history_end,
                    )

                    metric_labels = {
                        "Price $/m³": "price_per_m3",
                        "Margin $/m³": "margin_per_m3",
                        "Margin %": "margin_total_pct",
                    }

                    def _with_display_period(frame: pd.DataFrame, *, year_offset: int = 0) -> pd.DataFrame:
                        if frame.empty or "period" not in frame.columns:
                            return frame
                        result = frame.copy()
                        result["display_period"] = pd.to_datetime(result["period"], errors="coerce")
                        if year_offset:
                            result["display_period"] = result["display_period"] + pd.DateOffset(years=year_offset)
                        return result

                    current_overall = _with_display_period(
                        history_series.current_overall.assign(series="Current period")
                    )
                    previous_overall = _with_display_period(
                        history_series.previous_year_overall.assign(series="Previous year"),
                        year_offset=1,
                    )
                    combined_overall = pd.concat(
                        [
                            frame
                            for frame in (current_overall, previous_overall)
                            if not frame.empty
                        ],
                        ignore_index=True,
                    )

                    if combined_overall.empty:
                        st.info("Not enough data to build an aggregated price history chart.")
                    else:
                        metric_columns = [
                            column
                            for column in metric_labels.values()
                            if column in combined_overall.columns
                        ]
                        if not metric_columns:
                            st.info(
                                "Time series metrics are missing for the selected range. Add price or margin columns to compare trends."
                            )
                        else:
                            melted = combined_overall.melt(
                                id_vars=["display_period", "series"],
                                value_vars=metric_columns,
                                var_name="metric",
                                value_name="value",
                            ).dropna(subset=["value", "display_period"])
                            if melted.empty:
                                st.info(
                                    "Time series metrics are missing for the selected range. Add price or margin columns to compare trends."
                                )
                            else:
                                overall_fig = px.line(
                                    melted,
                                    x="display_period",
                                    y="value",
                                    color="metric",
                                    line_dash="series",
                                    markers=True,
                                    labels={
                                        "display_period": "Period",
                                        "value": "Value",
                                        "metric": "Metric",
                                        "series": "Series",
                                    },
                                )
                                overall_fig.update_layout(legend_title_text="Metric comparison")
                                st.plotly_chart(overall_fig, width="stretch")

                    available_metric_labels = [
                        label
                        for label, column in metric_labels.items()
                        if column in history_series.current_overall.columns
                        or column in history_series.current_by_origin.columns
                        or column in history_series.current_by_destination.columns
                        or column in history_series.previous_year_overall.columns
                        or column in history_series.previous_year_by_origin.columns
                        or column in history_series.previous_year_by_destination.columns
                    ]

                    metric_label = None
                    metric_column = None
                    if available_metric_labels:
                        metric_label = st.selectbox(
                            "Breakdown metric",
                            options=available_metric_labels,
                            key="price_history_metric",
                            help="Choose which metric drives the origin and destination breakdown charts.",
                        )
                        metric_column = metric_labels[metric_label]
                    else:
                        st.info(
                            "Add price or margin columns to explore origin and destination breakdowns."
                        )

                    metric_display_label = metric_label or "Value"

                    if metric_column:
                        current_origin = _with_display_period(
                            history_series.current_by_origin.assign(series="Current period")
                        )
                        previous_origin = _with_display_period(
                            history_series.previous_year_by_origin.assign(series="Previous year"),
                            year_offset=1,
                        )
                        origin_combined = pd.concat(
                            [
                                frame
                                for frame in (current_origin, previous_origin)
                                if not frame.empty
                            ],
                            ignore_index=True,
                        )

                        current_destination = _with_display_period(
                            history_series.current_by_destination.assign(series="Current period")
                        )
                        previous_destination = _with_display_period(
                            history_series.previous_year_by_destination.assign(series="Previous year"),
                            year_offset=1,
                        )
                        destination_combined = pd.concat(
                            [
                                frame
                                for frame in (current_destination, previous_destination)
                                if not frame.empty
                            ],
                            ignore_index=True,
                        )

                        origin_col, destination_col = st.columns(2)
                        if (
                            origin_combined.empty
                            or metric_column not in origin_combined.columns
                            or origin_combined.dropna(subset=[metric_column]).empty
                        ):
                            origin_col.info("No origin-level data available for the selected metric.")
                        else:
                            origin_fig = px.line(
                                origin_combined.dropna(subset=[metric_column, "display_period"]),
                                x="display_period",
                                y=metric_column,
                                color="origin",
                                line_dash="series",
                                markers=True,
                                labels={
                                    "display_period": "Period",
                                    "origin": "Origin",
                                    metric_column: metric_display_label,
                                    "series": "Series",
                                },
                            )
                            origin_fig.update_layout(legend_title_text="Origin")
                            origin_col.plotly_chart(origin_fig, width="stretch")

                        if (
                            destination_combined.empty
                            or metric_column not in destination_combined.columns
                            or destination_combined.dropna(subset=[metric_column]).empty
                        ):
                            destination_col.info(
                                "No destination-level data available for the selected metric."
                            )
                        else:
                            destination_fig = px.line(
                                destination_combined.dropna(
                                    subset=[metric_column, "display_period"]
                                ),
                                x="display_period",
                                y=metric_column,
                                color="destination",
                                line_dash="series",
                                markers=True,
                                labels={
                                    "display_period": "Period",
                                    "destination": "Destination",
                                    metric_column: metric_display_label,
                                    "series": "Series",
                                },
                            )
                            destination_fig.update_layout(legend_title_text="Destination")
                            destination_col.plotly_chart(destination_fig, width="stretch")

                    st.markdown("#### Previous year distribution snapshots")
                    histogram_metric = None
                    previous_overall_frame = previous_year_frames.get("overall", pd.DataFrame())
                    if not previous_overall_frame.empty:
                        if "price_per_m3" in previous_overall_frame.columns:
                            histogram_metric = "price_per_m3"
                        elif metric_column and metric_column in previous_overall_frame.columns:
                            histogram_metric = metric_column
                    if histogram_metric:
                        hist_source = previous_overall_frame.dropna(subset=[histogram_metric])
                        if hist_source.empty:
                            st.info(
                                "No numeric values available to plot the previous year histogram."
                            )
                        else:
                            hist_fig = px.histogram(
                                hist_source,
                                x=histogram_metric,
                                nbins=25,
                                color_discrete_sequence=[px.colors.qualitative.Plotly[0]],
                                labels={
                                    histogram_metric: (
                                        metric_display_label
                                        if histogram_metric == metric_column
                                        else "Price $/m³"
                                    )
                                },
                                title="Previous year distribution",
                            )
                            st.plotly_chart(hist_fig, width="stretch")
                    else:
                        st.info(
                            "Historical dataset from the previous year is unavailable for distribution comparisons."
                        )

                    if metric_column:
                        comparison_columns = st.columns(2)
                        previous_origin = previous_year_frames.get("by_origin", pd.DataFrame())
                        if (
                            previous_origin.empty
                            or metric_column not in previous_origin.columns
                            or previous_origin.dropna(subset=[metric_column]).empty
                        ):
                            comparison_columns[0].info(
                                "No previous-year origin data to summarise as a box plot."
                            )
                        else:
                            origin_box = px.box(
                                previous_origin.dropna(subset=[metric_column]),
                                x="origin",
                                y=metric_column,
                                points="outliers",
                                labels={
                                    "origin": "Origin",
                                    metric_column: metric_display_label,
                                },
                                title="Origin spread (previous year)",
                            )
                            comparison_columns[0].plotly_chart(origin_box, width="stretch")

                        previous_destination = previous_year_frames.get("by_destination", pd.DataFrame())
                        if (
                            previous_destination.empty
                            or metric_column not in previous_destination.columns
                            or previous_destination.dropna(subset=[metric_column]).empty
                        ):
                            comparison_columns[1].info(
                                "No previous-year destination data to summarise as a box plot."
                            )
                        else:
                            destination_box = px.box(
                                previous_destination.dropna(subset=[metric_column]),
                                x="destination",
                                y=metric_column,
                                points="outliers",
                                labels={
                                    "destination": "Destination",
                                    metric_column: metric_display_label,
                                },
                                title="Destination spread (previous year)",
                            )
                            comparison_columns[1].plotly_chart(
                                destination_box, width="stretch"
                            )

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
                    key="dashboard_profitability_view",
                )
                fig = view_options[selected_view](filtered_df)
                st.plotly_chart(fig, width="stretch")

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
                key="dashboard_route_map_mode",
            )
            metro_only = st.checkbox(
                "Limit to metro jobs (≤100 km)",
                value=False,
                help="Apply a distance filter using distance_km ≤ 100 to focus on metro corridors.",
                key="dashboard_route_map_metro_only",
            )

            scoped_df = _filter_by_distance(
                filtered_df, metro_only=metro_only, max_distance_km=metro_distance_km
            )
            map_df = scoped_df.copy()

            date_column: Optional[str] = None
            date_series: Optional[pd.Series] = None
            if not map_df.empty:
                candidate_columns = []
                if "job_date" in map_df.columns:
                    candidate_columns.append("job_date")
                if filtered_mapping.date and filtered_mapping.date in map_df.columns:
                    candidate_columns.append(filtered_mapping.date)

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
                required_columns = {"origin_lat", "origin_lon", "dest_lat", "dest_lon"}
                missing_coordinates = required_columns - set(map_df.columns)

                if map_df.empty:
                    st.info("No jobs match the metro filter for the current selection.")
                elif missing_coordinates:
                    st.info(
                        "Add geocoded origin and destination coordinates to visualise routes."
                    )
                else:
                    geocoded = map_df.dropna(subset=list(required_columns))
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

                        geometry_toggle_help = (
                            "Switch between straight-line haversine chords and the stored route geometry "
                            "when plotting route lines."
                        )
                        geometry_toggle_key = "dashboard_route_use_route_geometry"
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

                        missing_route_ids: List[int] = []
                        if use_route_geometry and show_routes:
                            geometry_series = geocoded.get("route_geojson")
                            if geometry_series is None:
                                st.caption(
                                    "Stored route geometry is not available for this dataset yet."
                                )
                            else:
                                def _has_geometry(value: Any) -> bool:
                                    if value is None:
                                        return False
                                    if isinstance(value, (bytes, bytearray, memoryview)):
                                        return bool(value)
                                    if isinstance(value, str):
                                        return bool(value.strip())
                                    return True

                                missing_mask = ~geometry_series.apply(_has_geometry)
                                missing_count = int(missing_mask.sum())
                                if missing_count > 0:
                                    st.info(
                                        f"{missing_count} route{'s' if missing_count != 1 else ''} "
                                        "are missing stored geometry."
                                    )
                                    if "id" in geocoded.columns:
                                        route_ids = geocoded.loc[missing_mask, "id"].dropna()
                                        missing_route_ids = [int(value) for value in route_ids.tolist()]
                                    else:
                                        st.caption(
                                            "Add an 'id' column to populate geometry for the selected routes."
                                        )

                                    if missing_route_ids and dataset_key in {"historical", "live"}:
                                        if st.button(
                                            "Populate route geometry",
                                            key="populate_route_geometry_button",
                                            help=(
                                                "Fetch routing-provider geometry for the filtered routes and store it "
                                                "for future map sessions."
                                            ),
                                        ):
                                            try:
                                                populated = populate_route_geometry(
                                                    conn,
                                                    missing_route_ids,
                                                    dataset=dataset_key,
                                                )
                                            except Exception as exc:  # pragma: no cover - streamlit feedback only
                                                st.error(f"Failed to populate geometry: {exc}")
                                            else:
                                                if populated:
                                                    st.success(
                                                        f"Stored geometry for {populated} route"
                                                        f"{'s' if populated != 1 else ''}."
                                                    )
                                                else:
                                                    st.warning(
                                                        "No route geometry could be retrieved for the current selection."
                                                    )
                                                _rerun_app()
                                    elif missing_route_ids:
                                        st.caption(
                                            "Populate the historical or live job tables to store route geometry."
                                        )
                                else:
                                    st.caption(
                                        "All displayed routes already have stored geometry."
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
                                    key="dashboard_route_colour_dimension",
                                )
                                selected_column = available_colour_dimensions[colour_label]
                                try:
                                    plotly_map_df = prepare_route_map_data(
                                        map_df, selected_column
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
                                    st.plotly_chart(route_map, width="stretch")
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
                                    key="dashboard_route_metric_dimension",
                                )
                                metric_spec = available_metric_options[metric_label]
                                metric_column = metric_spec["column"]
                                format_spec = metric_spec.get("format", "number")
                                try:
                                    metric_map_df = prepare_metric_route_map_data(
                                        map_df,
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
                                    st.plotly_chart(route_map, width="stretch")
            elif map_mode == "Heatmap":
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
                else:
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
                        heatmap_fig = px.density_map(
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
                            **plotly_map_layout(
                                centre,
                                zoom=4,
                                engine="map",
                            ),
                            margin={"l": 0, "r": 0, "t": 0, "b": 0},
                            coloraxis_colorbar={"title": weight_label},
                        )
                        st.plotly_chart(heatmap_fig, width="stretch")
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
                    map_df,
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

            st.divider()
            st.markdown("#### Saved job routes (Folium)")
            if folium is None or st_folium is None:
                st.info(
                    "Install 'folium' and 'streamlit-folium' to view the interactive route overlay."
                )
            else:
                try:
                    job_rows = fetch_job_route_rows(conn, include_actual=True)
                except sqlite3.OperationalError as exc:
                    st.warning(f"Unable to load saved routes from the jobs table: {exc}")
                else:
                    if not job_rows:
                        st.caption("No saved jobs with coordinates are available yet.")
                    else:
                        actual_available = any(
                            "route_geojson" in row.keys() and row["route_geojson"] for row in job_rows
                        )
                        toggle_help = (
                            "Overlay the stored routing-provider geometry instead of straight-line chords."
                        )
                        include_actual_key = "folium_job_routes_overlay"
                        default_toggle = (
                            bool(st.session_state.get(include_actual_key, actual_available))
                            if actual_available
                            else False
                        )

                        if hasattr(st, "toggle"):
                            include_actual = st.toggle(
                                "Overlay actual routed paths",
                                value=default_toggle,
                                key=include_actual_key,
                                help=toggle_help,
                                disabled=not actual_available,
                            )
                        else:
                            include_actual = st.checkbox(
                                "Overlay actual routed paths",
                                value=default_toggle,
                                key=include_actual_key,
                                help=toggle_help,
                                disabled=not actual_available,
                            )

                        if not actual_available and include_actual:
                            include_actual = False
                        if not actual_available:
                            st.caption(
                                "Stored route geometry has not been captured yet; showing straight-line connections instead."
                            )

                        folium_map = build_job_route_map(job_rows, include_actual=include_actual)
                        st_folium(
                            folium_map,
                            height=520,
                            key="folium_saved_job_routes",
                            returned_objects=[],
                        )
                        st.caption(
                            "Use the layer control to toggle marker, straight-line, and actual route overlays."
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
                        deck_kwargs = {
                            "initial_view_state": pdk.ViewState(
                                latitude=midpoint_lat,
                                longitude=midpoint_lon,
                                zoom=5,
                                pitch=30,
                            ),
                            "layers": [
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
                        }
                        deck_kwargs.update(
                            pydeck_map_kwargs("mapbox://styles/mapbox/light-v9")
                        )
                        deck = pdk.Deck(**deck_kwargs)
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
                if _QUOTE_COUNTRY_STATE_KEY not in st.session_state:
                    st.session_state[_QUOTE_COUNTRY_STATE_KEY] = (
                        default_country or COUNTRY_DEFAULT
                    )
                country_value = st.text_input(
                    "Country",
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
                    key="quote_builder_modifiers_multiselect_inline",
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
                        st.plotly_chart(chart, width="stretch")

                        st.dataframe(recommendations_df, width="stretch")

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
        filtered_display_df = filtered_df
        if "job_date" in filtered_df.columns:
            parsed_dates = pd.to_datetime(filtered_df["job_date"], errors="coerce")
            filtered_display_df = (
                filtered_df.assign(_job_sort_key=parsed_dates)
                .sort_values("_job_sort_key", ascending=False, na_position="last")
                .drop(columns="_job_sort_key")
            )

        st.dataframe(filtered_display_df[display_columns + remaining_columns])

        csv_buffer = io.StringIO()
        filtered_display_df.to_csv(csv_buffer, index=False)
        st.download_button(
            "Export filtered rows",
            csv_buffer.getvalue(),
            file_name="price_distribution_filtered.csv",
            mime="text/csv",
        )


def main() -> None:
    """Configure Streamlit and render the price distribution dashboard."""

    st.set_page_config(
        page_title="Price distribution by corridor",
        layout="wide",
    )
    render_price_distribution_dashboard()


if __name__ == "__main__":  # pragma: no cover - manual execution convenience
    main()
