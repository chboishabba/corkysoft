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
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import plotly.express as px
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

from analytics.dashboard_layouts import (
    ROLE_LAYOUT_DEFAULTS,
    get_dashboard_role_layouts,
    missing_recommended_primary_tabs,
    resolve_dashboard_layout,
    upsert_dashboard_role_layout,
)
from analytics.db import (
    ABSENCE_RECORD_STATUSES,
    ABSENCE_RECORD_TYPES,
    INVENTORY_ARCHITECTURES,
    INVENTORY_CUSTODY_TYPES,
    INVENTORY_EXECUTION_STAGES,
    INVENTORY_STATES,
    INVENTORY_SUBSTITUTION_APPROVER_ROLES,
    INVENTORY_SUBSTITUTION_STATUSES,
    allocate_inventory_to_segment,
    decide_inventory_substitution,
    ensure_dashboard_tables,
    get_allowed_inventory_execution_stages,
    import_inventory_items_from_dataframe,
    import_inventory_movements_from_dataframe,
    import_suppliers_from_google_sheet,
    import_workers_from_google_sheet,
    import_workers_from_staff_sheet,
    list_inventory,
    list_inventory_balances,
    list_inventory_execution_events,
    list_inventory_exceptions,
    list_inventory_movements,
    list_inventory_requirements,
    list_inventory_substitution_reason_codes,
    list_segment_inventory_coordination,
    list_inventory_substitutions,
    record_inventory_execution_event,
    record_inventory_movement,
    create_worker_absence_record,
    request_inventory_substitution,
    resolve_inventory_exception,
    upsert_inventory_substitution_reason_code,
    upsert_inventory_requirement,
    ensure_dashboard_tables,
    import_workers_from_staff_sheet,
    upsert_worker,
)
from analytics.db_connection import connection_scope
from analytics.driver_shifts import (
    DEFAULT_DRIVER_SHEET_NAME,
    import_driver_shifts_from_sheet,
    load_driver_shifts_dataframe,
)
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
    summarise_profitability,
    update_break_even,
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
from analytics.kent_ams_import import (
    get_kent_tender_policy_config,
    list_kent_override_reason_codes,
    list_kent_tender_override_history,
    list_prioritized_tenders,
    record_kent_tender_override,
    update_kent_tender_policy_config,
    upsert_kent_override_reason_code,
)
from analytics.labor_analytics import (
    OVERTIME_DAILY_HOURS_DEFAULT,
    build_payroll_labor_analytics,
)
from analytics.operations_assignment import (
    assign_worker_compliance,
    assign_worker_role,
    ensure_worker_compliance,
    ensure_worker_role,
    list_labor_reconciliation,
    list_operational_readiness_items,
    list_planned_labor_assignments,
    list_segments_for_worker,
    list_worker_assignment_summary,
)
from dashboard.components.operations import render_operations_tab
from dashboard.components.dispatch import render_dispatch_tab
from dashboard.components.planner import render_planner_tab
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
    _initial_view_state,
    render_network_map,
)
from dashboard.components.maintenance import render_fleet_tab, render_vehicle_maintenance_tab
from dashboard.components.route_maps import render_route_maps_tab
from dashboard.components.calls import render_calls_tab
from dashboard.components.price_history import render_price_history_tab
from dashboard.components.optimizer import render_optimizer
from dashboard.map_provider import (
    folium_map_configuration,
    google_maps_api_key,
    plotly_map_layout,
    pydeck_map_kwargs,
)
from corkysoft.call_ops import (
    decide_worker_time_capture_event,
    list_worker_time_capture_events,
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
PRICE_DASHBOARD_TABS = [
    "Histogram",
    "Price history",
    "Profitability insights",
    "Live network overview",
    "Route maps",
    "Dispatch",
    "Planner",
    "Operations",
    "Fleet",
    "Vehicle maintenance",
    "Quote builder",
    "Calls",
    "Kent tenders",
    "Kent admin",
    "Optimizer",
    "Inventory",
    "Staff",
    "Driver shifts",
    "Payroll / Labor analytics",
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
        map_kwargs, tile_layer_kwargs = folium_map_configuration()
        map_obj = folium.Map(
            location=[current_lat, current_lon],
            zoom_start=zoom,
            **map_kwargs,
        )
        if tile_layer_kwargs:
            folium.TileLayer(**tile_layer_kwargs).add_to(map_obj)
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
        ensure_dashboard_tables(conn)

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
        role_layouts = get_dashboard_role_layouts(conn, available_tabs=tab_labels)
        role_labels = {item["label"]: item for item in role_layouts}
        default_role_label = next((item["label"] for item in role_layouts if item["roleKey"] == "dispatcher"), role_layouts[0]["label"])
        layout_cols = st.columns([2, 2, 2, 1])
        selected_role_label = layout_cols[0].selectbox(
            "Role layout",
            options=list(role_labels.keys()),
            index=list(role_labels.keys()).index(st.session_state.get("dashboard_active_role", default_role_label))
            if st.session_state.get("dashboard_active_role", default_role_label) in role_labels
            else 0,
            key="dashboard_active_role",
        )
        selected_role_layout = role_labels[selected_role_label]
        stale_primary_tabs = missing_recommended_primary_tabs(
            role_key=str(selected_role_layout["roleKey"]),
            layout=selected_role_layout,
            available_tabs=tab_labels,
        )
        session_primary_tabs = layout_cols[1].multiselect(
            "Session focus tabs",
            options=tab_labels,
            default=st.session_state.get("dashboard_session_primary_tabs", selected_role_layout["primaryTabs"]),
            key="dashboard_session_primary_tabs",
        )
        session_show_all = layout_cols[2].checkbox(
            "Show all tabs this session",
            value=bool(st.session_state.get("dashboard_show_all_tabs", False)),
            key="dashboard_show_all_tabs",
        )
        if layout_cols[3].button("Reset layout", key="dashboard_reset_role_layout"):
            st.session_state["dashboard_session_primary_tabs"] = list(selected_role_layout["primaryTabs"])
            st.session_state["dashboard_session_hidden_tabs"] = list(selected_role_layout["hiddenTabs"])
            st.session_state["dashboard_session_landing_tab"] = selected_role_layout["defaultLandingTab"]
            st.session_state["dashboard_show_all_tabs"] = False
            _rerun_app()

        if selected_role_layout["roleKey"] == "dispatcher" and stale_primary_tabs:
            st.warning(
                "Dispatcher layout is missing recommended focus tabs: "
                + ", ".join(stale_primary_tabs)
                + "."
            )
            if st.button("Repair dispatcher layout", key="dashboard_repair_dispatcher_layout"):
                repaired = upsert_dashboard_role_layout(
                    conn,
                    role_key="dispatcher",
                    default_landing_tab=str(ROLE_LAYOUT_DEFAULTS["dispatcher"]["defaultLandingTab"]),
                    primary_tabs=list(ROLE_LAYOUT_DEFAULTS["dispatcher"]["primaryTabs"]),
                    hidden_tabs=list(ROLE_LAYOUT_DEFAULTS["dispatcher"]["hiddenTabs"]),
                    available_tabs=tab_labels,
                )
                st.session_state["dashboard_session_primary_tabs"] = list(repaired["primaryTabs"])
                st.session_state["dashboard_session_hidden_tabs"] = list(repaired["hiddenTabs"])
                st.session_state["dashboard_session_landing_tab"] = repaired["defaultLandingTab"]
                st.session_state["dashboard_show_all_tabs"] = False
                _rerun_app()

        params = _get_query_params()
        requested_tab = params.get("view", [tab_labels[0]])[0]
        if requested_tab not in tab_labels:
            requested_tab = tab_labels[0]
        resolved_layout = resolve_dashboard_layout(
            available_tabs=tab_labels,
            layout=selected_role_layout,
            requested_tab=requested_tab if "view" in params else None,
            session_primary_tabs=st.session_state.get("dashboard_session_primary_tabs", selected_role_layout["primaryTabs"]),
            session_hidden_tabs=st.session_state.get("dashboard_session_hidden_tabs", selected_role_layout["hiddenTabs"]),
            session_landing_tab=st.session_state.get("dashboard_session_landing_tab", selected_role_layout["defaultLandingTab"]),
            show_all_tabs=bool(st.session_state.get("dashboard_show_all_tabs", False)),
        )
        tab_labels = resolved_layout["tabOrder"]
        requested_tab = resolved_layout["landingTab"]
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

        if "Live network overview" in tab_map:
            with tab_map["Live network overview"]:
                render_network_map(
                    map_routes,
                    truck_positions,
                    active_routes,
                    toggle_key="dashboard_network_map_toggle_overview",
                )

        if "Histogram" in tab_map:
            with tab_map["Histogram"]:
                if has_filtered_data:
                    with st.popover("❓ Histogram stats", width='stretch'):
                        st.markdown(
                            """
                            ### **Break-even bands**
                            Vertical guide-lines centred on your **break-even $/m³**.
    
                            Each band shows your break-even target (**$/m³ needed to make profit**), along with percentages indicating how far real jobs fall above or below it.
    
                            You can quickly see:
    
                            - **Which corridors frequently underperform**
                            - **Whether a client consistently prices below your minimum**
                            - **How much “safety margin” you have on metro jobs**
                            - The **normal-fit overlay**, showing a bell curve fitted to your $/m³ distribution
                            - Real-world job pricing is messy and often skewed — the normal fit gives an *idealised baseline*.
                            - These are methods derived from the formal study of statistics and can be used to inform operators and managers about pricing trends.
    
                            ---
    
                            ### **Reading the curve**
                            - **Skew** → are there lots of cheap jobs or lots of expensive jobs?
                            - **Fat tails** → outliers on either side
                            - **Pricing stability** → is your pricing consistent or chaotic?
    
                            **Tall, narrow curve** → stable, predictable pricing
                            **Wide, flat curve** → highly variable pricing
    
                            ---
    
                            ### **Summary statistics**
                            These quantify the shape and behaviour of your pricing:
    
                            - **Percentiles** — e.g., 75th percentile means a value is higher than 75% of jobs.
                            - **Mean (μ)** — your overall *average* revenue density.
                            - **Median** — midpoint of all jobs.
                            More stable than the mean when outliers exist.
                            - **Standard deviation (σ)** — measures volatility.
                            **High σ** = inconsistent pricing; **Low σ** = tightly clustered pricing.
                            - **Kurtosis** — how “outlier-heavy” your distribution is.
                            Over > 3 = fat tails - some data is very unlike others; Under < 3 = tighter, more predictable.
                            - **Skewness** — asymmetry.
                            **Positive skew** → many cheap jobs, few expensive ones.
                            **Negative skew** → many expensive jobs, few cheap ones.
                            - **% below break-even** — proportion of unprofitable jobs.
                            **Ideal:** 0–10% **Warning:** 20–30% **Critical:** >30%
                            """,
                            unsafe_allow_html=True,
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

        if "Price history" in tab_map:
            with tab_map["Price history"]:
                render_price_history_tab(
                    filtered_df=filtered_df,
                    mapping=filtered_mapping,
                    start_date=start_date,
                    end_date=end_date,
                )

        if "Profitability insights" in tab_map:
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

        if "Route maps" in tab_map:
            with tab_map["Route maps"]:
                render_route_maps_tab(
                    filtered_df=filtered_df,
                    mapping=filtered_mapping,
                    conn=conn,
                    dataset_key=dataset_key,
                    metro_distance_km=metro_distance_km,
                )

        if "Dispatch" in tab_map:
            with tab_map["Dispatch"]:
                render_dispatch_tab(conn)

        if "Planner" in tab_map:
            with tab_map["Planner"]:
                render_planner_tab(filtered_df=filtered_df, conn=conn)

        if "Operations" in tab_map:
            with tab_map["Operations"]:
                render_operations_tab(conn)

        if "Fleet" in tab_map:
            with tab_map["Fleet"]:
                render_fleet_tab(conn)

        if "Vehicle maintenance" in tab_map:
            with tab_map["Vehicle maintenance"]:
                render_vehicle_maintenance_tab(conn)

        if "Quote builder" in tab_map:
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
                    if quote_result.profit_rule_mode:
                        policy_line = (
                            f"**Quote policy preview:** `{quote_result.profit_rule_mode}` | "
                            f"pass={bool(quote_result.policy_matched)} | "
                            f"thresholds ${float(quote_result.absolute_margin_threshold or 0.0):,.0f} / "
                            f"{float(quote_result.margin_percent_threshold or 0.0):.1f}%"
                        )
                        st.markdown(policy_line)
                    if quote_result.policy_fail_reasons:
                        st.warning(
                            "Profitability policy fail reasons: "
                            + ", ".join(quote_result.policy_fail_reasons)
                        )
                    if quote_result.loss_alert:
                        st.error("Loss alert: expected margin is below the configured floor.")
    
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

        if "Calls" in tab_map:
            with tab_map["Calls"]:
                render_calls_tab(conn)

        if "Kent tenders" in tab_map:
            with tab_map["Kent tenders"]:
                render_kent_tenders_tab(conn)

        if "Kent admin" in tab_map:
            with tab_map["Kent admin"]:
                render_kent_admin_tab(conn)

        if "Optimizer" in tab_map:
            with tab_map["Optimizer"]:

                with st.popover("❓ How to use the Optimiser", width='stretch'):
                    st.markdown(
                    """
                    ### **How it works**
    
                    The optimizer reviews your **historical jobs** and tells you where pricing can be safely increased.
    
                    Here’s the workflow:
    
                    1. **Apply filters (left sidebar)**
                    Choose the clients, corridors, job sizes, and date ranges you want the optimizer to analyse.
                    Only jobs that match these filters are used in the calculation.
    
                    2. **Set your pricing rules**
                    - **Margin buffer** → how much extra profit you want to build in
                    - **Max uplift cap** → the highest % increase you’re comfortable recommending
                    - **Minimum corridor volume** → ignore corridors with too few historic jobs
                    These guardrails control how aggressive or conservative the recommendations will be.
    
                    3. **Run the optimizer**
                    It evaluates each corridor by comparing its **historic median $/m³** to the **$/m³ needed to hit your target margin**.
                    If a corridor is underpriced, it suggests a recommended uplift *within the limits you set*.
    
                    ---
    
                    ### **What the recommendations mean**
    
                    For each corridor, the optimizer tells you:
    
                    - **How far the historic pricing is below your target margin**
                    - **A suggested uplift %** that brings it closer to your profitability goal
                    - **Whether the uplift is capped** (because it hit your max allowance)
                    - **Whether the corridor was skipped** (not enough data)
    
                    These suggestions are based entirely on **your real historic jobs**, so they reflect how you have actually priced and performed.
    
                    ---
    
                    ### **Exporting the results**
    
                    Use the **Download CSV** button to export a corridor-by-corridor report.
                    This makes it easy to:
    
                    - Share uplift recommendations with sales or pricing teams
                    - Review which corridors are consistently underperforming
                    - Apply controlled, data-backed price adjustments
    
                    """,unsafe_allow_html=True,
                    )
    
                render_optimizer(filtered_df)

        if "Inventory" in tab_map:
            with tab_map["Inventory"]:
                render_inventory_tab(conn)
        if "Staff" in tab_map:
            with tab_map["Staff"]:
                render_staff_tab(conn)

        if "Driver shifts" in tab_map:
            with tab_map["Driver shifts"]:
                render_driver_shifts_tab(conn)
        if "Payroll / Labor analytics" in tab_map:
            with tab_map["Payroll / Labor analytics"]:
                render_payroll_labor_analytics_tab(conn)

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

        st.dataframe(filtered_display_df[display_columns + remaining_columns], width='content')

        csv_buffer = io.StringIO()
        filtered_display_df.to_csv(csv_buffer, index=False)
        st.download_button(
            "Export filtered rows",
            csv_buffer.getvalue(),
            file_name="price_distribution_filtered.csv",
            mime="text/csv",
        )


def _read_uploaded_inventory_file(uploaded_file: Any | None) -> pd.DataFrame:
    """Parse a CSV or Excel upload into a dataframe."""

    if uploaded_file is None:
        return pd.DataFrame()

    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if filename.endswith(('.xls', '.xlsx')):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type. Please upload CSV or Excel.")


def render_kent_tenders_tab(conn: sqlite3.Connection) -> None:
    st.subheader("Kent AMS tender queue")
    st.caption(
        "Profitability rule mode drives priority, not hard blocking. Only safety/legal/compliance flags should hard-block."
    )

    policy = get_kent_tender_policy_config(conn)
    queue_cols = st.columns([1, 1, 1, 1])
    status_filter = queue_cols[0].selectbox(
        "Status",
        options=["open", "awarded", "closed", "all"],
        index=0,
        key="kent_tender_status_filter",
    )
    limit_value = int(
        queue_cols[1].number_input(
            "Rows",
            min_value=5,
            max_value=250,
            value=25,
            step=5,
            key="kent_tender_limit",
        )
    )
    operator_id = queue_cols[2].text_input(
        "Operator ID",
        value=st.session_state.get("kent_tender_operator_id", ""),
        key="kent_tender_operator_id",
    )
    queue_cols[3].metric("Rule mode", policy["ruleMode"])

    rows = list_prioritized_tenders(conn, status=status_filter, limit=limit_value)
    if not rows:
        st.info("No Kent tenders found for the selected filter.")
        return

    reason_options = {
        row["code"]: row["label"]
        for row in list_kent_override_reason_codes(conn)
        if row["active"]
    }
    if not reason_options:
        st.warning(
            "No active override reasons are configured. Operators can review the queue, but overrides are disabled until an admin activates at least one reason."
        )

    summary_rows = [
        {
            "Tender": row["tenderExternalId"],
            "Job": row["jobNumber"],
            "Client": row["clientName"],
            "Origin": row["origin"],
            "Destination": row["destination"],
            "Action": row["recommendedAction"],
            "Policy": "PASS" if row["policyMatched"] else "FAIL",
            "Margin": row["estimatedMargin"],
            "Margin %": row["estimatedMarginPct"],
            "Score": row["scoreTotal"],
            "Loss": "ALERT" if row["lossAlert"] else "",
            "Freshness": row["freshnessState"],
        }
        for row in rows
    ]
    st.dataframe(pd.DataFrame(summary_rows), width='stretch', hide_index=True)

    for row in rows:
        badge_parts = []
        if row["hardBlockFlags"]:
            badge_parts.append("HARD BLOCK")
        if row["lossAlert"]:
            badge_parts.append("LOSS ALERT")
        if not row["policyMatched"]:
            badge_parts.append("POLICY FAIL")
        header = " | ".join(
            part for part in [row["tenderExternalId"], row["jobNumber"], ", ".join(badge_parts)] if part
        )
        with st.expander(header or row["tenderExternalId"], expanded=False):
            detail_cols = st.columns(4)
            detail_cols[0].metric("Expected revenue", format_currency(row["expectedRevenue"] or 0.0))
            detail_cols[1].metric("Est. margin", format_currency(row["estimatedMargin"] or 0.0))
            detail_cols[2].metric(
                "Est. margin %",
                "n/a" if row["estimatedMarginPct"] is None else f"{row['estimatedMarginPct']:.1f}%",
            )
            detail_cols[3].metric("Priority score", f"{row['scoreTotal']:.1f}")
            st.caption(
                f"Rule mode `{row['profitRuleMode']}` | thresholds: ${row['absoluteMarginThreshold']:,.0f} and {row['marginPercentThreshold']:.1f}% | freshness `{row['freshnessState']}` ({row['confidenceScore']:.1f})"
            )
            if row["policyFailReasons"]:
                st.warning("Policy fail reasons: " + ", ".join(row["policyFailReasons"]))
            if row["overrideableFlags"]:
                st.info("Overrideable flags: " + ", ".join(row["overrideableFlags"]))
            if row["hardBlockFlags"]:
                st.error("Hard-block flags: " + ", ".join(row["hardBlockFlags"]))

            with st.form(f"kent_override_form_{row['tenderExternalId']}"):
                action = st.selectbox(
                    "Action",
                    options=["pursue", "review", "defer", "award_override"],
                    key=f"kent_override_action_{row['tenderExternalId']}",
                )
                reason_code = st.selectbox(
                    "Reason code",
                    options=list(reason_options.keys()) or ["<no-active-reasons>"],
                    format_func=lambda code: reason_options.get(code, code),
                    key=f"kent_override_reason_{row['tenderExternalId']}",
                )
                note = st.text_area(
                    "Optional note",
                    key=f"kent_override_note_{row['tenderExternalId']}",
                    height=80,
                )
                submit_disabled = (
                    bool(row["hardBlockFlags"])
                    or not operator_id.strip()
                    or not reason_options
                )
                if st.form_submit_button("Record override", disabled=submit_disabled):
                    try:
                        record_kent_tender_override(
                            conn,
                            tender_external_id=row["tenderExternalId"],
                            action=action,
                            operator_id=operator_id.strip(),
                            reason_code=reason_code,
                            note=note,
                        )
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        st.success("Override recorded.")
                        st.rerun()
                if not operator_id.strip():
                    st.caption("Enter an operator ID to record override actions.")


def render_kent_admin_tab(conn: sqlite3.Connection) -> None:
    st.subheader("Kent AMS admin")
    st.caption(
        "Use this surface for policy defaults, override reason governance, and review. Operators should work from the Kent tenders tab."
    )

    policy = get_kent_tender_policy_config(conn)
    with st.form("kent_tender_policy_form"):
        config_cols = st.columns(4)
        rule_mode = config_cols[0].selectbox(
            "Rule mode",
            options=["ABS_ONLY", "PCT_ONLY", "EITHER", "BOTH"],
            index=["ABS_ONLY", "PCT_ONLY", "EITHER", "BOTH"].index(policy["ruleMode"]),
        )
        abs_threshold = config_cols[1].number_input(
            "Abs margin threshold",
            value=float(policy["absoluteMarginThreshold"]),
            step=100.0,
        )
        pct_threshold = config_cols[2].number_input(
            "Margin % threshold",
            value=float(policy["marginPercentThreshold"]),
            step=1.0,
        )
        loss_floor = config_cols[3].number_input(
            "Loss alert floor",
            value=float(policy["lossAlertFloor"]),
            step=100.0,
        )
        if st.form_submit_button("Save policy defaults"):
            try:
                update_kent_tender_policy_config(
                    conn,
                    rule_mode=rule_mode,
                    absolute_margin_threshold=float(abs_threshold),
                    margin_percent_threshold=float(pct_threshold),
                    loss_alert_floor=float(loss_floor),
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.success("Kent tender policy defaults updated.")
                st.rerun()

    reasons = list_kent_override_reason_codes(conn)
    if reasons:
        st.dataframe(pd.DataFrame(reasons), width='stretch', hide_index=True)

    with st.form("kent_override_reason_form"):
        reason_cols = st.columns(4)
        new_code = reason_cols[0].text_input("Code")
        new_label = reason_cols[1].text_input("Label")
        new_description = reason_cols[2].text_input("Description")
        new_active = reason_cols[3].checkbox("Active", value=True)
        if st.form_submit_button("Save reason"):
            try:
                upsert_kent_override_reason_code(
                    conn,
                    code=new_code,
                    label=new_label,
                    description=new_description,
                    active=new_active,
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.success("Override reason saved.")
                st.rerun()

            history = list_kent_tender_override_history(
                conn, tender_external_id=row["tenderExternalId"]
            )
            if history:
                history_df = pd.DataFrame(
                    [
                        {
                            "At": item["createdAt"],
                            "Action": item["action"],
                            "Operator": item["operatorId"],
                            "Reason": item["reasonCode"],
                            "Note": item["note"],
                            "Policy matched": item["policyMatched"],
                            "Loss alert": item["lossAlert"],
                        }
                        for item in history
                    ]
                )
                st.dataframe(history_df, width='stretch', hide_index=True)


def render_inventory_tab(conn: sqlite3.Connection) -> None:
    st.subheader("Inventory and movements")
    st.caption(
        "Execution stages are constrained warehouse actions layered above the lower-level logistics states."
    )

    segment_coordination = list_segment_inventory_coordination(conn)
    st.markdown("#### Segment-linked inventory coordination")
    if segment_coordination:
        coordination_df = pd.DataFrame(segment_coordination)
        coordination_df["inventoryNames"] = coordination_df["inventoryNames"].apply(
            lambda values: ", ".join(values) if isinstance(values, list) else ""
        )
        coordination_df["supplierNames"] = coordination_df["supplierNames"].apply(
            lambda values: ", ".join(values) if isinstance(values, list) else ""
        )
        coordination_df["architectures"] = coordination_df["architectures"].apply(
            lambda values: ", ".join(values) if isinstance(values, list) else ""
        )
        coordination_df["requirementNames"] = coordination_df["requirementNames"].apply(
            lambda values: ", ".join(values) if isinstance(values, list) else ""
        )
        st.dataframe(
            coordination_df[
                [
                    "jobId",
                    "segmentSequence",
                    "fromLocation",
                    "toLocation",
                    "plannedStart",
                    "assignmentStatus",
                    "requirementCount",
                    "requiredQuantity",
                    "shipmentCount",
                    "allocatedQuantity",
                    "shortageQuantity",
                    "inventoryNames",
                    "requirementNames",
                    "architectures",
                    "supplierNames",
                ]
            ].rename(
                columns={
                    "jobId": "Job",
                    "segmentSequence": "Segment",
                    "fromLocation": "From",
                    "toLocation": "To",
                    "plannedStart": "Planned start",
                    "assignmentStatus": "Status",
                    "requirementCount": "Requirements",
                    "requiredQuantity": "Required qty",
                    "shipmentCount": "Shipments",
                    "allocatedQuantity": "Allocated qty",
                    "shortageQuantity": "Shortage qty",
                    "inventoryNames": "Inventory",
                    "requirementNames": "Requirement lines",
                    "architectures": "Architectures",
                    "supplierNames": "Suppliers",
                }
            ),
            width='stretch',
            hide_index=True,
        )
    else:
        st.caption("No segment-linked inventory allocations recorded yet.")

    state_filter = st.multiselect(
        "Filter by state",
        INVENTORY_STATES,
        default=list(INVENTORY_STATES),
        help="States are derived from movement events and item imports.",
    )

    job_filter_raw = st.text_input(
        "Filter by job (numeric id)",
        value="",
        help="Leave blank to show all jobs.",
    )
    job_filter: int | None = None
    if job_filter_raw.strip():
        try:
            job_filter = int(job_filter_raw)
        except ValueError:
            st.warning("Job filter must be a number if provided.")

    requirements = list_inventory_requirements(conn, job_id=job_filter)
    st.markdown("#### Requirement planning")
    if requirements:
        requirements_df = pd.DataFrame(requirements)
        st.dataframe(
            requirements_df[
                [
                    "jobId",
                    "segmentSequence",
                    "requirementName",
                    "inventoryName",
                    "architecture",
                    "requiredQuantity",
                    "allocatedQuantity",
                    "approvedSubstitutionQuantity",
                    "requestedSubstitutionQuantity",
                    "effectiveFulfilledQuantity",
                    "shortageQuantity",
                    "substitutionAllowed",
                    "hasPendingSubstitution",
                    "executionStage",
                    "executionActor",
                    "unit",
                    "notes",
                ]
            ].rename(
                columns={
                    "jobId": "Job",
                    "segmentSequence": "Segment",
                    "requirementName": "Requirement",
                    "inventoryName": "Inventory item",
                    "architecture": "Architecture",
                    "requiredQuantity": "Required qty",
                    "allocatedQuantity": "Allocated qty",
                    "approvedSubstitutionQuantity": "Approved substitution qty",
                    "requestedSubstitutionQuantity": "Requested substitution qty",
                    "effectiveFulfilledQuantity": "Effective fulfilled qty",
                    "shortageQuantity": "Shortage qty",
                    "substitutionAllowed": "Substitutable",
                    "hasPendingSubstitution": "Pending substitution",
                    "executionStage": "Execution stage",
                    "executionActor": "Last actor",
                    "unit": "Unit",
                    "notes": "Notes",
                }
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.caption("No requirement lines recorded yet. Add segment requirements to surface shortages before execution.")

    active_substitution_reasons = list_inventory_substitution_reason_codes(conn, active_only=True)

    with st.expander("Warehouse execution updates", expanded=False):
        if not requirements:
            st.caption("Add requirement lines before recording pick / pack / load activity.")
        else:
            requirement_options = {
                (
                    f"Job {row['jobId']} / Segment {row['segmentSequence']} / "
                    f"{row['requirementName']} ({row['requiredQuantity']} required)"
                ): row
                for row in requirements
            }
            event_label = st.selectbox(
                "Requirement line",
                options=list(requirement_options.keys()),
                key="inventory_execution_requirement",
            )
            selected_requirement = requirement_options[event_label]
            allowed_execution_stages = get_allowed_inventory_execution_stages(
                selected_requirement.get("executionStage"),
                architecture=str(selected_requirement.get("architecture") or "general"),
            )
            st.caption(
                f"Current stage: `{selected_requirement.get('executionStage') or 'required'}`"
            )
            if allowed_execution_stages:
                st.caption("Allowed next actions: " + ", ".join(allowed_execution_stages))
            else:
                st.caption("No further routine execution actions are available for this requirement.")
            execution_cols = st.columns(4)
            execution_stage = execution_cols[0].selectbox(
                "Next action",
                options=allowed_execution_stages or [selected_requirement.get("executionStage") or "required"],
                key="inventory_execution_stage",
                disabled=not allowed_execution_stages,
            )
            execution_quantity = execution_cols[1].number_input(
                "Quantity",
                min_value=0.1,
                value=float(
                    selected_requirement.get("shortageQuantity")
                    or selected_requirement.get("requiredQuantity")
                    or 1.0
                ),
                step=0.5,
                key="inventory_execution_quantity",
            )
            execution_actor = execution_cols[2].text_input(
                "Actor",
                value="",
                key="inventory_execution_actor",
            )
            execution_truck = execution_cols[3].text_input(
                "Truck (optional)",
                value=str(selected_requirement.get("executionTruckId") or ""),
                key="inventory_execution_truck",
            )
            execution_aux_cols = st.columns(4)
            execution_container = execution_aux_cols[0].text_input(
                "Container ref",
                value=str(selected_requirement.get("executionContainerRef") or ""),
                key="inventory_execution_container",
            )
            execution_location_type = execution_aux_cols[1].selectbox(
                "Location type",
                options=[""] + list(INVENTORY_CUSTODY_TYPES),
                key="inventory_execution_location_type",
            )
            execution_location_ref = execution_aux_cols[2].text_input(
                "Location ref",
                value="",
                key="inventory_execution_location_ref",
            )
            execution_location_label = execution_aux_cols[3].text_input(
                "Location label",
                value="",
                key="inventory_execution_location_label",
            )
            execution_note = st.text_input(
                "Note",
                value="",
                key="inventory_execution_note",
            )
            if st.button(
                "Record execution update",
                type="primary",
                key="inventory_execution_save",
                disabled=not allowed_execution_stages,
            ):
                try:
                    record_inventory_execution_event(
                        conn,
                        job_id=int(selected_requirement["jobId"]),
                        segment_id=int(selected_requirement["segmentId"]),
                        requirement_id=int(selected_requirement["requirementId"]),
                        inventory_item_id=selected_requirement.get("inventoryItemId"),
                        stage=execution_stage,
                        quantity=float(execution_quantity),
                        actor=execution_actor or None,
                        note=execution_note or None,
                        container_ref=execution_container or None,
                        truck_id=execution_truck or None,
                        location_type=execution_location_type or None,
                        location_ref=execution_location_ref or None,
                        location_label=execution_location_label or None,
                    )
                except Exception as exc:  # pragma: no cover
                    st.error(f"Failed to record execution update: {exc}")
                else:
                    st.success("Execution update recorded.")
                    _rerun_app()

    with st.expander("Substitutions", expanded=False):
        if not requirements:
            st.caption("Add requirement lines before requesting or approving substitutions.")
        else:
            requirement_options = {
                (
                    f"Job {row['jobId']} / Segment {row['segmentSequence']} / "
                    f"{row['requirementName']} ({row['shortageQuantity']} shortage)"
                ): row
                for row in requirements
            }
            inventory_items = list_inventory(conn)
            substitution_item_options = {
                "<no substitute item selected>": None,
                **{str(row["name"]): int(row["id"]) for row in inventory_items},
            }

            request_cols = st.columns(4)
            request_requirement_label = request_cols[0].selectbox(
                "Requirement for request",
                options=list(requirement_options.keys()),
                key="inventory_substitution_requirement",
            )
            selected_requirement = requirement_options[request_requirement_label]
            request_quantity = request_cols[1].number_input(
                "Requested quantity",
                min_value=0.1,
                value=max(float(selected_requirement.get("shortageQuantity") or 0.0), 0.1),
                step=0.5,
                key="inventory_substitution_quantity",
            )
            request_actor = request_cols[2].text_input(
                "Requested by",
                value="",
                key="inventory_substitution_requested_by",
            )
            request_reason = request_cols[3].selectbox(
                "Reason code",
                options=[row["code"] for row in active_substitution_reasons]
                if active_substitution_reasons
                else ["<no active reasons>"],
                format_func=lambda code: next(
                    (
                        f"{row['label']} ({row['code']})"
                        for row in active_substitution_reasons
                        if row["code"] == code
                    ),
                    code,
                ),
                key="inventory_substitution_reason_code",
                disabled=not active_substitution_reasons,
            )
            request_substitute_label = st.selectbox(
                "Proposed substitute item",
                options=list(substitution_item_options.keys()),
                key="inventory_substitution_item",
            )
            request_note = st.text_input(
                "Request note",
                value="",
                key="inventory_substitution_note",
            )
            if st.button(
                "Request substitution",
                type="primary",
                key="inventory_substitution_request_save",
                disabled=not active_substitution_reasons,
            ):
                try:
                    request_inventory_substitution(
                        conn,
                        requirement_id=int(selected_requirement["requirementId"]),
                        requested_quantity=float(request_quantity),
                        requested_by=request_actor or None,
                        reason_code=request_reason.strip(),
                        note=request_note or None,
                        substitute_inventory_item_id=substitution_item_options[request_substitute_label],
                    )
                except Exception as exc:  # pragma: no cover
                    st.error(f"Failed to request substitution: {exc}")
                else:
                    st.success("Substitution request recorded.")
                    _rerun_app()

            substitutions = list_inventory_substitutions(conn, job_id=job_filter)
            if substitutions:
                substitutions_df = pd.DataFrame(substitutions)
                st.dataframe(
                    substitutions_df[
                        [
                            "substitutionId",
                            "jobId",
                            "segmentId",
                            "requirementName",
                            "inventoryName",
                            "substituteInventoryName",
                            "requestedQuantity",
                            "approvedQuantity",
                            "status",
                            "requestedBy",
                            "approvedBy",
                            "approvedRole",
                            "reasonCode",
                            "note",
                            "createdAt",
                            "decidedAt",
                        ]
                    ].rename(
                        columns={
                            "substitutionId": "ID",
                            "jobId": "Job",
                            "segmentId": "Segment",
                            "requirementName": "Requirement",
                            "inventoryName": "Original item",
                            "substituteInventoryName": "Substitute item",
                            "requestedQuantity": "Requested qty",
                            "approvedQuantity": "Approved qty",
                            "status": "Status",
                            "requestedBy": "Requested by",
                            "approvedBy": "Approved by",
                            "approvedRole": "Approved role",
                            "reasonCode": "Reason code",
                            "note": "Note",
                            "createdAt": "Created",
                            "decidedAt": "Decided",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )
                pending = [row for row in substitutions if row["status"] == "requested"]
                if pending:
                    pending_options = {
                        (
                            f"#{row['substitutionId']} / Job {row['jobId']} / Segment {row['segmentId']} / "
                            f"{row['requirementName']}"
                        ): row
                        for row in pending
                    }
                    decision_cols = st.columns(5)
                    pending_label = decision_cols[0].selectbox(
                        "Pending request",
                        options=list(pending_options.keys()),
                        key="inventory_substitution_pending",
                    )
                    pending_row = pending_options[pending_label]
                    decision_status = decision_cols[1].selectbox(
                        "Decision",
                        options=["approved", "rejected"],
                        key="inventory_substitution_decision",
                    )
                    approved_quantity = decision_cols[2].number_input(
                        "Approved qty",
                        min_value=0.0,
                        value=float(pending_row.get("requestedQuantity") or 0.0),
                        step=0.5,
                        key="inventory_substitution_approved_qty",
                    )
                    approved_by = decision_cols[3].text_input(
                        "Approved by",
                        value="",
                        key="inventory_substitution_approved_by",
                    )
                    approved_role = decision_cols[4].selectbox(
                        "Approval role",
                        options=list(INVENTORY_SUBSTITUTION_APPROVER_ROLES),
                        key="inventory_substitution_approved_role",
                    )
                    decision_aux_cols = st.columns(2)
                    decision_substitute_label = decision_aux_cols[0].selectbox(
                        "Decision substitute item",
                        options=list(substitution_item_options.keys()),
                        index=list(substitution_item_options.keys()).index(
                            pending_row.get("substituteInventoryName")
                            if pending_row.get("substituteInventoryName") in substitution_item_options
                            else "<no substitute item selected>"
                        ),
                        key="inventory_substitution_decision_item",
                    )
                    decision_note = decision_aux_cols[1].text_input(
                        "Decision note",
                        value="",
                        key="inventory_substitution_decision_note",
                    )
                    if st.button(
                        "Apply substitution decision",
                        type="primary",
                        key="inventory_substitution_decision_save",
                    ):
                        try:
                            decide_inventory_substitution(
                                conn,
                                substitution_id=int(pending_row["substitutionId"]),
                                status=decision_status,
                                approved_by=approved_by or None,
                                approved_role=approved_role,
                                approved_quantity=(
                                    float(approved_quantity)
                                    if decision_status == "approved"
                                    else None
                                ),
                                note=decision_note or None,
                                substitute_inventory_item_id=substitution_item_options[
                                    decision_substitute_label
                                ],
                            )
                        except Exception as exc:  # pragma: no cover
                            st.error(f"Failed to apply substitution decision: {exc}")
                        else:
                            st.success("Substitution decision recorded.")
                            _rerun_app()
            else:
                st.caption("No substitution requests recorded yet.")

        with st.expander("Substitution reason governance", expanded=False):
            all_reasons = list_inventory_substitution_reason_codes(conn, active_only=False)
            if all_reasons:
                st.dataframe(
                    pd.DataFrame(all_reasons)[
                        ["code", "label", "description", "active", "systemSeeded"]
                    ].rename(
                        columns={
                            "code": "Code",
                            "label": "Label",
                            "description": "Description",
                            "active": "Active",
                            "systemSeeded": "Seeded",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )
            reason_cols = st.columns(4)
            reason_code = reason_cols[0].text_input("Code", key="inventory_reason_code")
            reason_label = reason_cols[1].text_input("Label", key="inventory_reason_label")
            reason_description = reason_cols[2].text_input(
                "Description",
                key="inventory_reason_description",
            )
            reason_active = reason_cols[3].checkbox(
                "Active",
                value=True,
                key="inventory_reason_active",
            )
            if st.button("Save reason code", key="inventory_reason_save"):
                try:
                    upsert_inventory_substitution_reason_code(
                        conn,
                        code=reason_code,
                        label=reason_label,
                        description=reason_description or None,
                        active=bool(reason_active),
                    )
                except Exception as exc:  # pragma: no cover
                    st.error(f"Failed to save substitution reason code: {exc}")
                else:
                    st.success("Substitution reason code saved.")
                    _rerun_app()

    execution_events = list_inventory_execution_events(conn, job_id=job_filter, limit=50)
    if execution_events:
        st.markdown("#### Recent execution events")
        events_df = pd.DataFrame(execution_events)
        st.dataframe(
            events_df[
                [
                    "jobId",
                    "segmentId",
                    "requirementName",
                    "inventoryName",
                    "stage",
                    "quantity",
                    "actor",
                    "containerRef",
                    "truckId",
                    "locationType",
                    "locationLabel",
                    "note",
                    "createdAt",
                ]
            ].rename(
                columns={
                    "jobId": "Job",
                    "segmentId": "Segment",
                    "requirementName": "Requirement",
                    "inventoryName": "Inventory item",
                    "stage": "Stage",
                    "quantity": "Qty",
                    "actor": "Actor",
                    "containerRef": "Container",
                    "truckId": "Truck",
                    "locationType": "Location type",
                    "locationLabel": "Location",
                    "note": "Note",
                    "createdAt": "Recorded",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    balances = list_inventory_balances(
        conn, job_id=job_filter, states=state_filter or None
    )
    balances_df = pd.DataFrame(balances)
    if balances_df.empty:
        st.info("No inventory items found. Import items to begin tracking balances.")
    else:
        display_columns = [
            "name",
            "state",
            "job_id",
            "on_hand_quantity",
            "allocated_quantity",
            "available_quantity",
            "architecture",
            "custody_location_type",
            "custody_location_label",
            "unit",
            "updated_at",
        ]
        present_columns = [col for col in display_columns if col in balances_df.columns]
        st.dataframe(balances_df[present_columns], width='stretch')

    with st.expander("Import inventory items", expanded=False):
        items_file = st.file_uploader(
            "Upload CSV or Excel for inventory items",
            type=["csv", "xlsx", "xls"],
            key="inventory_items_upload",
        )
        if st.button(
            "Import items",
            type="primary",
            disabled=items_file is None,
            key="inventory_items_import_button",
        ):
            try:
                df = _read_uploaded_inventory_file(items_file)
                imported = import_inventory_items_from_dataframe(conn, df)
            except Exception as exc:  # pragma: no cover - surfaced in UI
                st.error(f"Failed to import inventory items: {exc}")
            else:
                st.success(f"Imported or refreshed {imported} inventory rows.")
                _rerun_app()

    with st.expander("Import suppliers from Google Sheets", expanded=False):
        suppliers_sheet_reference = st.text_input(
            "Operations workbook ID or URL",
            value=_default_operations_sheet_reference(),
            help="Shared operations workbook containing the SUPPLIERS tab.",
            key="suppliers_sheet_reference",
        )
        suppliers_sheet_name = st.text_input(
            "Supplier tab name",
            value="SUPPLIERS",
            key="suppliers_sheet_name",
        )
        if st.button(
            "Import suppliers",
            type="primary",
            disabled=not suppliers_sheet_reference.strip(),
            key="suppliers_import_button",
        ):
            try:
                imported = import_suppliers_from_google_sheet(
                    conn,
                    sheet_id=suppliers_sheet_reference.strip(),
                    sheet_name=suppliers_sheet_name.strip() or "SUPPLIERS",
                )
            except Exception as exc:  # pragma: no cover - surfaced in UI
                st.error(f"Failed to import suppliers: {exc}")
            else:
                st.success(f"Imported or refreshed {imported} suppliers.")
                _rerun_app()

    with st.expander("Import movement events", expanded=False):
        movements_file = st.file_uploader(
            "Upload CSV or Excel for movement events",
            type=["csv", "xlsx", "xls"],
            key="inventory_movements_upload",
        )
        default_reason = st.text_input(
            "Default reason (optional)",
            value="",
            help="Applied when the upload does not specify a reason column.",
        )
        if st.button(
            "Import movements",
            type="primary",
            disabled=movements_file is None,
            key="inventory_movements_import_button",
        ):
            try:
                df = _read_uploaded_inventory_file(movements_file)
                imported = import_inventory_movements_from_dataframe(
                    conn, df, default_reason=default_reason or None
                )
            except Exception as exc:  # pragma: no cover - surfaced in UI
                st.error(f"Failed to import movement events: {exc}")
            else:
                st.success(f"Recorded {imported} movement events.")
                _rerun_app()

    with st.expander("Plan inventory requirements", expanded=False):
        segment_options = {
            f"Job {row['jobId']} / Segment {row['segmentSequence']}": int(row["segmentId"])
            for row in segment_coordination
        }
        item_rows = list_inventory(conn)
        item_lookup = {f"{row['name']}": int(row["id"]) for row in item_rows}
        if not segment_options:
            st.caption("Need at least one planned segment before defining inventory requirements.")
        else:
            req_cols = st.columns(4)
            segment_label = req_cols[0].selectbox(
                "Target segment",
                options=list(segment_options.keys()),
                key="inventory_requirement_segment",
            )
            selected_segment_id = segment_options[segment_label]
            selected_segment = next(
                row for row in segment_coordination if int(row["segmentId"]) == int(selected_segment_id)
            )
            item_label = req_cols[1].selectbox(
                "Inventory item (optional)",
                options=["<generic requirement>"] + list(item_lookup.keys()),
                key="inventory_requirement_item",
            )
            architecture = req_cols[2].selectbox(
                "Architecture",
                options=list(INVENTORY_ARCHITECTURES),
                index=list(INVENTORY_ARCHITECTURES).index("container"),
                key="inventory_requirement_architecture",
            )
            substitution_allowed = req_cols[3].checkbox(
                "Substitution allowed",
                value=False,
                key="inventory_requirement_substitution",
            )
            req_name_default = item_label if item_label != "<generic requirement>" else ""
            requirement_name = st.text_input(
                "Requirement name",
                value=req_name_default,
                key="inventory_requirement_name",
            )
            qty_cols = st.columns(2)
            required_quantity = qty_cols[0].number_input(
                "Required quantity",
                min_value=0.1,
                value=1.0,
                step=0.5,
                key="inventory_requirement_quantity",
            )
            requirement_notes = qty_cols[1].text_input(
                "Notes",
                value="",
                key="inventory_requirement_notes",
            )
            if st.button("Save requirement", type="primary", key="inventory_requirement_save"):
                try:
                    upsert_inventory_requirement(
                        conn,
                        job_id=int(selected_segment["jobId"]),
                        segment_id=int(selected_segment_id),
                        inventory_item_id=item_lookup.get(item_label) if item_label != "<generic requirement>" else None,
                        requirement_name=requirement_name,
                        required_quantity=float(required_quantity),
                        substitution_allowed=bool(substitution_allowed),
                        architecture=architecture,
                        notes=requirement_notes or None,
                    )
                except Exception as exc:  # pragma: no cover
                    st.error(f"Failed to save inventory requirement: {exc}")
                else:
                    st.success("Inventory requirement saved.")
                    _rerun_app()

    with st.expander("Reserve or release stock", expanded=False):
        if balances_df.empty:
            st.caption("Add inventory items to enable reservations and releases.")
        else:
            option_labels = {
                f"{row['name']} ({row['available_quantity']} available)": int(row["id"])
                for row in balances
            }
            selected_label = st.selectbox(
                "Inventory item",
                options=list(option_labels.keys()),
                key="inventory_reservation_item",
            )
            quantity = st.number_input(
                "Quantity", min_value=1, step=1, value=1, key="inventory_reservation_qty"
            )
            target_state = st.selectbox(
                "Set state",
                INVENTORY_STATES,
                index=INVENTORY_STATES.index("staged"),
                key="inventory_reservation_state",
            )

            item_id = option_labels.get(selected_label)
            cols = st.columns(2)
            with cols[0]:
                if st.button("Reserve allocation", type="primary"):
                    record_inventory_movement(
                        conn,
                        inventory_item_id=int(item_id),
                        change_allocated=int(quantity),
                        state=target_state,
                        job_id=job_filter,
                    )
                    st.success("Reserved stock and updated state.")
                    _rerun_app()
            with cols[1]:
                if st.button("Release allocation"):
                    record_inventory_movement(
                        conn,
                        inventory_item_id=int(item_id),
                        change_allocated=-int(quantity),
                        state=target_state,
                        job_id=job_filter,
                    )
                    st.success("Released stock and updated state.")
                    _rerun_app()

    with st.expander("Update custody / location", expanded=False):
        if balances_df.empty:
            st.caption("Add inventory items before recording custody/location changes.")
        else:
            custody_options = {
                f"{row['name']} ({row['available_quantity']} available)": int(row["id"])
                for row in balances
            }
            custody_label = st.selectbox(
                "Inventory item for custody update",
                options=list(custody_options.keys()),
                key="inventory_custody_item",
            )
            custody_cols = st.columns(3)
            location_type = custody_cols[0].selectbox(
                "Location type",
                options=list(INVENTORY_CUSTODY_TYPES),
                key="inventory_custody_type",
            )
            location_ref = custody_cols[1].text_input(
                "Location reference",
                value="",
                key="inventory_custody_ref",
            )
            location_label = custody_cols[2].text_input(
                "Location label",
                value="",
                key="inventory_custody_label",
            )
            custody_state = st.selectbox(
                "State",
                options=list(INVENTORY_STATES),
                index=list(INVENTORY_STATES).index("staged"),
                key="inventory_custody_state",
            )
            if st.button("Record custody update", type="primary", key="inventory_custody_save"):
                try:
                    record_inventory_movement(
                        conn,
                        inventory_item_id=int(custody_options[custody_label]),
                        reason="custody_update",
                        state=custody_state,
                        job_id=job_filter,
                        location_type=location_type,
                        location_ref=location_ref or None,
                        location_label=location_label or None,
                    )
                except Exception as exc:  # pragma: no cover
                    st.error(f"Failed to record custody update: {exc}")
                else:
                    st.success("Custody/location updated.")
                    _rerun_app()

    with st.expander("Allocate inventory to planned segment", expanded=False):
        segment_options = {
            f"Job {row['jobId']} / Segment {row['segmentSequence']}": int(row["segmentId"])
            for row in segment_coordination
        }
        item_options = (
            {
                f"{row['name']} ({row['available_quantity']} available)": int(row["id"])
                for row in balances
            }
            if balances
            else {
                f"{row['name']}": int(row["id"])
                for row in list_inventory(conn)
            }
        )
        if not segment_options or not item_options:
            st.caption("Need at least one planned segment and one inventory item before allocating stock.")
        else:
            segment_label = st.selectbox(
                "Target segment",
                options=list(segment_options.keys()),
                key="inventory_segment_target",
            )
            item_label = st.selectbox(
                "Inventory item for segment",
                options=list(item_options.keys()),
                key="inventory_segment_item",
            )
            alloc_quantity = st.number_input(
                "Allocation quantity",
                min_value=0.1,
                value=1.0,
                step=0.5,
                key="inventory_segment_quantity",
            )
            alloc_status = st.selectbox(
                "Shipment status",
                options=["planned", "staged", "loaded", "in_transit"],
                index=0,
                key="inventory_segment_status",
            )
            if st.button(
                "Allocate to segment",
                type="primary",
                key="inventory_segment_allocate_button",
            ):
                try:
                    allocate_inventory_to_segment(
                        conn,
                        segment_id=segment_options[segment_label],
                        inventory_item_id=item_options[item_label],
                        quantity=float(alloc_quantity),
                        status=alloc_status,
                    )
                except Exception as exc:  # pragma: no cover
                    st.error(f"Failed to allocate inventory to segment: {exc}")
                else:
                    st.success("Inventory allocated to segment.")
                    _rerun_app()

    with st.expander("Recent movements", expanded=True):
        movements = list_inventory_movements(
            conn, limit=100, job_id=job_filter, states=state_filter or None
        )
        movements_df = pd.DataFrame(movements)
        if movements_df.empty:
            st.caption("No movement history available for the current filters.")
        else:
            display_columns = [
                "inventory_name",
                "movement_state",
                "job_id",
                "change_on_hand",
                "change_allocated",
                "reason",
                "location_type_value",
                "location_label_value",
                "sequence_no",
                "created_at",
            ]
            present_columns = [
                col for col in display_columns if col in movements_df.columns
            ]
            st.dataframe(movements_df[present_columns], width='stretch')

    with st.expander("Inventory exceptions", expanded=True):
        exceptions = list_inventory_exceptions(conn, resolved=False)
        if not exceptions:
            st.caption("No outstanding exceptions detected by reconciliation jobs.")
        else:
            for exception in exceptions:
                cols = st.columns([4, 1])
                with cols[0]:
                    st.markdown(
                        f"**Item:** {exception.get('inventory_name') or 'Unknown'}  \
                        **State:** {exception.get('state') or 'n/a'}  \
                        **Job:** {exception.get('job_id') or exception.get('inventory_job_id') or 'n/a'}"
                    )
                    st.caption(exception.get("notes") or "No notes recorded.")
                with cols[1]:
                    if st.button(
                        "Reconcile",
                        key=f"inventory_exception_{exception['id']}",
                    ):
                        resolve_inventory_exception(
                            conn,
                            exception_id=int(exception["id"]),
                            note="Reconciled via dashboard",
                        )
                        st.success("Exception marked as reconciled.")
                        _rerun_app()
def _split_worker_name(name: str) -> tuple[str, str]:
    parts = name.strip().split(" ", 1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def _format_truck_list(truck_string: str | float | None) -> str:
    if truck_string is None or (isinstance(truck_string, float) and pd.isna(truck_string)):
        return ""
    trucks = {truck.strip() for truck in str(truck_string).split(",") if truck.strip()}
    return ", ".join(sorted(trucks))


def _default_operations_sheet_reference() -> str:
    return (
        os.environ.get("OPERATIONS_WORKBOOK_URL")
        or os.environ.get("OPERATIONS_WORKBOOK_SHEET_ID")
        or ""
    )


def _worker_time_events_df(
    conn: sqlite3.Connection,
    *,
    limit: int = 500,
) -> pd.DataFrame:
    rows = list_worker_time_capture_events(conn, limit=limit)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "createdAt" in df.columns:
        df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce")
    if "reviewedAt" in df.columns:
        df["reviewedAt"] = pd.to_datetime(df["reviewedAt"], errors="coerce")
    if "effectiveTimestamp" in df.columns:
        df["effectiveTimestamp"] = pd.to_datetime(df["effectiveTimestamp"], errors="coerce")
    if "rawPayload" in df.columns:
        df["anomalyFlags"] = df["rawPayload"].apply(
            lambda payload: ", ".join((payload or {}).get("anomalyFlags", []))
            if isinstance(payload, dict)
            else ""
        )
    else:
        df["anomalyFlags"] = ""
    return df


def _build_worker_time_shift_comparison(
    *,
    imported_shifts: pd.DataFrame,
    worker_time_events: pd.DataFrame,
) -> pd.DataFrame:
    def _norm(value: Any) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        if isinstance(value, int):
            return str(value)
        text = str(value).strip()
        if text.endswith(".0"):
            try:
                return str(int(float(text)))
            except ValueError:
                return text
        return text

    def _event_within_shift_window(
        shift_date_value: Any,
        shift_window_start: Any,
        shift_window_end: Any,
        effective_timestamp: Any,
    ) -> bool | None:
        if pd.isna(shift_date_value) or pd.isna(effective_timestamp):
            return None
        if not _norm(shift_window_start) or not _norm(shift_window_end):
            return None
        event_ts = pd.to_datetime(effective_timestamp, errors="coerce")
        if pd.isna(event_ts):
            return None
        if getattr(event_ts, "tzinfo", None) is not None:
            event_ts = event_ts.tz_localize(None)
        window_start = pd.to_datetime(
            f"{shift_date_value} {_norm(shift_window_start)}",
            errors="coerce",
        )
        window_end = pd.to_datetime(
            f"{shift_date_value} {_norm(shift_window_end)}",
            errors="coerce",
        )
        if pd.isna(window_start) or pd.isna(window_end):
            return None
        if window_end < window_start:
            window_end = window_end + pd.Timedelta(days=1)
            if event_ts < window_start:
                event_ts = event_ts + pd.Timedelta(days=1)
        return bool(window_start <= event_ts <= window_end)

    comparison_rows: list[dict[str, Any]] = []
    accepted_events = worker_time_events[
        worker_time_events["reviewStatus"] == "accepted"
    ].copy()
    accepted_events = accepted_events.reset_index(drop=True)
    matched_event_indexes: set[int] = set()

    for _, imported_row in imported_shifts.iterrows():
        shift_date = imported_row.get("shift_date")
        if pd.isna(shift_date):
            continue
        shift_date_str = str(shift_date)
        worker_name = _norm(imported_row.get("worker_name"))
        truck_id = _norm(imported_row.get("truck_id"))
        linked_job_id = _norm(imported_row.get("linked_job_id"))
        imported_window = " - ".join(
            part for part in [_norm(imported_row.get("shift_window_start")), _norm(imported_row.get("shift_window_end"))] if part
        ) or "n/a"

        candidate_mask = (
            accepted_events["effective_date"].astype(str) == shift_date_str
        ) & (accepted_events["workerName"].fillna("").astype(str).str.strip() == worker_name)
        candidates = accepted_events[candidate_mask].copy()

        status = "imported_only"
        call_truck = ""
        call_job = ""
        call_time = ""
        matched_event_index: int | None = None

        if not candidates.empty:
            in_window_candidates: list[tuple[int, pd.Series]] = []
            fallback_candidates: list[tuple[int, pd.Series]] = []
            for candidate_index, candidate in candidates.iterrows():
                within_window = _event_within_shift_window(
                    shift_date,
                    imported_row.get("shift_window_start"),
                    imported_row.get("shift_window_end"),
                    candidate.get("effectiveTimestamp"),
                )
                if within_window is True or within_window is None:
                    in_window_candidates.append((candidate_index, candidate))
                else:
                    fallback_candidates.append((candidate_index, candidate))

            candidate_groups = [in_window_candidates, fallback_candidates]
            for candidate_group in candidate_groups:
                for candidate_index, candidate in candidate_group:
                    candidate_truck = _norm(candidate.get("truckId"))
                    candidate_job = _norm(candidate.get("jobId"))
                    candidate_time = _norm(candidate.get("effectiveTimestamp"))
                    same_truck = candidate_truck == truck_id
                    same_job = candidate_job == linked_job_id
                    if same_truck and same_job:
                        status = "matched" if candidate_group is in_window_candidates else "time_mismatch"
                        call_truck = candidate_truck
                        call_job = candidate_job
                        call_time = candidate_time
                        matched_event_index = candidate_index
                        break
                if matched_event_index is not None:
                    break

            if matched_event_index is None and in_window_candidates:
                for candidate_index, candidate in in_window_candidates:
                    candidate_truck = _norm(candidate.get("truckId"))
                    candidate_job = _norm(candidate.get("jobId"))
                    candidate_time = _norm(candidate.get("effectiveTimestamp"))
                    same_truck = candidate_truck == truck_id
                    same_job = candidate_job == linked_job_id
                    if same_truck and not same_job:
                        status = "job_mismatch"
                    elif same_job and not same_truck:
                        status = "truck_mismatch"
                    else:
                        status = "assignment_mismatch"
                    call_truck = candidate_truck
                    call_job = candidate_job
                    call_time = candidate_time
                    matched_event_index = candidate_index
                    break

            if matched_event_index is None and fallback_candidates:
                candidate_index, candidate = fallback_candidates[0]
                candidate_truck = _norm(candidate.get("truckId"))
                candidate_job = _norm(candidate.get("jobId"))
                call_time = _norm(candidate.get("effectiveTimestamp"))
                same_truck = candidate_truck == truck_id
                same_job = candidate_job == linked_job_id
                if same_truck and same_job:
                    status = "time_mismatch"
                elif same_truck and not same_job:
                    status = "job_mismatch"
                elif same_job and not same_truck:
                    status = "truck_mismatch"
                else:
                    status = "assignment_mismatch"
                call_truck = candidate_truck
                call_job = candidate_job
                matched_event_index = candidate_index

        if matched_event_index is not None:
            matched_event_indexes.add(matched_event_index)

        comparison_rows.append(
            {
                "Status": status,
                "Date": shift_date_str,
                "Worker": worker_name,
                "Imported window": imported_window,
                "Call time": call_time or "n/a",
                "Imported truck": truck_id or "n/a",
                "Call truck": call_truck or "n/a",
                "Imported job": linked_job_id or "n/a",
                "Call job": call_job or "n/a",
            }
        )

    unmatched_events = accepted_events.drop(index=list(matched_event_indexes), errors="ignore")
    for _, event_row in unmatched_events.iterrows():
        effective_date = event_row.get("effective_date")
        if pd.isna(effective_date):
            continue
        comparison_rows.append(
            {
                "Status": "call_only",
                "Date": str(effective_date),
                "Worker": _norm(event_row.get("workerName")),
                "Imported window": "n/a",
                "Call time": _norm(event_row.get("effectiveTimestamp")) or "n/a",
                "Imported truck": "n/a",
                "Call truck": _norm(event_row.get("truckId")) or "n/a",
                "Imported job": "n/a",
                "Call job": _norm(event_row.get("jobId")) or "n/a",
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    if comparison_df.empty:
        return comparison_df
    return comparison_df.sort_values(
        by=["Date", "Worker", "Status", "Imported truck", "Call truck"],
        ascending=[True, True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)


def _display_worker_time_shift_comparison(
    comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    if comparison_df.empty:
        return comparison_df
    status_labels = {
        "matched": "Matched",
        "time_mismatch": "Timing drift",
        "truck_mismatch": "Truck mismatch",
        "job_mismatch": "Job mismatch",
        "assignment_mismatch": "Truck + job mismatch",
        "imported_only": "Imported only",
        "call_only": "Call-derived only",
    }
    status_explanations = {
        "matched": "Imported shift and accepted worker-time event align.",
        "time_mismatch": "Worker/job/truck align, but the accepted event falls outside the imported shift window.",
        "truck_mismatch": "Worker and job align, but truck assignment differs.",
        "job_mismatch": "Worker and truck align, but linked job differs.",
        "assignment_mismatch": "Worker matches, but both truck and job differ.",
        "imported_only": "Imported shift has no accepted call-derived worker-time match.",
        "call_only": "Accepted worker-time event has no imported shift match.",
    }
    display_df = comparison_df.copy()
    display_df["Reconciliation"] = display_df["Status"].map(status_labels).fillna(
        display_df["Status"]
    )
    display_df["Why"] = display_df["Status"].map(status_explanations).fillna("")
    ordered_columns = [
        "Reconciliation",
        "Why",
        "Date",
        "Worker",
        "Imported window",
        "Call time",
        "Imported truck",
        "Call truck",
        "Imported job",
        "Call job",
        "Status",
    ]
    present_columns = [column for column in ordered_columns if column in display_df.columns]
    return display_df[present_columns]


def _render_worker_time_review_controls(
    conn: sqlite3.Connection,
    *,
    pending_events: pd.DataFrame,
    key_prefix: str,
) -> None:
    if pending_events.empty:
        st.caption("No pending worker-time events for the current selection.")
        return

    option_map = {
        (
            f"#{int(row['id'])} · {row.get('eventType') or 'event'} · "
            f"{row.get('workerName') or row.get('workerNameRaw') or 'unknown'}"
        ): row
        for _, row in pending_events.iterrows()
    }
    selected_label = st.selectbox(
        "Pending worker-time event",
        options=list(option_map.keys()),
        key=f"{key_prefix}_pending_worker_time_event",
    )
    selected = option_map[selected_label]
    decision_cols = st.columns(5)
    review_status = decision_cols[0].selectbox(
        "Review decision",
        options=["accepted", "rejected"],
        key=f"{key_prefix}_worker_time_review_status",
    )
    reviewer = decision_cols[1].text_input(
        "Reviewer",
        value="",
        key=f"{key_prefix}_worker_time_reviewer",
    )
    resolved_worker_id = decision_cols[2].text_input(
        "Resolved worker id",
        value=str(selected.get("workerId") or ""),
        key=f"{key_prefix}_worker_time_worker_id",
    )
    resolved_job_id = decision_cols[3].text_input(
        "Resolved job id",
        value=str(selected.get("jobId") or ""),
        key=f"{key_prefix}_worker_time_job_id",
    )
    resolved_segment_id = decision_cols[4].text_input(
        "Resolved segment id",
        value=str(selected.get("segmentId") or ""),
        key=f"{key_prefix}_worker_time_segment_id",
    )
    follow_cols = st.columns(2)
    resolved_truck_id = follow_cols[0].text_input(
        "Resolved truck id",
        value=str(selected.get("truckId") or ""),
        key=f"{key_prefix}_worker_time_truck_id",
    )
    review_note = follow_cols[1].text_input(
        "Review note",
        value="",
        key=f"{key_prefix}_worker_time_review_note",
    )
    if st.button("Apply worker-time review", key=f"{key_prefix}_apply_worker_time_review"):
        try:
            decide_worker_time_capture_event(
                conn,
                event_id=int(selected["id"]),
                review_status=review_status,
                reviewer=reviewer or None,
                review_note=review_note or None,
                worker_id=_int_or_none(resolved_worker_id),
                job_id=_int_or_none(resolved_job_id),
                segment_id=_int_or_none(resolved_segment_id),
                truck_id=(resolved_truck_id or None),
            )
        except Exception as exc:  # pragma: no cover
            st.error(f"Failed to review worker-time event: {exc}")
        else:
            st.success("Worker-time review recorded.")
            _rerun_app()


def _prepare_staff_export(workers_df: pd.DataFrame) -> bytes:
    export_df = workers_df.copy()
    first_names: list[str] = []
    last_names: list[str] = []
    for _, row in export_df.iterrows():
        first, last = _split_worker_name(str(row.get("name", "")))
        first_names.append(first)
        last_names.append(last)

    export_df.insert(0, "FIRST NAME", first_names)
    export_df.insert(1, "LAST NAME", last_names)
    export_df = export_df[
        [
            "FIRST NAME",
            "LAST NAME",
            "role",
            "rate",
            "tickets",
            "phone",
            "active",
        ]
    ].rename(columns={"role": "ROLE", "rate": "RATE", "tickets": "TICKETS"})

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        export_df.to_excel(writer, sheet_name="STAFF", index=False)
    buffer.seek(0)
    return buffer.getvalue()


def render_staff_tab(conn: sqlite3.Connection) -> None:
    st.subheader("Staff roster (STAFF)")
    st.caption(
        "Import, audit, and edit the STAFF worksheet. Link workers to driver shifts and vehicles."
    )

    ensure_dashboard_tables(conn)
    assignment_summary = list_worker_assignment_summary(conn)

    import_feedback: Optional[tuple[str, str]] = None
    with st.expander("Import/export STAFF worksheet", expanded=False):
        google_col, import_col, export_col = st.columns(3)
        with google_col:
            staff_sheet_reference = st.text_input(
                "Google Sheets ID or URL",
                value=_default_operations_sheet_reference(),
                help="Shared operations workbook containing the STAFF tab.",
                key="staff_sheet_reference",
            )
            if st.button("Import STAFF from Google Sheet", key="staff_google_import_button"):
                try:
                    inserted, updated = import_workers_from_google_sheet(
                        conn,
                        sheet_id_or_url=staff_sheet_reference.strip() or None,
                    )
                except Exception as exc:  # pragma: no cover - surfaced in UI
                    import_feedback = (
                        "error",
                        f"Failed to import staff from Google Sheets: {exc}",
                    )
                else:
                    import_feedback = (
                        "success",
                        f"Imported {inserted} new staff and updated {updated} existing records from Google Sheets.",
                    )
        with import_col:
            staff_upload = st.file_uploader(
                "Upload STAFF workbook (.xlsx)",
                type=["xlsx"],
                help="Re-use the STAFF worksheet downloaded from Google Sheets.",
                key="staff_upload_widget",
            )
            if st.button("Import STAFF", key="staff_import_button"):
                if staff_upload is None:
                    import_feedback = (
                        "warning",
                        "Choose a STAFF workbook before importing.",
                    )
                else:
                    try:
                        inserted, updated = import_workers_from_staff_sheet(
                            conn, staff_upload
                        )
                    except Exception as exc:  # pragma: no cover - surfaced in UI
                        import_feedback = (
                            "error",
                            f"Failed to import staff: {exc}",
                        )
                    else:
                        import_feedback = (
                            "success",
                            f"Imported {inserted} new staff and updated {updated} existing records.",
                        )

        with export_col:
            workers_for_export = pd.read_sql_query(
                "SELECT name, role, rate, tickets, phone, active FROM workers ORDER BY name",
                conn,
            )
            if workers_for_export.empty:
                st.caption(
                    "Add staff before exporting a workbook compatible with the STAFF sheet."
                )
            else:
                export_bytes = _prepare_staff_export(workers_for_export)
                st.download_button(
                    "Download STAFF workbook",
                    export_bytes,
                    file_name="STAFF.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="staff_export_button",
                )

    if import_feedback:
        level, message = import_feedback
        if level == "success":
            st.success(message)
        elif level == "warning":
            st.info(message)
        else:
            st.error(message)

    workers_df = pd.read_sql_query(
        """
        SELECT
            w.id,
            w.name,
            w.role,
            w.phone,
            w.rate,
            w.tickets,
            w.active,
            w.hired_at,
            w.updated_at,
            COUNT(ds.id) AS shift_count,
            MAX(ds.shift_date) AS last_shift_date,
            GROUP_CONCAT(DISTINCT ds.truck_id) AS shift_trucks
        FROM workers AS w
        LEFT JOIN driver_shifts AS ds ON ds.worker_id = w.id
        GROUP BY w.id
        ORDER BY w.name
        """,
        conn,
    )

    vehicle_df = pd.read_sql_query(
        """
        SELECT truck_id, present_driver
        FROM vehicle_details
        WHERE present_driver IS NOT NULL AND TRIM(present_driver) != ''
        """,
        conn,
    )
    vehicle_assignments = (
        vehicle_df.groupby("present_driver")["truck_id"]
        .apply(lambda series: ", ".join(sorted({str(val).strip() for val in series if str(val).strip()})))
        .to_dict()
    )

    if not workers_df.empty:
        workers_df["active"] = workers_df["active"].astype(bool)
        workers_df["last_shift_date"] = pd.to_datetime(
            workers_df["last_shift_date"], errors="coerce"
        ).dt.date
        workers_df["shift_trucks"] = workers_df["shift_trucks"].apply(_format_truck_list)
        workers_df["imported_trucks"] = workers_df["name"].map(vehicle_assignments).fillna("")
        workers_df["planned_segment_count"] = workers_df["id"].map(
            lambda worker_id: assignment_summary.get(int(worker_id), {}).get("plannedSegmentCount", 0)
        )
        workers_df["planned_job_count"] = workers_df["id"].map(
            lambda worker_id: assignment_summary.get(int(worker_id), {}).get("plannedJobCount", 0)
        )
        workers_df["planned_trucks"] = workers_df["id"].map(
            lambda worker_id: ", ".join(
                assignment_summary.get(int(worker_id), {}).get("plannedTrucks", [])
            )
        )
        workers_df["next_planned_start"] = pd.to_datetime(
            workers_df["id"].map(
                lambda worker_id: assignment_summary.get(int(worker_id), {}).get("nextPlannedStart")
            ),
            errors="coerce",
        ).dt.date
        workers_df["shift_count"] = workers_df["shift_count"].fillna(0).astype(int)

    summary_cols = st.columns(3)
    summary_cols[0].metric("Total workers", int(len(workers_df)))
    active_count = int(workers_df[workers_df["active"]].shape[0]) if not workers_df.empty else 0
    summary_cols[1].metric("Active workers", active_count)
    summary_cols[2].metric(
        "Workers on planned segments",
        int((workers_df["planned_segment_count"] > 0).sum()) if not workers_df.empty else 0,
    )

    filter_cols = st.columns(3)
    name_filter = filter_cols[0].text_input("Search by name", key="staff_name_filter")
    role_options = (
        sorted(
            filter(
                lambda r: bool(r),
                workers_df.get("role").dropna().unique().tolist(),
            )
        )
        if not workers_df.empty and "role" in workers_df
        else []
    )
    role_filter = filter_cols[1].multiselect("Roles", role_options, key="staff_role_filter")
    status_filter = filter_cols[2].selectbox(
        "Active status",
        ["All", "Active", "Inactive"],
        key="staff_status_filter",
    )

    filtered_df = workers_df.copy()
    if name_filter:
        filtered_df = filtered_df[
            filtered_df["name"].str.contains(name_filter, case=False, na=False)
        ]
    if role_filter:
        filtered_df = filtered_df[filtered_df["role"].isin(role_filter)]
    if status_filter == "Active":
        filtered_df = filtered_df[filtered_df["active"]]
    elif status_filter == "Inactive":
        filtered_df = filtered_df[~filtered_df["active"]]

    display_columns = [
        "id",
        "name",
        "role",
        "phone",
        "rate",
        "tickets",
        "active",
        "last_shift_date",
        "shift_count",
        "planned_segment_count",
        "planned_job_count",
        "next_planned_start",
        "planned_trucks",
        "shift_trucks",
        "imported_trucks",
    ]
    present_columns = [col for col in display_columns if col in filtered_df.columns]
    edited_df = st.data_editor(
        filtered_df[present_columns],
        hide_index=True,
        width='stretch',
        num_rows="dynamic",
        column_config={
            "id": st.column_config.Column("ID", disabled=True, width="small"),
            "name": st.column_config.Column("Name"),
            "role": st.column_config.Column("Role"),
            "phone": st.column_config.Column("Phone"),
            "rate": st.column_config.NumberColumn("Rate", format="%.2f"),
            "tickets": st.column_config.NumberColumn("Tickets", format="%d"),
            "active": st.column_config.CheckboxColumn("Active"),
            "last_shift_date": st.column_config.DateColumn("Last shift", disabled=True),
            "shift_count": st.column_config.NumberColumn("Shift count", disabled=True),
            "planned_segment_count": st.column_config.NumberColumn("Planned segments", disabled=True),
            "planned_job_count": st.column_config.NumberColumn("Planned jobs", disabled=True),
            "next_planned_start": st.column_config.DateColumn("Next planned start", disabled=True),
            "planned_trucks": st.column_config.Column("Planned trucks", disabled=True),
            "shift_trucks": st.column_config.Column("Recent trucks", disabled=True),
            "imported_trucks": st.column_config.Column(
                "Imported sheet trucks", disabled=True
            ),
        },
    )

    if st.button("Save staff changes", type="primary", key="staff_save_button"):
        if edited_df.empty:
            st.info("No staff rows to save.")
        else:
            errors: list[str] = []
            saved = 0
            for idx, row in edited_df.iterrows():
                name = str(row.get("name") or "").strip()
                if not name:
                    errors.append(f"Row {idx + 1}: name is required.")
                    continue

                rate_raw = row.get("rate")
                rate_value: float | None
                if rate_raw in ("", None) or (isinstance(rate_raw, float) and math.isnan(rate_raw)):
                    rate_value = None
                else:
                    try:
                        rate_value = float(rate_raw)
                        if rate_value < 0:
                            raise ValueError("Rate cannot be negative")
                    except Exception as exc:
                        errors.append(f"{name}: invalid rate ({exc}).")
                        continue

                tickets_raw = row.get("tickets")
                tickets_value: int | None
                if tickets_raw in ("", None) or (isinstance(tickets_raw, float) and math.isnan(tickets_raw)):
                    tickets_value = None
                else:
                    try:
                        tickets_value = int(tickets_raw)
                        if tickets_value < 0:
                            raise ValueError("Tickets cannot be negative")
                    except Exception as exc:
                        errors.append(f"{name}: invalid tickets ({exc}).")
                        continue

                try:
                    upsert_worker(
                        conn,
                        name=name,
                        role=str(row.get("role") or ""),
                        phone=str(row.get("phone") or ""),
                        rate=rate_value,
                        tickets=tickets_value,
                        active=bool(row.get("active")),
                    )
                except Exception as exc:  # pragma: no cover - surfaced in UI
                    errors.append(f"{name}: failed to save ({exc}).")
                else:
                    saved += 1

            if errors:
                st.error("\n".join(errors))
            if saved and not errors:
                st.success(f"Saved {saved} staff record{'s' if saved != 1 else ''}.")
                _rerun_app()
            elif saved:
                st.info(
                    f"Saved {saved} staff record{'s' if saved != 1 else ''}. Fix the remaining issues and try again."
                )

    st.divider()
    st.subheader("Linked shifts and vehicle assignments")
    if workers_df.empty:
        st.info("No staff available to display shift links.")
        return

    worker_choice = st.selectbox(
        "Choose a worker to review recent shifts and vehicles",
        sorted(workers_df["name"].tolist()),
        key="staff_worker_review",
    )
    if worker_choice:
        worker_row = workers_df.loc[workers_df["name"] == worker_choice].iloc[0]
        worker_time_df = _worker_time_events_df(conn, limit=500)
        if not worker_time_df.empty:
            worker_time_df = worker_time_df[
                worker_time_df["workerId"].fillna(-1).astype(int) == int(worker_row["id"])
            ].copy()
        planned_segments = list_segments_for_worker(conn, worker_id=int(worker_row["id"]))
        if planned_segments:
            planned_df = pd.DataFrame(
                [
                    {
                        "Job": row["jobId"],
                        "Segment": row["segmentSequence"],
                        "From": row["fromLocation"] or row["jobOrigin"],
                        "To": row["toLocation"] or row["jobDestination"],
                        "Planned start": row["plannedStart"],
                        "Planned end": row["plannedEnd"],
                        "Status": row["assignmentStatus"],
                        "Trucks": ", ".join(
                            assignment["truckId"]
                            for assignment in row["truckAssignments"]
                            if assignment.get("truckId")
                        ),
                    }
                    for row in planned_segments
                ]
            )
            st.markdown("#### Planned segment assignments")
            st.dataframe(planned_df, width='stretch', hide_index=True)

        shift_df = load_driver_shifts_dataframe(conn, workers=[worker_choice])
        if not shift_df.empty:
            shift_df = shift_df.copy()
            shift_df["shift_date"] = pd.to_datetime(shift_df["shift_date"], errors="coerce").dt.date
            columns = [
                "shift_date",
                "truck_id",
                "truck_name",
                "shift_window_start",
                "shift_window_end",
                "shift_start",
                "shift_end",
                "hours",
                "hourly_rate",
                "cost_total",
                "source",
            ]
            present_shift_cols = [col for col in columns if col in shift_df.columns]
            st.dataframe(
                shift_df.sort_values(by="shift_date", ascending=False)[present_shift_cols],
                width='stretch',
            )
        else:
            st.caption("No driver shifts linked to this worker yet.")

        imported_trucks = vehicle_assignments.get(worker_choice)
        if imported_trucks:
            st.info(f"Imported sheet truck context: {imported_trucks}")
        elif not shift_df.empty:
            st.caption("No imported sheet truck assignment; trucks only appear in recorded shifts.")
        else:
            st.caption("No imported sheet truck assignment recorded for this worker.")

        st.markdown("#### Reviewed worker-time events")
        if worker_time_df.empty:
            st.caption("No worker-time capture events are linked to this worker yet.")
        else:
            review_cols = st.columns(4)
            review_cols[0].metric(
                "Pending review",
                int((worker_time_df["reviewStatus"] == "pending_review").sum()),
            )
            review_cols[1].metric(
                "Accepted",
                int((worker_time_df["reviewStatus"] == "accepted").sum()),
            )
            review_cols[2].metric(
                "Rejected",
                int((worker_time_df["reviewStatus"] == "rejected").sum()),
            )
            latest_reviewed = worker_time_df["reviewedAt"].dropna()
            review_cols[3].metric(
                "Latest reviewed",
                latest_reviewed.max().date().isoformat() if not latest_reviewed.empty else "n/a",
            )
            st.dataframe(
                worker_time_df[
                    [
                        "id",
                        "eventType",
                        "channel",
                        "effectiveTimestamp",
                        "confidence",
                        "reviewStatus",
                        "reviewer",
                        "reviewedAt",
                        "jobId",
                        "segmentId",
                        "truckId",
                        "anomalyFlags",
                    ]
                ].rename(
                    columns={
                        "id": "Event",
                        "eventType": "Event type",
                        "channel": "Channel",
                        "effectiveTimestamp": "Effective time",
                        "confidence": "Confidence",
                        "reviewStatus": "Review",
                        "reviewer": "Reviewer",
                        "reviewedAt": "Reviewed at",
                        "jobId": "Job",
                        "segmentId": "Segment",
                        "truckId": "Truck",
                        "anomalyFlags": "Anomalies",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
            with st.expander("Review pending worker-time events", expanded=False):
                pending_worker_time = worker_time_df[
                    worker_time_df["reviewStatus"] == "pending_review"
                ].copy()
                _render_worker_time_review_controls(
                    conn,
                    pending_events=pending_worker_time,
                    key_prefix=f"staff_worker_time_{int(worker_row['id'])}",
                )

    st.divider()
    st.subheader("Roles and compliances")
    if workers_df.empty:
        st.caption("Add staff before managing roles and compliances.")
        return

    admin_cols = st.columns(2)
    with admin_cols[0]:
        selected_admin_worker = st.selectbox(
            "Worker for role/compliance admin",
            sorted(workers_df["name"].tolist()),
            key="staff_worker_admin",
        )
    worker_admin_row = workers_df.loc[workers_df["name"] == selected_admin_worker].iloc[0]
    worker_admin_id = int(worker_admin_row["id"])

    role_rows = conn.execute(
        """
        SELECT wr.id, wr.name
        FROM worker_role_assignments AS wra
        JOIN worker_roles AS wr ON wr.id = wra.role_id
        WHERE wra.worker_id = ?
        ORDER BY wr.name
        """,
        (worker_admin_id,),
    ).fetchall()
    compliance_rows = conn.execute(
        """
        SELECT wc.id, wc.name, wca.expiry_date
        FROM worker_compliance_assignments AS wca
        JOIN worker_compliances AS wc ON wc.id = wca.compliance_id
        WHERE wca.worker_id = ?
        ORDER BY wc.name
        """,
        (worker_admin_id,),
    ).fetchall()

    role_col, compliance_col = st.columns(2)
    with role_col:
        st.markdown("#### Role assignments")
        if role_rows:
            st.dataframe(
                pd.DataFrame([{"Role": row["name"]} for row in role_rows]),
                width='stretch',
                hide_index=True,
            )
        else:
            st.caption("No role assignments recorded.")
        available_roles = conn.execute(
            "SELECT id, name FROM worker_roles ORDER BY name"
        ).fetchall()
        role_options = {row["name"]: int(row["id"]) for row in available_roles}
        selected_role_name = st.selectbox(
            "Existing role",
            options=["<new role>", *role_options.keys()],
            key="staff_role_existing_select",
        )
        new_role_name = st.text_input("New role name", value="", key="staff_new_role_name")
        if st.button("Assign role", key="staff_assign_role_button"):
            try:
                role_id = (
                    ensure_worker_role(conn, name=new_role_name.strip())
                    if selected_role_name == "<new role>"
                    else role_options[selected_role_name]
                )
                assign_worker_role(conn, worker_id=worker_admin_id, role_id=role_id)
            except Exception as exc:  # pragma: no cover
                st.error(f"Failed to assign role: {exc}")
            else:
                st.success("Role assignment saved.")
                _rerun_app()

    with compliance_col:
        st.markdown("#### Compliance assignments")
        if compliance_rows:
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Compliance": row["name"], "Expiry": row["expiry_date"]}
                        for row in compliance_rows
                    ]
                ),
                width='stretch',
                hide_index=True,
            )
        else:
            st.caption("No compliance assignments recorded.")
        available_compliances = conn.execute(
            "SELECT id, name FROM worker_compliances ORDER BY name"
        ).fetchall()
        compliance_options = {row["name"]: int(row["id"]) for row in available_compliances}
        selected_compliance_name = st.selectbox(
            "Existing compliance",
            options=["<new compliance>", *compliance_options.keys()],
            key="staff_compliance_existing_select",
        )
        new_compliance_name = st.text_input(
            "New compliance name", value="", key="staff_new_compliance_name"
        )
        expiry_value = st.text_input(
            "Compliance expiry (ISO date)",
            value="",
            placeholder="2026-12-31",
            key="staff_compliance_expiry",
        )
        if st.button("Assign compliance", key="staff_assign_compliance_button"):
            try:
                compliance_id = (
                    ensure_worker_compliance(conn, name=new_compliance_name.strip())
                    if selected_compliance_name == "<new compliance>"
                    else compliance_options[selected_compliance_name]
                )
                assign_worker_compliance(
                    conn,
                    worker_id=worker_admin_id,
                    compliance_id=compliance_id,
                    expiry_date=expiry_value.strip() or None,
                )
            except Exception as exc:  # pragma: no cover
                st.error(f"Failed to assign compliance: {exc}")
            else:
                st.success("Compliance assignment saved.")
                _rerun_app()

    worker_readiness_items = [
        item
        for item in list_operational_readiness_items(conn, resource_type="worker")
        if item["resourceId"] == str(worker_admin_id)
    ]
    if worker_readiness_items:
        st.markdown("#### Worker readiness alerts")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Status": item["status"],
                        "Rule": item["ruleType"],
                        "Due": item["dueAt"],
                        "Details": item["details"],
                    }
                    for item in worker_readiness_items
                ]
            ),
            width='stretch',
            hide_index=True,
        )


def render_driver_shifts_tab(conn: sqlite3.Connection) -> None:
    st.subheader("Labor planning and shift reconciliation")
    st.caption(
        "Use native planned labor from job segments as the planning surface. VEHICLE_DRIVER remains an imported reconciliation feed."
    )

    roster_df = pd.DataFrame(list_planned_labor_assignments(conn))
    st.markdown("#### Native planned labor roster")
    if roster_df.empty:
        st.caption("No planned labor assignments exist yet. Assign workers and trucks to job segments in Operations.")
    else:
        roster_display = roster_df.copy()
        roster_display["truckIds"] = roster_display["truckIds"].apply(
            lambda values: ", ".join(values) if isinstance(values, list) else ""
        )
        roster_display["truckNames"] = roster_display["truckNames"].apply(
            lambda values: ", ".join(values) if isinstance(values, list) else ""
        )
        st.dataframe(
            roster_display[
                [
                    "jobId",
                    "segmentSequence",
                    "workerName",
                    "truckIds",
                    "plannedStart",
                    "plannedEnd",
                    "assignmentStatus",
                ]
            ].rename(
                columns={
                    "jobId": "Job",
                    "segmentSequence": "Segment",
                    "workerName": "Worker",
                    "truckIds": "Trucks",
                    "plannedStart": "Planned start",
                    "plannedEnd": "Planned end",
                    "assignmentStatus": "Status",
                }
            ),
            width='stretch',
            hide_index=True,
        )

    reconciliation = pd.DataFrame(list_labor_reconciliation(conn))
    st.markdown("#### Plan vs imported shift reconciliation")
    if reconciliation.empty:
        st.caption("No planned/imported labor reconciliation items available yet.")
    else:
        recon_cols = st.columns(3)
        recon_cols[0].metric(
            "Planned only",
            int((reconciliation["status"] == "planned_only").sum()),
        )
        recon_cols[1].metric(
            "Imported only",
            int((reconciliation["status"] == "imported_only").sum()),
        )
        recon_cols[2].metric(
            "Matched",
            int((reconciliation["status"] == "matched").sum()),
        )
        recon_display = reconciliation.copy()
        recon_display["truckIds"] = recon_display["truckIds"].apply(
            lambda values: ", ".join(values) if isinstance(values, list) else ""
        )
        st.dataframe(
            recon_display[
                [
                    "status",
                    "shiftDate",
                    "workerName",
                    "truckIds",
                    "jobId",
                    "segmentId",
                    "source",
                ]
            ].rename(
                columns={
                    "status": "Status",
                    "shiftDate": "Date",
                    "workerName": "Worker",
                    "truckIds": "Trucks",
                    "jobId": "Job",
                    "segmentId": "Segment",
                    "source": "Source",
                }
            ),
            width='stretch',
            hide_index=True,
        )

    worker_time_df = _worker_time_events_df(conn, limit=500)
    st.markdown("#### Reviewed worker-time capture")
    if worker_time_df.empty:
        st.caption("No worker-time capture events recorded yet.")
    else:
        worker_time_metric_cols = st.columns(4)
        worker_time_metric_cols[0].metric(
            "Pending review",
            int((worker_time_df["reviewStatus"] == "pending_review").sum()),
        )
        worker_time_metric_cols[1].metric(
            "Accepted",
            int((worker_time_df["reviewStatus"] == "accepted").sum()),
        )
        worker_time_metric_cols[2].metric(
            "Rejected",
            int((worker_time_df["reviewStatus"] == "rejected").sum()),
        )
        accepted_hours_proxy = worker_time_df[
            worker_time_df["reviewStatus"] == "accepted"
        ].shape[0]
        worker_time_metric_cols[3].metric("Accepted events", int(accepted_hours_proxy))
        st.dataframe(
            worker_time_df[
                [
                    "id",
                    "workerName",
                    "eventType",
                    "channel",
                    "effectiveTimestamp",
                    "reviewStatus",
                    "jobId",
                    "segmentId",
                    "truckId",
                    "anomalyFlags",
                ]
            ].rename(
                columns={
                    "id": "Event",
                    "workerName": "Worker",
                    "eventType": "Event type",
                    "channel": "Channel",
                    "effectiveTimestamp": "Effective time",
                    "reviewStatus": "Review",
                    "jobId": "Job",
                    "segmentId": "Segment",
                    "truckId": "Truck",
                    "anomalyFlags": "Anomalies",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    st.divider()
    st.markdown("#### Imported VEHICLE_DRIVER feed")

    with st.expander("Import from Google Sheet", expanded=False):
        default_sheet_id = os.environ.get("VEHICLE_DRIVER_SHEET_ID", "")
        sheet_id = st.text_input(
            "Sheet ID or full URL",
            value=default_sheet_id,
            help="Paste the Google Sheet ID or sharing URL for the VEHICLE_DRIVER tab.",
            key="driver_shift_sheet_id",
        )
        sheet_name = st.text_input(
            "Sheet tab name",
            value=DEFAULT_DRIVER_SHEET_NAME,
            help="Defaults to the VEHICLE_DRIVER tab name.",
            key="driver_shift_sheet_name",
        )
        if st.button(
            "Import driver shifts",
            type="primary",
            key="driver_shift_import_button",
            disabled=not sheet_id.strip(),
        ):
            try:
                inserted, updated = import_driver_shifts_from_sheet(
                    conn,
                    sheet_id=sheet_id.strip(),
                    sheet_name=sheet_name.strip() or DEFAULT_DRIVER_SHEET_NAME,
                )
            except Exception as exc:  # pragma: no cover - surfaced in UI
                st.error(f"Failed to import driver shifts: {exc}")
            else:
                st.success(
                    f"Imported {inserted} new shift entries and refreshed {updated} existing rows."
                )

    df = load_driver_shifts_dataframe(conn)
    if df.empty:
        st.info(
            "No driver shifts available. Import the VEHICLE_DRIVER sheet to populate this view."
        )
        return

    df = df.copy()
    df["shift_date"] = pd.to_datetime(df["shift_date"], errors="coerce")
    df = df.dropna(subset=["shift_date"])
    if df.empty:
        st.info("Driver shift dates could not be parsed from the data.")
        return

    min_date = df["shift_date"].min().date()
    max_date = df["shift_date"].max().date()
    date_range = st.date_input(
        "Shift date range",
        value=(min_date, max_date),
    )
    selected_workers = st.multiselect(
        "Drivers/workers",
        sorted(df["worker_name"].dropna().unique().tolist()),
    )
    selected_trucks = st.multiselect(
        "Trucks",
        sorted(df["truck_id"].dropna().unique().tolist()),
    )

    filtered = df
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        if start_date:
            filtered = filtered[filtered["shift_date"] >= pd.to_datetime(start_date)]
        if end_date:
            filtered = filtered[filtered["shift_date"] <= pd.to_datetime(end_date)]
    if selected_workers:
        filtered = filtered[filtered["worker_name"].isin(selected_workers)]
    if selected_trucks:
        filtered = filtered[filtered["truck_id"].isin(selected_trucks)]

    worker_time_filtered = worker_time_df.copy()
    if not worker_time_filtered.empty:
        effective_dates = pd.to_datetime(
            worker_time_filtered["effectiveTimestamp"], errors="coerce"
        )
        worker_time_filtered = worker_time_filtered.assign(
            effective_date=effective_dates.dt.date
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            if start_date:
                worker_time_filtered = worker_time_filtered[
                    worker_time_filtered["effective_date"] >= start_date
                ]
            if end_date:
                worker_time_filtered = worker_time_filtered[
                    worker_time_filtered["effective_date"] <= end_date
                ]
        if selected_workers:
            worker_time_filtered = worker_time_filtered[
                worker_time_filtered["workerName"].isin(selected_workers)
            ]
        if selected_trucks:
            worker_time_filtered = worker_time_filtered[
                worker_time_filtered["truckId"].isin(selected_trucks)
            ]

    filtered = filtered.sort_values(
        by=["shift_date", "shift_start", "truck_id", "worker_name"],
        ascending=[False, True, True, True],
    )
    filtered = filtered.assign(shift_date=filtered["shift_date"].dt.date)

    shift_vs_call_df = _build_worker_time_shift_comparison(
        imported_shifts=filtered,
        worker_time_events=worker_time_filtered,
    )

    total_hours = filtered["hours"].sum(skipna=True) if "hours" in filtered else 0
    total_cost = (
        filtered["cost_total"].sum(skipna=True) if "cost_total" in filtered else 0
    )
    metric_cols = st.columns(2)
    metric_cols[0].metric("Total hours", f"{total_hours:,.2f}")
    metric_cols[1].metric("Total cost", f"${total_cost:,.2f}")

    display_columns = [
        "shift_date",
        "truck_id",
        "truck_name",
        "worker_name",
        "linked_job_id",
        "shipment_id",
        "role",
        "shift_window_start",
        "shift_window_end",
        "ticket_numbers",
        "shift_start",
        "shift_end",
        "hours",
        "hourly_rate",
        "cost_total",
        "source",
    ]
    present_columns = [col for col in display_columns if col in filtered.columns]
    st.dataframe(filtered[present_columns], width='stretch')

    if not worker_time_filtered.empty:
        st.markdown("#### Worker-time events in selected range")
        st.dataframe(
            worker_time_filtered[
                [
                    "effective_date",
                    "workerName",
                    "eventType",
                    "channel",
                    "reviewStatus",
                    "jobId",
                    "segmentId",
                    "truckId",
                    "anomalyFlags",
                ]
            ].rename(
                columns={
                    "effective_date": "Date",
                    "workerName": "Worker",
                    "eventType": "Event type",
                    "channel": "Channel",
                    "reviewStatus": "Review",
                    "jobId": "Job",
                    "segmentId": "Segment",
                    "truckId": "Truck",
                    "anomalyFlags": "Anomalies",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        with st.expander("Review pending worker-time events in this range", expanded=False):
            pending_worker_time = worker_time_filtered[
                worker_time_filtered["reviewStatus"] == "pending_review"
            ].copy()
            _render_worker_time_review_controls(
                conn,
                pending_events=pending_worker_time,
                key_prefix="driver_shifts_worker_time",
            )

    st.markdown("#### Imported shifts vs accepted call-derived worker time")
    if shift_vs_call_df.empty:
        st.caption("No imported/accepted comparison rows are available for the current selection.")
    else:
        display_shift_vs_call_df = _display_worker_time_shift_comparison(shift_vs_call_df)
        compare_cols = st.columns(4)
        compare_cols[0].metric(
            "Matched",
            int((shift_vs_call_df["Status"] == "matched").sum()),
        )
        compare_cols[1].metric(
            "Mismatch / timing drift",
            int(
                shift_vs_call_df["Status"].isin(
                    ["truck_mismatch", "job_mismatch", "assignment_mismatch", "time_mismatch"]
                ).sum()
            ),
        )
        compare_cols[2].metric(
            "Imported only",
            int((shift_vs_call_df["Status"] == "imported_only").sum()),
        )
        compare_cols[3].metric(
            "Call-derived only",
            int((shift_vs_call_df["Status"] == "call_only").sum()),
        )
        st.caption(
            "Rows below show the exact reconciliation class for each imported shift or accepted call-derived event."
        )
        st.dataframe(display_shift_vs_call_df, width="stretch", hide_index=True)


def render_payroll_labor_analytics_tab(conn: sqlite3.Connection) -> None:
    st.subheader("Payroll preparation and labor analytics")
    st.caption(
        "Aggregate-first labor forecasting and workforce insight built from planned assignments, imported shifts, and reviewed worker-time events. Corkysoft prepares payroll truth here; it does not execute payroll."
    )

    baseline = build_payroll_labor_analytics(conn)
    known_dates: list[date] = []
    for row in baseline.get("hoursCostDistributionRows", []):
        parsed = pd.to_datetime(row.get("date"), errors="coerce")
        if not pd.isna(parsed):
            known_dates.append(parsed.date())
    for row in baseline.get("payForecastRows", []):
        parsed = pd.to_datetime(row.get("date"), errors="coerce")
        if not pd.isna(parsed):
            known_dates.append(parsed.date())

    if known_dates:
        min_date = min(known_dates)
        max_date = max(known_dates)
    else:
        today = date.today()
        min_date = today
        max_date = today

    date_range = st.date_input(
        "Payroll / labor date range",
        value=(min_date, max_date),
        key="payroll_labor_analytics_date_range",
    )
    overtime_threshold = st.number_input(
        "Daily overtime threshold (hours)",
        min_value=0.0,
        max_value=24.0,
        value=float(OVERTIME_DAILY_HOURS_DEFAULT),
        step=0.5,
        key="payroll_labor_analytics_overtime_threshold",
    )

    start_date = None
    end_date = None
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date = date_range[0].isoformat() if date_range[0] else None
        end_date = date_range[1].isoformat() if date_range[1] else None

    analytics_payload = build_payroll_labor_analytics(
        conn,
        start_date=start_date,
        end_date=end_date,
        overtime_daily_hours=float(overtime_threshold),
    )

    summary = analytics_payload["summary"]
    pay_forecast_rows = analytics_payload["payForecastRows"]
    export_ready_rows = analytics_payload["exportReadyLaborSummaries"]
    distribution_rows = analytics_payload["hoursCostDistributionRows"]
    overtime_rows = analytics_payload["overtimeRows"]
    plan_vs_actual = analytics_payload["planVsActual"]
    confidence = analytics_payload["confidence"]
    absence_summary = analytics_payload["absenceSummary"]
    absence_rows = analytics_payload["absenceRows"]
    labor_cost_drivers = analytics_payload["laborCostDrivers"]

    summary_cols = st.columns(4)
    summary_cols[0].metric("Planned exposure", f"${summary['plannedExposure']:,.2f}")
    summary_cols[1].metric("Imported labor cost", f"${summary['importedCost']:,.2f}")
    summary_cols[2].metric("Reviewed actual cost", f"${summary['reviewedActualCost']:,.2f}")
    summary_cols[3].metric(
        "Payroll-prep confidence",
        f"{int(summary['confidenceScore'])} ({summary['confidenceLabel']})",
    )

    st.markdown("#### Pay Forecast")
    pay_forecast_df = pd.DataFrame(pay_forecast_rows)
    if pay_forecast_df.empty:
        st.caption("No planned or imported labor data is available for the selected range.")
    else:
        pay_forecast_df = pay_forecast_df.sort_values(
            by=["importedCost", "plannedExposure", "reviewedActualCost"],
            ascending=[False, False, False],
            kind="stable",
        )
        st.dataframe(
            pay_forecast_df.rename(
                columns={
                    "workerName": "Worker",
                    "plannedHours": "Planned hours",
                    "plannedExposure": "Planned exposure",
                    "importedHours": "Imported hours",
                    "importedCost": "Imported cost",
                    "reviewedActualCost": "Reviewed actual cost",
                    "acceptedEventCount": "Accepted events",
                    "hourlyRateBasis": "Hourly-rate basis",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        top_pay_df = pay_forecast_df.head(10).copy()
        pay_chart = px.bar(
            top_pay_df,
            x="workerName",
            y=["plannedExposure", "importedCost", "reviewedActualCost"],
            barmode="group",
            title="Top workers by planned/imported labor cost",
            labels={"workerName": "Worker", "value": "Amount", "variable": "Series"},
        )
        st.plotly_chart(pay_chart, width="stretch")

    st.markdown("#### Export-ready Labor Summary")
    export_df = pd.DataFrame(export_ready_rows)
    if export_df.empty:
        st.caption("No export-ready labor summary rows are available for the selected range.")
    else:
        st.dataframe(
            export_df.rename(
                columns={
                    "workerName": "Worker",
                    "dateRangeStart": "Range start",
                    "dateRangeEnd": "Range end",
                    "plannedExposure": "Planned exposure",
                    "importedCost": "Imported cost",
                    "reviewedActualCost": "Reviewed actual cost",
                    "importedHours": "Imported hours",
                    "overtimeHours": "Overtime hours",
                    "absenceDays": "Absence days",
                    "absenceHours": "Absence hours",
                    "acceptedEventCount": "Accepted events",
                    "pendingReviewCount": "Pending reviews",
                    "hourlyRateBasis": "Hourly-rate basis",
                    "exportReady": "Export ready",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        export_buffer = io.StringIO()
        export_df.to_csv(export_buffer, index=False)
        st.download_button(
            "Download payroll-prep summary CSV",
            export_buffer.getvalue(),
            file_name="payroll_labor_export_summary.csv",
            mime="text/csv",
            key="payroll_labor_export_summary_csv",
        )

    st.markdown("#### Hours / Cost Distribution")
    distribution_df = pd.DataFrame(distribution_rows)
    if distribution_df.empty:
        st.caption("No imported labor-cost rows are available for the selected range.")
    else:
        dist_cols = st.columns(2)
        hours_fig = px.histogram(
            distribution_df,
            x="hours",
            nbins=20,
            title="Imported shift hours distribution",
            labels={"hours": "Hours"},
        )
        cost_fig = px.histogram(
            distribution_df,
            x="costTotal",
            nbins=20,
            title="Imported labor cost distribution",
            labels={"costTotal": "Cost"},
        )
        dist_cols[0].plotly_chart(hours_fig, width="stretch")
        dist_cols[1].plotly_chart(cost_fig, width="stretch")

    st.markdown("#### Overtime Distribution")
    overtime_df = pd.DataFrame(overtime_rows)
    if overtime_df.empty:
        st.caption("No imported shift rows are available to evaluate overtime in the selected range.")
    else:
        overtime_worker_df = (
            overtime_df.groupby("workerName", dropna=False)[["overtimeHours", "totalHours", "totalCost"]]
            .sum()
            .reset_index()
            .sort_values(["overtimeHours", "totalHours"], ascending=[False, False], kind="stable")
        )
        st.caption(
            f"V1 overtime uses a simple daily-hours-above-threshold heuristic ({float(overtime_threshold):.1f} h/day), not award interpretation."
        )
        st.dataframe(
            overtime_worker_df.rename(
                columns={
                    "workerName": "Worker",
                    "overtimeHours": "Overtime hours",
                    "totalHours": "Total hours",
                    "totalCost": "Total cost",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        overtime_chart = px.bar(
            overtime_worker_df.head(10),
            x="workerName",
            y="overtimeHours",
            title="Overtime hours by worker",
            labels={"workerName": "Worker", "overtimeHours": "Overtime hours"},
        )
        st.plotly_chart(overtime_chart, width="stretch")

    st.markdown("#### Plan vs Actual")
    variance_cols = st.columns(5)
    variance_cols[0].metric("Planned only", int(plan_vs_actual["plannedOnlyCount"]))
    variance_cols[1].metric("Imported only", int(plan_vs_actual["importedOnlyCount"]))
    variance_cols[2].metric("Matched", int(plan_vs_actual["matchedCount"]))
    variance_cols[3].metric(
        "Accepted matched shifts",
        int(plan_vs_actual["acceptedMatchedShiftCount"]),
    )
    variance_cols[4].metric(
        "Accepted unmatched",
        int(plan_vs_actual["acceptedUnmatchedCount"]),
    )

    st.markdown("#### Confidence / Anomalies")
    confidence_cols = st.columns(4)
    confidence_cols[0].metric("Pending review", int(confidence["pendingReviewCount"]))
    confidence_cols[1].metric("Duplicate events", int(confidence["duplicateEventCount"]))
    confidence_cols[2].metric(
        "Missing prior clock-on",
        int(confidence["missingPriorClockOnCount"]),
    )
    confidence_cols[3].metric(
        "Accepted unmatched events",
        int(confidence["acceptedUnmatchedCount"]),
    )
    st.caption(
        "Confidence reflects worker-time review/anomaly health. Absence is now based on explicit recorded leave/absence rows rather than inferred missing shifts."
    )

    st.markdown("#### Absence / Leave")
    absence_cols = st.columns(4)
    absence_cols[0].metric("Recorded rows", int(absence_summary["recordCount"]))
    absence_cols[1].metric("Confirmed", int(absence_summary["confirmedCount"]))
    absence_cols[2].metric("Planned", int(absence_summary["plannedCount"]))
    absence_cols[3].metric("Sick days", f"{absence_summary['sickDays']:.1f}")
    secondary_absence_cols = st.columns(4)
    secondary_absence_cols[0].metric("Annual leave days", f"{absence_summary['annualLeaveDays']:.1f}")
    secondary_absence_cols[1].metric("Personal leave days", f"{absence_summary['personalLeaveDays']:.1f}")
    secondary_absence_cols[2].metric("Unpaid leave days", f"{absence_summary['unpaidLeaveDays']:.1f}")
    secondary_absence_cols[3].metric("Carer's leave days", f"{absence_summary['carersLeaveDays']:.1f}")

    worker_options = conn.execute("SELECT id, name FROM workers ORDER BY name").fetchall()
    if worker_options:
        with st.expander("Record absence / leave", expanded=False):
            with st.form("payroll_absence_record_form"):
                worker_label_map = {f"{row['name']} ({int(row['id'])})": int(row["id"]) for row in worker_options}
                selected_worker_label = st.selectbox(
                    "Worker",
                    options=list(worker_label_map.keys()),
                    key="payroll_absence_worker",
                )
                absence_form_cols = st.columns(3)
                absence_start_date = absence_form_cols[0].date_input(
                    "Start date",
                    value=min_date,
                    key="payroll_absence_start_date",
                )
                absence_end_date = absence_form_cols[1].date_input(
                    "End date",
                    value=min_date,
                    key="payroll_absence_end_date",
                )
                absence_type = absence_form_cols[2].selectbox(
                    "Type",
                    options=list(ABSENCE_RECORD_TYPES),
                    key="payroll_absence_type",
                )
                absence_meta_cols = st.columns(4)
                absence_status = absence_meta_cols[0].selectbox(
                    "Status",
                    options=list(ABSENCE_RECORD_STATUSES),
                    key="payroll_absence_status",
                )
                absence_hours = float(
                    absence_meta_cols[1].number_input(
                        "Hours per day",
                        min_value=0.0,
                        max_value=24.0,
                        value=8.0,
                        step=0.5,
                        key="payroll_absence_hours_per_day",
                    )
                )
                absence_source = absence_meta_cols[2].text_input(
                    "Source",
                    value="manual_manager",
                    key="payroll_absence_source",
                )
                absence_recorded_by = absence_meta_cols[3].text_input(
                    "Recorded by",
                    value="manager",
                    key="payroll_absence_recorded_by",
                )
                absence_note = st.text_area(
                    "Note",
                    key="payroll_absence_note",
                )
                if st.form_submit_button("Record absence / leave"):
                    try:
                        create_worker_absence_record(
                            conn,
                            worker_id=worker_label_map[selected_worker_label],
                            start_date=absence_start_date.isoformat(),
                            end_date=absence_end_date.isoformat(),
                            absence_type=absence_type,
                            status=absence_status,
                            hours_per_day=absence_hours,
                            note=absence_note.strip() or None,
                            source=absence_source.strip() or None,
                            recorded_by=absence_recorded_by.strip() or None,
                        )
                    except Exception as exc:
                        st.error(f"Failed to record absence / leave: {exc}")
                    else:
                        st.success("Absence / leave record saved.")
                        _trigger_rerun()

    absence_df = pd.DataFrame(absence_rows)
    if absence_df.empty:
        st.caption("No absence / leave rows are recorded for the selected range.")
    else:
        st.dataframe(
            absence_df.rename(
                columns={
                    "workerName": "Worker",
                    "startDate": "Start date",
                    "endDate": "End date",
                    "absenceType": "Type",
                    "status": "Status",
                    "hoursPerDay": "Hours / day",
                    "note": "Note",
                    "source": "Source",
                    "recordedBy": "Recorded by",
                }
            )[
                [
                    "Worker",
                    "Start date",
                    "End date",
                    "Type",
                    "Status",
                    "Hours / day",
                    "Source",
                    "Recorded by",
                    "Note",
                ]
            ],
            width="stretch",
            hide_index=True,
        )

    st.markdown("#### Labor Cost Drivers")
    driver_dimension = st.radio(
        "Cost-driver grouping",
        options=["worker", "client", "corridor", "truck", "job"],
        horizontal=True,
        key="payroll_labor_cost_driver_dimension",
    )
    driver_df = pd.DataFrame(labor_cost_drivers.get(driver_dimension, []))
    if driver_df.empty:
        st.caption("No labor cost-driver rows are available for the selected range.")
    else:
        st.dataframe(
            driver_df.rename(
                columns={
                    "dimensionValue": "Value",
                    "totalHours": "Total hours",
                    "totalCost": "Total cost",
                    "shiftCount": "Shift count",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        driver_chart = px.bar(
            driver_df.head(10),
            x="dimensionValue",
            y="totalCost",
            title=f"Top {driver_dimension} labor cost drivers",
            labels={"dimensionValue": driver_dimension.title(), "totalCost": "Total cost"},
        )
        st.plotly_chart(driver_chart, width="stretch")


def main() -> None:
    """Configure Streamlit and render the price distribution dashboard."""

    st.set_page_config(
        page_title="Price distribution by corridor",
        layout="wide",
    )
    render_price_distribution_dashboard()


if __name__ == "__main__":  # pragma: no cover - manual execution convenience
    main()
