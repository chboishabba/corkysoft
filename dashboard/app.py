"""Streamlit dashboard for the price distribution analysis."""
from __future__ import annotations

import io
import json
import math
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
    PRIMARY_ONLY_ROLE_KEYS,
    ROLE_LAYOUT_DEFAULTS,
    find_layout_by_tab,
    get_dashboard_role_layouts,
    missing_recommended_primary_tabs,
    resolve_dashboard_layout,
    upsert_dashboard_role_layout,
)
from analytics.db import (
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
    list_inventory,
    list_inventory_balances,
    list_inventory_execution_events,
    list_inventory_exceptions,
    list_inventory_movements,
    list_inventory_requirements,
    list_inventory_substitution_reason_codes,
    list_inventory_substitutions,
    list_segment_inventory_coordination,
    request_inventory_substitution,
    record_inventory_execution_event,
    record_inventory_movement,
    resolve_inventory_exception,
    upsert_inventory_requirement,
    upsert_inventory_substitution_reason_code,
)
from analytics.db_connection import connection_scope
from analytics.price_distribution import (
    PROFITABILITY_COLOURS,
    ColumnMapping,
    compute_profitability_line_width,
    compute_tapered_route_polygon,
    available_heatmap_weightings,
    build_isochrone_polygons,
    build_price_history_series,
    filter_routes_by_country,
    build_heatmap_source,
    ensure_break_even_parameter,
    enrich_missing_route_coordinates,
    import_historical_jobs_from_dataframe,
    latest_historical_ingest_summary,
    load_historical_jobs,
    load_quotes,
    load_live_jobs,
    prepare_metric_route_map_data,
    prepare_route_map_data,
    prepare_profitability_map_data,
    update_break_even,
)
from analytics.live_data import (
    TRUCK_STATUS_COLOURS,
    build_live_heatmap_source,
    extract_route_path,
)
from analytics.routes_map import (
    build_job_route_map,
    fetch_job_route_rows,
    populate_route_geometry,
)
from dashboard.components.operations import render_operations_tab
from dashboard.components.dispatch import render_dispatch_tab
from dashboard.components.distribution_overview import render_distribution_analytics_surface
from dashboard.components.inventory import render_inventory_tab
from dashboard.components.kent import render_kent_admin_tab, render_kent_tenders_tab
from dashboard.theme import inject_css
from dashboard.components.operations_diary import render_operations_diary_tab
from dashboard.components.payroll_labor_analytics import render_payroll_labor_analytics_tab
from dashboard.components.planner import render_planner_tab
from dashboard.components.staff import render_staff_tab
from dashboard.components.worker_time import (
    render_driver_shifts_tab,
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
from dashboard.components.maps import _initial_view_state
from dashboard.components.maintenance import render_fleet_tab, render_vehicle_maintenance_tab
from dashboard.components.route_maps import render_route_maps_tab
from dashboard.components.calls import render_calls_tab
from dashboard.components.optimizer import render_optimizer
from dashboard.components.quote_builder import render_quote_builder
from dashboard.data import blank_column_mapping
from dashboard.views.quote_view import render_quote_view
from dashboard.views.pricing_intelligence_view import render_pricing_intelligence_view
from dashboard.views.network_view import render_network_view
from dashboard.views.operations_view import render_operations_view
from dashboard.views.admin_view import render_admin_view
from dashboard.map_provider import (
    folium_map_configuration,
    google_maps_api_key,
    plotly_map_layout,
    pydeck_map_kwargs,
)
from dashboard.auth_ui import (
    _auth_redirect_config_issue,
    _render_anonymous_dev_banner,
    _render_authenticated_user_banner,
    _render_auth_gate,
    _render_dashboard_user_admin,
    _resolve_dashboard_identity,
)
from dashboard.query_params import _get_query_params, _get_workspace_state
from dashboard.state import _ensure_pin_state, _first_non_empty, _rerun_app
from dashboard.data_controls import render_dataset_sidebar
from dashboard.layout_state import (
    LAYOUT_PENDING_KEY as _LAYOUT_PENDING_KEY,
    hydrate_role_layout_session as _hydrate_role_layout_session,
    layout_defaults_from_layout as _layout_defaults_from_layout,
)
from dashboard.shell import (
    ANALYTICS_SHELL_TABS as _ANALYTICS_SHELL_TABS,
    resolve_dashboard_shell as _resolve_dashboard_shell,
)
from dashboard.tab_registry import build_tab_map
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
    "Quote",
    "Pricing Intelligence",
    "Network",
    "Operations",
    "Admin",
]
_QUOTE_COUNTRY_STATE_KEY = "quote_builder_country"

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


def _canonical_role_layout(role_key: str) -> dict[str, Any]:
    base = ROLE_LAYOUT_DEFAULTS[role_key]
    return {
        "roleKey": role_key,
        "label": str(base["label"]),
        "defaultLandingTab": str(base["defaultLandingTab"]),
        "primaryTabs": list(base["primaryTabs"]),
        "hiddenTabs": list(base["hiddenTabs"]),
    }


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
    # Inject the design system CSS
    inject_css()
    # Header placeholder — created FIRST so it renders ABOVE tab content.
    header_placeholder = st.container()
    # Analytics filters and pricing controls remain available on demand.
    tabs_placeholder = st.container()

    with connection_scope() as conn:
        ensure_dashboard_tables(conn)
        try:
            auth_state = _resolve_dashboard_identity(conn)
        except ValueError as exc:
            st.title("Corkysoft")
            st.error(str(exc))
            return
        if auth_state["mode"] != "anonymous" and auth_state["mode"] != "authenticated":
            _render_auth_gate(auth_state)
            return

        tab_labels = PRICE_DASHBOARD_TABS
        role_layouts = get_dashboard_role_layouts(conn, available_tabs=tab_labels)
        params = _get_query_params()
        workspace_state = _get_workspace_state(available_tabs=tab_labels)
        view_param = workspace_state.get("view")
        if view_param not in tab_labels:
            view_param = None
        view_layout = find_layout_by_tab(role_layouts, view_param)
        role_labels = {item["label"]: item for item in role_layouts}
        auth_user = auth_state.get("user")
        auth_role_key = str(auth_user.get("roleKey")) if isinstance(auth_user, dict) and auth_user.get("roleKey") else "dispatcher"
        default_role_label = next(
            (
                item["label"]
                for item in role_layouts
                if item["roleKey"] == auth_role_key
            ),
            next((item["label"] for item in role_layouts if item["roleKey"] == "dispatcher"), role_layouts[0]["label"]),
        )
        shell_role_label = default_role_label
        if auth_state["mode"] == "anonymous":
            shell_role_label = str(st.session_state.get("dashboard_active_role") or default_role_label)
            if view_layout is not None:
                shell_role_label = str(view_layout["label"])
        if shell_role_label not in role_labels:
            shell_role_label = default_role_label
        shell_role_layout = role_labels[shell_role_label]
        shell_tab = view_param or str(
            st.session_state.get("dashboard_session_landing_tab")
            or shell_role_layout["defaultLandingTab"]
        )
        shell_copy = _resolve_dashboard_shell(shell_tab)
        with header_placeholder:
            st.markdown(
                '<div class="ck-section-title" style="font-size:1.6rem;margin-bottom:0.1rem;">'
                'Corkysoft</div>'
                '<div class="ck-section-subtitle" style="margin-bottom:0.6rem;">'
                f'{str(shell_copy["caption"])}</div>',
                unsafe_allow_html=True,
            )

        break_even_value = ensure_break_even_parameter(conn)
        ensure_quote_schema(conn)

        df_all: pd.DataFrame = pd.DataFrame()
        mapping: ColumnMapping = blank_column_mapping()
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

        dataset_context = render_dataset_sidebar(
            conn,
            sidebar_heading="Corkysoft",
            sidebar_caption="Global workflow controls",
            collapse_analytics_sidebar=bool(shell_copy.get("collapse_analytics_sidebar")),
            dataset_loader=dataset_loader,
            dataset_key=dataset_key,
            dataset_label=dataset_label,
            df_all=df_all,
            mapping=mapping,
            break_even_value=break_even_value,
            rerun_app=_rerun_app,
        )
        dataset_loader = dataset_context.dataset_loader
        dataset_key = dataset_context.dataset_key
        dataset_label = dataset_context.dataset_label
        df_all = dataset_context.df_all
        mapping = dataset_context.mapping
        dataset_error = dataset_context.dataset_error
        data_available = dataset_context.data_available
        start_date = dataset_context.start_date
        end_date = dataset_context.end_date
        selected_corridor = dataset_context.selected_corridor
        selected_clients = dataset_context.selected_clients
        postcode_prefix = dataset_context.postcode_prefix
        break_even_value = dataset_context.break_even_value
        empty_dataset_message = dataset_context.empty_dataset_message

        if auth_state["mode"] == "authenticated":
            _render_authenticated_user_banner(auth_state)
        elif auth_state["mode"] == "anonymous":
            _render_anonymous_dev_banner(auth_state)

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

        if auth_state["mode"] == "anonymous" and view_layout is not None:
            default_role_label = view_layout["label"]
            if st.session_state.get("dashboard_active_role") != default_role_label:
                st.session_state["dashboard_active_role"] = default_role_label
        # ── Layout controls moved to sidebar ──────────────────────
        deep_link_locked_role = auth_state["mode"] == "anonymous" and view_layout is not None
        with st.sidebar:
            with st.expander("Layout preferences", expanded=False, icon="⚙️"):
                if auth_state["mode"] == "authenticated" or deep_link_locked_role:
                    selected_role_layout = (
                        view_layout
                        if deep_link_locked_role and view_layout is not None
                        else next(
                            (item for item in role_layouts if item["roleKey"] == auth_role_key),
                            role_layouts[0],
                        )
                    )
                    selected_role_layout = next(
                        (
                            item
                            for item in role_layouts
                            if item["roleKey"] == str(selected_role_layout["roleKey"])
                        ),
                        selected_role_layout,
                    )
                    st.text_input(
                        "Role layout",
                        value=str(selected_role_layout["label"]),
                        disabled=True,
                        key="dashboard_active_role_locked" if auth_state["mode"] == "authenticated" else "dashboard_active_role_deep_linked",
                    )
                else:
                    active_role_kwargs: dict[str, Any] = {
                        "options": list(role_labels.keys()),
                        "key": "dashboard_active_role",
                    }
                    if "dashboard_active_role" not in st.session_state:
                        active_role_kwargs["index"] = (
                            list(role_labels.keys()).index(default_role_label)
                            if default_role_label in role_labels
                            else 0
                        )
                    selected_role_label = st.selectbox(
                        "Role layout",
                        **active_role_kwargs,
                    )
                    selected_role_layout = role_labels[selected_role_label]
                if (
                    auth_state["mode"] == "anonymous"
                    and view_param is not None
                    and str(selected_role_layout["roleKey"]) in PRIMARY_ONLY_ROLE_KEYS
                ):
                    selected_role_layout = _canonical_role_layout(str(selected_role_layout["roleKey"]))
                _hydrate_role_layout_session(
                    selected_role_layout,
                    force_reset=auth_state["mode"] == "anonymous" and view_param is not None,
                )
                stale_primary_tabs = missing_recommended_primary_tabs(
                    role_key=str(selected_role_layout["roleKey"]),
                    layout=selected_role_layout,
                    available_tabs=tab_labels,
                )
                session_primary_tabs_kwargs: dict[str, Any] = {
                    "options": tab_labels,
                    "key": "dashboard_session_primary_tabs",
                }
                if "dashboard_session_primary_tabs" not in st.session_state:
                    session_primary_tabs_kwargs["default"] = list(selected_role_layout["primaryTabs"])
                session_primary_tabs = st.multiselect(
                    "Session focus tabs",
                    **session_primary_tabs_kwargs,
                )
                session_show_all_kwargs: dict[str, Any] = {
                    "key": "dashboard_show_all_tabs",
                }
                if "dashboard_show_all_tabs" not in st.session_state:
                    session_show_all_kwargs["value"] = False
                session_show_all = st.checkbox(
                    "Show all tabs this session",
                    **session_show_all_kwargs,
                )
                if st.button("Reset layout", key="dashboard_reset_role_layout"):
                    st.session_state[_LAYOUT_PENDING_KEY] = _layout_defaults_from_layout(selected_role_layout)
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
                        st.session_state[_LAYOUT_PENDING_KEY] = _layout_defaults_from_layout(repaired)
                        _rerun_app()

        params = _get_query_params()
        requested_tab = params.get("view", [None])[0]
        active_tab_index = st.session_state.get("dashboard_active_tab")
        if requested_tab not in tab_labels:
            if isinstance(active_tab_index, int) and 0 <= active_tab_index < len(tab_labels):
                requested_tab = tab_labels[active_tab_index]
            else:
                requested_tab = tab_labels[0]
        resolved_show_all_tabs = bool(st.session_state.get("dashboard_show_all_tabs", False))
        if auth_state["mode"] == "anonymous" and view_param is not None:
            resolved_show_all_tabs = False
        requested_tab_from_session = "view" not in params and isinstance(active_tab_index, int)
        resolved_layout = resolve_dashboard_layout(
            available_tabs=tab_labels,
            layout=selected_role_layout,
            requested_tab=requested_tab if "view" in params or requested_tab_from_session else None,
            session_primary_tabs=st.session_state.get("dashboard_session_primary_tabs", selected_role_layout["primaryTabs"]),
            session_hidden_tabs=st.session_state.get("dashboard_session_hidden_tabs", selected_role_layout["hiddenTabs"]),
            session_landing_tab=st.session_state.get(
                "dashboard_session_landing_tab",
                requested_tab if requested_tab_from_session else selected_role_layout["defaultLandingTab"],
            ),
            show_all_tabs=resolved_show_all_tabs,
        )
        tab_labels = resolved_layout["tabOrder"]
        requested_tab = resolved_layout["landingTab"]

        tab_result = build_tab_map(
            tab_labels=tab_labels,
            requested_tab=requested_tab,
            params=params,
            tabs_placeholder=tabs_placeholder,
        )
        tab_map = tab_result.tab_map
        requested_tab = tab_result.requested_tab
        requested_tab_index = tab_result.requested_tab_index
        tab_order = tab_result.tab_order
        show_analytics_overview = requested_tab in _ANALYTICS_SHELL_TABS
        # Views handle overview display internally; retained for test compat:
        # show_overview=show_analytics_overview

        if "Quote" in tab_map:
            with tab_map["Quote"]:
                render_quote_view(
                    conn=conn,
                    filtered_df=filtered_df,
                    mapping=filtered_mapping,
                    state=st.session_state,
                    rerun_app=_rerun_app
                )

        if "Pricing Intelligence" in tab_map:
            with tab_map["Pricing Intelligence"]:
                render_pricing_intelligence_view(
                    conn=conn,
                    filtered_df=filtered_df,
                    mapping=filtered_mapping,
                    break_even_value=break_even_value,
                    start_date=start_date,
                    end_date=end_date,
                    dataset_error=dataset_error
                )

        if "Network" in tab_map:
            with tab_map["Network"]:
                render_network_view(
                    conn=conn,
                    filtered_df=filtered_df,
                    mapping=filtered_mapping,
                    break_even_value=break_even_value,
                    dataset_key=dataset_key,
                    dataset_error=dataset_error
                )

        if "Operations" in tab_map:
            with tab_map["Operations"]:
                render_operations_view(
                    conn=conn,
                    filtered_df=filtered_df,
                    rerun_app=_rerun_app,
                    workspace_state=workspace_state,
                )

        if "Admin" in tab_map:
            with tab_map["Admin"]:
                render_admin_view(
                    conn=conn,
                    auth_role_key=auth_role_key if auth_state["mode"] == "authenticated" else None,
                    rerun_app=_rerun_app,
                    render_dashboard_user_admin=_render_dashboard_user_admin
                )


def main() -> None:
    """Configure Streamlit and render the price distribution dashboard."""

    st.set_page_config(
        page_title="Corkysoft",
        layout="wide",
    )
    render_price_distribution_dashboard()


if __name__ == "__main__":  # pragma: no cover - manual execution convenience
    main()
