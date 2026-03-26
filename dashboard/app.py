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
    INVENTORY_ARCHITECTURES,
    INVENTORY_CUSTODY_TYPES,
    INVENTORY_EXECUTION_STAGES,
    INVENTORY_STATES,
    INVENTORY_SUBSTITUTION_APPROVER_ROLES,
    INVENTORY_SUBSTITUTION_STATUSES,
    allocate_inventory_to_segment,
    bootstrap_dashboard_admin,
    decide_inventory_substitution,
    ensure_dashboard_tables,
    get_allowed_inventory_execution_stages,
    get_dashboard_user_by_email,
    import_inventory_items_from_dataframe,
    import_inventory_movements_from_dataframe,
    import_suppliers_from_google_sheet,
    import_workers_from_google_sheet,
    import_workers_from_staff_sheet,
    list_dashboard_users,
    list_inventory,
    list_inventory_balances,
    list_inventory_execution_events,
    list_inventory_exceptions,
    list_inventory_movements,
    list_inventory_requirements,
    list_inventory_substitution_reason_codes,
    list_inventory_substitutions,
    list_segment_inventory_coordination,
    normalize_user_email,
    request_inventory_substitution,
    record_dashboard_user_login,
    record_inventory_execution_event,
    record_inventory_movement,
    resolve_ui_auth_policy,
    resolve_inventory_exception,
    upsert_dashboard_user,
    upsert_inventory_requirement,
    upsert_inventory_substitution_reason_code,
    upsert_worker,
)
from analytics.db_connection import connection_scope
from analytics.driver_shifts import load_driver_shifts_dataframe
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
from analytics.kent_ams_import import (
    get_kent_tender_policy_config,
    list_kent_override_reason_codes,
    list_kent_tender_override_history,
    list_prioritized_tenders,
    record_kent_tender_override,
    update_kent_tender_policy_config,
    upsert_kent_override_reason_code,
)
from analytics.operations_assignment import (
    assign_worker_compliance,
    assign_worker_role,
    ensure_worker_compliance,
    ensure_worker_role,
    list_operational_readiness_items,
    list_segments_for_worker,
    list_worker_assignment_summary,
)
from dashboard.components.operations import render_operations_tab
from dashboard.components.dispatch import render_dispatch_tab
from dashboard.components.distribution_overview import render_distribution_analytics_surface
from dashboard.components.operations_diary import render_operations_diary_tab
from dashboard.components.payroll_labor_analytics import render_payroll_labor_analytics_tab
from dashboard.components.planner import render_planner_tab
from dashboard.components.worker_time import (
    render_driver_shifts_tab,
    render_worker_time_review_controls,
    worker_time_events_df,
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
from dashboard.map_provider import (
    folium_map_configuration,
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
PRICE_DASHBOARD_TABS = [
    "Histogram",
    "Price history",
    "Profitability insights",
    "Live network overview",
    "Route maps",
    "Dispatch",
    "Operations diary",
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


def _streamlit_auth_configured() -> bool:
    user_obj = getattr(st, "user", None)
    return getattr(user_obj, "is_logged_in", None) is not None


def _streamlit_user_claims() -> Dict[str, Any]:
    user_obj = getattr(st, "user", None)
    if user_obj is None:
        return {}
    claims: Dict[str, Any] = {}
    for key in ("email", "name", "sub"):
        value = getattr(user_obj, key, None)
        if isinstance(value, str) and value.strip():
            claims[key] = value.strip()
    claims["is_logged_in"] = bool(getattr(user_obj, "is_logged_in", False))
    return claims


def _resolve_dashboard_identity(
    conn: sqlite3.Connection,
) -> dict[str, Any]:
    policy = resolve_ui_auth_policy()
    bootstrap_dashboard_admin(conn, allowed_role_keys=tuple(ROLE_LAYOUT_DEFAULTS.keys()))

    if not policy["requireAuth"]:
        return {
            "mode": "anonymous",
            "policy": policy,
            "user": None,
            "claims": {},
            "configured": _streamlit_auth_configured(),
        }

    configured = _streamlit_auth_configured()
    claims = _streamlit_user_claims()
    if not configured:
        return {
            "mode": "misconfigured",
            "policy": policy,
            "user": None,
            "claims": claims,
            "configured": False,
        }

    if not claims.get("is_logged_in"):
        return {
            "mode": "login_required",
            "policy": policy,
            "user": None,
            "claims": claims,
            "configured": True,
        }

    email = normalize_user_email(claims.get("email"))
    local_user = get_dashboard_user_by_email(conn, email=email)
    if local_user is None:
        return {
            "mode": "unauthorized",
            "policy": policy,
            "user": None,
            "claims": claims,
            "configured": True,
        }
    if not local_user["active"]:
        return {
            "mode": "inactive",
            "policy": policy,
            "user": local_user,
            "claims": claims,
            "configured": True,
        }

    refreshed_user = record_dashboard_user_login(
        conn,
        email=email,
        google_sub=claims.get("sub"),
        display_name=claims.get("name"),
    ) or local_user
    return {
        "mode": "authenticated",
        "policy": policy,
        "user": refreshed_user,
        "claims": claims,
        "configured": True,
    }


def _render_auth_gate(auth_state: dict[str, Any]) -> None:
    st.title("Corkysoft")
    st.caption("Private dashboard access is controlled by Google sign-in plus a local Corkysoft allowlist.")

    mode = str(auth_state["mode"])
    if mode == "misconfigured":
        st.error(
            "UI auth is required but Streamlit OIDC is not configured. Add `.streamlit/secrets.toml` auth settings before starting this deployment."
        )
        st.stop()

    if mode == "login_required":
        st.info("Sign in with Google to continue.")
        login = getattr(st, "login", None)
        if not callable(login):
            st.error("Streamlit login support is unavailable in this runtime.")
        elif st.button("Sign in with Google", key="dashboard_auth_login_google"):
            login("google")
        st.stop()

    claims = auth_state.get("claims", {})
    email = claims.get("email") or "Unknown account"
    if mode == "unauthorized":
        st.error(f"`{email}` is not in the local Corkysoft allowlist.")
    elif mode == "inactive":
        st.error(f"`{email}` is currently inactive in Corkysoft.")
    else:
        st.error("Authentication failed.")

    logout = getattr(st, "logout", None)
    if callable(logout) and st.button("Sign out", key="dashboard_auth_logout_gate"):
        logout()
    st.stop()


def _render_authenticated_user_banner(auth_state: dict[str, Any]) -> None:
    user = auth_state.get("user") or {}
    claims = auth_state.get("claims") or {}
    display_name = user.get("displayName") or claims.get("name") or user.get("email") or "Unknown user"
    email = user.get("email") or claims.get("email") or ""
    role_key = user.get("roleKey") or "dispatcher"
    role_label = ROLE_LAYOUT_DEFAULTS.get(role_key, {}).get("label", role_key)

    st.success(
        f"Authenticated via Google as {display_name} ({email}) · role: {role_label}"
    )
    banner_cols = st.columns([3, 2, 1])
    banner_cols[0].caption(f"Google account: **{display_name}**")
    banner_cols[1].caption(f"{email} · {role_label}")
    logout = getattr(st, "logout", None)
    if callable(logout) and banner_cols[2].button("Log out", key="dashboard_auth_logout_button"):
        logout()


def _render_anonymous_dev_banner(auth_state: dict[str, Any]) -> None:
    policy = auth_state.get("policy") or {}
    environment = str(policy.get("environment") or "development")
    st.warning(
        "Anonymous development mode is active. Google sign-in is bypassed for this local run."
    )
    st.caption(
        f"Mode: anonymous local development · environment: {environment} · set `CORKYSOFT_REQUIRE_UI_AUTH=1` and unset `CORKYSOFT_ALLOW_ANONYMOUS_UI` to force login."
    )


def _render_dashboard_user_admin(
    conn: sqlite3.Connection,
    *,
    current_role_key: str,
) -> None:
    if current_role_key != "system_rollout_admin":
        return

    st.markdown("#### Dashboard users")
    users = list_dashboard_users(conn)
    if users:
        users_df = pd.DataFrame(
            [
                {
                    "Email": item["email"],
                    "Name": item["displayName"] or "",
                    "Role": item["roleKey"],
                    "Active": item["active"],
                    "Provider": item["authProvider"],
                    "Google sub": item["googleSub"] or "",
                    "Last login": item["lastLoginAt"] or "",
                }
                for item in users
            ]
        )
        st.dataframe(users_df, width="stretch", hide_index=True)
    else:
        st.caption("No local dashboard users have been created yet.")

    with st.form("dashboard_user_admin_form"):
        user_cols = st.columns(4)
        email = user_cols[0].text_input("Email")
        display_name = user_cols[1].text_input("Name")
        role_key = user_cols[2].selectbox(
            "Role",
            options=list(ROLE_LAYOUT_DEFAULTS.keys()),
            format_func=lambda key: str(ROLE_LAYOUT_DEFAULTS[key]["label"]),
        )
        active = user_cols[3].checkbox("Active", value=True)
        if st.form_submit_button("Save dashboard user"):
            try:
                upsert_dashboard_user(
                    conn,
                    email=email,
                    display_name=display_name or None,
                    role_key=role_key,
                    active=active,
                    allowed_role_keys=tuple(ROLE_LAYOUT_DEFAULTS.keys()),
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.success("Dashboard user saved.")
                _rerun_app()


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

        st.title("Price distribution (Airbnb-style)")
        st.caption(
            "Visualise $ per m³ by corridor and client, with break-even bands to spot loss-leaders."
        )

        if auth_state["mode"] == "authenticated":
            _render_authenticated_user_banner(auth_state)
        elif auth_state["mode"] == "anonymous":
            _render_anonymous_dev_banner(auth_state)

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
                    ingest_summary = latest_historical_ingest_summary(conn)
                    if ingest_summary is not None:
                        ingest_cols = st.columns(4)
                        ingest_cols[0].metric(
                            "Readiness",
                            str(ingest_summary.get("readiness_status") or "unknown"),
                        )
                        ingest_cols[1].metric(
                            "Inserted",
                            int(ingest_summary.get("inserted_rows") or 0),
                        )
                        ingest_cols[2].metric(
                            "Skipped",
                            int(ingest_summary.get("skipped_rows") or 0),
                        )
                        ingest_cols[3].metric(
                            "Issues",
                            int(ingest_summary.get("issue_count") or 0),
                        )
                        st.caption(
                            "Latest ingest: "
                            + str(ingest_summary.get("source_name") or "unknown source")
                            + " at "
                            + str(ingest_summary.get("completed_at") or "unknown time")
                        )
                        coverage = ingest_summary.get("coverage_summary") or {}
                        top_issues = coverage.get("topIssueCodes") or []
                        if top_issues:
                            st.caption(
                                "Top issues: "
                                + ", ".join(
                                    f"{item['issueCode']} ({item['count']})"
                                    for item in top_issues
                                )
                            )
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
        layout_cols = st.columns([2, 2, 2, 1])
        if auth_state["mode"] == "authenticated":
            selected_role_layout = next(
                (item for item in role_layouts if item["roleKey"] == auth_role_key),
                role_layouts[0],
            )
            layout_cols[0].text_input(
                "Role layout",
                value=str(selected_role_layout["label"]),
                disabled=True,
                key="dashboard_active_role_locked",
            )
        else:
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

        render_distribution_analytics_surface(
            tab_map=tab_map,
            filtered_df=filtered_df,
            filtered_mapping=filtered_mapping,
            break_even_value=break_even_value,
            dataset_error=dataset_error,
            conn=conn,
            start_date=start_date,
            end_date=end_date,
        )

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

        if "Operations diary" in tab_map:
            with tab_map["Operations diary"]:
                render_operations_diary_tab(conn)

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
                render_quote_builder(
                    filtered_df=filtered_df,
                    mapping=filtered_mapping,
                    conn=conn,
                    state=state,
                )
        if "Calls" in tab_map:
            with tab_map["Calls"]:
                render_calls_tab(conn)

        if "Kent tenders" in tab_map:
            with tab_map["Kent tenders"]:
                render_kent_tenders_tab(conn)

        if "Kent admin" in tab_map:
            with tab_map["Kent admin"]:
                render_kent_admin_tab(
                    conn,
                    current_role_key=auth_role_key if auth_state["mode"] == "authenticated" else None,
                )

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
                render_driver_shifts_tab(conn, rerun_app=_rerun_app)
        if "Payroll / Labor analytics" in tab_map:
            with tab_map["Payroll / Labor analytics"]:
                render_payroll_labor_analytics_tab(conn, rerun_app=_rerun_app)

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
    triage_filters = st.columns(3)
    hard_block_scope = triage_filters[0].selectbox(
        "Hard-block scope",
        options=["all", "hide_hard_blocked", "only_hard_blocked"],
        index=0,
        key="kent_tender_hard_block_scope",
    )
    policy_scope = triage_filters[1].selectbox(
        "Policy scope",
        options=["all", "policy_fail_only", "policy_matched_only"],
        index=0,
        key="kent_tender_policy_scope",
    )
    loss_scope = triage_filters[2].selectbox(
        "Loss scope",
        options=["all", "loss_alert_only", "hide_loss_alerts"],
        index=0,
        key="kent_tender_loss_scope",
    )

    rows = list_prioritized_tenders(conn, status=status_filter, limit=limit_value)
    if hard_block_scope == "hide_hard_blocked":
        rows = [row for row in rows if not row["hardBlockFlags"]]
    elif hard_block_scope == "only_hard_blocked":
        rows = [row for row in rows if row["hardBlockFlags"]]
    if policy_scope == "policy_fail_only":
        rows = [row for row in rows if not row["policyMatched"]]
    elif policy_scope == "policy_matched_only":
        rows = [row for row in rows if row["policyMatched"]]
    if loss_scope == "loss_alert_only":
        rows = [row for row in rows if row["lossAlert"]]
    elif loss_scope == "hide_loss_alerts":
        rows = [row for row in rows if not row["lossAlert"]]
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


def render_kent_admin_tab(
    conn: sqlite3.Connection,
    *,
    current_role_key: str | None = None,
) -> None:
    st.subheader("Kent AMS admin")
    st.caption(
        "Use this surface for policy defaults, override reason governance, and review. Operators should work from the Kent tenders tab."
    )

    if current_role_key:
        _render_dashboard_user_admin(conn, current_role_key=current_role_key)

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

    review_rows = list_prioritized_tenders(conn, status="all", limit=100)
    review_summary = {
        "hardBlocked": sum(1 for row in review_rows if row["hardBlockFlags"]),
        "policyFail": sum(1 for row in review_rows if not row["policyMatched"]),
        "lossAlert": sum(1 for row in review_rows if row["lossAlert"]),
        "overrideable": sum(1 for row in review_rows if row["overrideableFlags"]),
    }
    review_cols = st.columns(4)
    review_cols[0].metric("Hard blocked", review_summary["hardBlocked"])
    review_cols[1].metric("Policy fail", review_summary["policyFail"])
    review_cols[2].metric("Loss alerts", review_summary["lossAlert"])
    review_cols[3].metric("Overrideable", review_summary["overrideable"])

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
    recent_tenders = list_prioritized_tenders(conn, status="all", limit=10)
    recent_override_rows: list[dict[str, Any]] = []
    for tender_row in recent_tenders:
        history = list_kent_tender_override_history(
            conn, tender_external_id=tender_row["tenderExternalId"]
        )
        for item in history[:3]:
            recent_override_rows.append(
                {
                    "Tender": tender_row["tenderExternalId"],
                    "At": item["createdAt"],
                    "Action": item["action"],
                    "Operator": item["operatorId"],
                    "Reason": item["reasonCode"],
                    "Note": item["note"],
                    "Policy matched": item["policyMatched"],
                    "Loss alert": item["lossAlert"],
                }
            )
    if recent_override_rows:
        st.markdown("#### Recent override history")
        st.dataframe(
            pd.DataFrame(recent_override_rows).sort_values("At", ascending=False),
            width='stretch',
            hide_index=True,
        )


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
        worker_time_df = worker_time_events_df(conn, limit=500)
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
                render_worker_time_review_controls(
                    conn,
                    pending_events=pending_worker_time,
                    key_prefix=f"staff_worker_time_{int(worker_row['id'])}",
                    rerun_app=_rerun_app,
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


def main() -> None:
    """Configure Streamlit and render the price distribution dashboard."""

    st.set_page_config(
        page_title="Price distribution by corridor",
        layout="wide",
    )
    render_price_distribution_dashboard()


if __name__ == "__main__":  # pragma: no cover - manual execution convenience
    main()
