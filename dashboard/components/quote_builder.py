"""Streamlit quote builder component."""

from __future__ import annotations

import math
from datetime import date
from typing import Any, Dict, List, MutableMapping, Optional, Sequence, Tuple

import pandas as pd
import pydeck as pdk
import streamlit as st

try:
    import folium
    from streamlit_folium import st_folium
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    folium = None  # type: ignore[assignment]
    st_folium = None  # type: ignore[assignment]

from analytics.price_distribution import (
    ColumnMapping,
    enrich_missing_route_coordinates,
    filter_routes_by_country,
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
    find_client_matches,
    format_client_display,
    persist_quote,
)
from corkysoft.routing import snap_coordinates_to_road

DEFAULT_TARGET_MARGIN_PERCENT = 20.0
_AUS_LAT_LON = (-25.2744, 133.7751)
_PIN_NOTE = "Manual pin override used for routing"
_HAVERSINE_MODAL_STATE_KEY = "quote_haversine_modal_ack"
_NULL_CLIENT_MODAL_STATE_KEY = "quote_null_client_modal_open"
_NULL_CLIENT_COMPANY_KEY = "quote_null_client_company"
_NULL_CLIENT_NOTES_KEY = "quote_null_client_notes"
_NULL_CLIENT_DEFAULT_COMPANY = "Null (filler) client"
_NULL_CLIENT_DEFAULT_NOTES = "Placeholder client captured via quote builder."
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


def _set_query_params(**params: str) -> None:
    """Set Streamlit query parameters using the stable API when available."""
    query_params = getattr(st, "query_params", None)
    if query_params is not None:
        query_params.from_dict(params)
        return
    # Fallback for older Streamlit versions.
    st.experimental_set_query_params(**params)


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


def render_quote_builder(
    filtered_df: pd.DataFrame,
    mapping: ColumnMapping,
    conn: Any,
    state: MutableMapping[str, Any],
) -> Dict[str, Any]:
    """Render the quote builder tab and return session state hints."""

    before_keys = set(st.session_state.keys())
    before_snapshot = {key: st.session_state.get(key) for key in before_keys}

    filtered_mapping = mapping

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

    after_keys = set(st.session_state.keys())
    after_snapshot = {key: st.session_state.get(key) for key in after_keys}

    updated: Dict[str, Any] = {}
    for key in after_keys:
        if key not in before_keys:
            updated[key] = after_snapshot[key]
            continue
        before_value = before_snapshot.get(key)
        after_value = after_snapshot[key]
        changed = False
        if before_value is not after_value:
            try:
                changed = before_value != after_value
            except Exception:
                changed = True
        if changed:
            updated[key] = after_value

    cleared = sorted(key for key in before_keys if key not in after_keys)

    if state is not st.session_state:
        for key in cleared:
            state.pop(key, None)
        for key, value in after_snapshot.items():
            state[key] = value

    return {
        "updated": updated,
        "cleared": cleared,
    }
