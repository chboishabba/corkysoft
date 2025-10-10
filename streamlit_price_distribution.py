"""Streamlit dashboard for the price distribution analysis."""
from __future__ import annotations

import io
import math
from datetime import date
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import pydeck as pdk
import streamlit as st

from analytics.db import connection_scope
from analytics.price_distribution import (
    DistributionSummary,
    ProfitabilitySummary,
    create_histogram,
    create_m3_margin_figure,
    create_m3_vs_km_figure,
    ensure_break_even_parameter,
    load_historical_jobs,
    prepare_route_map_data,
    PROFITABILITY_COLOURS,
    summarise_distribution,
    summarise_profitability,
    update_break_even,
)
from analytics.live_data import (
    TRUCK_STATUS_COLOURS,
    load_active_routes,
    load_truck_positions,
from corkysoft.quote_service import (
    COUNTRY_DEFAULT,
    DEFAULT_MODIFIERS,
    QuoteInput,
    QuoteResult,
    calculate_quote,
    ensure_schema as ensure_quote_schema,
    format_currency,
    persist_quote,
)

st.set_page_config(
    page_title="Price distribution by corridor",
    layout="wide",
)

st.title("Price distribution (Airbnb-style)")
st.caption("Visualise $ per m³ by corridor and client, with break-even bands to spot loss-leaders.")


def render_summary(
    summary: DistributionSummary,
    break_even: float,
    profitability_summary: ProfitabilitySummary,
) -> None:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Jobs in filter", summary.job_count)
    valid_label = f"Valid $/m³ ({summary.priced_job_count})"
    col2.metric(valid_label, f"{summary.median:,.2f}" if summary.priced_job_count else "n/a")
    col3.metric("25th percentile", f"{summary.percentile_25:,.2f}" if summary.priced_job_count else "n/a")
    col4.metric("75th percentile", f"{summary.percentile_75:,.2f}" if summary.priced_job_count else "n/a")
    below_pct = summary.below_break_even_ratio * 100 if summary.priced_job_count else 0.0
    col5.metric(
        "% below break-even",
        f"{below_pct:.1f}%",
        help=f"Break-even: ${break_even:,.2f} per m³",
    )

    def _format_value(value: Optional[float], *, currency: bool = False, percentage: bool = False) -> str:
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
        column.metric(label, _format_value(value, currency=as_currency, percentage=as_percentage))

    profitability_cols = st.columns(4)
    profitability_metrics = [
        ("Median $/km", profitability_summary.revenue_per_km_median, True, False),
        ("Average $/km", profitability_summary.revenue_per_km_mean, True, False),
        ("Median margin $/m³", profitability_summary.margin_per_m3_median, True, False),
        ("Median margin %", profitability_summary.margin_per_m3_pct_median, False, True),
    ]
    for column, (label, value, as_currency, as_percentage) in zip(profitability_cols, profitability_metrics):
        column.metric(label, _format_value(value, currency=as_currency, percentage=as_percentage))


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
) -> None:
    st.markdown("### Live network overview")

    if historical_routes.empty and trucks.empty:
        st.info("No geocoded historical jobs or live telemetry available to plot yet.")
        return

    truck_data = trucks.copy()
    if not truck_data.empty:
        truck_data["colour"] = truck_data["status"].map(TRUCK_STATUS_COLOURS)
        truck_data["colour"] = truck_data["colour"].apply(
            lambda value: value if isinstance(value, (list, tuple)) else [0, 122, 204]
        )
        truck_data["tooltip"] = truck_data.apply(
            lambda row: f"{row['truck_id']} ({row['status']})", axis=1
        )

    layers: list[pdk.Layer] = []

    if not historical_routes.empty:
        base_layer = pdk.Layer(
            "LineLayer",
            data=historical_routes,
            get_source_position="[origin_lon, origin_lat]",
            get_target_position="[dest_lon, dest_lat]",
            get_color="colour",
            get_width="line_width",
            pickable=True,
            opacity=0.4,
        )
        layers.append(base_layer)

    if not active_routes.empty:
        if "job_id" in active_routes.columns and not historical_routes.empty:
            enriched = active_routes.merge(
                historical_routes[["id", "colour", "profit_band", "tooltip"]],
                left_on="job_id",
                right_on="id",
                how="left",
                suffixes=("", "_hist"),
            )
            if "colour" not in enriched.columns and "colour_hist" in enriched.columns:
                enriched["colour"] = enriched["colour_hist"]
            if "profit_band" not in enriched.columns and "profit_band_hist" in enriched.columns:
                enriched["profit_band"] = enriched["profit_band_hist"]
            if "tooltip" not in enriched.columns and "tooltip_hist" in enriched.columns:
                enriched["tooltip"] = enriched["tooltip_hist"]
        else:
            enriched = active_routes.copy()
            enriched["colour"] = [PROFITABILITY_COLOURS["Unknown"]] * len(enriched)
            enriched["profit_band"] = "Unknown"
            enriched["tooltip"] = "Active route"

        enriched["colour"] = enriched["colour"].apply(
            lambda value: value if isinstance(value, (list, tuple)) else [255, 255, 255]
        )
        enriched["tooltip"] = enriched.apply(
            lambda row: f"Truck {row['truck_id']} ({row.get('profit_band', 'Unknown')})", axis=1
        )

        active_layer = pdk.Layer(
            "LineLayer",
            data=enriched,
            get_source_position="[origin_lon, origin_lat]",
            get_target_position="[dest_lon, dest_lat]",
            get_color="colour",
            get_width=250,
            pickable=True,
            opacity=0.9,
        )
        layers.append(active_layer)

    if not truck_data.empty:
        trucks_layer = pdk.Layer(
            "ScatterplotLayer",
            data=truck_data,
            get_position="[lon, lat]",
            get_fill_color="colour",
            get_radius=800,
            pickable=True,
        )
        layers.append(trucks_layer)

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
        layers.append(text_layer)

    view_df_candidates: list[pd.DataFrame] = []
    if not historical_routes.empty:
        view_df_candidates.append(
            historical_routes.rename(columns={"origin_lat": "lat", "origin_lon": "lon"})[["lat", "lon"]]
        )
    if not truck_data.empty:
        view_df_candidates.append(truck_data[["lat", "lon"]])
    view_df = pd.concat(view_df_candidates) if view_df_candidates else pd.DataFrame()

    tooltip = {"html": "<b>{tooltip}</b>", "style": {"color": "white"}}

    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=_initial_view_state(view_df),
            tooltip=tooltip,
            map_style="mapbox://styles/mapbox/light-v9",
        )
    )

    legend_cols = st.columns(len(PROFITABILITY_COLOURS))
    for (band, colour), column in zip(PROFITABILITY_COLOURS.items(), legend_cols):
        colour_hex = "#" + "".join(f"{int(c):02x}" for c in colour)
        column.markdown(
            f"<div style='color:{colour_hex}; font-weight:bold'>{band}</div>",
            unsafe_allow_html=True,
        )

def _activate_quote_tab() -> None:
    """Switch the interface to the quote builder tab."""

    st.experimental_set_query_params(view="Quote builder")
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
            except Exception:  # pragma: no cover - defensive parsing
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

    with st.sidebar:
        st.header("Filters")
        try:
            df_all, mapping = load_historical_jobs(conn)
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()

        if df_all.empty:
            st.info("historical_jobs table has no rows yet. Import historical jobs to populate the view.")
            st.stop()

        date_column = "job_date" if "job_date" in df_all.columns else mapping.date
        if date_column and date_column in df_all.columns:
            df_all[date_column] = pd.to_datetime(df_all[date_column], errors="coerce")
            min_date = df_all[date_column].min()
            max_date = df_all[date_column].max()
            default_start = min_date.date() if isinstance(min_date, pd.Timestamp) else date.today()
            default_end = max_date.date() if isinstance(max_date, pd.Timestamp) else date.today()
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
            start_date = end_date = None

        corridor_options = sorted(df_all["corridor_display"].dropna().unique())
        selected_corridor: Optional[str] = st.selectbox(
            "Corridor",
            options=["All corridors"] + corridor_options,
            index=0,
        )
        if selected_corridor == "All corridors":
            selected_corridor = None

        client_options = sorted(df_all["client_display"].dropna().unique())
        selected_clients = st.multiselect("Client", options=client_options, default=client_options)

        postcode_prefix = st.text_input(
            "Corridor contains postcode prefix",
            help="Match origin or destination postcode prefixes (e.g. 40 to match 4000-4099).",
        )

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

    filtered_df, filtered_mapping = load_historical_jobs(
        conn,
        start_date=start_date,
        end_date=end_date,
        clients=selected_clients or None,
        corridor=selected_corridor,
        postcode_prefix=postcode_prefix,
    )

    if filtered_df.empty:
        st.warning("No jobs match the selected filters.")
        st.stop()

    summary_tab, map_tab = st.tabs(["Profitability", "Map"])

    with summary_tab:
        summary = summarise_distribution(filtered_df, break_even_value)
        profitability_summary = summarise_profitability(filtered_df)
        render_summary(summary, break_even_value, profitability_summary)

    truck_positions = load_truck_positions(conn)
    active_routes = load_active_routes(conn)
    map_routes = prepare_route_map_data(filtered_df, break_even_value)

    render_network_map(map_routes, truck_positions, active_routes)

    histogram_tab, profitability_tab = st.tabs([
        "Histogram",
        "Profitability insights",
    ])
        histogram_tab, profitability_tab = st.tabs([
            "Histogram",
            "Profitability insights",
        ])

        with histogram_tab:
            histogram = create_histogram(filtered_df, break_even_value)
            st.plotly_chart(histogram, use_container_width=True)
            st.caption(
                "Histogram overlays include the normal distribution fit plus kurtosis and dispersion markers for context."
            )

        with profitability_tab:
            st.markdown("### Profitability insights")
            view_options = {
                "m³ vs km profitability": create_m3_vs_km_figure,
                "Quoted vs calculated $/m³": create_m3_margin_figure,
            }
            selected_view = st.radio(
                "Choose a view",
                list(view_options.keys()),
                horizontal=True,
                help="Switch between per-kilometre earnings and quoted-versus-cost comparisons.",
    st.button(
        "Open quote builder",
        on_click=_activate_quote_tab,
        help="Jump to the quote builder tab to build a quick quote from a historical route.",
    )

    tab_labels = ["Histogram", "Profitability insights", "Quote builder"]
    params = st.experimental_get_query_params()
    requested_tab = params.get("view", [tab_labels[0]])[0]
    if requested_tab not in tab_labels:
        requested_tab = tab_labels[0]
    if requested_tab != tab_labels[0]:
        ordered_labels = [requested_tab] + [label for label in tab_labels if label != requested_tab]
    else:
        ordered_labels = tab_labels

    streamlit_tabs = st.tabs(ordered_labels)
    tab_map: Dict[str, Any] = {
        label: tab for label, tab in zip(ordered_labels, streamlit_tabs)
    }

    with tab_map["Histogram"]:
        histogram = create_histogram(filtered_df, break_even_value)
        st.plotly_chart(histogram, use_container_width=True)
        st.caption(
            "Histogram overlays include the normal distribution fit plus kurtosis and dispersion markers for context."
        )

    with tab_map["Profitability insights"]:
        st.markdown("### Profitability insights")
        view_options = {
            "m³ vs km profitability": create_m3_vs_km_figure,
            "Quoted vs calculated $/m³": create_m3_margin_figure,
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

        if "margin_per_m3" in filtered_df.columns:
            st.markdown("#### Margin outliers")
            ranked = filtered_df.dropna(subset=["margin_per_m3"]).sort_values(
                "margin_per_m3"
            )
            fig = view_options[selected_view](filtered_df)
            st.plotly_chart(fig, use_container_width=True)

            if "margin_per_m3" in filtered_df.columns:
                st.markdown("#### Margin outliers")
                ranked = (
                    filtered_df.dropna(subset=["margin_per_m3"])
                    .sort_values("margin_per_m3")
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

        st.subheader("Filtered jobs")
        display_columns = [
            col for col in [
                "job_date",
                "corridor_display",
                "client_display",
                "price_per_m3",
            ]
            if col in filtered_df.columns
                    if col in ranked.columns
                ]
                low_cols.write("Lowest margin jobs")
                low_cols.dataframe(ranked.head(5)[display_fields])
                high_cols.write("Highest margin jobs")
                high_cols.dataframe(ranked.tail(5).iloc[::-1][display_fields])
            else:
                st.info("No margin data available to highlight outliers yet.")

    with tab_map["Quote builder"]:
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

        if map_columns.issubset(filtered_df.columns):
            map_routes = filtered_df.dropna(subset=list(map_columns)).copy()
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
                            "color": [239, 83, 80, 200],
                        },
                    ]
                    deck = pdk.Deck(
                        map_style="mapbox://styles/mapbox/light-v9",
                        initial_view_state=pdk.ViewState(
                            latitude=midpoint_lat,
                            longitude=midpoint_lon,
                            zoom=6,
                        ),
                        layers=[
                            pdk.Layer(
                                "LineLayer",
                                line_data,
                                get_source_position="from",
                                get_target_position="to",
                                get_width=4,
                                get_color=[33, 150, 243, 160],
                            ),
                            pdk.Layer(
                                "ScatterplotLayer",
                                scatter_data,
                                get_position="position",
                                get_color="color",
                                get_radius=8000,
                                pickable=True,
                            ),
                        ],
                        tooltip={"text": "{label}"},
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
        for candidate in (
            filtered_mapping.volume,
            mapping.volume,
        ):
            if candidate and candidate not in base_candidates:
                base_candidates.append(candidate)

        default_origin = session_inputs.origin if session_inputs else ""
        default_destination = session_inputs.destination if session_inputs else ""
        default_volume = session_inputs.cubic_m if session_inputs else 30.0
        default_date = session_inputs.quote_date if session_inputs else date.today()
        default_modifiers = list(session_inputs.modifiers) if session_inputs else []
        default_margin_percent = (
            session_inputs.target_margin_percent if session_inputs else None
        )
        default_country = session_inputs.country if session_inputs else COUNTRY_DEFAULT

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
                default_country = route_country

        modifier_options = [mod.id for mod in DEFAULT_MODIFIERS]
        modifier_labels: Dict[str, str] = {
            mod.id: mod.label for mod in DEFAULT_MODIFIERS
        }

        with st.form("quote_builder_form"):
            origin_value = st.text_input("Origin", value=default_origin)
            destination_value = st.text_input(
                "Destination", value=default_destination
            )
            country_value = st.text_input(
                "Country", value=default_country or COUNTRY_DEFAULT
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
                    default_margin_percent if default_margin_percent is not None else 20.0
                ),
                step=1.0,
                disabled=not apply_margin,
            )
            submitted = st.form_submit_button("Calculate quote")

        stored_inputs = session_inputs

        if submitted:
            if not origin_value or not destination_value:
                st.error("Origin and destination are required to calculate a quote.")
            else:
                margin_to_apply = (
                    float(margin_percent_value) if apply_margin else None
                )
                quote_inputs = QuoteInput(
                    origin=origin_value,
                    destination=destination_value,
                    cubic_m=float(cubic_m_value),
                    quote_date=quote_date_value,
                    modifiers=list(selected_modifier_ids),
                    target_margin_percent=margin_to_apply,
                    country=country_value or COUNTRY_DEFAULT,
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
                    st.experimental_set_query_params(view="Quote builder")
                    st.success("Quote calculated. Review the breakdown below.")
                    stored_inputs = quote_inputs
                    quote_result = result

        stored_inputs = st.session_state.get("quote_inputs")
        quote_result = st.session_state.get("quote_result")

        if quote_result and stored_inputs:
            st.markdown("#### Quote output")
            st.write(
                f"**Route:** {quote_result.origin_resolved} → {quote_result.destination_resolved}"
            )
            st.write(
                f"**Distance:** {quote_result.distance_km:.1f} km ({quote_result.duration_hr:.1f} h)"
            )
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

            action_cols = st.columns(2)
            if action_cols[0].button("Persist quote", type="primary"):
                try:
                    persist_quote(conn, stored_inputs, quote_result)
                    rowid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                except Exception as exc:  # pragma: no cover - UI feedback path
                    st.error(f"Failed to persist quote: {exc}")
                else:
                    st.success(f"Quote saved as record #{rowid}.")
            if action_cols[1].button("Reset quote builder"):
                st.session_state.pop("quote_result", None)
                st.session_state.pop("quote_inputs", None)
                st.experimental_set_query_params(view="Quote builder")
                st.experimental_rerun()

    st.subheader("Filtered jobs")
    display_columns = [
        col for col in [
            "job_date",
            "corridor_display",
            "client_display",
            "price_per_m3",
        ]
        remaining_columns = [
            col for col in filtered_df.columns
            if col not in display_columns
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

    with map_tab:
        st.markdown("### Route visualisation")
        required_columns = {"origin_lon", "origin_lat", "dest_lon", "dest_lat"}
        if not required_columns.issubset(filtered_df.columns):
            st.info("Mapping requires geocoded origin and destination coordinates.")
        else:
            map_df = filtered_df.dropna(subset=required_columns).copy()

            if map_df.empty:
                st.info("No geocoded jobs are available for the current filters.")
            else:
                def _profit_band(value: Optional[float]) -> str:
                    if pd.isna(value):
                        return "Unknown"
                    if value < break_even_value:
                        return "Below break-even"
                    threshold = break_even_value * 0.1 if break_even_value else 0
                    if value - break_even_value <= threshold:
                        return "Near break-even"
                    return "Above break-even"

                if "price_per_m3" in map_df.columns:
                    map_df["profit_band"] = map_df["price_per_m3"].apply(_profit_band)
                else:
                    map_df["profit_band"] = "Unknown"

                colour_by_options = {"Profit band": "profit_band"}
                if "client_display" in map_df.columns:
                    colour_by_options["Client"] = "client_display"
                if "corridor_display" in map_df.columns:
                    colour_by_options["Corridor"] = "corridor_display"

                colour_choice = st.selectbox(
                    "Colour routes by",
                    options=list(colour_by_options.keys()),
                    help="Adjust route colouring to explore profitability or segment trends.",
                )
                colour_column = colour_by_options[colour_choice]

                colour_series = map_df[colour_column].fillna("Unknown")
                unique_colour_values = list(dict.fromkeys(colour_series))
                palette = [
                    [31, 119, 180],
                    [255, 127, 14],
                    [44, 160, 44],
                    [214, 39, 40],
                    [148, 103, 189],
                    [140, 86, 75],
                    [227, 119, 194],
                    [127, 127, 127],
                    [188, 189, 34],
                    [23, 190, 207],
                ]
                colour_mapping = {
                    value: palette[index % len(palette)]
                    for index, value in enumerate(unique_colour_values)
                }
                map_df["map_colour"] = colour_series.map(colour_mapping)

                latitudes = pd.concat([map_df["origin_lat"], map_df["dest_lat"]])
                longitudes = pd.concat([map_df["origin_lon"], map_df["dest_lon"]])
                view_state = pdk.ViewState(
                    latitude=float(latitudes.mean()),
                    longitude=float(longitudes.mean()),
                    zoom=4,
                    pitch=30,
                )

                route_layer = pdk.Layer(
                    "GreatCircleLayer",
                    data=map_df,
                    get_source_position="[origin_lon, origin_lat]",
                    get_target_position="[dest_lon, dest_lat]",
                    get_source_color="map_colour",
                    get_target_color="map_colour",
                    get_width=5,
                    width_scale=10,
                    width_min_pixels=2,
                    pickable=True,
                )
                origin_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=map_df,
                    get_position="[origin_lon, origin_lat]",
                    get_fill_color="map_colour",
                    get_radius=40000,
                    radius_min_pixels=3,
                    pickable=True,
                )
                destination_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=map_df,
                    get_position="[dest_lon, dest_lat]",
                    get_fill_color="map_colour",
                    get_radius=40000,
                    radius_min_pixels=3,
                    pickable=True,
                )

                deck = pdk.Deck(
                    map_style="mapbox://styles/mapbox/light-v9",
                    initial_view_state=view_state,
                    layers=[route_layer, origin_layer, destination_layer],
                    tooltip={
                        "html": "<b>{client_display}</b><br/>{corridor_display}<br/>$ {price_per_m3} per m³",
                        "style": {"color": "white"},
                    },
                )

                st.pydeck_chart(deck)

                legend_columns = st.columns(min(len(colour_mapping), 4) or 1)
                legend_items = list(colour_mapping.items())
                for index, (label, colour) in enumerate(legend_items):
                    column = legend_columns[index % len(legend_columns)]
                    column.markdown(
                        f"<div style='display:flex;align-items:center;gap:0.5rem;'>"
                        f"<span style='display:inline-block;width:1.5rem;height:1.5rem;border-radius:0.25rem;background-color: rgba({colour[0]}, {colour[1]}, {colour[2]}, 0.85);'></span>"
                        f"<span>{label}</span>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
