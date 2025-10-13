"""Streamlit dashboard for the price distribution analysis."""
from __future__ import annotations

import io
import math
from datetime import date
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from analytics.db import connection_scope
from analytics.price_distribution import (
    DistributionSummary,
    ProfitabilitySummary,
    PROFITABILITY_COLOURS,
    available_heatmap_weightings,
    build_heatmap_source,
    create_histogram,
    create_metro_profitability_figure,
    create_m3_margin_figure,
    create_m3_vs_km_figure,
    ensure_break_even_parameter,
    load_historical_jobs,
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
    load_active_routes,
    load_truck_positions,
)
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


def _prepare_plotly_map_data(
    df: pd.DataFrame,
    colour_column: str,
    *,
    placeholder: str = "Unknown",
) -> pd.DataFrame:
    """Return a dataframe suitable for categorical colouring on a Plotly map."""
    required_columns = ["origin_lat", "origin_lon", "dest_lat", "dest_lon"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise KeyError(
            f"Dataframe is missing required coordinate columns: {missing_str}"
        )

    if colour_column not in df.columns:
        raise KeyError(f"'{colour_column}' column is required to colour the map")

    filtered = df.dropna(subset=required_columns).copy()
    if filtered.empty:
        return filtered

    filtered["map_colour_value"] = (
        filtered[colour_column].fillna(placeholder).astype(str)
    )
    return filtered


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
) -> go.Figure:
    """Construct a Plotly Mapbox figure showing coloured routes and points."""
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

    if show_routes:
        for value in colour_values:
            category_df = df[df["map_colour_value"] == value]
            if category_df.empty:
                continue
            lat_values: list[float] = []
            lon_values: list[float] = []
            for _, row in category_df.iterrows():
                lat_values.extend([row["origin_lat"], row["dest_lat"], None])
                lon_values.extend([row["origin_lon"], row["dest_lon"], None])
            figure.add_trace(
                go.Scattermapbox(
                    lat=lat_values,
                    lon=lon_values,
                    mode="lines",
                    line={"width": 2, "color": colour_map[value]},
                    name=value,
                    legendgroup=value,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    if show_points:
        for value in colour_values:
            category_df = df[df["map_colour_value"] == value]
            if category_df.empty:
                continue
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
                    f"{colour_label}: {value}<br>Origin: {origin_label}<br>Job ID: {job_id}"
                )
                marker_lat.append(row["dest_lat"])
                marker_lon.append(row["dest_lon"])
                marker_text.append(
                    f"{colour_label}: {value}<br>Destination: {destination_label}<br>Job ID: {job_id}"
                )

            figure.add_trace(
                go.Scattermapbox(
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
                    name=value,
                    legendgroup=value,
                )
            )

    all_lat = pd.concat([df["origin_lat"], df["dest_lat"]])
    all_lon = pd.concat([df["origin_lon"], df["dest_lon"]])
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
) -> None:
    st.markdown("### Live network overview")

    show_live_overlay = st.toggle(
        "Show live network overlay",
        value=True,
        help=(
            "Toggle the live overlay of active routes and truck telemetry without "
            "hiding the base map."
        ),
        key="network_map_live_overlay_toggle",
    )

    if historical_routes.empty and trucks.empty and active_routes.empty:
        st.info("No geocoded historical jobs or live telemetry available to plot yet.")

    truck_data = trucks.copy()
    if not truck_data.empty:
        truck_data["colour"] = truck_data["status"].map(TRUCK_STATUS_COLOURS)
        truck_data["colour"] = truck_data["colour"].apply(
            lambda value: value if isinstance(value, (list, tuple)) else [0, 122, 204]
        )
        truck_data["tooltip"] = truck_data.apply(
            lambda row: f"{row['truck_id']} ({row['status']})", axis=1
        )

    base_map_layer = pdk.Layer(
        "TileLayer",
        data="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        min_zoom=0,
        max_zoom=19,
        tile_size=256,
        attribution="© OpenStreetMap contributors",
    )

    overlay_layers: list[pdk.Layer] = []

    if show_live_overlay and not historical_routes.empty:
        history_layer = pdk.Layer(
            "LineLayer",
            data=historical_routes,
            get_source_position="[origin_lon, origin_lat]",
            get_target_position="[dest_lon, dest_lat]",
            get_color="colour",
            get_width="line_width",
            pickable=True,
            opacity=0.4,
        )
        overlay_layers.append(history_layer)

    if show_live_overlay and not active_routes.empty:
        if "job_id" in active_routes.columns and not historical_routes.empty:
            enriched = active_routes.merge(
                historical_routes[
                    ["id", "colour", "profit_band", "profitability_status", "tooltip"]
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


def _activate_quote_tab() -> None:
    """Switch the interface to the quote builder tab."""
    _set_query_params(view="Quote builder")
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
            st.info(
                "historical_jobs table has no rows yet. Import historical jobs to populate the view."
            )
            st.stop()

        date_column = "job_date" if "job_date" in df_all.columns else mapping.date
        if date_column and date_column in df_all.columns:
            df_all[date_column] = pd.to_datetime(df_all[date_column], errors="coerce")
            min_date = df_all[date_column].min()
            max_date = df_all[date_column].max()
            default_start = (
                min_date.date() if isinstance(min_date, pd.Timestamp) else date.today()
            )
            default_end = (
                max_date.date() if isinstance(max_date, pd.Timestamp) else date.today()
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
        default_clients = client_options if client_options else []
        selected_clients = st.multiselect(
            "Client",
            options=client_options,
            default=default_clients,
        )

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

    summary = summarise_distribution(filtered_df, break_even_value)
    profitability_summary = summarise_profitability(filtered_df)

    metro_distance_km = 100.0
    metro_df = _filter_by_distance(filtered_df, metro_only=True, max_distance_km=metro_distance_km)
    metro_summary = (
        summarise_distribution(metro_df, break_even_value)
        if not metro_df.empty
        else None
    )
    metro_profitability = (
        summarise_profitability(metro_df) if not metro_df.empty else None
    )

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

    render_network_map(map_routes, truck_positions, active_routes)

    st.button(
        "Open quote builder",
        on_click=_activate_quote_tab,
        help="Jump to the quote builder tab to build a quick quote from a historical route.",
    )

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

    truck_positions = load_truck_positions(conn)
    active_routes = load_active_routes(conn)

    with tab_map["Route maps"]:
        st.markdown("### Corridor visualisation")
        map_mode = st.radio(
            "Visualisation mode",
            ("Routes/points", "Heatmap"),
            horizontal=True,
            help="Switch between individual routes/points and an aggregate density heatmap.",
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
            colour_dimensions = {
                "Job ID": "id",
                "Client": "client_display",
                "Destination city": "destination_city",
                "Origin city": "origin_city",
            }
            available_colour_dimensions = {
                label: column
                for label, column in colour_dimensions.items()
                if column in scoped_df.columns
            }

            if not available_colour_dimensions:
                st.info("No categorical columns available to colour the route map.")
            elif scoped_df.empty:
                st.info("No jobs match the metro filter for the current selection.")
            else:
                colour_label = st.selectbox(
                    "Colour routes by",
                    options=list(available_colour_dimensions.keys()),
                    help="Choose which attribute drives the route and point colouring.",
                )
                show_routes = st.checkbox("Show route lines", value=True)
                show_points = st.checkbox("Show origin/destination points", value=True)

                selected_column = available_colour_dimensions[colour_label]
                try:
                    plotly_map_df = _prepare_plotly_map_data(scoped_df, selected_column)
                except KeyError as exc:
                    st.warning(str(exc))
                    plotly_map_df = pd.DataFrame()

                if plotly_map_df.empty:
                    st.info(
                        "No routes with coordinates are available for the current filters."
                    )
                elif not show_routes and not show_points:
                    st.info("Enable at least one layer to view the route map.")
                else:
                    route_map = build_route_map(
                        plotly_map_df,
                        colour_label,
                        show_routes=show_routes,
                        show_points=show_points,
                    )
                    st.plotly_chart(route_map, use_container_width=True)
        else:
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

        network_routes = prepare_profitability_map_data(scoped_df, break_even_value)
        render_network_map(network_routes, truck_positions, active_routes)

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
        modifier_labels: Dict[str, str] = {mod.id: mod.label for mod in DEFAULT_MODIFIERS}

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
                help=(
                    "Enter the desired margin percentage. The value is only used when 'Apply margin'"
                    " is enabled."
                ),
            )
            submitted = st.form_submit_button("Calculate quote")

        stored_inputs = session_inputs

        if submitted:
            if not origin_value or not destination_value:
                st.error("Origin and destination are required to calculate a quote.")
            else:
                margin_to_apply = float(margin_percent_value) if apply_margin else None
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
                    _set_query_params(view="Quote builder")
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
                _set_query_params(view="Quote builder")
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
