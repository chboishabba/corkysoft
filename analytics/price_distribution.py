"""Helpers for the price-distribution Streamlit view."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .db import bootstrap_parameters, ensure_global_parameters_table, get_parameter_value, set_parameter_value

BREAK_EVEN_KEY = "break_even_per_m3"
DEFAULT_BREAK_EVEN_VALUE = 250.0
DEFAULT_BREAK_EVEN_DESCRIPTION = "Baseline break-even $/m³ across the network"

# Candidate column names used in legacy exports.
DATE_COLUMNS = [
    "job_date",
    "move_date",
    "delivery_date",
    "created_at",
    "updated_at",
]
CLIENT_COLUMNS = [
    "client",
    "client_name",
    "account",
    "customer",
]
VOLUME_COLUMNS = [
    "volume_m3",
    "volume_cbm",
    "cbm",
    "cubic_meters",
    "m3",
]
REVENUE_COLUMNS = [
    "revenue_total",
    "sell_total",
    "total_revenue",
    "price_total",
    "quoted_sell",
]
PRICE_COLUMNS = [
    "revenue_per_m3",
    "price_per_m3",
    "sell_per_m3",
    "rate_per_m3",
]
ORIGIN_COLUMNS = [
    "origin",
    "origin_suburb",
    "origin_city",
]
DESTINATION_COLUMNS = [
    "destination",
    "destination_suburb",
    "destination_city",
]
POSTCODE_COLUMNS = [
    "origin_postcode",
    "origin_postal",
    "origin_pc",
    "destination_postcode",
    "destination_postal",
    "destination_pc",
]
CORRIDOR_COLUMNS = [
    "corridor",
    "lane",
    "lane_name",
]
DISTANCE_COLUMNS = [
    "distance_km",
    "distance",
    "km",
    "kms",
    "kilometers",
    "kilometres",
]
FINAL_COST_COLUMNS = [
    "final_cost",
    "final_total",
    "actual_cost",
    "actual_total",
    "final_sell",
    "final_sell_total",
    "actual_sell",
    "cost_total",
    "total_cost",
    "final_price",
    "final_amount",
]


@dataclass
class ColumnMapping:
    date: Optional[str]
    client: Optional[str]
    price: Optional[str]
    revenue: Optional[str]
    volume: Optional[str]
    origin: Optional[str]
    destination: Optional[str]
    corridor: Optional[str]
    distance: Optional[str]
    final_cost: Optional[str]


def _first_present(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    columns_lower = {c.lower(): c for c in columns}
    for candidate in candidates:
        lower = candidate.lower()
        if lower in columns_lower:
            return columns_lower[lower]
    return None


def infer_columns(df: pd.DataFrame) -> ColumnMapping:
    cols = df.columns
    return ColumnMapping(
        date=_first_present(cols, DATE_COLUMNS),
        client=_first_present(cols, CLIENT_COLUMNS),
        price=_first_present(cols, PRICE_COLUMNS),
        revenue=_first_present(cols, REVENUE_COLUMNS),
        volume=_first_present(cols, VOLUME_COLUMNS),
        origin=_first_present(cols, ORIGIN_COLUMNS),
        destination=_first_present(cols, DESTINATION_COLUMNS),
        corridor=_first_present(cols, CORRIDOR_COLUMNS),
        distance=_first_present(cols, DISTANCE_COLUMNS),
        final_cost=_first_present(cols, FINAL_COST_COLUMNS),
    )


def _historical_jobs_query() -> str:
    """Return the default query joining address metadata for historical jobs."""

    return """
        SELECT
            hj.*,
            COALESCE(o.city, o.normalized, o.raw_input) AS origin,
            COALESCE(d.city, d.normalized, d.raw_input) AS destination,
            o.raw_input AS origin_raw,
            o.normalized AS origin_normalized,
            o.city AS origin_city,
            o.state AS origin_state,
            o.postcode AS origin_postcode,
            o.country AS origin_country,
            o.lon AS origin_lon,
            o.lat AS origin_lat,
            d.raw_input AS destination_raw,
            d.normalized AS destination_normalized,
            d.city AS destination_city,
            d.state AS destination_state,
            d.postcode AS destination_postcode,
            d.country AS destination_country,
            d.lon AS dest_lon,
            d.lat AS dest_lat
        FROM historical_jobs AS hj
        LEFT JOIN addresses AS o ON hj.origin_address_id = o.id
        LEFT JOIN addresses AS d ON hj.destination_address_id = d.id
    """


def load_historical_jobs(
    conn,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    clients: Optional[Sequence[str]] = None,
    corridor: Optional[str] = None,
    postcode_prefix: Optional[str] = None,
) -> tuple[pd.DataFrame, ColumnMapping]:
    """Load historical job data applying the requested filters."""
    ensure_global_parameters_table(conn)

    query = _historical_jobs_query()
    try:
        df = pd.read_sql_query(query, conn)
    except Exception:  # pragma: no cover - fall back for legacy schemas
        try:
            df = pd.read_sql_query("SELECT * FROM historical_jobs", conn)
        except Exception as exc:  # pragma: no cover - surfaces friendly error in UI
            raise RuntimeError("historical_jobs table is required for this view") from exc

    if df.empty:
        mapping = infer_columns(df)
        return df, mapping

    mapping = infer_columns(df)

    if mapping.date and mapping.date in df.columns:
        df[mapping.date] = pd.to_datetime(df[mapping.date], errors="coerce")
        if start_date is not None:
            df = df[df[mapping.date] >= pd.to_datetime(start_date)]
        if end_date is not None:
            df = df[df[mapping.date] <= pd.to_datetime(end_date)]

    if mapping.client and clients:
        df = df[df[mapping.client].isin(clients)]

    if postcode_prefix:
        postcode_prefix = str(postcode_prefix).strip()
        if postcode_prefix:
            prefix_lower = postcode_prefix.lower()
            known_postcodes = {pc.lower() for pc in POSTCODE_COLUMNS}
            postcode_columns = [c for c in df.columns if c.lower() in known_postcodes]
            text_columns = list(dict.fromkeys(postcode_columns))  # de-duplicate while preserving order
            if mapping.corridor and mapping.corridor in df.columns:
                text_columns.append(mapping.corridor)
            if mapping.origin and mapping.origin in df.columns:
                text_columns.append(mapping.origin)
            if mapping.destination and mapping.destination in df.columns:
                text_columns.append(mapping.destination)
            if text_columns:
                mask = pd.Series(False, index=df.index)
                for col in text_columns:
                    mask = mask | df[col].astype(str).str.lower().str.contains(prefix_lower, na=False)
                df = df[mask]

    df = df.copy()

    revenue_series: Optional[pd.Series] = None
    volume_series: Optional[pd.Series] = None
    distance_series: Optional[pd.Series] = None
    final_cost_series: Optional[pd.Series] = None

    if mapping.revenue and mapping.revenue in df.columns:
        revenue_series = pd.to_numeric(df[mapping.revenue], errors="coerce")
        df[mapping.revenue] = revenue_series
    if mapping.volume and mapping.volume in df.columns:
        volume_series = pd.to_numeric(df[mapping.volume], errors="coerce")
        df[mapping.volume] = volume_series
    if mapping.distance and mapping.distance in df.columns:
        distance_series = pd.to_numeric(df[mapping.distance], errors="coerce")
        df[mapping.distance] = distance_series
        df["distance_km"] = distance_series
    if mapping.final_cost and mapping.final_cost in df.columns:
        final_cost_series = pd.to_numeric(df[mapping.final_cost], errors="coerce")
        df[mapping.final_cost] = final_cost_series

    if mapping.price and mapping.price in df.columns:
        df["price_per_m3"] = pd.to_numeric(df[mapping.price], errors="coerce")
    else:
        if revenue_series is None or volume_series is None:
            raise RuntimeError(
                "historical_jobs must contain a per-m³ price column or both revenue and volume columns"
            )
        df["price_per_m3"] = revenue_series / volume_series.replace({0: np.nan})

    if revenue_series is not None and distance_series is not None:
        df["revenue_per_km"] = revenue_series / distance_series.replace({0: np.nan})

    if final_cost_series is not None:
        df["final_cost_total"] = final_cost_series
        safe_cost = final_cost_series.replace({0: np.nan})
        if revenue_series is not None:
            margin_total = revenue_series - final_cost_series
            df["margin_total"] = margin_total
            df["margin_total_pct"] = margin_total / safe_cost
        if volume_series is not None:
            safe_volume = volume_series.replace({0: np.nan})
            cost_per_m3 = final_cost_series / safe_volume
            df["final_cost_per_m3"] = cost_per_m3
            margin_per_m3 = df["price_per_m3"] - cost_per_m3
            df["margin_per_m3"] = margin_per_m3
            safe_cost_per_m3 = cost_per_m3.replace({0: np.nan})
            df["margin_per_m3_pct"] = margin_per_m3 / safe_cost_per_m3

    if mapping.corridor and mapping.corridor in df.columns:
        df["corridor_display"] = df[mapping.corridor]
    else:
        origin = df[mapping.origin] if mapping.origin else None
        destination = df[mapping.destination] if mapping.destination else None
        if origin is not None and destination is not None:
            df["corridor_display"] = origin.fillna("?") + " → " + destination.fillna("?")
        else:
            df["corridor_display"] = "Unknown"

    if corridor:
        if mapping.corridor and mapping.corridor in df.columns:
            df = df[df[mapping.corridor] == corridor]
        else:
            df = df[df["corridor_display"] == corridor]

    if mapping.client and mapping.client in df.columns:
        df["client_display"] = df[mapping.client]
    else:
        df["client_display"] = "Unknown"

    if mapping.date:
        df["job_date"] = df[mapping.date]

    numeric_cols = [
        c
        for c in [mapping.revenue, mapping.volume, mapping.distance, mapping.final_cost]
        if c
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, mapping


@dataclass
class DistributionSummary:
    job_count: int
    priced_job_count: int
    median: Optional[float]
    percentile_25: Optional[float]
    percentile_75: Optional[float]
    below_break_even_count: int
    below_break_even_ratio: float
    mean: Optional[float]
    std_dev: Optional[float]
    kurtosis: Optional[float]
    skewness: Optional[float]


def summarise_distribution(df: pd.DataFrame, break_even: float) -> DistributionSummary:
    """Return KPI statistics for the filtered dataset."""
    priced = df["price_per_m3"].dropna()
    job_count = len(df)
    priced_job_count = len(priced)
    if priced_job_count:
        median = float(priced.median())
        percentile_25 = float(priced.quantile(0.25))
        percentile_75 = float(priced.quantile(0.75))
        below_break_even_count = int((priced < break_even).sum())
        below_break_even_ratio = below_break_even_count / priced_job_count
        mean = float(priced.mean())
        std_dev = float(priced.std(ddof=1)) if priced_job_count > 1 else math.nan
        kurtosis = float(priced.kurtosis()) if priced_job_count > 3 else math.nan
        skewness = float(priced.skew()) if priced_job_count > 2 else math.nan
    else:
        median = percentile_25 = percentile_75 = math.nan
        below_break_even_count = 0
        below_break_even_ratio = 0.0
        mean = std_dev = kurtosis = skewness = math.nan

    return DistributionSummary(
        job_count=job_count,
        priced_job_count=priced_job_count,
        median=median,
        percentile_25=percentile_25,
        percentile_75=percentile_75,
        below_break_even_count=below_break_even_count,
        below_break_even_ratio=below_break_even_ratio,
        mean=mean,
        std_dev=std_dev,
        kurtosis=kurtosis,
        skewness=skewness,
    )


@dataclass
class ProfitabilitySummary:
    revenue_per_km_median: Optional[float]
    revenue_per_km_mean: Optional[float]
    margin_per_m3_median: Optional[float]
    margin_per_m3_pct_median: Optional[float]
    margin_total_median: Optional[float]
    margin_total_pct_median: Optional[float]


def summarise_profitability(df: pd.DataFrame) -> ProfitabilitySummary:
    """Calculate profitability KPIs used by the optional views."""

    def _median(series: pd.Series) -> Optional[float]:
        series = series.dropna()
        if series.empty:
            return math.nan
        return float(series.median())

    def _mean(series: pd.Series) -> Optional[float]:
        series = series.dropna()
        if series.empty:
            return math.nan
        return float(series.mean())

    revenue_per_km_median = revenue_per_km_mean = math.nan
    if "revenue_per_km" in df:
        revenue_per_km_series = pd.to_numeric(df["revenue_per_km"], errors="coerce")
        revenue_per_km_median = _median(revenue_per_km_series)
        revenue_per_km_mean = _mean(revenue_per_km_series)

    margin_per_m3_median = margin_per_m3_pct_median = math.nan
    if "margin_per_m3" in df:
        margin_per_m3_series = pd.to_numeric(df["margin_per_m3"], errors="coerce")
        margin_per_m3_median = _median(margin_per_m3_series)
    if "margin_per_m3_pct" in df:
        margin_per_m3_pct_series = pd.to_numeric(df["margin_per_m3_pct"], errors="coerce")
        margin_per_m3_pct_median = _median(margin_per_m3_pct_series)

    margin_total_median = margin_total_pct_median = math.nan
    if "margin_total" in df:
        margin_total_series = pd.to_numeric(df["margin_total"], errors="coerce")
        margin_total_median = _median(margin_total_series)
    if "margin_total_pct" in df:
        margin_total_pct_series = pd.to_numeric(df["margin_total_pct"], errors="coerce")
        margin_total_pct_median = _median(margin_total_pct_series)

    return ProfitabilitySummary(
        revenue_per_km_median=revenue_per_km_median,
        revenue_per_km_mean=revenue_per_km_mean,
        margin_per_m3_median=margin_per_m3_median,
        margin_per_m3_pct_median=margin_per_m3_pct_median,
        margin_total_median=margin_total_median,
        margin_total_pct_median=margin_total_pct_median,
    )


def _band_styles():
    return {
        -0.5: ("-50%", "rgba(214, 39, 40, 0.55)", "dot"),
        -0.2: ("-20%", "rgba(255, 127, 14, 0.7)", "dash"),
        -0.1: ("-10%", "rgba(255, 187, 120, 0.8)", "dashdot"),
        0.0: ("Break-even", "rgba(33, 33, 33, 0.9)", "solid"),
        0.1: ("+10%", "rgba(31, 119, 180, 0.7)", "dashdot"),
        0.2: ("+20%", "rgba(44, 160, 44, 0.7)", "dash"),
        0.5: ("+50%", "rgba(148, 103, 189, 0.8)", "dot"),
    }


def create_histogram(df: pd.DataFrame, break_even: float, bins: Optional[int] = None) -> go.Figure:
    """Create a histogram with break-even bands."""
    priced = df.dropna(subset=["price_per_m3"])
    if priced.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No jobs available for the selected filters",
            xaxis_title="$ per m³",
            yaxis_title="Job count",
        )
        return fig

    if bins is None:
        bins = min(50, max(10, int(math.sqrt(len(priced)))))

    fig = px.histogram(
        priced,
        x="price_per_m3",
        nbins=bins,
        labels={"price_per_m3": "$ per m³", "count": "Job count"},
        title="$ per m³ distribution",
        opacity=0.85,
    )

    styles = _band_styles()
    for pct, (label, color, dash) in styles.items():
        if break_even <= 0 and pct != 0.0:
            continue
        x_val = break_even * (1 + pct)
        fig.add_vline(
            x=x_val,
            line_width=2 if pct == 0 else 1.5,
            line_dash=dash,
            line_color=color,
            annotation_text=label,
            annotation_position="top",
            annotation_font_color="#111",
            annotation_bgcolor="rgba(255, 255, 255, 0.85)",
            annotation_bordercolor=color,
        )

    priced_values = priced["price_per_m3"].dropna()
    mean_val = float(priced_values.mean()) if not priced_values.empty else math.nan
    std_val = float(priced_values.std(ddof=1)) if len(priced_values) > 1 else math.nan
    kurtosis_val = float(priced_values.kurtosis()) if len(priced_values) > 3 else math.nan

    if len(priced_values) > 1 and std_val and not math.isnan(std_val) and std_val > 0:
        _, bin_edges = np.histogram(priced_values, bins=bins)
        if len(bin_edges) > 1:
            bin_width = float(np.mean(np.diff(bin_edges)))
        else:
            bin_width = 0.0
        if bin_width > 0:
            x_vals = np.linspace(bin_edges[0], bin_edges[-1], 200)
            pdf = (1.0 / (std_val * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_vals - mean_val) / std_val) ** 2)
            y_vals = pdf * len(priced_values) * bin_width
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines",
                    name="Normal fit",
                    line=dict(color="rgba(17, 17, 17, 0.85)", width=2),
                )
            )

    stats_bits = []
    if not math.isnan(mean_val):
        stats_bits.append(f"μ={mean_val:,.2f}")
    if not math.isnan(std_val):
        stats_bits.append(f"σ={std_val:,.2f}")
    if not math.isnan(kurtosis_val):
        stats_bits.append(f"kurtosis={kurtosis_val:,.2f}")
    if stats_bits:
        fig.add_annotation(
            text=" | ".join(stats_bits),
            xref="paper",
            yref="paper",
            x=0.99,
            y=0.98,
            showarrow=False,
            align="right",
            bgcolor="rgba(255, 255, 255, 0.85)",
            bordercolor="rgba(17, 17, 17, 0.2)",
            font=dict(color="#111", size=12),
        )

    fig.update_layout(
        bargap=0.02,
        xaxis_title="$ per m³",
        yaxis_title="Job count",
        showlegend=False,
        hovermode="x unified",
    )

    return fig


def _empty_figure(title: str, x_title: str, y_title: str, message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(color="#555", size=13),
    )
    return fig


def create_m3_vs_km_figure(df: pd.DataFrame) -> go.Figure:
    """Visualise $/m³ profitability relative to $/km earnings."""

    if "price_per_m3" not in df.columns or "revenue_per_km" not in df.columns:
        return _empty_figure(
            title="m³ vs km profitability",
            x_title="$ per m³",
            y_title="$ per km",
            message="No revenue or distance data available.",
        )

    subset = df.dropna(subset=["price_per_m3", "revenue_per_km"])
    if subset.empty:
        return _empty_figure(
            title="m³ vs km profitability",
            x_title="$ per m³",
            y_title="$ per km",
            message="Add jobs with both revenue and distance to unlock this view.",
        )

    hover_data: dict[str, object] = {}
    for column, fmt in [
        ("client_display", True),
        ("corridor_display", True),
        ("volume_m3", ":.1f"),
        ("volume", ":.1f"),
        ("distance_km", ":.1f"),
        ("margin_total", ":.0f"),
        ("margin_total_pct", ":.1%"),
    ]:
        if column in subset.columns:
            hover_data[column] = fmt

    color_col = "corridor_display" if "corridor_display" in subset.columns else None

    fig = px.scatter(
        subset,
        x="price_per_m3",
        y="revenue_per_km",
        color=color_col,
        hover_data=hover_data,
        labels={"price_per_m3": "$ per m³", "revenue_per_km": "$ per km"},
        title="m³ vs km profitability",
    )
    fig.update_traces(marker=dict(size=10, opacity=0.8))
    fig.update_layout(
        xaxis=dict(zeroline=False),
        yaxis=dict(zeroline=False),
        legend_title_text="Corridor" if color_col else None,
    )
    return fig


def create_m3_margin_figure(df: pd.DataFrame) -> go.Figure:
    """Compare quoted $/m³ to cost-derived $/m³ and highlight margin deltas."""

    required_cols = {"price_per_m3", "final_cost_per_m3"}
    if not required_cols.issubset(df.columns):
        return _empty_figure(
            title="Quoted vs calculated $/m³",
            x_title="Cost-derived $ per m³",
            y_title="Quoted $ per m³",
            message="Final cost data is unavailable.",
        )

    subset = df.dropna(subset=list(required_cols))
    if subset.empty:
        return _empty_figure(
            title="Quoted vs calculated $/m³",
            x_title="Cost-derived $ per m³",
            y_title="Quoted $ per m³",
            message="No jobs contain both quoted and calculated $/m³ values.",
        )

    hover_data: dict[str, object] = {}
    for column, fmt in [
        ("client_display", True),
        ("corridor_display", True),
        ("margin_per_m3", ":.2f"),
        ("margin_per_m3_pct", ":.1%"),
        ("margin_total", ":.0f"),
        ("margin_total_pct", ":.1%"),
        ("volume_m3", ":.1f"),
        ("distance_km", ":.1f"),
    ]:
        if column in subset.columns:
            hover_data[column] = fmt

    color_col = "margin_per_m3_pct" if "margin_per_m3_pct" in subset.columns else None
    color_args = {}
    if color_col:
        color_args = {
            "color": subset[color_col],
            "color_continuous_scale": "RdYlGn",
        }

    fig = px.scatter(
        subset,
        x="final_cost_per_m3",
        y="price_per_m3",
        hover_data=hover_data,
        labels={
            "final_cost_per_m3": "Cost-derived $ per m³",
            "price_per_m3": "Quoted $ per m³",
        },
        title="Quoted vs calculated $/m³",
        **color_args,
    )
    fig.update_traces(marker=dict(size=10, opacity=0.8))

    min_val = float(subset[["final_cost_per_m3", "price_per_m3"]].min().min())
    max_val = float(subset[["final_cost_per_m3", "price_per_m3"]].max().max())
    if max_val > min_val:
        parity_line = go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="rgba(17, 17, 17, 0.6)", dash="dash"),
            name="Parity",
            showlegend=False,
        )
        fig.add_trace(parity_line)

    fig.update_layout(coloraxis_colorbar=dict(title="Margin %"), legend_title_text=None)
    return fig


def ensure_break_even_parameter(conn) -> float:
    """Ensure the break-even parameter exists and return its value."""
    bootstrap_parameters(
        conn,
        [
            (BREAK_EVEN_KEY, DEFAULT_BREAK_EVEN_VALUE, DEFAULT_BREAK_EVEN_DESCRIPTION),
        ],
    )
    value = get_parameter_value(conn, BREAK_EVEN_KEY)
    assert value is not None
    return value


def update_break_even(conn, value: float) -> None:
    """Update the break-even value."""
    set_parameter_value(conn, BREAK_EVEN_KEY, value, DEFAULT_BREAK_EVEN_DESCRIPTION)
