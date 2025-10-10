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

    if mapping.price and mapping.price in df.columns:
        df["price_per_m3"] = pd.to_numeric(df[mapping.price], errors="coerce")
    else:
        if not (mapping.revenue and mapping.volume):
            raise RuntimeError(
                "historical_jobs must contain a per-m³ price column or both revenue and volume columns"
            )
        revenue_series = pd.to_numeric(df[mapping.revenue], errors="coerce")
        volume_series = pd.to_numeric(df[mapping.volume], errors="coerce")
        df["price_per_m3"] = revenue_series / volume_series.replace({0: np.nan})

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

    numeric_cols = [c for c in [mapping.revenue, mapping.volume] if c]
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
    else:
        median = percentile_25 = percentile_75 = math.nan
        below_break_even_count = 0
        below_break_even_ratio = 0.0

    return DistributionSummary(
        job_count=job_count,
        priced_job_count=priced_job_count,
        median=median,
        percentile_25=percentile_25,
        percentile_75=percentile_75,
        below_break_even_count=below_break_even_count,
        below_break_even_ratio=below_break_even_ratio,
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
            annotation_font_color=color,
        )

    fig.update_layout(
        bargap=0.02,
        xaxis_title="$ per m³",
        yaxis_title="Job count",
        showlegend=False,
    )

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
