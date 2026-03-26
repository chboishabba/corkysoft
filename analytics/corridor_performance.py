from __future__ import annotations

from typing import Any, Optional

import pandas as pd


def _numeric_series(
    df: pd.DataFrame,
    candidates: list[str],
) -> pd.Series:
    """Return the first matching numeric series or an all-NA aligned series."""

    for candidate in candidates:
        if candidate and candidate in df.columns:
            return pd.to_numeric(df[candidate], errors="coerce")
    return pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")


def _clean_location(value: Any) -> str:
    """Return a normalised string representation for origin/destination labels."""

    if pd.isna(value):
        return "Unknown"
    text = str(value).strip()
    return text or "Unknown"


def format_bidirectional_corridor(origin: Any, destination: Any) -> str:
    """Return a canonical bidirectional corridor label."""

    cleaned_origin = _clean_location(origin)
    cleaned_destination = _clean_location(destination)
    if cleaned_origin == cleaned_destination:
        return cleaned_origin
    ordered = sorted([cleaned_origin, cleaned_destination], key=str.lower)
    return f"{ordered[0]} ↔ {ordered[1]}"


def _split_corridor_label(value: Any) -> tuple[str, str]:
    """Best-effort parsing for corridor labels lacking explicit endpoints."""

    if pd.isna(value):
        return "Unknown", "Unknown"
    text = str(value).strip()
    if not text:
        return "Unknown", "Unknown"
    for delimiter in ("↔", "→", "<->", "-", "—", " to ", "/", "|"):
        if delimiter in text:
            parts = [part.strip() for part in text.split(delimiter) if part.strip()]
            if len(parts) >= 2:
                return parts[0], parts[-1]
    return text, "Unknown"


def _resolve_corridor_pairs(df: pd.DataFrame) -> pd.Series:
    """Return a Series of bidirectional corridor labels for ``df`` rows."""

    from .price_distribution import (
        CORRIDOR_COLUMNS,
        DESTINATION_COLUMNS,
        ORIGIN_COLUMNS,
        _first_present,
    )

    origin_column = _first_present(df.columns, ORIGIN_COLUMNS)
    destination_column = _first_present(df.columns, DESTINATION_COLUMNS)

    if origin_column and destination_column:
        origins = df[origin_column]
        destinations = df[destination_column]
    else:
        corridor_column = None
        for candidate in ("corridor_display", *CORRIDOR_COLUMNS):
            if candidate in df.columns:
                corridor_column = candidate
                break
        if corridor_column:
            pairs = df[corridor_column].apply(_split_corridor_label)
            origins = pairs.str[0]
            destinations = pairs.str[1]
        else:
            origins = pd.Series(["Unknown"] * len(df), index=df.index)
            destinations = pd.Series(["Unknown"] * len(df), index=df.index)

    labels = [
        format_bidirectional_corridor(origin, destination)
        for origin, destination in zip(origins, destinations)
    ]
    return pd.Series(labels, index=df.index)


def aggregate_corridor_performance(
    df: pd.DataFrame,
    break_even: float,
    *,
    volume_column: Optional[str] = None,
    revenue_column: Optional[str] = None,
) -> pd.DataFrame:
    """Aggregate systemic performance metrics by bidirectional corridor."""

    from .price_distribution import LANE_STATUS_ASSIGNED

    columns = [
        "corridor_pair",
        "job_count",
        "share_of_jobs",
        "priced_job_count",
        "priced_job_ratio",
        "median_price_per_m3",
        "mean_price_per_m3",
        "price_per_m3_p25",
        "price_per_m3_p75",
        "weighted_price_per_m3",
        "below_break_even_ratio",
        "total_volume_m3",
        "share_of_volume",
        "total_revenue",
        "margin_per_m3_median",
        "margin_total_sum",
        "share_of_margin",
        "margin_total_pct_median",
        "revenue_per_km_median",
        "median_distance_km",
    ]

    if df.empty:
        return pd.DataFrame(columns=columns)

    working = df.copy()
    working["corridor_pair"] = _resolve_corridor_pairs(working)
    if {
        "lane_assignment_status",
        "origin_cluster_key",
        "destination_cluster_key",
    }.issubset(working.columns):
        assigned_mask = (
            working["lane_assignment_status"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .eq(LANE_STATUS_ASSIGNED)
        )
        origin_cluster = working["origin_cluster_key"].fillna("").astype(str).str.strip()
        destination_cluster = (
            working["destination_cluster_key"].fillna("").astype(str).str.strip()
        )
        canonical_pairs = pd.Series(
            [
                format_bidirectional_corridor(origin, destination)
                if origin and destination
                else pair
                for origin, destination, pair in zip(
                    origin_cluster,
                    destination_cluster,
                    working["corridor_pair"],
                )
            ],
            index=working.index,
        )
        working.loc[assigned_mask, "corridor_pair"] = canonical_pairs.loc[assigned_mask]

    priced_series = _numeric_series(working, ["price_per_m3"])
    volume_series = _numeric_series(
        working,
        [value for value in [volume_column, "volume_m3", "volume"] if value],
    )
    revenue_series = _numeric_series(
        working,
        [value for value in [revenue_column, "revenue_total", "revenue"] if value],
    )
    margin_per_m3_series = _numeric_series(working, ["margin_per_m3"])
    margin_total_series = _numeric_series(working, ["margin_total"])
    margin_total_pct_series = _numeric_series(working, ["margin_total_pct"])
    revenue_per_km_series = _numeric_series(working, ["revenue_per_km"])
    distance_series = _numeric_series(working, ["distance_km"])

    working["price_per_m3_numeric"] = priced_series
    working["volume_numeric"] = volume_series
    working["revenue_numeric"] = revenue_series
    working["margin_per_m3_numeric"] = margin_per_m3_series
    working["margin_total_numeric"] = margin_total_series
    working["margin_total_pct_numeric"] = margin_total_pct_series
    working["revenue_per_km_numeric"] = revenue_per_km_series
    working["distance_numeric"] = distance_series
    working["priced_flag"] = priced_series.notna()
    working["below_break_even_flag"] = priced_series.lt(break_even)

    grouped = working.groupby("corridor_pair", dropna=False)
    summary = grouped.agg(
        job_count=("corridor_pair", "size"),
        priced_job_count=("priced_flag", "sum"),
        median_price_per_m3=("price_per_m3_numeric", "median"),
        mean_price_per_m3=("price_per_m3_numeric", "mean"),
        price_per_m3_p25=("price_per_m3_numeric", lambda values: values.quantile(0.25)),
        price_per_m3_p75=("price_per_m3_numeric", lambda values: values.quantile(0.75)),
        below_break_even_ratio=("below_break_even_flag", "mean"),
        total_volume_m3=("volume_numeric", "sum"),
        total_revenue=("revenue_numeric", "sum"),
        margin_per_m3_median=("margin_per_m3_numeric", "median"),
        margin_total_sum=("margin_total_numeric", "sum"),
        margin_total_pct_median=("margin_total_pct_numeric", "median"),
        revenue_per_km_median=("revenue_per_km_numeric", "median"),
        median_distance_km=("distance_numeric", "median"),
    ).reset_index()

    total_jobs = float(len(working))
    total_volume = float(volume_series.fillna(0.0).sum())
    total_margin = float(margin_total_series.fillna(0.0).sum())

    weighted_prices: list[float] = []
    for _, group in grouped:
        price_values = pd.to_numeric(group["price_per_m3_numeric"], errors="coerce")
        volume_values = pd.to_numeric(group["volume_numeric"], errors="coerce")
        valid_mask = price_values.notna() & volume_values.notna()
        if not valid_mask.any():
            weighted_prices.append(float("nan"))
            continue
        valid_prices = price_values.loc[valid_mask]
        valid_volumes = volume_values.loc[valid_mask]
        volume_sum = float(valid_volumes.sum())
        if volume_sum <= 0:
            weighted_prices.append(float("nan"))
            continue
        weighted_prices.append(float((valid_prices * valid_volumes).sum() / volume_sum))

    summary["weighted_price_per_m3"] = weighted_prices
    summary["share_of_jobs"] = summary["job_count"] / total_jobs if total_jobs else 0.0
    summary["priced_job_ratio"] = summary["priced_job_count"] / summary["job_count"].replace(
        0, pd.NA
    )
    summary["share_of_volume"] = (
        summary["total_volume_m3"] / total_volume if total_volume else 0.0
    )
    summary["share_of_margin"] = (
        summary["margin_total_sum"] / total_margin if total_margin else 0.0
    )

    return summary[columns].sort_values(
        by=["share_of_jobs", "total_volume_m3", "corridor_pair"],
        ascending=[False, False, True],
        ignore_index=True,
    )
