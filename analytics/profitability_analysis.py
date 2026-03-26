from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

PROFITABILITY_BANDS: Sequence[tuple[float, float, str]] = (
    (-float("inf"), 0.0, "Below break-even"),
    (0.0, 50.0, "0-50 above break-even"),
    (50.0, 100.0, "50-100 above break-even"),
    (100.0, float("inf"), "100+ above break-even"),
)
DEFAULT_BREAK_EVEN_ABS_TOLERANCE = 5.0
DEFAULT_BREAK_EVEN_REL_TOLERANCE = 0.02


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
    if "price_per_m3" not in df.columns:
        raise KeyError("'price_per_m3' column is required for distribution summaries")

    price_series = pd.to_numeric(df["price_per_m3"], errors="coerce")
    priced = price_series.dropna()
    job_count = len(df)
    priced_job_count = len(priced)
    if priced_job_count:
        median = float(priced.median())
        percentile_25 = float(priced.quantile(0.25))
        percentile_75 = float(priced.quantile(0.75))
        if "break_even_per_m3" in df.columns:
            break_even_series = pd.to_numeric(
                df.loc[priced.index, "break_even_per_m3"], errors="coerce"
            ).fillna(break_even)
            comparison_target = break_even_series
        else:
            comparison_target = pd.Series(break_even, index=priced.index)
        below_break_even_count = int((priced < comparison_target).sum())
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


def _format_ratio(ratio: float) -> str:
    if ratio is None or math.isnan(ratio):
        return "n/a"
    return f"{ratio:.1%}"


def _safe_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return math.nan
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    if math.isnan(numeric):
        return math.nan
    return numeric


def _format_corridor_notes(row: pd.Series, break_even: float) -> str:
    details: list[str] = []

    avg_margin_m3 = _safe_float(row.get("avg_margin_per_m3"))
    if not math.isnan(avg_margin_m3):
        details.append(f"Avg margin per m³ ${avg_margin_m3:.2f}")

    avg_margin_pct = _safe_float(row.get("avg_margin_pct"))
    if not math.isnan(avg_margin_pct):
        details.append(f"Avg margin % {_format_ratio(avg_margin_pct)}")

    avg_price_m3 = _safe_float(row.get("avg_price_per_m3"))
    if not math.isnan(avg_price_m3):
        delta = avg_price_m3 - break_even
        details.append(f"Δ vs break-even ${delta:.2f}")

    total_margin = _safe_float(row.get("total_margin"))
    if not math.isnan(total_margin):
        details.append(f"Total margin ${total_margin:.2f}")

    total_volume = _safe_float(row.get("total_volume"))
    if not math.isnan(total_volume):
        details.append(f"Volume {total_volume:.1f} m³")

    job_count = row.get("job_count")
    if pd.notna(job_count):
        details.append(f"Jobs {int(job_count)}")

    return " | ".join(details)


def build_profitability_export(
    df: pd.DataFrame,
    break_even: float,
    *,
    top_n_corridors: int = 3,
) -> pd.DataFrame:
    distribution_summary = summarise_distribution(df, break_even)
    profitability_summary = summarise_profitability(df)

    rows: list[dict[str, object]] = []

    def append_row(section: str, metric: str, value: object, unit: str = "", notes: str = "") -> None:
        rows.append(
            {
                "section": section,
                "metric": metric,
                "value": value,
                "unit": unit,
                "notes": notes,
            }
        )

    append_row("Assumptions", "Break-even assumption", float(break_even), "$/m³")
    append_row("Distribution", "Jobs analysed", distribution_summary.job_count, "jobs")
    append_row("Distribution", "Jobs with price", distribution_summary.priced_job_count, "jobs")

    if not math.isnan(distribution_summary.median):
        append_row("Distribution", "Median price per m³", distribution_summary.median, "$/m³")
        append_row("Distribution", "25th percentile price per m³", distribution_summary.percentile_25, "$/m³")
        append_row("Distribution", "75th percentile price per m³", distribution_summary.percentile_75, "$/m³")
        append_row("Distribution", "Mean price per m³", distribution_summary.mean, "$/m³")
        append_row("Distribution", "Std deviation", distribution_summary.std_dev, "$/m³")
        append_row("Distribution", "Kurtosis", distribution_summary.kurtosis)
        append_row("Distribution", "Skewness", distribution_summary.skewness)

    below_ratio_note = f"{_format_ratio(distribution_summary.below_break_even_ratio)} of priced jobs"
    append_row(
        "Distribution",
        "Below break-even jobs",
        distribution_summary.below_break_even_count,
        "jobs",
        below_ratio_note,
    )

    append_row("Profitability", "Median revenue per km", profitability_summary.revenue_per_km_median, "$/km")
    append_row("Profitability", "Mean revenue per km", profitability_summary.revenue_per_km_mean, "$/km")
    append_row("Profitability", "Median margin per m³", profitability_summary.margin_per_m3_median, "$/m³")
    append_row("Profitability", "Median margin per m³ %", profitability_summary.margin_per_m3_pct_median, "ratio")
    append_row("Profitability", "Median margin total", profitability_summary.margin_total_median, "$")
    append_row("Profitability", "Median margin total %", profitability_summary.margin_total_pct_median, "ratio")

    if "price_per_m3" in df.columns:
        price_series = pd.to_numeric(df["price_per_m3"], errors="coerce")
        bands = price_series.apply(lambda value: classify_profit_band(value, break_even))
        band_counts = bands.value_counts()
        priced_jobs = len(price_series.dropna()) or 1
        for label, count in band_counts.items():
            ratio = _format_ratio(count / priced_jobs)
            append_row(
                "Profitability",
                f"Band - {label}",
                int(count),
                "jobs",
                f"{ratio} of priced jobs",
            )

    if "corridor_display" in df.columns:
        numeric_df = df.copy()
        column_pairs = {
            "price_per_m3_numeric": "price_per_m3",
            "margin_per_m3_numeric": "margin_per_m3",
            "margin_per_m3_pct_numeric": "margin_per_m3_pct",
            "margin_total_numeric": "margin_total",
            "volume_m3_numeric": "volume_m3",
        }
        for numeric_column, source_column in column_pairs.items():
            if source_column in numeric_df.columns:
                numeric_df[numeric_column] = pd.to_numeric(
                    numeric_df[source_column], errors="coerce"
                )
            else:
                numeric_df[numeric_column] = math.nan

        grouped = numeric_df.groupby("corridor_display", dropna=False)
        corridor_stats = grouped.agg(
            avg_price_per_m3=("price_per_m3_numeric", "mean"),
            avg_margin_per_m3=("margin_per_m3_numeric", "mean"),
            avg_margin_pct=("margin_per_m3_pct_numeric", "mean"),
            total_margin=("margin_total_numeric", "sum"),
            total_volume=("volume_m3_numeric", "sum"),
        )
        corridor_stats["job_count"] = grouped.size()
        corridor_stats = corridor_stats.replace({np.inf: math.nan, -np.inf: math.nan})

        if not corridor_stats.empty and corridor_stats["avg_margin_per_m3"].notna().any():
            sorted_corridors = corridor_stats.sort_values("avg_margin_per_m3", ascending=False)
            for idx, (corridor, row) in enumerate(sorted_corridors.head(top_n_corridors).iterrows(), start=1):
                append_row(
                    "Optimisation",
                    f"Top corridor #{idx} by avg margin per m³",
                    corridor,
                    notes=_format_corridor_notes(row, break_even),
                )

            bottom_corridors = sorted_corridors.tail(top_n_corridors).sort_values("avg_margin_per_m3", ascending=True)
            for idx, (corridor, row) in enumerate(bottom_corridors.iterrows(), start=1):
                append_row(
                    "Optimisation",
                    f"Lowest margin corridor #{idx}",
                    corridor,
                    notes=_format_corridor_notes(row, break_even),
                )

        if "avg_price_per_m3" in corridor_stats.columns:
            below_break_even = corridor_stats[corridor_stats["avg_price_per_m3"] < float(break_even)]
            if not below_break_even.empty:
                entries: list[str] = []
                for corridor, row in below_break_even.sort_values("avg_price_per_m3").iterrows():
                    avg_price = _safe_float(row.get("avg_price_per_m3"))
                    if math.isnan(avg_price):
                        continue
                    delta = avg_price - float(break_even)
                    job_count = int(row.get("job_count", 0))
                    entries.append(f"{corridor} ({job_count} jobs, Δ ${delta:.2f})")
                if entries:
                    append_row(
                        "Optimisation",
                        "Corridors below break-even",
                        ", ".join(entries),
                        notes="Negative Δ indicates pricing below break-even",
                    )

    return pd.DataFrame(rows, columns=["section", "metric", "value", "unit", "notes"])


def classify_profit_band(value: Optional[float], break_even: float) -> str:
    if value is None:
        return "Unknown"
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "Unknown"
    if math.isnan(numeric_value):
        return "Unknown"

    diff = numeric_value - break_even
    for lower, upper, label in PROFITABILITY_BANDS:
        if lower <= diff < upper:
            return label
    return "Unknown"


def classify_profitability_status(
    value: Optional[float],
    break_even: float,
    *,
    abs_tolerance: float = DEFAULT_BREAK_EVEN_ABS_TOLERANCE,
    rel_tolerance: float = DEFAULT_BREAK_EVEN_REL_TOLERANCE,
) -> str:
    if value is None:
        return "Unknown"
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "Unknown"
    if math.isnan(numeric_value):
        return "Unknown"

    diff = numeric_value - break_even
    tolerance = max(abs_tolerance, abs(break_even) * rel_tolerance)
    if abs(diff) <= tolerance:
        return "Break-even"
    if diff > 0:
        return "Profitable"
    return "Loss-leading"
