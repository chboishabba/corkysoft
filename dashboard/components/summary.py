"""Summary metric components for the dashboard."""
from __future__ import annotations

import math
from typing import Optional

import streamlit as st

from analytics.price_distribution import DistributionSummary, ProfitabilitySummary

__all__ = ["render_summary"]


def _format_value(
    value: Optional[float], *, currency: bool = False, percentage: bool = False
) -> str:
    """Format ``value`` for display in a Streamlit metric widget."""

    if value is None:
        return "n/a"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "n/a"
    if currency:
        return f"${value:,.2f}"
    if percentage:
        return f"{value * 100:.1f}%"
    return f"{value:,.2f}"


def render_summary(
    summary: DistributionSummary,
    break_even: float,
    profitability_summary: ProfitabilitySummary,
    *,
    metro_summary: Optional[DistributionSummary] = None,
    metro_profitability: Optional[ProfitabilitySummary] = None,
    metro_distance_km: float = 100.0,
) -> None:
    """Render the headline price and profitability metrics."""

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
        st.markdown(f"**Metro subset (≤{metro_distance_km:,.0f} km)**")
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
