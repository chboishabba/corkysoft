from __future__ import annotations

import math
import sqlite3
from datetime import date
from typing import Any, Mapping, Optional

import pandas as pd
import streamlit as st

from analytics.live_data import load_active_routes, load_truck_positions
from analytics.price_distribution import (
    ColumnMapping,
    DistributionSummary,
    ProfitabilitySummary,
    compute_cost_vs_price_percentage,
    create_histogram,
    create_m3_margin_figure,
    create_m3_vs_km_figure,
    create_metro_profitability_figure,
    filter_jobs_by_distance,
    prepare_profitability_route_data,
    summarise_distribution,
    summarise_profitability,
)
from dashboard.components.lane_scope import apply_lane_status_scope
from dashboard.components.maps import render_network_map
from dashboard.components.price_history import render_price_history_tab


def _filter_by_distance(
    df: pd.DataFrame,
    *,
    metro_only: bool = False,
    max_distance_km: float = 100.0,
) -> pd.DataFrame:
    return filter_jobs_by_distance(
        df,
        metro_only=metro_only,
        max_distance_km=max_distance_km,
    )


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


def render_distribution_analytics_surface(
    *,
    tab_map: Mapping[str, Any],
    filtered_df: pd.DataFrame,
    filtered_mapping: ColumnMapping,
    break_even_value: float,
    dataset_error: str | None,
    conn: sqlite3.Connection,
    start_date: Optional[date],
    end_date: Optional[date],
) -> None:
    has_filtered_data = not filtered_df.empty
    metro_distance_km = 100.0
    summary_scope_df = filtered_df

    if has_filtered_data:
        filtered_df = filtered_df.copy()
        filtered_df["cost_vs_price_pct"] = compute_cost_vs_price_percentage(filtered_df)
        summary_scope_df = apply_lane_status_scope(
            filtered_df,
            scope_key="dashboard_summary_lane_scope",
            label="Lane assignment scope",
            help_text=(
                "These analytics default to canonically assigned lane history. "
                "Include ambiguous or unassigned rows only when deliberately exploring unresolved data."
            ),
            caption_prefix="Summary/histogram rows after lane-status filter",
        )

        summary = summarise_distribution(summary_scope_df, break_even_value)
        profitability_summary = summarise_profitability(summary_scope_df)

        metro_df = _filter_by_distance(
            summary_scope_df, metro_only=True, max_distance_km=metro_distance_km
        )
        metro_summary = None
        metro_profitability = None
        if not metro_df.empty:
            metro_summary = summarise_distribution(metro_df, break_even_value)
            metro_profitability = summarise_profitability(metro_df)

        render_summary(
            summary,
            break_even_value,
            profitability_summary,
            metro_summary=metro_summary,
            metro_profitability=metro_profitability,
            metro_distance_km=metro_distance_km,
        )

    if "Live network overview" in tab_map:
        with tab_map["Live network overview"]:
            network_df = apply_lane_status_scope(
                filtered_df,
                scope_key="dashboard_live_network_lane_scope",
                label="Lane assignment scope",
                help_text=(
                    "These analytics default to canonically assigned lane history. "
                    "Include ambiguous or unassigned rows only when deliberately exploring unresolved data."
                ),
                caption_prefix="Live-network analytic rows after lane-status filter",
            )
            render_network_map(
                prepare_profitability_route_data(network_df, break_even_value),
                load_truck_positions(conn),
                load_active_routes(conn),
                toggle_key="dashboard_network_map_toggle_overview",
            )

    if "Histogram" in tab_map:
        with tab_map["Histogram"]:
            if has_filtered_data:
                with st.popover("❓ Histogram stats", width="stretch"):
                    st.markdown(
                        """
                        ### **Break-even bands**
                        Vertical guide-lines centred on your **break-even $/m³**.

                        Each band shows your break-even target (**$/m³ needed to make profit**), along with percentages indicating how far real jobs fall above or below it.

                        You can quickly see:

                        - **Which corridors frequently underperform**
                        - **Whether a client consistently prices below your minimum**
                        - **How much “safety margin” you have on metro jobs**
                        - The **normal-fit overlay**, showing a bell curve fitted to your $/m³ distribution
                        - Real-world job pricing is messy and often skewed — the normal fit gives an *idealised baseline*.
                        - These are methods derived from the formal study of statistics and can be used to inform operators and managers about pricing trends.

                        ---

                        ### **Reading the curve**
                        - **Skew** → are there lots of cheap jobs or lots of expensive jobs?
                        - **Fat tails** → outliers on either side
                        - **Pricing stability** → is your pricing consistent or chaotic?

                        **Tall, narrow curve** → stable, predictable pricing
                        **Wide, flat curve** → highly variable pricing

                        ---

                        ### **Summary statistics**
                        These quantify the shape and behaviour of your pricing:

                        - **Percentiles** — e.g., 75th percentile means a value is higher than 75% of jobs.
                        - **Mean (μ)** — your overall *average* revenue density.
                        - **Median** — midpoint of all jobs.
                        More stable than the mean when outliers exist.
                        - **Standard deviation (σ)** — measures volatility.
                        **High σ** = inconsistent pricing; **Low σ** = tightly clustered pricing.
                        - **Kurtosis** — how “outlier-heavy” your distribution is.
                        Over > 3 = fat tails - some data is very unlike others; Under < 3 = tighter, more predictable.
                        - **Skewness** — asymmetry.
                        **Positive skew** → many cheap jobs, few expensive ones.
                        **Negative skew** → many expensive jobs, few cheap ones.
                        - **% below break-even** — proportion of unprofitable jobs.
                        **Ideal:** 0–10% **Warning:** 20–30% **Critical:** >30%
                        """,
                        unsafe_allow_html=True,
                    )

                histogram = create_histogram(summary_scope_df, break_even_value)
                st.plotly_chart(histogram, width="stretch")
                st.caption(
                    "Histogram overlays include the normal distribution fit plus kurtosis and dispersion markers for context."
                )
            elif dataset_error:
                st.error("Unable to load jobs — initialise the database and retry.")
            else:
                st.info("Import historical jobs to plot the price distribution histogram.")

    if "Price history" in tab_map:
        with tab_map["Price history"]:
            render_price_history_tab(
                filtered_df=filtered_df,
                mapping=filtered_mapping,
                start_date=start_date,
                end_date=end_date,
            )

    if "Profitability insights" in tab_map:
        with tab_map["Profitability insights"]:
            if has_filtered_data:
                profitability_df = apply_lane_status_scope(
                    filtered_df,
                    scope_key="dashboard_profitability_lane_scope",
                    label="Lane assignment scope",
                    help_text=(
                        "These analytics default to canonically assigned lane history. "
                        "Include ambiguous or unassigned rows only when deliberately exploring unresolved data."
                    ),
                    caption_prefix="Profitability analytic rows after lane-status filter",
                )
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
                    key="dashboard_profitability_view",
                )
                st.plotly_chart(view_options[selected_view](profitability_df), width="stretch")

                if selected_view == "Metro profitability spotlight":
                    st.caption(
                        "Metro view highlights close-in routes with margin and cost sensitivity overlays."
                    )

                if "margin_per_m3" in profitability_df.columns:
                    st.markdown("#### Margin outliers")
                    ranked = profitability_df.dropna(subset=["margin_per_m3"]).sort_values(
                        "margin_per_m3"
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
            elif dataset_error:
                st.error("Unable to calculate profitability without job data.")
            else:
                st.info("Import jobs with price and cost data to unlock profitability insights.")
