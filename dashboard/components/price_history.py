"""Price history tab rendering helpers."""
from __future__ import annotations

from datetime import date
from typing import Any, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.lane_scope import apply_lane_status_scope
from analytics.price_distribution import (
    ColumnMapping,
    build_price_history_series,
    summarise_last_year_distributions,
)


def _with_display_period(frame: pd.DataFrame, *, year_offset: int = 0) -> pd.DataFrame:
    """Attach a display period column for timeline plotting."""
    if frame.empty or "period" not in frame.columns:
        return frame

    result = frame.copy()
    result["display_period"] = pd.to_datetime(result["period"], errors="coerce")
    if year_offset:
        result["display_period"] = result["display_period"] + pd.DateOffset(
            years=year_offset
        )
    return result


def render_price_history_tab(
    filtered_df: pd.DataFrame,
    mapping: ColumnMapping,
    start_date: Optional[date],
    end_date: Optional[date],
) -> None:
    """Render the price history dashboard tab."""
    scoped_df = apply_lane_status_scope(
        filtered_df,
        scope_key="price_history_lane_assignment_scope",
        label="Lane assignment scope",
        help_text=(
            "Price-history trends default to canonically assigned lane history. "
            "Include ambiguous or unassigned rows only when exploring unresolved data."
        ),
        caption_prefix="Price-history rows after lane-status filter",
    )
    has_filtered_data = not scoped_df.empty
    date_column = mapping.date or "job_date"

    if not has_filtered_data:
        st.info("Apply filters or import jobs to analyse price history trends.")
        return

    if date_column not in scoped_df.columns:
        st.warning("No date column available to build the price history view.")
        return

    st.markdown("### Price history")
    frequency_options = {"Daily": "daily", "Weekly": "weekly", "Monthly": "monthly"}
    selected_frequency_label = st.radio(
        "Aggregation frequency",
        list(frequency_options.keys()),
        horizontal=True,
        key="price_history_frequency",
    )
    frequency_value = frequency_options[selected_frequency_label]

    dates = pd.to_datetime(scoped_df[date_column], errors="coerce")
    valid_dates = dates.dropna()
    default_start = (
        start_date or (valid_dates.min().date() if not valid_dates.empty else date.today())
    )
    default_end = (
        end_date or (valid_dates.max().date() if not valid_dates.empty else date.today())
    )

    history_range_kwargs: dict[str, Any] = {"key": "price_history_range"}
    if not valid_dates.empty:
        history_range_kwargs["min_value"] = valid_dates.min().date()
        history_range_kwargs["max_value"] = valid_dates.max().date()

    history_range = st.date_input(
        "History range",
        value=(default_start, default_end),
        **history_range_kwargs,
    )

    if isinstance(history_range, tuple) and len(history_range) == 2:
        history_start, history_end = history_range
    else:
        history_start, history_end = default_start, default_end

    if history_start > history_end:
        st.warning("Select a start date that precedes the end date to render the charts.")
        return

    history_series = build_price_history_series(
        scoped_df,
        frequency=frequency_value,
        date_column=date_column,
        start_date=history_start,
        end_date=history_end,
    )
    previous_year_frames = summarise_last_year_distributions(
        scoped_df,
        date_column=date_column,
        start_date=history_start,
        end_date=history_end,
    )

    metric_labels = {
        "Price $/m³": "price_per_m3",
        "Margin $/m³": "margin_per_m3",
        "Margin %": "margin_total_pct",
    }

    current_overall = _with_display_period(
        history_series.current_overall.assign(series="Current period")
    )
    previous_overall = _with_display_period(
        history_series.previous_year_overall.assign(series="Previous year"),
        year_offset=1,
    )
    combined_overall = pd.concat(
        [frame for frame in (current_overall, previous_overall) if not frame.empty],
        ignore_index=True,
    )

    if combined_overall.empty:
        st.info("Not enough data to build an aggregated price history chart.")
    else:
        metric_columns = [
            column
            for column in metric_labels.values()
            if column in combined_overall.columns
        ]
        if not metric_columns:
            st.info(
                "Time series metrics are missing for the selected range. Add price or margin columns to compare trends."
            )
        else:
            melted = combined_overall.melt(
                id_vars=["display_period", "series"],
                value_vars=metric_columns,
                var_name="metric",
                value_name="value",
            ).dropna(subset=["value", "display_period"])
            if melted.empty:
                st.info(
                    "Time series metrics are missing for the selected range. Add price or margin columns to compare trends."
                )
            else:
                overall_fig = px.line(
                    melted,
                    x="display_period",
                    y="value",
                    color="metric",
                    line_dash="series",
                    markers=True,
                    labels={
                        "display_period": "Period",
                        "value": "Value",
                        "metric": "Metric",
                        "series": "Series",
                    },
                )
                overall_fig.update_layout(legend_title_text="Metric comparison")
                st.plotly_chart(overall_fig, width="stretch")

    available_metric_labels = [
        label
        for label, column in metric_labels.items()
        if column in history_series.current_overall.columns
        or column in history_series.current_by_origin.columns
        or column in history_series.current_by_destination.columns
        or column in history_series.previous_year_overall.columns
        or column in history_series.previous_year_by_origin.columns
        or column in history_series.previous_year_by_destination.columns
    ]

    metric_label = None
    metric_column = None
    if available_metric_labels:
        metric_label = st.selectbox(
            "Breakdown metric",
            options=available_metric_labels,
            key="price_history_metric",
            help="Choose which metric drives the origin and destination breakdown charts.",
        )
        metric_column = metric_labels[metric_label]
    else:
        st.info("Add price or margin columns to explore origin and destination breakdowns.")

    metric_display_label = metric_label or "Value"

    if metric_column:
        current_origin = _with_display_period(
            history_series.current_by_origin.assign(series="Current period")
        )
        previous_origin = _with_display_period(
            history_series.previous_year_by_origin.assign(series="Previous year"),
            year_offset=1,
        )
        origin_combined = pd.concat(
            [frame for frame in (current_origin, previous_origin) if not frame.empty],
            ignore_index=True,
        )

        current_destination = _with_display_period(
            history_series.current_by_destination.assign(series="Current period")
        )
        previous_destination = _with_display_period(
            history_series.previous_year_by_destination.assign(series="Previous year"),
            year_offset=1,
        )
        destination_combined = pd.concat(
            [
                frame
                for frame in (current_destination, previous_destination)
                if not frame.empty
            ],
            ignore_index=True,
        )

        origin_col, destination_col = st.columns(2)
        if (
            origin_combined.empty
            or metric_column not in origin_combined.columns
            or origin_combined.dropna(subset=[metric_column]).empty
        ):
            origin_col.info("No origin-level data available for the selected metric.")
        else:
            origin_fig = px.line(
                origin_combined.dropna(subset=[metric_column, "display_period"]),
                x="display_period",
                y=metric_column,
                color="origin",
                line_dash="series",
                markers=True,
                labels={
                    "display_period": "Period",
                    "origin": "Origin",
                    metric_column: metric_display_label,
                    "series": "Series",
                },
            )
            origin_fig.update_layout(legend_title_text="Origin")
            origin_col.plotly_chart(origin_fig, width="stretch")

        if (
            destination_combined.empty
            or metric_column not in destination_combined.columns
            or destination_combined.dropna(subset=[metric_column]).empty
        ):
            destination_col.info(
                "No destination-level data available for the selected metric."
            )
        else:
            destination_fig = px.line(
                destination_combined.dropna(subset=[metric_column, "display_period"]),
                x="display_period",
                y=metric_column,
                color="destination",
                line_dash="series",
                markers=True,
                labels={
                    "display_period": "Period",
                    "destination": "Destination",
                    metric_column: metric_display_label,
                    "series": "Series",
                },
            )
            destination_fig.update_layout(legend_title_text="Destination")
            destination_col.plotly_chart(destination_fig, width="stretch")

    st.markdown("#### Previous year distribution snapshots")
    histogram_metric = None
    previous_overall_frame = previous_year_frames.get("overall", pd.DataFrame())
    if not previous_overall_frame.empty:
        if "price_per_m3" in previous_overall_frame.columns:
            histogram_metric = "price_per_m3"
        elif metric_column and metric_column in previous_overall_frame.columns:
            histogram_metric = metric_column
    if histogram_metric:
        hist_source = previous_overall_frame.dropna(subset=[histogram_metric])
        if hist_source.empty:
            st.info("No numeric values available to plot the previous year histogram.")
        else:
            hist_fig = px.histogram(
                hist_source,
                x=histogram_metric,
                nbins=25,
                color_discrete_sequence=[px.colors.qualitative.Plotly[0]],
                labels={
                    histogram_metric: (
                        metric_display_label
                        if histogram_metric == metric_column
                        else "Price $/m³"
                    )
                },
                title="Previous year distribution",
            )
            st.plotly_chart(hist_fig, width="stretch")
    else:
        st.info(
            "Historical dataset from the previous year is unavailable for distribution comparisons."
        )

    if metric_column:
        comparison_columns = st.columns(2)
        previous_origin = previous_year_frames.get("by_origin", pd.DataFrame())
        if (
            previous_origin.empty
            or metric_column not in previous_origin.columns
            or previous_origin.dropna(subset=[metric_column]).empty
        ):
            comparison_columns[0].info(
                "No previous-year origin data to summarise as a box plot."
            )
        else:
            origin_box = px.box(
                previous_origin.dropna(subset=[metric_column]),
                x="origin",
                y=metric_column,
                points="outliers",
                labels={
                    "origin": "Origin",
                    metric_column: metric_display_label,
                },
                title="Origin spread (previous year)",
            )
            comparison_columns[0].plotly_chart(origin_box, width="stretch")

        previous_destination = previous_year_frames.get("by_destination", pd.DataFrame())
        if (
            previous_destination.empty
            or metric_column not in previous_destination.columns
            or previous_destination.dropna(subset=[metric_column]).empty
        ):
            comparison_columns[1].info(
                "No previous-year destination data to summarise as a box plot."
            )
        else:
            destination_box = px.box(
                previous_destination.dropna(subset=[metric_column]),
                x="destination",
                y=metric_column,
                points="outliers",
                labels={
                    "destination": "Destination",
                    metric_column: metric_display_label,
                },
                title="Destination spread (previous year)",
            )
            comparison_columns[1].plotly_chart(destination_box, width="stretch")
