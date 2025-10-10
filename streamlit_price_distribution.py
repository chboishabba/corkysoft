"""Streamlit dashboard for the price distribution analysis."""
from __future__ import annotations

import io
import math
from datetime import date
from typing import Optional

import pandas as pd
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
    summarise_distribution,
    summarise_profitability,
    update_break_even,
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


with connection_scope() as conn:
    break_even_value = ensure_break_even_parameter(conn)

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

    filtered_df, _ = load_historical_jobs(
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
    render_summary(summary, break_even_value, profitability_summary)

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
