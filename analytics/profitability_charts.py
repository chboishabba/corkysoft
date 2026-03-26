from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

METRO_HISTOGRAM_BINS = 15


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

    for pct, (label, color, dash) in _band_styles().items():
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
        bin_width = float(np.mean(np.diff(bin_edges))) if len(bin_edges) > 1 else 0.0
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


def filter_metro_jobs(df: pd.DataFrame, max_distance_km: float = 100.0) -> pd.DataFrame:
    if "distance_km" not in df.columns:
        return df.copy()
    distances = df["distance_km"].astype(float)
    within_threshold = distances.fillna(np.inf) <= max_distance_km
    return df.loc[within_threshold].copy()


def create_m3_vs_km_figure(df: pd.DataFrame) -> go.Figure:
    if "price_per_m3" not in df.columns or "revenue_per_km" not in df.columns:
        return _empty_figure("m³ vs km profitability", "$ per m³", "$ per km", "No revenue or distance data available.")

    subset = df.dropna(subset=["price_per_m3", "revenue_per_km"])
    if subset.empty:
        return _empty_figure("m³ vs km profitability", "$ per m³", "$ per km", "Add jobs with both revenue and distance to unlock this view.")

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
    fig.update_layout(xaxis=dict(zeroline=False), yaxis=dict(zeroline=False), legend_title_text="Corridor" if color_col else None)
    return fig


def create_m3_margin_figure(df: pd.DataFrame) -> go.Figure:
    required_cols = {"price_per_m3", "final_cost_per_m3"}
    if not required_cols.issubset(df.columns):
        return _empty_figure("Quoted vs calculated $/m³", "Cost-derived $ per m³", "Quoted $ per m³", "Final cost data is unavailable.")

    subset = df.dropna(subset=list(required_cols))
    if subset.empty:
        return _empty_figure("Quoted vs calculated $/m³", "Cost-derived $ per m³", "Quoted $ per m³", "No jobs contain both quoted and calculated $/m³ values.")

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
        color_args = {"color": subset[color_col], "color_continuous_scale": "RdYlGn"}

    fig = px.scatter(
        subset,
        x="final_cost_per_m3",
        y="price_per_m3",
        hover_data=hover_data,
        labels={"final_cost_per_m3": "Cost-derived $ per m³", "price_per_m3": "Quoted $ per m³"},
        title="Quoted vs calculated $/m³",
        **color_args,
    )
    fig.update_traces(marker=dict(size=10, opacity=0.8))

    min_val = float(subset[["final_cost_per_m3", "price_per_m3"]].min().min())
    max_val = float(subset[["final_cost_per_m3", "price_per_m3"]].max().max())
    if max_val > min_val:
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(color="rgba(17, 17, 17, 0.6)", dash="dash"),
                name="Parity",
                showlegend=False,
            )
        )

    fig.update_layout(coloraxis_colorbar=dict(title="Margin %"), legend_title_text=None)
    return fig


def create_metro_profitability_figure(
    df: pd.DataFrame, *, max_distance_km: float = 100.0
) -> go.Figure:
    title = f"Metro profitability (≤{max_distance_km:,.0f} km)"
    metro_df = filter_metro_jobs(df, max_distance_km=max_distance_km)

    if metro_df.empty:
        return _empty_figure(title, "$ per m³", "$ per km", "No jobs fall within the metro distance threshold.")

    required = {"price_per_m3", "revenue_per_km"}
    missing = required - set(metro_df.columns)
    if missing:
        return _empty_figure(title, "$ per m³", "$ per km", f"Metro view requires columns: {', '.join(sorted(missing))}.")

    scatter_df = metro_df.dropna(subset=list(required))
    if scatter_df.empty:
        return _empty_figure(title, "$ per m³", "$ per km", "Metro jobs lack both price and revenue per km values.")

    margin_available = "margin_per_m3" in metro_df.columns and not metro_df["margin_per_m3"].dropna().empty
    cost_ratio_series = pd.Series(dtype=float)
    if "final_cost_per_m3" in metro_df.columns:
        ratio_df = metro_df.dropna(subset=["final_cost_per_m3", "price_per_m3"])
        if not ratio_df.empty:
            denom = ratio_df["price_per_m3"].replace(0, np.nan)
            cost_ratio_series = (ratio_df["final_cost_per_m3"] / denom).replace([np.inf, -np.inf], np.nan).dropna()
    cost_available = not cost_ratio_series.empty

    subplot_titles = ["Prices ($/m³)"]
    specs: list[dict[str, str]] = [{"type": "xy"}]
    if margin_available:
        subplot_titles.append("Margins ($/m³)")
        specs.append({"type": "xy"})
    if cost_available:
        subplot_titles.append("Cost vs quote share")
        specs.append({"type": "xy"})

    fig = make_subplots(rows=1, cols=len(subplot_titles), subplot_titles=subplot_titles, specs=[specs], horizontal_spacing=0.08)

    hover_bits: list[list[str]] = []
    hover_columns = [
        ("client_display", "Client"),
        ("corridor_display", "Corridor"),
        ("job_date", "Date"),
        ("volume_m3", "Volume (m³)"),
        ("distance_km", "Distance (km)"),
        ("margin_per_m3", "Margin $/m³"),
        ("margin_per_m3_pct", "Margin %"),
    ]
    for _, row in scatter_df.iterrows():
        parts = [f"Quoted $/m³: {row['price_per_m3']:,.2f}", f"$ per km: {row['revenue_per_km']:,.2f}"]
        for column, label in hover_columns:
            if column not in scatter_df.columns:
                continue
            value = row.get(column)
            if pd.isna(value):
                continue
            if column.endswith("pct"):
                parts.append(f"{label}: {value * 100:.1f}%")
            elif isinstance(value, (int, float)):
                parts.append(f"{label}: {value:,.2f}")
            else:
                parts.append(f"{label}: {value}")
        hover_bits.append(parts)

    marker_args: dict[str, object] = {"size": 10, "opacity": 0.85}
    if "margin_per_m3_pct" in scatter_df.columns and not scatter_df["margin_per_m3_pct"].dropna().empty:
        marker_args.update(
            {
                "color": scatter_df["margin_per_m3_pct"],
                "colorscale": "RdYlGn",
                "showscale": True,
                "colorbar": {"title": "Margin % sensitivity", "tickformat": ".0%"},
            }
        )

    fig.add_trace(
        go.Scatter(
            x=scatter_df["price_per_m3"],
            y=scatter_df["revenue_per_km"],
            mode="markers",
            name="Metro jobs",
            marker=marker_args,
            hovertemplate="%{text}<extra></extra>",
            text=["<br>".join(parts) for parts in hover_bits],
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="Prices ($/m³)", row=1, col=1)
    fig.update_yaxes(title_text="Revenue ($/km)", row=1, col=1)

    current_col = 2
    if margin_available:
        fig.add_trace(
            go.Histogram(
                x=metro_df["margin_per_m3"].dropna(),
                name="Margins ($/m³)",
                nbinsx=METRO_HISTOGRAM_BINS,
                marker=dict(color="rgba(91, 192, 222, 0.85)"),
                hovertemplate="Margin $/m³: %{x:,.2f}<br>Count: %{y}<extra></extra>",
            ),
            row=1,
            col=current_col,
        )
        fig.update_xaxes(title_text="Margins ($/m³)", row=1, col=current_col)
        fig.update_yaxes(title_text="Job count", row=1, col=current_col)
        current_col += 1

    if cost_available:
        fig.add_trace(
            go.Histogram(
                x=cost_ratio_series,
                name="Cost sensitivity",
                nbinsx=METRO_HISTOGRAM_BINS,
                marker=dict(color="rgba(217, 83, 79, 0.7)"),
                hovertemplate="Cost/price: %{x:.1%}<br>Count: %{y}<extra></extra>",
            ),
            row=1,
            col=current_col,
        )
        fig.update_xaxes(title_text="Cost as share of quoted price", tickformat=".0%", row=1, col=current_col)
        fig.update_yaxes(title_text="Job count", row=1, col=current_col)

    fig.update_layout(
        title=title,
        bargap=0.05,
        hovermode="closest",
        legend_title_text=None,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    return fig
