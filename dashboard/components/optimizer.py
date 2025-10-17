"""Streamlit components for the dashboard optimizer tab."""
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from analytics.optimizer import (
    OptimizerParameters,
    OptimizerRun,
    can_run_optimizer,
    recommendations_to_frame,
    run_margin_optimizer,
)


def _get_optimizer_state() -> Dict[str, Any]:
    """Return the optimizer state stored in the Streamlit session."""
    return st.session_state.setdefault("optimizer_state", {})


def _render_optimizer_form(defaults: Dict[str, Any]) -> Optional[OptimizerParameters]:
    """Render the optimizer form and return parameters when submitted."""
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

    if not submitted:
        return None

    return OptimizerParameters(
        target_margin_per_m3=target_margin,
        max_uplift_pct=max_uplift_pct,
        min_job_count=min_job_count,
    )


def _render_optimizer_results(run: OptimizerRun) -> None:
    """Display optimizer results including charts and downloads."""
    run_time = run.executed_at.strftime("%Y-%m-%d %H:%M UTC")
    st.caption(
        f"Last run: {run_time} · Target margin ${run.parameters.target_margin_per_m3:,.0f}/m³ · "
        f"Max uplift {run.parameters.max_uplift_pct:.0f}%"
    )

    recommendations_df = recommendations_to_frame(run.recommendations)
    if recommendations_df.empty:
        st.info(
            "No eligible corridors were found — adjust parameters or widen the dashboard filters."
        )
        return

    metric_cols = st.columns(3)
    metric_cols[0].metric("Corridors analysed", len(recommendations_df))
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


def render_optimizer(filtered_df: pd.DataFrame) -> None:
    """Render the optimizer workflow for the provided filtered dataframe."""
    st.markdown("### Margin optimizer")
    st.caption("Generate corridor-level price uplift suggestions using the filtered job set.")

    optimizer_state = _get_optimizer_state()

    if not can_run_optimizer(filtered_df):
        st.info(
            "Optimizer requires price and cost per m³ columns. Import jobs with $ / m³ and cost data to enable recommendations."
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

        params = _render_optimizer_form(defaults)
        if params:
            run = run_margin_optimizer(filtered_df, params)
            optimizer_state["last_run"] = run
            optimizer_state["defaults"] = {
                "target_margin": params.target_margin_per_m3,
                "max_uplift": params.max_uplift_pct,
                "min_job_count": params.min_job_count,
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
            _render_optimizer_results(last_run)

    st.info(
        "Optimizer works on the same filters applied across the dashboard, making it safe for non-technical teams to explore 'what if' pricing scenarios."
    )
