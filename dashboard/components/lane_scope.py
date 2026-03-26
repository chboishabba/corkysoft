"""Shared lane-assignment scope helpers for dashboard surfaces."""
from __future__ import annotations

import pandas as pd
import streamlit as st


LANE_STATUS_ORDER = ("assigned", "ambiguous", "unassigned")


def apply_lane_status_scope(
    df: pd.DataFrame,
    *,
    scope_key: str,
    label: str,
    help_text: str,
    caption_prefix: str | None = None,
) -> pd.DataFrame:
    """Normalize and filter a dataframe by lane-assignment status."""

    scoped = df.copy()
    if "lane_assignment_status" not in scoped.columns:
        return scoped

    normalized_status = (
        scoped["lane_assignment_status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .replace("", "unassigned")
    )
    scoped = scoped.assign(lane_assignment_status=normalized_status)

    lane_status_options = [
        status
        for status in LANE_STATUS_ORDER
        if normalized_status.eq(status).any()
    ]
    if not lane_status_options:
        return scoped

    selected_lane_statuses = st.multiselect(
        label,
        options=lane_status_options,
        default=["assigned"] if "assigned" in lane_status_options else lane_status_options,
        help=help_text,
        key=scope_key,
    )
    scoped = scoped.loc[
        scoped["lane_assignment_status"].isin(selected_lane_statuses)
    ].copy()

    if caption_prefix:
        st.caption(f"{caption_prefix}: {len(scoped)}")
    return scoped


__all__ = ["apply_lane_status_scope"]
