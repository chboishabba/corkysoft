"""Streamlit components for vehicle maintenance history."""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import streamlit as st

from analytics.vehicle_repairs import (
    import_vehicle_repairs_from_sheet,
    load_vehicle_repairs,
)


def _format_currency(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "$0"
    return f"${value:,.0f}"


def render_vehicle_maintenance_tab(conn) -> None:
    """Render the maintenance history tab."""

    st.markdown("### Vehicle maintenance history")
    st.caption("Import workshop visits from Google Sheets and keep a running log of spend per truck.")

    default_sheet = os.environ.get("VEHICLE_REPAIRS_SHEET_URL") or os.environ.get(
        "VEHICLE_REPAIRS_SHEET"
    )

    with st.expander("Import VEHICLE_REPAIRS sheet", expanded=False):
        sheet_url = st.text_input(
            "Google Sheets CSV/Excel URL",
            value=default_sheet or "",
            help=(
                "Paste the CSV export link for the VEHICLE_REPAIRS sheet. "
                "Both CSV and XLSX feeds are supported."
            ),
            key="vehicle_repairs_sheet_url",
        )
        if st.button("Import vehicle repairs", key="vehicle_repairs_import_button"):
            if not sheet_url:
                st.error("Provide a Google Sheets URL before importing.")
            else:
                try:
                    inserted, updated = import_vehicle_repairs_from_sheet(
                        conn, sheet_url=sheet_url
                    )
                except Exception as exc:  # pragma: no cover - surfaced in UI only
                    st.error(f"Failed to import repairs: {exc}")
                else:
                    st.success(
                        f"Imported {inserted} new record{'s' if inserted != 1 else ''}. "
                        f"Updated {updated} existing row{'s' if updated != 1 else ''}."
                    )

    repairs_df = load_vehicle_repairs(conn)
    if repairs_df.empty:
        st.info(
            "No vehicle repairs captured yet. Import the VEHICLE_REPAIRS sheet to populate this view."
        )
        return

    display_df = repairs_df.copy()
    for column in ("service_date", "next_service_date", "created_at", "updated_at"):
        if column in display_df.columns:
            display_df[column] = pd.to_datetime(display_df[column], errors="coerce")

    display_df = display_df.sort_values(
        by=["service_date", "created_at"], ascending=[False, False], na_position="last"
    )

    st.markdown("#### Spend by vehicle")
    spend_summary = (
        display_df.assign(price_numeric=pd.to_numeric(display_df["price"], errors="coerce"))
        .groupby("truck_id")
        .agg(
            total_spend=pd.NamedAgg(column="price_numeric", aggfunc="sum"),
            jobs=pd.NamedAgg(column="job_item", aggfunc="count"),
            last_service=pd.NamedAgg(column="service_date", aggfunc="max"),
        )
        .reset_index()
        .sort_values("total_spend", ascending=False)
    )

    summary_cols = st.columns(max(1, min(3, len(spend_summary))))
    for idx, row in spend_summary.iterrows():
        column = summary_cols[idx % len(summary_cols)]
        column.metric(
            f"{row['truck_id']} spend",
            _format_currency(row["total_spend"]),
            help="Total repair spend captured for this vehicle.",
        )
        column.metric(
            f"{row['truck_id']} jobs",
            f"{int(row['jobs'])}",
            help="Count of repair log entries.",
        )
        if pd.notna(row["last_service"]):
            column.caption(f"Last service: {pd.to_datetime(row['last_service']).date()}")

    st.dataframe(
        spend_summary.rename(
            columns={
                "truck_id": "Vehicle",
                "total_spend": "Spend",
                "jobs": "Repairs logged",
                "last_service": "Most recent service",
            }
        ),
        use_container_width=True,
    )

    st.markdown("#### Repair log")
    st.dataframe(display_df, use_container_width=True)

