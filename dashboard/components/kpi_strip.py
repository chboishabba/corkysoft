"""Reusable KPI strip component for the guided workflow UI.

Renders a horizontal row of metric cards with colour-coded deltas.
"""
from __future__ import annotations

from html import escape
from typing import Sequence

import streamlit as st


def render_kpi_strip(
    metrics: Sequence[dict],
) -> None:
    """Render a horizontal KPI strip.

    Parameters
    ----------
    metrics : list of dict
        Each dict must contain:
            - label: str   — metric name (e.g. "Quote Win Rate")
            - value: str   — display value (e.g. "34%")
        Optional keys:
            - delta: str   — delta text (e.g. "+2%")
            - direction: str — "up" | "down" | "neutral" (default "neutral")
    """
    cards_html_parts: list[str] = []
    for m in metrics:
        direction = m.get("direction", "neutral")
        delta_class = {
            "up": "delta-up",
            "down": "delta-down",
        }.get(direction, "delta-neutral")
        label = escape(str(m.get("label", "")))
        value = escape(str(m.get("value", "")))

        delta_html = ""
        if m.get("delta"):
            arrow = {"up": "▲", "down": "▼"}.get(direction, "—")
            delta_text_class = {"up": "up", "down": "down"}.get(direction, "neutral")
            delta_text = escape(str(m["delta"]))
            delta_html = (
                f'<div class="ck-kpi-delta {delta_text_class}">'
                f"{arrow} {delta_text}</div>"
            )

        cards_html_parts.append(
            f'<div class="ck-kpi-card {delta_class}">'
            f"  <div class=\"ck-kpi-label\">{label}</div>"
            f"  <div class=\"ck-kpi-value\">{value}</div>"
            f'  {delta_html}'
            f'</div>'
        )

    html = '<div class="ck-kpi-strip">' + "".join(cards_html_parts) + '</div>'
    st.markdown(html, unsafe_allow_html=True)
