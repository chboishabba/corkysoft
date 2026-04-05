"""Reusable alert banner component with severity coding.

Renders severity-coded alert banners — critical (red/pulsing),
warning (amber), and info (blue).
"""
from __future__ import annotations

from html import escape

import streamlit as st

_SEVERITY_ICONS = {
    "critical": "🔴",
    "warning": "⚠️",
    "info": "ℹ️",
}

_SEVERITY_TITLES = {
    "critical": "Action Required",
    "warning": "Attention",
    "info": "Status",
}


def render_alert_banner(
    message: str,
    *,
    severity: str = "info",
    title: str | None = None,
) -> None:
    """Render a severity-coded alert banner.

    Parameters
    ----------
    message : str
        The alert message body.
    severity : str
        One of "critical", "warning", "info".
    title : str or None
        Optional title override.  Defaults to a standard label per severity.
    """
    sev = severity if severity in _SEVERITY_ICONS else "info"
    icon = _SEVERITY_ICONS[sev]
    display_title = escape(title or _SEVERITY_TITLES[sev])
    safe_message = escape(message)

    html = (
        f'<div class="ck-alert {sev}">'
        f'  <span class="ck-alert-icon">{icon}</span>'
        f'  <div class="ck-alert-body">'
        f'    <div class="ck-alert-title">{display_title}</div>'
        f"    {safe_message}"
        f'  </div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)
