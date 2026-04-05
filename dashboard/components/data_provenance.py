"""Shared governance notices for scaffold and sourced dashboard signals."""

from __future__ import annotations

import streamlit as st

from dashboard.shell_signals import ShellSignalBundle
from dashboard.theme import provenance_chip


def render_signal_contract_notice(bundle: ShellSignalBundle) -> None:
    """Render the governance notice that matches the current signal contract state."""

    scope_label = bundle.scope_label
    if bundle.freshness_state == "scaffold":
        provenance_chip(
            f"{scope_label} · scaffold values · {bundle.owner} · non-decision-grade",
            icon="🏗️",
        )
        return

    if bundle.freshness_state == "stale":
        provenance_chip(
            f"{scope_label} · stale · {bundle.owner} · fallback active",
            icon="⏳",
        )
        return

    if bundle.freshness_state == "unknown":
        provenance_chip(
            f"{scope_label} · unavailable · {bundle.owner} · advisory only",
            icon="❓",
        )
        return

    grade = bundle.decision_grade.replace('_', ' ')
    provenance_chip(
        f"{scope_label} · {grade} · {bundle.freshness_state} · {bundle.owner}",
        icon="✓" if bundle.freshness_state == "fresh" else "🔗",
    )
