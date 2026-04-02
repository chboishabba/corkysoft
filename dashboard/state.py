"""State and session helpers for the Streamlit dashboards."""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Literal, Optional, Sequence

import math

import pandas as pd
import streamlit as st

from corkysoft.quote_service import QuoteInput, QuoteResult, calculate_quote
from dashboard.query_params import _get_query_params, _set_query_params


__all__ = [
    "_set_query_params",
    "_get_query_params",
    "_rerun_app",
    "_initial_pin_state",
    "_ensure_pin_state",
    "_first_non_empty",
    "_format_route_label",
    "_apply_quote_suggestion",
]


def _rerun_app() -> None:
    """Trigger a Streamlit rerun using the available API."""
    rerun = getattr(st, "rerun", None)
    if callable(rerun):
        rerun()
        return

    experimental_rerun = getattr(st, "experimental_rerun", None)
    if callable(experimental_rerun):
        experimental_rerun()
        return

    raise RuntimeError("Streamlit rerun API is unavailable.")


def _initial_pin_state(result: QuoteResult) -> Dict[str, Any]:
    """Return the default pin state derived from a ``QuoteResult``."""

    return {
        "origin": {
            "lon": float(result.origin_lon),
            "lat": float(result.origin_lat),
        },
        "destination": {
            "lon": float(result.dest_lon),
            "lat": float(result.dest_lat),
        },
        "enabled": False,
    }


def _ensure_pin_state(result: QuoteResult) -> Dict[str, Any]:
    """Ensure the pin state exists in ``st.session_state`` and return it."""

    state: Dict[str, Any] = st.session_state.get("quote_pin_override", {})
    if not state or "origin" not in state or "destination" not in state:
        state = _initial_pin_state(result)
    else:
        state.setdefault("enabled", False)

        # When result coordinates change, refresh defaults so pins move with them.
        origin_state = state.get("origin") or {}
        dest_state = state.get("destination") or {}

        if not origin_state:
            origin_state = {}
        if not dest_state:
            dest_state = {}

        origin_state.setdefault("lon", float(result.origin_lon))
        origin_state.setdefault("lat", float(result.origin_lat))
        dest_state.setdefault("lon", float(result.dest_lon))
        dest_state.setdefault("lat", float(result.dest_lat))

        state["origin"] = origin_state
        state["destination"] = dest_state

    st.session_state["quote_pin_override"] = state
    return state


def _first_non_empty(route: pd.Series, columns: Sequence[str]) -> Optional[str]:
    """Return the first non-empty string value from ``route`` across ``columns``."""

    for column in columns:
        if column in route and isinstance(route[column], str):
            value = route[column].strip()
            if value:
                return value
    return None


def _format_route_label(route: pd.Series) -> str:
    """Construct the label used in quote and route maps."""

    origin = _first_non_empty(
        route,
        [
            "corridor_display",
            "origin",
            "origin_city",
            "origin_normalized",
            "origin_raw",
        ],
    ) or "Origin"
    destination = _first_non_empty(
        route,
        [
            "destination",
            "destination_city",
            "destination_normalized",
            "destination_raw",
        ],
    ) or "Destination"
    distance_value: Optional[float] = None
    for column in ("distance_km", "distance", "km", "kms"):
        if column in route and pd.notna(route[column]):
            try:
                distance_value = float(route[column])
            except (TypeError, ValueError):
                continue
            break
    if distance_value is not None and not math.isnan(distance_value):
        return f"{origin} → {destination} ({distance_value:.1f} km)"
    return f"{origin} → {destination}"


def _apply_quote_suggestion(
    conn: Any,
    field: Literal["origin", "destination"],
    suggestion: str,
) -> Optional[QuoteResult]:
    """Update stored quote inputs with a selected suggestion and recalculate."""

    if not suggestion:
        return None

    text_key = "Origin" if field == "origin" else "Destination"
    st.session_state[text_key] = suggestion

    existing_inputs = st.session_state.get("quote_inputs")
    if not isinstance(existing_inputs, QuoteInput):
        st.session_state["quote_suggestion_error"] = (
            "Unable to apply suggestion without an existing quote input."
        )
        return None

    updated_inputs = (
        replace(existing_inputs, origin=suggestion)
        if field == "origin"
        else replace(existing_inputs, destination=suggestion)
    )
    st.session_state["quote_inputs"] = updated_inputs

    try:
        updated_result = calculate_quote(conn, updated_inputs)
    except (RuntimeError, ValueError) as exc:
        st.session_state["quote_suggestion_error"] = str(exc)
        st.session_state.pop("quote_result", None)
        return None

    st.session_state.pop("quote_suggestion_error", None)
    st.session_state["quote_result"] = updated_result
    st.session_state["quote_manual_override_enabled"] = False
    st.session_state["quote_manual_override_amount"] = float(updated_result.final_quote)
    return updated_result
