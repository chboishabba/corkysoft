"""State and session helpers for the Streamlit dashboards."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import streamlit as st

from corkysoft.quote_service import QuoteResult


__all__ = [
    "_set_query_params",
    "_get_query_params",
    "_rerun_app",
    "_initial_pin_state",
    "_ensure_pin_state",
    "_first_non_empty",
]


def _set_query_params(**params: str) -> None:
    """Set Streamlit query parameters using the stable API when available."""

    query_params = getattr(st, "query_params", None)
    if query_params is not None:
        query_params.from_dict(params)
        return

    # Fallback for older Streamlit versions.
    st.experimental_set_query_params(**params)


def _get_query_params() -> Dict[str, List[str]]:
    """Return query parameters as a dictionary of lists."""

    query_params = getattr(st, "query_params", None)
    if query_params is not None:
        return {key: query_params.get_all(key) for key in query_params.keys()}
    return st.experimental_get_query_params()


def _rerun_app() -> None:
    """Trigger a Streamlit rerun using the available API."""

    rerun = getattr(st, "rerun", None)
    if rerun is not None:
        rerun()
        return

    st.experimental_rerun()


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
