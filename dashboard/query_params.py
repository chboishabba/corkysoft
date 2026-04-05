"""Helpers to read and write Streamlit query parameters in a stable API."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

import streamlit as st

from dashboard.workspace_state import (
    workspace_state_from_query_params,
    workspace_state_to_query_params,
)


def _set_query_params(**params: str) -> None:
    """Set Streamlit query parameters using the stable API when available."""
    query_params = getattr(st, "query_params", None)
    if query_params is not None:
        query_params.from_dict(params)
        return
    st.experimental_set_query_params(**params)


def _get_query_params() -> Dict[str, List[str]]:
    """Return Streamlit query parameters as a dictionary of lists."""
    query_params = getattr(st, "query_params", None)
    if query_params is not None:
        return {key: query_params.get_all(key) for key in query_params.keys()}
    return st.experimental_get_query_params()


def _get_workspace_state(*, available_tabs: Iterable[str]) -> dict[str, Any]:
    """Return normalized workspace state derived from current query parameters."""

    return workspace_state_from_query_params(
        _get_query_params(),
        available_tabs=available_tabs,
    )


def _set_workspace_query_params(
    *,
    available_tabs: Iterable[str],
    **state: Any,
) -> None:
    """Persist canonical workspace state into query params with legacy compatibility keys."""

    _set_query_params(
        **workspace_state_to_query_params(
            state,
            available_tabs=available_tabs,
        )
    )
