"""Helpers to read and write Streamlit query parameters in a stable API."""
from __future__ import annotations

from typing import Dict, List

import streamlit as st


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
