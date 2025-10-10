"""Analytics utilities for Corkysoft dashboards."""

from .db import get_connection, ensure_global_parameters_table, get_parameter_value, set_parameter_value
from .price_distribution import load_historical_jobs

__all__ = [
    "get_connection",
    "ensure_global_parameters_table",
    "get_parameter_value",
    "set_parameter_value",
    "load_historical_jobs",
]
