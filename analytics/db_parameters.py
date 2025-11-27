"""Global parameter helpers used across analytics features.

Backwards-compatible wrapper around the packaged parameter helpers.
"""
from __future__ import annotations

from .db.parameters import (
    bootstrap_parameters,
    ensure_global_parameters_table,
    get_parameter_value,
    set_parameter_value,
)

__all__ = [
    "bootstrap_parameters",
    "ensure_global_parameters_table",
    "get_parameter_value",
    "set_parameter_value",
]
