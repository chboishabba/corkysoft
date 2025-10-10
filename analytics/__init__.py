"""Analytics utilities for Corkysoft dashboards."""

# Re-export lightweight database helpers at the package level. More
# feature-specific utilities (e.g. price distribution loading) should be
# imported from their dedicated modules to avoid pulling heavy dependencies
# such as pandas/numpy during package initialisation.
from .db import (
    ensure_global_parameters_table,
    ensure_historical_job_routes_table,
    get_connection,
    get_parameter_value,
    migrate_geojson_to_routes,
    set_parameter_value,
)

__all__ = [
    "get_connection",
    "ensure_global_parameters_table",
    "ensure_historical_job_routes_table",
    "get_parameter_value",
    "migrate_geojson_to_routes",
    "set_parameter_value",
]
