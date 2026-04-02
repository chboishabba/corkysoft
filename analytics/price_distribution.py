"""Pricing facade for dashboard analytics and compatibility imports.

This module is intentionally thin: domain subsystems now live in dedicated
modules while callers continue importing the established pricing surface from
here.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Literal, Optional, Sequence, TYPE_CHECKING

import numpy as np
import pandas as pd

from .db import (
    bootstrap_parameters,
    ensure_global_parameters_table,
    get_parameter_value,
    set_parameter_value,
)
from .corridor_performance import (
    aggregate_corridor_performance,
    format_bidirectional_corridor,
)
from .historical_ingest import (
    CORRIDOR_COLUMNS,
    DESTINATION_COLUMNS,
    DESTINATION_POSTCODE_CANDIDATES,
    ORIGIN_COLUMNS,
    ORIGIN_POSTCODE_CANDIDATES,
    POSTCODE_COLUMNS,
    ColumnMapping,
    _first_present,
    _infer_datetime_parse_kwargs,
    enrich_missing_route_coordinates,
    geocode_cached,
    import_historical_jobs_from_dataframe,
    infer_columns,
    latest_historical_ingest_summary,
)
from .lane_assignment import LANE_STATUS_ASSIGNED
from .price_history_analysis import (
    PriceHistorySeries,
    build_price_history_series,
    summarise_last_year_distributions,
)
from .routes_map import populate_route_geometry
from .profitability_analysis import (
    DEFAULT_BREAK_EVEN_ABS_TOLERANCE,
    DEFAULT_BREAK_EVEN_REL_TOLERANCE,
    PROFITABILITY_BANDS,
    DistributionSummary,
    ProfitabilitySummary,
    build_profitability_export,
    classify_profit_band,
    classify_profitability_status,
    summarise_distribution,
    summarise_profitability,
)
from .profitability_map_prep import (
    PROFITABILITY_COLOURS,
    compute_profitability_line_width,
    compute_tapered_route_polygon,
    prepare_profitability_map_data,
    prepare_profitability_route_data,
    _coerce_float,
    _route_display_name,
)
from .profitability_charts import (
    create_histogram,
    create_m3_margin_figure,
    create_m3_vs_km_figure,
    create_metro_profitability_figure,
    filter_metro_jobs,
)
from .route_map_prep import (
    HEATMAP_WEIGHTING_CANDIDATES,
    METRO_DISTANCE_THRESHOLD_KM,
    build_heatmap_source,
    build_isochrone_polygons,
    explain_isochrone_unavailability,
    compute_cost_vs_price_percentage,
    filter_jobs_by_distance,
    prepare_metric_route_map_data,
    prepare_route_map_data,
    available_heatmap_weightings,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from openrouteservice import Client as ORSClient
else:
    ORSClient = Any  # type: ignore[misc, assignment]

# Local constants and lightweight helpers that still belong to the facade.
METRO_HISTOGRAM_BINS = 15


PRICE_HISTORY_FREQUENCIES: dict[str, str] = {
    "D": "D",
    "DAILY": "D",
    "DAY": "D",
    "W": "W",
    "WEEKLY": "W",
    "WEEK": "W",
    "M": "M",
    "MONTHLY": "M",
    "MONTH": "M",
}


def _has_geometry(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text and text not in {"{}", "[]", "null", "None"})


BREAK_EVEN_KEY = "break_even_per_m3"
DEFAULT_BREAK_EVEN_VALUE = 250.0
DEFAULT_BREAK_EVEN_DESCRIPTION = "Baseline break-even $/m³ across the network"

FUEL_COST_KEY = "base_cost.fuel_per_km"
DEFAULT_FUEL_COST_PER_KM = 0.95

DRIVER_COST_KEY = "base_cost.driver_per_km"
DEFAULT_DRIVER_COST_PER_KM = 6.5

MAINTENANCE_COST_KEY = "base_cost.maintenance_per_km"
DEFAULT_MAINTENANCE_COST_PER_KM = 1.1

OVERHEAD_COST_KEY = "base_cost.overhead_per_job"
DEFAULT_OVERHEAD_COST_PER_JOB = 3200.0

BASE_COST_DEFAULTS: Sequence[tuple[str, float, str]] = (
    (FUEL_COST_KEY, DEFAULT_FUEL_COST_PER_KM, "Fuel cost per kilometre (AUD)"),
    (DRIVER_COST_KEY, DEFAULT_DRIVER_COST_PER_KM, "Driver labour cost per kilometre (AUD)"),
    (
        MAINTENANCE_COST_KEY,
        DEFAULT_MAINTENANCE_COST_PER_KM,
        "Maintenance and tyre cost per kilometre (AUD)",
    ),
    (OVERHEAD_COST_KEY, DEFAULT_OVERHEAD_COST_PER_JOB, "Fixed overhead per job (AUD)"),
)


@dataclass(frozen=True)
class BaseCostConfig:
    fuel_per_km: float
    driver_per_km: float
    maintenance_per_km: float
    overhead_per_job: float

    @property
    def per_km_total(self) -> float:
        return self.fuel_per_km + self.driver_per_km + self.maintenance_per_km


def _deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with duplicate column labels collapsed via coalescing.

    Legacy exports may contain denormalised location columns (``origin`` and
    ``destination`` coordinates) that are later joined again during dashboard
    loads. Pandas keeps all duplicate column labels which forces callers to
    handle :class:`~pandas.DataFrame` objects when selecting a column and loses
    the original coordinate data if the joined copy is missing.  Instead of
    dropping the earlier duplicates outright we coalesce the values by
    preferring the right-most non-null entry and falling back to preceding
    columns so that sparsely populated joins keep the stored coordinates.
    """

    if not df.columns.duplicated().any():
        return df

    df = df.copy()
    duplicate_labels = df.columns[df.columns.duplicated(keep=False)]
    for label in pd.unique(duplicate_labels):
        duplicate_indices = [idx for idx, col in enumerate(df.columns) if col == label]
        if len(duplicate_indices) < 2:
            continue

        duplicates = df.iloc[:, duplicate_indices]
        combined = duplicates.iloc[:, ::-1].bfill(axis=1).ffill(axis=1).iloc[:, 0]
        df.iloc[:, duplicate_indices[-1]] = combined

    return df.loc[:, ~df.columns.duplicated(keep="last")]


def _coalesce_string_columns(
    df: pd.DataFrame, primary: str, fallback: str, target: str
) -> None:
    """Populate ``target`` by preferring ``primary`` and falling back to ``fallback``."""

    primary_series = (
        df[primary].astype(str).str.strip()
        if primary in df.columns
        else pd.Series("", index=df.index)
    )
    fallback_series = (
        df[fallback].astype(str).str.strip()
        if fallback in df.columns
        else pd.Series("", index=df.index)
    )
    combined = primary_series.where(primary_series != "", fallback_series)
    df[target] = combined.replace("", np.nan)


def ensure_base_cost_parameters(conn) -> BaseCostConfig:
    """Ensure operating cost parameters exist and return their values."""

    bootstrap_parameters(conn, BASE_COST_DEFAULTS)
    fuel = get_parameter_value(conn, FUEL_COST_KEY, DEFAULT_FUEL_COST_PER_KM)
    driver = get_parameter_value(conn, DRIVER_COST_KEY, DEFAULT_DRIVER_COST_PER_KM)
    maintenance = get_parameter_value(
        conn, MAINTENANCE_COST_KEY, DEFAULT_MAINTENANCE_COST_PER_KM
    )
    overhead = get_parameter_value(conn, OVERHEAD_COST_KEY, DEFAULT_OVERHEAD_COST_PER_JOB)
    assert fuel is not None
    assert driver is not None
    assert maintenance is not None
    assert overhead is not None
    return BaseCostConfig(
        fuel_per_km=float(fuel),
        driver_per_km=float(driver),
        maintenance_per_km=float(maintenance),
        overhead_per_job=float(overhead),
    )


def compute_break_even_series(
    distance_km: pd.Series, volume_m3: pd.Series, base_costs: BaseCostConfig
) -> tuple[pd.Series, pd.Series]:
    """Return total and per-m³ break-even costs for each job."""

    per_km_cost = base_costs.per_km_total
    distance_values = pd.to_numeric(distance_km, errors="coerce")
    volume_values = pd.to_numeric(volume_m3, errors="coerce")
    total_cost = distance_values * per_km_cost + base_costs.overhead_per_job
    safe_volume = volume_values.replace({0: np.nan})
    per_m3_cost = total_cost / safe_volume
    mask = distance_values.isna() | safe_volume.isna()
    total_cost = total_cost.where(~mask, np.nan)
    per_m3_cost = per_m3_cost.where(~mask, np.nan)
    return total_cost.astype(float), per_m3_cost.astype(float)

from .job_loading import (
    filter_routes_by_country,
    load_historical_jobs,
    load_live_jobs,
    load_quotes,
)


def ensure_break_even_parameter(conn) -> float:
    """Ensure the break-even parameter exists and return its value."""
    bootstrap_parameters(
        conn,
        [
            (BREAK_EVEN_KEY, DEFAULT_BREAK_EVEN_VALUE, DEFAULT_BREAK_EVEN_DESCRIPTION),
        ],
    )
    value = get_parameter_value(conn, BREAK_EVEN_KEY)
    assert value is not None
    return value


def update_break_even(conn, value: float) -> None:
    """Update the break-even value."""
    set_parameter_value(conn, BREAK_EVEN_KEY, value, DEFAULT_BREAK_EVEN_DESCRIPTION)
