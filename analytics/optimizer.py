"""Utilities for running pricing optimisations on historical jobs."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Candidate columns reused by other analytics modules. They are duplicated here to
# avoid the heavier imports inside :mod:`analytics.price_distribution`.
PRICE_COLUMNS: Sequence[str] = (
    "price_per_m3",
    "revenue_per_m3",
    "sell_per_m3",
    "rate_per_m3",
)
COST_PER_M3_COLUMNS: Sequence[str] = (
    "final_cost_per_m3",
    "cost_per_m3",
)
TOTAL_COST_COLUMNS: Sequence[str] = (
    "final_cost",
    "final_total",
    "actual_cost",
    "actual_total",
    "cost_total",
)
VOLUME_COLUMNS: Sequence[str] = (
    "volume_m3",
    "volume_cbm",
    "cbm",
    "cubic_meters",
    "m3",
)
CORRIDOR_COLUMNS: Sequence[str] = (
    "corridor_display",
    "corridor",
    "lane",
    "lane_name",
)

DEFAULT_CORRIDOR_LABEL = "Unmapped corridor"


@dataclass
class OptimizerParameters:
    """User-configurable optimisation inputs."""

    target_margin_per_m3: float = 120.0
    max_uplift_pct: float = 25.0
    min_job_count: int = 3


@dataclass
class OptimizationRecommendation:
    """Recommended pricing change for a single corridor."""

    corridor: str
    job_count: int
    current_price_per_m3: float
    current_margin_per_m3: float
    recommended_price_per_m3: float
    recommended_margin_per_m3: float
    uplift_per_m3: float
    uplift_pct: float
    notes: Optional[str] = None


@dataclass
class OptimizerRun:
    """Result of executing the optimisation routine."""

    executed_at: datetime
    parameters: OptimizerParameters
    recommendations: List[OptimizationRecommendation]


def _first_present(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    """Return the first column name from *candidates* that exists."""

    columns_lower = {col.lower(): col for col in columns}
    for candidate in candidates:
        lower = candidate.lower()
        if lower in columns_lower:
            return columns_lower[lower]
    return None


def can_run_optimizer(df: pd.DataFrame) -> bool:
    """Return ``True`` when the dataframe has enough data for optimisation."""

    if df.empty:
        return False
    price_column = _first_present(df.columns, PRICE_COLUMNS)
    cost_per_m3_column = _first_present(df.columns, COST_PER_M3_COLUMNS)
    total_cost_column = _first_present(df.columns, TOTAL_COST_COLUMNS)
    volume_column = _first_present(df.columns, VOLUME_COLUMNS)
    return bool(price_column) and bool(
        cost_per_m3_column or (total_cost_column and volume_column)
    )


def _safe_corridor_label(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return DEFAULT_CORRIDOR_LABEL
    text = str(value).strip()
    return text or DEFAULT_CORRIDOR_LABEL


def _aligned_price_cost(
    group: pd.DataFrame,
    price_column: str,
    cost_per_m3_column: Optional[str],
    total_cost_column: Optional[str],
    volume_column: Optional[str],
) -> Tuple[pd.Series, pd.Series]:
    price_series = pd.to_numeric(group[price_column], errors="coerce")

    if cost_per_m3_column:
        cost_series = pd.to_numeric(group[cost_per_m3_column], errors="coerce")
    elif total_cost_column and volume_column:
        total_cost = pd.to_numeric(group[total_cost_column], errors="coerce")
        volume = pd.to_numeric(group[volume_column], errors="coerce")
        volume = volume.replace(0, np.nan)
        cost_series = total_cost / volume
    else:
        cost_series = pd.Series(dtype=float)

    aligned = pd.DataFrame({"price": price_series, "cost": cost_series})
    aligned.dropna(inplace=True)
    return aligned["price"], aligned["cost"]


def run_margin_optimizer(
    df: pd.DataFrame, parameters: OptimizerParameters | None = None
) -> OptimizerRun:
    """Calculate recommended per-corridor uplifts to hit a margin target."""

    params = parameters or OptimizerParameters()
    executed_at = datetime.now(timezone.utc)
    if df.empty or not can_run_optimizer(df):
        return OptimizerRun(executed_at, params, [])

    price_column = _first_present(df.columns, PRICE_COLUMNS)
    cost_per_m3_column = _first_present(df.columns, COST_PER_M3_COLUMNS)
    total_cost_column = _first_present(df.columns, TOTAL_COST_COLUMNS)
    volume_column = _first_present(df.columns, VOLUME_COLUMNS)
    corridor_column = _first_present(df.columns, CORRIDOR_COLUMNS)

    recommendations: List[OptimizationRecommendation] = []

    group_iterable: Iterable[tuple[object, pd.DataFrame]]
    if corridor_column:
        group_iterable = df.groupby(corridor_column, dropna=False)
    else:
        group_iterable = [(DEFAULT_CORRIDOR_LABEL, df)]

    for corridor_value, group in group_iterable:
        prices, costs = _aligned_price_cost(
            group,
            price_column=price_column,
            cost_per_m3_column=cost_per_m3_column,
            total_cost_column=total_cost_column,
            volume_column=volume_column,
        )

        if prices.empty or costs.empty:
            continue

        price_median = float(np.median(prices))
        margin_series = prices - costs
        margin_median = float(np.median(margin_series))

        uplift_needed = max(0.0, params.target_margin_per_m3 - margin_median)
        max_uplift_amount = (
            price_median * (params.max_uplift_pct / 100.0)
            if params.max_uplift_pct >= 0
            else 0.0
        )
        uplift_capped = min(uplift_needed, max_uplift_amount) if price_median else 0.0
        recommended_price = price_median + uplift_capped
        recommended_margin = margin_median + uplift_capped

        uplift_pct = (uplift_capped / price_median * 100.0) if price_median else 0.0
        job_count = int(len(prices))

        notes: Optional[str] = None
        if job_count < params.min_job_count:
            plural = "s" if job_count != 1 else ""
            notes = f"Only {job_count} job{plural}" if job_count else "No valid jobs"
        elif uplift_capped <= 0.0:
            notes = "Already meeting target"

        corridor_label = _safe_corridor_label(corridor_value)
        recommendation = OptimizationRecommendation(
            corridor=corridor_label,
            job_count=job_count,
            current_price_per_m3=price_median,
            current_margin_per_m3=margin_median,
            recommended_price_per_m3=recommended_price,
            recommended_margin_per_m3=recommended_margin,
            uplift_per_m3=uplift_capped,
            uplift_pct=uplift_pct,
            notes=notes,
        )
        recommendations.append(recommendation)

    recommendations.sort(key=lambda rec: rec.uplift_per_m3, reverse=True)
    return OptimizerRun(executed_at, params, recommendations)


def recommendations_to_frame(
    recommendations: Sequence[OptimizationRecommendation],
) -> pd.DataFrame:
    """Convert recommendations into a user-friendly dataframe."""

    columns = [
        "Corridor",
        "Jobs analysed",
        "Current $/m³",
        "Current margin $/m³",
        "Recommended $/m³",
        "Recommended margin $/m³",
        "Uplift $/m³",
        "Uplift %",
        "Notes",
    ]
    if not recommendations:
        return pd.DataFrame(columns=columns)

    rows = []
    for rec in recommendations:
        rows.append(
            {
                "Corridor": rec.corridor,
                "Jobs analysed": rec.job_count,
                "Current $/m³": round(rec.current_price_per_m3, 2),
                "Current margin $/m³": round(rec.current_margin_per_m3, 2),
                "Recommended $/m³": round(rec.recommended_price_per_m3, 2),
                "Recommended margin $/m³": round(rec.recommended_margin_per_m3, 2),
                "Uplift $/m³": round(rec.uplift_per_m3, 2),
                "Uplift %": round(rec.uplift_pct, 2),
                "Notes": rec.notes or "",
            }
        )
    return pd.DataFrame(rows, columns=columns)


__all__ = [
    "OptimizerParameters",
    "OptimizationRecommendation",
    "OptimizerRun",
    "can_run_optimizer",
    "recommendations_to_frame",
    "run_margin_optimizer",
]
