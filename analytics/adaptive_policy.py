"""Helpers for the bounded adaptive policy parameter state."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from .db.parameters import (
    bootstrap_parameters,
    ensure_global_parameters_table,
    get_parameter_value,
    set_parameter_value,
)


LANE_RATE_PER_M3_KEY = "adaptive.lane_rate_per_m3"
LANE_ETA_MULTIPLIER_KEY = "adaptive.lane_eta_multiplier"
WEATHER_RISK_MULTIPLIER_KEY = "adaptive.weather_risk_multiplier"
CLOSURE_DELAY_FACTOR_KEY = "adaptive.closure_delay_factor"
TRUCK_EFFICIENCY_SCORE_KEY = "adaptive.truck_efficiency_score"
DRIVER_EFFICIENCY_SCORE_KEY = "adaptive.driver_efficiency_score"
SEASONAL_MARGIN_UPLIFT_KEY = "adaptive.seasonal_margin_uplift"

ADAPTIVE_POLICY_DEFAULTS: tuple[tuple[str, float, str], ...] = (
    (
        LANE_RATE_PER_M3_KEY,
        1.0,
        "Relative lane pricing multiplier used by adaptive policy review.",
    ),
    (
        LANE_ETA_MULTIPLIER_KEY,
        1.0,
        "Relative ETA multiplier used by adaptive policy review.",
    ),
    (
        WEATHER_RISK_MULTIPLIER_KEY,
        1.0,
        "Relative weather risk multiplier used by adaptive policy review.",
    ),
    (
        CLOSURE_DELAY_FACTOR_KEY,
        1.0,
        "Relative road-closure delay factor used by adaptive policy review.",
    ),
    (
        TRUCK_EFFICIENCY_SCORE_KEY,
        1.0,
        "Relative truck efficiency score used by adaptive policy review.",
    ),
    (
        DRIVER_EFFICIENCY_SCORE_KEY,
        1.0,
        "Relative driver efficiency score used by adaptive policy review.",
    ),
    (
        SEASONAL_MARGIN_UPLIFT_KEY,
        0.0,
        "Seasonal margin uplift used by adaptive policy review.",
    ),
)


@dataclass(frozen=True)
class AdaptivePolicySnapshot:
    """Current adaptive-policy parameter state."""

    lane_rate_per_m3: float
    lane_eta_multiplier: float
    weather_risk_multiplier: float
    closure_delay_factor: float
    truck_efficiency_score: float
    driver_efficiency_score: float
    seasonal_margin_uplift: float


def ensure_adaptive_policy_defaults(conn: sqlite3.Connection) -> None:
    """Ensure the adaptive-policy defaults exist in ``global_parameters``."""

    ensure_global_parameters_table(conn)
    bootstrap_parameters(conn, ADAPTIVE_POLICY_DEFAULTS)


def load_adaptive_policy_snapshot(conn: sqlite3.Connection) -> AdaptivePolicySnapshot:
    """Return the current adaptive-policy state."""

    ensure_adaptive_policy_defaults(conn)
    return AdaptivePolicySnapshot(
        lane_rate_per_m3=float(get_parameter_value(conn, LANE_RATE_PER_M3_KEY, 1.0) or 1.0),
        lane_eta_multiplier=float(
            get_parameter_value(conn, LANE_ETA_MULTIPLIER_KEY, 1.0) or 1.0
        ),
        weather_risk_multiplier=float(
            get_parameter_value(conn, WEATHER_RISK_MULTIPLIER_KEY, 1.0) or 1.0
        ),
        closure_delay_factor=float(
            get_parameter_value(conn, CLOSURE_DELAY_FACTOR_KEY, 1.0) or 1.0
        ),
        truck_efficiency_score=float(
            get_parameter_value(conn, TRUCK_EFFICIENCY_SCORE_KEY, 1.0) or 1.0
        ),
        driver_efficiency_score=float(
            get_parameter_value(conn, DRIVER_EFFICIENCY_SCORE_KEY, 1.0) or 1.0
        ),
        seasonal_margin_uplift=float(
            get_parameter_value(conn, SEASONAL_MARGIN_UPLIFT_KEY, 0.0) or 0.0
        ),
    )


def apply_bounded_parameter_target(
    conn: sqlite3.Connection,
    key: str,
    target_value: float,
    *,
    max_delta: float = 0.1,
    min_value: float | None = 0.0,
    max_value: float | None = None,
    description: str | None = None,
) -> float:
    """Move one adaptive parameter toward ``target_value`` within a bounded step."""

    if max_delta < 0:
        raise ValueError("max_delta must be non-negative")

    ensure_adaptive_policy_defaults(conn)
    current_value = float(get_parameter_value(conn, key, 0.0) or 0.0)
    target = float(target_value)
    delta = target - current_value

    if delta > max_delta:
        new_value = current_value + max_delta
    elif delta < -max_delta:
        new_value = current_value - max_delta
    else:
        new_value = target

    if min_value is not None:
        new_value = max(min_value, new_value)
    if max_value is not None:
        new_value = min(max_value, new_value)

    set_parameter_value(conn, key, float(new_value), description)
    return float(new_value)


__all__ = [
    "ADAPTIVE_POLICY_DEFAULTS",
    "AdaptivePolicySnapshot",
    "CLOSURE_DELAY_FACTOR_KEY",
    "DRIVER_EFFICIENCY_SCORE_KEY",
    "LANE_ETA_MULTIPLIER_KEY",
    "LANE_RATE_PER_M3_KEY",
    "SEASONAL_MARGIN_UPLIFT_KEY",
    "TRUCK_EFFICIENCY_SCORE_KEY",
    "WEATHER_RISK_MULTIPLIER_KEY",
    "apply_bounded_parameter_target",
    "ensure_adaptive_policy_defaults",
    "load_adaptive_policy_snapshot",
]
