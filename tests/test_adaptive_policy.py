from __future__ import annotations

import sqlite3

from analytics.adaptive_policy import (
    ADAPTIVE_POLICY_DEFAULTS,
    LANE_RATE_PER_M3_KEY,
    SEASONAL_MARGIN_UPLIFT_KEY,
    apply_bounded_parameter_target,
    ensure_adaptive_policy_defaults,
    load_adaptive_policy_snapshot,
)


def build_conn() -> sqlite3.Connection:
    return sqlite3.connect(":memory:")


def test_ensure_adaptive_policy_defaults_bootstraps_parameters() -> None:
    conn = build_conn()

    ensure_adaptive_policy_defaults(conn)

    rows = conn.execute(
        "SELECT key, value_numeric FROM global_parameters ORDER BY key"
    ).fetchall()

    assert len(rows) == len(ADAPTIVE_POLICY_DEFAULTS)
    assert dict(rows)[LANE_RATE_PER_M3_KEY] == 1.0
    assert dict(rows)[SEASONAL_MARGIN_UPLIFT_KEY] == 0.0


def test_load_adaptive_policy_snapshot_returns_default_state() -> None:
    conn = build_conn()

    snapshot = load_adaptive_policy_snapshot(conn)

    assert snapshot.lane_rate_per_m3 == 1.0
    assert snapshot.lane_eta_multiplier == 1.0
    assert snapshot.weather_risk_multiplier == 1.0
    assert snapshot.closure_delay_factor == 1.0
    assert snapshot.truck_efficiency_score == 1.0
    assert snapshot.driver_efficiency_score == 1.0
    assert snapshot.seasonal_margin_uplift == 0.0


def test_apply_bounded_parameter_target_caps_positive_delta() -> None:
    conn = build_conn()
    ensure_adaptive_policy_defaults(conn)

    updated = apply_bounded_parameter_target(
        conn,
        LANE_RATE_PER_M3_KEY,
        1.4,
        max_delta=0.1,
    )

    assert updated == 1.1


def test_apply_bounded_parameter_target_caps_negative_delta_and_min_value() -> None:
    conn = build_conn()
    ensure_adaptive_policy_defaults(conn)

    updated = apply_bounded_parameter_target(
        conn,
        SEASONAL_MARGIN_UPLIFT_KEY,
        -0.5,
        max_delta=0.25,
        min_value=0.0,
    )

    assert updated == 0.0
