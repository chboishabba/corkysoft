from __future__ import annotations

import pandas as pd
import pytest

from analytics.optimizer import (
    OptimizerParameters,
    can_run_optimizer,
    recommendations_to_frame,
    run_margin_optimizer,
)


def test_run_margin_optimizer_caps_uplift() -> None:
    df = pd.DataFrame(
        {
            "corridor_display": ["BNE-SYD", "BNE-SYD", "BNE-SYD", "BNE-SYD"],
            "price_per_m3": [300.0, 310.0, 305.0, 295.0],
            "final_cost_per_m3": [220.0, 230.0, 225.0, 215.0],
        }
    )
    params = OptimizerParameters(
        target_margin_per_m3=120.0, max_uplift_pct=10.0, min_job_count=1
    )

    run = run_margin_optimizer(df, params)
    assert run.recommendations, "Expected at least one recommendation"
    recommendation = run.recommendations[0]
    assert recommendation.corridor == "BNE-SYD"
    assert pytest.approx(recommendation.uplift_per_m3, rel=1e-4) == 30.25
    assert pytest.approx(recommendation.recommended_margin_per_m3, rel=1e-4) == 110.25
    frame = recommendations_to_frame(run.recommendations)
    assert "Corridor" in frame.columns
    assert frame.iloc[0]["Corridor"] == "BNE-SYD"


def test_run_margin_optimizer_flags_low_sample() -> None:
    df = pd.DataFrame(
        {
            "corridor_display": ["SYD-MEL", "SYD-MEL"],
            "price_per_m3": [280.0, 275.0],
            "final_cost_per_m3": [250.0, 248.0],
        }
    )
    params = OptimizerParameters(
        target_margin_per_m3=80.0, max_uplift_pct=50.0, min_job_count=3
    )

    run = run_margin_optimizer(df, params)
    assert run.recommendations
    recommendation = run.recommendations[0]
    assert recommendation.notes and "Only 2" in recommendation.notes


def test_optimizer_uses_total_cost_when_per_m3_missing() -> None:
    df = pd.DataFrame(
        {
            "corridor": ["BNE-GLD", "BNE-GLD", "BNE-GLD"],
            "price_per_m3": [260.0, 255.0, 262.0],
            "final_cost": [4800.0, 5000.0, 4700.0],
            "volume_m3": [20.0, 22.0, 19.0],
        }
    )
    assert can_run_optimizer(df)
    run = run_margin_optimizer(df, OptimizerParameters(target_margin_per_m3=70.0))
    assert run.recommendations
    recommendation = run.recommendations[0]
    assert recommendation.corridor == "BNE-GLD"
    # Ensure the optimizer derived cost per m3 correctly
    assert recommendation.current_margin_per_m3 > 0
