from __future__ import annotations

import math

import pandas as pd
import pytest

from analytics.margin_regression import (
    build_corridor_margin_preview,
    summarise_corridor_margin_model,
    summarise_corridor_margin_validation,
    build_margin_regression_preview,
    summarise_margin_regression,
)


def test_summarise_margin_regression_recovers_distance_and_season_signal() -> None:
    df = pd.DataFrame(
        {
            "job_date": [
                "2026-01-15",
                "2026-02-15",
                "2026-04-10",
                "2026-05-10",
                "2026-07-20",
                "2026-08-20",
                "2026-10-05",
                "2026-11-05",
            ],
            "distance_km": [100, 300, 100, 300, 100, 300, 100, 300],
            "margin_per_m3": [95, 85, 85, 75, 70, 60, 90, 80],
        }
    )

    summary = summarise_margin_regression(df)

    assert summary.fitted_job_count == 8
    assert summary.baseline_season == "summer"
    assert summary.distance_coeff_per_km == pytest.approx(-0.05, abs=1e-6)
    assert summary.distance_coeff_per_100km == pytest.approx(-5.0, abs=1e-6)
    assert summary.seasonal_effects["autumn"] == pytest.approx(-10.0, abs=1e-6)
    assert summary.seasonal_effects["winter"] == pytest.approx(-25.0, abs=1e-6)
    assert summary.seasonal_effects["spring"] == pytest.approx(-5.0, abs=1e-6)
    assert summary.r_squared == pytest.approx(1.0, abs=1e-6)


def test_build_margin_regression_preview_shapes_predictions() -> None:
    df = pd.DataFrame(
        {
            "job_date": ["2026-01-15", "2026-04-15", "2026-07-15", "2026-10-15"],
            "distance_km": [100, 100, 100, 100],
            "margin_per_m3": [90, 80, 70, 85],
        }
    )

    summary = summarise_margin_regression(df)
    preview = build_margin_regression_preview(summary, distances_km=(100.0, 500.0))

    assert list(preview.columns) == ["Season", "Distance km", "Predicted margin per m3"]
    assert len(preview) == 8
    assert set(preview["Season"]) == {"Summer", "Autumn", "Winter", "Spring"}


def test_summarise_margin_regression_returns_nan_summary_when_insufficient_rows() -> None:
    df = pd.DataFrame(
        {
            "job_date": ["2026-01-15", "2026-04-15", "2026-07-15"],
            "distance_km": [100, 200, 300],
            "margin_per_m3": [90, 80, 70],
        }
    )

    summary = summarise_margin_regression(df)
    preview = build_margin_regression_preview(summary)

    assert summary.fitted_job_count == 3
    assert math.isnan(summary.intercept)
    assert preview.empty


def test_summarise_margin_regression_requires_expected_columns() -> None:
    df = pd.DataFrame({"distance_km": [100], "margin_per_m3": [50]})

    with pytest.raises(KeyError):
        summarise_margin_regression(df)


def test_summarise_corridor_margin_model_recovers_corridor_effects_and_improves_fit() -> None:
    df = pd.DataFrame(
        {
            "job_date": [
                "2026-01-15",
                "2026-02-15",
                "2026-01-20",
                "2026-02-20",
                "2026-04-15",
                "2026-05-15",
                "2026-04-20",
                "2026-05-20",
            ],
            "distance_km": [100, 300, 100, 300, 100, 300, 100, 300],
            "corridor_display": [
                "Alpha",
                "Alpha",
                "Beta",
                "Beta",
                "Alpha",
                "Alpha",
                "Beta",
                "Beta",
            ],
            "margin_per_m3": [95, 85, 70, 60, 85, 75, 60, 50],
        }
    )

    summary = summarise_corridor_margin_model(df)

    assert summary.fitted_job_count == 8
    assert summary.baseline_corridor == "Alpha"
    assert summary.corridor_effects["Alpha"] == pytest.approx(0.0, abs=1e-6)
    assert summary.corridor_effects["Beta"] == pytest.approx(-25.0, abs=1e-6)
    assert summary.r_squared > summary.baseline_r_squared
    assert summary.rmse < summary.baseline_rmse


def test_summarise_corridor_margin_model_groups_rare_corridors_into_other() -> None:
    df = pd.DataFrame(
        {
            "job_date": [
                "2026-01-15",
                "2026-02-15",
                "2026-03-15",
                "2026-04-15",
                "2026-05-15",
                "2026-06-15",
            ],
            "distance_km": [100, 200, 300, 100, 200, 300],
            "corridor_display": ["Alpha", "Alpha", "Alpha", "Beta", "Gamma", "Gamma"],
            "margin_per_m3": [90, 80, 70, 88, 75, 65],
        }
    )

    summary = summarise_corridor_margin_model(df, min_corridor_jobs=2)
    preview = build_corridor_margin_preview(summary, distances_km=(100.0,))

    assert "Other corridors" in summary.corridor_job_counts
    assert "Other corridors" in summary.corridor_effects
    assert "Other corridors" in set(preview["Corridor"])


def test_summarise_corridor_margin_model_uses_corridor_group_key_when_available() -> None:
    df = pd.DataFrame(
        {
            "job_date": [
                "2026-01-15",
                "2026-02-15",
                "2026-03-15",
                "2026-04-15",
                "2026-05-15",
                "2026-06-15",
            ],
            "distance_km": [100, 200, 300, 100, 200, 300],
            "corridor_group_key": ["A<->B", "A<->B", "A<->B", "C<->D", "C<->D", "C<->D"],
            "corridor_display": ["Noise"] * 6,
            "margin_per_m3": [90, 80, 70, 95, 85, 75],
        }
    )

    summary = summarise_corridor_margin_model(df)

    assert summary.corridor_column == "corridor_group_key"


def test_summarise_corridor_margin_validation_reports_reviewable_when_corridor_model_beats_baseline() -> None:
    df = pd.DataFrame(
        {
            "job_date": [
                "2026-01-01",
                "2026-01-05",
                "2026-01-10",
                "2026-01-15",
                "2026-01-20",
                "2026-01-25",
                "2026-02-01",
                "2026-02-05",
                "2026-02-10",
                "2026-02-15",
                "2026-02-20",
                "2026-02-25",
            ],
            "distance_km": [100, 200, 300, 100, 200, 300, 100, 200, 300, 100, 200, 300],
            "corridor_display": [
                "Alpha",
                "Alpha",
                "Alpha",
                "Beta",
                "Beta",
                "Beta",
                "Alpha",
                "Alpha",
                "Alpha",
                "Beta",
                "Beta",
                "Beta",
            ],
            "margin_per_m3": [95, 90, 85, 70, 65, 60, 94, 89, 84, 69, 64, 59],
        }
    )

    summary = summarise_corridor_margin_validation(df)

    assert summary.train_job_count == 9
    assert summary.holdout_job_count == 3
    assert summary.holdout_rmse < summary.baseline_holdout_rmse
    assert summary.rmse_improvement > 0
    assert summary.trust_label == "reviewable"


def test_summarise_corridor_margin_validation_reports_low_support_for_novel_holdout_corridors() -> None:
    df = pd.DataFrame(
        {
            "job_date": [
                "2026-01-01",
                "2026-01-05",
                "2026-01-10",
                "2026-01-15",
                "2026-01-20",
                "2026-01-25",
                "2026-02-01",
                "2026-02-05",
            ],
            "distance_km": [100, 200, 300, 100, 200, 300, 150, 250],
            "corridor_display": [
                "Alpha",
                "Alpha",
                "Alpha",
                "Beta",
                "Beta",
                "Beta",
                "Novel-1",
                "Novel-2",
            ],
            "margin_per_m3": [95, 90, 85, 70, 65, 60, 82, 78],
        }
    )

    summary = summarise_corridor_margin_validation(df)

    assert summary.holdout_job_count == 2
    assert summary.novel_holdout_corridor_count >= 2
    assert summary.trust_label == "low_support"
