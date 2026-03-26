from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

SEASON_ORDER = ("summer", "autumn", "winter", "spring")


@dataclass
class MarginRegressionSummary:
    target_column: str
    fitted_job_count: int
    intercept: float
    distance_coeff_per_km: float
    distance_coeff_per_100km: float
    r_squared: float
    rmse: float
    baseline_season: str
    seasonal_effects: dict[str, float]
    seasonal_job_counts: dict[str, int]


def _season_from_timestamp(value: object) -> str | None:
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return None
    month = int(timestamp.month)
    if month in {12, 1, 2}:
        return "summer"
    if month in {3, 4, 5}:
        return "autumn"
    if month in {6, 7, 8}:
        return "winter"
    return "spring"


def summarise_margin_regression(
    df: pd.DataFrame,
    *,
    target_column: str = "margin_per_m3",
) -> MarginRegressionSummary:
    required_columns = {"distance_km", target_column, "job_date"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns for regression summary: {sorted(missing)}")

    working = pd.DataFrame(
        {
            "distance_km": pd.to_numeric(df["distance_km"], errors="coerce"),
            "target": pd.to_numeric(df[target_column], errors="coerce"),
            "season": df["job_date"].map(_season_from_timestamp),
        }
    ).dropna(subset=["distance_km", "target", "season"])

    if working.empty or len(working) < 4:
        return MarginRegressionSummary(
            target_column=target_column,
            fitted_job_count=len(working),
            intercept=math.nan,
            distance_coeff_per_km=math.nan,
            distance_coeff_per_100km=math.nan,
            r_squared=math.nan,
            rmse=math.nan,
            baseline_season="summer",
            seasonal_effects={season: math.nan for season in SEASON_ORDER},
            seasonal_job_counts={
                season: int((working["season"] == season).sum()) if "season" in working else 0
                for season in SEASON_ORDER
            },
        )

    seasonal_counts = {
        season: int((working["season"] == season).sum())
        for season in SEASON_ORDER
    }
    baseline_season = next(
        (season for season in SEASON_ORDER if seasonal_counts[season] > 0),
        SEASON_ORDER[0],
    )
    non_baseline_seasons = [season for season in SEASON_ORDER if season != baseline_season]

    design_columns = [
        np.ones(len(working), dtype=float),
        working["distance_km"].to_numpy(dtype=float),
    ]
    for season in non_baseline_seasons:
        design_columns.append((working["season"] == season).astype(float).to_numpy(dtype=float))
    design = np.column_stack(design_columns)
    target = working["target"].to_numpy(dtype=float)

    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    predictions = design @ coefficients
    residuals = target - predictions
    target_mean = float(target.mean()) if len(target) else 0.0
    ss_res = float(np.sum(np.square(residuals)))
    ss_tot = float(np.sum(np.square(target - target_mean)))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else math.nan
    rmse = math.sqrt(ss_res / len(target)) if len(target) else math.nan

    seasonal_effects = {season: 0.0 for season in SEASON_ORDER}
    seasonal_effects[baseline_season] = 0.0
    for index, season in enumerate(non_baseline_seasons, start=2):
        seasonal_effects[season] = float(coefficients[index])

    return MarginRegressionSummary(
        target_column=target_column,
        fitted_job_count=len(working),
        intercept=float(coefficients[0]),
        distance_coeff_per_km=float(coefficients[1]),
        distance_coeff_per_100km=float(coefficients[1] * 100.0),
        r_squared=float(r_squared),
        rmse=float(rmse),
        baseline_season=baseline_season,
        seasonal_effects=seasonal_effects,
        seasonal_job_counts=seasonal_counts,
    )


def build_margin_regression_preview(
    summary: MarginRegressionSummary,
    *,
    distances_km: tuple[float, ...] = (100.0, 500.0, 1000.0),
) -> pd.DataFrame:
    if math.isnan(summary.intercept) or math.isnan(summary.distance_coeff_per_km):
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for season in SEASON_ORDER:
        seasonal_adjustment = float(summary.seasonal_effects.get(season, 0.0) or 0.0)
        for distance in distances_km:
            predicted_margin = (
                float(summary.intercept)
                + float(summary.distance_coeff_per_km) * float(distance)
                + seasonal_adjustment
            )
            rows.append(
                {
                    "Season": season.title(),
                    "Distance km": float(distance),
                    "Predicted margin per m3": round(predicted_margin, 2),
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "MarginRegressionSummary",
    "build_margin_regression_preview",
    "summarise_margin_regression",
]
