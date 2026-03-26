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


@dataclass
class CorridorMarginModelSummary:
    target_column: str
    corridor_column: str
    fitted_job_count: int
    intercept: float
    distance_coeff_per_km: float
    distance_coeff_per_100km: float
    r_squared: float
    rmse: float
    baseline_r_squared: float
    baseline_rmse: float
    baseline_season: str
    baseline_corridor: str
    seasonal_effects: dict[str, float]
    corridor_effects: dict[str, float]
    seasonal_job_counts: dict[str, int]
    corridor_job_counts: dict[str, int]


def _fit_linear_model(design: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float, float]:
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    predictions = design @ coefficients
    residuals = target - predictions
    target_mean = float(target.mean()) if len(target) else 0.0
    ss_res = float(np.sum(np.square(residuals)))
    ss_tot = float(np.sum(np.square(target - target_mean)))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else math.nan
    rmse = math.sqrt(ss_res / len(target)) if len(target) else math.nan
    return coefficients, float(r_squared), float(rmse)


def _corridor_series(df: pd.DataFrame) -> tuple[str, pd.Series]:
    if "corridor_group_key" in df.columns:
        raw = df["corridor_group_key"].astype(str).replace({"": np.nan})
        if raw.dropna().any():
            return "corridor_group_key", raw
    if "corridor_display" in df.columns:
        raw = df["corridor_display"].astype(str).replace({"": np.nan})
        return "corridor_display", raw
    raise KeyError("Missing required corridor column for corridor-aware regression summary")


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

    coefficients, r_squared, rmse = _fit_linear_model(design, target)

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


def summarise_corridor_margin_model(
    df: pd.DataFrame,
    *,
    target_column: str = "margin_per_m3",
    min_corridor_jobs: int = 2,
    max_corridors: int = 8,
) -> CorridorMarginModelSummary:
    required_columns = {"distance_km", target_column, "job_date"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns for corridor-aware regression summary: {sorted(missing)}")

    corridor_column, corridor_values = _corridor_series(df)
    working = pd.DataFrame(
        {
            "distance_km": pd.to_numeric(df["distance_km"], errors="coerce"),
            "target": pd.to_numeric(df[target_column], errors="coerce"),
            "season": df["job_date"].map(_season_from_timestamp),
            "corridor": corridor_values,
        }
    ).dropna(subset=["distance_km", "target", "season", "corridor"])

    if working.empty or len(working) < 6:
        return CorridorMarginModelSummary(
            target_column=target_column,
            corridor_column=corridor_column,
            fitted_job_count=len(working),
            intercept=math.nan,
            distance_coeff_per_km=math.nan,
            distance_coeff_per_100km=math.nan,
            r_squared=math.nan,
            rmse=math.nan,
            baseline_r_squared=math.nan,
            baseline_rmse=math.nan,
            baseline_season="summer",
            baseline_corridor="Other corridors",
            seasonal_effects={season: math.nan for season in SEASON_ORDER},
            corridor_effects={},
            seasonal_job_counts={
                season: int((working["season"] == season).sum()) if "season" in working else 0
                for season in SEASON_ORDER
            },
            corridor_job_counts={},
        )

    corridor_counts = working["corridor"].value_counts()
    kept_corridors = corridor_counts[corridor_counts >= max(1, int(min_corridor_jobs))].head(max_corridors)
    kept_corridor_labels = list(kept_corridors.index)
    working["corridor_bucket"] = working["corridor"].where(
        working["corridor"].isin(kept_corridor_labels),
        other="Other corridors",
    )
    bucket_counts = working["corridor_bucket"].value_counts()
    baseline_corridor = str(bucket_counts.idxmax())
    non_baseline_corridors = [
        corridor for corridor in bucket_counts.index.tolist() if str(corridor) != baseline_corridor
    ]

    seasonal_counts = {season: int((working["season"] == season).sum()) for season in SEASON_ORDER}
    baseline_season = next(
        (season for season in SEASON_ORDER if seasonal_counts[season] > 0),
        SEASON_ORDER[0],
    )
    non_baseline_seasons = [season for season in SEASON_ORDER if season != baseline_season]

    baseline_design_columns = [
        np.ones(len(working), dtype=float),
        working["distance_km"].to_numpy(dtype=float),
    ]
    for season in non_baseline_seasons:
        baseline_design_columns.append((working["season"] == season).astype(float).to_numpy(dtype=float))
    baseline_design = np.column_stack(baseline_design_columns)
    target = working["target"].to_numpy(dtype=float)
    _, baseline_r_squared, baseline_rmse = _fit_linear_model(baseline_design, target)

    design_columns = list(baseline_design_columns)
    for corridor in non_baseline_corridors:
        design_columns.append((working["corridor_bucket"] == corridor).astype(float).to_numpy(dtype=float))
    design = np.column_stack(design_columns)
    coefficients, r_squared, rmse = _fit_linear_model(design, target)

    seasonal_effects = {season: 0.0 for season in SEASON_ORDER}
    seasonal_effects[baseline_season] = 0.0
    for index, season in enumerate(non_baseline_seasons, start=2):
        seasonal_effects[season] = float(coefficients[index])

    corridor_effects = {baseline_corridor: 0.0}
    start_index = 2 + len(non_baseline_seasons)
    for offset, corridor in enumerate(non_baseline_corridors):
        corridor_effects[str(corridor)] = float(coefficients[start_index + offset])

    return CorridorMarginModelSummary(
        target_column=target_column,
        corridor_column=corridor_column,
        fitted_job_count=len(working),
        intercept=float(coefficients[0]),
        distance_coeff_per_km=float(coefficients[1]),
        distance_coeff_per_100km=float(coefficients[1] * 100.0),
        r_squared=float(r_squared),
        rmse=float(rmse),
        baseline_r_squared=float(baseline_r_squared),
        baseline_rmse=float(baseline_rmse),
        baseline_season=baseline_season,
        baseline_corridor=baseline_corridor,
        seasonal_effects=seasonal_effects,
        corridor_effects=corridor_effects,
        seasonal_job_counts=seasonal_counts,
        corridor_job_counts={str(key): int(value) for key, value in bucket_counts.items()},
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


def build_corridor_margin_preview(
    summary: CorridorMarginModelSummary,
    *,
    distances_km: tuple[float, ...] = (100.0, 500.0, 1000.0),
    season: str | None = None,
    max_corridors: int = 4,
) -> pd.DataFrame:
    if math.isnan(summary.intercept) or math.isnan(summary.distance_coeff_per_km):
        return pd.DataFrame()

    target_season = str(season or summary.baseline_season)
    seasonal_adjustment = float(summary.seasonal_effects.get(target_season, 0.0) or 0.0)
    ranked_corridors = sorted(
        summary.corridor_job_counts.items(),
        key=lambda item: (-int(item[1]), str(item[0])),
    )[:max_corridors]
    rows: list[dict[str, object]] = []
    for corridor, job_count in ranked_corridors:
        corridor_adjustment = float(summary.corridor_effects.get(str(corridor), 0.0) or 0.0)
        for distance in distances_km:
            predicted_margin = (
                float(summary.intercept)
                + float(summary.distance_coeff_per_km) * float(distance)
                + seasonal_adjustment
                + corridor_adjustment
            )
            rows.append(
                {
                    "Corridor": str(corridor),
                    "Jobs": int(job_count),
                    "Season": target_season.title(),
                    "Distance km": float(distance),
                    "Predicted margin per m3": round(predicted_margin, 2),
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "CorridorMarginModelSummary",
    "MarginRegressionSummary",
    "build_corridor_margin_preview",
    "build_margin_regression_preview",
    "summarise_corridor_margin_model",
    "summarise_margin_regression",
]
