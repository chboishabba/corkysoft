from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import pandas as pd

from .lane_assignment import LANE_STATUS_ASSIGNED
from .profitability_analysis import (
    classify_profit_band,
    classify_profitability_status,
)

PROFITABILITY_COLOURS = {
    "Below break-even": [217, 83, 79],
    "0-50 above break-even": [240, 173, 78],
    "50-100 above break-even": [91, 192, 222],
    "100+ above break-even": [92, 184, 92],
    "Unknown": [128, 128, 128],
}

PROFITABILITY_BAND_INTENSITY = {
    "Below break-even": 0,
    "0-50 above break-even": 1,
    "50-100 above break-even": 2,
    "100+ above break-even": 3,
    "Unknown": 2,
}

MAX_LANE_WIDTH = 28.0
MIN_LANE_WIDTH = 9.0
LANE_WIDTH_CURVE_EXPONENT = 0.75
ROUTE_WIDTH_METRE_SCALE = 0.25
VOLUME_COLUMNS = [
    "volume_m3",
    "volume_cbm",
    "cbm",
    "cubic_meters",
    "m3",
    "cubic_m",
]


def _profit_band_intensity(band: str) -> float:
    return PROFITABILITY_BAND_INTENSITY.get(band, PROFITABILITY_BAND_INTENSITY["Unknown"])


def compute_profitability_line_width(
    profit_band: str, *, job_scale: Optional[float] = None
) -> float:
    if job_scale is None:
        band_index = _profit_band_intensity(profit_band)
        max_index = max(PROFITABILITY_BAND_INTENSITY.values()) or 1
        position = min(max(band_index / max_index, 0.0), 1.0)
    else:
        try:
            position = float(job_scale)
        except (TypeError, ValueError):
            position = 0.0
        position = min(max(position, 0.0), 1.0)

    width = MIN_LANE_WIDTH + (
        (MAX_LANE_WIDTH - MIN_LANE_WIDTH) * (position ** LANE_WIDTH_CURVE_EXPONENT)
    )
    return round(width, 2)


def _normalise_job_counts(job_counts: pd.Series) -> pd.Series:
    if job_counts.empty:
        return pd.Series(dtype="float64")

    numeric = pd.to_numeric(job_counts, errors="coerce").fillna(0.0).astype(float)
    log_counts = np.log1p(numeric)
    min_log = float(log_counts.min())
    max_log = float(log_counts.max())

    if math.isclose(max_log, min_log):
        baseline = 0.5 if max_log > 0 else 0.0
        return pd.Series(baseline, index=job_counts.index, dtype="float64")

    scale = (log_counts - min_log) / (max_log - min_log)
    return scale.astype(float)


def compute_tapered_route_polygon(row: pd.Series) -> list[list[float]]:
    start_lon = float(row["origin_lon"])
    start_lat = float(row["origin_lat"])
    end_lon = float(row["dest_lon"])
    end_lat = float(row["dest_lat"])

    dx = end_lon - start_lon
    dy = end_lat - start_lat
    if dx == 0 and dy == 0:
        return [[start_lon, start_lat]]

    mid_lon = (start_lon + end_lon) / 2
    mid_lat = (start_lat + end_lat) / 2

    metres_per_degree_lat = 111_320
    metres_per_degree_lon = max(1e-9, metres_per_degree_lat * math.cos(math.radians(mid_lat)))

    dir_x_m = dx * metres_per_degree_lon
    dir_y_m = dy * metres_per_degree_lat
    vector_length = math.hypot(dir_x_m, dir_y_m)
    if vector_length == 0:
        return [[start_lon, start_lat]]

    unit_dir_x = dir_x_m / vector_length
    unit_dir_y = dir_y_m / vector_length

    perp_x = -unit_dir_y
    perp_y = unit_dir_x

    width_m = float(row["line_width"]) * ROUTE_WIDTH_METRE_SCALE
    offset_x_m = perp_x * width_m / 2
    offset_y_m = perp_y * width_m / 2

    offset_lon = offset_x_m / metres_per_degree_lon
    offset_lat = offset_y_m / metres_per_degree_lat

    left_mid = [mid_lon - offset_lon, mid_lat - offset_lat]
    right_mid = [mid_lon + offset_lon, mid_lat + offset_lat]

    return [
        [start_lon, start_lat],
        left_mid,
        [end_lon, end_lat],
        right_mid,
    ]


def _first_present(columns, candidates: list[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _route_display_name(route: pd.Series) -> str:
    for column in (
        "corridor_display",
        "corridor",
        "lane",
        "lane_name",
    ):
        value = route.get(column)
        if isinstance(value, str) and value.strip():
            return value.strip()

    origin_column = _first_present(route.index, ["origin", "origin_suburb", "origin_city"])
    destination_column = _first_present(
        route.index, ["destination", "destination_suburb", "destination_city"]
    )
    origin = str(route.get(origin_column, "")).strip() if origin_column else ""
    destination = str(route.get(destination_column, "")).strip() if destination_column else ""
    if origin and destination:
        return f"{origin} → {destination}"
    if origin or destination:
        return origin or destination
    return "Unknown corridor"


def prepare_profitability_route_data(
    df: pd.DataFrame,
    break_even: float,
) -> pd.DataFrame:
    required_columns = {
        "origin_lat",
        "origin_lon",
        "dest_lat",
        "dest_lon",
        "price_per_m3",
    }
    if not required_columns.issubset(df.columns):
        return pd.DataFrame(
            columns=[
                "id",
                "origin_lat",
                "origin_lon",
                "dest_lat",
                "dest_lon",
                "price_per_m3",
                "profit_band",
                "profitability_status",
                "colour",
                "tooltip",
            ]
        )

    map_df = df.dropna(subset=["origin_lat", "origin_lon", "dest_lat", "dest_lon"]).copy()
    if map_df.empty:
        return map_df

    map_df["_origin_lat"] = pd.to_numeric(map_df["origin_lat"], errors="coerce")
    map_df["_origin_lon"] = pd.to_numeric(map_df["origin_lon"], errors="coerce")
    map_df["_dest_lat"] = pd.to_numeric(map_df["dest_lat"], errors="coerce")
    map_df["_dest_lon"] = pd.to_numeric(map_df["dest_lon"], errors="coerce")
    map_df = map_df.dropna(
        subset=["_origin_lat", "_origin_lon", "_dest_lat", "_dest_lon"]
    ).copy()
    if map_df.empty:
        return map_df

    map_df["_price_numeric"] = pd.to_numeric(map_df["price_per_m3"], errors="coerce")
    if "break_even_per_m3" in map_df.columns:
        map_df["_break_even_numeric"] = pd.to_numeric(
            map_df["break_even_per_m3"], errors="coerce"
        ).fillna(break_even)
    else:
        map_df["_break_even_numeric"] = pd.Series(break_even, index=map_df.index, dtype="float64")

    volume_column = _first_present(list(map_df.columns), VOLUME_COLUMNS)
    if volume_column:
        map_df["_volume_numeric"] = pd.to_numeric(map_df[volume_column], errors="coerce")
    else:
        map_df["_volume_numeric"] = math.nan

    map_df["_route_label"] = map_df.apply(_route_display_name, axis=1)

    def _lane_identifier(row: pd.Series) -> str:
        if str(row.get("lane_assignment_status", "")).strip().lower() == LANE_STATUS_ASSIGNED:
            lane_key = str(row.get("lane_key", "")).strip()
            if lane_key:
                return lane_key
        label = row.get("_route_label") or "Unknown corridor"
        origin = (row.get("_origin_lat"), row.get("_origin_lon"))
        destination = (row.get("_dest_lat"), row.get("_dest_lon"))
        return f"{label}|{origin[0]:.2f},{origin[1]:.2f}→{destination[0]:.2f},{destination[1]:.2f}"

    map_df["_lane_key"] = map_df.apply(_lane_identifier, axis=1)

    total_jobs = len(map_df)
    lane_rows: list[dict[str, Any]] = []
    grouped = map_df.groupby("_lane_key", dropna=False)
    for lane_key, group in grouped:
        job_count = int(len(group))
        price_values = group["_price_numeric"].dropna()
        priced_job_count = int(len(price_values))

        if priced_job_count:
            weights = (
                group.loc[price_values.index, "_volume_numeric"].fillna(0.0)
                if "_volume_numeric" in group
                else pd.Series(0.0, index=price_values.index)
            )
            positive_weights = weights > 0
            if positive_weights.any():
                lane_price = float(
                    np.average(
                        price_values.loc[positive_weights],
                        weights=weights.loc[positive_weights],
                    )
                )
            else:
                lane_price = float(price_values.mean())
        else:
            lane_price = math.nan

        break_even_values = group["_break_even_numeric"].dropna()
        lane_break_even = float(break_even_values.mean()) if not break_even_values.empty else float(break_even)
        lane_volume = group["_volume_numeric"].sum(min_count=1)
        lane_volume = float(lane_volume) if pd.notna(lane_volume) else math.nan

        lane_rows.append(
            {
                "lane_key": lane_key,
                "corridor_display": group["_route_label"].iloc[0] or "Unknown corridor",
                "origin_lat": float(group["_origin_lat"].mean()),
                "origin_lon": float(group["_origin_lon"].mean()),
                "dest_lat": float(group["_dest_lat"].mean()),
                "dest_lon": float(group["_dest_lon"].mean()),
                "price_per_m3": lane_price,
                "break_even_per_m3": lane_break_even,
                "job_count": job_count,
                "priced_job_count": priced_job_count,
                "total_volume_m3": lane_volume,
                "share_of_jobs": job_count / total_jobs if total_jobs else 0.0,
            }
        )

    lane_df = pd.DataFrame(lane_rows)
    if lane_df.empty:
        return map_df

    lane_df["profit_band"] = [
        classify_profit_band(price, be)
        for price, be in zip(lane_df["price_per_m3"], lane_df["break_even_per_m3"])
    ]
    lane_df["profitability_status"] = [
        classify_profitability_status(price, be)
        for price, be in zip(lane_df["price_per_m3"], lane_df["break_even_per_m3"])
    ]
    lane_df["colour"] = lane_df["profit_band"].map(PROFITABILITY_COLOURS)
    lane_df["colour"] = lane_df["colour"].apply(
        lambda value: value if isinstance(value, (list, tuple)) else [128, 128, 128]
    )
    lane_df["fill_colour"] = lane_df["colour"].apply(
        lambda value: [int(component) for component in list(value)[:3]] + [102]
    )

    job_scale = _normalise_job_counts(lane_df["job_count"])
    lane_df["line_width"] = [
        compute_profitability_line_width(band, job_scale=scale)
        for band, scale in zip(lane_df["profit_band"], job_scale)
    ]
    lane_df["route_polygon"] = lane_df.apply(compute_tapered_route_polygon, axis=1)

    def _format_tooltip(row: pd.Series) -> str:
        corridor = row.get("corridor_display") or "Corridor"
        status = row.get("profitability_status") or row.get("profit_band", "Unknown")
        band = row.get("profit_band")
        descriptor = f"{status} – {band}" if band and band not in {"Unknown", status} else status

        price = row.get("price_per_m3")
        price_text = "n/a" if pd.isna(price) else f"${float(price):,.0f} per m³"
        break_even_value = row.get("break_even_per_m3")
        diff_detail: Optional[str] = None
        if not pd.isna(price) and not pd.isna(break_even_value):
            diff_detail = f"Δ {float(price) - float(break_even_value):+.0f} vs break-even"

        job_count_value = row.get("job_count")
        job_detail: Optional[str] = None
        if isinstance(job_count_value, (int, float)) and not math.isnan(job_count_value):
            job_int = int(job_count_value)
            job_detail = f"{job_int} job{'s' if job_int != 1 else ''}"

        details = [detail for detail in (price_text, diff_detail, job_detail) if detail]
        return f"{corridor}: {descriptor} ({'; '.join(details)})" if details else f"{corridor}: {descriptor}"

    lane_df["tooltip"] = lane_df.apply(_format_tooltip, axis=1)
    lane_df = lane_df.set_index("lane_key")

    drop_columns = [
        "origin_lat",
        "origin_lon",
        "dest_lat",
        "dest_lon",
        "price_per_m3",
        "break_even_per_m3",
        "corridor_display",
    ]
    existing_drop = [column for column in drop_columns if column in map_df.columns]
    if existing_drop:
        map_df = map_df.drop(columns=existing_drop)

    map_df = map_df.join(lane_df, on="_lane_key")
    map_df = map_df.rename(columns={"_lane_key": "lane_key"})

    cleanup_columns = [
        "_route_label",
        "_price_numeric",
        "_break_even_numeric",
        "_volume_numeric",
        "_origin_lat",
        "_origin_lon",
        "_dest_lat",
        "_dest_lon",
    ]
    drop_cleanup = [column for column in cleanup_columns if column in map_df.columns]
    if drop_cleanup:
        map_df = map_df.drop(columns=drop_cleanup)

    return map_df


def prepare_profitability_map_data(
    df: pd.DataFrame,
    break_even: float,
    *,
    placeholder: str = "Unknown",
) -> pd.DataFrame:
    map_df = prepare_profitability_route_data(df, break_even)
    if map_df.empty:
        return map_df

    colour_values = map_df.get("profit_band")
    if colour_values is None:
        map_df["map_colour_value"] = placeholder
    else:
        map_df["map_colour_value"] = colour_values.fillna(placeholder).astype(str)
    return map_df
