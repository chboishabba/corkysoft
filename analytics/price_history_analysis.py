from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, Sequence

import pandas as pd

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

ORIGIN_COLUMNS = [
    "origin",
    "origin_suburb",
    "origin_city",
]

DESTINATION_COLUMNS = [
    "destination",
    "destination_suburb",
    "destination_city",
]


def _first_present(columns, candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


@dataclass
class PriceHistorySeries:
    current_overall: pd.DataFrame
    previous_year_overall: pd.DataFrame
    current_by_origin: pd.DataFrame
    previous_year_by_origin: pd.DataFrame
    current_by_destination: pd.DataFrame
    previous_year_by_destination: pd.DataFrame


def _empty_history_series() -> PriceHistorySeries:
    empty = pd.DataFrame()
    return PriceHistorySeries(
        current_overall=empty.copy(),
        previous_year_overall=empty.copy(),
        current_by_origin=empty.copy(),
        previous_year_by_origin=empty.copy(),
        current_by_destination=empty.copy(),
        previous_year_by_destination=empty.copy(),
    )


def _normalise_history_frequency(value: str) -> str:
    if not value:
        return "W"
    key = value.upper()
    frequency = PRICE_HISTORY_FREQUENCIES.get(key)
    if frequency is None:
        raise ValueError(f"Unsupported resample frequency: {value}")
    return frequency


def _coerce_timestamp(value: Optional[object]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, (datetime, date)):
        return pd.Timestamp(value)
    try:
        return pd.to_datetime(value, errors="coerce")
    except Exception:
        return None


def _prepare_history_frame(
    df: pd.DataFrame,
    *,
    date_column: str,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    if date_column not in df.columns:
        raise KeyError(f"'{date_column}' column is required for history analysis")

    frame = df.copy()
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    frame = frame.dropna(subset=[date_column])
    if frame.empty:
        return frame

    if start is not None:
        frame = frame[frame[date_column] >= start]
    if end is not None:
        frame = frame[frame[date_column] <= end]
    return frame


def _resample_history(
    df: pd.DataFrame,
    *,
    frequency: str,
    date_column: str,
    group_column: Optional[str] = None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df = df.copy()

    metrics = [
        column
        for column in ("price_per_m3", "margin_per_m3", "margin_total_pct")
        if column in df.columns
    ]

    for column in metrics:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if group_column:
        df = df.dropna(subset=[group_column])
        if df.empty:
            return pd.DataFrame()

    grouper = [pd.Grouper(key=date_column, freq=frequency)]
    rename_map: dict[str, str] = {}
    if group_column:
        grouper.insert(0, group_column)
        if group_column in ORIGIN_COLUMNS:
            rename_map[group_column] = "origin"
        elif group_column in DESTINATION_COLUMNS:
            rename_map[group_column] = "destination"

    grouped = df.groupby(grouper)
    counts = grouped.size().rename("job_count")

    if metrics:
        means = grouped[metrics].mean()
        if isinstance(means, pd.Series):
            means = means.to_frame()
    else:
        means = pd.DataFrame(index=counts.index)

    combined = pd.concat([means, counts.to_frame()], axis=1).reset_index()
    combined = combined.rename(columns={date_column: "period", **rename_map})
    order: list[str] = []
    if group_column:
        order.append(rename_map.get(group_column, group_column))
    order.append("period")
    order.extend(metrics)
    order.append("job_count")
    available = [column for column in order if column in combined.columns]
    combined = combined.loc[:, available]
    combined = combined.sort_values("period")
    return combined


def build_price_history_series(
    df: pd.DataFrame,
    *,
    frequency: str = "W",
    date_column: str = "job_date",
    start_date: Optional[object] = None,
    end_date: Optional[object] = None,
) -> PriceHistorySeries:
    if df.empty:
        return _empty_history_series()

    frequency_alias = _normalise_history_frequency(str(frequency))
    start_ts = _coerce_timestamp(start_date)
    end_ts = _coerce_timestamp(end_date)

    frame = _prepare_history_frame(df, date_column=date_column, start=start_ts, end=end_ts)
    if frame.empty:
        return _empty_history_series()

    derived_start = frame[date_column].min() if start_ts is None else start_ts
    derived_end = frame[date_column].max() if end_ts is None else end_ts

    if pd.isna(derived_start) or pd.isna(derived_end) or derived_start > derived_end:
        return _empty_history_series()

    origin_column = _first_present(frame.columns, ORIGIN_COLUMNS)
    destination_column = _first_present(frame.columns, DESTINATION_COLUMNS)

    current = _prepare_history_frame(
        frame, date_column=date_column, start=derived_start, end=derived_end
    )

    previous_start = derived_start - pd.DateOffset(years=1)
    previous_end = derived_end - pd.DateOffset(years=1)
    previous = _prepare_history_frame(
        df,
        date_column=date_column,
        start=previous_start,
        end=previous_end,
    )

    current_overall = _resample_history(
        current, frequency=frequency_alias, date_column=date_column
    )
    previous_overall = _resample_history(
        previous, frequency=frequency_alias, date_column=date_column
    )

    current_by_origin = (
        _resample_history(
            current,
            frequency=frequency_alias,
            date_column=date_column,
            group_column=origin_column,
        )
        if origin_column
        else pd.DataFrame()
    )
    previous_by_origin = (
        _resample_history(
            previous,
            frequency=frequency_alias,
            date_column=date_column,
            group_column=origin_column,
        )
        if origin_column
        else pd.DataFrame()
    )

    current_by_destination = (
        _resample_history(
            current,
            frequency=frequency_alias,
            date_column=date_column,
            group_column=destination_column,
        )
        if destination_column
        else pd.DataFrame()
    )
    previous_by_destination = (
        _resample_history(
            previous,
            frequency=frequency_alias,
            date_column=date_column,
            group_column=destination_column,
        )
        if destination_column
        else pd.DataFrame()
    )

    return PriceHistorySeries(
        current_overall=current_overall,
        previous_year_overall=previous_overall,
        current_by_origin=current_by_origin,
        previous_year_by_origin=previous_by_origin,
        current_by_destination=current_by_destination,
        previous_year_by_destination=previous_by_destination,
    )


def summarise_last_year_distributions(
    df: pd.DataFrame,
    *,
    date_column: str = "job_date",
    start_date: Optional[object] = None,
    end_date: Optional[object] = None,
) -> dict[str, pd.DataFrame]:
    if df.empty:
        return {
            "overall": pd.DataFrame(),
            "by_origin": pd.DataFrame(),
            "by_destination": pd.DataFrame(),
        }

    start_ts = _coerce_timestamp(start_date)
    end_ts = _coerce_timestamp(end_date)

    frame = _prepare_history_frame(df, date_column=date_column, start=None, end=None)
    if frame.empty:
        return {
            "overall": pd.DataFrame(),
            "by_origin": pd.DataFrame(),
            "by_destination": pd.DataFrame(),
        }

    derived_start = frame[date_column].min() if start_ts is None else start_ts
    derived_end = frame[date_column].max() if end_ts is None else end_ts

    previous_start = derived_start - pd.DateOffset(years=1)
    previous_end = derived_end - pd.DateOffset(years=1)
    previous = _prepare_history_frame(
        frame,
        date_column=date_column,
        start=previous_start,
        end=previous_end,
    )

    if previous.empty:
        return {
            "overall": pd.DataFrame(),
            "by_origin": pd.DataFrame(),
            "by_destination": pd.DataFrame(),
        }

    metrics = [
        column
        for column in ("price_per_m3", "margin_per_m3", "margin_total_pct")
        if column in previous.columns
    ]
    for column in metrics:
        previous[column] = pd.to_numeric(previous[column], errors="coerce")

    origin_column = _first_present(previous.columns, ORIGIN_COLUMNS)
    destination_column = _first_present(previous.columns, DESTINATION_COLUMNS)

    overall_columns = [column for column in metrics if column in previous.columns]
    if date_column in previous.columns:
        overall_columns.insert(0, date_column)
    overall = previous.loc[:, overall_columns].copy()
    overall["series"] = "Previous year"

    by_origin = pd.DataFrame()
    if origin_column:
        origin_columns = [origin_column, *metrics]
        origin_frame = previous.loc[:, [col for col in origin_columns if col in previous.columns]]
        origin_frame = origin_frame.dropna(subset=[origin_column])
        if not origin_frame.empty:
            by_origin = origin_frame.rename(columns={origin_column: "origin"})
            by_origin["series"] = "Previous year"

    by_destination = pd.DataFrame()
    if destination_column:
        destination_columns = [destination_column, *metrics]
        dest_frame = previous.loc[
            :, [col for col in destination_columns if col in previous.columns]
        ]
        dest_frame = dest_frame.dropna(subset=[destination_column])
        if not dest_frame.empty:
            by_destination = dest_frame.rename(columns={destination_column: "destination"})
            by_destination["series"] = "Previous year"

    return {"overall": overall, "by_origin": by_origin, "by_destination": by_destination}
