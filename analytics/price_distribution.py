"""Helpers for the price-distribution Streamlit view."""
from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Dict, Iterable, Literal, Optional, Sequence, TYPE_CHECKING


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pandas.api.types import is_datetime64_any_dtype

from .db import (
    bootstrap_parameters,
    ensure_dashboard_tables,
    ensure_global_parameters_table,
    get_parameter_value,
    migrate_geojson_to_routes,
    set_parameter_value,
)
from .lane_assignment import (
    LANE_STATUS_ASSIGNED,
    backfill_lane_assignments,
    ensure_lane_assignment_schema,
)
from .price_history_analysis import (
    PriceHistorySeries,
    build_price_history_series,
    summarise_last_year_distributions,
)
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
from .routing_provider import RoutingProvider, get_routing_provider
from .routes_map import populate_route_geometry


try:  # pragma: no cover - availability exercised via tests
    from corkysoft.routing import geocode_cached as _geocode_cached
except Exception:  # pragma: no cover - optional dependency
    geocode_cached = None  # type: ignore[assignment]
else:
    geocode_cached = _geocode_cached

_QUOTE_COUNTRY_DEFAULT = "Australia"

if TYPE_CHECKING:  # pragma: no cover - typing only
    from openrouteservice import Client as ORSClient
else:
    ORSClient = Any  # type: ignore[misc, assignment]


logger = logging.getLogger(__name__)

METRO_HISTOGRAM_BINS = 15
HISTORICAL_INGEST_USABLE = "usable"
HISTORICAL_INGEST_USABLE_WITH_GAPS = "usable_with_gaps"
HISTORICAL_INGEST_NOT_READY = "not_ready"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


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
METRO_DISTANCE_THRESHOLD_KM = 100.0

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

HEATMAP_WEIGHTING_CANDIDATES: Sequence[tuple[str, Optional[str]]] = (
    ("Job count", None),
    ("Volume (m³)", "volume_m3"),
    ("Margin ($)", "margin_total"),
    ("Margin per m³", "margin_per_m3"),
    ("Margin %", "margin_total_pct"),
    ("Margin per m³ %", "margin_per_m3_pct"),
)

# Candidate column names used in legacy exports.
DATE_COLUMNS = [
    "job_date",
    "move_date",
    "delivery_date",
    "created_at",
    "updated_at",
    "date",
    "quote_date",
]
CLIENT_COLUMNS = [
    "client",
    "client_name",
    "account",
    "customer",
]
VOLUME_COLUMNS = [
    "volume_m3",
    "volume_cbm",
    "cbm",
    "cubic_meters",
    "m3",
    "cubic_m",
]
REVENUE_COLUMNS = [
    "revenue_total",
    "sell_total",
    "total_revenue",
    "price_total",
    "quoted_sell",
    "final_quote",
]
PRICE_COLUMNS = [
    "revenue_per_m3",
    "price_per_m3",
    "sell_per_m3",
    "rate_per_m3",
]
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
POSTCODE_COLUMNS = [
    "origin_postcode",
    "origin_postal",
    "origin_pc",
    "destination_postcode",
    "destination_postal",
    "destination_pc",
]
CORRIDOR_COLUMNS = [
    "corridor",
    "lane",
    "lane_name",
]
DISTANCE_COLUMNS = [
    "distance_km",
    "distance",
    "km",
    "kms",
    "kilometers",
    "kilometres",
]
FINAL_COST_COLUMNS = [
    "final_cost",
    "final_total",
    "actual_cost",
    "actual_total",
    "final_sell",
    "final_sell_total",
    "actual_sell",
    "cost_total",
    "total_cost",
    "final_price",
    "final_amount",
    "total_before_margin",
]

ORIGIN_POSTCODE_CANDIDATES = [
    "origin_postcode",
    "origin_postal",
    "origin_pc",
]

DESTINATION_POSTCODE_CANDIDATES = [
    "destination_postcode",
    "destination_postal",
    "destination_pc",
]


@dataclass
class ColumnMapping:
    date: Optional[str]
    client: Optional[str]
    price: Optional[str]
    revenue: Optional[str]
    volume: Optional[str]
    origin: Optional[str]
    destination: Optional[str]
    corridor: Optional[str]
    distance: Optional[str]
    final_cost: Optional[str]


@dataclass(frozen=True)
class BaseCostConfig:
    fuel_per_km: float
    driver_per_km: float
    maintenance_per_km: float
    overhead_per_job: float

    @property
    def per_km_total(self) -> float:
        return self.fuel_per_km + self.driver_per_km + self.maintenance_per_km


def _first_present(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    columns_lower = {c.lower(): c for c in columns}
    for candidate in candidates:
        lower = candidate.lower()
        if lower in columns_lower:
            return columns_lower[lower]
    return None


def _infer_datetime_parse_kwargs(series: pd.Series) -> dict[str, Any]:
    """Infer keyword arguments for :func:`pandas.to_datetime` for *series*."""
    if is_datetime64_any_dtype(series):
        return {}

    sample = (
        series.dropna()
        .astype(str)
        .str.strip()
        .replace({"": np.nan})
        .dropna()
    )

    if sample.empty:
        return {}

    sample_values = sample.head(20).tolist()

    iso_date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    if all(iso_date_pattern.match(value) for value in sample_values):
        return {"format": "%Y-%m-%d"}

    slash_pattern = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")
    slash_values = [value for value in sample_values if slash_pattern.match(value)]
    if slash_values and len(slash_values) == len(sample_values):
        numeric_parts = [tuple(int(part) for part in value.split("/")) for value in slash_values]
        if any(parts[0] > 12 for parts in numeric_parts):
            return {"dayfirst": True}
        if any(parts[1] > 12 for parts in numeric_parts):
            return {"dayfirst": False}
        # Ambiguous day/month ordering; prefer day-first to match AU/EU data dumps.
        return {"dayfirst": True}

    dash_pattern = re.compile(r"^\d{1,2}-\d{1,2}-\d{4}$")
    dash_values = [value for value in sample_values if dash_pattern.match(value)]
    if dash_values and len(dash_values) == len(sample_values):
        numeric_parts = [tuple(int(part) for part in value.split("-")) for value in dash_values]
        if any(parts[0] > 12 for parts in numeric_parts):
            return {"dayfirst": True}
        if any(parts[1] > 12 for parts in numeric_parts):
            return {"dayfirst": False}
        return {"dayfirst": True}

    return {}


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


def infer_columns(df: pd.DataFrame) -> ColumnMapping:
    cols = df.columns
    return ColumnMapping(
        date=_first_present(cols, DATE_COLUMNS),
        client=_first_present(cols, CLIENT_COLUMNS),
        price=_first_present(cols, PRICE_COLUMNS),
        revenue=_first_present(cols, REVENUE_COLUMNS),
        volume=_first_present(cols, VOLUME_COLUMNS),
        origin=_first_present(cols, ORIGIN_COLUMNS),
        destination=_first_present(cols, DESTINATION_COLUMNS),
        corridor=_first_present(cols, CORRIDOR_COLUMNS),
        distance=_first_present(cols, DISTANCE_COLUMNS),
        final_cost=_first_present(cols, FINAL_COST_COLUMNS),
    )


def _clean_string(value: Any) -> Optional[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return str(value).strip() or None


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
        return float(value)
    except (TypeError, ValueError):
        return None


_DEDUP_FLOAT_PRECISION = 6


def _normalise_float_for_key(value: Any) -> Optional[float]:
    numeric = _safe_float(value)
    if numeric is None or math.isnan(numeric):
        return None
    return round(numeric, _DEDUP_FLOAT_PRECISION)


def _build_job_identity_key(
    job_date: Any,
    origin: Any,
    destination: Any,
    client: Any,
    price_per_m3: Any,
    volume_m3: Any,
    revenue_total: Any,
    distance_km: Any,
    final_cost: Any,
) -> tuple[Any, ...]:
    origin_clean = _clean_string(origin)
    destination_clean = _clean_string(destination)
    client_clean = _clean_string(client) or ""
    job_date_clean = str(job_date).strip() if job_date is not None else ""

    return (
        job_date_clean,
        origin_clean,
        destination_clean,
        client_clean,
        _normalise_float_for_key(price_per_m3),
        _normalise_float_for_key(volume_m3),
        _normalise_float_for_key(revenue_total),
        _normalise_float_for_key(distance_km),
        _normalise_float_for_key(final_cost),
    )


def _infer_postcode_columns(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    lower_to_column = {column.lower(): column for column in df.columns}
    origin_column = next(
        (
            lower_to_column[candidate]
            for candidate in (c.lower() for c in ORIGIN_POSTCODE_CANDIDATES)
            if candidate in lower_to_column
        ),
        None,
    )
    destination_column = next(
        (
            lower_to_column[candidate]
            for candidate in (c.lower() for c in DESTINATION_POSTCODE_CANDIDATES)
            if candidate in lower_to_column
        ),
        None,
    )
    return origin_column, destination_column


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


def _first_non_empty_value(row: pd.Series, columns: Sequence[str]) -> Optional[str]:
    for column in columns:
        if column not in row:
            continue
        value = row[column]
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                return cleaned
            continue
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def enrich_missing_route_coordinates(
    df: pd.DataFrame,
    conn: sqlite3.Connection,
    *,
    country: Optional[str] = None,
    origin_candidates: Optional[Sequence[str]] = None,
    destination_candidates: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Ensure ``df`` contains latitude/longitude columns by geocoding missing values."""

    if df.empty:
        return df

    if geocode_cached is None:  # pragma: no cover - dependency not available
        return df

    if conn is None:  # pragma: no cover - defensive guard for callers
        return df

    map_columns = ("origin_lon", "origin_lat", "dest_lon", "dest_lat")
    columns = set(df.columns)
    needs_enrichment = any(column not in columns for column in map_columns)
    if not needs_enrichment:
        needs_enrichment = any(
            df[column].isna().all()
            for column in map_columns
            if column in df.columns
        )

    if not needs_enrichment:
        return df

    working = df.copy()
    for column in map_columns:
        if column not in working.columns:
            working[column] = pd.Series(np.nan, index=working.index, dtype=float)

    resolved_origin_candidates = origin_candidates or (
        "origin_resolved",
        "origin_normalized",
        "origin",
        "origin_raw",
        "origin_city",
    )
    resolved_destination_candidates = destination_candidates or (
        "destination_resolved",
        "destination_normalized",
        "destination",
        "destination_raw",
        "destination_city",
    )

    default_country = (country or "").strip() or _QUOTE_COUNTRY_DEFAULT
    origin_cache: dict[tuple[str, str], Optional[tuple[float, float]]] = {}
    destination_cache: dict[tuple[str, str], Optional[tuple[float, float]]] = {}

    for idx, row in working.iterrows():
        if pd.isna(working.at[idx, "origin_lon"]) or pd.isna(working.at[idx, "origin_lat"]):
            origin_place = _first_non_empty_value(row, resolved_origin_candidates)
            if origin_place:
                origin_country = _first_non_empty_value(row, ("origin_country",)) or default_country
                cache_key = (origin_place, origin_country)
                coords = origin_cache.get(cache_key)
                if cache_key not in origin_cache:
                    try:
                        result = geocode_cached(conn, origin_place, origin_country)
                    except Exception as exc:  # pragma: no cover - surfaces via logging
                        logger.warning(
                            "Failed to geocode origin '%s' for country '%s': %s",
                            origin_place,
                            origin_country,
                            exc,
                        )
                        origin_cache[cache_key] = None
                        coords = None
                    else:
                        coords = (float(result.lon), float(result.lat))
                        origin_cache[cache_key] = coords
                if coords:
                    working.at[idx, "origin_lon"] = coords[0]
                    working.at[idx, "origin_lat"] = coords[1]

        if pd.isna(working.at[idx, "dest_lon"]) or pd.isna(working.at[idx, "dest_lat"]):
            destination_place = _first_non_empty_value(row, resolved_destination_candidates)
            if destination_place:
                destination_country = (
                    _first_non_empty_value(row, ("destination_country",)) or default_country
                )
                cache_key = (destination_place, destination_country)
                coords = destination_cache.get(cache_key)
                if cache_key not in destination_cache:
                    try:
                        result = geocode_cached(conn, destination_place, destination_country)
                    except Exception as exc:  # pragma: no cover - surfaces via logging
                        logger.warning(
                            "Failed to geocode destination '%s' for country '%s': %s",
                            destination_place,
                            destination_country,
                            exc,
                        )
                        destination_cache[cache_key] = None
                        coords = None
                    else:
                        coords = (float(result.lon), float(result.lat))
                        destination_cache[cache_key] = coords
                if coords:
                    working.at[idx, "dest_lon"] = coords[0]
                    working.at[idx, "dest_lat"] = coords[1]

    return working


def import_historical_jobs_from_dataframe(
    conn,
    df: pd.DataFrame,
    *,
    source_name: str | None = None,
) -> tuple[int, int]:
    """Insert ``df`` rows into ``historical_jobs`` and return ``(inserted, skipped)``.

    The importer uses :func:`infer_columns` to discover relevant fields and performs
    light validation before inserting rows. Rows missing a job date, origin,
    destination or price signal are skipped. Duplicate rows, identified via the
    combination of ``(job_date, origin, destination, client, price_per_m3,
    volume_m3, revenue_total, distance_km, final_cost)``, are ignored. When the
    distinguishing metrics are missing they fall back to ``NULL`` so exact
    duplicates remain filtered while legitimate same-corridor jobs with
    different volumes or pricing are retained.
    """

    ensure_global_parameters_table(conn)
    ensure_dashboard_tables(conn)
    ensure_historical_ingest_tables(conn)
    ensure_lane_assignment_schema(conn)

    if df.empty:
        create_historical_ingest_run(
            conn,
            source_name=source_name or "dataframe_upload",
            total_rows=0,
            inserted_rows=0,
            skipped_rows=0,
            duplicate_rows=0,
            invalid_rows=0,
            issues=[],
        )
        return 0, 0

    mapping = infer_columns(df)
    if mapping.date is None:
        raise ValueError("Unable to infer a job date column from the uploaded data.")
    if mapping.origin is None or mapping.destination is None:
        raise ValueError(
            "Uploaded data must include origin and destination columns."
        )
    if mapping.price is None and (mapping.revenue is None or mapping.volume is None):
        raise ValueError(
            "Uploaded data must include a price-per-m³ column or both revenue and volume columns."
        )

    parse_kwargs = _infer_datetime_parse_kwargs(df[mapping.date])
    dates = pd.to_datetime(df[mapping.date], errors="coerce", **parse_kwargs)

    origin_pc_col, dest_pc_col = _infer_postcode_columns(df)

    existing_rows = conn.execute(
        """
        SELECT
            job_date,
            origin,
            destination,
            client,
            price_per_m3,
            volume_m3,
            revenue_total,
            distance_km,
            final_cost
        FROM historical_jobs
        """
    ).fetchall()
    existing_keys = {
        _build_job_identity_key(
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            row[5],
            row[6],
            row[7],
            row[8],
        )
        for row in existing_rows
    }

    now = datetime.now(UTC).isoformat()
    to_insert: list[tuple[Any, ...]] = []
    issues: list[dict[str, Any]] = []
    skipped = 0
    duplicate_rows = 0
    invalid_rows = 0
    valid_rows = 0

    for idx in range(len(df)):
        source_row_ref = _historical_source_row_ref(df, idx)
        job_date = dates.iloc[idx]
        if pd.isna(job_date):
            invalid_rows += 1
            skipped += 1
            issues.append(
                _historical_ingest_issue(
                    idx,
                    source_row_ref,
                    "missing_job_date",
                    "error",
                    "Row skipped because the job date could not be parsed.",
                )
            )
            continue

        origin_value = _clean_string(df.iloc[idx][mapping.origin])
        destination_value = _clean_string(df.iloc[idx][mapping.destination])
        if not origin_value or not destination_value:
            invalid_rows += 1
            skipped += 1
            issues.append(
                _historical_ingest_issue(
                    idx,
                    source_row_ref,
                    "missing_route_endpoint",
                    "error",
                    "Row skipped because origin or destination was missing.",
                )
            )
            continue

        client_value = _clean_string(df.iloc[idx][mapping.client]) if mapping.client else None

        price_value = (
            _safe_float(df.iloc[idx][mapping.price]) if mapping.price else None
        )
        revenue_value = (
            _safe_float(df.iloc[idx][mapping.revenue]) if mapping.revenue else None
        )
        volume_value = (
            _safe_float(df.iloc[idx][mapping.volume]) if mapping.volume else None
        )
        distance_value = (
            _safe_float(df.iloc[idx][mapping.distance]) if mapping.distance else None
        )
        final_cost_value = (
            _safe_float(df.iloc[idx][mapping.final_cost]) if mapping.final_cost else None
        )

        if price_value is None and revenue_value is not None and volume_value:
            if volume_value == 0:
                price_value = None
            else:
                price_value = revenue_value / volume_value

        if price_value is None:
            invalid_rows += 1
            skipped += 1
            issues.append(
                _historical_ingest_issue(
                    idx,
                    source_row_ref,
                    "missing_price_signal",
                    "error",
                    "Row skipped because price per m3 could not be derived.",
                )
            )
            continue

        corridor_value: Optional[str]
        if mapping.corridor and mapping.corridor in df.columns:
            corridor_value = _clean_string(df.iloc[idx][mapping.corridor])
        else:
            corridor_value = f"{origin_value} → {destination_value}"

        origin_postcode = (
            _clean_string(df.iloc[idx][origin_pc_col]) if origin_pc_col else None
        )
        dest_postcode = (
            _clean_string(df.iloc[idx][dest_pc_col]) if dest_pc_col else None
        )

        valid_rows += 1
        key = _build_job_identity_key(
            job_date.date().isoformat(),
            origin_value,
            destination_value,
            client_value,
            price_value,
            volume_value,
            revenue_value,
            distance_value,
            final_cost_value,
        )
        if key in existing_keys:
            duplicate_rows += 1
            skipped += 1
            issues.append(
                _historical_ingest_issue(
                    idx,
                    source_row_ref,
                    "duplicate_row",
                    "warning",
                    "Row skipped because an identical historical job already exists.",
                )
            )
            continue

        existing_keys.add(key)

        to_insert.append(
            (
                key[0],
                client_value,
                corridor_value,
                float(price_value),
                revenue_value,
                revenue_value,
                volume_value,
                volume_value,
                distance_value,
                final_cost_value,
                origin_value,
                destination_value,
                origin_postcode,
                dest_postcode,
                now,
                now,
            )
        )

    inserted_ids: list[int] = []
    if to_insert:
        for row in to_insert:
            cursor = conn.execute(
                """
                INSERT INTO historical_jobs (
                    job_date,
                    client,
                    corridor_display,
                    price_per_m3,
                    revenue_total,
                    revenue,
                    volume_m3,
                    volume,
                    distance_km,
                    final_cost,
                    origin,
                    destination,
                    origin_postcode,
                    destination_postcode,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )
            inserted_ids.append(int(cursor.lastrowid))
        conn.commit()

    if inserted_ids:
        try:
            populate_route_geometry(conn, inserted_ids, dataset="historical")
        except Exception:
            # Route enrichment is best-effort; imports should not fail because a provider is unavailable.
            pass
        backfill_lane_assignments(conn, dataset="historical", row_ids=inserted_ids)

    create_historical_ingest_run(
        conn,
        source_name=source_name or "dataframe_upload",
        total_rows=len(df),
        inserted_rows=len(to_insert),
        skipped_rows=skipped,
        duplicate_rows=duplicate_rows,
        invalid_rows=invalid_rows,
        valid_rows=valid_rows,
        issues=issues,
    )

    return len(to_insert), skipped


def ensure_historical_ingest_tables(conn: sqlite3.Connection) -> None:
    """Ensure durable ingest-run and row-issue tables exist."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS historical_ingest_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_name TEXT NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT NOT NULL,
            total_rows INTEGER NOT NULL DEFAULT 0,
            valid_rows INTEGER NOT NULL DEFAULT 0,
            inserted_rows INTEGER NOT NULL DEFAULT 0,
            skipped_rows INTEGER NOT NULL DEFAULT 0,
            duplicate_rows INTEGER NOT NULL DEFAULT 0,
            invalid_rows INTEGER NOT NULL DEFAULT 0,
            issue_count INTEGER NOT NULL DEFAULT 0,
            readiness_status TEXT NOT NULL,
            coverage_summary TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS historical_ingest_issues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            row_index INTEGER NOT NULL,
            source_row_ref TEXT,
            issue_code TEXT NOT NULL,
            severity TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES historical_ingest_runs(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_historical_ingest_runs_completed
        ON historical_ingest_runs(completed_at DESC, id DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_historical_ingest_issues_run
        ON historical_ingest_issues(run_id, severity, issue_code)
        """
    )
    conn.commit()


def create_historical_ingest_run(
    conn: sqlite3.Connection,
    *,
    source_name: str,
    total_rows: int,
    inserted_rows: int,
    skipped_rows: int,
    duplicate_rows: int,
    invalid_rows: int,
    issues: Sequence[dict[str, Any]],
    valid_rows: int | None = None,
) -> dict[str, Any]:
    """Persist one historical-ingest run and any row-level issues."""

    ensure_historical_ingest_tables(conn)
    started_at = _utc_now_iso()
    completed_at = _utc_now_iso()
    valid = int(valid_rows if valid_rows is not None else max(0, total_rows - invalid_rows))
    coverage = summarize_historical_ingest_counts(
        total_rows=total_rows,
        valid_rows=valid,
        inserted_rows=inserted_rows,
        skipped_rows=skipped_rows,
        duplicate_rows=duplicate_rows,
        invalid_rows=invalid_rows,
        issues=issues,
    )
    cursor = conn.execute(
        """
        INSERT INTO historical_ingest_runs (
            source_name,
            started_at,
            completed_at,
            total_rows,
            valid_rows,
            inserted_rows,
            skipped_rows,
            duplicate_rows,
            invalid_rows,
            issue_count,
            readiness_status,
            coverage_summary
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            source_name,
            started_at,
            completed_at,
            int(total_rows),
            valid,
            int(inserted_rows),
            int(skipped_rows),
            int(duplicate_rows),
            int(invalid_rows),
            len(issues),
            coverage["readinessStatus"],
            json.dumps(coverage, sort_keys=True),
        ),
    )
    run_id = int(cursor.lastrowid)
    if issues:
        conn.executemany(
            """
            INSERT INTO historical_ingest_issues (
                run_id,
                row_index,
                source_row_ref,
                issue_code,
                severity,
                message,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    int(issue["rowIndex"]),
                    issue.get("sourceRowRef"),
                    issue["issueCode"],
                    issue["severity"],
                    issue["message"],
                    completed_at,
                )
                for issue in issues
            ],
        )
    conn.commit()
    return get_historical_ingest_run(conn, run_id)


def summarize_historical_ingest_counts(
    *,
    total_rows: int,
    valid_rows: int,
    inserted_rows: int,
    skipped_rows: int,
    duplicate_rows: int,
    invalid_rows: int,
    issues: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Return coverage and readiness summary for one ingest batch."""

    total = int(total_rows)
    valid = int(valid_rows)
    inserted = int(inserted_rows)
    skipped = int(skipped_rows)
    duplicates = int(duplicate_rows)
    invalid = int(invalid_rows)
    coverage_ratio = (valid / total) if total else 0.0
    inserted_ratio = (inserted / total) if total else 0.0
    issue_counts: dict[str, int] = {}
    for issue in issues:
        code = str(issue["issueCode"])
        issue_counts[code] = issue_counts.get(code, 0) + 1
    top_issue_codes = [
        {"issueCode": code, "count": count}
        for code, count in sorted(issue_counts.items(), key=lambda item: (-item[1], item[0]))
    ][:5]
    if valid == 0:
        readiness = HISTORICAL_INGEST_NOT_READY
    elif coverage_ratio >= 0.95:
        readiness = HISTORICAL_INGEST_USABLE
    elif coverage_ratio >= 0.5:
        readiness = HISTORICAL_INGEST_USABLE_WITH_GAPS
    else:
        readiness = HISTORICAL_INGEST_NOT_READY
    return {
        "totalRows": total,
        "validRows": valid,
        "insertedRows": inserted,
        "skippedRows": skipped,
        "duplicateRows": duplicates,
        "invalidRows": invalid,
        "issueCount": len(issues),
        "coverageRatio": round(coverage_ratio, 4),
        "insertedRatio": round(inserted_ratio, 4),
        "readinessStatus": readiness,
        "topIssueCodes": top_issue_codes,
    }


def get_historical_ingest_run(conn: sqlite3.Connection, run_id: int) -> dict[str, Any]:
    """Return one historical-ingest run with issues."""

    ensure_historical_ingest_tables(conn)
    cursor = conn.execute(
        """
        SELECT
            id,
            source_name,
            started_at,
            completed_at,
            total_rows,
            valid_rows,
            inserted_rows,
            skipped_rows,
            duplicate_rows,
            invalid_rows,
            issue_count,
            readiness_status,
            coverage_summary
        FROM historical_ingest_runs
        WHERE id = ?
        """,
        (int(run_id),),
    )
    row = cursor.fetchone()
    if row is None:
        raise ValueError(f"Unknown historical ingest run: {run_id}")
    columns = [column[0] for column in cursor.description or []]
    payload = dict(zip(columns, row, strict=False))
    payload["coverage_summary"] = json.loads(payload["coverage_summary"])
    issue_cursor = conn.execute(
        """
        SELECT row_index, source_row_ref, issue_code, severity, message, created_at
        FROM historical_ingest_issues
        WHERE run_id = ?
        ORDER BY row_index, id
        """,
        (int(run_id),),
    )
    issue_columns = [column[0] for column in issue_cursor.description or []]
    payload["issues"] = [dict(zip(issue_columns, issue_row, strict=False)) for issue_row in issue_cursor.fetchall()]
    return payload


def latest_historical_ingest_summary(conn: sqlite3.Connection) -> dict[str, Any] | None:
    """Return the most recent historical-ingest run summary if present."""

    ensure_historical_ingest_tables(conn)
    row = conn.execute(
        "SELECT id FROM historical_ingest_runs ORDER BY completed_at DESC, id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    return get_historical_ingest_run(conn, int(row[0]))


def _historical_ingest_issue(
    row_index: int,
    source_row_ref: str | None,
    issue_code: str,
    severity: str,
    message: str,
) -> dict[str, Any]:
    return {
        "rowIndex": int(row_index),
        "sourceRowRef": source_row_ref,
        "issueCode": issue_code,
        "severity": severity,
        "message": message,
    }


def _historical_source_row_ref(df: pd.DataFrame, idx: int) -> str | None:
    for column in ("id", "job_id", "job_number", "reference", "external_id"):
        if column in df.columns:
            value = _clean_string(df.iloc[idx][column])
            if value:
                return value
    return str(idx)


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


def prepare_route_map_data(
    df: pd.DataFrame,
    colour_column: str,
    *,
    placeholder: str = "Unknown",
) -> pd.DataFrame:
    """Return map-ready rows ensuring coordinates exist and colour labels are set.

    Parameters
    ----------
    df:
        The dataframe containing the historical job data.
    colour_column:
        Name of the column used to colour the map traces.
    placeholder:
        Value used when ``colour_column`` has missing entries to keep the legend stable.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` filtered to rows containing coordinates with extra
        ``map_colour_value`` and ``map_colour_display`` columns suitable for
        categorical colouring on the route map.
    """

    if colour_column not in df.columns:
        raise KeyError(f"'{colour_column}' column is required to colour the map")

    required_columns = ["origin_lat", "origin_lon", "dest_lat", "dest_lon"]
    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        missing_str = ", ".join(missing_required)
        raise KeyError(f"Dataframe is missing required coordinate columns: {missing_str}")

    filtered = df.dropna(subset=required_columns).copy()
    colour_series = filtered[colour_column].fillna(placeholder)
    filtered["map_colour_value"] = colour_series.astype(str)
    filtered["map_colour_display"] = filtered["map_colour_value"]
    return filtered


def _format_metric_value(value: float, format_spec: str) -> str:
    """Return a human-friendly string for ``value`` based on ``format_spec``."""

    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "n/a"

    if format_spec == "currency":
        return f"${value:,.2f}"
    if format_spec == "currency_per_m3":
        return f"${value:,.2f}/m³"
    if format_spec == "percentage":
        return f"{value * 100:.1f}%"
    if format_spec == "volume":
        return f"{value:,.1f} m³"
    if format_spec == "distance":
        return f"{value:,.1f} km"
    if format_spec == "hours":
        return f"{value:,.1f} hr"

    return f"{value:,.2f}"


def compute_cost_vs_price_percentage(df: pd.DataFrame) -> pd.Series:
    """Return cost as a share of price expressed as a percentage ratio.

    The output is a float series named ``cost_vs_price_pct`` where each value is
    ``final_cost_per_m3 / price_per_m3`` for the corresponding row. When either
    input column is missing, non-numeric, or zero the result contains ``NaN`` to
    avoid introducing infinities into downstream visualisations.
    """

    series_name = "cost_vs_price_pct"
    if df.empty:
        return pd.Series(dtype="float64", name=series_name)

    if "price_per_m3" not in df.columns or "final_cost_per_m3" not in df.columns:
        return pd.Series(
            np.nan,
            index=df.index,
            dtype="float64",
            name=series_name,
        )

    price_series = pd.to_numeric(df["price_per_m3"], errors="coerce")
    cost_series = pd.to_numeric(df["final_cost_per_m3"], errors="coerce")
    safe_denominator = price_series.replace({0: np.nan})
    ratio = cost_series.divide(safe_denominator)
    ratio = ratio.replace([math.inf, -math.inf], np.nan)
    ratio.name = series_name
    return ratio.astype("float64")


def prepare_metric_route_map_data(
    df: pd.DataFrame,
    metric_column: str,
    *,
    format_spec: str = "number",
) -> pd.DataFrame:
    """Return map rows with numeric metrics for continuous colouring.

    Parameters
    ----------
    df:
        Source dataframe containing the historical job data.
    metric_column:
        Name of the numeric column used to drive the colour scale.
    format_spec:
        Formatting hint used when presenting the metric values in hover labels.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` filtered to rows containing coordinates and numeric values
        in ``metric_column``. The dataframe includes ``map_colour_value`` as a
        ``float`` and ``map_colour_display`` for formatted hover labels.
    """

    if metric_column not in df.columns:
        raise KeyError(f"'{metric_column}' column is required to colour the map")

    required_columns = ["origin_lat", "origin_lon", "dest_lat", "dest_lon"]
    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        missing_str = ", ".join(missing_required)
        raise KeyError(f"Dataframe is missing required coordinate columns: {missing_str}")

    filtered = df.dropna(subset=required_columns).copy()
    numeric_series = pd.to_numeric(filtered[metric_column], errors="coerce")
    numeric_series = numeric_series.replace([math.inf, -math.inf], pd.NA)
    valid_mask = numeric_series.notna()
    filtered = filtered.loc[valid_mask].copy()
    if filtered.empty:
        return filtered

    numeric_values = numeric_series.loc[valid_mask].astype(float)
    filtered["map_colour_value"] = numeric_values
    filtered["map_colour_display"] = numeric_values.apply(
        lambda value: _format_metric_value(value, format_spec)
    )
    return filtered


def filter_jobs_by_distance(
    df: pd.DataFrame,
    *,
    metro_only: bool = False,
    threshold_km: float = METRO_DISTANCE_THRESHOLD_KM,
) -> pd.DataFrame:
    """Filter jobs by distance when metro-only mode is requested.

    When the canonical ``distance_km`` column is unavailable the function attempts to
    locate an alternative distance column and gracefully skips filtering if none are
    present instead of raising an exception. This keeps consumer UIs resilient when
    operating on partially populated datasets.
    """

    if not metro_only or df.empty:
        return df.copy()

    candidate_columns = ("distance_km", "distance", "km", "kms")
    distance_column = next((col for col in candidate_columns if col in df.columns), None)
    if distance_column is None:
        return df.copy()

    distances = pd.to_numeric(df[distance_column], errors="coerce")
    mask = distances <= threshold_km
    filtered = df.loc[mask].copy()

    if "distance_km" not in filtered.columns and distance_column != "distance_km":
        filtered["distance_km"] = distances.loc[filtered.index]

    return filtered


def available_heatmap_weightings(df: pd.DataFrame) -> dict[str, Optional[str]]:
    """Return the heatmap weighting options available for the dataframe."""

    options: dict[str, Optional[str]] = {}
    for label, column in HEATMAP_WEIGHTING_CANDIDATES:
        if column is None or column in df.columns:
            options[label] = column
    return options


def build_heatmap_source(
    df: pd.DataFrame,
    weight_column: Optional[str] = None,
    *,
    metro_only: bool = False,
    threshold_km: float = METRO_DISTANCE_THRESHOLD_KM,
) -> pd.DataFrame:
    """Build a point-based dataframe suitable for density heatmaps."""

    if df.empty:
        return pd.DataFrame(columns=["lat", "lon", "weight"])

    scoped = filter_jobs_by_distance(
        df,
        metro_only=metro_only,
        threshold_km=threshold_km,
    )
    if scoped.empty:
        return pd.DataFrame(columns=["lat", "lon", "weight"])

    if weight_column is None:
        weights = pd.Series(1.0, index=scoped.index, dtype=float)
    else:
        if weight_column not in scoped.columns:
            raise KeyError(
                f"'{weight_column}' column is required for heatmap weighting"
            )
        weights = pd.to_numeric(scoped[weight_column], errors="coerce")

    coordinate_pairs = [
        ("origin_lat", "origin_lon"),
        ("dest_lat", "dest_lon"),
    ]

    frames: list[pd.DataFrame] = []
    for lat_column, lon_column in coordinate_pairs:
        if lat_column not in scoped.columns or lon_column not in scoped.columns:
            continue
        coords = scoped[[lat_column, lon_column]].copy()
        coords = coords.rename(columns={lat_column: "lat", lon_column: "lon"})
        coords["weight"] = weights
        coords = coords.dropna(subset=["lat", "lon"])
        coords["weight"] = pd.to_numeric(coords["weight"], errors="coerce")
        coords = coords.dropna(subset=["weight"])
        if not coords.empty:
            frames.append(coords)

    if not frames:
        return pd.DataFrame(columns=["lat", "lon", "weight"])

    result = pd.concat(frames, ignore_index=True)
    result["lat"] = pd.to_numeric(result["lat"], errors="coerce")
    result["lon"] = pd.to_numeric(result["lon"], errors="coerce")
    result = result.dropna(subset=["lat", "lon", "weight"])
    result["weight"] = result["weight"].astype(float)
    return result.reset_index(drop=True)


def _clean_location(value: Any) -> str:
    """Return a normalised string representation for origin/destination labels."""

    if pd.isna(value):
        return "Unknown"
    text = str(value).strip()
    return text or "Unknown"


def format_bidirectional_corridor(origin: Any, destination: Any) -> str:
    """Return a canonical bidirectional corridor label.

    Parameters
    ----------
    origin, destination:
        Raw origin and destination labels which may include mixed casing or
        missing values. The function ensures labels are cleaned and sorted so
        ``Brisbane → Melbourne`` and ``Melbourne → Brisbane`` collapse into the
        shared label ``Brisbane ↔ Melbourne``.
    """

    cleaned_origin = _clean_location(origin)
    cleaned_destination = _clean_location(destination)
    if cleaned_origin == cleaned_destination:
        return cleaned_origin
    ordered = sorted([cleaned_origin, cleaned_destination], key=str.lower)
    return f"{ordered[0]} ↔ {ordered[1]}"


def _split_corridor_label(value: Any) -> tuple[str, str]:
    """Best-effort parsing for corridor labels lacking explicit endpoints."""

    if pd.isna(value):
        return "Unknown", "Unknown"
    text = str(value).strip()
    if not text:
        return "Unknown", "Unknown"
    for delimiter in ("↔", "→", "<->", "-", "—", " to ", "/", "|"):
        if delimiter in text:
            parts = [part.strip() for part in text.split(delimiter) if part.strip()]
            if len(parts) >= 2:
                return parts[0], parts[-1]
    return text, "Unknown"


def _resolve_corridor_pairs(df: pd.DataFrame) -> pd.Series:
    """Return a Series of bidirectional corridor labels for ``df`` rows."""

    origin_column = _first_present(df.columns, ORIGIN_COLUMNS)
    destination_column = _first_present(df.columns, DESTINATION_COLUMNS)

    if origin_column and destination_column:
        origins = df[origin_column]
        destinations = df[destination_column]
    else:
        corridor_column = None
        for candidate in ("corridor_display", *CORRIDOR_COLUMNS):
            if candidate in df.columns:
                corridor_column = candidate
                break
        if corridor_column:
            pairs = df[corridor_column].apply(_split_corridor_label)
            origins = pairs.str[0]
            destinations = pairs.str[1]
        else:
            origins = pd.Series(["Unknown"] * len(df), index=df.index)
            destinations = pd.Series(["Unknown"] * len(df), index=df.index)

    labels = [
        format_bidirectional_corridor(origin, destination)
        for origin, destination in zip(origins, destinations)
    ]
    return pd.Series(labels, index=df.index)


def aggregate_corridor_performance(
    df: pd.DataFrame,
    break_even: float,
    *,
    volume_column: Optional[str] = None,
    revenue_column: Optional[str] = None,
) -> pd.DataFrame:
    """Aggregate systemic performance metrics by bidirectional corridor.

    Parameters
    ----------
    df:
        Historical job records, typically produced by
        :func:`load_historical_jobs`.
    break_even:
        Break-even price per cubic metre used to classify loss-making lanes.
    volume_column, revenue_column:
        Optional overrides for the volume and revenue column names. When
        omitted, the function searches for known volume/revenue aliases.
    """

    columns = [
        "corridor_pair",
        "job_count",
        "share_of_jobs",
        "priced_job_count",
        "priced_job_ratio",
        "median_price_per_m3",
        "mean_price_per_m3",
        "price_per_m3_p25",
        "price_per_m3_p75",
        "weighted_price_per_m3",
        "below_break_even_ratio",
        "total_volume_m3",
        "share_of_volume",
        "total_revenue",
        "margin_per_m3_median",
        "margin_total_sum",
        "share_of_margin",
        "margin_total_pct_median",
        "revenue_per_km_median",
        "median_distance_km",
    ]

    if df.empty:
        return pd.DataFrame(columns=columns)

    working = df.copy()
    working["corridor_pair"] = _resolve_corridor_pairs(working)
    if {
        "lane_assignment_status",
        "origin_cluster_key",
        "destination_cluster_key",
    }.issubset(working.columns):
        assigned_mask = (
            working["lane_assignment_status"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .eq(LANE_STATUS_ASSIGNED)
        )
        origin_cluster = working["origin_cluster_key"].fillna("").astype(str).str.strip()
        destination_cluster = (
            working["destination_cluster_key"].fillna("").astype(str).str.strip()
        )
        canonical_pairs = pd.Series(
            [
                " <-> ".join(sorted((origin_value, destination_value)))
                if origin_value and destination_value
                else ""
                for origin_value, destination_value in zip(origin_cluster, destination_cluster)
            ],
            index=working.index,
        )
        valid_assigned = assigned_mask & canonical_pairs.ne("")
        working.loc[valid_assigned, "corridor_pair"] = canonical_pairs.loc[valid_assigned]

    if "price_per_m3" in working:
        working["_price_per_m3"] = pd.to_numeric(working["price_per_m3"], errors="coerce")
    else:
        working["_price_per_m3"] = np.nan

    if volume_column is None:
        volume_column = _first_present(working.columns, VOLUME_COLUMNS)
    if volume_column and volume_column in working:
        working["_volume_numeric"] = pd.to_numeric(working[volume_column], errors="coerce")
    else:
        working["_volume_numeric"] = np.nan

    if revenue_column is None:
        revenue_column = _first_present(working.columns, REVENUE_COLUMNS)
    if revenue_column and revenue_column in working:
        working["_revenue_numeric"] = pd.to_numeric(working[revenue_column], errors="coerce")
    else:
        working["_revenue_numeric"] = np.nan

    if "margin_per_m3" in working:
        working["_margin_per_m3_numeric"] = pd.to_numeric(
            working["margin_per_m3"], errors="coerce"
        )
    else:
        working["_margin_per_m3_numeric"] = np.nan

    if "margin_total" in working:
        working["_margin_total_numeric"] = pd.to_numeric(
            working["margin_total"], errors="coerce"
        )
    else:
        working["_margin_total_numeric"] = np.nan

    if "margin_total_pct" in working:
        working["_margin_total_pct_numeric"] = pd.to_numeric(
            working["margin_total_pct"], errors="coerce"
        )
    else:
        working["_margin_total_pct_numeric"] = np.nan

    if "revenue_per_km" in working:
        working["_revenue_per_km_numeric"] = pd.to_numeric(
            working["revenue_per_km"], errors="coerce"
        )
    else:
        working["_revenue_per_km_numeric"] = np.nan

    distance_column = "distance_km" if "distance_km" in working else _first_present(
        working.columns, DISTANCE_COLUMNS
    )
    if distance_column and distance_column in working:
        working["_distance_numeric"] = pd.to_numeric(
            working[distance_column], errors="coerce"
        )
    else:
        working["_distance_numeric"] = np.nan

    total_jobs = len(working)
    total_volume = working["_volume_numeric"].sum(min_count=1)
    total_volume = float(total_volume) if pd.notna(total_volume) else math.nan
    total_margin = working["_margin_total_numeric"].sum(min_count=1)
    total_margin = float(total_margin) if pd.notna(total_margin) else math.nan

    rows: list[dict[str, Any]] = []
    grouped = working.groupby("corridor_pair", dropna=False)
    for corridor, group in grouped:
        job_count = int(len(group))
        share_of_jobs = job_count / total_jobs if total_jobs else 0.0

        priced = group["_price_per_m3"].dropna()
        priced_job_count = int(len(priced))
        priced_job_ratio = priced_job_count / job_count if job_count else 0.0
        if priced_job_count:
            median_price = float(priced.median())
            mean_price = float(priced.mean())
            percentile_25 = float(priced.quantile(0.25))
            percentile_75 = float(priced.quantile(0.75))
            below_break_even_ratio = float((priced < break_even).sum() / priced_job_count)
        else:
            median_price = mean_price = percentile_25 = percentile_75 = math.nan
            below_break_even_ratio = 0.0

        volume_total = group["_volume_numeric"].sum(min_count=1)
        volume_total = float(volume_total) if pd.notna(volume_total) else math.nan
        revenue_total = group["_revenue_numeric"].sum(min_count=1)
        revenue_total = float(revenue_total) if pd.notna(revenue_total) else math.nan
        weighted_price = (
            revenue_total / volume_total
            if not math.isnan(revenue_total)
            and not math.isnan(volume_total)
            and volume_total != 0
            else math.nan
        )

        share_of_volume = (
            volume_total / total_volume
            if not math.isnan(volume_total)
            and not math.isnan(total_volume)
            and total_volume != 0
            else math.nan
        )

        margin_per_m3_series = group["_margin_per_m3_numeric"].dropna()
        margin_per_m3_median = (
            float(margin_per_m3_series.median())
            if not margin_per_m3_series.empty
            else math.nan
        )

        margin_total_sum = group["_margin_total_numeric"].sum(min_count=1)
        margin_total_sum = float(margin_total_sum) if pd.notna(margin_total_sum) else math.nan
        share_of_margin = (
            margin_total_sum / total_margin
            if not math.isnan(margin_total_sum)
            and not math.isnan(total_margin)
            and total_margin != 0
            else math.nan
        )

        margin_total_pct_series = group["_margin_total_pct_numeric"].dropna()
        margin_total_pct_median = (
            float(margin_total_pct_series.median())
            if not margin_total_pct_series.empty
            else math.nan
        )

        revenue_per_km_series = group["_revenue_per_km_numeric"].dropna()
        revenue_per_km_median = (
            float(revenue_per_km_series.median())
            if not revenue_per_km_series.empty
            else math.nan
        )

        distance_series = group["_distance_numeric"].dropna()
        median_distance = (
            float(distance_series.median()) if not distance_series.empty else math.nan
        )

        rows.append(
            {
                "corridor_pair": corridor,
                "job_count": job_count,
                "share_of_jobs": share_of_jobs,
                "priced_job_count": priced_job_count,
                "priced_job_ratio": priced_job_ratio,
                "median_price_per_m3": median_price,
                "mean_price_per_m3": mean_price,
                "price_per_m3_p25": percentile_25,
                "price_per_m3_p75": percentile_75,
                "weighted_price_per_m3": weighted_price,
                "below_break_even_ratio": below_break_even_ratio,
                "total_volume_m3": volume_total,
                "share_of_volume": share_of_volume,
                "total_revenue": revenue_total,
                "margin_per_m3_median": margin_per_m3_median,
                "margin_total_sum": margin_total_sum,
                "share_of_margin": share_of_margin,
                "margin_total_pct_median": margin_total_pct_median,
                "revenue_per_km_median": revenue_per_km_median,
                "median_distance_km": median_distance,
            }
        )

    result = pd.DataFrame(rows, columns=columns)
    if not result.empty:
        result = result.sort_values(
            by=["margin_total_sum", "total_revenue", "job_count"],
            ascending=[False, False, False],
        )
        result = result.reset_index(drop=True)
    return result


def _circle_coordinates(
    lat: float,
    lon: float,
    radius_km: float,
    *,
    points: int = 60,
) -> tuple[list[float], list[float]]:
    """Return an approximate circle around ``lat``/``lon`` with radius ``radius_km``.

    The approximation assumes a spherical Earth and adjusts longitudinal degrees
    for the latitude.  It is sufficient for visualisation purposes without
    adding heavier geographic dependencies.
    """

    if radius_km <= 0 or not math.isfinite(radius_km):
        return [], []

    lat_rad = math.radians(lat)
    cos_lat = math.cos(lat_rad)
    if abs(cos_lat) < 1e-6:
        cos_lat = 1e-6 if cos_lat >= 0 else -1e-6

    lat_deg_per_km = 1.0 / 110.574
    lon_deg_per_km = 1.0 / (111.320 * cos_lat)

    angles = np.linspace(0.0, 2.0 * math.pi, points, endpoint=False)
    lat_offsets = radius_km * np.sin(angles)
    lon_offsets = radius_km * np.cos(angles)

    latitudes = (lat + lat_offsets * lat_deg_per_km).tolist()
    longitudes = (lon + lon_offsets * lon_deg_per_km).tolist()

    if latitudes and longitudes:
        latitudes.append(latitudes[0])
        longitudes.append(longitudes[0])

    return latitudes, longitudes



def build_isochrone_polygons(
    df: pd.DataFrame,
    *,
    centre: Literal["origin", "destination"] = "origin",
    horizon_hours: float = 4.0,
    default_speed_kmh: float = 70.0,
    max_routes: int = 50,
    points: int = 60,
    routing_provider: Optional[RoutingProvider] = None,
    ors_client: Optional[ORSClient] = None,
    ors_profile: str = "driving-hgv",
) -> pd.DataFrame:
    """Return approximate isochrone polygons for each route in ``df``.

    Parameters
    ----------
    df:
        DataFrame containing at least coordinate and distance information for
        each route.  ``origin_lat``/``origin_lon`` or ``dest_lat``/``dest_lon``
        columns are required depending on the selected ``centre``.  Distances are
        sourced from ``distance_km``/``distance`` columns while travel durations
        are taken from ``duration_hr``/``duration`` columns when available.
    centre:
        Which endpoint of the route anchors the isochrone.  ``"origin"`` draws
        circles around origin coordinates, whereas ``"destination"`` uses the
        destination coordinates.
    horizon_hours:
        Travel time horizon used to scale the isochrone radius.  The function
        multiplies the inferred average speed (distance / duration) by this
        value.  When no duration is available the ``default_speed_kmh`` is used
        instead.
    default_speed_kmh:
        Fallback speed applied when a route does not provide usable duration
        information.
    max_routes:
        Maximum number of routes to include.  This prevents map visualisations
        from being overwhelmed by hundreds of polygons.
    points:
        Number of vertices used to approximate each circular polygon.
    routing_provider:
        Optional routing provider implementing :class:`RoutingProvider`.
        When omitted the helper attempts to resolve one via
        :func:`analytics.routing_provider.get_routing_provider` using the
        supplied ``ors_client`` (if any) and the ``ROUTING_PROVIDER``
        environment variable.
    ors_client:
        Backwards-compatible argument for supplying an OpenRouteService client
        when the default provider requires one.
    ors_profile:
        Routing profile supplied to the OpenRouteService isochrone request.  The
        default of ``"driving-hgv"`` aligns with heavy vehicle routing.

    Returns
    -------
    pandas.DataFrame
        Dataframe with ``label``, ``centre_lat``, ``centre_lon``,
        ``radius_km``, ``speed_kmh``, ``latitudes``, ``longitudes`` and
        ``tooltip`` columns.  Each row describes one isochrone polygon, using
        network-aware shapes when OpenRouteService is available and circular
        fallbacks otherwise.
    """

    if df.empty:
        return pd.DataFrame(
            columns=
            [
                "label",
                "centre_lat",
                "centre_lon",
                "radius_km",
                "speed_kmh",
                "latitudes",
                "longitudes",
                "tooltip",
            ]
        )

    centre_key = centre.lower()
    if centre_key not in {"origin", "destination"}:
        raise ValueError("centre must be 'origin' or 'destination'")

    lat_column = "origin_lat" if centre_key == "origin" else "dest_lat"
    lon_column = "origin_lon" if centre_key == "origin" else "dest_lon"
    if lat_column not in df.columns or lon_column not in df.columns:
        return pd.DataFrame(
            columns=
            [
                "label",
                "centre_lat",
                "centre_lon",
                "radius_km",
                "speed_kmh",
                "latitudes",
                "longitudes",
                "tooltip",
            ]
        )

    distance_column = next(
        (column for column in ("distance_km", "distance", "km", "kms") if column in df.columns),
        None,
    )
    if distance_column is None:
        return pd.DataFrame(
            columns=
            [
                "label",
                "centre_lat",
                "centre_lon",
                "radius_km",
                "speed_kmh",
                "latitudes",
                "longitudes",
                "tooltip",
            ]
        )

    duration_column = next(
        (
            column
            for column in ("duration_hr", "duration_hours", "travel_hours", "duration")
            if column in df.columns
        ),
        None,
    )

    provider: Optional[RoutingProvider]
    if routing_provider is not None:
        provider = routing_provider
    else:
        try:
            provider = get_routing_provider(client=ors_client)
        except Exception as exc:  # pragma: no cover - exercised via fallback path
            logger.debug("Unable to initialise routing provider for isochrones: %s", exc)
            provider = None

    range_seconds: list[int] = []
    if horizon_hours > 0 and math.isfinite(horizon_hours):
        seconds = int(round(horizon_hours * 3600.0))
        if seconds > 0:
            range_seconds = [seconds]

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        lat_value = _coerce_float(row.get(lat_column))
        lon_value = _coerce_float(row.get(lon_column))
        if lat_value is None or lon_value is None:
            continue

        distance_value = _coerce_float(row.get(distance_column))
        if distance_value is None or distance_value <= 0:
            continue

        if duration_column is not None:
            duration_value = _coerce_float(row.get(duration_column))
        else:
            duration_value = None

        if duration_value is not None and duration_value > 0:
            speed_kmh = distance_value / duration_value
        else:
            speed_kmh = default_speed_kmh

        if not math.isfinite(speed_kmh) or speed_kmh <= 0:
            speed_kmh = default_speed_kmh

        radius_km = speed_kmh * horizon_hours
        if radius_km <= 0 or not math.isfinite(radius_km):
            continue

        label = _route_display_name(row)

        latitudes: list[float]
        longitudes: list[float]
        if provider is not None and range_seconds:
            try:
                result = provider.isochrone(
                    centre=(float(lon_value), float(lat_value)),
                    profile=ors_profile,
                    range_seconds=range_seconds,
                )
            except NotImplementedError:
                result = None
            except Exception as exc:  # pragma: no cover - network failures are environment-dependent
                logger.debug("Routing provider isochrone request failed for %s: %s", label, exc)
                result = None
            if result:
                latitudes, longitudes = result.to_lat_lon_lists()
            else:
                latitudes, longitudes = [], []
            if not latitudes or not longitudes:
                latitudes, longitudes = _circle_coordinates(
                    lat_value,
                    lon_value,
                    radius_km,
                    points=points,
                )
        else:
            latitudes, longitudes = _circle_coordinates(
                lat_value,
                lon_value,
                radius_km,
                points=points,
            )
        if not latitudes or not longitudes:
            continue

        tooltip = (
            f"{label} — {horizon_hours:.1f} hr reach ≈ {radius_km:.0f} km "
            f"(avg {speed_kmh:.0f} km/h)"
        )

        records.append(
            {
                "label": label,
                "centre_lat": lat_value,
                "centre_lon": lon_value,
                "radius_km": radius_km,
                "speed_kmh": speed_kmh,
                "latitudes": latitudes,
                "longitudes": longitudes,
                "tooltip": tooltip,
            }
        )

        if len(records) >= max_routes:
            break

    return pd.DataFrame.from_records(records)


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
