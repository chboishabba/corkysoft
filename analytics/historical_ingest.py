"""Historical pricing ingest and column inference helpers."""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from .db import ensure_dashboard_tables, ensure_global_parameters_table
from .lane_assignment import backfill_lane_assignments, ensure_lane_assignment_schema
from .routes_map import populate_route_geometry

try:  # pragma: no cover - availability exercised via tests
    from corkysoft.routing import geocode_cached as _geocode_cached
except Exception:  # pragma: no cover - optional dependency
    geocode_cached = None  # type: ignore[assignment]
else:
    geocode_cached = _geocode_cached


logger = logging.getLogger(__name__)

_QUOTE_COUNTRY_DEFAULT = "Australia"
HISTORICAL_INGEST_USABLE = "usable"
HISTORICAL_INGEST_USABLE_WITH_GAPS = "usable_with_gaps"
HISTORICAL_INGEST_NOT_READY = "not_ready"

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

_DEDUP_FLOAT_PRECISION = 6


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


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

    sample = series.dropna().astype(str).str.strip().replace({"": np.nan}).dropna()
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

    from . import price_distribution as price_distribution_module

    geocode = getattr(price_distribution_module, "geocode_cached", geocode_cached)
    if df.empty or geocode is None or conn is None:  # pragma: no cover - dependency/guard
        return df

    map_columns = ("origin_lon", "origin_lat", "dest_lon", "dest_lat")
    columns = set(df.columns)
    needs_enrichment = any(column not in columns for column in map_columns)
    if not needs_enrichment:
        needs_enrichment = any(df[column].isna().all() for column in map_columns if column in df.columns)
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
                        result = geocode(conn, origin_place, origin_country)
                    except Exception as exc:  # pragma: no cover
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
                destination_country = _first_non_empty_value(row, ("destination_country",)) or default_country
                cache_key = (destination_place, destination_country)
                coords = destination_cache.get(cache_key)
                if cache_key not in destination_cache:
                    try:
                        result = geocode(conn, destination_place, destination_country)
                    except Exception as exc:  # pragma: no cover
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
    """Insert ``df`` rows into ``historical_jobs`` and return ``(inserted, skipped)``."""

    from . import price_distribution as price_distribution_module

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
        raise ValueError("Uploaded data must include origin and destination columns.")
    if mapping.price is None and (mapping.revenue is None or mapping.volume is None):
        raise ValueError(
            "Uploaded data must include a price-per-m3 column or both revenue and volume columns."
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
            row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8]
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
        price_value = _safe_float(df.iloc[idx][mapping.price]) if mapping.price else None
        revenue_value = _safe_float(df.iloc[idx][mapping.revenue]) if mapping.revenue else None
        volume_value = _safe_float(df.iloc[idx][mapping.volume]) if mapping.volume else None
        distance_value = _safe_float(df.iloc[idx][mapping.distance]) if mapping.distance else None
        final_cost_value = _safe_float(df.iloc[idx][mapping.final_cost]) if mapping.final_cost else None

        if price_value is None and revenue_value is not None and volume_value:
            price_value = None if volume_value == 0 else revenue_value / volume_value

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

        if mapping.corridor and mapping.corridor in df.columns:
            corridor_value = _clean_string(df.iloc[idx][mapping.corridor])
        else:
            corridor_value = f"{origin_value} → {destination_value}"

        origin_postcode = _clean_string(df.iloc[idx][origin_pc_col]) if origin_pc_col else None
        dest_postcode = _clean_string(df.iloc[idx][dest_pc_col]) if dest_pc_col else None

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
        populate_geometry = getattr(
            price_distribution_module,
            "populate_route_geometry",
            populate_route_geometry,
        )
        try:
            populate_geometry(conn, inserted_ids, dataset="historical")
        except Exception:
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
    payload["issues"] = [
        dict(zip(issue_columns, issue_row, strict=False))
        for issue_row in issue_cursor.fetchall()
    ]
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
