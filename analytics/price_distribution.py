"""Helpers for the price-distribution Streamlit view."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Iterable, Optional, Sequence

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


PROFITABILITY_BANDS: Sequence[tuple[float, float, str]] = (
    (-float("inf"), 0.0, "Below break-even"),
    (0.0, 50.0, "0-50 above break-even"),
    (50.0, 100.0, "50-100 above break-even"),
    (100.0, float("inf"), "100+ above break-even"),
)

DEFAULT_BREAK_EVEN_ABS_TOLERANCE = 5.0
DEFAULT_BREAK_EVEN_REL_TOLERANCE = 0.02

PROFITABILITY_COLOURS = {
    "Below break-even": [217, 83, 79],
    "0-50 above break-even": [240, 173, 78],
    "50-100 above break-even": [91, 192, 222],
    "100+ above break-even": [92, 184, 92],
    "Unknown": [128, 128, 128],
}

PROFITABILITY_WIDTHS = {
    "Below break-even": 200,
    "0-50 above break-even": 120,
    "50-100 above break-even": 100,
    "100+ above break-even": 80,
    "Unknown": 80,
}


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
    """Return ``df`` with duplicate column labels removed.

    When importing legacy exports the ``historical_jobs`` table may already
    contain denormalised location columns (``origin``, ``destination`` etc.).
    The default dashboard query joins the addresses table and aliases the
    enriched columns using the same names which results in duplicate column
    labels. Pandas returns a :class:`~pandas.DataFrame` rather than a
    :class:`~pandas.Series` when selecting a column with duplicate labels which
    later breaks string/arithmetical operations with ``Columns must be same
    length as key`` errors.  Keeping the last occurrence (the enriched join
    columns) matches the behaviour of the old dashboard and avoids the
    ambiguity.
    """

    if df.columns.duplicated().any():
        return df.loc[:, ~df.columns.duplicated(keep="last")]
    return df


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


def import_historical_jobs_from_dataframe(
    conn,
    df: pd.DataFrame,
) -> tuple[int, int]:
    """Insert ``df`` rows into ``historical_jobs`` and return ``(inserted, skipped)``.

    The importer uses :func:`infer_columns` to discover relevant fields and performs
    light validation before inserting rows. Rows missing a job date, origin,
    destination or price signal are skipped. Duplicate rows, identified via the
    combination of ``(job_date, origin, destination, client)``, are ignored.
    """

    ensure_global_parameters_table(conn)
    ensure_dashboard_tables(conn)

    if df.empty:
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
        "SELECT job_date, origin, destination, client FROM historical_jobs"
    ).fetchall()
    existing_keys = {
        (
            row[0],
            row[1] or None,
            row[2] or None,
            (row[3] or "").strip(),
        )
        for row in existing_rows
    }

    now = datetime.now(UTC).isoformat()
    to_insert: list[tuple[Any, ...]] = []
    skipped = 0

    for idx in range(len(df)):
        job_date = dates.iloc[idx]
        if pd.isna(job_date):
            skipped += 1
            continue

        origin_value = _clean_string(df.iloc[idx][mapping.origin])
        destination_value = _clean_string(df.iloc[idx][mapping.destination])
        if not origin_value or not destination_value:
            skipped += 1
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
            skipped += 1
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

        key = (
            job_date.date().isoformat(),
            origin_value,
            destination_value,
            (client_value or ""),
        )
        if key in existing_keys:
            skipped += 1
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

    if to_insert:
        conn.executemany(
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
            to_insert,
        )
        conn.commit()

    return len(to_insert), skipped


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


def _historical_jobs_query() -> str:
    """Return the default query joining address metadata for historical jobs."""

    return """
        SELECT
            hj.*,
            COALESCE(o.city, o.normalized, o.raw_input) AS origin,
            COALESCE(d.city, d.normalized, d.raw_input) AS destination,
            o.raw_input AS origin_raw,
            o.normalized AS origin_normalized,
            o.city AS origin_city,
            o.state AS origin_state,
            o.postcode AS origin_postcode,
            o.country AS origin_country,
            o.lon AS origin_lon,
            o.lat AS origin_lat,
            d.raw_input AS destination_raw,
            d.normalized AS destination_normalized,
            d.city AS destination_city,
            d.state AS destination_state,
            d.postcode AS destination_postcode,
            d.country AS destination_country,
            d.lon AS dest_lon,
            d.lat AS dest_lat
        FROM historical_jobs AS hj
        LEFT JOIN addresses AS o ON hj.origin_address_id = o.id
        LEFT JOIN addresses AS d ON hj.destination_address_id = d.id
    """


def filter_routes_by_country(
    routes: pd.DataFrame, country: Optional[str]
) -> pd.DataFrame:
    """Return ``routes`` limited to rows that match ``country`` when metadata exists."""

    if routes.empty:
        return routes.copy()

    if country is None:
        return routes.copy()

    normalized = str(country).strip().lower()
    if not normalized:
        return routes.copy()

    candidate_columns = [
        column for column in ("origin_country", "destination_country") if column in routes.columns
    ]
    if not candidate_columns:
        return routes.copy()

    mask = pd.Series(False, index=routes.index)
    for column in candidate_columns:
        values = (
            routes[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        mask = mask | (values == normalized)

    return routes.loc[mask].copy()


def _prepare_loaded_jobs(
    df: pd.DataFrame,
    mapping: ColumnMapping,
    base_costs: BaseCostConfig,
    *,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    clients: Optional[Sequence[str]] = None,
    corridor: Optional[str] = None,
    postcode_prefix: Optional[str] = None,
) -> tuple[pd.DataFrame, ColumnMapping]:
    """Return ``df`` filtered and enriched for downstream visualisations."""

    if df.empty:
        return df, mapping

    working = df.copy()

    if start_date is not None and not isinstance(start_date, pd.Timestamp):
        start_date = pd.Timestamp(start_date)
    if end_date is not None and not isinstance(end_date, pd.Timestamp):
        end_date = pd.Timestamp(end_date)

    if mapping.date and mapping.date in working.columns:
        parse_kwargs = _infer_datetime_parse_kwargs(working[mapping.date])
        working[mapping.date] = pd.to_datetime(
            working[mapping.date], errors="coerce", **parse_kwargs
        )
        parsed_dates = working[mapping.date]
        if start_date is not None:
            working = working[parsed_dates >= start_date]
            parsed_dates = working[mapping.date]
        if end_date is not None:
            working = working[parsed_dates <= end_date]

    if mapping.client and clients:
        working = working[working[mapping.client].isin(clients)]

    if postcode_prefix:
        prefix = str(postcode_prefix).strip()
        if prefix:
            prefix_lower = prefix.lower()
            known_postcodes = {pc.lower() for pc in POSTCODE_COLUMNS}
            postcode_columns = [
                column for column in working.columns if column.lower() in known_postcodes
            ]
            text_columns = list(dict.fromkeys(postcode_columns))
            if mapping.corridor and mapping.corridor in working.columns:
                text_columns.append(mapping.corridor)
            if mapping.origin and mapping.origin in working.columns:
                text_columns.append(mapping.origin)
            if mapping.destination and mapping.destination in working.columns:
                text_columns.append(mapping.destination)
            if text_columns:
                mask = pd.Series(False, index=working.index)
                for col in text_columns:
                    mask = mask | working[col].astype(str).str.lower().str.contains(
                        prefix_lower, na=False
                    )
                working = working[mask]

    revenue_series: Optional[pd.Series] = None
    volume_series: Optional[pd.Series] = None
    distance_series: Optional[pd.Series] = None
    final_cost_series: Optional[pd.Series] = None

    if mapping.revenue and mapping.revenue in working.columns:
        revenue_series = pd.to_numeric(working[mapping.revenue], errors="coerce")
        working[mapping.revenue] = revenue_series
    if mapping.volume and mapping.volume in working.columns:
        volume_series = pd.to_numeric(working[mapping.volume], errors="coerce")
        working[mapping.volume] = volume_series
    if mapping.distance and mapping.distance in working.columns:
        distance_series = pd.to_numeric(working[mapping.distance], errors="coerce")
        working[mapping.distance] = distance_series
        working["distance_km"] = distance_series
    if mapping.final_cost and mapping.final_cost in working.columns:
        final_cost_series = pd.to_numeric(working[mapping.final_cost], errors="coerce")
        working[mapping.final_cost] = final_cost_series

    if mapping.price and mapping.price in working.columns:
        working["price_per_m3"] = pd.to_numeric(working[mapping.price], errors="coerce")
    else:
        if revenue_series is None or volume_series is None:
            raise RuntimeError(
                "Jobs must contain a per-m³ price column or both revenue and volume columns"
            )
        working["price_per_m3"] = revenue_series / volume_series.replace({0: np.nan})

    if revenue_series is not None and distance_series is not None:
        working["revenue_per_km"] = revenue_series / distance_series.replace({0: np.nan})

    if final_cost_series is not None:
        working["final_cost_total"] = final_cost_series
        safe_cost = final_cost_series.replace({0: np.nan})
        if revenue_series is not None:
            margin_total = revenue_series - final_cost_series
            working["margin_total"] = margin_total
            working["margin_total_pct"] = margin_total / safe_cost
        if volume_series is not None:
            safe_volume = volume_series.replace({0: np.nan})
            cost_per_m3 = final_cost_series / safe_volume
            working["final_cost_per_m3"] = cost_per_m3
            margin_per_m3 = working["price_per_m3"] - cost_per_m3
            working["margin_per_m3"] = margin_per_m3
            safe_cost_per_m3 = cost_per_m3.replace({0: np.nan})
            working["margin_per_m3_pct"] = margin_per_m3 / safe_cost_per_m3

    if mapping.corridor and mapping.corridor in working.columns:
        working["corridor_display"] = working[mapping.corridor]
    else:
        origin = working[mapping.origin] if mapping.origin else None
        destination = working[mapping.destination] if mapping.destination else None
        if origin is not None and destination is not None:
            working["corridor_display"] = origin.fillna("?") + " → " + destination.fillna("?")
        else:
            working["corridor_display"] = "Unknown"

    if corridor:
        if mapping.corridor and mapping.corridor in working.columns:
            working = working[working[mapping.corridor] == corridor]
        else:
            working = working[working["corridor_display"] == corridor]

    if mapping.client and mapping.client in working.columns:
        working["client_display"] = working[mapping.client]
    else:
        working["client_display"] = "Unknown"

    if mapping.date and mapping.date in working.columns:
        working["job_date"] = working[mapping.date]

    numeric_cols = [
        c
        for c in [mapping.revenue, mapping.volume, mapping.distance, mapping.final_cost]
        if c
    ]
    for col in numeric_cols:
        working[col] = pd.to_numeric(working[col], errors="coerce")

    if volume_series is not None and distance_series is not None:
        break_even_total, break_even_per_m3 = compute_break_even_series(
            distance_series,
            volume_series,
            base_costs,
        )
        working["break_even_total"] = break_even_total
        working["break_even_per_m3"] = break_even_per_m3
        if "price_per_m3" in working.columns:
            working["margin_vs_break_even"] = working["price_per_m3"] - break_even_per_m3

    return working.reset_index(drop=True), mapping


def load_historical_jobs(
    conn,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    clients: Optional[Sequence[str]] = None,
    corridor: Optional[str] = None,
    postcode_prefix: Optional[str] = None,
) -> tuple[pd.DataFrame, ColumnMapping]:
    """Load historical job data applying the requested filters."""
    ensure_global_parameters_table(conn)
    migrate_geojson_to_routes(conn)
    base_costs = ensure_base_cost_parameters(conn)

    query = _historical_jobs_query()
    try:
        df = pd.read_sql_query(query, conn)
    except Exception:  # pragma: no cover - fall back for legacy schemas
        try:
            df = pd.read_sql_query("SELECT * FROM historical_jobs", conn)
        except Exception as exc:  # pragma: no cover - surfaces friendly error in UI
            raise RuntimeError("historical_jobs table is required for this view") from exc

    df = _deduplicate_columns(df)
    mapping = infer_columns(df)
    return _prepare_loaded_jobs(
        df,
        mapping,
        base_costs,
        start_date=start_date,
        end_date=end_date,
        clients=clients,
        corridor=corridor,
        postcode_prefix=postcode_prefix,
    )


def load_quotes(
    conn,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    clients: Optional[Sequence[str]] = None,
    corridor: Optional[str] = None,
    postcode_prefix: Optional[str] = None,
) -> tuple[pd.DataFrame, ColumnMapping]:
    """Load saved quick quote data from the ``quotes`` table."""

    ensure_global_parameters_table(conn)
    base_costs = ensure_base_cost_parameters(conn)

    try:
        df = pd.read_sql_query(
            """
            SELECT
                id,
                created_at,
                quote_date,
                origin_input,
                destination_input,
                origin_resolved,
                destination_resolved,
                origin_lon,
                origin_lat,
                dest_lon,
                dest_lat,
                distance_km,
                duration_hr,
                cubic_m,
                pricing_model,
                base_subtotal,
                base_components,
                modifiers_applied,
                modifiers_total,
                seasonal_multiplier,
                seasonal_label,
                total_before_margin,
                margin_percent,
                manual_quote,
                final_quote,
                summary
            FROM quotes
            ORDER BY quote_date DESC, created_at DESC
            """,
            conn,
        )
    except Exception as exc:  # pragma: no cover - surfaces friendly error in UI
        raise RuntimeError("quotes table is required for this view") from exc

    if df.empty:
        mapping = infer_columns(df)
        return df, mapping

    df = _deduplicate_columns(df)

    if "quote_date" in df.columns:
        df["job_date"] = df["quote_date"]

    _coalesce_string_columns(df, "origin_resolved", "origin_input", "origin")
    _coalesce_string_columns(
        df, "destination_resolved", "destination_input", "destination"
    )

    quote_total = df["manual_quote"].where(df["manual_quote"].notna(), df["final_quote"])
    quote_total = pd.to_numeric(quote_total, errors="coerce")
    df["quote_total"] = quote_total
    df["revenue_total"] = quote_total
    df["revenue"] = quote_total

    if "cubic_m" in df.columns:
        volume_series = pd.to_numeric(df["cubic_m"], errors="coerce")
    else:  # pragma: no cover - quotes schema always includes cubic_m
        volume_series = pd.Series(np.nan, index=df.index, dtype=float)
    df["volume_m3"] = volume_series
    df["volume"] = volume_series

    if "distance_km" in df.columns:
        distance_series = pd.to_numeric(df["distance_km"], errors="coerce")
        df["distance_km"] = distance_series

    if "total_before_margin" in df.columns:
        final_cost_series = pd.to_numeric(df["total_before_margin"], errors="coerce")
        df["final_cost"] = final_cost_series

    safe_volume = volume_series.replace({0: np.nan})
    df["price_per_m3"] = quote_total / safe_volume

    df["client"] = "Quote builder"

    mapping = infer_columns(df)
    return _prepare_loaded_jobs(
        df,
        mapping,
        base_costs,
        start_date=start_date,
        end_date=end_date,
        clients=clients,
        corridor=corridor,
        postcode_prefix=postcode_prefix,
    )


def load_live_jobs(
    conn,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    clients: Optional[Sequence[str]] = None,
    corridor: Optional[str] = None,
    postcode_prefix: Optional[str] = None,
) -> tuple[pd.DataFrame, ColumnMapping]:
    """Load live job data from the ``jobs`` table for real-time monitoring."""

    ensure_global_parameters_table(conn)
    base_costs = ensure_base_cost_parameters(conn)

    try:
        df = pd.read_sql_query("SELECT * FROM jobs", conn)
    except Exception as exc:
        raise RuntimeError("jobs table is required for live monitoring") from exc

    df = _deduplicate_columns(df)

    mapping = infer_columns(df)
    return _prepare_loaded_jobs(
        df,
        mapping,
        base_costs,
        start_date=start_date,
        end_date=end_date,
        clients=clients,
        corridor=corridor,
        postcode_prefix=postcode_prefix,
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
        Copy of ``df`` filtered to rows containing coordinates with an extra
        ``map_colour_value`` column suitable for categorical colouring.
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


@dataclass
class DistributionSummary:
    job_count: int
    priced_job_count: int
    median: Optional[float]
    percentile_25: Optional[float]
    percentile_75: Optional[float]
    below_break_even_count: int
    below_break_even_ratio: float
    mean: Optional[float]
    std_dev: Optional[float]
    kurtosis: Optional[float]
    skewness: Optional[float]


def summarise_distribution(df: pd.DataFrame, break_even: float) -> DistributionSummary:
    """Return KPI statistics for the filtered dataset."""

    if "price_per_m3" not in df.columns:
        raise KeyError("'price_per_m3' column is required for distribution summaries")

    price_series = pd.to_numeric(df["price_per_m3"], errors="coerce")
    priced = price_series.dropna()
    job_count = len(df)
    priced_job_count = len(priced)
    if priced_job_count:
        median = float(priced.median())
        percentile_25 = float(priced.quantile(0.25))
        percentile_75 = float(priced.quantile(0.75))
        if "break_even_per_m3" in df.columns:
            break_even_series = pd.to_numeric(
                df.loc[priced.index, "break_even_per_m3"], errors="coerce"
            ).fillna(break_even)
            comparison_target = break_even_series
        else:
            comparison_target = pd.Series(break_even, index=priced.index)
        below_break_even_count = int((priced < comparison_target).sum())
        below_break_even_ratio = below_break_even_count / priced_job_count
        mean = float(priced.mean())
        std_dev = float(priced.std(ddof=1)) if priced_job_count > 1 else math.nan
        kurtosis = float(priced.kurtosis()) if priced_job_count > 3 else math.nan
        skewness = float(priced.skew()) if priced_job_count > 2 else math.nan
    else:
        median = percentile_25 = percentile_75 = math.nan
        below_break_even_count = 0
        below_break_even_ratio = 0.0
        mean = std_dev = kurtosis = skewness = math.nan

    return DistributionSummary(
        job_count=job_count,
        priced_job_count=priced_job_count,
        median=median,
        percentile_25=percentile_25,
        percentile_75=percentile_75,
        below_break_even_count=below_break_even_count,
        below_break_even_ratio=below_break_even_ratio,
        mean=mean,
        std_dev=std_dev,
        kurtosis=kurtosis,
        skewness=skewness,
    )


@dataclass
class ProfitabilitySummary:
    revenue_per_km_median: Optional[float]
    revenue_per_km_mean: Optional[float]
    margin_per_m3_median: Optional[float]
    margin_per_m3_pct_median: Optional[float]
    margin_total_median: Optional[float]
    margin_total_pct_median: Optional[float]


def summarise_profitability(df: pd.DataFrame) -> ProfitabilitySummary:
    """Calculate profitability KPIs used by the optional views."""

    def _median(series: pd.Series) -> Optional[float]:
        series = series.dropna()
        if series.empty:
            return math.nan
        return float(series.median())

    def _mean(series: pd.Series) -> Optional[float]:
        series = series.dropna()
        if series.empty:
            return math.nan
        return float(series.mean())

    revenue_per_km_median = revenue_per_km_mean = math.nan
    if "revenue_per_km" in df:
        revenue_per_km_series = pd.to_numeric(df["revenue_per_km"], errors="coerce")
        revenue_per_km_median = _median(revenue_per_km_series)
        revenue_per_km_mean = _mean(revenue_per_km_series)

    margin_per_m3_median = margin_per_m3_pct_median = math.nan
    if "margin_per_m3" in df:
        margin_per_m3_series = pd.to_numeric(df["margin_per_m3"], errors="coerce")
        margin_per_m3_median = _median(margin_per_m3_series)
    if "margin_per_m3_pct" in df:
        margin_per_m3_pct_series = pd.to_numeric(df["margin_per_m3_pct"], errors="coerce")
        margin_per_m3_pct_median = _median(margin_per_m3_pct_series)

    margin_total_median = margin_total_pct_median = math.nan
    if "margin_total" in df:
        margin_total_series = pd.to_numeric(df["margin_total"], errors="coerce")
        margin_total_median = _median(margin_total_series)
    if "margin_total_pct" in df:
        margin_total_pct_series = pd.to_numeric(df["margin_total_pct"], errors="coerce")
        margin_total_pct_median = _median(margin_total_pct_series)

    return ProfitabilitySummary(
        revenue_per_km_median=revenue_per_km_median,
        revenue_per_km_mean=revenue_per_km_mean,
        margin_per_m3_median=margin_per_m3_median,
        margin_per_m3_pct_median=margin_per_m3_pct_median,
        margin_total_median=margin_total_median,
        margin_total_pct_median=margin_total_pct_median,
    )


def _format_ratio(ratio: float) -> str:
    if ratio is None or math.isnan(ratio):
        return "n/a"
    return f"{ratio:.1%}"


def _safe_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return math.nan
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    if math.isnan(numeric):
        return math.nan
    return numeric


def _format_corridor_notes(row: pd.Series, break_even: float) -> str:
    details: list[str] = []

    avg_margin_m3 = _safe_float(row.get("avg_margin_per_m3"))
    if not math.isnan(avg_margin_m3):
        details.append(f"Avg margin per m³ ${avg_margin_m3:.2f}")

    avg_margin_pct = _safe_float(row.get("avg_margin_pct"))
    if not math.isnan(avg_margin_pct):
        details.append(f"Avg margin % {_format_ratio(avg_margin_pct)}")

    avg_price_m3 = _safe_float(row.get("avg_price_per_m3"))
    if not math.isnan(avg_price_m3):
        delta = avg_price_m3 - break_even
        details.append(f"Δ vs break-even ${delta:.2f}")

    total_margin = _safe_float(row.get("total_margin"))
    if not math.isnan(total_margin):
        details.append(f"Total margin ${total_margin:.2f}")

    total_volume = _safe_float(row.get("total_volume"))
    if not math.isnan(total_volume):
        details.append(f"Volume {total_volume:.1f} m³")

    job_count = row.get("job_count")
    if pd.notna(job_count):
        details.append(f"Jobs {int(job_count)}")

    return " | ".join(details)


def build_profitability_export(
    df: pd.DataFrame,
    break_even: float,
    *,
    top_n_corridors: int = 3,
) -> pd.DataFrame:
    """Return a tabular export of profitability and optimisation highlights.

    The export flattens the key metrics used throughout the dashboard into a
    lightweight dataframe so that Streamlit and CLI callers can expose a
    ready-to-download CSV summarising the filtered dataset.
    """

    distribution_summary = summarise_distribution(df, break_even)
    profitability_summary = summarise_profitability(df)

    rows: list[dict[str, object]] = []

    def append_row(section: str, metric: str, value: object, unit: str = "", notes: str = "") -> None:
        rows.append(
            {
                "section": section,
                "metric": metric,
                "value": value,
                "unit": unit,
                "notes": notes,
            }
        )

    append_row("Assumptions", "Break-even assumption", float(break_even), "$/m³")

    append_row("Distribution", "Jobs analysed", distribution_summary.job_count, "jobs")
    append_row("Distribution", "Jobs with price", distribution_summary.priced_job_count, "jobs")

    if not math.isnan(distribution_summary.median):
        append_row("Distribution", "Median price per m³", distribution_summary.median, "$/m³")
        append_row(
            "Distribution",
            "25th percentile price per m³",
            distribution_summary.percentile_25,
            "$/m³",
        )
        append_row(
            "Distribution",
            "75th percentile price per m³",
            distribution_summary.percentile_75,
            "$/m³",
        )
        append_row("Distribution", "Mean price per m³", distribution_summary.mean, "$/m³")
        append_row("Distribution", "Std deviation", distribution_summary.std_dev, "$/m³")
        append_row("Distribution", "Kurtosis", distribution_summary.kurtosis)
        append_row("Distribution", "Skewness", distribution_summary.skewness)

    below_ratio_note = f"{_format_ratio(distribution_summary.below_break_even_ratio)} of priced jobs"
    append_row(
        "Distribution",
        "Below break-even jobs",
        distribution_summary.below_break_even_count,
        "jobs",
        below_ratio_note,
    )

    append_row(
        "Profitability",
        "Median revenue per km",
        profitability_summary.revenue_per_km_median,
        "$/km",
    )
    append_row(
        "Profitability",
        "Mean revenue per km",
        profitability_summary.revenue_per_km_mean,
        "$/km",
    )
    append_row(
        "Profitability",
        "Median margin per m³",
        profitability_summary.margin_per_m3_median,
        "$/m³",
    )
    append_row(
        "Profitability",
        "Median margin per m³ %",
        profitability_summary.margin_per_m3_pct_median,
        "ratio",
    )
    append_row(
        "Profitability",
        "Median margin total",
        profitability_summary.margin_total_median,
        "$",
    )
    append_row(
        "Profitability",
        "Median margin total %",
        profitability_summary.margin_total_pct_median,
        "ratio",
    )

    if "price_per_m3" in df.columns:
        price_series = pd.to_numeric(df["price_per_m3"], errors="coerce")
        bands = price_series.apply(lambda value: classify_profit_band(value, break_even))
        band_counts = bands.value_counts()
        priced_jobs = len(price_series.dropna()) or 1
        for label, count in band_counts.items():
            ratio = _format_ratio(count / priced_jobs)
            append_row(
                "Profitability",
                f"Band - {label}",
                int(count),
                "jobs",
                f"{ratio} of priced jobs",
            )

    if "corridor_display" in df.columns:
        numeric_df = df.copy()
        column_pairs = {
            "price_per_m3_numeric": "price_per_m3",
            "margin_per_m3_numeric": "margin_per_m3",
            "margin_per_m3_pct_numeric": "margin_per_m3_pct",
            "margin_total_numeric": "margin_total",
            "volume_m3_numeric": "volume_m3",
        }
        for numeric_column, source_column in column_pairs.items():
            if source_column in numeric_df.columns:
                numeric_df[numeric_column] = pd.to_numeric(
                    numeric_df[source_column], errors="coerce"
                )
            else:
                numeric_df[numeric_column] = math.nan

        grouped = numeric_df.groupby("corridor_display", dropna=False)
        corridor_stats = grouped.agg(
            avg_price_per_m3=("price_per_m3_numeric", "mean"),
            avg_margin_per_m3=("margin_per_m3_numeric", "mean"),
            avg_margin_pct=("margin_per_m3_pct_numeric", "mean"),
            total_margin=("margin_total_numeric", "sum"),
            total_volume=("volume_m3_numeric", "sum"),
        )
        corridor_stats["job_count"] = grouped.size()

        corridor_stats = corridor_stats.replace({np.inf: math.nan, -np.inf: math.nan})

        if not corridor_stats.empty and corridor_stats["avg_margin_per_m3"].notna().any():
            sorted_corridors = corridor_stats.sort_values(
                "avg_margin_per_m3", ascending=False
            )
            top_corridors = sorted_corridors.head(top_n_corridors)
            for idx, (corridor, row) in enumerate(top_corridors.iterrows(), start=1):
                append_row(
                    "Optimisation",
                    f"Top corridor #{idx} by avg margin per m³",
                    corridor,
                    notes=_format_corridor_notes(row, break_even),
                )

            bottom_corridors = sorted_corridors.tail(top_n_corridors).sort_values(
                "avg_margin_per_m3", ascending=True
            )
            for idx, (corridor, row) in enumerate(bottom_corridors.iterrows(), start=1):
                append_row(
                    "Optimisation",
                    f"Lowest margin corridor #{idx}",
                    corridor,
                    notes=_format_corridor_notes(row, break_even),
                )

        if "avg_price_per_m3" in corridor_stats.columns:
            below_break_even = corridor_stats[
                corridor_stats["avg_price_per_m3"] < float(break_even)
            ]
            if not below_break_even.empty:
                entries: list[str] = []
                ordered = below_break_even.sort_values("avg_price_per_m3")
                for corridor, row in ordered.iterrows():
                    avg_price = _safe_float(row.get("avg_price_per_m3"))
                    if math.isnan(avg_price):
                        continue
                    delta = avg_price - float(break_even)
                    job_count = int(row.get("job_count", 0))
                    entries.append(
                        f"{corridor} ({job_count} jobs, Δ ${delta:.2f})"
                    )
                if entries:
                    append_row(
                        "Optimisation",
                        "Corridors below break-even",
                        ", ".join(entries),
                        notes="Negative Δ indicates pricing below break-even",
                    )

    return pd.DataFrame(rows, columns=["section", "metric", "value", "unit", "notes"])


def classify_profit_band(value: Optional[float], break_even: float) -> str:
    """Return the profitability band label for a per-m³ price."""

    if value is None:
        return "Unknown"
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "Unknown"
    if math.isnan(numeric_value):
        return "Unknown"

    diff = numeric_value - break_even
    for lower, upper, label in PROFITABILITY_BANDS:
        if lower <= diff < upper:
            return label
    return "Unknown"


def classify_profitability_status(
    value: Optional[float],
    break_even: float,
    *,
    abs_tolerance: float = DEFAULT_BREAK_EVEN_ABS_TOLERANCE,
    rel_tolerance: float = DEFAULT_BREAK_EVEN_REL_TOLERANCE,
) -> str:
    """Classify a price-per-m³ value as profitable, break-even or loss-leading.

    Parameters
    ----------
    value:
        Observed price-per-m³ for a job or lane. ``None`` and non-numeric values
        are treated as ``"Unknown"``.
    break_even:
        Baseline break-even price-per-m³ used as the reference value.
    abs_tolerance:
        Absolute tolerance (in $/m³) when considering whether a value is within
        the break-even band.  This defaults to ``5`` which equates to ±$5/m³.
    rel_tolerance:
        Relative tolerance expressed as a fraction of the break-even value.  The
        effective tolerance is the larger of ``abs_tolerance`` and
        ``break_even * rel_tolerance`` so the comparison remains stable across
        different break-even baselines.
    """

    if value is None:
        return "Unknown"
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "Unknown"
    if math.isnan(numeric_value):
        return "Unknown"

    diff = numeric_value - break_even
    tolerance = max(abs_tolerance, abs(break_even) * rel_tolerance)
    if abs(diff) <= tolerance:
        return "Break-even"
    if diff > 0:
        return "Profitable"
    return "Loss-leading"


def prepare_profitability_route_data(
    df: pd.DataFrame,
    break_even: float,
) -> pd.DataFrame:
    """Prepare per-job profitability records for the telemetry map."""

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

    if "break_even_per_m3" in map_df.columns:
        break_even_series = pd.to_numeric(map_df["break_even_per_m3"], errors="coerce").fillna(break_even)
        price_series = pd.to_numeric(map_df["price_per_m3"], errors="coerce")
        map_df["profit_band"] = [
            classify_profit_band(price, be)
            for price, be in zip(price_series, break_even_series)
        ]
    else:
        map_df["profit_band"] = map_df["price_per_m3"].apply(
            lambda value: classify_profit_band(value, break_even)
        )
    map_df["profit_band"] = map_df["price_per_m3"].apply(lambda value: classify_profit_band(value, break_even))
    map_df["profitability_status"] = map_df["price_per_m3"].apply(
        lambda value: classify_profitability_status(value, break_even)
    )
    map_df["colour"] = map_df["profit_band"].map(PROFITABILITY_COLOURS)
    map_df["colour"] = map_df["colour"].apply(
        lambda value: value if isinstance(value, (list, tuple)) else [128, 128, 128]
    )
    map_df["line_width"] = map_df["profit_band"].map(PROFITABILITY_WIDTHS).fillna(80)

    def _format_tooltip(row: pd.Series) -> str:
        corridor = row.get("corridor_display", "Corridor")
        price = row.get("price_per_m3")
        price_text = "n/a" if pd.isna(price) else f"${price:,.0f} per m³"
        status = row.get("profitability_status") or row.get("profit_band", "Unknown")
        band = row.get("profit_band")
        if band and band not in {"Unknown", status}:
            descriptor = f"{status} – {band}"
        else:
            descriptor = status
        return f"{corridor}: {descriptor} ({price_text})"

    map_df["tooltip"] = map_df.apply(_format_tooltip, axis=1)
    return map_df


def prepare_profitability_map_data(
    df: pd.DataFrame,
    break_even: float,
    *,
    placeholder: str = "Unknown",
) -> pd.DataFrame:
    """Return profitability-enriched routes ready for map visualisations.

    The Streamlit app renders the profitability network map using the columns
    generated by :func:`prepare_profitability_route_data`.  This helper applies
    the same profitability band classification and additionally exposes a
    ``map_colour_value`` column so the data can be reused for categorical colour
    legends when required.
    """

    map_df = prepare_profitability_route_data(df, break_even)
    if map_df.empty:
        return map_df

    colour_values = map_df.get("profit_band")
    if colour_values is None:
        map_df["map_colour_value"] = placeholder
    else:
        map_df["map_colour_value"] = colour_values.fillna(placeholder).astype(str)
    return map_df


def _band_styles():
    return {
        -0.5: ("-50%", "rgba(214, 39, 40, 0.55)", "dot"),
        -0.2: ("-20%", "rgba(255, 127, 14, 0.7)", "dash"),
        -0.1: ("-10%", "rgba(255, 187, 120, 0.8)", "dashdot"),
        0.0: ("Break-even", "rgba(33, 33, 33, 0.9)", "solid"),
        0.1: ("+10%", "rgba(31, 119, 180, 0.7)", "dashdot"),
        0.2: ("+20%", "rgba(44, 160, 44, 0.7)", "dash"),
        0.5: ("+50%", "rgba(148, 103, 189, 0.8)", "dot"),
    }


def create_histogram(df: pd.DataFrame, break_even: float, bins: Optional[int] = None) -> go.Figure:
    """Create a histogram with break-even bands."""
    priced = df.dropna(subset=["price_per_m3"])
    if priced.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No jobs available for the selected filters",
            xaxis_title="$ per m³",
            yaxis_title="Job count",
        )
        return fig

    if bins is None:
        bins = min(50, max(10, int(math.sqrt(len(priced)))))

    fig = px.histogram(
        priced,
        x="price_per_m3",
        nbins=bins,
        labels={"price_per_m3": "$ per m³", "count": "Job count"},
        title="$ per m³ distribution",
        opacity=0.85,
    )

    styles = _band_styles()
    for pct, (label, color, dash) in styles.items():
        if break_even <= 0 and pct != 0.0:
            continue
        x_val = break_even * (1 + pct)
        fig.add_vline(
            x=x_val,
            line_width=2 if pct == 0 else 1.5,
            line_dash=dash,
            line_color=color,
            annotation_text=label,
            annotation_position="top",
            annotation_font_color="#111",
            annotation_bgcolor="rgba(255, 255, 255, 0.85)",
            annotation_bordercolor=color,
        )

    priced_values = priced["price_per_m3"].dropna()
    mean_val = float(priced_values.mean()) if not priced_values.empty else math.nan
    std_val = float(priced_values.std(ddof=1)) if len(priced_values) > 1 else math.nan
    kurtosis_val = float(priced_values.kurtosis()) if len(priced_values) > 3 else math.nan

    if len(priced_values) > 1 and std_val and not math.isnan(std_val) and std_val > 0:
        _, bin_edges = np.histogram(priced_values, bins=bins)
        if len(bin_edges) > 1:
            bin_width = float(np.mean(np.diff(bin_edges)))
        else:
            bin_width = 0.0
        if bin_width > 0:
            x_vals = np.linspace(bin_edges[0], bin_edges[-1], 200)
            pdf = (1.0 / (std_val * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_vals - mean_val) / std_val) ** 2)
            y_vals = pdf * len(priced_values) * bin_width
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines",
                    name="Normal fit",
                    line=dict(color="rgba(17, 17, 17, 0.85)", width=2),
                )
            )

    stats_bits = []
    if not math.isnan(mean_val):
        stats_bits.append(f"μ={mean_val:,.2f}")
    if not math.isnan(std_val):
        stats_bits.append(f"σ={std_val:,.2f}")
    if not math.isnan(kurtosis_val):
        stats_bits.append(f"kurtosis={kurtosis_val:,.2f}")
    if stats_bits:
        fig.add_annotation(
            text=" | ".join(stats_bits),
            xref="paper",
            yref="paper",
            x=0.99,
            y=0.98,
            showarrow=False,
            align="right",
            bgcolor="rgba(255, 255, 255, 0.85)",
            bordercolor="rgba(17, 17, 17, 0.2)",
            font=dict(color="#111", size=12),
        )

    fig.update_layout(
        bargap=0.02,
        xaxis_title="$ per m³",
        yaxis_title="Job count",
        showlegend=False,
        hovermode="x unified",
    )

    return fig


def _empty_figure(title: str, x_title: str, y_title: str, message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(color="#555", size=13),
    )
    return fig


def filter_metro_jobs(df: pd.DataFrame, max_distance_km: float = 100.0) -> pd.DataFrame:
    """Return a copy filtered to jobs within the metro distance threshold."""

    if "distance_km" not in df.columns:
        return df.copy()

    distances = df["distance_km"].astype(float)
    within_threshold = distances.fillna(np.inf) <= max_distance_km
    return df.loc[within_threshold].copy()


def create_m3_vs_km_figure(df: pd.DataFrame) -> go.Figure:
    """Visualise $/m³ profitability relative to $/km earnings."""

    if "price_per_m3" not in df.columns or "revenue_per_km" not in df.columns:
        return _empty_figure(
            title="m³ vs km profitability",
            x_title="$ per m³",
            y_title="$ per km",
            message="No revenue or distance data available.",
        )

    subset = df.dropna(subset=["price_per_m3", "revenue_per_km"])
    if subset.empty:
        return _empty_figure(
            title="m³ vs km profitability",
            x_title="$ per m³",
            y_title="$ per km",
            message="Add jobs with both revenue and distance to unlock this view.",
        )

    hover_data: dict[str, object] = {}
    for column, fmt in [
        ("client_display", True),
        ("corridor_display", True),
        ("volume_m3", ":.1f"),
        ("volume", ":.1f"),
        ("distance_km", ":.1f"),
        ("margin_total", ":.0f"),
        ("margin_total_pct", ":.1%"),
    ]:
        if column in subset.columns:
            hover_data[column] = fmt

    color_col = "corridor_display" if "corridor_display" in subset.columns else None

    fig = px.scatter(
        subset,
        x="price_per_m3",
        y="revenue_per_km",
        color=color_col,
        hover_data=hover_data,
        labels={"price_per_m3": "$ per m³", "revenue_per_km": "$ per km"},
        title="m³ vs km profitability",
    )
    fig.update_traces(marker=dict(size=10, opacity=0.8))
    fig.update_layout(
        xaxis=dict(zeroline=False),
        yaxis=dict(zeroline=False),
        legend_title_text="Corridor" if color_col else None,
    )
    return fig


def create_m3_margin_figure(df: pd.DataFrame) -> go.Figure:
    """Compare quoted $/m³ to cost-derived $/m³ and highlight margin deltas."""

    required_cols = {"price_per_m3", "final_cost_per_m3"}
    if not required_cols.issubset(df.columns):
        return _empty_figure(
            title="Quoted vs calculated $/m³",
            x_title="Cost-derived $ per m³",
            y_title="Quoted $ per m³",
            message="Final cost data is unavailable.",
        )

    subset = df.dropna(subset=list(required_cols))
    if subset.empty:
        return _empty_figure(
            title="Quoted vs calculated $/m³",
            x_title="Cost-derived $ per m³",
            y_title="Quoted $ per m³",
            message="No jobs contain both quoted and calculated $/m³ values.",
        )

    hover_data: dict[str, object] = {}
    for column, fmt in [
        ("client_display", True),
        ("corridor_display", True),
        ("margin_per_m3", ":.2f"),
        ("margin_per_m3_pct", ":.1%"),
        ("margin_total", ":.0f"),
        ("margin_total_pct", ":.1%"),
        ("volume_m3", ":.1f"),
        ("distance_km", ":.1f"),
    ]:
        if column in subset.columns:
            hover_data[column] = fmt

    color_col = "margin_per_m3_pct" if "margin_per_m3_pct" in subset.columns else None
    color_args = {}
    if color_col:
        color_args = {
            "color": subset[color_col],
            "color_continuous_scale": "RdYlGn",
        }

    fig = px.scatter(
        subset,
        x="final_cost_per_m3",
        y="price_per_m3",
        hover_data=hover_data,
        labels={
            "final_cost_per_m3": "Cost-derived $ per m³",
            "price_per_m3": "Quoted $ per m³",
        },
        title="Quoted vs calculated $/m³",
        **color_args,
    )
    fig.update_traces(marker=dict(size=10, opacity=0.8))

    min_val = float(subset[["final_cost_per_m3", "price_per_m3"]].min().min())
    max_val = float(subset[["final_cost_per_m3", "price_per_m3"]].max().max())
    if max_val > min_val:
        parity_line = go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="rgba(17, 17, 17, 0.6)", dash="dash"),
            name="Parity",
            showlegend=False,
        )
        fig.add_trace(parity_line)

    fig.update_layout(coloraxis_colorbar=dict(title="Margin %"), legend_title_text=None)
    return fig


def create_metro_profitability_figure(
    df: pd.DataFrame, *, max_distance_km: float = 100.0
) -> go.Figure:
    """Summarise metro-only profitability with scatter and distribution views."""

    title = f"Metro profitability (≤{max_distance_km:,.0f} km)"
    metro_df = filter_metro_jobs(df, max_distance_km=max_distance_km)

    if metro_df.empty:
        return _empty_figure(
            title=title,
            x_title="$ per m³",
            y_title="$ per km",
            message="No jobs fall within the metro distance threshold.",
        )

    required = {"price_per_m3", "revenue_per_km"}
    missing = required - set(metro_df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        return _empty_figure(
            title=title,
            x_title="$ per m³",
            y_title="$ per km",
            message=f"Metro view requires columns: {missing_list}.",
        )

    scatter_df = metro_df.dropna(subset=list(required))
    if scatter_df.empty:
        return _empty_figure(
            title=title,
            x_title="$ per m³",
            y_title="$ per km",
            message="Metro jobs lack both price and revenue per km values.",
        )

    margin_available = (
        "margin_per_m3" in metro_df.columns
        and not metro_df["margin_per_m3"].dropna().empty
    )
    cost_ratio_series = pd.Series(dtype=float)
    if "final_cost_per_m3" in metro_df.columns:
        ratio_df = metro_df.dropna(subset=["final_cost_per_m3", "price_per_m3"])
        if not ratio_df.empty:
            denom = ratio_df["price_per_m3"].replace(0, np.nan)
            cost_ratio_series = (ratio_df["final_cost_per_m3"] / denom).replace(
                [np.inf, -np.inf], np.nan
            )
            cost_ratio_series = cost_ratio_series.dropna()
    cost_available = not cost_ratio_series.empty

    subplot_titles = ["Price vs $/km (metro)"]
    specs: list[dict[str, str]] = [{"type": "xy"}]
    if margin_available:
        subplot_titles.append("Margin $/m³ (metro)")
        specs.append({"type": "xy"})
    if cost_available:
        subplot_titles.append("Cost vs quote share")
        specs.append({"type": "xy"})

    fig = make_subplots(
        rows=1,
        cols=len(subplot_titles),
        subplot_titles=subplot_titles,
        specs=[specs],
        horizontal_spacing=0.08,
    )

    hover_bits: list[list[str]] = []
    hover_columns = [
        ("client_display", "Client"),
        ("corridor_display", "Corridor"),
        ("job_date", "Date"),
        ("volume_m3", "Volume (m³)"),
        ("distance_km", "Distance (km)"),
        ("margin_per_m3", "Margin $/m³"),
        ("margin_per_m3_pct", "Margin %"),
    ]
    for _, row in scatter_df.iterrows():
        parts = [
            f"Quoted $/m³: {row['price_per_m3']:,.2f}",
            f"$ per km: {row['revenue_per_km']:,.2f}",
        ]
        for column, label in hover_columns:
            if column not in scatter_df.columns:
                continue
            value = row.get(column)
            if pd.isna(value):
                continue
            if column.endswith("pct"):
                parts.append(f"{label}: {value * 100:.1f}%")
            elif isinstance(value, (int, float)):
                parts.append(f"{label}: {value:,.2f}")
            else:
                parts.append(f"{label}: {value}")
        hover_bits.append(parts)

    hover_texts = ["<br>".join(parts) for parts in hover_bits]

    marker_args: dict[str, object] = {"size": 10, "opacity": 0.85}
    if "margin_per_m3_pct" in scatter_df.columns and not scatter_df["margin_per_m3_pct"].dropna().empty:
        marker_args.update(
            {
                "color": scatter_df["margin_per_m3_pct"],
                "colorscale": "RdYlGn",
                "showscale": True,
                "colorbar": {"title": "Margin %", "tickformat": ".0%"},
            }
        )

    fig.add_trace(
        go.Scatter(
            x=scatter_df["price_per_m3"],
            y=scatter_df["revenue_per_km"],
            mode="markers",
            name="Metro jobs",
            marker=marker_args,
            hovertemplate="%{text}<extra></extra>",
            text=hover_texts,
        ),
        row=1,
        col=1,
    )

    fig.update_xaxes(title_text="Quoted $ per m³", row=1, col=1)
    fig.update_yaxes(title_text="Revenue $ per km", row=1, col=1)

    current_col = 2
    if margin_available:
        margin_series = metro_df["margin_per_m3"].dropna()
        fig.add_trace(
            go.Histogram(
                x=margin_series,
                name="Margin $/m³",
                marker=dict(color="rgba(91, 192, 222, 0.85)"),
                hovertemplate="Margin $/m³: %{x:,.2f}<br>Count: %{y}<extra></extra>",
            ),
            row=1,
            col=current_col,
        )
        fig.update_xaxes(title_text="Margin $ per m³", row=1, col=current_col)
        fig.update_yaxes(title_text="Job count", row=1, col=current_col)
        current_col += 1

    if cost_available:
        fig.add_trace(
            go.Histogram(
                x=cost_ratio_series,
                name="Cost sensitivity",
                marker=dict(color="rgba(217, 83, 79, 0.7)"),
                hovertemplate="Cost/price: %{x:.1%}<br>Count: %{y}<extra></extra>",
            ),
            row=1,
            col=current_col,
        )
        fig.update_xaxes(
            title_text="Cost as share of quoted price",
            tickformat=".0%",
            row=1,
            col=current_col,
        )
        fig.update_yaxes(title_text="Job count", row=1, col=current_col)

    fig.update_layout(
        title=title,
        bargap=0.05,
        hovermode="closest",
        legend_title_text=None,
    )

    return fig


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
