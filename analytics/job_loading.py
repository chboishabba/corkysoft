from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .db import ensure_global_parameters_table, migrate_geojson_to_routes
from .lane_assignment import backfill_lane_assignments, ensure_lane_assignment_schema
from .routes_map import populate_route_geometry


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
            d.lat AS dest_lat,
            hr.geojson AS route_geojson
        FROM historical_jobs AS hj
        LEFT JOIN addresses AS o ON hj.origin_address_id = o.id
        LEFT JOIN addresses AS d ON hj.destination_address_id = d.id
        LEFT JOIN historical_job_routes AS hr ON hr.historical_job_id = hj.id
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
        column
        for column in ("origin_country", "destination_country")
        if column in routes.columns
    ]
    if not candidate_columns:
        return routes.copy()

    mask = pd.Series(False, index=routes.index)
    for column in candidate_columns:
        values = routes[column].fillna("").astype(str).str.strip().str.lower()
        mask = mask | (values == normalized)

    return routes.loc[mask].copy()


def _prepare_loaded_jobs(
    df: pd.DataFrame,
    mapping,
    base_costs,
    *,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    clients: Optional[Sequence[str]] = None,
    corridor: Optional[str] = None,
    postcode_prefix: Optional[str] = None,
) -> tuple[pd.DataFrame, object]:
    """Return ``df`` filtered and enriched for downstream visualisations."""

    from .price_distribution import (
        POSTCODE_COLUMNS,
        _infer_datetime_parse_kwargs,
        compute_break_even_series,
    )

    if df.empty:
        return df, mapping

    working = df.copy()
    if "lane_assignment_status" not in working.columns:
        working["lane_assignment_status"] = ""
    if "lane_key" not in working.columns:
        working["lane_key"] = ""
    if "corridor_group_key" not in working.columns:
        working["corridor_group_key"] = ""

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
            working["margin_vs_break_even"] = (
                working["price_per_m3"] - break_even_per_m3
            )

    return working.reset_index(drop=True), mapping


def load_historical_jobs(
    conn,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    clients: Optional[Sequence[str]] = None,
    corridor: Optional[str] = None,
    postcode_prefix: Optional[str] = None,
):
    """Load historical job data applying the requested filters."""

    from .price_distribution import (
        _deduplicate_columns,
        _has_geometry,
        ensure_base_cost_parameters,
        infer_columns,
    )

    ensure_global_parameters_table(conn)
    migrate_geojson_to_routes(conn)
    base_costs = ensure_base_cost_parameters(conn)
    ensure_lane_assignment_schema(conn)
    backfill_lane_assignments(conn, dataset="historical")

    query = _historical_jobs_query()
    try:
        df = pd.read_sql_query(query, conn)
    except Exception:
        try:
            df = pd.read_sql_query("SELECT * FROM historical_jobs", conn)
        except Exception as exc:
            raise RuntimeError("historical_jobs table is required for this view") from exc

    if "id" in df.columns and "route_geojson" in df.columns:
        missing_ids = [
            int(value)
            for value in df.loc[
                ~df["route_geojson"].apply(_has_geometry), "id"
            ].dropna().tolist()
        ]
        if missing_ids:
            try:
                populated = populate_route_geometry(conn, missing_ids, dataset="historical")
            except Exception:
                populated = 0
            if populated:
                df = pd.read_sql_query(query, conn)

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
):
    """Load saved quick quote data from the ``quotes`` table."""

    from .price_distribution import (
        _coalesce_string_columns,
        _deduplicate_columns,
        ensure_base_cost_parameters,
        infer_columns,
    )

    ensure_global_parameters_table(conn)
    base_costs = ensure_base_cost_parameters(conn)
    ensure_lane_assignment_schema(conn)
    backfill_lane_assignments(conn, dataset="live")

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
    except Exception as exc:
        raise RuntimeError("quotes table is required for this view") from exc

    if df.empty:
        mapping = infer_columns(df)
        return df, mapping

    df = _deduplicate_columns(df)

    if "quote_date" in df.columns:
        df["job_date"] = df["quote_date"]

    _coalesce_string_columns(df, "origin_resolved", "origin_input", "origin")
    _coalesce_string_columns(df, "destination_resolved", "destination_input", "destination")

    quote_total = df["manual_quote"].where(df["manual_quote"].notna(), df["final_quote"])
    quote_total = pd.to_numeric(quote_total, errors="coerce")
    df["quote_total"] = quote_total
    df["revenue_total"] = quote_total
    df["revenue"] = quote_total

    if "cubic_m" in df.columns:
        volume_series = pd.to_numeric(df["cubic_m"], errors="coerce")
    else:
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

    if "client_display" in df.columns:
        df["client"] = df["client_display"].fillna("Quote builder")
    else:
        df["client"] = "Quote builder"

    client_lookup: Dict[int, str] = {}
    try:
        for quote_id, client_display in conn.execute("SELECT id, client_display FROM quotes"):
            if client_display:
                client_lookup[int(quote_id)] = str(client_display)
    except Exception:
        client_lookup = {}

    mapping = infer_columns(df)
    prepared_df, mapping = _prepare_loaded_jobs(
        df,
        mapping,
        base_costs,
        start_date=start_date,
        end_date=end_date,
        clients=clients,
        corridor=corridor,
        postcode_prefix=postcode_prefix,
    )

    if client_lookup and "id" in prepared_df.columns:
        mapped_clients = prepared_df["id"].map(client_lookup)
        prepared_df["client_display"] = mapped_clients.where(
            mapped_clients.notna(), prepared_df["client_display"]
        )
        prepared_df["client"] = prepared_df["client_display"].fillna("Quote builder")

    return prepared_df, mapping


def load_live_jobs(
    conn,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    clients: Optional[Sequence[str]] = None,
    corridor: Optional[str] = None,
    postcode_prefix: Optional[str] = None,
):
    """Load live job data from the ``jobs`` table for real-time monitoring."""

    from .price_distribution import (
        _deduplicate_columns,
        _has_geometry,
        ensure_base_cost_parameters,
        infer_columns,
    )

    ensure_global_parameters_table(conn)
    base_costs = ensure_base_cost_parameters(conn)

    try:
        df = pd.read_sql_query("SELECT * FROM jobs", conn)
    except Exception as exc:
        raise RuntimeError("jobs table is required for live monitoring") from exc

    if "id" in df.columns and "route_geojson" in df.columns:
        missing_ids = [
            int(value)
            for value in df.loc[
                ~df["route_geojson"].apply(_has_geometry), "id"
            ].dropna().tolist()
        ]
        if missing_ids:
            try:
                populated = populate_route_geometry(conn, missing_ids, dataset="live")
            except Exception:
                populated = 0
            if populated:
                df = pd.read_sql_query("SELECT * FROM jobs", conn)

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
