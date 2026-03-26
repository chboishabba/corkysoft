"""Benchmarking and quote-guidance helpers for commercial workflows."""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Optional


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row[1])
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }


def _normalise_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


@dataclass
class QuoteBenchmarkOverlay:
    benchmark_available: bool
    current_price_per_m3: float
    benchmark_scope: str
    direct_job_count: int
    reverse_job_count: int
    benchmark_job_count: int
    benchmark_price_per_m3_median: Optional[float] = None
    benchmark_price_per_m3_p25: Optional[float] = None
    benchmark_price_per_m3_p75: Optional[float] = None
    benchmark_weighted_price_per_m3: Optional[float] = None
    benchmark_margin_per_m3_median: Optional[float] = None
    benchmark_position_label: str = "No benchmark"
    recommended_price_per_m3: Optional[float] = None
    recommended_quote_total: Optional[float] = None
    recommended_adjustment: Optional[float] = None
    recommended_adjustment_pct: Optional[float] = None
    backhaul_opportunity: bool = False
    backhaul_label: str = "none"
    backhaul_score: float = 0.0
    suggested_discount_pct: float = 0.0
    suggested_discount_amount: float = 0.0
    notes: list[str] = field(default_factory=list)


def build_quote_benchmark_overlay(
    conn: sqlite3.Connection,
    *,
    origin_resolved: str,
    destination_resolved: str,
    origin_postcode: str | None,
    destination_postcode: str | None,
    cubic_m: float,
    current_quote_total: float,
    spare_capacity_signal: dict[str, Any] | None = None,
) -> QuoteBenchmarkOverlay:
    """Benchmark a quote against matched history and derive backhaul guidance."""

    volume = _safe_float(cubic_m) or 0.0
    quote_total = _safe_float(current_quote_total) or 0.0
    current_price_per_m3 = quote_total / volume if volume > 0 else 0.0

    if volume <= 0 or not _table_exists(conn, "historical_jobs"):
        return QuoteBenchmarkOverlay(
            benchmark_available=False,
            current_price_per_m3=current_price_per_m3,
            benchmark_scope="unavailable",
            direct_job_count=0,
            reverse_job_count=0,
            benchmark_job_count=0,
            recommended_price_per_m3=current_price_per_m3 or None,
            recommended_quote_total=quote_total or None,
        )

    columns = _table_columns(conn, "historical_jobs")
    select_columns = [
        "price_per_m3",
        "revenue_total",
        "volume_m3",
        "final_cost",
        "origin",
        "destination",
        "origin_postcode",
        "destination_postcode",
    ]
    present_select = [column for column in select_columns if column in columns]
    if "price_per_m3" not in present_select:
        return QuoteBenchmarkOverlay(
            benchmark_available=False,
            current_price_per_m3=current_price_per_m3,
            benchmark_scope="unavailable",
            direct_job_count=0,
            reverse_job_count=0,
            benchmark_job_count=0,
            recommended_price_per_m3=current_price_per_m3 or None,
            recommended_quote_total=quote_total or None,
            notes=["Historical price data is missing price_per_m3 values."],
        )

    rows = conn.execute(
        f"SELECT {', '.join(present_select)} FROM historical_jobs"
    ).fetchall()

    direct_prices: list[float] = []
    reverse_prices: list[float] = []
    benchmark_prices: list[float] = []
    benchmark_margin_per_m3: list[float] = []
    weighted_revenue = 0.0
    weighted_volume = 0.0

    origin_norm = _normalise_text(origin_resolved)
    destination_norm = _normalise_text(destination_resolved)
    origin_postcode_norm = str(origin_postcode or "").strip()
    destination_postcode_norm = str(destination_postcode or "").strip()

    for row in rows:
        row_dict = dict(row)
        price_per_m3 = _safe_float(row_dict.get("price_per_m3"))
        if price_per_m3 is None:
            continue

        row_origin = _normalise_text(row_dict.get("origin"))
        row_destination = _normalise_text(row_dict.get("destination"))
        row_origin_postcode = str(row_dict.get("origin_postcode") or "").strip()
        row_destination_postcode = str(row_dict.get("destination_postcode") or "").strip()

        direct_match = False
        reverse_match = False
        if (
            origin_postcode_norm
            and destination_postcode_norm
            and row_origin_postcode
            and row_destination_postcode
        ):
            direct_match = (
                row_origin_postcode == origin_postcode_norm
                and row_destination_postcode == destination_postcode_norm
            )
            reverse_match = (
                row_origin_postcode == destination_postcode_norm
                and row_destination_postcode == origin_postcode_norm
            )
        else:
            direct_match = row_origin == origin_norm and row_destination == destination_norm
            reverse_match = row_origin == destination_norm and row_destination == origin_norm

        if not direct_match and not reverse_match:
            continue

        if direct_match:
            direct_prices.append(price_per_m3)
        if reverse_match:
            reverse_prices.append(price_per_m3)

        benchmark_prices.append(price_per_m3)

        row_revenue = _safe_float(row_dict.get("revenue_total"))
        row_volume = _safe_float(row_dict.get("volume_m3"))
        if row_revenue is not None and row_volume is not None and row_volume > 0:
            weighted_revenue += row_revenue
            weighted_volume += row_volume

        final_cost = _safe_float(row_dict.get("final_cost"))
        if final_cost is not None and row_volume is not None and row_volume > 0:
            benchmark_margin_per_m3.append(price_per_m3 - (final_cost / row_volume))

    direct_job_count = len(direct_prices)
    reverse_job_count = len(reverse_prices)
    benchmark_job_count = len(benchmark_prices)
    benchmark_available = benchmark_job_count > 0

    benchmark_median = _median(benchmark_prices)
    benchmark_p25 = _quantile(benchmark_prices, 0.25)
    benchmark_p75 = _quantile(benchmark_prices, 0.75)
    benchmark_margin_median = _median(benchmark_margin_per_m3)
    benchmark_weighted = (
        weighted_revenue / weighted_volume if weighted_volume > 0 else None
    )

    notes: list[str] = []
    if direct_job_count and reverse_job_count:
        benchmark_scope = "direct_and_reverse"
        notes.append(
            "Benchmark blends matched direction history with reverse-lane backhaul context."
        )
    elif direct_job_count:
        benchmark_scope = "direct_only"
        notes.append("Benchmark is based on matched same-direction lane history.")
    elif reverse_job_count:
        benchmark_scope = "reverse_only"
        notes.append("No same-direction history found; benchmark uses reverse-lane backhaul history.")
    else:
        benchmark_scope = "unavailable"
        notes.append("No comparable historical jobs were found for this route.")

    if benchmark_median is None:
        position_label = "No benchmark"
    elif benchmark_p25 is not None and current_price_per_m3 < benchmark_p25:
        position_label = "Below market band"
        notes.append("Current quote is below the observed benchmark band.")
    elif benchmark_p75 is not None and current_price_per_m3 > benchmark_p75:
        position_label = "Above market band"
        notes.append("Current quote sits above the upper observed benchmark band.")
    else:
        position_label = "Within market band"
        notes.append("Current quote sits inside the observed benchmark band.")

    spare_capacity_signal = spare_capacity_signal or {}
    matching_spare = int(spare_capacity_signal.get("matchingSpareTrucks") or 0)
    destination_spare = int(spare_capacity_signal.get("destinationSpareTrucks") or 0)
    signal_label = str(spare_capacity_signal.get("label") or "").strip().lower()

    backhaul_score = 0.0
    if reverse_job_count:
        backhaul_score += min(40.0, reverse_job_count * 8.0)
    if destination_spare:
        backhaul_score += min(30.0, destination_spare * 15.0)
    if matching_spare:
        backhaul_score += min(20.0, matching_spare * 10.0)
    if signal_label == "favorable":
        backhaul_score += 10.0
    elif signal_label == "workable":
        backhaul_score += 5.0
    backhaul_score = max(0.0, min(100.0, backhaul_score))

    if backhaul_score >= 70:
        backhaul_label = "strong"
    elif backhaul_score >= 40:
        backhaul_label = "moderate"
    elif backhaul_score > 0:
        backhaul_label = "limited"
    else:
        backhaul_label = "none"
    backhaul_opportunity = backhaul_label != "none"

    suggested_discount_pct = 0.0
    if backhaul_opportunity and current_price_per_m3 > 0:
        suggested_discount_pct += min(4.0, reverse_job_count * 1.5)
        suggested_discount_pct += min(4.0, destination_spare * 2.0)
        suggested_discount_pct += min(3.0, matching_spare * 1.5)
        if signal_label == "favorable":
            suggested_discount_pct += 1.0
        market_floor = benchmark_p25 if benchmark_p25 is not None else benchmark_median
        if market_floor is not None and current_price_per_m3 > 0:
            market_headroom_pct = max(
                0.0,
                ((current_price_per_m3 - market_floor) / current_price_per_m3) * 100.0,
            )
            suggested_discount_pct = min(suggested_discount_pct, market_headroom_pct)
        suggested_discount_pct = max(0.0, min(12.0, suggested_discount_pct))

    suggested_discount_amount = quote_total * (suggested_discount_pct / 100.0)

    recommended_price = current_price_per_m3 if current_price_per_m3 > 0 else None
    if benchmark_median is not None and recommended_price is not None:
        if benchmark_p25 is not None and current_price_per_m3 < benchmark_p25:
            recommended_price = benchmark_p25
        elif backhaul_opportunity and suggested_discount_pct > 0:
            lower_bound = benchmark_median
            if backhaul_label == "strong" and benchmark_p25 is not None:
                lower_bound = benchmark_p25
            discounted_price = current_price_per_m3 * (1.0 - (suggested_discount_pct / 100.0))
            recommended_price = max(lower_bound, discounted_price)
            if recommended_price < current_price_per_m3:
                notes.append("Backhaul conditions support a controlled discount without dropping outside the observed lane band.")
        elif benchmark_p75 is not None and current_price_per_m3 > benchmark_p75:
            recommended_price = benchmark_p75

    recommended_quote_total = (
        recommended_price * volume if recommended_price is not None and volume > 0 else None
    )
    recommended_adjustment = (
        recommended_quote_total - quote_total
        if recommended_quote_total is not None
        else None
    )
    recommended_adjustment_pct = (
        ((recommended_quote_total / quote_total) - 1.0) * 100.0
        if recommended_quote_total is not None and quote_total > 0
        else None
    )

    if backhaul_opportunity:
        notes.append(
            f"Backhaul signal is {backhaul_label}; discount headroom is capped at {suggested_discount_pct:.1f}%."
        )

    return QuoteBenchmarkOverlay(
        benchmark_available=benchmark_available,
        current_price_per_m3=current_price_per_m3,
        benchmark_scope=benchmark_scope,
        direct_job_count=direct_job_count,
        reverse_job_count=reverse_job_count,
        benchmark_job_count=benchmark_job_count,
        benchmark_price_per_m3_median=benchmark_median,
        benchmark_price_per_m3_p25=benchmark_p25,
        benchmark_price_per_m3_p75=benchmark_p75,
        benchmark_weighted_price_per_m3=benchmark_weighted,
        benchmark_margin_per_m3_median=benchmark_margin_median,
        benchmark_position_label=position_label,
        recommended_price_per_m3=recommended_price,
        recommended_quote_total=recommended_quote_total,
        recommended_adjustment=recommended_adjustment,
        recommended_adjustment_pct=recommended_adjustment_pct,
        backhaul_opportunity=backhaul_opportunity,
        backhaul_label=backhaul_label,
        backhaul_score=backhaul_score,
        suggested_discount_pct=suggested_discount_pct,
        suggested_discount_amount=suggested_discount_amount,
        notes=notes,
    )


__all__ = ["QuoteBenchmarkOverlay", "build_quote_benchmark_overlay"]
