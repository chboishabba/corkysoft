"""Operational capacity and route-fit signals reused across ingest and quoting."""
from __future__ import annotations

from datetime import UTC, datetime
import sqlite3
from typing import Any


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def _norm(value: str | None) -> str:
    return (value or "").strip().lower()


def _capacity_ratio(capacity_m3: float | None, active_load: float | None) -> float:
    if capacity_m3 is None or capacity_m3 <= 0:
        return 0.0
    load = max(0.0, float(active_load or 0.0))
    return max(0.0, min(1.0, (capacity_m3 - load) / capacity_m3))


def compute_route_spare_capacity_signal(
    conn: sqlite3.Connection,
    *,
    origin: str | None,
    destination: str | None,
    required_trucks: int = 1,
    estimated_volume_m3: float | None = None,
) -> dict[str, Any]:
    """Estimate how favorable a tender/quote is given live route spare capacity."""

    required_trucks = max(1, int(required_trucks))
    if not _table_exists(conn, "shipments") or not _table_exists(conn, "trucks"):
        return {
            "score": 50.0,
            "label": "neutral",
            "matchingSpareTrucks": 0,
            "destinationSpareTrucks": 0,
            "activeTrucks": 0,
        }

    origin_key = _norm(origin)
    destination_key = _norm(destination)
    has_route = bool(origin_key and destination_key)

    rows = conn.execute(
        """
        SELECT
            s.truck_id,
            t.capacity_m3,
            SUM(CASE
                WHEN s.status IN ('planned', 'in_transit', 'assigned')
                THEN COALESCE(s.quantity, 1)
                ELSE 0
            END) AS active_load,
            MAX(CASE
                WHEN LOWER(TRIM(COALESCE(s.from_location, j.origin, h.origin, ''))) = ?
                 AND LOWER(TRIM(COALESCE(s.to_location, j.destination, h.destination, ''))) = ?
                THEN 1 ELSE 0
            END) AS direct_match,
            MAX(CASE
                WHEN LOWER(TRIM(COALESCE(s.to_location, j.destination, h.destination, ''))) = ?
                THEN 1 ELSE 0
            END) AS destination_match
        FROM shipments s
        LEFT JOIN trucks t ON t.truck_id = s.truck_id
        LEFT JOIN jobs j ON j.id = s.job_id
        LEFT JOIN historical_jobs h ON h.id = s.historical_job_id
        WHERE s.status IN ('planned', 'in_transit', 'assigned')
          AND s.truck_id IS NOT NULL
        GROUP BY s.truck_id, t.capacity_m3
        """,
        (origin_key, destination_key, destination_key),
    ).fetchall()

    active_trucks = len(rows)
    if not rows:
        return {
            "score": 40.0,
            "label": "constrained",
            "matchingSpareTrucks": 0,
            "destinationSpareTrucks": 0,
            "activeTrucks": 0,
        }

    matching_spare = 0
    destination_spare = 0
    for row in rows:
        spare_ratio = _capacity_ratio(row["capacity_m3"], row["active_load"])
        is_spare = spare_ratio >= 0.20
        if has_route and row["direct_match"] and is_spare:
            matching_spare += 1
        elif has_route and row["destination_match"] and is_spare:
            destination_spare += 1

    score = 45.0
    if matching_spare >= required_trucks:
        score += 45.0
    elif matching_spare > 0:
        score += 28.0 * (matching_spare / required_trucks)
    elif destination_spare > 0:
        score += 12.0

    if estimated_volume_m3 is not None and estimated_volume_m3 > 0:
        # Mild penalty for very large moves when we do not see direct spare trucks.
        if matching_spare == 0 and estimated_volume_m3 > 35:
            score -= 10.0

    score = max(0.0, min(100.0, score))
    if score >= 75:
        label = "favorable"
    elif score >= 55:
        label = "workable"
    else:
        label = "constrained"

    return {
        "score": round(score, 2),
        "label": label,
        "matchingSpareTrucks": matching_spare,
        "destinationSpareTrucks": destination_spare,
        "activeTrucks": active_trucks,
    }


def upsert_job_operational_signal(
    conn: sqlite3.Connection,
    *,
    job_number: str,
    origin: str | None,
    destination: str | None,
    estimated_volume_m3: float | None = None,
    profitability_rule_mode: str | None = None,
    absolute_margin_threshold: float | None = None,
    margin_percent_threshold: float | None = None,
    policy_matched: bool | None = None,
    policy_fail_reasons: list[str] | None = None,
    loss_alert: bool | None = None,
    estimated_margin: float | None = None,
    estimated_margin_pct: float | None = None,
    source: str = "ingest",
) -> dict[str, Any]:
    """Compute and persist a route spare-capacity signal for a job record."""

    if not job_number.strip():
        raise ValueError("job_number is required for operational signal persistence")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS job_operational_signals (
            job_number TEXT PRIMARY KEY,
            signal_score REAL NOT NULL,
            signal_label TEXT NOT NULL,
            matching_spare_trucks INTEGER NOT NULL DEFAULT 0,
            destination_spare_trucks INTEGER NOT NULL DEFAULT 0,
            active_trucks INTEGER NOT NULL DEFAULT 0,
            profitability_rule_mode TEXT,
            absolute_margin_threshold REAL,
            margin_percent_threshold REAL,
            policy_matched INTEGER,
            policy_fail_reasons TEXT,
            loss_alert INTEGER,
            estimated_margin REAL,
            estimated_margin_pct REAL,
            source TEXT NOT NULL DEFAULT 'ingest',
            computed_at TEXT NOT NULL
        )
        """
    )
    columns = {
        row["name"] if hasattr(row, "keys") else row[1]
        for row in conn.execute("PRAGMA table_info(job_operational_signals)").fetchall()
    }
    optional_columns = {
        "profitability_rule_mode": "TEXT",
        "absolute_margin_threshold": "REAL",
        "margin_percent_threshold": "REAL",
        "policy_matched": "INTEGER",
        "policy_fail_reasons": "TEXT",
        "loss_alert": "INTEGER",
        "estimated_margin": "REAL",
        "estimated_margin_pct": "REAL",
    }
    for column, ddl in optional_columns.items():
        if column not in columns:
            conn.execute(f"ALTER TABLE job_operational_signals ADD COLUMN {column} {ddl}")

    signal = compute_route_spare_capacity_signal(
        conn,
        origin=origin,
        destination=destination,
        required_trucks=1,
        estimated_volume_m3=estimated_volume_m3,
    )
    timestamp = datetime.now(UTC).isoformat()
    conn.execute(
        """
        INSERT INTO job_operational_signals (
            job_number,
            signal_score,
            signal_label,
            matching_spare_trucks,
            destination_spare_trucks,
            active_trucks,
            profitability_rule_mode,
            absolute_margin_threshold,
            margin_percent_threshold,
            policy_matched,
            policy_fail_reasons,
            loss_alert,
            estimated_margin,
            estimated_margin_pct,
            source,
            computed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(job_number) DO UPDATE SET
            signal_score = excluded.signal_score,
            signal_label = excluded.signal_label,
            matching_spare_trucks = excluded.matching_spare_trucks,
            destination_spare_trucks = excluded.destination_spare_trucks,
            active_trucks = excluded.active_trucks,
            profitability_rule_mode = excluded.profitability_rule_mode,
            absolute_margin_threshold = excluded.absolute_margin_threshold,
            margin_percent_threshold = excluded.margin_percent_threshold,
            policy_matched = excluded.policy_matched,
            policy_fail_reasons = excluded.policy_fail_reasons,
            loss_alert = excluded.loss_alert,
            estimated_margin = excluded.estimated_margin,
            estimated_margin_pct = excluded.estimated_margin_pct,
            source = excluded.source,
            computed_at = excluded.computed_at
        """,
        (
            job_number.strip(),
            float(signal["score"]),
            str(signal["label"]),
            int(signal["matchingSpareTrucks"]),
            int(signal["destinationSpareTrucks"]),
            int(signal["activeTrucks"]),
            profitability_rule_mode,
            absolute_margin_threshold,
            margin_percent_threshold,
            None if policy_matched is None else int(policy_matched),
            None if policy_fail_reasons is None else ",".join(policy_fail_reasons),
            None if loss_alert is None else int(loss_alert),
            estimated_margin,
            estimated_margin_pct,
            source,
            timestamp,
        ),
    )
    conn.commit()
    return signal


__all__ = ["compute_route_spare_capacity_signal", "upsert_job_operational_signal"]
