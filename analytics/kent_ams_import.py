"""Utilities for importing Kent AMS adapter payloads into SQLite tables."""
from __future__ import annotations

import json
import sqlite3
from datetime import date
from datetime import UTC, datetime
from typing import Iterable, Mapping, Sequence, Any

from analytics.db import ensure_dashboard_tables, upsert_job_by_number, upsert_supplier
from analytics.db.parameters import (
    ensure_global_parameters_table,
    get_parameter_text,
    get_parameter_value,
    set_parameter_text,
    set_parameter_value,
)
from analytics.operational_signals import (
    compute_route_spare_capacity_signal,
    upsert_job_operational_signal,
)
from corkysoft.pricing import choose_pricing_model, compute_base_subtotal, seasonal_uplift


KENT_RULE_MODES = {"ABS_ONLY", "PCT_ONLY", "EITHER", "BOTH"}
KENT_TENDER_RULE_MODE_KEY = "kent_tender_profit_rule_mode"
KENT_TENDER_ABS_THRESHOLD_KEY = "kent_tender_abs_margin_threshold"
KENT_TENDER_PCT_THRESHOLD_KEY = "kent_tender_margin_pct_threshold"
KENT_TENDER_LOSS_ALERT_KEY = "kent_tender_loss_margin_floor"
KENT_DEFAULT_RULE_MODE = "EITHER"
KENT_DEFAULT_ABS_THRESHOLD = 750.0
KENT_DEFAULT_PCT_THRESHOLD = 12.0
KENT_DEFAULT_LOSS_ALERT = 0.0
KENT_OVERRIDE_REASON_SEED = (
    ("retention", "Retention", "Retain a strategic customer or account."),
    ("backhaul_positioning", "Backhaul positioning", "Position fleet for downstream utilization or backhaul recovery."),
    ("relationship", "Relationship", "Protect a commercial or referral relationship."),
    ("overflow_relief", "Overflow relief", "Relieve peak-period operational pressure or overflow."),
    ("manual_route_knowledge", "Manual route knowledge", "Operator knows route/traffic/site details not captured by the model."),
    ("sla_recovery", "SLA recovery", "Accept a weaker tender to recover service commitments elsewhere."),
    ("other", "Other", "Operator-selected exception outside the standard categories."),
)
KENT_ALLOWED_HARD_BLOCK_FLAGS = {
    "capacity_compliance_block",
    "licensing_block",
    "fatigue_block",
    "dangerous_goods_block",
}


def _first_present(
    record: Mapping[str, object], *keys: str, default: object | None = None
) -> object | None:
    for key in keys:
        if key in record:
            return record[key]
    return default


def _clean_str(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    return str(value).strip() or None


def _coerce_float(value: object | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: object | None) -> int | None:
    number = _coerce_float(value)
    if number is None:
        return None
    try:
        return int(number)
    except (TypeError, ValueError):
        return None


def _coerce_timestamp(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    text = _clean_str(value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.isoformat()


def _coerce_date(value: object | None) -> date | None:
    ts = _coerce_timestamp(value)
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _coerce_bool(value: object | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _normalise_rule_mode(value: object | None) -> str:
    candidate = _clean_str(value)
    if not candidate:
        return KENT_DEFAULT_RULE_MODE
    normalized = candidate.upper()
    if normalized not in KENT_RULE_MODES:
        raise ValueError(f"Unsupported Kent profitability rule mode '{candidate}'")
    return normalized


def _parse_flag_list(value: object | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return sorted({_clean_str(item) for item in value if _clean_str(item)})
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return sorted({_clean_str(item) for item in parsed if _clean_str(item)})
        return sorted({_clean_str(part) for part in text.split(",") if _clean_str(part)})
    return []


def _serialize_flag_list(flags: Sequence[str]) -> str:
    unique_flags = sorted({_clean_str(flag) for flag in flags if _clean_str(flag)})
    return json.dumps(unique_flags)


def _deserialize_flag_list(value: object | None) -> list[str]:
    return _parse_flag_list(value)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _table_has_column(conn, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any((row["name"] if hasattr(row, "keys") else row[1]) == column for row in rows)


def _bootstrap_kent_policy(conn) -> None:
    ensure_global_parameters_table(conn)
    if get_parameter_text(conn, KENT_TENDER_RULE_MODE_KEY) is None:
        set_parameter_text(
            conn,
            KENT_TENDER_RULE_MODE_KEY,
            KENT_DEFAULT_RULE_MODE,
            "Default Kent AMS profitability rule mode.",
        )
    if get_parameter_value(conn, KENT_TENDER_ABS_THRESHOLD_KEY) is None:
        set_parameter_value(
            conn,
            KENT_TENDER_ABS_THRESHOLD_KEY,
            KENT_DEFAULT_ABS_THRESHOLD,
            "Default Kent AMS minimum absolute margin threshold in AUD.",
        )
    if get_parameter_value(conn, KENT_TENDER_PCT_THRESHOLD_KEY) is None:
        set_parameter_value(
            conn,
            KENT_TENDER_PCT_THRESHOLD_KEY,
            KENT_DEFAULT_PCT_THRESHOLD,
            "Default Kent AMS minimum margin percentage threshold.",
        )
    if get_parameter_value(conn, KENT_TENDER_LOSS_ALERT_KEY) is None:
        set_parameter_value(
            conn,
            KENT_TENDER_LOSS_ALERT_KEY,
            KENT_DEFAULT_LOSS_ALERT,
            "Loss alert floor for Kent AMS tenders. Values below this remain visible but are highlighted.",
        )


def _seed_override_reason_codes(conn) -> None:
    timestamp = _utc_now_iso()
    conn.executemany(
        """
        INSERT INTO kent_tender_override_reason_codes (
            code,
            label,
            description,
            active,
            system_seeded,
            created_at,
            updated_at
        ) VALUES (?, ?, ?, 1, 1, ?, ?)
        ON CONFLICT(code) DO NOTHING
        """,
        [(code, label, description, timestamp, timestamp) for code, label, description in KENT_OVERRIDE_REASON_SEED],
    )
    conn.commit()


def get_kent_tender_policy_config(conn) -> dict[str, Any]:
    _bootstrap_kent_policy(conn)
    return {
        "ruleMode": _normalise_rule_mode(
            get_parameter_text(conn, KENT_TENDER_RULE_MODE_KEY, KENT_DEFAULT_RULE_MODE)
        ),
        "absoluteMarginThreshold": float(
            get_parameter_value(conn, KENT_TENDER_ABS_THRESHOLD_KEY, KENT_DEFAULT_ABS_THRESHOLD)
            or 0.0
        ),
        "marginPercentThreshold": float(
            get_parameter_value(conn, KENT_TENDER_PCT_THRESHOLD_KEY, KENT_DEFAULT_PCT_THRESHOLD)
            or 0.0
        ),
        "lossAlertFloor": float(
            get_parameter_value(conn, KENT_TENDER_LOSS_ALERT_KEY, KENT_DEFAULT_LOSS_ALERT)
            or 0.0
        ),
        "updatedAt": _optional_parameter_updated_at(conn),
    }


def _optional_parameter_updated_at(conn) -> str | None:
    row = conn.execute(
        """
        SELECT MAX(updated_at) AS updated_at
        FROM global_parameters
        WHERE key IN (?, ?, ?, ?)
        """,
        (
            KENT_TENDER_RULE_MODE_KEY,
            KENT_TENDER_ABS_THRESHOLD_KEY,
            KENT_TENDER_PCT_THRESHOLD_KEY,
            KENT_TENDER_LOSS_ALERT_KEY,
        ),
    ).fetchone()
    if not row:
        return None
    updated_at = row["updated_at"] if isinstance(row, sqlite3.Row) else row[0]
    return _clean_str(updated_at)


def update_kent_tender_policy_config(
    conn,
    *,
    rule_mode: str,
    absolute_margin_threshold: float,
    margin_percent_threshold: float,
    loss_alert_floor: float,
) -> dict[str, Any]:
    normalized_rule_mode = _normalise_rule_mode(rule_mode)
    _bootstrap_kent_policy(conn)
    set_parameter_text(
        conn,
        KENT_TENDER_RULE_MODE_KEY,
        normalized_rule_mode,
        "Default Kent AMS profitability rule mode.",
    )
    set_parameter_value(
        conn,
        KENT_TENDER_ABS_THRESHOLD_KEY,
        float(absolute_margin_threshold),
        "Default Kent AMS minimum absolute margin threshold in AUD.",
    )
    set_parameter_value(
        conn,
        KENT_TENDER_PCT_THRESHOLD_KEY,
        float(margin_percent_threshold),
        "Default Kent AMS minimum margin percentage threshold.",
    )
    set_parameter_value(
        conn,
        KENT_TENDER_LOSS_ALERT_KEY,
        float(loss_alert_floor),
        "Loss alert floor for Kent AMS tenders.",
    )
    return get_kent_tender_policy_config(conn)


def list_kent_override_reason_codes(conn) -> list[dict[str, Any]]:
    _ensure_kent_tables(conn)
    rows = conn.execute(
        """
        SELECT code, label, description, active, system_seeded, updated_at
        FROM kent_tender_override_reason_codes
        ORDER BY active DESC, system_seeded DESC, label ASC
        """
    ).fetchall()
    return [
        {
            "code": row["code"],
            "label": row["label"],
            "description": row["description"],
            "active": bool(row["active"]),
            "systemSeeded": bool(row["system_seeded"]),
            "updatedAt": row["updated_at"],
        }
        for row in rows
    ]


def upsert_kent_override_reason_code(
    conn,
    *,
    code: str,
    label: str,
    description: str | None = None,
    active: bool = True,
) -> dict[str, Any]:
    _ensure_kent_tables(conn)
    normalized_code = _clean_str(code)
    normalized_label = _clean_str(label)
    if not normalized_code or not normalized_label:
        raise ValueError("Override reason code and label are required")
    timestamp = _utc_now_iso()
    conn.execute(
        """
        INSERT INTO kent_tender_override_reason_codes (
            code,
            label,
            description,
            active,
            system_seeded,
            created_at,
            updated_at
        ) VALUES (?, ?, ?, ?, 0, ?, ?)
        ON CONFLICT(code) DO UPDATE SET
            label = excluded.label,
            description = excluded.description,
            active = excluded.active,
            updated_at = excluded.updated_at
        """,
        (
            normalized_code,
            normalized_label,
            _clean_str(description),
            1 if active else 0,
            timestamp,
            timestamp,
        ),
    )
    conn.commit()
    row = conn.execute(
        """
        SELECT code, label, description, active, system_seeded, updated_at
        FROM kent_tender_override_reason_codes
        WHERE code = ?
        """,
        (normalized_code,),
    ).fetchone()
    return {
        "code": row["code"],
        "label": row["label"],
        "description": row["description"],
        "active": bool(row["active"]),
        "systemSeeded": bool(row["system_seeded"]),
        "updatedAt": row["updated_at"],
    }


def _ensure_kent_tables(conn) -> None:
    _bootstrap_kent_policy(conn)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS kent_job_tenders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tender_external_id TEXT NOT NULL UNIQUE,
            job_number TEXT NOT NULL,
            client_name TEXT,
            origin TEXT,
            destination TEXT,
            estimated_volume_m3 REAL,
            estimated_distance_km REAL,
            expected_revenue REAL,
            estimated_cost REAL,
            required_trucks INTEGER,
            required_workers INTEGER,
            tender_status TEXT NOT NULL DEFAULT 'open',
            due_at TEXT,
            move_date TEXT,
            notes TEXT,
            score_total REAL NOT NULL DEFAULT 0,
            score_profitability REAL NOT NULL DEFAULT 0,
            score_capacity REAL NOT NULL DEFAULT 0,
            score_urgency REAL NOT NULL DEFAULT 0,
            score_seasonality REAL NOT NULL DEFAULT 0,
            score_route_location REAL NOT NULL DEFAULT 0,
            score_spare_capacity REAL NOT NULL DEFAULT 0,
            overrideable_flags TEXT NOT NULL DEFAULT '[]',
            hard_block_flags TEXT NOT NULL DEFAULT '[]',
            recommended_action TEXT NOT NULL DEFAULT 'review',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS kent_job_bids (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bid_external_id TEXT NOT NULL UNIQUE,
            job_number TEXT NOT NULL,
            subcontractor_name TEXT,
            bid_amount REAL,
            currency TEXT NOT NULL DEFAULT 'AUD',
            bid_status TEXT,
            submitted_at TEXT,
            selected_at TEXT,
            notes TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS kent_job_awards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            award_external_id TEXT NOT NULL UNIQUE,
            job_number TEXT NOT NULL,
            bid_external_id TEXT,
            subcontractor_name TEXT NOT NULL,
            awarded_amount REAL,
            currency TEXT NOT NULL DEFAULT 'AUD',
            awarded_at TEXT,
            status TEXT,
            notes TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS kent_tender_override_reason_codes (
            code TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            description TEXT,
            active INTEGER NOT NULL DEFAULT 1,
            system_seeded INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS kent_tender_overrides (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tender_external_id TEXT NOT NULL,
            action TEXT NOT NULL,
            operator_id TEXT NOT NULL,
            reason_code TEXT NOT NULL,
            note TEXT,
            overrideable_flags TEXT NOT NULL DEFAULT '[]',
            hard_block_flags TEXT NOT NULL DEFAULT '[]',
            profitability_rule_mode TEXT,
            absolute_margin_threshold REAL,
            margin_percent_threshold REAL,
            expected_revenue REAL,
            estimated_cost REAL,
            estimated_margin REAL,
            estimated_margin_pct REAL,
            policy_matched INTEGER NOT NULL DEFAULT 0,
            policy_fail_reasons TEXT NOT NULL DEFAULT '[]',
            loss_alert INTEGER NOT NULL DEFAULT 0,
            score_total REAL,
            score_profitability REAL,
            score_capacity REAL,
            score_urgency REAL,
            score_seasonality REAL,
            score_route_location REAL,
            score_spare_capacity REAL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(tender_external_id) REFERENCES kent_job_tenders(tender_external_id),
            FOREIGN KEY(reason_code) REFERENCES kent_tender_override_reason_codes(code)
        );
        """
    )
    tender_column_defaults = {
        "origin": "TEXT",
        "destination": "TEXT",
        "estimated_volume_m3": "REAL",
        "estimated_distance_km": "REAL",
        "required_trucks": "INTEGER",
        "required_workers": "INTEGER",
        "due_at": "TEXT",
        "move_date": "TEXT",
        "notes": "TEXT",
        "score_profitability": "REAL NOT NULL DEFAULT 0",
        "score_capacity": "REAL NOT NULL DEFAULT 0",
        "score_urgency": "REAL NOT NULL DEFAULT 0",
        "score_seasonality": "REAL NOT NULL DEFAULT 0",
        "score_route_location": "REAL NOT NULL DEFAULT 0",
        "score_spare_capacity": "REAL NOT NULL DEFAULT 0",
        "overrideable_flags": "TEXT NOT NULL DEFAULT '[]'",
        "hard_block_flags": "TEXT NOT NULL DEFAULT '[]'",
        "recommended_action": "TEXT NOT NULL DEFAULT 'review'",
        "created_at": "TEXT NOT NULL DEFAULT ''",
        "updated_at": "TEXT NOT NULL DEFAULT ''",
    }
    for column, ddl in tender_column_defaults.items():
        if not _table_has_column(conn, "kent_job_tenders", column):
            conn.execute(f"ALTER TABLE kent_job_tenders ADD COLUMN {column} {ddl}")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_kent_job_tenders_status_score
        ON kent_job_tenders(tender_status, score_total DESC, due_at)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_kent_job_bids_job_number
        ON kent_job_bids(job_number)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_kent_job_awards_job_number
        ON kent_job_awards(job_number)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_kent_tender_overrides_tender
        ON kent_tender_overrides(tender_external_id, created_at DESC)
        """
    )
    _seed_override_reason_codes(conn)
    conn.commit()


def _estimate_cost_for_tender(
    *, estimated_distance_km: float | None, estimated_volume_m3: float | None, move_date: date | None
) -> float | None:
    if estimated_distance_km is None or estimated_volume_m3 is None:
        return None
    model = choose_pricing_model(max(estimated_distance_km, 0.0))
    base, _ = compute_base_subtotal(
        distance_km=max(estimated_distance_km, 0.0),
        cubic_m=max(estimated_volume_m3, 0.0),
        model=model,
    )
    seasonal = seasonal_uplift(move_date or date.today())
    return base * seasonal.multiplier


def _current_capacity_snapshot(conn) -> dict[str, float]:
    truck_row = conn.execute(
        "SELECT COUNT(*) FROM trucks WHERE COALESCE(active, 1) = 1"
    ).fetchone()
    worker_row = conn.execute(
        "SELECT COUNT(*) FROM workers WHERE COALESCE(active, 1) = 1"
    ).fetchone()
    shipment_row = conn.execute(
        "SELECT COUNT(*) FROM shipments WHERE status IN ('planned', 'in_transit', 'assigned')"
    ).fetchone()
    active_trucks = float(truck_row[0] if truck_row else 0)
    active_workers = float(worker_row[0] if worker_row else 0)
    active_shipments = float(shipment_row[0] if shipment_row else 0)
    return {
        "active_trucks": active_trucks,
        "active_workers": active_workers,
        "active_shipments": active_shipments,
    }


def _normalize_margin_amount_score(margin_amount: float | None) -> float:
    if margin_amount is None:
        return 50.0
    if margin_amount <= -500.0:
        return 0.0
    if margin_amount >= 5000.0:
        return 100.0
    return _clamp(((margin_amount + 500.0) / 5500.0) * 100.0)


def _normalize_margin_pct_score(margin_pct: float | None) -> float:
    if margin_pct is None:
        return 50.0
    return _clamp((margin_pct + 0.20) / 0.60 * 100.0)


def _evaluate_profitability_policy(
    *,
    expected_revenue: float | None,
    estimated_cost: float | None,
    policy_config: Mapping[str, Any],
) -> dict[str, Any]:
    margin_amount = None
    margin_pct = None
    if expected_revenue is not None and estimated_cost is not None:
        margin_amount = expected_revenue - estimated_cost
        if expected_revenue > 0:
            margin_pct = (margin_amount / expected_revenue) * 100.0

    absolute_threshold = float(policy_config["absoluteMarginThreshold"])
    margin_pct_threshold = float(policy_config["marginPercentThreshold"])
    loss_alert_floor = float(policy_config["lossAlertFloor"])
    rule_mode = _normalise_rule_mode(policy_config["ruleMode"])

    abs_pass = margin_amount is not None and margin_amount >= absolute_threshold
    pct_pass = margin_pct is not None and margin_pct >= margin_pct_threshold

    if rule_mode == "ABS_ONLY":
        matched = abs_pass
    elif rule_mode == "PCT_ONLY":
        matched = pct_pass
    elif rule_mode == "BOTH":
        matched = abs_pass and pct_pass
    else:
        matched = abs_pass or pct_pass

    fail_reasons: list[str] = []
    if not matched:
        if rule_mode in {"ABS_ONLY", "EITHER", "BOTH"} and not abs_pass:
            fail_reasons.append(
                f"abs_margin_below_threshold:{round(absolute_threshold, 2)}"
            )
        if rule_mode in {"PCT_ONLY", "EITHER", "BOTH"} and not pct_pass:
            fail_reasons.append(
                f"margin_pct_below_threshold:{round(margin_pct_threshold, 2)}"
            )

    loss_alert = margin_amount is not None and margin_amount < loss_alert_floor
    if loss_alert:
        fail_reasons.append(f"loss_alert_below_floor:{round(loss_alert_floor, 2)}")

    return {
        "ruleMode": rule_mode,
        "absoluteMarginThreshold": absolute_threshold,
        "marginPercentThreshold": margin_pct_threshold,
        "lossAlertFloor": loss_alert_floor,
        "marginAmount": round(margin_amount, 2) if margin_amount is not None else None,
        "marginPercent": round(margin_pct, 2) if margin_pct is not None else None,
        "matched": bool(matched),
        "failReasons": fail_reasons,
        "lossAlert": bool(loss_alert),
    }


def evaluate_kent_profitability_policy(
    *,
    expected_revenue: float | None,
    estimated_cost: float | None,
    policy_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Public wrapper for Kent profitability policy evaluation."""

    return _evaluate_profitability_policy(
        expected_revenue=expected_revenue,
        estimated_cost=estimated_cost,
        policy_config=policy_config,
    )


def _freshness_state(updated_at: str | None) -> tuple[str, float]:
    if not updated_at:
        return ("unknown", 0.0)
    try:
        parsed = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
    except ValueError:
        return ("unknown", 0.0)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    age_hours = max(0.0, (datetime.now(UTC) - parsed).total_seconds() / 3600.0)
    if age_hours <= 1.0:
        return ("fresh", 1.0)
    if age_hours <= 6.0:
        return ("recent", 0.8)
    if age_hours <= 24.0:
        return ("aging", 0.5)
    return ("stale", 0.2)


def _score_tender(
    *,
    expected_revenue: float | None,
    estimated_cost: float | None,
    required_trucks: int,
    required_workers: int,
    due_at: str | None,
    move_date: date | None,
    capacity: Mapping[str, float],
    score_route_location: float,
    score_spare_capacity: float,
    policy_config: Mapping[str, Any],
    hard_block_flags: Sequence[str],
    overrideable_flags: Sequence[str],
) -> dict[str, float | str]:
    policy_eval = _evaluate_profitability_policy(
        expected_revenue=expected_revenue,
        estimated_cost=estimated_cost,
        policy_config=policy_config,
    )
    score_profitability = round(
        (
            _normalize_margin_amount_score(policy_eval["marginAmount"])
            + _normalize_margin_pct_score(
                None
                if policy_eval["marginPercent"] is None
                else float(policy_eval["marginPercent"]) / 100.0
            )
        )
        / 2.0,
        2,
    )

    available_trucks = max(1.0, capacity.get("active_trucks", 0.0))
    available_workers = max(1.0, capacity.get("active_workers", 0.0))
    active_shipments = capacity.get("active_shipments", 0.0)
    truck_pressure = max(0.0, required_trucks / available_trucks)
    worker_pressure = max(0.0, required_workers / available_workers)
    flow_pressure = max(0.0, active_shipments / available_trucks)
    capacity_pressure = max(truck_pressure, worker_pressure, flow_pressure)
    score_capacity = _clamp(100.0 - (capacity_pressure * 40.0), lo=0.0, hi=100.0)

    urgency = 50.0
    due_dt = None
    if due_at:
        try:
            due_dt = datetime.fromisoformat(due_at.replace("Z", "+00:00"))
        except ValueError:
            due_dt = None
    if due_dt is not None:
        days_to_due = (due_dt - datetime.now(UTC)).total_seconds() / 86400.0
        if days_to_due <= 0:
            urgency = 100.0
        elif days_to_due <= 1:
            urgency = 95.0
        elif days_to_due <= 3:
            urgency = 85.0
        elif days_to_due <= 7:
            urgency = 70.0
        elif days_to_due <= 14:
            urgency = 55.0
        else:
            urgency = 40.0
    score_urgency = _clamp(urgency)

    seasonal = seasonal_uplift(move_date or date.today())
    # Base season=50, shoulder=70, peak=95
    if seasonal.multiplier >= 1.29:
        score_seasonality = 95.0
    elif seasonal.multiplier >= 1.10:
        score_seasonality = 70.0
    else:
        score_seasonality = 50.0

    score_total = (
        0.42 * score_profitability
        + 0.16 * score_capacity
        + 0.14 * score_urgency
        + 0.08 * score_seasonality
        + 0.12 * _clamp(score_route_location)
        + 0.08 * _clamp(score_spare_capacity)
    )

    if hard_block_flags:
        recommended_action = "hard_blocked"
    elif not policy_eval["matched"] and policy_eval["lossAlert"]:
        recommended_action = "review_with_override"
    elif not policy_eval["matched"]:
        recommended_action = "review_if_strategic"
    elif score_total >= 80:
        recommended_action = "pursue_now"
    elif score_total >= 60:
        recommended_action = "review_today"
    elif score_total >= 40:
        recommended_action = "review_if_capacity"
    else:
        recommended_action = "defer"

    if overrideable_flags and recommended_action == "pursue_now":
        recommended_action = "pursue_with_flags"

    return {
        "score_total": round(score_total, 2),
        "score_profitability": round(score_profitability, 2),
        "score_capacity": round(score_capacity, 2),
        "score_urgency": round(score_urgency, 2),
        "score_seasonality": round(score_seasonality, 2),
        "score_route_location": round(_clamp(score_route_location), 2),
        "score_spare_capacity": round(_clamp(score_spare_capacity), 2),
        "recommended_action": recommended_action,
        "policy_matched": int(policy_eval["matched"]),
        "policy_fail_reasons": _serialize_flag_list(policy_eval["failReasons"]),
        "loss_alert": int(policy_eval["lossAlert"]),
        "estimated_margin": policy_eval["marginAmount"],
        "estimated_margin_pct": policy_eval["marginPercent"],
    }


def _score_route_location(
    conn,
    *,
    origin: str | None,
    destination: str | None,
    estimated_distance_km: float | None,
) -> float:
    score = 0.0
    has_origin_dest = bool(_clean_str(origin) and _clean_str(destination))
    if has_origin_dest:
        score += 25.0

    if estimated_distance_km is not None and estimated_distance_km > 0:
        score += 20.0
    else:
        score += 8.0

    if not has_origin_dest:
        return _clamp(score)

    origin_key = _clean_str(origin).lower()
    destination_key = _clean_str(destination).lower()

    lane_history = conn.execute(
        """
        SELECT
            COUNT(*) AS lane_count,
            AVG(CASE
                WHEN revenue_total IS NOT NULL AND final_cost IS NOT NULL
                THEN (revenue_total - final_cost)
                ELSE NULL
            END) AS avg_margin
        FROM (
            SELECT origin, destination, revenue_total, final_cost FROM jobs
            UNION ALL
            SELECT origin, destination, revenue_total, final_cost FROM historical_jobs
        ) lane_rows
        WHERE LOWER(TRIM(origin)) = ? AND LOWER(TRIM(destination)) = ?
        """,
        (origin_key, destination_key),
    ).fetchone()

    lane_count = int(lane_history["lane_count"] or 0)
    avg_margin = _coerce_float(lane_history["avg_margin"])

    if lane_count >= 10:
        score += 35.0
    elif lane_count >= 5:
        score += 28.0
    elif lane_count >= 2:
        score += 18.0
    elif lane_count == 1:
        score += 10.0
    else:
        score += 4.0

    if avg_margin is not None:
        if avg_margin > 0:
            score += 20.0
        elif avg_margin < 0:
            score -= 12.0

    return _clamp(score)


def _display_name(record: Mapping[str, object]) -> str | None:
    company_name = _clean_str(
        _first_present(record, "company_name", "companyName", "subcontractor_name")
    )
    if company_name:
        return company_name
    first_name = _clean_str(_first_present(record, "first_name", "firstName"))
    last_name = _clean_str(_first_present(record, "last_name", "lastName"))
    if first_name and last_name:
        return f"{first_name} {last_name}"
    return first_name or last_name


def import_jobs(conn, records: Sequence[Mapping[str, object]], *, dry_run: bool = False) -> tuple[int, int]:
    inserted = 0
    updated = 0

    for record in records:
        job_number = _clean_str(
            _first_present(
                record,
                "job_number",
                "jobNumber",
                "move_id",
                "moveId",
                "external_id",
                "externalId",
                "id",
            )
        )
        if not job_number:
            raise ValueError("Kent AMS job record missing job identifier")

        existing = None
        can_query_existing = _table_has_column(conn, "jobs", "job_number")
        if not dry_run and can_query_existing:
            existing = conn.execute(
                "SELECT id FROM jobs WHERE job_number = ?", (job_number,)
            ).fetchone()

        if not dry_run:
            profitability_policy = evaluate_kent_profitability_policy(
                expected_revenue=_coerce_float(
                    _first_present(record, "revenue_total", "revenueTotal", "quoted_price")
                ),
                estimated_cost=_coerce_float(_first_present(record, "final_cost", "finalCost")),
                policy_config=get_kent_tender_policy_config(conn),
            )
            upsert_job_by_number(
                conn,
                job_number=job_number,
                job_date=_clean_str(_first_present(record, "job_date", "jobDate")),
                client=_clean_str(_first_present(record, "client", "client_name", "assignee_name")),
                client_reference=_clean_str(
                    _first_present(record, "client_reference", "clientReference", "customer_reference")
                ),
                origin=_clean_str(
                    _first_present(record, "origin", "origin_address", "pickup_address")
                ),
                destination=_clean_str(
                    _first_present(
                        record, "destination", "destination_address", "delivery_address"
                    )
                ),
                revenue_total=_coerce_float(
                    _first_present(record, "revenue_total", "revenueTotal", "quoted_price")
                ),
                revenue=_coerce_float(_first_present(record, "revenue", "award_value")),
                volume_m3=_coerce_float(
                    _first_present(record, "volume_m3", "volumeM3", "estimated_volume_m3")
                ),
                volume=_coerce_float(_first_present(record, "volume", "estimated_volume")),
                distance_km=_coerce_float(_first_present(record, "distance_km", "distanceKm")),
                final_cost=_coerce_float(_first_present(record, "final_cost", "finalCost")),
                origin_postcode=_clean_str(
                    _first_present(record, "origin_postcode", "originPostcode")
                ),
                destination_postcode=_clean_str(
                    _first_present(record, "destination_postcode", "destinationPostcode")
                ),
                origin_lat=_coerce_float(_first_present(record, "origin_lat", "originLat")),
                origin_lon=_coerce_float(_first_present(record, "origin_lon", "originLon")),
                dest_lat=_coerce_float(_first_present(record, "dest_lat", "destLat")),
                dest_lon=_coerce_float(_first_present(record, "dest_lon", "destLon")),
                created_at=_coerce_timestamp(
                    _first_present(record, "created_at", "createdAt", "created")
                ),
                updated_at=_coerce_timestamp(
                    _first_present(record, "updated_at", "updatedAt", "lastUpdated")
                ),
            )
            upsert_job_operational_signal(
                conn,
                job_number=job_number,
                origin=_clean_str(
                    _first_present(record, "origin", "origin_address", "pickup_address")
                ),
                destination=_clean_str(
                    _first_present(
                        record, "destination", "destination_address", "delivery_address"
                    )
                ),
                estimated_volume_m3=_coerce_float(
                    _first_present(record, "volume_m3", "volumeM3", "estimated_volume_m3")
                ),
                profitability_rule_mode=profitability_policy["ruleMode"],
                absolute_margin_threshold=profitability_policy["absoluteMarginThreshold"],
                margin_percent_threshold=profitability_policy["marginPercentThreshold"],
                policy_matched=profitability_policy["matched"],
                policy_fail_reasons=list(profitability_policy["failReasons"]),
                loss_alert=profitability_policy["lossAlert"],
                estimated_margin=profitability_policy["marginAmount"],
                estimated_margin_pct=profitability_policy["marginPercent"],
                source="kent_ams_import",
            )
        if dry_run or existing is None:
            inserted += 1
        else:
            updated += 1
    return inserted, updated


def import_subcontractors(
    conn, records: Sequence[Mapping[str, object]], *, dry_run: bool = False
) -> tuple[int, int]:
    inserted = 0
    updated = 0

    for record in records:
        company_name = _display_name(record)
        if not company_name:
            raise ValueError("Subcontractor record missing display name/company")

        existing = conn.execute(
            "SELECT id FROM suppliers WHERE company_name = ?", (company_name,)
        ).fetchone()
        if not dry_run:
            upsert_supplier(
                conn,
                company_name=company_name,
                contact_name=_clean_str(
                    _first_present(record, "contact_name", "contactName")
                ),
                contact_number=_clean_str(
                    _first_present(record, "contact_number", "contactNumber", "phone")
                ),
                email=_clean_str(record.get("email")),
                notes=_clean_str(record.get("notes")),
            )
        if existing is None:
            inserted += 1
        else:
            updated += 1

    return inserted, updated


def import_bids(conn, records: Sequence[Mapping[str, object]], *, dry_run: bool = False) -> tuple[int, int]:
    inserted = 0
    updated = 0

    for record in records:
        bid_external_id = _clean_str(
            _first_present(record, "bid_id", "bidId", "id", "external_id", "externalId")
        )
        job_number = _clean_str(
            _first_present(record, "job_number", "jobNumber", "move_id", "moveId")
        )
        if not bid_external_id or not job_number:
            raise ValueError("Bid record requires bid identifier and job identifier")

        existing = conn.execute(
            "SELECT id FROM kent_job_bids WHERE bid_external_id = ?",
            (bid_external_id,),
        ).fetchone()

        if not dry_run:
            timestamp = datetime.now(UTC).isoformat()
            conn.execute(
                """
                INSERT INTO kent_job_bids (
                    bid_external_id,
                    job_number,
                    subcontractor_name,
                    bid_amount,
                    currency,
                    bid_status,
                    submitted_at,
                    selected_at,
                    notes,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(bid_external_id) DO UPDATE SET
                    job_number = excluded.job_number,
                    subcontractor_name = excluded.subcontractor_name,
                    bid_amount = excluded.bid_amount,
                    currency = excluded.currency,
                    bid_status = excluded.bid_status,
                    submitted_at = excluded.submitted_at,
                    selected_at = excluded.selected_at,
                    notes = excluded.notes,
                    updated_at = excluded.updated_at
                """,
                (
                    bid_external_id,
                    job_number,
                    _clean_str(
                        _first_present(
                            record, "subcontractor_name", "subcontractorName", "vendor_name"
                        )
                    ),
                    _coerce_float(_first_present(record, "bid_amount", "bidAmount", "amount")),
                    _clean_str(record.get("currency")) or "AUD",
                    _clean_str(_first_present(record, "status", "bid_status")),
                    _coerce_timestamp(_first_present(record, "submitted_at", "submittedAt")),
                    _coerce_timestamp(_first_present(record, "selected_at", "selectedAt")),
                    _clean_str(record.get("notes")),
                    timestamp,
                    timestamp,
                ),
            )
            conn.commit()
        if existing is None:
            inserted += 1
        else:
            updated += 1
    return inserted, updated


def import_awards(conn, records: Sequence[Mapping[str, object]], *, dry_run: bool = False) -> tuple[int, int]:
    inserted = 0
    updated = 0

    for record in records:
        award_external_id = _clean_str(
            _first_present(record, "award_id", "awardId", "id", "external_id", "externalId")
        )
        job_number = _clean_str(
            _first_present(record, "job_number", "jobNumber", "move_id", "moveId")
        )
        subcontractor_name = _clean_str(
            _first_present(record, "subcontractor_name", "subcontractorName", "vendor_name")
        )
        if not award_external_id or not job_number or not subcontractor_name:
            raise ValueError(
                "Award record requires award identifier, job identifier, and subcontractor name"
            )

        existing = conn.execute(
            "SELECT id FROM kent_job_awards WHERE award_external_id = ?",
            (award_external_id,),
        ).fetchone()

        if not dry_run:
            timestamp = datetime.now(UTC).isoformat()
            conn.execute(
                """
                INSERT INTO kent_job_awards (
                    award_external_id,
                    job_number,
                    bid_external_id,
                    subcontractor_name,
                    awarded_amount,
                    currency,
                    awarded_at,
                    status,
                    notes,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(award_external_id) DO UPDATE SET
                    job_number = excluded.job_number,
                    bid_external_id = excluded.bid_external_id,
                    subcontractor_name = excluded.subcontractor_name,
                    awarded_amount = excluded.awarded_amount,
                    currency = excluded.currency,
                    awarded_at = excluded.awarded_at,
                    status = excluded.status,
                    notes = excluded.notes,
                    updated_at = excluded.updated_at
                """,
                (
                    award_external_id,
                    job_number,
                    _clean_str(_first_present(record, "bid_id", "bidId", "bid_external_id")),
                    subcontractor_name,
                    _coerce_float(
                        _first_present(record, "awarded_amount", "awardedAmount", "amount")
                    ),
                    _clean_str(record.get("currency")) or "AUD",
                    _coerce_timestamp(_first_present(record, "awarded_at", "awardedAt")),
                    _clean_str(_first_present(record, "status", "award_status")) or "selected",
                    _clean_str(record.get("notes")),
                    timestamp,
                    timestamp,
                ),
            )
            conn.commit()
        if existing is None:
            inserted += 1
        else:
            updated += 1
    return inserted, updated


def _extract_overrideable_flags(
    record: Mapping[str, object], *, spare_capacity_score: float
) -> list[str]:
    flags = _parse_flag_list(
        _first_present(record, "overrideable_flags", "overrideableFlags", "soft_flags", "softFlags")
    )
    if _coerce_bool(_first_present(record, "transfer_rule_violated", "transferRuleViolated")):
        flags.append("beyond_transfer_rule")
    if _coerce_bool(_first_present(record, "sla_risk_increased", "slaRiskIncreased")):
        flags.append("sla_risk_increased")
    if _coerce_bool(_first_present(record, "capacity_relief_achieved", "capacityReliefAchieved")):
        flags.append("capacity_relief_achieved")
    if spare_capacity_score >= 70.0:
        flags.append("capacity_relief_achieved")
    return sorted({_clean_str(flag) for flag in flags if _clean_str(flag)})


def _extract_hard_block_flags(record: Mapping[str, object]) -> list[str]:
    flags = [
        flag
        for flag in _parse_flag_list(_first_present(record, "hard_block_flags", "hardBlockFlags"))
        if flag in KENT_ALLOWED_HARD_BLOCK_FLAGS
    ]
    if _coerce_bool(_first_present(record, "capacity_blocked", "capacityBlocked")):
        flags.append("capacity_compliance_block")
    if _coerce_bool(_first_present(record, "licensing_blocked", "licensingBlocked")):
        flags.append("licensing_block")
    if _coerce_bool(_first_present(record, "fatigue_blocked", "fatigueBlocked")):
        flags.append("fatigue_block")
    if _coerce_bool(_first_present(record, "dangerous_goods_blocked", "dangerousGoodsBlocked")):
        flags.append("dangerous_goods_block")
    return sorted(
        {
            _clean_str(flag)
            for flag in flags
            if _clean_str(flag) in KENT_ALLOWED_HARD_BLOCK_FLAGS
        }
    )


def import_tenders(conn, records: Sequence[Mapping[str, object]], *, dry_run: bool = False) -> tuple[int, int]:
    inserted = 0
    updated = 0
    capacity = _current_capacity_snapshot(conn)
    policy_config = get_kent_tender_policy_config(conn)

    for record in records:
        tender_external_id = _clean_str(
            _first_present(record, "tender_id", "tenderId", "id", "external_id", "externalId")
        )
        job_number = _clean_str(
            _first_present(record, "job_number", "jobNumber", "move_id", "moveId")
        )
        if not tender_external_id or not job_number:
            raise ValueError("Tender record requires tender identifier and job identifier")

        expected_revenue = _coerce_float(
            _first_present(record, "expected_revenue", "expectedRevenue", "quoted_price")
        )
        estimated_cost = _coerce_float(_first_present(record, "estimated_cost", "estimatedCost"))
        estimated_volume_m3 = _coerce_float(
            _first_present(record, "estimated_volume_m3", "estimatedVolumeM3", "volume_m3")
        )
        estimated_distance_km = _coerce_float(
            _first_present(record, "estimated_distance_km", "estimatedDistanceKm", "distance_km")
        )
        move_date_value = _coerce_date(_first_present(record, "move_date", "moveDate"))
        origin_value = _clean_str(
            _first_present(record, "origin", "origin_address", "pickup_address")
        )
        destination_value = _clean_str(
            _first_present(record, "destination", "destination_address", "delivery_address")
        )
        if estimated_cost is None:
            estimated_cost = _estimate_cost_for_tender(
                estimated_distance_km=estimated_distance_km,
                estimated_volume_m3=estimated_volume_m3,
                move_date=move_date_value,
            )
        required_trucks = _coerce_int(_first_present(record, "required_trucks", "requiredTrucks")) or 1
        required_workers = _coerce_int(_first_present(record, "required_workers", "requiredWorkers")) or 2
        due_at = _coerce_timestamp(_first_present(record, "due_at", "dueAt", "tender_due_at"))
        route_location_score = _score_route_location(
            conn,
            origin=origin_value,
            destination=destination_value,
            estimated_distance_km=estimated_distance_km,
        )
        spare_capacity_signal = compute_route_spare_capacity_signal(
            conn,
            origin=origin_value,
            destination=destination_value,
            required_trucks=required_trucks,
            estimated_volume_m3=estimated_volume_m3,
        )
        overrideable_flags = _extract_overrideable_flags(
            record,
            spare_capacity_score=float(spare_capacity_signal["score"]),
        )
        hard_block_flags = _extract_hard_block_flags(record)

        scores = _score_tender(
            expected_revenue=expected_revenue,
            estimated_cost=estimated_cost,
            required_trucks=required_trucks,
            required_workers=required_workers,
            due_at=due_at,
            move_date=move_date_value,
            capacity=capacity,
            score_route_location=route_location_score,
            score_spare_capacity=float(spare_capacity_signal["score"]),
            policy_config=policy_config,
            hard_block_flags=hard_block_flags,
            overrideable_flags=overrideable_flags,
        )

        existing = conn.execute(
            "SELECT id FROM kent_job_tenders WHERE tender_external_id = ?",
            (tender_external_id,),
        ).fetchone()
        if not dry_run:
            timestamp = datetime.now(UTC).isoformat()
            conn.execute(
                """
                INSERT INTO kent_job_tenders (
                    tender_external_id,
                    job_number,
                    client_name,
                    origin,
                    destination,
                    estimated_volume_m3,
                    estimated_distance_km,
                    expected_revenue,
                    estimated_cost,
                    required_trucks,
                    required_workers,
                    tender_status,
                    due_at,
                    move_date,
                    notes,
                    score_total,
                    score_profitability,
                    score_capacity,
                    score_urgency,
                    score_seasonality,
                    score_route_location,
                    score_spare_capacity,
                    overrideable_flags,
                    hard_block_flags,
                    recommended_action,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tender_external_id) DO UPDATE SET
                    job_number = excluded.job_number,
                    client_name = excluded.client_name,
                    origin = excluded.origin,
                    destination = excluded.destination,
                    estimated_volume_m3 = excluded.estimated_volume_m3,
                    estimated_distance_km = excluded.estimated_distance_km,
                    expected_revenue = excluded.expected_revenue,
                    estimated_cost = excluded.estimated_cost,
                    required_trucks = excluded.required_trucks,
                    required_workers = excluded.required_workers,
                    tender_status = excluded.tender_status,
                    due_at = excluded.due_at,
                    move_date = excluded.move_date,
                    notes = excluded.notes,
                    score_total = excluded.score_total,
                    score_profitability = excluded.score_profitability,
                    score_capacity = excluded.score_capacity,
                    score_urgency = excluded.score_urgency,
                    score_seasonality = excluded.score_seasonality,
                    score_route_location = excluded.score_route_location,
                    score_spare_capacity = excluded.score_spare_capacity,
                    overrideable_flags = excluded.overrideable_flags,
                    hard_block_flags = excluded.hard_block_flags,
                    recommended_action = excluded.recommended_action,
                    updated_at = excluded.updated_at
                """,
                (
                    tender_external_id,
                    job_number,
                    _clean_str(_first_present(record, "client_name", "clientName", "assignee_name")),
                    origin_value,
                    destination_value,
                    estimated_volume_m3,
                    estimated_distance_km,
                    expected_revenue,
                    estimated_cost,
                    required_trucks,
                    required_workers,
                    _clean_str(_first_present(record, "status", "tender_status")) or "open",
                    due_at,
                    _coerce_timestamp(_first_present(record, "move_date", "moveDate")),
                    _clean_str(record.get("notes")),
                    scores["score_total"],
                    scores["score_profitability"],
                    scores["score_capacity"],
                    scores["score_urgency"],
                    scores["score_seasonality"],
                    scores["score_route_location"],
                    scores["score_spare_capacity"],
                    _serialize_flag_list(overrideable_flags),
                    _serialize_flag_list(hard_block_flags),
                    scores["recommended_action"],
                    timestamp,
                    timestamp,
                ),
            )
            conn.commit()
        if existing is None:
            inserted += 1
        else:
            updated += 1
    return inserted, updated


def list_prioritized_tenders(
    conn, *, status: str = "open", limit: int = 50
) -> list[dict[str, Any]]:
    policy_config = get_kent_tender_policy_config(conn)
    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='kent_job_tenders'"
    ).fetchone() is None:
        return []
    rows = conn.execute(
        """
        SELECT
            tender_external_id,
            job_number,
            client_name,
            origin,
            destination,
            expected_revenue,
            estimated_cost,
            required_trucks,
            required_workers,
            due_at,
            move_date,
            tender_status,
            score_total,
            score_profitability,
            score_capacity,
            score_urgency,
            score_seasonality,
            score_route_location,
            score_spare_capacity,
            overrideable_flags,
            hard_block_flags,
            recommended_action,
            updated_at
        FROM kent_job_tenders
        WHERE (? = 'all' OR tender_status = ?)
        """,
        (status, status),
    ).fetchall()

    prioritized: list[dict[str, Any]] = []
    for row in rows:
        revenue = _coerce_float(row["expected_revenue"])
        cost = _coerce_float(row["estimated_cost"])
        policy_eval = _evaluate_profitability_policy(
            expected_revenue=revenue,
            estimated_cost=cost,
            policy_config=policy_config,
        )
        freshness_state, confidence = _freshness_state(_clean_str(row["updated_at"]))
        hard_block_flags = [
            flag
            for flag in _deserialize_flag_list(row["hard_block_flags"])
            if flag in KENT_ALLOWED_HARD_BLOCK_FLAGS
        ]
        overrideable_flags = _deserialize_flag_list(row["overrideable_flags"])
        recommended_action = row["recommended_action"]
        if hard_block_flags:
            recommended_action = "hard_blocked"
        elif not policy_eval["matched"]:
            recommended_action = "review_with_override" if policy_eval["lossAlert"] else "review_if_strategic"
        prioritized.append(
            {
                "tenderExternalId": row["tender_external_id"],
                "jobNumber": row["job_number"],
                "clientName": row["client_name"],
                "origin": row["origin"],
                "destination": row["destination"],
                "expectedRevenue": revenue,
                "estimatedCost": cost,
                "estimatedMargin": policy_eval["marginAmount"],
                "estimatedMarginPct": policy_eval["marginPercent"],
                "requiredTrucks": row["required_trucks"],
                "requiredWorkers": row["required_workers"],
                "dueAt": row["due_at"],
                "moveDate": row["move_date"],
                "status": row["tender_status"],
                "scoreTotal": row["score_total"],
                "scoreProfitability": row["score_profitability"],
                "scoreCapacity": row["score_capacity"],
                "scoreUrgency": row["score_urgency"],
                "scoreSeasonality": row["score_seasonality"],
                "scoreRouteLocation": row["score_route_location"],
                "scoreSpareCapacity": row["score_spare_capacity"],
                "recommendedAction": recommended_action,
                "updatedAt": row["updated_at"],
                "profitRuleMode": policy_eval["ruleMode"],
                "absoluteMarginThreshold": policy_eval["absoluteMarginThreshold"],
                "marginPercentThreshold": policy_eval["marginPercentThreshold"],
                "policyMatched": policy_eval["matched"],
                "policyFailReasons": policy_eval["failReasons"],
                "lossAlert": policy_eval["lossAlert"],
                "overrideableFlags": overrideable_flags,
                "hardBlockFlags": hard_block_flags,
                "freshnessState": freshness_state,
                "confidenceScore": confidence,
            }
        )
    prioritized.sort(
        key=lambda item: (
            1 if item["hardBlockFlags"] else 0,
            0 if item["policyMatched"] else 1,
            1 if item["lossAlert"] else 0,
            -float(item["scoreTotal"]),
            -(item["estimatedMargin"] if item["estimatedMargin"] is not None else -10**9),
            -(item["estimatedMarginPct"] if item["estimatedMarginPct"] is not None else -10**9),
            item["dueAt"] or "9999-12-31T23:59:59+00:00",
        ),
    )
    return prioritized[: max(1, int(limit))]


def record_kent_tender_override(
    conn,
    *,
    tender_external_id: str,
    action: str,
    operator_id: str,
    reason_code: str,
    note: str | None = None,
) -> dict[str, Any]:
    _ensure_kent_tables(conn)
    tender_id = _clean_str(tender_external_id)
    normalized_action = _clean_str(action)
    normalized_operator = _clean_str(operator_id)
    normalized_reason = _clean_str(reason_code)
    if not tender_id or not normalized_action or not normalized_operator or not normalized_reason:
        raise ValueError("Tender override requires tender id, action, operator id, and reason code")

    tender_row = conn.execute(
        """
        SELECT *
        FROM kent_job_tenders
        WHERE tender_external_id = ?
        """,
        (tender_id,),
    ).fetchone()
    if tender_row is None:
        raise ValueError(f"Kent tender '{tender_id}' not found")

    reason_row = conn.execute(
        """
        SELECT code, active
        FROM kent_tender_override_reason_codes
        WHERE code = ?
        """,
        (normalized_reason,),
    ).fetchone()
    if reason_row is None or not bool(reason_row["active"]):
        raise ValueError(f"Kent override reason '{normalized_reason}' is not active")

    policy_config = get_kent_tender_policy_config(conn)
    policy_eval = _evaluate_profitability_policy(
        expected_revenue=_coerce_float(tender_row["expected_revenue"]),
        estimated_cost=_coerce_float(tender_row["estimated_cost"]),
        policy_config=policy_config,
    )
    hard_block_flags = [
        flag
        for flag in _deserialize_flag_list(tender_row["hard_block_flags"])
        if flag in KENT_ALLOWED_HARD_BLOCK_FLAGS
    ]
    if hard_block_flags:
        raise ValueError("Hard-blocked tenders cannot be overridden through this workflow")

    timestamp = _utc_now_iso()
    conn.execute(
        """
        INSERT INTO kent_tender_overrides (
            tender_external_id,
            action,
            operator_id,
            reason_code,
            note,
            overrideable_flags,
            hard_block_flags,
            profitability_rule_mode,
            absolute_margin_threshold,
            margin_percent_threshold,
            expected_revenue,
            estimated_cost,
            estimated_margin,
            estimated_margin_pct,
            policy_matched,
            policy_fail_reasons,
            loss_alert,
            score_total,
            score_profitability,
            score_capacity,
            score_urgency,
            score_seasonality,
            score_route_location,
            score_spare_capacity,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            tender_id,
            normalized_action,
            normalized_operator,
            normalized_reason,
            _clean_str(note),
            tender_row["overrideable_flags"],
            tender_row["hard_block_flags"],
            policy_eval["ruleMode"],
            policy_eval["absoluteMarginThreshold"],
            policy_eval["marginPercentThreshold"],
            _coerce_float(tender_row["expected_revenue"]),
            _coerce_float(tender_row["estimated_cost"]),
            policy_eval["marginAmount"],
            policy_eval["marginPercent"],
            1 if policy_eval["matched"] else 0,
            _serialize_flag_list(policy_eval["failReasons"]),
            1 if policy_eval["lossAlert"] else 0,
            _coerce_float(tender_row["score_total"]),
            _coerce_float(tender_row["score_profitability"]),
            _coerce_float(tender_row["score_capacity"]),
            _coerce_float(tender_row["score_urgency"]),
            _coerce_float(tender_row["score_seasonality"]),
            _coerce_float(tender_row["score_route_location"]),
            _coerce_float(tender_row["score_spare_capacity"]),
            timestamp,
        ),
    )
    conn.commit()
    override_id = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    return {
        "id": int(override_id),
        "tenderExternalId": tender_id,
        "action": normalized_action,
        "operatorId": normalized_operator,
        "reasonCode": normalized_reason,
        "note": _clean_str(note),
        "overrideableFlags": _deserialize_flag_list(tender_row["overrideable_flags"]),
        "hardBlockFlags": hard_block_flags,
        "policyMatched": policy_eval["matched"],
        "policyFailReasons": policy_eval["failReasons"],
        "lossAlert": policy_eval["lossAlert"],
        "createdAt": timestamp,
    }


def list_kent_tender_override_history(
    conn, *, tender_external_id: str
) -> list[dict[str, Any]]:
    _ensure_kent_tables(conn)
    rows = conn.execute(
        """
        SELECT
            o.id,
            o.tender_external_id,
            o.action,
            o.operator_id,
            o.reason_code,
            r.label AS reason_label,
            o.note,
            o.overrideable_flags,
            o.hard_block_flags,
            o.profitability_rule_mode,
            o.absolute_margin_threshold,
            o.margin_percent_threshold,
            o.estimated_margin,
            o.estimated_margin_pct,
            o.policy_matched,
            o.policy_fail_reasons,
            o.loss_alert,
            o.score_total,
            o.created_at
        FROM kent_tender_overrides o
        LEFT JOIN kent_tender_override_reason_codes r
            ON r.code = o.reason_code
        WHERE o.tender_external_id = ?
        ORDER BY o.created_at DESC, o.id DESC
        """,
        (_clean_str(tender_external_id),),
    ).fetchall()
    return [
        {
            "id": row["id"],
            "tenderExternalId": row["tender_external_id"],
            "action": row["action"],
            "operatorId": row["operator_id"],
            "reasonCode": row["reason_code"],
            "reasonLabel": row["reason_label"],
            "note": row["note"],
            "overrideableFlags": _deserialize_flag_list(row["overrideable_flags"]),
            "hardBlockFlags": [
                flag
                for flag in _deserialize_flag_list(row["hard_block_flags"])
                if flag in KENT_ALLOWED_HARD_BLOCK_FLAGS
            ],
            "profitRuleMode": row["profitability_rule_mode"],
            "absoluteMarginThreshold": _coerce_float(row["absolute_margin_threshold"]),
            "marginPercentThreshold": _coerce_float(row["margin_percent_threshold"]),
            "estimatedMargin": _coerce_float(row["estimated_margin"]),
            "estimatedMarginPct": _coerce_float(row["estimated_margin_pct"]),
            "policyMatched": bool(row["policy_matched"]),
            "policyFailReasons": _deserialize_flag_list(row["policy_fail_reasons"]),
            "lossAlert": bool(row["loss_alert"]),
            "scoreTotal": _coerce_float(row["score_total"]),
            "createdAt": row["created_at"],
        }
        for row in rows
    ]


def get_tender_calibration(
    conn, *, lookback_days: int = 180
) -> dict[str, Any]:
    """Return score-band calibration metrics for Kent tender triage.

    Calibration focuses on:
    - win rate per score band (awarded tenders / total tenders)
    - realized margin metrics where downstream job outcomes are present
    """

    lookback_days = max(1, int(lookback_days))

    has_tenders = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='kent_job_tenders'"
    ).fetchone()
    if has_tenders is None:
        return {
            "summary": {
                "lookbackDays": lookback_days,
                "tenders": 0,
                "wins": 0,
                "overallWinRate": 0.0,
                "avgRealizedMargin": None,
                "meanAbsMarginError": None,
            },
            "bands": [
                {
                    "scoreBand": label,
                    "tenders": 0,
                    "wins": 0,
                    "winRate": 0.0,
                    "avgPredictedMargin": None,
                    "avgRealizedMargin": None,
                    "meanAbsMarginError": None,
                }
                for label in ("90-100", "75-89", "60-74", "40-59", "0-39")
            ],
        }

    job_columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
    }
    if {"job_number", "revenue_total", "final_cost"}.issubset(job_columns):
        jobs_join = "LEFT JOIN jobs j ON j.job_number = t.job_number"
        realized_revenue_expr = "j.revenue_total AS realized_revenue"
        realized_cost_expr = "j.final_cost AS realized_cost"
    else:
        jobs_join = ""
        realized_revenue_expr = "NULL AS realized_revenue"
        realized_cost_expr = "NULL AS realized_cost"

    rows = conn.execute(
        f"""
        SELECT
            t.tender_external_id,
            t.job_number,
            t.score_total,
            t.tender_status,
            t.expected_revenue,
            t.estimated_cost,
            t.updated_at,
            CASE
                WHEN LOWER(COALESCE(t.tender_status, '')) IN ('awarded', 'won', 'selected') THEN 1
                WHEN EXISTS (
                    SELECT 1
                    FROM kent_job_awards a
                    WHERE a.job_number = t.job_number
                ) THEN 1
                ELSE 0
            END AS won_flag,
            {realized_revenue_expr},
            {realized_cost_expr}
        FROM kent_job_tenders t
        {jobs_join}
        WHERE datetime(COALESCE(t.updated_at, t.created_at)) >= datetime('now', ?)
        """,
        (f"-{lookback_days} days",),
    ).fetchall()

    bands: list[dict[str, Any]] = [
        {"label": "90-100", "min": 90.0, "max": 100.0, "rows": []},
        {"label": "75-89", "min": 75.0, "max": 90.0, "rows": []},
        {"label": "60-74", "min": 60.0, "max": 75.0, "rows": []},
        {"label": "40-59", "min": 40.0, "max": 60.0, "rows": []},
        {"label": "0-39", "min": 0.0, "max": 40.0, "rows": []},
    ]

    def _assign_band(score: float) -> dict[str, Any]:
        for band in bands:
            if band["min"] <= score and (
                score < band["max"] or (band["max"] == 100.0 and score <= 100.0)
            ):
                return band
        return bands[-1]

    total_tenders = 0
    total_wins = 0
    total_realized_margin = 0.0
    total_realized_margin_count = 0
    total_prediction_error = 0.0
    total_prediction_error_count = 0

    for row in rows:
        score = float(row["score_total"] or 0.0)
        won_flag = int(row["won_flag"] or 0)
        expected_revenue = _coerce_float(row["expected_revenue"])
        estimated_cost = _coerce_float(row["estimated_cost"])
        predicted_margin = None
        if expected_revenue is not None and estimated_cost is not None:
            predicted_margin = expected_revenue - estimated_cost

        realized_revenue = _coerce_float(row["realized_revenue"])
        realized_cost = _coerce_float(row["realized_cost"])
        realized_margin = None
        if realized_revenue is not None and realized_cost is not None:
            realized_margin = realized_revenue - realized_cost

        band = _assign_band(score)
        band["rows"].append(
            {
                "won": won_flag,
                "predicted_margin": predicted_margin,
                "realized_margin": realized_margin,
            }
        )

        total_tenders += 1
        total_wins += won_flag
        if realized_margin is not None:
            total_realized_margin += realized_margin
            total_realized_margin_count += 1
        if predicted_margin is not None and realized_margin is not None:
            total_prediction_error += abs(realized_margin - predicted_margin)
            total_prediction_error_count += 1

    band_metrics: list[dict[str, Any]] = []
    for band in bands:
        entries = band["rows"]
        count = len(entries)
        wins = sum(1 for e in entries if e["won"])
        realized = [e["realized_margin"] for e in entries if e["realized_margin"] is not None]
        predicted = [e["predicted_margin"] for e in entries if e["predicted_margin"] is not None]
        errors = [
            abs(e["realized_margin"] - e["predicted_margin"])
            for e in entries
            if e["predicted_margin"] is not None and e["realized_margin"] is not None
        ]
        band_metrics.append(
            {
                "scoreBand": band["label"],
                "tenders": count,
                "wins": wins,
                "winRate": round((wins / count) if count else 0.0, 4),
                "avgPredictedMargin": round(sum(predicted) / len(predicted), 2) if predicted else None,
                "avgRealizedMargin": round(sum(realized) / len(realized), 2) if realized else None,
                "meanAbsMarginError": round(sum(errors) / len(errors), 2) if errors else None,
            }
        )

    summary = {
        "lookbackDays": lookback_days,
        "tenders": total_tenders,
        "wins": total_wins,
        "overallWinRate": round((total_wins / total_tenders) if total_tenders else 0.0, 4),
        "avgRealizedMargin": round(total_realized_margin / total_realized_margin_count, 2)
        if total_realized_margin_count
        else None,
        "meanAbsMarginError": round(total_prediction_error / total_prediction_error_count, 2)
        if total_prediction_error_count
        else None,
    }

    return {"summary": summary, "bands": band_metrics}


def import_kent_ams_records(
    conn, resource: str, records: Iterable[Mapping[str, object]], *, dry_run: bool = False
) -> tuple[int, int]:
    """Dispatch Kent AMS imports based on the resource path segment."""

    ensure_dashboard_tables(conn)
    _ensure_kent_tables(conn)
    normalized = resource.strip().lower()
    record_list = list(records)

    if normalized == "jobs":
        return import_jobs(conn, record_list, dry_run=dry_run)
    if normalized in {"subcontractors", "vendors"}:
        return import_subcontractors(conn, record_list, dry_run=dry_run)
    if normalized == "tenders":
        return import_tenders(conn, record_list, dry_run=dry_run)
    if normalized == "bids":
        return import_bids(conn, record_list, dry_run=dry_run)
    if normalized == "awards":
        return import_awards(conn, record_list, dry_run=dry_run)
    raise ValueError(f"Unsupported Kent AMS resource '{resource}'")


__all__ = [
    "evaluate_kent_profitability_policy",
    "get_kent_tender_policy_config",
    "import_kent_ams_records",
    "import_jobs",
    "import_subcontractors",
    "import_tenders",
    "import_bids",
    "import_awards",
    "list_prioritized_tenders",
    "get_tender_calibration",
    "list_kent_override_reason_codes",
    "list_kent_tender_override_history",
    "record_kent_tender_override",
    "update_kent_tender_policy_config",
    "upsert_kent_override_reason_code",
]
