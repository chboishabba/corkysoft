"""Situational-awareness ingestion helpers and adaptive-policy updates."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Mapping

from analytics.adaptive_policy import (
    AdaptivePolicyProposalItem,
    CLOSURE_DELAY_FACTOR_KEY,
    LANE_ETA_MULTIPLIER_KEY,
    WEATHER_RISK_MULTIPLIER_KEY,
    apply_bounded_parameter_target,
    create_adaptive_policy_proposal,
    ensure_adaptive_policy_defaults,
    load_adaptive_policy_snapshot,
)

DEFAULT_LOOKBACK_HOURS = 6

EVENT_TYPE_CLOSURE = "closure"
EVENT_TYPE_WEATHER = "weather"
EVENT_TYPE_TRAFFIC = "traffic"

DEFAULT_EVENT_TYPES = {EVENT_TYPE_CLOSURE, EVENT_TYPE_WEATHER, EVENT_TYPE_TRAFFIC}


@dataclass(frozen=True)
class DisruptionEvent:
    """Represents a situational-awareness ingestion record."""

    event_type: str
    severity: float
    start_time: datetime | str
    end_time: datetime | str | None = None
    location: str | None = None
    source: str | None = None
    description: str | None = None
    created_at: datetime | str | None = None


def _normalize_timestamp(value: datetime | str) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return value


def ensure_disruption_events_table(conn: sqlite3.Connection) -> None:
    """Create the disruption events table if it does not exist."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS disruption_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            severity REAL NOT NULL CHECK (severity >= 0),
            start_time TEXT NOT NULL,
            end_time TEXT,
            location TEXT,
            source TEXT,
            description TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def insert_disruption_event(conn: sqlite3.Connection, event: DisruptionEvent) -> int:
    """Persist a disruption event and return the new row id."""

    ensure_disruption_events_table(conn)
    created_at = event.created_at or datetime.now(timezone.utc)
    row = conn.execute(
        """
        INSERT INTO disruption_events (
            event_type,
            severity,
            start_time,
            end_time,
            location,
            source,
            description,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event.event_type.strip().lower(),
            max(0.0, float(event.severity)),
            _normalize_timestamp(event.start_time),
            _normalize_timestamp(event.end_time) if event.end_time is not None else None,
            event.location,
            event.source,
            event.description,
            _normalize_timestamp(created_at),
        ),
    )
    conn.commit()
    return row.lastrowid  # type: ignore[attr-defined]


def summarize_disruption_severity(
    conn: sqlite3.Connection,
    *,
    since: datetime | str | None = None,
    event_types: Iterable[str] | None = None,
) -> dict[str, float]:
    """Return aggregated severity totals for disruption types since the given time."""

    ensure_disruption_events_table(conn)
    since_time = _normalize_since(since)
    types = _normalize_event_types(event_types)

    params: list[str] = [since_time]
    filter_clause = ""
    if types:
        placeholders = ",".join("?" for _ in types)
        filter_clause = f"AND event_type IN ({placeholders})"
        params.extend(sorted(types))

    query = f"""
    SELECT event_type, SUM(severity) as total_severity
    FROM disruption_events
    WHERE start_time >= ?
    {filter_clause}
    GROUP BY event_type
    """
    rows = conn.execute(query, params).fetchall()
    totals: dict[str, float] = {}
    for event_type, total in rows:
        totals[event_type] = float(total or 0.0)
    if types:
        for event_type in types:
            totals.setdefault(event_type, 0.0)
    return totals


def update_adaptive_policy_from_disruptions(
    conn: sqlite3.Connection,
    *,
    actor: str | None = None,
    approval_mode: str = "proposal",
    lookback: timedelta = timedelta(hours=DEFAULT_LOOKBACK_HOURS),
    max_delta: float = 0.1,
    weather_scale: float = 0.1,
    closure_scale: float = 0.12,
    traffic_scale: float = 0.08,
    description: str | None = None,
    note: str | None = None,
) -> Mapping[str, float | int | str]:
    """Create or apply bounded adaptive-policy changes using recent disruptions."""

    ensure_disruption_events_table(conn)
    ensure_adaptive_policy_defaults(conn)
    since = datetime.now(timezone.utc) - lookback
    summary = summarize_disruption_severity(
        conn, since=since, event_types=DEFAULT_EVENT_TYPES
    )
    weather_target = 1.0 + summary.get(EVENT_TYPE_WEATHER, 0.0) * weather_scale
    closure_target = 1.0 + summary.get(EVENT_TYPE_CLOSURE, 0.0) * closure_scale
    eta_target = 1.0 + summary.get(EVENT_TYPE_TRAFFIC, 0.0) * traffic_scale
    has_activity = any(value for value in summary.values())
    meta_description = description or (
        f"auto update from situational summary {summary}"
        if has_activity
        else "auto update from situational summary (no events)"
    )
    snapshot = load_adaptive_policy_snapshot(conn)
    bounded_targets = {
        "weather_risk_multiplier": min(
            weather_target,
            snapshot.weather_risk_multiplier + max_delta,
        ),
        "closure_delay_factor": min(
            closure_target,
            snapshot.closure_delay_factor + max_delta,
        ),
        "lane_eta_multiplier": min(
            eta_target,
            snapshot.lane_eta_multiplier + max_delta,
        ),
    }
    if approval_mode == "proposal":
        proposal = create_adaptive_policy_proposal(
            conn,
            proposal_type="situational_awareness",
            actor=str(actor or "").strip(),
            note=note or meta_description,
            source_summary={
                "lookbackHours": lookback.total_seconds() / 3600,
                "summary": summary,
            },
            items=[
                AdaptivePolicyProposalItem(
                    key=WEATHER_RISK_MULTIPLIER_KEY,
                    current_value=snapshot.weather_risk_multiplier,
                    proposed_value=float(bounded_targets["weather_risk_multiplier"]),
                    target_value=weather_target,
                    max_delta=max_delta,
                    description=f"weather {meta_description}",
                ),
                AdaptivePolicyProposalItem(
                    key=CLOSURE_DELAY_FACTOR_KEY,
                    current_value=snapshot.closure_delay_factor,
                    proposed_value=float(bounded_targets["closure_delay_factor"]),
                    target_value=closure_target,
                    max_delta=max_delta,
                    description=f"closure {meta_description}",
                ),
                AdaptivePolicyProposalItem(
                    key=LANE_ETA_MULTIPLIER_KEY,
                    current_value=snapshot.lane_eta_multiplier,
                    proposed_value=float(bounded_targets["lane_eta_multiplier"]),
                    target_value=eta_target,
                    max_delta=max_delta,
                    description=f"traffic {meta_description}",
                ),
            ],
        )
        return {
            "proposal_id": int(proposal["id"]),
            "status": str(proposal["status"]),
            "weather_risk_multiplier": float(bounded_targets["weather_risk_multiplier"]),
            "closure_delay_factor": float(bounded_targets["closure_delay_factor"]),
            "lane_eta_multiplier": float(bounded_targets["lane_eta_multiplier"]),
        }
    if approval_mode != "apply":
        raise ValueError("approval_mode must be either 'proposal' or 'apply'")
    weather_value = apply_bounded_parameter_target(
        conn,
        WEATHER_RISK_MULTIPLIER_KEY,
        weather_target,
        max_delta=max_delta,
        description=f"weather {meta_description}",
    )
    closure_value = apply_bounded_parameter_target(
        conn,
        CLOSURE_DELAY_FACTOR_KEY,
        closure_target,
        max_delta=max_delta,
        description=f"closure {meta_description}",
    )
    eta_value = apply_bounded_parameter_target(
        conn,
        LANE_ETA_MULTIPLIER_KEY,
        eta_target,
        max_delta=max_delta,
        description=f"traffic {meta_description}",
    )
    return {
        "weather_risk_multiplier": weather_value,
        "closure_delay_factor": closure_value,
        "lane_eta_multiplier": eta_value,
    }


def _normalize_event_types(event_types: Iterable[str] | None) -> set[str]:
    if event_types is None:
        return set()
    return {event_type.strip().lower() for event_type in event_types if event_type}


def _normalize_since(since: datetime | str | None) -> str:
    if since is None:
        since_time = datetime.now(timezone.utc) - timedelta(hours=DEFAULT_LOOKBACK_HOURS)
    else:
        since_time = since if isinstance(since, datetime) else _normalize_timestamp(since)
        if isinstance(since_time, str):
            return since_time
        since_time = since_time.astimezone(timezone.utc)
    return since_time.isoformat()
