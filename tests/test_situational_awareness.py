from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from analytics.situational_awareness import (
    DisruptionEvent,
    EVENT_TYPE_CLOSURE,
    EVENT_TYPE_TRAFFIC,
    EVENT_TYPE_WEATHER,
    insert_disruption_event,
    summarize_disruption_severity,
    update_adaptive_policy_from_disruptions,
)


def build_conn() -> sqlite3.Connection:
    return sqlite3.connect(":memory:")


def test_summarize_disruption_severity_aggregates_by_type() -> None:
    conn = build_conn()
    now = datetime.now(timezone.utc)
    insert_disruption_event(
        conn,
        DisruptionEvent(
            event_type=EVENT_TYPE_WEATHER,
            severity=0.4,
            start_time=now - timedelta(hours=1),
        ),
    )
    insert_disruption_event(
        conn,
        DisruptionEvent(
            event_type=EVENT_TYPE_CLOSURE,
            severity=0.2,
            start_time=now - timedelta(hours=2),
        ),
    )
    insert_disruption_event(
        conn,
        DisruptionEvent(
            event_type=EVENT_TYPE_TRAFFIC,
            severity=0.1,
            start_time=now - timedelta(minutes=30),
        ),
    )

    summary = summarize_disruption_severity(
        conn,
        since=now - timedelta(hours=3),
        event_types=[EVENT_TYPE_WEATHER, EVENT_TYPE_CLOSURE],
    )

    assert summary[EVENT_TYPE_WEATHER] == 0.4
    assert summary[EVENT_TYPE_CLOSURE] == 0.2
    assert EVENT_TYPE_TRAFFIC not in summary


def test_update_adaptive_policy_from_disruptions_applies_bounded_updates() -> None:
    conn = build_conn()
    now = datetime.now(timezone.utc)
    insert_disruption_event(
        conn,
        DisruptionEvent(
            event_type=EVENT_TYPE_WEATHER,
            severity=0.4,
            start_time=now - timedelta(hours=1),
        ),
    )
    insert_disruption_event(
        conn,
        DisruptionEvent(
            event_type=EVENT_TYPE_CLOSURE,
            severity=0.3,
            start_time=now - timedelta(hours=2),
        ),
    )
    insert_disruption_event(
        conn,
        DisruptionEvent(
            event_type=EVENT_TYPE_TRAFFIC,
            severity=0.2,
            start_time=now - timedelta(minutes=10),
        ),
    )

    result = update_adaptive_policy_from_disruptions(
        conn,
        actor="ops_manager",
        approval_mode="proposal",
        lookback=timedelta(hours=4),
        max_delta=0.5,
        weather_scale=0.2,
        closure_scale=0.15,
        traffic_scale=0.12,
    )

    assert result["proposal_id"] > 0
    assert result["status"] == "pending_review"
    assert result["weather_risk_multiplier"] == pytest.approx(1.08)
    assert result["closure_delay_factor"] == pytest.approx(1.045)
    assert result["lane_eta_multiplier"] == pytest.approx(1.024)

    row = conn.execute(
        "SELECT key, value_numeric FROM global_parameters WHERE key = ?",
        ("adaptive.weather_risk_multiplier",),
    ).fetchone()
    assert row is not None and row[1] == pytest.approx(1.0)
