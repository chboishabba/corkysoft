from __future__ import annotations

import sqlite3

from analytics.adaptive_policy import (
    ADAPTIVE_POLICY_DEFAULTS,
    LANE_RATE_PER_M3_KEY,
    SEASONAL_MARGIN_UPLIFT_KEY,
    WEATHER_RISK_MULTIPLIER_KEY,
    AdaptivePolicyProposalItem,
    approve_adaptive_policy_proposal,
    apply_bounded_parameter_target,
    apply_adaptive_policy_proposal,
    create_adaptive_policy_proposal,
    ensure_adaptive_policy_defaults,
    list_adaptive_policy_proposals,
    load_adaptive_policy_snapshot,
    reject_adaptive_policy_proposal,
)


def build_conn() -> sqlite3.Connection:
    return sqlite3.connect(":memory:")


def test_ensure_adaptive_policy_defaults_bootstraps_parameters() -> None:
    conn = build_conn()

    ensure_adaptive_policy_defaults(conn)

    rows = conn.execute(
        "SELECT key, value_numeric FROM global_parameters ORDER BY key"
    ).fetchall()

    assert len(rows) == len(ADAPTIVE_POLICY_DEFAULTS)
    assert dict(rows)[LANE_RATE_PER_M3_KEY] == 1.0
    assert dict(rows)[SEASONAL_MARGIN_UPLIFT_KEY] == 0.0


def test_load_adaptive_policy_snapshot_returns_default_state() -> None:
    conn = build_conn()

    snapshot = load_adaptive_policy_snapshot(conn)

    assert snapshot.lane_rate_per_m3 == 1.0
    assert snapshot.lane_eta_multiplier == 1.0
    assert snapshot.weather_risk_multiplier == 1.0
    assert snapshot.closure_delay_factor == 1.0
    assert snapshot.truck_efficiency_score == 1.0
    assert snapshot.driver_efficiency_score == 1.0
    assert snapshot.seasonal_margin_uplift == 0.0


def test_apply_bounded_parameter_target_caps_positive_delta() -> None:
    conn = build_conn()
    ensure_adaptive_policy_defaults(conn)

    updated = apply_bounded_parameter_target(
        conn,
        LANE_RATE_PER_M3_KEY,
        1.4,
        max_delta=0.1,
    )

    assert updated == 1.1


def test_apply_bounded_parameter_target_caps_negative_delta_and_min_value() -> None:
    conn = build_conn()
    ensure_adaptive_policy_defaults(conn)

    updated = apply_bounded_parameter_target(
        conn,
        SEASONAL_MARGIN_UPLIFT_KEY,
        -0.5,
        max_delta=0.25,
        min_value=0.0,
    )

    assert updated == 0.0


def test_create_and_list_adaptive_policy_proposal_keeps_active_state_unchanged() -> None:
    conn = build_conn()
    ensure_adaptive_policy_defaults(conn)

    proposal = create_adaptive_policy_proposal(
        conn,
        proposal_type="situational_awareness",
        actor="ops_manager",
        items=[
            AdaptivePolicyProposalItem(
                key=WEATHER_RISK_MULTIPLIER_KEY,
                current_value=1.0,
                proposed_value=1.1,
                target_value=1.2,
                max_delta=0.1,
                description="weather proposal",
            )
        ],
    )

    active_value = conn.execute(
        "SELECT value_numeric FROM global_parameters WHERE key = ?",
        (WEATHER_RISK_MULTIPLIER_KEY,),
    ).fetchone()
    proposals = list_adaptive_policy_proposals(conn)

    assert proposal["status"] == "pending_review"
    assert active_value is not None and active_value[0] == 1.0
    assert len(proposals) == 1
    assert proposals[0]["items"][0]["proposed_value"] == 1.1


def test_adaptive_policy_proposal_requires_approval_before_apply() -> None:
    conn = build_conn()
    ensure_adaptive_policy_defaults(conn)
    proposal = create_adaptive_policy_proposal(
        conn,
        proposal_type="manual",
        actor="ops_manager",
        items=[
            AdaptivePolicyProposalItem(
                key=WEATHER_RISK_MULTIPLIER_KEY,
                current_value=1.0,
                proposed_value=1.08,
                target_value=1.08,
                max_delta=0.08,
            )
        ],
    )

    try:
        apply_adaptive_policy_proposal(conn, proposal_id=int(proposal["id"]), actor="admin")
    except ValueError as exc:
        assert "approved before apply" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected apply to fail without approval")


def test_approved_adaptive_policy_proposal_updates_live_parameters() -> None:
    conn = build_conn()
    ensure_adaptive_policy_defaults(conn)
    proposal = create_adaptive_policy_proposal(
        conn,
        proposal_type="manual",
        actor="ops_manager",
        items=[
            AdaptivePolicyProposalItem(
                key=WEATHER_RISK_MULTIPLIER_KEY,
                current_value=1.0,
                proposed_value=1.08,
                target_value=1.08,
                max_delta=0.08,
                description="manual weather nudge",
            )
        ],
    )

    approve_adaptive_policy_proposal(conn, proposal_id=int(proposal["id"]), actor="commercial_owner")
    applied = apply_adaptive_policy_proposal(conn, proposal_id=int(proposal["id"]), actor="admin")
    row = conn.execute(
        "SELECT value_numeric FROM global_parameters WHERE key = ?",
        (WEATHER_RISK_MULTIPLIER_KEY,),
    ).fetchone()

    assert applied["status"] == "applied"
    assert row is not None and row[0] == 1.08


def test_rejected_adaptive_policy_proposal_records_rejection() -> None:
    conn = build_conn()
    ensure_adaptive_policy_defaults(conn)
    proposal = create_adaptive_policy_proposal(
        conn,
        proposal_type="manual",
        actor="ops_manager",
        items=[
            AdaptivePolicyProposalItem(
                key=WEATHER_RISK_MULTIPLIER_KEY,
                current_value=1.0,
                proposed_value=1.04,
                target_value=1.04,
                max_delta=0.04,
            )
        ],
    )

    rejected = reject_adaptive_policy_proposal(
        conn,
        proposal_id=int(proposal["id"]),
        actor="commercial_owner",
        note="Need more evidence.",
    )

    assert rejected["status"] == "rejected"
    assert rejected["rejected_by"] == "commercial_owner"
