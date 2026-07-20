from __future__ import annotations

import sqlite3

import pytest

from corkysoft.api_shared import ApiAuthContext, EVIDENCE_REVIEW_SCOPE
from corkysoft.call_ops import create_call_event, record_transcript_artifact
from corkysoft.evidence_promotion import (
    decide_evidence,
    decide_evidence_promotion,
    hold_evidence,
    propose_evidence_promotion,
    require_accepted_evidence,
)


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def _artifact(conn: sqlite3.Connection) -> dict:
    event = create_call_event(
        conn,
        event_kind="ops_call",
        direction="internal",
        status="completed",
        source_channel="manual_note",
        title="Evidence test",
        operator_id="operator@example.test",
    )
    return record_transcript_artifact(
        conn,
        call_event_id=event["id"],
        service_key="test",
        status="completed",
        transcript_text="Worker confirmed the corrected time entry.",
        is_final=True,
    )


def _reviewer(*, scopes: tuple[str, ...] = (EVIDENCE_REVIEW_SCOPE,)) -> ApiAuthContext:
    return ApiAuthContext(
        credential_id="reviewer-1",
        actor="ops-reviewer@example.test",
        scopes=scopes,
        request_id="req-review-1",
    )


def _proposal(conn: sqlite3.Connection) -> dict:
    artifact = _artifact(conn)
    return propose_evidence_promotion(
        conn,
        source_artifact_id=artifact["id"],
        proposed_target="worker_time_event",
        proposed_action="worker_time.adjust",
        proposed_payload={"minutes": 90, "targetId": "42"},
        proposed_by="transcript-model",
        request_id="req-proposal-1",
    )


def test_advisory_evidence_is_held_and_cannot_authorize_write() -> None:
    conn = _conn()
    proposal = _proposal(conn)

    assert proposal["state"] == "held"
    with pytest.raises(PermissionError, match="not been accepted"):
        require_accepted_evidence(
            conn,
            promotion_id=proposal["id"],
            proposed_action="worker_time.adjust",
            proposed_target="worker_time_event",
            target_id="42",
        )


def test_proposal_replay_is_idempotent_and_payload_conflicts_fail_closed() -> None:
    conn = _conn()
    artifact = _artifact(conn)
    kwargs = {
        "source_artifact_id": artifact["id"],
        "proposed_target": "worker_time_event",
        "proposed_action": "worker_time.adjust",
        "proposed_payload": {"minutes": 90, "targetId": "42"},
        "proposed_by": "transcript-model",
        "request_id": "req-proposal-1",
    }

    first = propose_evidence_promotion(conn, **kwargs)
    second = propose_evidence_promotion(conn, **kwargs)

    assert first["id"] == second["id"]
    assert conn.execute("SELECT COUNT(*) FROM evidence_promotions").fetchone()[0] == 1

    with pytest.raises(ValueError, match="conflicts"):
        propose_evidence_promotion(
            conn,
            **{**kwargs, "proposed_payload": {"minutes": 120, "targetId": "42"}},
        )


def test_scoped_reviewer_accepts_and_binds_exact_action_and_target() -> None:
    conn = _conn()
    proposal = _proposal(conn)

    accepted = decide_evidence(
        conn,
        promotion_id=proposal["id"],
        decision="accepted",
        auth=_reviewer(),
        reason="Matched the roster and call recording.",
    )
    promoted = require_accepted_evidence(
        conn,
        promotion_id=proposal["id"],
        proposed_action="worker_time.adjust",
        proposed_target="worker_time_event",
        target_id="42",
    )

    assert accepted["state"] == "accepted"
    assert accepted["decidedBy"] == "ops-reviewer@example.test"
    assert promoted["proposedPayload"]["minutes"] == 90

    with pytest.raises(PermissionError, match="different action"):
        require_accepted_evidence(
            conn,
            promotion_id=proposal["id"],
            proposed_action="customer_status.publish",
        )
    with pytest.raises(PermissionError, match="different target"):
        require_accepted_evidence(
            conn,
            promotion_id=proposal["id"],
            target_id="43",
        )


def test_unscoped_actor_cannot_decide_even_when_calling_domain_function() -> None:
    conn = _conn()
    proposal = _proposal(conn)

    with pytest.raises(PermissionError, match="not authorized"):
        decide_evidence_promotion(
            conn,
            promotion_id=proposal["id"],
            state="accepted",
            actor="unscoped@example.test",
            credential_id="unscoped-1",
            scopes=("worker_time:write",),
            request_id="req-unscoped",
            reason="Attempted bypass",
        )


def test_terminal_decision_replay_is_idempotent_but_change_is_rejected() -> None:
    conn = _conn()
    proposal = _proposal(conn)
    auth = _reviewer()

    first = decide_evidence(
        conn,
        promotion_id=proposal["id"],
        decision="rejected",
        auth=auth,
        reason="Speaker identity was ambiguous.",
    )
    replay = decide_evidence(
        conn,
        promotion_id=proposal["id"],
        decision="rejected",
        auth=auth,
        reason="Speaker identity was ambiguous.",
    )

    assert first == replay
    assert conn.execute("SELECT COUNT(*) FROM evidence_promotion_decisions").fetchone()[0] == 1
    with pytest.raises(ValueError, match="terminal"):
        decide_evidence(
            conn,
            promotion_id=proposal["id"],
            decision="accepted",
            auth=auth,
            reason="Attempted second decision.",
        )


def test_reviewed_hold_is_idempotent_and_does_not_grant_authority() -> None:
    conn = _conn()
    proposal = _proposal(conn)
    auth = _reviewer()

    held = hold_evidence(
        conn,
        promotion_id=proposal["id"],
        auth=auth,
        reason="Waiting for payroll source data.",
    )
    replay = hold_evidence(
        conn,
        promotion_id=proposal["id"],
        auth=auth,
        reason="Waiting for payroll source data.",
    )

    assert held == replay
    assert held["state"] == "held"
    assert conn.execute("SELECT COUNT(*) FROM evidence_promotion_decisions").fetchone()[0] == 1
    with pytest.raises(PermissionError):
        require_accepted_evidence(conn, promotion_id=proposal["id"])
