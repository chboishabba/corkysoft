"""Reviewed-evidence promotion boundary for advisory operational inputs."""
from __future__ import annotations

import json
import sqlite3
from typing import Any

from .api_shared import ApiAuthContext, EVIDENCE_REVIEW_SCOPE
from .call_ops import _utc_now, ensure_call_ops_tables

PROMOTION_STATES = ("held", "accepted", "rejected")
PROMOTION_TARGETS = (
    "operational_note",
    "worker_time_event",
    "job_state",
    "customer_projection",
)


def ensure_evidence_promotion_tables(conn: sqlite3.Connection) -> None:
    """Create additive review records for advisory evidence."""

    ensure_call_ops_tables(conn)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS evidence_promotions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_type TEXT NOT NULL,
            source_id INTEGER NOT NULL,
            source_classification TEXT NOT NULL,
            proposed_target TEXT NOT NULL,
            proposed_action TEXT NOT NULL,
            proposed_payload_json TEXT NOT NULL DEFAULT '{}',
            state TEXT NOT NULL DEFAULT 'held',
            proposed_by TEXT NOT NULL,
            decided_by TEXT,
            decision_reason TEXT,
            decision_credential_id TEXT,
            decision_scopes_json TEXT,
            request_id TEXT,
            created_at TEXT NOT NULL,
            decided_at TEXT,
            FOREIGN KEY(source_id) REFERENCES call_transcript_artifacts(id) ON DELETE RESTRICT
        );
        CREATE INDEX IF NOT EXISTS idx_evidence_promotions_source
            ON evidence_promotions(source_type, source_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_evidence_promotions_state
            ON evidence_promotions(state, proposed_target, created_at DESC);
        CREATE TABLE IF NOT EXISTS evidence_promotion_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            promotion_id INTEGER NOT NULL,
            state TEXT NOT NULL,
            actor TEXT NOT NULL,
            reason TEXT,
            credential_id TEXT NOT NULL,
            scopes_json TEXT NOT NULL,
            request_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(promotion_id) REFERENCES evidence_promotions(id) ON DELETE CASCADE
        );
        """
    )
    columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(evidence_promotions)").fetchall()
    }
    if "target_id" not in columns:
        conn.execute("ALTER TABLE evidence_promotions ADD COLUMN target_id TEXT")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_evidence_promotions_identity
        ON evidence_promotions(
            source_type,
            source_id,
            proposed_target,
            proposed_action,
            target_id
        )
        """
    )
    conn.commit()


def _source_artifact(conn: sqlite3.Connection, source_id: int) -> sqlite3.Row:
    row = conn.execute(
        """
        SELECT id, status, data_classification, authority_class
        FROM call_transcript_artifacts WHERE id = ?
        """,
        (source_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Unknown transcript artifact: {source_id}")
    if row["status"] != "completed" or row["data_classification"] == "failed_artifact":
        raise ValueError("Only completed, non-failed evidence can be proposed for review")
    return row


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["sourceId"] = result.pop("source_id")
    result["sourceType"] = result.pop("source_type")
    result["sourceClassification"] = result.pop("source_classification")
    result["proposedTarget"] = result.pop("proposed_target")
    result["proposedAction"] = result.pop("proposed_action")
    result["proposedPayload"] = json.loads(result.pop("proposed_payload_json") or "{}")
    result["targetId"] = result.pop("target_id", None)
    result["proposedBy"] = result.pop("proposed_by")
    result["decidedBy"] = result.pop("decided_by")
    result["decisionReason"] = result.pop("decision_reason")
    result["decisionCredentialId"] = result.pop("decision_credential_id")
    result["decisionScopes"] = json.loads(result.pop("decision_scopes_json") or "[]")
    result["requestId"] = result.pop("request_id")
    result["createdAt"] = result.pop("created_at")
    result["decidedAt"] = result.pop("decided_at")
    return result


def _target_id_from_payload(payload: dict[str, Any]) -> str | None:
    for key in (
        "targetId",
        "target_id",
        "workerTimeEventId",
        "worker_time_event_id",
        "jobId",
        "job_id",
    ):
        value = payload.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def propose_evidence_promotion(
    conn: sqlite3.Connection,
    *,
    source_artifact_id: int,
    proposed_target: str,
    proposed_action: str,
    proposed_payload: dict[str, Any] | None,
    proposed_by: str,
    target_id: str | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Persist a held proposal without applying the proposed effect.

    Replaying the same immutable proposal identity is idempotent. Reusing that
    identity with a different payload fails closed instead of silently replacing
    the reviewed evidence.
    """

    ensure_evidence_promotion_tables(conn)
    if proposed_target not in PROMOTION_TARGETS:
        raise ValueError(f"Unsupported evidence promotion target: {proposed_target}")
    action = proposed_action.strip()
    if not action:
        raise ValueError("A proposed action is required")
    source = _source_artifact(conn, source_artifact_id)
    payload = proposed_payload or {}
    payload_json = json.dumps(payload, sort_keys=True)
    normalized_target_id = str(target_id).strip() if target_id not in (None, "") else None
    normalized_target_id = normalized_target_id or _target_id_from_payload(payload)
    conn.row_factory = sqlite3.Row
    existing = conn.execute(
        """
        SELECT * FROM evidence_promotions
        WHERE source_type = 'transcript_artifact'
          AND source_id = ?
          AND proposed_target = ?
          AND proposed_action = ?
          AND COALESCE(target_id, '') = COALESCE(?, '')
        ORDER BY id ASC
        LIMIT 1
        """,
        (source_artifact_id, proposed_target, action, normalized_target_id),
    ).fetchone()
    if existing is not None:
        current = _row_to_dict(existing)
        if (
            current["sourceClassification"] != source["data_classification"]
            or current["proposedPayload"] != payload
        ):
            raise ValueError("Evidence proposal identity conflicts with an existing payload")
        return current

    timestamp = _utc_now()
    cursor = conn.execute(
        """
        INSERT INTO evidence_promotions (
            source_type,
            source_id,
            source_classification,
            proposed_target,
            proposed_action,
            proposed_payload_json,
            target_id,
            state,
            proposed_by,
            request_id,
            created_at
        ) VALUES ('transcript_artifact', ?, ?, ?, ?, ?, ?, 'held', ?, ?, ?)
        """,
        (
            source_artifact_id,
            source["data_classification"],
            proposed_target,
            action,
            payload_json,
            normalized_target_id,
            proposed_by.strip() or "unknown",
            request_id.strip() if request_id else None,
            timestamp,
        ),
    )
    conn.commit()
    return get_evidence_promotion(conn, int(cursor.lastrowid))


def get_evidence_promotion(conn: sqlite3.Connection, promotion_id: int) -> dict[str, Any]:
    ensure_evidence_promotion_tables(conn)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM evidence_promotions WHERE id = ?",
        (promotion_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Unknown evidence promotion: {promotion_id}")
    return _row_to_dict(row)


def _decision_matches(
    current: dict[str, Any],
    *,
    state: str,
    actor: str,
    credential_id: str,
    scopes: tuple[str, ...],
    request_id: str,
    reason: str,
) -> bool:
    return (
        current["state"] == state
        and current["decidedBy"] == actor
        and current["decisionCredentialId"] == credential_id
        and tuple(current["decisionScopes"]) == tuple(scopes)
        and current["requestId"] == request_id
        and current["decisionReason"] == reason
    )


def decide_evidence_promotion(
    conn: sqlite3.Connection,
    *,
    promotion_id: int,
    state: str,
    actor: str,
    credential_id: str,
    scopes: tuple[str, ...],
    request_id: str,
    reason: str | None = None,
) -> dict[str, Any]:
    """Record an actor-bound scoped decision without applying an effect."""

    if state not in PROMOTION_STATES:
        raise ValueError("Evidence promotion decision must be held, accepted, or rejected")
    if EVIDENCE_REVIEW_SCOPE not in scopes:
        raise PermissionError("Credential scope is not authorized for evidence review")
    normalized_actor = actor.strip()
    normalized_credential = credential_id.strip()
    normalized_request = request_id.strip()
    normalized_reason = str(reason or "").strip()
    if not normalized_actor or not normalized_credential or not normalized_request:
        raise ValueError("Actor, credential, and request identity are required")
    if not normalized_reason:
        raise ValueError("An evidence review reason is required")

    current = get_evidence_promotion(conn, promotion_id)
    if current["state"] != "held":
        if _decision_matches(
            current,
            state=state,
            actor=normalized_actor,
            credential_id=normalized_credential,
            scopes=scopes,
            request_id=normalized_request,
            reason=normalized_reason,
        ):
            return current
        raise ValueError("Accepted or rejected evidence decisions are terminal")
    if _decision_matches(
        current,
        state=state,
        actor=normalized_actor,
        credential_id=normalized_credential,
        scopes=scopes,
        request_id=normalized_request,
        reason=normalized_reason,
    ):
        return current

    timestamp = _utc_now()
    conn.execute(
        """
        INSERT INTO evidence_promotion_decisions (
            promotion_id,
            state,
            actor,
            reason,
            credential_id,
            scopes_json,
            request_id,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            promotion_id,
            state,
            normalized_actor,
            normalized_reason,
            normalized_credential,
            json.dumps(list(scopes)),
            normalized_request,
            timestamp,
        ),
    )
    cursor = conn.execute(
        """
        UPDATE evidence_promotions
        SET state = ?,
            decided_by = ?,
            decision_reason = ?,
            decision_credential_id = ?,
            decision_scopes_json = ?,
            request_id = ?,
            decided_at = ?
        WHERE id = ? AND state = 'held'
        """,
        (
            state,
            normalized_actor,
            normalized_reason,
            normalized_credential,
            json.dumps(list(scopes)),
            normalized_request,
            timestamp,
            promotion_id,
        ),
    )
    if cursor.rowcount != 1:
        conn.rollback()
        raise ValueError("Evidence decision lost a concurrent state race")
    conn.commit()
    return get_evidence_promotion(conn, promotion_id)


def decide_evidence(
    conn: sqlite3.Connection,
    *,
    promotion_id: int,
    decision: str,
    auth: ApiAuthContext,
    reason: str,
) -> dict[str, Any]:
    """Apply a decision using an authenticated, actor-bound context."""

    return decide_evidence_promotion(
        conn,
        promotion_id=promotion_id,
        state=decision,
        actor=auth.actor,
        credential_id=auth.credential_id,
        scopes=auth.scopes,
        request_id=auth.request_id,
        reason=reason,
    )


def hold_evidence(
    conn: sqlite3.Connection,
    *,
    promotion_id: int,
    auth: ApiAuthContext,
    reason: str,
) -> dict[str, Any]:
    """Record a reviewed hold without granting authority."""

    return decide_evidence(
        conn,
        promotion_id=promotion_id,
        decision="held",
        auth=auth,
        reason=reason,
    )


def require_accepted_evidence(
    conn: sqlite3.Connection,
    *,
    promotion_id: int,
    proposed_action: str | None = None,
    proposed_target: str | None = None,
    target_type: str | None = None,
    target_id: str | None = None,
) -> dict[str, Any]:
    """Return exactly-bound accepted evidence or fail before a write."""

    promotion = get_evidence_promotion(conn, promotion_id)
    if promotion["state"] != "accepted":
        raise PermissionError("Evidence has not been accepted for promotion")
    if proposed_action is not None and promotion["proposedAction"] != proposed_action:
        raise PermissionError("Evidence was accepted for a different action")
    expected_target = proposed_target or target_type
    if expected_target is not None and promotion["proposedTarget"] != expected_target:
        raise PermissionError("Evidence was accepted for a different target type")
    if target_id is not None and str(promotion.get("targetId") or "") != str(target_id):
        raise PermissionError("Evidence was accepted for a different target")
    return promotion


__all__ = [
    "PROMOTION_STATES",
    "PROMOTION_TARGETS",
    "decide_evidence",
    "decide_evidence_promotion",
    "ensure_evidence_promotion_tables",
    "get_evidence_promotion",
    "hold_evidence",
    "propose_evidence_promotion",
    "require_accepted_evidence",
]
