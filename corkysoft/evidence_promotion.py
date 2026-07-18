"""Reviewed-evidence promotion boundary for advisory operational inputs."""
from __future__ import annotations

import json
import sqlite3
from typing import Any

from .call_ops import _utc_now, ensure_call_ops_tables

PROMOTION_STATES = ("held", "accepted", "rejected")
PROMOTION_TARGETS = (
    "operational_note",
    "worker_time_event",
    "job_state",
    "customer_projection",
)


def ensure_evidence_promotion_tables(conn: sqlite3.Connection) -> None:
    """Create additive, append-only review records for advisory evidence."""

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
    result["proposedBy"] = result.pop("proposed_by")
    result["decidedBy"] = result.pop("decided_by")
    result["decisionReason"] = result.pop("decision_reason")
    result["decisionCredentialId"] = result.pop("decision_credential_id")
    result["decisionScopes"] = json.loads(result.pop("decision_scopes_json") or "[]")
    result["requestId"] = result.pop("request_id")
    result["createdAt"] = result.pop("created_at")
    result["decidedAt"] = result.pop("decided_at")
    return result


def propose_evidence_promotion(
    conn: sqlite3.Connection,
    *,
    source_artifact_id: int,
    proposed_target: str,
    proposed_action: str,
    proposed_payload: dict[str, Any] | None,
    proposed_by: str,
) -> dict[str, Any]:
    """Persist a held proposal; this function never applies the proposed effect."""

    ensure_evidence_promotion_tables(conn)
    if proposed_target not in PROMOTION_TARGETS:
        raise ValueError(f"Unsupported evidence promotion target: {proposed_target}")
    if not proposed_action.strip():
        raise ValueError("A proposed action is required")
    source = _source_artifact(conn, source_artifact_id)
    timestamp = _utc_now()
    cursor = conn.execute(
        """
        INSERT INTO evidence_promotions (
            source_type, source_id, source_classification, proposed_target,
            proposed_action, proposed_payload_json, state, proposed_by, created_at
        ) VALUES ('transcript_artifact', ?, ?, ?, ?, ?, 'held', ?, ?)
        """,
        (
            source_artifact_id,
            source["data_classification"],
            proposed_target,
            proposed_action.strip(),
            json.dumps(proposed_payload or {}, sort_keys=True),
            proposed_by.strip() or "unknown",
            timestamp,
        ),
    )
    conn.commit()
    return get_evidence_promotion(conn, int(cursor.lastrowid))


def get_evidence_promotion(conn: sqlite3.Connection, promotion_id: int) -> dict[str, Any]:
    ensure_evidence_promotion_tables(conn)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM evidence_promotions WHERE id = ?", (promotion_id,)).fetchone()
    if row is None:
        raise ValueError(f"Unknown evidence promotion: {promotion_id}")
    return _row_to_dict(row)


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
    """Record a scoped human decision without applying an operational effect."""

    if state not in PROMOTION_STATES:
        raise ValueError("Evidence promotion decision must be held, accepted, or rejected")
    current = get_evidence_promotion(conn, promotion_id)
    timestamp = _utc_now()
    conn.execute(
        """
        INSERT INTO evidence_promotion_decisions (
            promotion_id, state, actor, reason, credential_id, scopes_json, request_id, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (promotion_id, state, actor, reason, credential_id, json.dumps(list(scopes)), request_id, timestamp),
    )
    conn.execute(
        """
        UPDATE evidence_promotions
        SET state = ?, decided_by = ?, decision_reason = ?, decision_credential_id = ?,
            decision_scopes_json = ?, request_id = ?, decided_at = ?
        WHERE id = ?
        """,
        (state, actor, reason, credential_id, json.dumps(list(scopes)), request_id, timestamp, promotion_id),
    )
    conn.commit()
    return get_evidence_promotion(conn, int(current["id"]))
