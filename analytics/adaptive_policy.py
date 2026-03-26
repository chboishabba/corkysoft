"""Helpers for the bounded adaptive policy parameter state."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Mapping, Sequence

from .db.parameters import (
    bootstrap_parameters,
    ensure_global_parameters_table,
    get_parameter_value,
    set_parameter_value,
)


LANE_RATE_PER_M3_KEY = "adaptive.lane_rate_per_m3"
LANE_ETA_MULTIPLIER_KEY = "adaptive.lane_eta_multiplier"
WEATHER_RISK_MULTIPLIER_KEY = "adaptive.weather_risk_multiplier"
CLOSURE_DELAY_FACTOR_KEY = "adaptive.closure_delay_factor"
TRUCK_EFFICIENCY_SCORE_KEY = "adaptive.truck_efficiency_score"
DRIVER_EFFICIENCY_SCORE_KEY = "adaptive.driver_efficiency_score"
SEASONAL_MARGIN_UPLIFT_KEY = "adaptive.seasonal_margin_uplift"
PROPOSAL_STATUS_PENDING_REVIEW = "pending_review"
PROPOSAL_STATUS_APPROVED = "approved"
PROPOSAL_STATUS_REJECTED = "rejected"
PROPOSAL_STATUS_APPLIED = "applied"

ADAPTIVE_POLICY_DEFAULTS: tuple[tuple[str, float, str], ...] = (
    (
        LANE_RATE_PER_M3_KEY,
        1.0,
        "Relative lane pricing multiplier used by adaptive policy review.",
    ),
    (
        LANE_ETA_MULTIPLIER_KEY,
        1.0,
        "Relative ETA multiplier used by adaptive policy review.",
    ),
    (
        WEATHER_RISK_MULTIPLIER_KEY,
        1.0,
        "Relative weather risk multiplier used by adaptive policy review.",
    ),
    (
        CLOSURE_DELAY_FACTOR_KEY,
        1.0,
        "Relative road-closure delay factor used by adaptive policy review.",
    ),
    (
        TRUCK_EFFICIENCY_SCORE_KEY,
        1.0,
        "Relative truck efficiency score used by adaptive policy review.",
    ),
    (
        DRIVER_EFFICIENCY_SCORE_KEY,
        1.0,
        "Relative driver efficiency score used by adaptive policy review.",
    ),
    (
        SEASONAL_MARGIN_UPLIFT_KEY,
        0.0,
        "Seasonal margin uplift used by adaptive policy review.",
    ),
)


@dataclass(frozen=True)
class AdaptivePolicySnapshot:
    """Current adaptive-policy parameter state."""

    lane_rate_per_m3: float
    lane_eta_multiplier: float
    weather_risk_multiplier: float
    closure_delay_factor: float
    truck_efficiency_score: float
    driver_efficiency_score: float
    seasonal_margin_uplift: float


@dataclass(frozen=True)
class AdaptivePolicyProposalItem:
    """One proposed adaptive-policy parameter change."""

    key: str
    current_value: float
    proposed_value: float
    target_value: float
    min_value: float | None = 0.0
    max_value: float | None = None
    max_delta: float = 0.1
    description: str | None = None


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def ensure_adaptive_policy_defaults(conn: sqlite3.Connection) -> None:
    """Ensure the adaptive-policy defaults exist in ``global_parameters``."""

    ensure_global_parameters_table(conn)
    bootstrap_parameters(conn, ADAPTIVE_POLICY_DEFAULTS)


def ensure_adaptive_policy_governance_tables(conn: sqlite3.Connection) -> None:
    """Ensure proposal and review tables for adaptive-policy governance exist."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS adaptive_policy_proposals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            proposal_type TEXT NOT NULL,
            status TEXT NOT NULL,
            requested_by TEXT NOT NULL,
            request_note TEXT,
            source_summary TEXT,
            created_at TEXT NOT NULL,
            approved_by TEXT,
            approval_note TEXT,
            approved_at TEXT,
            rejected_by TEXT,
            rejection_note TEXT,
            rejected_at TEXT,
            applied_by TEXT,
            applied_note TEXT,
            applied_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS adaptive_policy_proposal_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            proposal_id INTEGER NOT NULL,
            parameter_key TEXT NOT NULL,
            current_value REAL NOT NULL,
            proposed_value REAL NOT NULL,
            target_value REAL NOT NULL,
            min_value REAL,
            max_value REAL,
            max_delta REAL NOT NULL,
            description TEXT,
            FOREIGN KEY(proposal_id) REFERENCES adaptive_policy_proposals(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_adaptive_policy_proposals_status_created
        ON adaptive_policy_proposals(status, created_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_adaptive_policy_proposal_items_proposal
        ON adaptive_policy_proposal_items(proposal_id, parameter_key)
        """
    )
    conn.commit()


def load_adaptive_policy_snapshot(conn: sqlite3.Connection) -> AdaptivePolicySnapshot:
    """Return the current adaptive-policy state."""

    ensure_adaptive_policy_defaults(conn)
    return AdaptivePolicySnapshot(
        lane_rate_per_m3=float(get_parameter_value(conn, LANE_RATE_PER_M3_KEY, 1.0) or 1.0),
        lane_eta_multiplier=float(
            get_parameter_value(conn, LANE_ETA_MULTIPLIER_KEY, 1.0) or 1.0
        ),
        weather_risk_multiplier=float(
            get_parameter_value(conn, WEATHER_RISK_MULTIPLIER_KEY, 1.0) or 1.0
        ),
        closure_delay_factor=float(
            get_parameter_value(conn, CLOSURE_DELAY_FACTOR_KEY, 1.0) or 1.0
        ),
        truck_efficiency_score=float(
            get_parameter_value(conn, TRUCK_EFFICIENCY_SCORE_KEY, 1.0) or 1.0
        ),
        driver_efficiency_score=float(
            get_parameter_value(conn, DRIVER_EFFICIENCY_SCORE_KEY, 1.0) or 1.0
        ),
        seasonal_margin_uplift=float(
            get_parameter_value(conn, SEASONAL_MARGIN_UPLIFT_KEY, 0.0) or 0.0
        ),
    )


def _bounded_parameter_value(
    current_value: float,
    target_value: float,
    *,
    max_delta: float = 0.1,
    min_value: float | None = 0.0,
    max_value: float | None = None,
) -> float:
    if max_delta < 0:
        raise ValueError("max_delta must be non-negative")

    current = float(current_value)
    target = float(target_value)
    delta = target - current

    if delta > max_delta:
        new_value = current + max_delta
    elif delta < -max_delta:
        new_value = current - max_delta
    else:
        new_value = target

    if min_value is not None:
        new_value = max(min_value, new_value)
    if max_value is not None:
        new_value = min(max_value, new_value)
    return float(new_value)


def apply_bounded_parameter_target(
    conn: sqlite3.Connection,
    key: str,
    target_value: float,
    *,
    max_delta: float = 0.1,
    min_value: float | None = 0.0,
    max_value: float | None = None,
    description: str | None = None,
) -> float:
    """Move one adaptive parameter toward ``target_value`` within a bounded step."""

    ensure_adaptive_policy_defaults(conn)
    current_value = float(get_parameter_value(conn, key, 0.0) or 0.0)
    new_value = _bounded_parameter_value(
        current_value,
        target_value,
        max_delta=max_delta,
        min_value=min_value,
        max_value=max_value,
    )

    set_parameter_value(conn, key, float(new_value), description)
    return float(new_value)


def create_adaptive_policy_proposal(
    conn: sqlite3.Connection,
    *,
    proposal_type: str,
    actor: str,
    items: Sequence[AdaptivePolicyProposalItem],
    note: str | None = None,
    source_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a proposal for later review before changing active policy."""

    ensure_adaptive_policy_defaults(conn)
    ensure_adaptive_policy_governance_tables(conn)
    actor_name = str(actor).strip()
    if not actor_name:
        raise ValueError("Adaptive-policy proposal actor is required.")
    if not items:
        raise ValueError("Adaptive-policy proposal requires at least one parameter change.")

    created_at = _utc_now_iso()
    cursor = conn.execute(
        """
        INSERT INTO adaptive_policy_proposals (
            proposal_type,
            status,
            requested_by,
            request_note,
            source_summary,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            proposal_type.strip() or "manual",
            PROPOSAL_STATUS_PENDING_REVIEW,
            actor_name,
            note,
            json.dumps(source_summary or {}, sort_keys=True),
            created_at,
        ),
    )
    proposal_id = int(cursor.lastrowid)
    conn.executemany(
        """
        INSERT INTO adaptive_policy_proposal_items (
            proposal_id,
            parameter_key,
            current_value,
            proposed_value,
            target_value,
            min_value,
            max_value,
            max_delta,
            description
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                proposal_id,
                item.key,
                float(item.current_value),
                float(item.proposed_value),
                float(item.target_value),
                item.min_value,
                item.max_value,
                float(item.max_delta),
                item.description,
            )
            for item in items
        ],
    )
    conn.commit()
    return get_adaptive_policy_proposal(conn, proposal_id)


def _proposal_row_to_dict(row: sqlite3.Row | tuple[Any, ...], columns: Sequence[str]) -> dict[str, Any]:
    if isinstance(row, sqlite3.Row):
        return {column: row[column] for column in columns}
    return dict(zip(columns, row, strict=False))


def get_adaptive_policy_proposal(conn: sqlite3.Connection, proposal_id: int) -> dict[str, Any]:
    """Return one adaptive-policy proposal with child items."""

    ensure_adaptive_policy_governance_tables(conn)
    proposal_cursor = conn.execute(
        """
        SELECT
            id,
            proposal_type,
            status,
            requested_by,
            request_note,
            source_summary,
            created_at,
            approved_by,
            approval_note,
            approved_at,
            rejected_by,
            rejection_note,
            rejected_at,
            applied_by,
            applied_note,
            applied_at
        FROM adaptive_policy_proposals
        WHERE id = ?
        """,
        (int(proposal_id),),
    )
    proposal_row = proposal_cursor.fetchone()
    if proposal_row is None:
        raise ValueError(f"Unknown adaptive-policy proposal: {proposal_id}")
    proposal_columns = [column[0] for column in proposal_cursor.description or []]
    payload = _proposal_row_to_dict(proposal_row, proposal_columns)
    item_cursor = conn.execute(
        """
        SELECT
            parameter_key,
            current_value,
            proposed_value,
            target_value,
            min_value,
            max_value,
            max_delta,
            description
        FROM adaptive_policy_proposal_items
        WHERE proposal_id = ?
        ORDER BY parameter_key
        """,
        (int(proposal_id),),
    )
    item_columns = [column[0] for column in item_cursor.description or []]
    items = [
        _proposal_row_to_dict(row, item_columns)
        for row in item_cursor.fetchall()
    ]
    payload["items"] = items
    payload["source_summary"] = json.loads(payload.get("source_summary") or "{}")
    return payload


def list_adaptive_policy_proposals(
    conn: sqlite3.Connection,
    *,
    limit: int = 25,
    status: str | None = None,
) -> list[dict[str, Any]]:
    """Return recent adaptive-policy proposals."""

    ensure_adaptive_policy_governance_tables(conn)
    query = """
        SELECT id
        FROM adaptive_policy_proposals
    """
    params: list[Any] = []
    if status:
        query += " WHERE status = ?"
        params.append(status)
    query += " ORDER BY created_at DESC, id DESC LIMIT ?"
    params.append(int(limit))
    rows = conn.execute(query, params).fetchall()
    proposal_ids = [int(row[0]) for row in rows]
    return [get_adaptive_policy_proposal(conn, proposal_id) for proposal_id in proposal_ids]


def approve_adaptive_policy_proposal(
    conn: sqlite3.Connection,
    *,
    proposal_id: int,
    actor: str,
    note: str | None = None,
) -> dict[str, Any]:
    """Mark a pending adaptive-policy proposal as approved."""

    proposal = get_adaptive_policy_proposal(conn, proposal_id)
    actor_name = str(actor).strip()
    if not actor_name:
        raise ValueError("Adaptive-policy approval actor is required.")
    if proposal["status"] != PROPOSAL_STATUS_PENDING_REVIEW:
        raise ValueError("Only pending adaptive-policy proposals can be approved.")
    conn.execute(
        """
        UPDATE adaptive_policy_proposals
        SET
            status = ?,
            approved_by = ?,
            approval_note = ?,
            approved_at = ?
        WHERE id = ?
        """,
        (
            PROPOSAL_STATUS_APPROVED,
            actor_name,
            note,
            _utc_now_iso(),
            int(proposal_id),
        ),
    )
    conn.commit()
    return get_adaptive_policy_proposal(conn, proposal_id)


def reject_adaptive_policy_proposal(
    conn: sqlite3.Connection,
    *,
    proposal_id: int,
    actor: str,
    note: str,
) -> dict[str, Any]:
    """Reject a pending adaptive-policy proposal."""

    proposal = get_adaptive_policy_proposal(conn, proposal_id)
    actor_name = str(actor).strip()
    rejection_note = str(note).strip()
    if not actor_name:
        raise ValueError("Adaptive-policy rejection actor is required.")
    if not rejection_note:
        raise ValueError("Adaptive-policy rejection note is required.")
    if proposal["status"] != PROPOSAL_STATUS_PENDING_REVIEW:
        raise ValueError("Only pending adaptive-policy proposals can be rejected.")
    conn.execute(
        """
        UPDATE adaptive_policy_proposals
        SET
            status = ?,
            rejected_by = ?,
            rejection_note = ?,
            rejected_at = ?
        WHERE id = ?
        """,
        (
            PROPOSAL_STATUS_REJECTED,
            actor_name,
            rejection_note,
            _utc_now_iso(),
            int(proposal_id),
        ),
    )
    conn.commit()
    return get_adaptive_policy_proposal(conn, proposal_id)


def apply_adaptive_policy_proposal(
    conn: sqlite3.Connection,
    *,
    proposal_id: int,
    actor: str,
    note: str | None = None,
) -> dict[str, Any]:
    """Apply an approved adaptive-policy proposal to the active parameter store."""

    proposal = get_adaptive_policy_proposal(conn, proposal_id)
    actor_name = str(actor).strip()
    if not actor_name:
        raise ValueError("Adaptive-policy apply actor is required.")
    if proposal["status"] != PROPOSAL_STATUS_APPROVED:
        raise ValueError("Adaptive-policy proposal must be approved before apply.")
    for item in proposal["items"]:
        set_parameter_value(
            conn,
            str(item["parameter_key"]),
            float(item["proposed_value"]),
            (
                f"Applied adaptive-policy proposal #{proposal_id}"
                + (f": {item['description']}" if item.get("description") else "")
            ),
        )
    conn.execute(
        """
        UPDATE adaptive_policy_proposals
        SET
            status = ?,
            applied_by = ?,
            applied_note = ?,
            applied_at = ?
        WHERE id = ?
        """,
        (
            PROPOSAL_STATUS_APPLIED,
            actor_name,
            note,
            _utc_now_iso(),
            int(proposal_id),
        ),
    )
    conn.commit()
    return get_adaptive_policy_proposal(conn, proposal_id)


__all__ = [
    "ADAPTIVE_POLICY_DEFAULTS",
    "AdaptivePolicySnapshot",
    "AdaptivePolicyProposalItem",
    "CLOSURE_DELAY_FACTOR_KEY",
    "DRIVER_EFFICIENCY_SCORE_KEY",
    "LANE_ETA_MULTIPLIER_KEY",
    "LANE_RATE_PER_M3_KEY",
    "PROPOSAL_STATUS_APPLIED",
    "PROPOSAL_STATUS_APPROVED",
    "PROPOSAL_STATUS_PENDING_REVIEW",
    "PROPOSAL_STATUS_REJECTED",
    "SEASONAL_MARGIN_UPLIFT_KEY",
    "TRUCK_EFFICIENCY_SCORE_KEY",
    "WEATHER_RISK_MULTIPLIER_KEY",
    "approve_adaptive_policy_proposal",
    "apply_bounded_parameter_target",
    "apply_adaptive_policy_proposal",
    "create_adaptive_policy_proposal",
    "ensure_adaptive_policy_defaults",
    "ensure_adaptive_policy_governance_tables",
    "get_adaptive_policy_proposal",
    "list_adaptive_policy_proposals",
    "load_adaptive_policy_snapshot",
    "reject_adaptive_policy_proposal",
]
