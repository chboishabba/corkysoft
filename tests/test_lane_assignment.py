from __future__ import annotations

import sqlite3

from analytics.db import ensure_dashboard_tables
from analytics.lane_assignment import (
    LANE_PROPOSAL_STATUS_APPLIED,
    LANE_PROPOSAL_STATUS_APPROVED,
    LANE_PROPOSAL_STATUS_PENDING_REVIEW,
    LANE_STATUS_AMBIGUOUS,
    LANE_STATUS_ASSIGNED,
    LANE_STATUS_UNASSIGNED,
    apply_lane_promotion_proposal,
    approve_lane_promotion_proposal,
    backfill_lane_assignments,
    create_lane_promotion_proposal,
    ensure_lane_assignment_schema,
    reject_lane_promotion_proposal,
)


def _assignment_row(conn: sqlite3.Connection, table: str, row_id: int) -> sqlite3.Row:
    return conn.execute(
        f"""
        SELECT
            origin_cluster_key,
            destination_cluster_key,
            lane_key,
            corridor_group_key,
            lane_assignment_status,
            lane_assignment_source,
            lane_assignment_note
        FROM {table}
        WHERE id = ?
        """,
        (row_id,),
    ).fetchone()


def test_historical_backfill_reuses_existing_lane_and_preserves_corridor_label() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)
    ensure_lane_assignment_schema(conn)

    conn.execute(
        """
        INSERT INTO historical_jobs (
            job_date,
            corridor_display,
            origin,
            destination,
            origin_postcode,
            destination_postcode,
            price_per_m3
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "2026-03-01",
            "Brisbane → Gold Coast",
            "Brisbane",
            "Gold Coast",
            "4000",
            "4217",
            100.0,
        ),
    )
    first_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    backfill_lane_assignments(conn, dataset="historical", row_ids=[first_id])

    conn.execute(
        """
        INSERT INTO historical_jobs (
            job_date,
            corridor_display,
            origin,
            destination,
            origin_postcode,
            destination_postcode,
            price_per_m3
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "2026-03-02",
            "Legacy display should stay",
            "Brisbane",
            "Gold Coast",
            "4000",
            "4217",
            120.0,
        ),
    )
    second_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    backfill_lane_assignments(conn, dataset="historical", row_ids=[second_id])

    row = _assignment_row(conn, "historical_jobs", second_id)
    assert row["lane_assignment_status"] == LANE_STATUS_ASSIGNED
    assert row["origin_cluster_key"] == "pc:4000"
    assert row["destination_cluster_key"] == "pc:4217"
    assert row["lane_key"] == "pc:4000->pc:4217"
    assert row["corridor_group_key"] == "pc:4000<->pc:4217"
    assert row["lane_assignment_source"] == "postcode|postcode"
    assert conn.execute(
        "SELECT corridor_display FROM historical_jobs WHERE id = ?",
        (second_id,),
    ).fetchone()[0] == "Legacy display should stay"


def test_live_backfill_marks_missing_evidence_unassigned() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)
    ensure_lane_assignment_schema(conn)

    conn.execute(
        "INSERT INTO jobs (client) VALUES (?)",
        ("Kent",),
    )
    row_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    backfill_lane_assignments(conn, dataset="live", row_ids=[row_id])

    row = _assignment_row(conn, "jobs", row_id)
    assert row["lane_assignment_status"] == LANE_STATUS_UNASSIGNED
    assert row["lane_key"] is None
    assert row["lane_assignment_source"] is None


def test_live_backfill_marks_broad_overlap_ambiguous() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)
    ensure_lane_assignment_schema(conn)

    conn.execute(
        """
        INSERT INTO historical_jobs (
            job_date,
            origin,
            destination,
            origin_postcode,
            destination_postcode,
            price_per_m3
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("2026-03-01", "Brisbane", "Gold Coast", "4000", "4217", 100.0),
    )
    historical_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    backfill_lane_assignments(conn, dataset="historical", row_ids=[historical_id])

    conn.execute(
        """
        INSERT INTO jobs (
            origin,
            destination,
            origin_postcode,
            destination_postcode
        ) VALUES (?, ?, ?, ?)
        """,
        ("Brisbane Airport", "Gold Coast South", "4001", "4218"),
    )
    live_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    backfill_lane_assignments(conn, dataset="live", row_ids=[live_id])

    row = _assignment_row(conn, "jobs", live_id)
    assert row["lane_assignment_status"] == LANE_STATUS_AMBIGUOUS
    assert row["lane_key"] is None
    assert row["lane_assignment_source"] == "postcode|postcode"


def test_lane_promotion_proposal_lifecycle_promotes_candidate_lane() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)
    ensure_lane_assignment_schema(conn)

    conn.execute(
        """
        INSERT INTO historical_jobs (
            job_date,
            origin,
            destination,
            origin_postcode,
            destination_postcode,
            price_per_m3
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("2026-03-01", "Brisbane", "Gold Coast", "4000", "4217", 100.0),
    )
    historical_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    backfill_lane_assignments(conn, dataset="historical", row_ids=[historical_id])

    conn.execute(
        """
        INSERT INTO jobs (
            origin,
            destination,
            origin_postcode,
            destination_postcode
        ) VALUES (?, ?, ?, ?)
        """,
        ("Brisbane Airport", "Gold Coast South", "4001", "4218"),
    )
    live_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    backfill_lane_assignments(conn, dataset="live", row_ids=[live_id])

    proposal = create_lane_promotion_proposal(
        conn,
        dataset="live",
        row_id=live_id,
        actor="ops_manager",
        note="Promote reviewed lane candidate.",
    )
    assert proposal["status"] == LANE_PROPOSAL_STATUS_PENDING_REVIEW
    assert proposal["lane_key"] == "pc:4001->pc:4218"

    approved = approve_lane_promotion_proposal(
        conn,
        proposal_id=int(proposal["id"]),
        actor="commercial_owner",
        note="Approved after review.",
    )
    assert approved["status"] == LANE_PROPOSAL_STATUS_APPROVED

    applied = apply_lane_promotion_proposal(
        conn,
        proposal_id=int(proposal["id"]),
        actor="ops_manager",
        note="Apply approved lane.",
    )
    assert applied["status"] == LANE_PROPOSAL_STATUS_APPLIED

    lane_row = conn.execute(
        "SELECT lane_key FROM directional_lanes WHERE lane_key = ?",
        ("pc:4001->pc:4218",),
    ).fetchone()
    assert lane_row is not None

    updated_row = _assignment_row(conn, "jobs", live_id)
    assert updated_row["lane_assignment_status"] == LANE_STATUS_ASSIGNED
    assert updated_row["lane_key"] == "pc:4001->pc:4218"


def test_lane_promotion_rejection_requires_note() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_dashboard_tables(conn)
    ensure_lane_assignment_schema(conn)

    conn.execute(
        """
        INSERT INTO jobs (
            origin,
            destination,
            origin_postcode,
            destination_postcode,
            lane_assignment_status,
            origin_cluster_key,
            destination_cluster_key,
            lane_assignment_source,
            lane_assignment_note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "Townsville",
            "Cairns",
            "4810",
            "4870",
            LANE_STATUS_UNASSIGNED,
            "pc:4810",
            "pc:4870",
            "postcode|postcode",
            "Manual staging for governance test.",
        ),
    )
    row_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    proposal = create_lane_promotion_proposal(
        conn,
        dataset="live",
        row_id=row_id,
        actor="ops_manager",
    )

    try:
        reject_lane_promotion_proposal(
            conn,
            proposal_id=int(proposal["id"]),
            actor="commercial_owner",
            note="",
        )
    except ValueError as exc:
        assert "note is required" in str(exc)
    else:
        raise AssertionError("Expected lane promotion rejection to require a note.")
