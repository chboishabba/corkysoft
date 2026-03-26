"""Canonical lane assignment helpers for historical and live jobs."""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import UTC, datetime
from typing import Any, Mapping, Sequence

LANE_STATUS_ASSIGNED = "assigned"
LANE_STATUS_AMBIGUOUS = "ambiguous"
LANE_STATUS_UNASSIGNED = "unassigned"
LANE_PROPOSAL_STATUS_PENDING_REVIEW = "pending_review"
LANE_PROPOSAL_STATUS_APPROVED = "approved"
LANE_PROPOSAL_STATUS_REJECTED = "rejected"
LANE_PROPOSAL_STATUS_APPLIED = "applied"

_GEOHASH_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    return {
        str(row[1])
        for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    }


def ensure_lane_assignment_schema(conn: sqlite3.Connection) -> None:
    """Ensure canonical lane-assignment tables and columns exist."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS location_clusters (
            cluster_key TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            cluster_type TEXT NOT NULL,
            source TEXT NOT NULL,
            centroid_lat REAL,
            centroid_lon REAL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS corridor_groups (
            corridor_group_key TEXT PRIMARY KEY,
            cluster_a_key TEXT NOT NULL,
            cluster_b_key TEXT NOT NULL,
            display_name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS directional_lanes (
            lane_key TEXT PRIMARY KEY,
            corridor_group_key TEXT NOT NULL,
            origin_cluster_key TEXT NOT NULL,
            destination_cluster_key TEXT NOT NULL,
            display_name TEXT NOT NULL,
            promotion_status TEXT NOT NULL DEFAULT 'promoted',
            source TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(corridor_group_key) REFERENCES corridor_groups(corridor_group_key),
            FOREIGN KEY(origin_cluster_key) REFERENCES location_clusters(cluster_key),
            FOREIGN KEY(destination_cluster_key) REFERENCES location_clusters(cluster_key)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_directional_lanes_group
        ON directional_lanes(corridor_group_key, origin_cluster_key, destination_cluster_key)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lane_promotion_proposals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            status TEXT NOT NULL,
            requested_by TEXT NOT NULL,
            request_note TEXT,
            created_at TEXT NOT NULL,
            dataset TEXT NOT NULL,
            source_row_id INTEGER,
            source_summary TEXT NOT NULL,
            origin_cluster_key TEXT NOT NULL,
            destination_cluster_key TEXT NOT NULL,
            corridor_group_key TEXT NOT NULL,
            lane_key TEXT NOT NULL,
            lane_display_name TEXT NOT NULL,
            corridor_display_name TEXT NOT NULL,
            source TEXT,
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
        CREATE INDEX IF NOT EXISTS idx_lane_promotion_proposals_status_created
        ON lane_promotion_proposals(status, created_at DESC)
        """
    )
    for table_name in ("historical_jobs", "jobs"):
        if not _table_exists(conn, table_name):
            continue
        columns = _table_columns(conn, table_name)
        declarations = {
            "origin_cluster_key": "TEXT",
            "destination_cluster_key": "TEXT",
            "lane_key": "TEXT",
            "corridor_group_key": "TEXT",
            "lane_assignment_status": "TEXT",
            "lane_assignment_source": "TEXT",
            "lane_assignment_note": "TEXT",
        }
        for column, declaration in declarations.items():
            if column not in columns:
                conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column} {declaration}")
    conn.commit()


def create_lane_promotion_proposal(
    conn: sqlite3.Connection,
    *,
    dataset: str,
    row_id: int,
    actor: str,
    note: str | None = None,
) -> dict[str, Any]:
    """Create a promotion proposal from one historical/live job row."""

    ensure_lane_assignment_schema(conn)
    actor_name = str(actor).strip()
    if not actor_name:
        raise ValueError("Lane promotion proposal actor is required.")
    row = _load_governance_row(conn, dataset=dataset, row_id=int(row_id))
    if not row.get("origin_cluster_key") or not row.get("destination_cluster_key"):
        raise ValueError("Lane promotion proposal requires both origin and destination clusters.")
    if row.get("lane_assignment_status") == LANE_STATUS_ASSIGNED:
        raise ValueError("Assigned rows do not need a lane promotion proposal.")

    lane_key = f"{row['origin_cluster_key']}->{row['destination_cluster_key']}"
    if conn.execute("SELECT 1 FROM directional_lanes WHERE lane_key = ?", (lane_key,)).fetchone():
        raise ValueError("Directional lane already exists for this candidate.")

    lane_display_name = (
        f"{str(row.get('origin_label') or row['origin_cluster_key']).strip()} → "
        f"{str(row.get('destination_label') or row['destination_cluster_key']).strip()}"
    )
    corridor_display_name = _bidirectional_label(
        str(row.get("origin_label") or row["origin_cluster_key"]).strip(),
        str(row.get("destination_label") or row["destination_cluster_key"]).strip(),
    )
    source_summary = {
        "dataset": dataset,
        "rowId": int(row_id),
        "reference": row.get("reference"),
        "corridorDisplay": row.get("corridor_display"),
        "laneAssignmentStatus": row.get("lane_assignment_status"),
        "laneAssignmentSource": row.get("lane_assignment_source"),
        "laneAssignmentNote": row.get("lane_assignment_note"),
    }
    cursor = conn.execute(
        """
        INSERT INTO lane_promotion_proposals (
            status,
            requested_by,
            request_note,
            created_at,
            dataset,
            source_row_id,
            source_summary,
            origin_cluster_key,
            destination_cluster_key,
            corridor_group_key,
            lane_key,
            lane_display_name,
            corridor_display_name,
            source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            LANE_PROPOSAL_STATUS_PENDING_REVIEW,
            actor_name,
            note,
            _utc_now_iso(),
            dataset,
            int(row_id),
            json.dumps(source_summary, sort_keys=True),
            str(row["origin_cluster_key"]),
            str(row["destination_cluster_key"]),
            _corridor_group_key(str(row["origin_cluster_key"]), str(row["destination_cluster_key"])),
            lane_key,
            lane_display_name,
            corridor_display_name,
            row.get("lane_assignment_source"),
        ),
    )
    conn.commit()
    return get_lane_promotion_proposal(conn, int(cursor.lastrowid))


def create_lane_promotion_proposal_for_clusters(
    conn: sqlite3.Connection,
    *,
    dataset: str,
    origin_cluster_key: str,
    destination_cluster_key: str,
    actor: str,
    note: str | None = None,
) -> dict[str, Any]:
    """Create one promotion proposal covering all matching candidate rows for a cluster pair."""

    ensure_lane_assignment_schema(conn)
    actor_name = str(actor).strip()
    if not actor_name:
        raise ValueError("Lane promotion proposal actor is required.")
    origin_key = str(origin_cluster_key).strip()
    destination_key = str(destination_cluster_key).strip()
    if not origin_key or not destination_key:
        raise ValueError("Lane promotion proposal requires both cluster keys.")
    lane_key = f"{origin_key}->{destination_key}"
    if conn.execute("SELECT 1 FROM directional_lanes WHERE lane_key = ?", (lane_key,)).fetchone():
        raise ValueError("Directional lane already exists for this candidate.")

    table = _dataset_table(dataset)
    rows = conn.execute(
        f"""
        SELECT id
        FROM {table}
        WHERE origin_cluster_key = ? AND destination_cluster_key = ?
          AND COALESCE(NULLIF(TRIM(lane_assignment_status), ''), 'unassigned') != ?
        ORDER BY id
        """,
        (origin_key, destination_key, LANE_STATUS_ASSIGNED),
    ).fetchall()
    if not rows:
        raise ValueError("No unresolved candidate rows found for this cluster pair.")
    representative_row_id = int(rows[0][0])
    proposal = create_lane_promotion_proposal(
        conn,
        dataset=dataset,
        row_id=representative_row_id,
        actor=actor_name,
        note=note,
    )
    source_summary = dict(proposal.get("source_summary") or {})
    source_summary["candidateRowIds"] = [int(row[0]) for row in rows]
    source_summary["candidateCount"] = len(rows)
    conn.execute(
        "UPDATE lane_promotion_proposals SET source_summary = ? WHERE id = ?",
        (json.dumps(source_summary, sort_keys=True), int(proposal["id"])),
    )
    conn.commit()
    return get_lane_promotion_proposal(conn, int(proposal["id"]))


def get_lane_promotion_proposal(conn: sqlite3.Connection, proposal_id: int) -> dict[str, Any]:
    ensure_lane_assignment_schema(conn)
    cursor = conn.execute(
        """
        SELECT
            id,
            status,
            requested_by,
            request_note,
            created_at,
            dataset,
            source_row_id,
            source_summary,
            origin_cluster_key,
            destination_cluster_key,
            corridor_group_key,
            lane_key,
            lane_display_name,
            corridor_display_name,
            source,
            approved_by,
            approval_note,
            approved_at,
            rejected_by,
            rejection_note,
            rejected_at,
            applied_by,
            applied_note,
            applied_at
        FROM lane_promotion_proposals
        WHERE id = ?
        """,
        (int(proposal_id),),
    )
    row = cursor.fetchone()
    if row is None:
        raise ValueError(f"Unknown lane promotion proposal: {proposal_id}")
    columns = [column[0] for column in cursor.description or []]
    payload = {column: row[column] for column in columns} if isinstance(row, sqlite3.Row) else dict(zip(columns, row, strict=False))
    payload["source_summary"] = json.loads(payload.get("source_summary") or "{}")
    return payload


def list_lane_promotion_proposals(
    conn: sqlite3.Connection,
    *,
    limit: int = 25,
    status: str | None = None,
) -> list[dict[str, Any]]:
    ensure_lane_assignment_schema(conn)
    query = "SELECT id FROM lane_promotion_proposals"
    params: list[Any] = []
    if status:
        query += " WHERE status = ?"
        params.append(status)
    query += " ORDER BY created_at DESC, id DESC LIMIT ?"
    params.append(int(limit))
    rows = conn.execute(query, params).fetchall()
    return [get_lane_promotion_proposal(conn, int(row[0])) for row in rows]


def approve_lane_promotion_proposal(
    conn: sqlite3.Connection,
    *,
    proposal_id: int,
    actor: str,
    note: str | None = None,
) -> dict[str, Any]:
    proposal = get_lane_promotion_proposal(conn, proposal_id)
    actor_name = str(actor).strip()
    if not actor_name:
        raise ValueError("Lane promotion approval actor is required.")
    if proposal["status"] != LANE_PROPOSAL_STATUS_PENDING_REVIEW:
        raise ValueError("Only pending lane promotion proposals can be approved.")
    conn.execute(
        """
        UPDATE lane_promotion_proposals
        SET status = ?, approved_by = ?, approval_note = ?, approved_at = ?
        WHERE id = ?
        """,
        (LANE_PROPOSAL_STATUS_APPROVED, actor_name, note, _utc_now_iso(), int(proposal_id)),
    )
    conn.commit()
    return get_lane_promotion_proposal(conn, proposal_id)


def reject_lane_promotion_proposal(
    conn: sqlite3.Connection,
    *,
    proposal_id: int,
    actor: str,
    note: str,
) -> dict[str, Any]:
    proposal = get_lane_promotion_proposal(conn, proposal_id)
    actor_name = str(actor).strip()
    rejection_note = str(note).strip()
    if not actor_name:
        raise ValueError("Lane promotion rejection actor is required.")
    if not rejection_note:
        raise ValueError("Lane promotion rejection note is required.")
    if proposal["status"] != LANE_PROPOSAL_STATUS_PENDING_REVIEW:
        raise ValueError("Only pending lane promotion proposals can be rejected.")
    conn.execute(
        """
        UPDATE lane_promotion_proposals
        SET status = ?, rejected_by = ?, rejection_note = ?, rejected_at = ?
        WHERE id = ?
        """,
        (LANE_PROPOSAL_STATUS_REJECTED, actor_name, rejection_note, _utc_now_iso(), int(proposal_id)),
    )
    conn.commit()
    return get_lane_promotion_proposal(conn, proposal_id)


def apply_lane_promotion_proposal(
    conn: sqlite3.Connection,
    *,
    proposal_id: int,
    actor: str,
    note: str | None = None,
) -> dict[str, Any]:
    proposal = get_lane_promotion_proposal(conn, proposal_id)
    actor_name = str(actor).strip()
    if not actor_name:
        raise ValueError("Lane promotion apply actor is required.")
    if proposal["status"] != LANE_PROPOSAL_STATUS_APPROVED:
        raise ValueError("Lane promotion proposal must be approved before apply.")

    _upsert_corridor_group(
        conn,
        corridor_group_key=str(proposal["corridor_group_key"]),
        cluster_a_key=min(str(proposal["origin_cluster_key"]), str(proposal["destination_cluster_key"])),
        cluster_b_key=max(str(proposal["origin_cluster_key"]), str(proposal["destination_cluster_key"])),
        display_name=str(proposal["corridor_display_name"]),
    )
    _upsert_directional_lane(
        conn,
        lane_key=str(proposal["lane_key"]),
        corridor_group_key=str(proposal["corridor_group_key"]),
        origin_cluster_key=str(proposal["origin_cluster_key"]),
        destination_cluster_key=str(proposal["destination_cluster_key"]),
        display_name=str(proposal["lane_display_name"]),
        source=str(proposal.get("source") or "lane_promotion_review"),
    )
    historical_ids = _matching_row_ids_for_clusters(
        conn,
        dataset="historical",
        origin_cluster_key=str(proposal["origin_cluster_key"]),
        destination_cluster_key=str(proposal["destination_cluster_key"]),
    )
    live_ids = _matching_row_ids_for_clusters(
        conn,
        dataset="live",
        origin_cluster_key=str(proposal["origin_cluster_key"]),
        destination_cluster_key=str(proposal["destination_cluster_key"]),
    )
    if historical_ids:
        backfill_lane_assignments(conn, dataset="historical", row_ids=historical_ids)
    if live_ids:
        backfill_lane_assignments(conn, dataset="live", row_ids=live_ids)
    conn.execute(
        """
        UPDATE lane_promotion_proposals
        SET status = ?, applied_by = ?, applied_note = ?, applied_at = ?
        WHERE id = ?
        """,
        (LANE_PROPOSAL_STATUS_APPLIED, actor_name, note, _utc_now_iso(), int(proposal_id)),
    )
    conn.commit()
    return get_lane_promotion_proposal(conn, proposal_id)


def backfill_lane_assignments(
    conn: sqlite3.Connection,
    *,
    dataset: str,
    row_ids: Sequence[int] | None = None,
) -> int:
    """Assign canonical lane metadata to historical or live jobs."""

    ensure_lane_assignment_schema(conn)
    rows = _load_assignment_rows(conn, dataset=dataset, row_ids=row_ids)
    updated = 0
    for row in rows:
        assignment = assign_lane_for_record(conn, row)
        conn.execute(
            f"""
            UPDATE {_dataset_table(dataset)}
            SET
                origin_cluster_key = ?,
                destination_cluster_key = ?,
                lane_key = ?,
                corridor_group_key = ?,
                lane_assignment_status = ?,
                lane_assignment_source = ?,
                lane_assignment_note = ?
            WHERE id = ?
            """,
            (
                assignment.get("origin_cluster_key"),
                assignment.get("destination_cluster_key"),
                assignment.get("lane_key"),
                assignment.get("corridor_group_key"),
                assignment["lane_assignment_status"],
                assignment.get("lane_assignment_source"),
                assignment.get("lane_assignment_note"),
                int(row["id"]),
            ),
        )
        updated += 1
    conn.commit()
    return updated


def assign_lane_for_record(conn: sqlite3.Connection, row: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve one row into canonical lane-assignment metadata."""

    ensure_lane_assignment_schema(conn)
    origin_cluster = _resolve_cluster(
        label=row.get("origin"),
        postcode=row.get("origin_postcode"),
        lat=row.get("origin_lat"),
        lon=row.get("origin_lon"),
    )
    destination_cluster = _resolve_cluster(
        label=row.get("destination"),
        postcode=row.get("destination_postcode"),
        lat=row.get("dest_lat"),
        lon=row.get("dest_lon"),
    )
    if origin_cluster is None or destination_cluster is None:
        return {
            "lane_assignment_status": LANE_STATUS_UNASSIGNED,
            "lane_assignment_source": None,
            "lane_assignment_note": "Insufficient endpoint evidence for canonical lane assignment.",
            "origin_cluster_key": origin_cluster["cluster_key"] if origin_cluster else None,
            "destination_cluster_key": destination_cluster["cluster_key"] if destination_cluster else None,
            "lane_key": None,
            "corridor_group_key": None,
        }

    _upsert_cluster(conn, origin_cluster)
    _upsert_cluster(conn, destination_cluster)

    lane_key = f"{origin_cluster['cluster_key']}->{destination_cluster['cluster_key']}"
    reverse_lane_key = f"{destination_cluster['cluster_key']}->{origin_cluster['cluster_key']}"
    corridor_group_key = _corridor_group_key(
        origin_cluster["cluster_key"], destination_cluster["cluster_key"]
    )

    if conn.execute("SELECT lane_key FROM directional_lanes WHERE lane_key = ?", (lane_key,)).fetchone():
        return {
            "lane_assignment_status": LANE_STATUS_ASSIGNED,
            "lane_assignment_source": _assignment_source(origin_cluster, destination_cluster),
            "lane_assignment_note": "Matched existing promoted directional lane.",
            "origin_cluster_key": origin_cluster["cluster_key"],
            "destination_cluster_key": destination_cluster["cluster_key"],
            "lane_key": lane_key,
            "corridor_group_key": corridor_group_key,
        }

    reverse_lane = conn.execute(
        "SELECT lane_key FROM directional_lanes WHERE lane_key = ?",
        (reverse_lane_key,),
    ).fetchone()
    broad_origin = _cluster_broad_key(origin_cluster["cluster_key"])
    broad_destination = _cluster_broad_key(destination_cluster["cluster_key"])
    existing_lanes = conn.execute(
        "SELECT origin_cluster_key, destination_cluster_key FROM directional_lanes"
    ).fetchall()
    broad_overlap = any(
        _cluster_broad_key(str(candidate[0])) == broad_origin
        and _cluster_broad_key(str(candidate[1])) == broad_destination
        for candidate in existing_lanes
    )
    if broad_overlap and reverse_lane is None:
        return {
            "lane_assignment_status": LANE_STATUS_AMBIGUOUS,
            "lane_assignment_source": _assignment_source(origin_cluster, destination_cluster),
            "lane_assignment_note": (
                "Candidate lane overlaps an existing broad corridor but does not cleanly "
                "match a promoted directional lane."
            ),
            "origin_cluster_key": origin_cluster["cluster_key"],
            "destination_cluster_key": destination_cluster["cluster_key"],
            "lane_key": None,
            "corridor_group_key": corridor_group_key,
        }

    _upsert_corridor_group(
        conn,
        corridor_group_key=corridor_group_key,
        cluster_a_key=min(origin_cluster["cluster_key"], destination_cluster["cluster_key"]),
        cluster_b_key=max(origin_cluster["cluster_key"], destination_cluster["cluster_key"]),
        display_name=_bidirectional_label(
            origin_cluster["display_name"], destination_cluster["display_name"]
        ),
    )
    _upsert_directional_lane(
        conn,
        lane_key=lane_key,
        corridor_group_key=corridor_group_key,
        origin_cluster_key=origin_cluster["cluster_key"],
        destination_cluster_key=destination_cluster["cluster_key"],
        display_name=f"{origin_cluster['display_name']} → {destination_cluster['display_name']}",
        source=_assignment_source(origin_cluster, destination_cluster),
    )
    note = (
        "Created directional lane from reverse-lane parity."
        if reverse_lane is not None
        else "Created new promoted directional lane."
    )
    return {
        "lane_assignment_status": LANE_STATUS_ASSIGNED,
        "lane_assignment_source": _assignment_source(origin_cluster, destination_cluster),
        "lane_assignment_note": note,
        "origin_cluster_key": origin_cluster["cluster_key"],
        "destination_cluster_key": destination_cluster["cluster_key"],
        "lane_key": lane_key,
        "corridor_group_key": corridor_group_key,
    }


def lane_grouping_key(row: Mapping[str, Any]) -> str | None:
    if str(row.get("lane_assignment_status") or "") == LANE_STATUS_ASSIGNED:
        return str(row.get("lane_key") or "") or None
    return None


def corridor_grouping_key(row: Mapping[str, Any]) -> str | None:
    if str(row.get("lane_assignment_status") or "") == LANE_STATUS_ASSIGNED:
        return str(row.get("corridor_group_key") or "") or None
    return None


def _dataset_table(dataset: str) -> str:
    if dataset == "historical":
        return "historical_jobs"
    if dataset == "live":
        return "jobs"
    raise ValueError(f"Unsupported lane-assignment dataset: {dataset}")


def _load_assignment_rows(
    conn: sqlite3.Connection,
    *,
    dataset: str,
    row_ids: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    if dataset == "historical":
        sql = """
            SELECT
                hj.id,
                COALESCE(hj.origin, o.city, o.normalized, o.raw_input) AS origin,
                COALESCE(hj.destination, d.city, d.normalized, d.raw_input) AS destination,
                COALESCE(hj.origin_postcode, o.postcode) AS origin_postcode,
                COALESCE(hj.destination_postcode, d.postcode) AS destination_postcode,
                o.lat AS origin_lat,
                o.lon AS origin_lon,
                d.lat AS dest_lat,
                d.lon AS dest_lon
            FROM historical_jobs AS hj
            LEFT JOIN addresses AS o ON hj.origin_address_id = o.id
            LEFT JOIN addresses AS d ON hj.destination_address_id = d.id
        """
    elif dataset == "live":
        sql = """
            SELECT
                id,
                origin,
                destination,
                origin_postcode,
                destination_postcode,
                origin_lat,
                origin_lon,
                dest_lat,
                dest_lon
            FROM jobs
        """
    else:
        raise ValueError(f"Unsupported lane-assignment dataset: {dataset}")

    params: list[Any] = []
    conditions: list[str] = []
    if row_ids:
        placeholders = ",".join("?" for _ in row_ids)
        row_id_column = "hj.id" if dataset == "historical" else "id"
        conditions.append(f"{row_id_column} IN ({placeholders})")
        params.extend(int(value) for value in row_ids)
    else:
        status_column = (
            "hj.lane_assignment_status" if dataset == "historical" else "lane_assignment_status"
        )
        conditions.append(f"({status_column} IS NULL OR TRIM({status_column}) = '')")
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    cursor = conn.execute(sql, params)
    columns = [column[0] for column in cursor.description or []]
    return [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]


def _load_governance_row(
    conn: sqlite3.Connection,
    *,
    dataset: str,
    row_id: int,
) -> dict[str, Any]:
    if dataset == "historical":
        sql = """
            SELECT
                hj.id,
                COALESCE(hj.client, CAST(hj.id AS TEXT)) AS reference,
                hj.corridor_display,
                COALESCE(hj.origin, o.city, o.normalized, o.raw_input) AS origin_label,
                COALESCE(hj.destination, d.city, d.normalized, d.raw_input) AS destination_label,
                hj.origin_cluster_key,
                hj.destination_cluster_key,
                hj.lane_assignment_status,
                hj.lane_assignment_source,
                hj.lane_assignment_note
            FROM historical_jobs AS hj
            LEFT JOIN addresses AS o ON hj.origin_address_id = o.id
            LEFT JOIN addresses AS d ON hj.destination_address_id = d.id
            WHERE hj.id = ?
        """
    elif dataset == "live":
        sql = """
            SELECT
                id,
                COALESCE(client, CAST(id AS TEXT)) AS reference,
                COALESCE(origin, '?') || ' → ' || COALESCE(destination, '?') AS corridor_display,
                origin AS origin_label,
                destination AS destination_label,
                origin_cluster_key,
                destination_cluster_key,
                lane_assignment_status,
                lane_assignment_source,
                lane_assignment_note
            FROM jobs
            WHERE id = ?
        """
    else:
        raise ValueError(f"Unsupported lane-assignment dataset: {dataset}")
    cursor = conn.execute(sql, (int(row_id),))
    row = cursor.fetchone()
    if row is None:
        raise ValueError(f"Unknown {dataset} lane candidate row: {row_id}")
    columns = [column[0] for column in cursor.description or []]
    return {column: row[column] for column in columns} if isinstance(row, sqlite3.Row) else dict(zip(columns, row, strict=False))


def _matching_row_ids_for_clusters(
    conn: sqlite3.Connection,
    *,
    dataset: str,
    origin_cluster_key: str,
    destination_cluster_key: str,
) -> list[int]:
    table = _dataset_table(dataset)
    rows = conn.execute(
        f"""
        SELECT id
        FROM {table}
        WHERE origin_cluster_key = ? AND destination_cluster_key = ?
        """,
        (origin_cluster_key, destination_cluster_key),
    ).fetchall()
    return [int(row[0]) for row in rows]


def _resolve_cluster(*, label: Any, postcode: Any, lat: Any, lon: Any) -> dict[str, Any] | None:
    lat_value = _safe_float(lat)
    lon_value = _safe_float(lon)
    postcode_text = _clean_string(postcode)
    label_text = _clean_string(label) or postcode_text
    if lat_value is not None and lon_value is not None:
        geohash = _encode_geohash(lat_value, lon_value, precision=5)
        return {
            "cluster_key": f"gh5:{geohash}",
            "display_name": label_text or geohash,
            "cluster_type": "geohash",
            "source": "geohash_5",
            "centroid_lat": lat_value,
            "centroid_lon": lon_value,
        }
    if postcode_text:
        return {
            "cluster_key": f"pc:{postcode_text[:4]}",
            "display_name": label_text or postcode_text[:4],
            "cluster_type": "postcode",
            "source": "postcode",
            "centroid_lat": None,
            "centroid_lon": None,
        }
    if label_text:
        slug = re.sub(r"[^a-z0-9]+", "-", label_text.lower()).strip("-")
        if slug:
            return {
                "cluster_key": f"txt:{slug}",
                "display_name": label_text,
                "cluster_type": "text",
                "source": "text_normalized",
                "centroid_lat": None,
                "centroid_lon": None,
            }
    return None


def _upsert_cluster(conn: sqlite3.Connection, cluster: Mapping[str, Any]) -> None:
    timestamp = _utc_now_iso()
    conn.execute(
        """
        INSERT INTO location_clusters (
            cluster_key,
            display_name,
            cluster_type,
            source,
            centroid_lat,
            centroid_lon,
            created_at,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(cluster_key) DO UPDATE SET
            display_name = excluded.display_name,
            cluster_type = excluded.cluster_type,
            source = excluded.source,
            centroid_lat = COALESCE(excluded.centroid_lat, location_clusters.centroid_lat),
            centroid_lon = COALESCE(excluded.centroid_lon, location_clusters.centroid_lon),
            updated_at = excluded.updated_at
        """,
        (
            cluster["cluster_key"],
            cluster["display_name"],
            cluster["cluster_type"],
            cluster["source"],
            cluster["centroid_lat"],
            cluster["centroid_lon"],
            timestamp,
            timestamp,
        ),
    )


def _upsert_corridor_group(
    conn: sqlite3.Connection,
    *,
    corridor_group_key: str,
    cluster_a_key: str,
    cluster_b_key: str,
    display_name: str,
) -> None:
    timestamp = _utc_now_iso()
    conn.execute(
        """
        INSERT INTO corridor_groups (
            corridor_group_key,
            cluster_a_key,
            cluster_b_key,
            display_name,
            created_at,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(corridor_group_key) DO UPDATE SET
            display_name = excluded.display_name,
            updated_at = excluded.updated_at
        """,
        (corridor_group_key, cluster_a_key, cluster_b_key, display_name, timestamp, timestamp),
    )


def _upsert_directional_lane(
    conn: sqlite3.Connection,
    *,
    lane_key: str,
    corridor_group_key: str,
    origin_cluster_key: str,
    destination_cluster_key: str,
    display_name: str,
    source: str,
) -> None:
    timestamp = _utc_now_iso()
    conn.execute(
        """
        INSERT INTO directional_lanes (
            lane_key,
            corridor_group_key,
            origin_cluster_key,
            destination_cluster_key,
            display_name,
            promotion_status,
            source,
            created_at,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, 'promoted', ?, ?, ?)
        ON CONFLICT(lane_key) DO UPDATE SET
            display_name = excluded.display_name,
            source = excluded.source,
            updated_at = excluded.updated_at
        """,
        (
            lane_key,
            corridor_group_key,
            origin_cluster_key,
            destination_cluster_key,
            display_name,
            source,
            timestamp,
            timestamp,
        ),
    )


def _cluster_broad_key(cluster_key: str) -> str:
    text = str(cluster_key)
    if text.startswith("gh5:"):
        return f"gh3:{text[4:7]}"
    if text.startswith("pc:"):
        return f"pc3:{text[3:6]}"
    return text


def _corridor_group_key(origin_cluster_key: str, destination_cluster_key: str) -> str:
    ordered = sorted([origin_cluster_key, destination_cluster_key])
    return f"{ordered[0]}<->{ordered[1]}"


def _bidirectional_label(origin_label: str, destination_label: str) -> str:
    if origin_label == destination_label:
        return origin_label
    ordered = sorted([origin_label, destination_label], key=str.lower)
    return f"{ordered[0]} ↔ {ordered[1]}"


def _assignment_source(origin_cluster: Mapping[str, Any], destination_cluster: Mapping[str, Any]) -> str:
    return f"{origin_cluster['source']}|{destination_cluster['source']}"


def _clean_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _encode_geohash(latitude: float, longitude: float, *, precision: int = 5) -> str:
    lat_interval = [-90.0, 90.0]
    lon_interval = [-180.0, 180.0]
    geohash = []
    bits = [16, 8, 4, 2, 1]
    bit = 0
    ch = 0
    even = True
    while len(geohash) < precision:
        if even:
            mid = sum(lon_interval) / 2
            if longitude > mid:
                ch |= bits[bit]
                lon_interval[0] = mid
            else:
                lon_interval[1] = mid
        else:
            mid = sum(lat_interval) / 2
            if latitude > mid:
                ch |= bits[bit]
                lat_interval[0] = mid
            else:
                lat_interval[1] = mid
        even = not even
        if bit < 4:
            bit += 1
        else:
            geohash.append(_GEOHASH_BASE32[ch])
            bit = 0
            ch = 0
    return "".join(geohash)


__all__ = [
    "LANE_PROPOSAL_STATUS_APPLIED",
    "LANE_PROPOSAL_STATUS_APPROVED",
    "LANE_PROPOSAL_STATUS_PENDING_REVIEW",
    "LANE_PROPOSAL_STATUS_REJECTED",
    "LANE_STATUS_AMBIGUOUS",
    "LANE_STATUS_ASSIGNED",
    "LANE_STATUS_UNASSIGNED",
    "approve_lane_promotion_proposal",
    "apply_lane_promotion_proposal",
    "assign_lane_for_record",
    "backfill_lane_assignments",
    "corridor_grouping_key",
    "create_lane_promotion_proposal",
    "ensure_lane_assignment_schema",
    "get_lane_promotion_proposal",
    "lane_grouping_key",
    "list_lane_promotion_proposals",
    "reject_lane_promotion_proposal",
]
