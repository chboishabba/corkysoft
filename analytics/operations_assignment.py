"""Segment-based operational planning and readiness helpers."""
from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any, Iterable, Sequence

from analytics.db.parameters import (
    bootstrap_parameters,
    ensure_global_parameters_table,
    get_parameter_value,
    set_parameter_value,
)
from analytics.db.schema import ensure_dashboard_tables
from analytics.db.legacy import (
    _ensure_job_segment_tables,
    _link_vehicle_to_segment,
    _link_worker_to_segment,
    get_or_create_job_segment,
)

REGO_WARNING_DAYS_KEY = "operations.warning_days.rego"
COI_WARNING_DAYS_KEY = "operations.warning_days.coi"
SERVICE_WARNING_DAYS_KEY = "operations.warning_days.service"
COMPLIANCE_WARNING_DAYS_KEY = "operations.warning_days.compliance"
SERVICE_OVERDUE_BLOCKS_KEY = "operations.block.service_overdue"
CONFLICT_BLOCKS_KEY = "operations.block.conflicts"
SERVICE_OVERRIDE_ALLOWED_KEY = "operations.override.service_overdue"
CONFLICT_OVERRIDE_ALLOWED_KEY = "operations.override.conflicts"

OPERATIONS_DEFAULTS = (
    (REGO_WARNING_DAYS_KEY, 30.0, "Days before rego expiry to raise warnings."),
    (COI_WARNING_DAYS_KEY, 30.0, "Days before COI expiry to raise warnings."),
    (SERVICE_WARNING_DAYS_KEY, 14.0, "Days before next service due to raise warnings."),
    (COMPLIANCE_WARNING_DAYS_KEY, 30.0, "Days before worker compliance expiry to raise warnings."),
    (SERVICE_OVERDUE_BLOCKS_KEY, 1.0, "Whether overdue service blocks assignment by default."),
    (CONFLICT_BLOCKS_KEY, 1.0, "Whether assignment conflicts block assignment by default."),
    (SERVICE_OVERRIDE_ALLOWED_KEY, 1.0, "Whether service-overdue blocks can be overridden."),
    (CONFLICT_OVERRIDE_ALLOWED_KEY, 1.0, "Whether assignment conflicts can be overridden."),
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _parse_iso_datetime(value: object | None) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_iso_date(value: object | None) -> datetime | None:
    parsed = _parse_iso_datetime(value)
    if parsed is not None:
        return parsed
    if value in (None, ""):
        return None
    try:
        return datetime.fromisoformat(str(value).strip()).replace(tzinfo=UTC)
    except ValueError:
        return None


def _serialize_flags(values: Iterable[str]) -> str:
    return json.dumps(sorted({value for value in values if value}))


def _deserialize_flags(value: object | None) -> list[str]:
    if value in (None, ""):
        return []
    try:
        parsed = json.loads(str(value))
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if str(item).strip()]


def _bool_param(conn: sqlite3.Connection, key: str, default: bool) -> bool:
    raw = get_parameter_value(conn, key, 1.0 if default else 0.0)
    return bool(raw) if raw is not None else default


def get_operations_policy(conn: sqlite3.Connection) -> dict[str, Any]:
    ensure_global_parameters_table(conn)
    bootstrap_parameters(conn, OPERATIONS_DEFAULTS)
    return {
        "regoWarningDays": int(get_parameter_value(conn, REGO_WARNING_DAYS_KEY, 30.0) or 30),
        "coiWarningDays": int(get_parameter_value(conn, COI_WARNING_DAYS_KEY, 30.0) or 30),
        "serviceWarningDays": int(get_parameter_value(conn, SERVICE_WARNING_DAYS_KEY, 14.0) or 14),
        "complianceWarningDays": int(
            get_parameter_value(conn, COMPLIANCE_WARNING_DAYS_KEY, 30.0) or 30
        ),
        "serviceOverdueBlocks": _bool_param(conn, SERVICE_OVERDUE_BLOCKS_KEY, True),
        "conflictBlocks": _bool_param(conn, CONFLICT_BLOCKS_KEY, True),
        "serviceOverrideAllowed": _bool_param(conn, SERVICE_OVERRIDE_ALLOWED_KEY, True),
        "conflictOverrideAllowed": _bool_param(conn, CONFLICT_OVERRIDE_ALLOWED_KEY, True),
    }


def update_operations_policy(
    conn: sqlite3.Connection,
    *,
    rego_warning_days: int,
    coi_warning_days: int,
    service_warning_days: int,
    compliance_warning_days: int,
    service_overdue_blocks: bool,
    conflict_blocks: bool,
    service_override_allowed: bool,
    conflict_override_allowed: bool,
) -> dict[str, Any]:
    ensure_global_parameters_table(conn)
    set_parameter_value(conn, REGO_WARNING_DAYS_KEY, float(rego_warning_days))
    set_parameter_value(conn, COI_WARNING_DAYS_KEY, float(coi_warning_days))
    set_parameter_value(conn, SERVICE_WARNING_DAYS_KEY, float(service_warning_days))
    set_parameter_value(conn, COMPLIANCE_WARNING_DAYS_KEY, float(compliance_warning_days))
    set_parameter_value(conn, SERVICE_OVERDUE_BLOCKS_KEY, 1.0 if service_overdue_blocks else 0.0)
    set_parameter_value(conn, CONFLICT_BLOCKS_KEY, 1.0 if conflict_blocks else 0.0)
    set_parameter_value(conn, SERVICE_OVERRIDE_ALLOWED_KEY, 1.0 if service_override_allowed else 0.0)
    set_parameter_value(conn, CONFLICT_OVERRIDE_ALLOWED_KEY, 1.0 if conflict_override_allowed else 0.0)
    return get_operations_policy(conn)


def _ensure_operations_columns(conn: sqlite3.Connection) -> None:
    ensure_dashboard_tables(conn)
    _ensure_job_segment_tables(conn)
    ensure_global_parameters_table(conn)
    bootstrap_parameters(conn, OPERATIONS_DEFAULTS)
    segment_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(job_segments)").fetchall()
    }
    for column, declaration in {
        "assignment_status": "TEXT DEFAULT 'draft'",
        "warning_flags": "TEXT DEFAULT '[]'",
        "blocking_flags": "TEXT DEFAULT '[]'",
        "overrideable_flags": "TEXT DEFAULT '[]'",
        "override_required": "INTEGER DEFAULT 0",
        "override_reason_code": "TEXT",
        "override_note": "TEXT",
    }.items():
        if column not in segment_columns:
            conn.execute(f"ALTER TABLE job_segments ADD COLUMN {column} {declaration}")

    segment_worker_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(job_segment_workers)").fetchall()
    }
    if "required_compliance_ids" not in segment_worker_columns:
        conn.execute("ALTER TABLE job_segment_workers ADD COLUMN required_compliance_ids TEXT")
    conn.commit()


def _backfill_job_segments(conn: sqlite3.Connection) -> None:
    _ensure_operations_columns(conn)
    jobs = conn.execute("SELECT id, origin, destination FROM jobs").fetchall()
    for row in jobs:
        get_or_create_job_segment(
            conn,
            job_id=int(row["id"]),
            segment_sequence=1,
            from_location=row["origin"],
            to_location=row["destination"],
            status="planned",
        )
    conn.commit()


def _overlaps(
    start_a: datetime | None,
    end_a: datetime | None,
    start_b: datetime | None,
    end_b: datetime | None,
) -> bool:
    if not all((start_a, end_a, start_b, end_b)):
        return False
    return start_a < end_b and start_b < end_a


def _assignment_conflicts(
    conn: sqlite3.Connection,
    *,
    segment_id: int,
    planned_start: datetime | None,
    planned_end: datetime | None,
    truck_ids: Sequence[str],
    worker_ids: Sequence[int],
) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    overrideable: list[str] = []
    if not planned_start or not planned_end:
        return warnings, overrideable

    rows = conn.execute(
        """
        SELECT
            js.id AS segment_id,
            js.planned_start,
            js.planned_end,
            jsv.truck_id,
            jsw.worker_id
        FROM job_segments AS js
        LEFT JOIN job_segment_vehicles AS jsv ON jsv.segment_id = js.id
        LEFT JOIN job_segment_workers AS jsw ON jsw.segment_id = js.id
        WHERE js.id != ?
        """,
        (segment_id,),
    ).fetchall()
    for row in rows:
        other_start = _parse_iso_datetime(row["planned_start"])
        other_end = _parse_iso_datetime(row["planned_end"])
        if not _overlaps(planned_start, planned_end, other_start, other_end):
            continue
        if row["truck_id"] and str(row["truck_id"]) in set(truck_ids):
            overrideable.append(f"truck_conflict:{row['truck_id']}:segment_{row['segment_id']}")
        if row["worker_id"] and int(row["worker_id"]) in set(worker_ids):
            overrideable.append(f"worker_conflict:{row['worker_id']}:segment_{row['segment_id']}")
    return warnings, overrideable


def _date_readiness_flags(
    *,
    prefix: str,
    label: str,
    due_value: object | None,
    warning_days: int,
    blocking: bool,
    warning_flags: list[str],
    blocking_flags: list[str],
    overrideable_flags: list[str],
) -> None:
    due = _parse_iso_date(due_value)
    if due is None:
        return
    now = _utc_now()
    if due < now:
        flag = f"{prefix}:{label}_expired"
        if blocking:
            blocking_flags.append(flag)
        else:
            overrideable_flags.append(flag)
    elif due <= now + timedelta(days=warning_days):
        warning_flags.append(f"{prefix}:{label}_due_soon")


def _worker_compliance_flags(
    conn: sqlite3.Connection,
    *,
    worker_id: int,
    required_ids: Sequence[int],
    warning_days: int,
    warning_flags: list[str],
    blocking_flags: list[str],
) -> None:
    now = _utc_now()
    for compliance_id in required_ids:
        row = conn.execute(
            """
            SELECT wc.name, wca.expiry_date
            FROM worker_compliances AS wc
            LEFT JOIN worker_compliance_assignments AS wca
                ON wca.compliance_id = wc.id AND wca.worker_id = ?
            WHERE wc.id = ?
            """,
            (worker_id, compliance_id),
        ).fetchone()
        compliance_name = row["name"] if row else str(compliance_id)
        if row is None or row["expiry_date"] is None:
            blocking_flags.append(f"worker:{worker_id}:missing_compliance:{compliance_name}")
            continue
        expiry = _parse_iso_date(row["expiry_date"])
        if expiry is None or expiry < now:
            blocking_flags.append(f"worker:{worker_id}:expired_compliance:{compliance_name}")
        elif expiry <= now + timedelta(days=warning_days):
            warning_flags.append(f"worker:{worker_id}:compliance_due_soon:{compliance_name}")


def evaluate_segment_readiness(
    conn: sqlite3.Connection,
    *,
    segment_id: int,
    truck_ids: Sequence[str] | None = None,
    worker_assignments: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    _ensure_operations_columns(conn)
    policy = get_operations_policy(conn)
    segment = conn.execute("SELECT * FROM job_segments WHERE id = ?", (segment_id,)).fetchone()
    if segment is None:
        raise ValueError(f"Segment {segment_id} not found")

    assigned_trucks = list(truck_ids or [])
    if not assigned_trucks:
        assigned_trucks = [
            str(row["truck_id"])
            for row in conn.execute(
                "SELECT truck_id FROM job_segment_vehicles WHERE segment_id = ? ORDER BY truck_id",
                (segment_id,),
            ).fetchall()
        ]

    assignments = list(worker_assignments or [])
    if not assignments:
        rows = conn.execute(
            """
            SELECT worker_id, role_id, required_compliance_ids
            FROM job_segment_workers
            WHERE segment_id = ?
            ORDER BY worker_id
            """,
            (segment_id,),
        ).fetchall()
        assignments = [
            {
                "workerId": int(row["worker_id"]),
                "roleId": row["role_id"],
                "requiredComplianceIds": _deserialize_flags(row["required_compliance_ids"]),
            }
            for row in rows
        ]

    warning_flags: list[str] = []
    blocking_flags: list[str] = []
    overrideable_flags: list[str] = []

    if not assigned_trucks:
        warning_flags.append("segment:no_truck_assigned")
    if not assignments:
        warning_flags.append("segment:no_worker_assigned")

    for truck_id in assigned_trucks:
        detail = conn.execute(
            "SELECT * FROM vehicle_details WHERE truck_id = ?",
            (truck_id,),
        ).fetchone()
        if detail is None:
            warning_flags.append(f"truck:{truck_id}:missing_vehicle_details")
            continue
        _date_readiness_flags(
            prefix=f"truck:{truck_id}",
            label="rego",
            due_value=detail["rego_expiry"],
            warning_days=policy["regoWarningDays"],
            blocking=True,
            warning_flags=warning_flags,
            blocking_flags=blocking_flags,
            overrideable_flags=overrideable_flags,
        )
        _date_readiness_flags(
            prefix=f"truck:{truck_id}",
            label="coi",
            due_value=detail["coi_due"],
            warning_days=policy["coiWarningDays"],
            blocking=True,
            warning_flags=warning_flags,
            blocking_flags=blocking_flags,
            overrideable_flags=overrideable_flags,
        )
        _date_readiness_flags(
            prefix=f"truck:{truck_id}",
            label="service",
            due_value=detail["next_service"],
            warning_days=policy["serviceWarningDays"],
            blocking=policy["serviceOverdueBlocks"] and not policy["serviceOverrideAllowed"],
            warning_flags=warning_flags,
            blocking_flags=blocking_flags,
            overrideable_flags=overrideable_flags,
        )
        if detail["next_service"] and _parse_iso_date(detail["next_service"]) and _parse_iso_date(detail["next_service"]) < _utc_now():
            if policy["serviceOverdueBlocks"] and policy["serviceOverrideAllowed"]:
                overrideable_flags.append(f"truck:{truck_id}:service_expired")
        if detail["daily_check_complete"] in (0, False):
            warning_flags.append(f"truck:{truck_id}:daily_check_incomplete")

    for assignment in assignments:
        worker_id = int(assignment["workerId"])
        role_id = assignment.get("roleId")
        worker = conn.execute("SELECT * FROM workers WHERE id = ?", (worker_id,)).fetchone()
        if worker is None:
            blocking_flags.append(f"worker:{worker_id}:missing")
            continue
        if role_id is not None:
            has_role = conn.execute(
                "SELECT 1 FROM worker_role_assignments WHERE worker_id = ? AND role_id = ?",
                (worker_id, role_id),
            ).fetchone()
            if has_role is None:
                blocking_flags.append(f"worker:{worker_id}:missing_role:{role_id}")
        required_ids = [int(item) for item in assignment.get("requiredComplianceIds", [])]
        if required_ids:
            _worker_compliance_flags(
                conn,
                worker_id=worker_id,
                required_ids=required_ids,
                warning_days=policy["complianceWarningDays"],
                warning_flags=warning_flags,
                blocking_flags=blocking_flags,
            )

    _, conflicts = _assignment_conflicts(
        conn,
        segment_id=segment_id,
        planned_start=_parse_iso_datetime(segment["planned_start"]),
        planned_end=_parse_iso_datetime(segment["planned_end"]),
        truck_ids=assigned_trucks,
        worker_ids=[int(item["workerId"]) for item in assignments],
    )
    if conflicts:
        if policy["conflictBlocks"] and not policy["conflictOverrideAllowed"]:
            blocking_flags.extend(conflicts)
        else:
            overrideable_flags.extend(conflicts)

    assignment_status = (
        "blocked"
        if blocking_flags
        else "override_required"
        if overrideable_flags
        else "planned"
        if assigned_trucks or assignments
        else "draft"
    )
    return {
        "segmentId": int(segment["id"]),
        "jobId": int(segment["job_id"]),
        "segmentSequence": int(segment["segment_sequence"]),
        "assignmentStatus": assignment_status,
        "warningFlags": sorted(set(warning_flags)),
        "blockingFlags": sorted(set(blocking_flags)),
        "overrideableFlags": sorted(set(overrideable_flags)),
        "overrideRequired": bool(overrideable_flags),
        "truckIds": assigned_trucks,
        "workerAssignments": assignments,
    }


def _persist_segment_readiness(
    conn: sqlite3.Connection,
    *,
    segment_id: int,
    readiness: dict[str, Any],
    override_reason_code: str | None = None,
    override_note: str | None = None,
) -> None:
    conn.execute(
        """
        UPDATE job_segments
        SET assignment_status = ?,
            warning_flags = ?,
            blocking_flags = ?,
            overrideable_flags = ?,
            override_required = ?,
            override_reason_code = ?,
            override_note = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (
            readiness["assignmentStatus"],
            _serialize_flags(readiness["warningFlags"]),
            _serialize_flags(readiness["blockingFlags"]),
            _serialize_flags(readiness["overrideableFlags"]),
            int(bool(readiness["overrideRequired"])),
            override_reason_code,
            override_note,
            _utc_now_iso(),
            segment_id,
        ),
    )
    conn.commit()


def assign_segment_resources(
    conn: sqlite3.Connection,
    *,
    segment_id: int,
    truck_ids: Sequence[str],
    worker_assignments: Sequence[dict[str, Any]],
    override: bool = False,
    override_reason_code: str | None = None,
    override_note: str | None = None,
) -> dict[str, Any]:
    _ensure_operations_columns(conn)
    segment = conn.execute("SELECT 1 FROM job_segments WHERE id = ?", (segment_id,)).fetchone()
    if segment is None:
        raise ValueError(f"Segment {segment_id} not found")

    readiness = evaluate_segment_readiness(
        conn,
        segment_id=segment_id,
        truck_ids=truck_ids,
        worker_assignments=worker_assignments,
    )
    if readiness["blockingFlags"]:
        raise ValueError("Segment has blocking readiness failures and cannot be assigned")
    if readiness["overrideRequired"] and not override:
        raise ValueError("Segment assignment requires override")
    if override and not override_reason_code:
        raise ValueError("Override reason code is required when overriding assignment policy")

    conn.execute("DELETE FROM job_segment_vehicles WHERE segment_id = ?", (segment_id,))
    conn.execute("DELETE FROM job_segment_workers WHERE segment_id = ?", (segment_id,))

    for truck_id in truck_ids:
        _link_vehicle_to_segment(
            conn,
            segment_id=segment_id,
            truck_id=str(truck_id).strip(),
            requirement_met=True,
        )

    for assignment in worker_assignments:
        worker_id = int(assignment["workerId"])
        role_id = assignment.get("roleId")
        compliance_ids = [int(item) for item in assignment.get("requiredComplianceIds", [])]
        _link_worker_to_segment(
            conn,
            segment_id=segment_id,
            worker_id=worker_id,
            role_id=int(role_id) if role_id not in (None, "") else None,
            required_compliance_ids=compliance_ids,
            start_time=assignment.get("startTime"),
            end_time=assignment.get("endTime"),
        )
        conn.execute(
            """
            UPDATE job_segment_workers
            SET required_compliance_ids = ?
            WHERE segment_id = ? AND worker_id = ?
            """,
            (_serialize_flags(str(item) for item in compliance_ids), segment_id, worker_id),
        )

    final_readiness = evaluate_segment_readiness(conn, segment_id=segment_id)
    if override and final_readiness["overrideRequired"]:
        final_readiness["assignmentStatus"] = "overridden"
    _persist_segment_readiness(
        conn,
        segment_id=segment_id,
        readiness=final_readiness,
        override_reason_code=override_reason_code if override else None,
        override_note=override_note if override else None,
    )
    return final_readiness


def list_segment_readiness(
    conn: sqlite3.Connection,
    *,
    job_id: int | None = None,
    assignment_status: str | None = None,
) -> list[dict[str, Any]]:
    _backfill_job_segments(conn)
    filters: list[str] = []
    params: list[Any] = []
    if job_id is not None:
        filters.append("js.job_id = ?")
        params.append(job_id)
    if assignment_status:
        filters.append("COALESCE(js.assignment_status, 'draft') = ?")
        params.append(assignment_status)
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    rows = conn.execute(
        f"""
        SELECT
            js.*,
            j.client,
            j.origin AS job_origin,
            j.destination AS job_destination
        FROM job_segments AS js
        JOIN jobs AS j ON j.id = js.job_id
        {where}
        ORDER BY js.job_id, js.segment_sequence
        """,
        params,
    ).fetchall()
    payload: list[dict[str, Any]] = []
    for row in rows:
        readiness = evaluate_segment_readiness(conn, segment_id=int(row["id"]))
        truck_rows = conn.execute(
            """
            SELECT t.truck_id, t.name, vd.source_imported_at
            FROM job_segment_vehicles AS jsv
            JOIN trucks AS t ON t.truck_id = jsv.truck_id
            LEFT JOIN vehicle_details AS vd ON vd.truck_id = t.truck_id
            WHERE jsv.segment_id = ?
            ORDER BY t.truck_id
            """,
            (row["id"],),
        ).fetchall()
        worker_rows = conn.execute(
            """
            SELECT
                w.id,
                w.name,
                jsw.role_id,
                jsw.required_compliance_ids,
                w.source_imported_at
            FROM job_segment_workers AS jsw
            JOIN workers AS w ON w.id = jsw.worker_id
            WHERE jsw.segment_id = ?
            ORDER BY w.name
            """,
            (row["id"],),
        ).fetchall()
        payload.append(
            {
                "segmentId": int(row["id"]),
                "jobId": int(row["job_id"]),
                "jobClient": row["client"],
                "jobOrigin": row["job_origin"],
                "jobDestination": row["job_destination"],
                "segmentSequence": int(row["segment_sequence"]),
                "fromLocation": row["from_location"],
                "toLocation": row["to_location"],
                "plannedStart": row["planned_start"],
                "plannedEnd": row["planned_end"],
                "assignmentStatus": readiness["assignmentStatus"],
                "warningFlags": readiness["warningFlags"],
                "blockingFlags": readiness["blockingFlags"],
                "overrideableFlags": readiness["overrideableFlags"],
                "overrideRequired": readiness["overrideRequired"],
                "overrideReasonCode": row["override_reason_code"],
                "overrideNote": row["override_note"],
                "truckAssignments": [
                    {
                        "truckId": item["truck_id"],
                        "truckName": item["name"],
                        "sourceImportedAt": item["source_imported_at"],
                    }
                    for item in truck_rows
                ],
                "workerAssignments": [
                    {
                        "workerId": int(item["id"]),
                        "workerName": item["name"],
                        "roleId": item["role_id"],
                        "requiredComplianceIds": _deserialize_flags(item["required_compliance_ids"]),
                        "sourceImportedAt": item["source_imported_at"],
                    }
                    for item in worker_rows
                ],
            }
        )
    return payload


def list_operational_conflicts(
    conn: sqlite3.Connection,
    *,
    job_id: int | None = None,
) -> list[dict[str, Any]]:
    segments = list_segment_readiness(conn, job_id=job_id)
    conflicts: list[dict[str, Any]] = []
    for item in segments:
        for flag in item["blockingFlags"] + item["overrideableFlags"]:
            if "conflict:" in flag or "truck_conflict" in flag or "worker_conflict" in flag:
                conflicts.append(
                    {
                        "segmentId": item["segmentId"],
                        "jobId": item["jobId"],
                        "assignmentStatus": item["assignmentStatus"],
                        "flag": flag,
                    }
                )
    return conflicts


def list_worker_assignment_summary(conn: sqlite3.Connection) -> dict[int, dict[str, Any]]:
    _backfill_job_segments(conn)
    rows = conn.execute(
        """
        SELECT
            w.id AS worker_id,
            COUNT(DISTINCT js.id) AS planned_segment_count,
            COUNT(DISTINCT js.job_id) AS planned_job_count,
            MIN(js.planned_start) AS next_planned_start,
            GROUP_CONCAT(DISTINCT jsv.truck_id) AS planned_trucks
        FROM workers AS w
        LEFT JOIN job_segment_workers AS jsw ON jsw.worker_id = w.id
        LEFT JOIN job_segments AS js ON js.id = jsw.segment_id
        LEFT JOIN job_segment_vehicles AS jsv ON jsv.segment_id = js.id
        GROUP BY w.id
        """
    ).fetchall()
    summary: dict[int, dict[str, Any]] = {}
    for row in rows:
        trucks = sorted(
            {
                item.strip()
                for item in str(row["planned_trucks"] or "").split(",")
                if item and item.strip()
            }
        )
        summary[int(row["worker_id"])] = {
            "plannedSegmentCount": int(row["planned_segment_count"] or 0),
            "plannedJobCount": int(row["planned_job_count"] or 0),
            "nextPlannedStart": row["next_planned_start"],
            "plannedTrucks": trucks,
        }
    return summary


def list_truck_assignment_summary(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    _backfill_job_segments(conn)
    rows = conn.execute(
        """
        SELECT
            t.truck_id,
            COUNT(DISTINCT js.id) AS planned_segment_count,
            COUNT(DISTINCT js.job_id) AS planned_job_count,
            MIN(js.planned_start) AS next_planned_start,
            GROUP_CONCAT(DISTINCT w.name) AS planned_workers
        FROM trucks AS t
        LEFT JOIN job_segment_vehicles AS jsv ON jsv.truck_id = t.truck_id
        LEFT JOIN job_segments AS js ON js.id = jsv.segment_id
        LEFT JOIN job_segment_workers AS jsw ON jsw.segment_id = js.id
        LEFT JOIN workers AS w ON w.id = jsw.worker_id
        GROUP BY t.truck_id
        """
    ).fetchall()
    summary: dict[str, dict[str, Any]] = {}
    for row in rows:
        workers = sorted(
            {
                item.strip()
                for item in str(row["planned_workers"] or "").split(",")
                if item and item.strip()
            }
        )
        summary[str(row["truck_id"])] = {
            "plannedSegmentCount": int(row["planned_segment_count"] or 0),
            "plannedJobCount": int(row["planned_job_count"] or 0),
            "nextPlannedStart": row["next_planned_start"],
            "plannedWorkers": workers,
        }
    return summary


def list_segments_for_worker(conn: sqlite3.Connection, *, worker_id: int) -> list[dict[str, Any]]:
    return [
        item
        for item in list_segment_readiness(conn)
        if any(int(assignment["workerId"]) == worker_id for assignment in item["workerAssignments"])
    ]


def list_segments_for_truck(conn: sqlite3.Connection, *, truck_id: str) -> list[dict[str, Any]]:
    target = str(truck_id).strip()
    return [
        item
        for item in list_segment_readiness(conn)
        if any(str(assignment["truckId"]).strip() == target for assignment in item["truckAssignments"])
    ]


def ensure_segment(
    conn: sqlite3.Connection,
    *,
    job_id: int,
    segment_sequence: int,
    from_location: str | None = None,
    to_location: str | None = None,
    planned_start: str | None = None,
    planned_end: str | None = None,
) -> sqlite3.Row:
    _backfill_job_segments(conn)
    segment = get_or_create_job_segment(
        conn,
        job_id=job_id,
        segment_sequence=segment_sequence,
        from_location=from_location,
        to_location=to_location,
        planned_start=planned_start,
        planned_end=planned_end,
        status="planned",
    )
    updates: list[str] = []
    params: list[Any] = []
    for column, value in (
        ("from_location", from_location),
        ("to_location", to_location),
        ("planned_start", planned_start),
        ("planned_end", planned_end),
    ):
        if value not in (None, ""):
            updates.append(f"{column} = ?")
            params.append(value)
    if updates:
        updates.append("updated_at = ?")
        params.append(_utc_now_iso())
        params.append(int(segment["id"]))
        conn.execute(
            f"UPDATE job_segments SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        conn.commit()
        segment = conn.execute("SELECT * FROM job_segments WHERE id = ?", (segment["id"],)).fetchone()
    readiness = evaluate_segment_readiness(conn, segment_id=int(segment["id"]))
    _persist_segment_readiness(conn, segment_id=int(segment["id"]), readiness=readiness)
    return conn.execute("SELECT * FROM job_segments WHERE id = ?", (segment["id"],)).fetchone()


__all__ = [
    "assign_segment_resources",
    "ensure_segment",
    "evaluate_segment_readiness",
    "get_operations_policy",
    "list_operational_conflicts",
    "list_segments_for_truck",
    "list_segments_for_worker",
    "list_segment_readiness",
    "list_truck_assignment_summary",
    "list_worker_assignment_summary",
    "update_operations_policy",
]
