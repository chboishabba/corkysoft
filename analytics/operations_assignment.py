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

OPERATIONS_CUTOVER_DEFAULTS: tuple[dict[str, Any], ...] = (
    {
        "workflow_key": "dispatch_execution",
        "label": "Dispatch execution",
        "native_surface": "Dispatch tab",
        "spreadsheet_source": "FLEET / STAFF / ad hoc dispatch sheets",
        "cutover_status": "native_primary",
        "owner_role": "dispatcher",
        "snapshot_mode": "on_demand",
        "snapshot_fields": [
            "jobId",
            "jobClient",
            "jobOrigin",
            "jobDestination",
            "jobStatus",
            "segmentCount",
            "truckIds",
            "workerNames",
        ],
        "fallback_mode": "import_only",
        "cutover_target_percent": 100.0,
        "native_usage_percent": 100.0,
        "fallback_usage_count": 0,
        "open_issue_count": 0,
        "snapshot_consumer_count": 1,
        "native_ready": True,
        "dual_run_complete": True,
        "fallback_drill_complete": False,
        "operator_trained": False,
        "last_review_at": None,
        "rollback_instructions": (
            "Keep spreadsheet imports read-only. If dispatch board data is stale or unavailable, "
            "refresh workbook imports, export a dispatch snapshot CSV, and continue operations from "
            "that snapshot until the board is restored."
        ),
        "notes": "Primary daily execution board now lives in Corkysoft.",
    },
    {
        "workflow_key": "maintenance_compliance",
        "label": "Maintenance and compliance",
        "native_surface": "Fleet / Vehicle maintenance tabs",
        "spreadsheet_source": "FLEET / VEHICLE_REPAIRS",
        "cutover_status": "dual_run",
        "owner_role": "fleet_manager",
        "snapshot_mode": "none",
        "snapshot_fields": [],
        "fallback_mode": "import_only",
        "cutover_target_percent": 90.0,
        "native_usage_percent": 60.0,
        "fallback_usage_count": 0,
        "open_issue_count": 1,
        "snapshot_consumer_count": 0,
        "native_ready": True,
        "dual_run_complete": False,
        "fallback_drill_complete": False,
        "operator_trained": False,
        "last_review_at": None,
        "rollback_instructions": (
            "If readiness signals look incorrect, rerun workbook sync and VEHICLE_REPAIRS import, "
            "then validate blocked/due-soon items before falling back to the source sheet."
        ),
        "notes": "Cockpit is implemented; cutover drill still pending.",
    },
    {
        "workflow_key": "labor_planning",
        "label": "Labor planning",
        "native_surface": "Driver shifts tab",
        "spreadsheet_source": "VEHICLE_DRIVER",
        "cutover_status": "dual_run",
        "owner_role": "operations_manager",
        "snapshot_mode": "none",
        "snapshot_fields": [],
        "fallback_mode": "import_only",
        "cutover_target_percent": 90.0,
        "native_usage_percent": 55.0,
        "fallback_usage_count": 0,
        "open_issue_count": 1,
        "snapshot_consumer_count": 0,
        "native_ready": True,
        "dual_run_complete": False,
        "fallback_drill_complete": False,
        "operator_trained": False,
        "last_review_at": None,
        "rollback_instructions": (
            "Use the reconciliation view to compare planned labor against imported shifts. If native "
            "planning is unavailable, import VEHICLE_DRIVER and continue from the last reconciled roster."
        ),
        "notes": "Native planning exists but remains in dual-run with imported reconciliation.",
    },
    {
        "workflow_key": "inventory_coordination",
        "label": "Inventory and suppliers",
        "native_surface": "Inventory tab",
        "spreadsheet_source": "SUPPLIERS / stock sheets",
        "cutover_status": "dual_run",
        "owner_role": "warehouse",
        "snapshot_mode": "none",
        "snapshot_fields": [],
        "fallback_mode": "import_only",
        "cutover_target_percent": 90.0,
        "native_usage_percent": 50.0,
        "fallback_usage_count": 0,
        "open_issue_count": 1,
        "snapshot_consumer_count": 0,
        "native_ready": True,
        "dual_run_complete": False,
        "fallback_drill_complete": False,
        "operator_trained": False,
        "last_review_at": None,
        "rollback_instructions": (
            "If segment-linked stock data diverges, rerun supplier/stock imports and validate shipment "
            "allocations before using spreadsheet balances as a temporary reference."
        ),
        "notes": "Segment-linked allocations exist; supplier-side process cutover still pending.",
    },
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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS operations_cutover_workflows (
            workflow_key TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            native_surface TEXT NOT NULL,
            spreadsheet_source TEXT NOT NULL,
            cutover_status TEXT NOT NULL DEFAULT 'dual_run',
            owner_role TEXT,
            snapshot_mode TEXT NOT NULL DEFAULT 'none',
            snapshot_fields TEXT NOT NULL DEFAULT '[]',
            fallback_mode TEXT NOT NULL DEFAULT 'import_only',
            cutover_target_percent REAL NOT NULL DEFAULT 100,
            native_usage_percent REAL NOT NULL DEFAULT 0,
            fallback_usage_count INTEGER NOT NULL DEFAULT 0,
            open_issue_count INTEGER NOT NULL DEFAULT 0,
            snapshot_consumer_count INTEGER NOT NULL DEFAULT 0,
            native_ready INTEGER NOT NULL DEFAULT 0,
            dual_run_complete INTEGER NOT NULL DEFAULT 0,
            fallback_drill_complete INTEGER NOT NULL DEFAULT 0,
            operator_trained INTEGER NOT NULL DEFAULT 0,
            last_drill_at TEXT,
            last_review_at TEXT,
            rollback_instructions TEXT,
            notes TEXT,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS operations_cutover_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workflow_key TEXT NOT NULL,
            event_type TEXT NOT NULL,
            actor TEXT,
            note TEXT,
            event_value TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(workflow_key) REFERENCES operations_cutover_workflows(workflow_key)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_operations_cutover_events_workflow_type
        ON operations_cutover_events(workflow_key, event_type, created_at)
        """
    )
    cutover_columns = {row[1] for row in conn.execute("PRAGMA table_info(operations_cutover_workflows)").fetchall()}
    for column, declaration in {
        "cutover_target_percent": "REAL NOT NULL DEFAULT 100",
        "native_usage_percent": "REAL NOT NULL DEFAULT 0",
        "fallback_usage_count": "INTEGER NOT NULL DEFAULT 0",
        "open_issue_count": "INTEGER NOT NULL DEFAULT 0",
        "snapshot_consumer_count": "INTEGER NOT NULL DEFAULT 0",
        "last_review_at": "TEXT",
    }.items():
        if column not in cutover_columns:
            conn.execute(f"ALTER TABLE operations_cutover_workflows ADD COLUMN {column} {declaration}")
    for item in OPERATIONS_CUTOVER_DEFAULTS:
        exists = conn.execute(
            "SELECT 1 FROM operations_cutover_workflows WHERE workflow_key = ?",
            (item["workflow_key"],),
        ).fetchone()
        if exists:
            continue
        conn.execute(
            """
            INSERT INTO operations_cutover_workflows (
                workflow_key,
                label,
                native_surface,
                spreadsheet_source,
                cutover_status,
                owner_role,
                snapshot_mode,
                snapshot_fields,
                fallback_mode,
                cutover_target_percent,
                native_usage_percent,
                fallback_usage_count,
                open_issue_count,
                snapshot_consumer_count,
                native_ready,
                dual_run_complete,
                fallback_drill_complete,
                operator_trained,
                last_review_at,
                rollback_instructions,
                notes,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item["workflow_key"],
                item["label"],
                item["native_surface"],
                item["spreadsheet_source"],
                item["cutover_status"],
                item["owner_role"],
                item["snapshot_mode"],
                json.dumps(item["snapshot_fields"]),
                item["fallback_mode"],
                item["cutover_target_percent"],
                item["native_usage_percent"],
                item["fallback_usage_count"],
                item["open_issue_count"],
                item["snapshot_consumer_count"],
                1 if item["native_ready"] else 0,
                1 if item["dual_run_complete"] else 0,
                1 if item["fallback_drill_complete"] else 0,
                1 if item["operator_trained"] else 0,
                item["last_review_at"],
                item["rollback_instructions"],
                item["notes"],
                _utc_now_iso(),
            ),
        )
    conn.commit()


def record_operations_cutover_event(
    conn: sqlite3.Connection,
    *,
    workflow_key: str,
    event_type: str,
    actor: str | None = None,
    note: str | None = None,
    event_value: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    _ensure_operations_columns(conn)
    exists = conn.execute(
        "SELECT 1 FROM operations_cutover_workflows WHERE workflow_key = ?",
        (workflow_key,),
    ).fetchone()
    if exists is None:
        raise ValueError(f"Unknown cutover workflow: {workflow_key}")
    event_time = created_at or _utc_now_iso()
    cursor = conn.execute(
        """
        INSERT INTO operations_cutover_events (
            workflow_key,
            event_type,
            actor,
            note,
            event_value,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (workflow_key, event_type, actor, note, event_value, event_time),
    )
    conn.commit()
    row = conn.execute(
        "SELECT * FROM operations_cutover_events WHERE id = ?",
        (int(cursor.lastrowid),),
    ).fetchone()
    return {
        "id": int(row["id"]),
        "workflowKey": row["workflow_key"],
        "eventType": row["event_type"],
        "actor": row["actor"],
        "note": row["note"],
        "eventValue": row["event_value"],
        "createdAt": row["created_at"],
    }


def list_operations_cutover_events(
    conn: sqlite3.Connection,
    *,
    workflow_key: str | None = None,
    event_type: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    _ensure_operations_columns(conn)
    filters: list[str] = []
    params: list[Any] = []
    if workflow_key:
        filters.append("workflow_key = ?")
        params.append(workflow_key)
    if event_type:
        filters.append("event_type = ?")
        params.append(event_type)
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    rows = conn.execute(
        f"""
        SELECT *
        FROM operations_cutover_events
        {where}
        ORDER BY created_at DESC, id DESC
        LIMIT ?
        """,
        [*params, int(limit)],
    ).fetchall()
    return [
        {
            "id": int(row["id"]),
            "workflowKey": row["workflow_key"],
            "eventType": row["event_type"],
            "actor": row["actor"],
            "note": row["note"],
            "eventValue": row["event_value"],
            "createdAt": row["created_at"],
        }
        for row in rows
    ]


def _computed_cutover_metrics(conn: sqlite3.Connection, *, workflow_key: str) -> dict[str, Any]:
    _ensure_operations_columns(conn)
    events = list_operations_cutover_events(conn, workflow_key=workflow_key, limit=500)
    last_review_at = next(
        (item["createdAt"] for item in events if item["eventType"] == "review"),
        None,
    )
    last_drill_at = next(
        (item["createdAt"] for item in events if item["eventType"] == "fallback_drill"),
        None,
    )
    fallback_usage_count = sum(1 for item in events if item["eventType"] == "fallback_use")
    snapshot_consumer_count = len(
        {
            str(item["eventValue"]).strip().lower()
            for item in events
            if item["eventType"] == "snapshot_issued" and str(item.get("eventValue") or "").strip()
        }
    )

    if workflow_key == "dispatch_execution":
        board_rows = list_job_operations_board(conn)
        total_jobs = len(board_rows)
        native_jobs = sum(
            1
            for row in board_rows
            if row["jobStatus"] in {"planned", "overridden", "override_required", "blocked"}
            and (row["truckIds"] or row["workerNames"] or row["segmentCount"] > 0)
        )
        native_usage_percent = (native_jobs / total_jobs * 100.0) if total_jobs else 0.0
        open_issue_count = sum(
            1 for row in board_rows if row["jobStatus"] in {"blocked", "override_required"}
        )
    elif workflow_key == "maintenance_compliance":
        readiness = list_operational_readiness_items(conn)
        vehicle_rows = conn.execute(
            "SELECT COUNT(*) AS count FROM trucks WHERE active = 1"
        ).fetchone()
        active_vehicles = int(vehicle_rows["count"] or 0)
        covered_vehicles = int(
            conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM vehicle_details
                WHERE COALESCE(rego_expiry, '') != ''
                  AND COALESCE(coi_due, '') != ''
                  AND COALESCE(next_service, '') != ''
                """
            ).fetchone()["count"]
            or 0
        )
        native_usage_percent = (
            min(covered_vehicles, active_vehicles) / active_vehicles * 100.0
            if active_vehicles
            else 0.0
        )
        open_issue_count = sum(1 for item in readiness if item["status"] == "blocked")
    elif workflow_key == "labor_planning":
        reconciliation = list_labor_reconciliation(conn)
        total_rows = len(reconciliation)
        native_rows = sum(1 for item in reconciliation if item["status"] in {"matched", "planned_only"})
        native_usage_percent = (native_rows / total_rows * 100.0) if total_rows else 0.0
        open_issue_count = sum(
            1 for item in reconciliation if item["status"] in {"planned_only", "imported_only"}
        )
    elif workflow_key == "inventory_coordination":
        shipment_counts = conn.execute(
            """
            SELECT
                COUNT(*) AS total_shipments,
                SUM(CASE WHEN segment_id IS NOT NULL THEN 1 ELSE 0 END) AS segment_linked
            FROM shipments
            """
        ).fetchone()
        total_shipments = int(shipment_counts["total_shipments"] or 0)
        segment_linked = int(shipment_counts["segment_linked"] or 0)
        native_usage_percent = (segment_linked / total_shipments * 100.0) if total_shipments else 0.0
        open_issue_count = max(total_shipments - segment_linked, 0)
    else:
        native_usage_percent = 0.0
        open_issue_count = 0

    return {
        "nativeUsagePercent": round(native_usage_percent, 1),
        "fallbackUsageCount": fallback_usage_count,
        "openIssueCount": open_issue_count,
        "snapshotConsumerCount": snapshot_consumer_count,
        "lastReviewAt": last_review_at,
        "lastDrillAt": last_drill_at,
    }


def _transition_candidate(row: dict[str, Any]) -> dict[str, Any]:
    current = str(row["cutoverStatus"])
    checklist = row["checklist"]
    metrics = row["metrics"]
    reasons: list[str] = []
    if not checklist["nativeReady"]:
        reasons.append("native workflow not marked ready")
    if not checklist["dualRunComplete"]:
        reasons.append("dual-run not complete")
    if not checklist["fallbackDrillComplete"]:
        reasons.append("fallback drill not complete")
    if not checklist["operatorTrained"]:
        reasons.append("operator training not complete")
    if metrics["nativeUsagePercent"] < metrics["cutoverTargetPercent"]:
        reasons.append("native usage below target")
    if metrics["openIssueCount"] > 0:
        reasons.append("open issues still present")

    if current == "dual_run" and not reasons:
        return {
            "targetStatus": "native_primary",
            "eligible": True,
            "reason": "All checklist gates complete and target met.",
        }
    if current == "native_primary" and not reasons and metrics["fallbackUsageCount"] == 0:
        return {
            "targetStatus": "fallback_only",
            "eligible": True,
            "reason": "Native-primary stable with no fallback usage and no open issues.",
        }
    if current == "native_primary" and not reasons:
        reasons.append("fallback usage still recorded")
    if current in {"fallback_only", "sheet_primary"}:
        return {
            "targetStatus": current,
            "eligible": False,
            "reason": "No automatic transition recommended from current state.",
        }
    return {
        "targetStatus": current,
        "eligible": False,
        "reason": "; ".join(reasons) if reasons else "No transition recommended yet.",
    }


def _event_sort_key(event: dict[str, Any]) -> tuple[str, int]:
    return (str(event.get("createdAt") or ""), int(event.get("id") or 0))


def _latest_matching_event(
    events: Sequence[dict[str, Any]],
    *,
    event_type: str,
    target_status: str,
    after: tuple[str, int] | None = None,
) -> dict[str, Any] | None:
    matched = [
        event
        for event in events
        if event.get("eventType") == event_type
        and str(event.get("eventValue") or "") == target_status
        and (after is None or _event_sort_key(event) >= after)
    ]
    if not matched:
        return None
    return max(matched, key=_event_sort_key)


def _cutover_approval_state(
    conn: sqlite3.Connection,
    *,
    workflow_key: str,
    target_status: str | None,
) -> dict[str, Any]:
    approval = {
        "targetStatus": target_status,
        "status": "not_required" if not target_status else "not_requested",
        "requestPending": False,
        "approvalSatisfied": False,
        "blockedByApproval": False,
        "requestedAt": None,
        "requestedBy": None,
        "requestNote": None,
        "approvedAt": None,
        "approvedBy": None,
        "approvalNote": None,
        "rejectedAt": None,
        "rejectedBy": None,
        "rejectionNote": None,
    }
    if not target_status:
        return approval

    events = list_operations_cutover_events(conn, workflow_key=workflow_key, limit=500)
    latest_request = _latest_matching_event(
        events,
        event_type="promotion_requested",
        target_status=target_status,
    )
    if latest_request is None:
        approval["blockedByApproval"] = True
        return approval

    approval.update(
        {
            "requestedAt": latest_request["createdAt"],
            "requestedBy": latest_request["actor"],
            "requestNote": latest_request["note"],
        }
    )
    request_marker = _event_sort_key(latest_request)
    latest_approval = _latest_matching_event(
        events,
        event_type="promotion_approved",
        target_status=target_status,
        after=request_marker,
    )
    latest_rejection = _latest_matching_event(
        events,
        event_type="promotion_rejected",
        target_status=target_status,
        after=request_marker,
    )

    if latest_rejection and (
        latest_approval is None or _event_sort_key(latest_rejection) > _event_sort_key(latest_approval)
    ):
        approval.update(
            {
                "status": "rejected",
                "blockedByApproval": True,
                "rejectedAt": latest_rejection["createdAt"],
                "rejectedBy": latest_rejection["actor"],
                "rejectionNote": latest_rejection["note"],
            }
        )
        return approval

    if latest_approval is not None:
        approval.update(
            {
                "status": "approved",
                "approvalSatisfied": True,
                "approvedAt": latest_approval["createdAt"],
                "approvedBy": latest_approval["actor"],
                "approvalNote": latest_approval["note"],
            }
        )
        return approval

    approval.update(
        {
            "status": "requested",
            "requestPending": True,
            "blockedByApproval": True,
        }
    )
    return approval


def list_operations_cutover_workflows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    _ensure_operations_columns(conn)
    rows = conn.execute(
        """
        SELECT
            workflow_key,
            label,
            native_surface,
            spreadsheet_source,
            cutover_status,
            owner_role,
            snapshot_mode,
            snapshot_fields,
            fallback_mode,
            cutover_target_percent,
            native_usage_percent,
            fallback_usage_count,
            open_issue_count,
            snapshot_consumer_count,
            native_ready,
            dual_run_complete,
            fallback_drill_complete,
            operator_trained,
            last_drill_at,
            last_review_at,
            rollback_instructions,
            notes,
            updated_at
        FROM operations_cutover_workflows
        ORDER BY label
        """
    ).fetchall()
    payload: list[dict[str, Any]] = []
    for row in rows:
        checklist = {
            "nativeReady": bool(row["native_ready"]),
            "dualRunComplete": bool(row["dual_run_complete"]),
            "fallbackDrillComplete": bool(row["fallback_drill_complete"]),
            "operatorTrained": bool(row["operator_trained"]),
        }
        computed_metrics = _computed_cutover_metrics(conn, workflow_key=row["workflow_key"])
        metrics = {
            "cutoverTargetPercent": float(row["cutover_target_percent"] or 0),
            "nativeUsagePercent": computed_metrics["nativeUsagePercent"],
            "fallbackUsageCount": computed_metrics["fallbackUsageCount"],
            "openIssueCount": computed_metrics["openIssueCount"],
            "snapshotConsumerCount": computed_metrics["snapshotConsumerCount"],
            "lastReviewAt": computed_metrics["lastReviewAt"],
        }
        payload.append(
            {
                "workflowKey": row["workflow_key"],
                "label": row["label"],
                "nativeSurface": row["native_surface"],
                "spreadsheetSource": row["spreadsheet_source"],
                "cutoverStatus": row["cutover_status"],
                "ownerRole": row["owner_role"],
                "snapshotMode": row["snapshot_mode"],
                "snapshotFields": _deserialize_flags(row["snapshot_fields"]),
                "fallbackMode": row["fallback_mode"],
                "metrics": metrics,
                "checklist": checklist,
                "allChecksComplete": all(checklist.values()),
                "targetMet": metrics["nativeUsagePercent"] >= metrics["cutoverTargetPercent"]
                and metrics["openIssueCount"] == 0,
                "lastDrillAt": computed_metrics["lastDrillAt"],
                "rollbackInstructions": row["rollback_instructions"],
                "notes": row["notes"],
                "updatedAt": row["updated_at"],
            }
        )
    return payload


def _recommended_cutover_transition(
    row: dict[str, Any],
    *,
    approval: dict[str, Any],
) -> dict[str, Any]:
    candidate = _transition_candidate(row)
    target_status = candidate["targetStatus"]
    if not candidate["eligible"]:
        return {
            "recommendedStatus": target_status,
            "actionable": False,
            "reason": candidate["reason"],
            "approvalRequired": False,
            "approvalSatisfied": False,
            "blockedByApproval": False,
        }
    if approval["status"] == "not_requested":
        return {
            "recommendedStatus": target_status,
            "actionable": False,
            "reason": "Evidence gate passed. approval path not started; ops manager promotion request is still required.",
            "approvalRequired": True,
            "approvalSatisfied": False,
            "blockedByApproval": True,
        }
    if approval["status"] == "requested":
        return {
            "recommendedStatus": target_status,
            "actionable": False,
            "reason": "Promotion requested. Awaiting commercial owner approval.",
            "approvalRequired": True,
            "approvalSatisfied": False,
            "blockedByApproval": True,
        }
    if approval["status"] == "rejected":
        return {
            "recommendedStatus": target_status,
            "actionable": False,
            "reason": "Promotion rejected. Address feedback and submit a new request.",
            "approvalRequired": True,
            "approvalSatisfied": False,
            "blockedByApproval": True,
        }
    return {
        "recommendedStatus": target_status,
        "actionable": True,
        "reason": candidate["reason"] + " Approval chain complete.",
        "approvalRequired": True,
        "approvalSatisfied": True,
        "blockedByApproval": False,
    }


def list_operations_cutover_rollout(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = list_operations_cutover_workflows(conn)
    payload: list[dict[str, Any]] = []
    for row in rows:
        candidate = _transition_candidate(row)
        approval = _cutover_approval_state(
            conn,
            workflow_key=row["workflowKey"],
            target_status=candidate["targetStatus"] if candidate["eligible"] else None,
        )
        recommendation = _recommended_cutover_transition(row, approval=approval)
        enriched = dict(row)
        enriched["approval"] = approval
        enriched["recommendation"] = recommendation
        payload.append(enriched)
    return payload


def request_operations_cutover_promotion(
    conn: sqlite3.Connection,
    *,
    workflow_key: str,
    actor: str,
    note: str | None = None,
    target_status: str | None = None,
) -> dict[str, Any]:
    rollout_rows = list_operations_cutover_rollout(conn)
    current = next((row for row in rollout_rows if row["workflowKey"] == workflow_key), None)
    if current is None:
        raise ValueError(f"Unknown cutover workflow: {workflow_key}")
    candidate = _transition_candidate(current)
    if not candidate["eligible"]:
        raise ValueError(candidate["reason"])
    requested_status = target_status or candidate["targetStatus"]
    if requested_status != candidate["targetStatus"]:
        raise ValueError("Requested promotion target does not match the current recommendation.")
    if not str(actor).strip():
        raise ValueError("Promotion request actor is required.")
    if current["approval"]["status"] == "requested":
        raise ValueError("Promotion request is already pending approval.")
    if current["approval"]["status"] == "approved":
        raise ValueError("Promotion is already approved and ready to apply.")
    record_operations_cutover_event(
        conn,
        workflow_key=workflow_key,
        event_type="promotion_requested",
        actor=actor.strip(),
        note=note or "Promotion requested from Fleet cutover admin.",
        event_value=requested_status,
    )
    return next(
        row for row in list_operations_cutover_rollout(conn) if row["workflowKey"] == workflow_key
    )


def approve_operations_cutover_promotion(
    conn: sqlite3.Connection,
    *,
    workflow_key: str,
    actor: str,
    note: str | None = None,
    target_status: str | None = None,
) -> dict[str, Any]:
    rollout_rows = list_operations_cutover_rollout(conn)
    current = next((row for row in rollout_rows if row["workflowKey"] == workflow_key), None)
    if current is None:
        raise ValueError(f"Unknown cutover workflow: {workflow_key}")
    candidate = _transition_candidate(current)
    if not candidate["eligible"]:
        raise ValueError(candidate["reason"])
    approval_target = target_status or candidate["targetStatus"]
    if approval_target != candidate["targetStatus"]:
        raise ValueError("Approval target does not match the current recommendation.")
    if current["approval"]["status"] != "requested":
        raise ValueError("Promotion request is required before approval.")
    if not str(actor).strip():
        raise ValueError("Promotion approval actor is required.")
    record_operations_cutover_event(
        conn,
        workflow_key=workflow_key,
        event_type="promotion_approved",
        actor=actor.strip(),
        note=note or "Promotion approved.",
        event_value=approval_target,
    )
    return next(
        row for row in list_operations_cutover_rollout(conn) if row["workflowKey"] == workflow_key
    )


def reject_operations_cutover_promotion(
    conn: sqlite3.Connection,
    *,
    workflow_key: str,
    actor: str,
    note: str,
    target_status: str | None = None,
) -> dict[str, Any]:
    rollout_rows = list_operations_cutover_rollout(conn)
    current = next((row for row in rollout_rows if row["workflowKey"] == workflow_key), None)
    if current is None:
        raise ValueError(f"Unknown cutover workflow: {workflow_key}")
    candidate = _transition_candidate(current)
    if not candidate["eligible"]:
        raise ValueError(candidate["reason"])
    rejection_target = target_status or candidate["targetStatus"]
    if rejection_target != candidate["targetStatus"]:
        raise ValueError("Rejection target does not match the current recommendation.")
    if current["approval"]["status"] != "requested":
        raise ValueError("Promotion request is required before rejection.")
    if not str(actor).strip():
        raise ValueError("Promotion rejection actor is required.")
    if not str(note).strip():
        raise ValueError("Promotion rejection note is required.")
    record_operations_cutover_event(
        conn,
        workflow_key=workflow_key,
        event_type="promotion_rejected",
        actor=actor.strip(),
        note=note.strip(),
        event_value=rejection_target,
    )
    return next(
        row for row in list_operations_cutover_rollout(conn) if row["workflowKey"] == workflow_key
    )


def upsert_operations_cutover_workflow(
    conn: sqlite3.Connection,
    *,
    workflow_key: str,
    cutover_status: str,
    owner_role: str | None,
    snapshot_mode: str,
    snapshot_fields: Sequence[str],
    fallback_mode: str,
    cutover_target_percent: float,
    native_ready: bool,
    dual_run_complete: bool,
    fallback_drill_complete: bool,
    operator_trained: bool,
    rollback_instructions: str | None,
    notes: str | None,
) -> dict[str, Any]:
    _ensure_operations_columns(conn)
    existing = conn.execute(
        """
        SELECT workflow_key, label, native_surface, spreadsheet_source
        FROM operations_cutover_workflows
        WHERE workflow_key = ?
        """,
        (workflow_key,),
    ).fetchone()
    if existing is None:
        raise ValueError(f"Unknown cutover workflow: {workflow_key}")
    conn.execute(
        """
        UPDATE operations_cutover_workflows
        SET
            cutover_status = ?,
            owner_role = ?,
            snapshot_mode = ?,
            snapshot_fields = ?,
            fallback_mode = ?,
            cutover_target_percent = ?,
            native_ready = ?,
            dual_run_complete = ?,
            fallback_drill_complete = ?,
            operator_trained = ?,
            rollback_instructions = ?,
            notes = ?,
            updated_at = ?
        WHERE workflow_key = ?
        """,
        (
            cutover_status,
            owner_role,
            snapshot_mode,
            json.dumps(sorted({str(item).strip() for item in snapshot_fields if str(item).strip()})),
            fallback_mode,
            float(cutover_target_percent),
            1 if native_ready else 0,
            1 if dual_run_complete else 0,
            1 if fallback_drill_complete else 0,
            1 if operator_trained else 0,
            rollback_instructions,
            notes,
            _utc_now_iso(),
            workflow_key,
        ),
    )
    conn.commit()
    for row in list_operations_cutover_workflows(conn):
        if row["workflowKey"] == workflow_key:
            return row
    raise RuntimeError(f"Failed to reload cutover workflow: {workflow_key}")


def apply_operations_cutover_recommendation(
    conn: sqlite3.Connection,
    *,
    workflow_key: str,
    actor: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    rollout_rows = list_operations_cutover_rollout(conn)
    current = next((row for row in rollout_rows if row["workflowKey"] == workflow_key), None)
    if current is None:
        raise ValueError(f"Unknown cutover workflow: {workflow_key}")
    recommendation = current["recommendation"]
    if not recommendation["actionable"]:
        raise ValueError(recommendation["reason"])
    if not current["approval"]["approvalSatisfied"]:
        raise ValueError("Commercial approval is required before applying this transition.")
    updated = upsert_operations_cutover_workflow(
        conn,
        workflow_key=workflow_key,
        cutover_status=recommendation["recommendedStatus"],
        owner_role=current.get("ownerRole"),
        snapshot_mode=current["snapshotMode"],
        snapshot_fields=current.get("snapshotFields", []),
        fallback_mode=current["fallbackMode"],
        cutover_target_percent=float(current["metrics"]["cutoverTargetPercent"]),
        native_ready=bool(current["checklist"]["nativeReady"]),
        dual_run_complete=bool(current["checklist"]["dualRunComplete"]),
        fallback_drill_complete=bool(current["checklist"]["fallbackDrillComplete"]),
        operator_trained=bool(current["checklist"]["operatorTrained"]),
        rollback_instructions=current.get("rollbackInstructions"),
        notes=current.get("notes"),
    )
    record_operations_cutover_event(
        conn,
        workflow_key=workflow_key,
        event_type="status_transition",
        actor=actor,
        note=note or recommendation["reason"],
        event_value=recommendation["recommendedStatus"],
    )
    refreshed = next(
        row for row in list_operations_cutover_rollout(conn) if row["workflowKey"] == workflow_key
    )
    return refreshed


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
    from analytics.db.inventory import list_inventory_requirements

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

    inventory_requirements = list_inventory_requirements(conn, segment_id=segment_id)
    inventory_shortages: list[dict[str, Any]] = []
    for requirement in inventory_requirements:
        shortage_quantity = float(requirement.get("shortageQuantity") or 0.0)
        if shortage_quantity <= 0:
            continue
        requirement_name = str(
            requirement.get("inventoryName")
            or requirement.get("requirementName")
            or f"requirement-{requirement.get('requirementId')}"
        )
        inventory_shortages.append(
            {
                "requirementId": requirement.get("requirementId"),
                "requirementName": requirement_name,
                "shortageQuantity": shortage_quantity,
                "requiredQuantity": float(requirement.get("requiredQuantity") or 0.0),
                "allocatedQuantity": float(requirement.get("allocatedQuantity") or 0.0),
                "substitutionAllowed": bool(requirement.get("substitutionAllowed")),
                "architecture": requirement.get("architecture") or "general",
            }
        )
        flag = f"segment:inventory_shortage:{requirement_name}:{shortage_quantity:g}"
        if bool(requirement.get("substitutionAllowed")):
            overrideable_flags.append(flag)
        else:
            blocking_flags.append(flag)

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
        "inventoryShortages": inventory_shortages,
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
                "inventoryShortages": readiness.get("inventoryShortages", []),
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


def ensure_worker_role(conn: sqlite3.Connection, *, name: str, description: str = "") -> int:
    _ensure_operations_columns(conn)
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Role name is required")
    conn.execute(
        """
        INSERT OR IGNORE INTO worker_roles (name, description) VALUES (?, ?)
        """,
        (clean_name, description.strip()),
    )
    conn.commit()
    row = conn.execute("SELECT id FROM worker_roles WHERE name = ?", (clean_name,)).fetchone()
    return int(row["id"])


def assign_worker_role(conn: sqlite3.Connection, *, worker_id: int, role_id: int) -> None:
    _ensure_operations_columns(conn)
    if conn.execute("SELECT 1 FROM workers WHERE id = ?", (worker_id,)).fetchone() is None:
        raise ValueError(f"Worker {worker_id} not found")
    if conn.execute("SELECT 1 FROM worker_roles WHERE id = ?", (role_id,)).fetchone() is None:
        raise ValueError(f"Role {role_id} not found")
    conn.execute(
        """
        INSERT INTO worker_role_assignments (worker_id, role_id, assigned_at)
        VALUES (?, ?, ?)
        ON CONFLICT(worker_id, role_id) DO UPDATE SET assigned_at = excluded.assigned_at
        """,
        (worker_id, role_id, _utc_now_iso()),
    )
    conn.commit()


def ensure_worker_compliance(conn: sqlite3.Connection, *, name: str, description: str = "") -> int:
    _ensure_operations_columns(conn)
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Compliance name is required")
    conn.execute(
        """
        INSERT OR IGNORE INTO worker_compliances (name, description) VALUES (?, ?)
        """,
        (clean_name, description.strip()),
    )
    conn.commit()
    row = conn.execute("SELECT id FROM worker_compliances WHERE name = ?", (clean_name,)).fetchone()
    return int(row["id"])


def assign_worker_compliance(
    conn: sqlite3.Connection,
    *,
    worker_id: int,
    compliance_id: int,
    expiry_date: str | None = None,
) -> None:
    _ensure_operations_columns(conn)
    if conn.execute("SELECT 1 FROM workers WHERE id = ?", (worker_id,)).fetchone() is None:
        raise ValueError(f"Worker {worker_id} not found")
    if conn.execute(
        "SELECT 1 FROM worker_compliances WHERE id = ?", (compliance_id,)
    ).fetchone() is None:
        raise ValueError(f"Compliance {compliance_id} not found")
    conn.execute(
        """
        INSERT INTO worker_compliance_assignments (worker_id, compliance_id, expiry_date, assigned_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(worker_id, compliance_id) DO UPDATE SET
            expiry_date = excluded.expiry_date,
            assigned_at = excluded.assigned_at
        """,
        (worker_id, compliance_id, expiry_date, _utc_now_iso()),
    )
    conn.commit()


def list_operational_readiness_items(
    conn: sqlite3.Connection,
    *,
    resource_type: str | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    _ensure_operations_columns(conn)
    policy = get_operations_policy(conn)
    items: list[dict[str, Any]] = []
    now = _utc_now()

    if resource_type in (None, "", "vehicle"):
        vehicle_rows = conn.execute(
            """
            SELECT
                t.truck_id,
                t.name,
                vd.rego_expiry,
                vd.coi_due,
                vd.next_service,
                vd.source_imported_at
            FROM trucks AS t
            LEFT JOIN vehicle_details AS vd ON vd.truck_id = t.truck_id
            ORDER BY t.truck_id
            """
        ).fetchall()
        for row in vehicle_rows:
            checks = (
                ("rego", row["rego_expiry"], policy["regoWarningDays"], False),
                ("coi", row["coi_due"], policy["coiWarningDays"], False),
                (
                    "service",
                    row["next_service"],
                    policy["serviceWarningDays"],
                    bool(policy["serviceOverrideAllowed"]),
                ),
            )
            for rule_type, due_raw, warning_days, overrideable in checks:
                due = _parse_iso_date(due_raw)
                if due is None:
                    continue
                if due < now:
                    item_status = "blocked"
                elif due <= now + timedelta(days=warning_days):
                    item_status = "warning"
                else:
                    continue
                items.append(
                    {
                        "resourceType": "vehicle",
                        "resourceId": str(row["truck_id"]),
                        "resourceName": row["name"] or str(row["truck_id"]),
                        "status": item_status,
                        "ruleType": rule_type,
                        "dueAt": due.date().isoformat(),
                        "overrideable": bool(overrideable and item_status == "blocked"),
                        "sourceImportedAt": row["source_imported_at"],
                        "details": f"{rule_type.upper()} {'expired' if item_status == 'blocked' else 'due soon'}",
                    }
                )

    if resource_type in (None, "", "worker"):
        worker_rows = conn.execute(
            """
            SELECT
                w.id AS worker_id,
                w.name AS worker_name,
                wc.name AS compliance_name,
                wca.expiry_date,
                w.source_imported_at
            FROM worker_compliance_assignments AS wca
            JOIN workers AS w ON w.id = wca.worker_id
            JOIN worker_compliances AS wc ON wc.id = wca.compliance_id
            ORDER BY w.name, wc.name
            """
        ).fetchall()
        for row in worker_rows:
            due = _parse_iso_date(row["expiry_date"])
            if due is None:
                continue
            if due < now:
                item_status = "blocked"
            elif due <= now + timedelta(days=policy["complianceWarningDays"]):
                item_status = "warning"
            else:
                continue
            items.append(
                {
                    "resourceType": "worker",
                    "resourceId": str(row["worker_id"]),
                    "resourceName": row["worker_name"],
                    "status": item_status,
                    "ruleType": "compliance",
                    "dueAt": due.date().isoformat(),
                    "overrideable": False,
                    "sourceImportedAt": row["source_imported_at"],
                    "details": row["compliance_name"],
                }
            )

    if status:
        items = [item for item in items if item["status"] == status]

    status_rank = {"blocked": 0, "warning": 1}
    return sorted(
        items,
        key=lambda item: (
            status_rank.get(str(item["status"]), 9),
            str(item["dueAt"] or ""),
            str(item["resourceType"]),
            str(item["resourceName"]),
            str(item["ruleType"]),
        ),
    )


def _date_in_range(
    value: str | None,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> bool:
    if not value:
        return False
    parsed = _parse_iso_datetime(value) or _parse_iso_date(value)
    if parsed is None:
        return False
    current_date = parsed.date().isoformat()
    if start_date and current_date < start_date:
        return False
    if end_date and current_date > end_date:
        return False
    return True


def list_planned_labor_assignments(
    conn: sqlite3.Connection,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    worker_id: int | None = None,
    truck_id: str | None = None,
) -> list[dict[str, Any]]:
    segments = list_segment_readiness(conn)
    payload: list[dict[str, Any]] = []
    for segment in segments:
        if not _date_in_range(segment.get("plannedStart"), start_date=start_date, end_date=end_date):
            if start_date or end_date:
                continue
        if truck_id and not any(
            str(assignment["truckId"]).strip() == str(truck_id).strip()
            for assignment in segment["truckAssignments"]
        ):
            continue
        truck_ids = [assignment["truckId"] for assignment in segment["truckAssignments"]]
        truck_names = [assignment.get("truckName") for assignment in segment["truckAssignments"]]
        for assignment in segment["workerAssignments"]:
            current_worker_id = int(assignment["workerId"])
            if worker_id is not None and current_worker_id != int(worker_id):
                continue
            payload.append(
                {
                    "segmentId": segment["segmentId"],
                    "jobId": segment["jobId"],
                    "jobClient": segment.get("jobClient"),
                    "segmentSequence": segment["segmentSequence"],
                    "workerId": current_worker_id,
                    "workerName": assignment.get("workerName"),
                    "roleId": assignment.get("roleId"),
                    "truckIds": truck_ids,
                    "truckNames": [name for name in truck_names if name],
                    "plannedStart": segment.get("plannedStart"),
                    "plannedEnd": segment.get("plannedEnd"),
                    "fromLocation": segment.get("fromLocation") or segment.get("jobOrigin"),
                    "toLocation": segment.get("toLocation") or segment.get("jobDestination"),
                    "assignmentStatus": segment.get("assignmentStatus"),
                }
            )
    return sorted(
        payload,
        key=lambda item: (
            str(item.get("plannedStart") or ""),
            str(item.get("workerName") or ""),
            str(item.get("jobId") or ""),
            str(item.get("segmentSequence") or ""),
        ),
    )


def list_labor_reconciliation(
    conn: sqlite3.Connection,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict[str, Any]]:
    planned_rows = list_planned_labor_assignments(conn, start_date=start_date, end_date=end_date)
    imported_rows = conn.execute(
        """
        SELECT
            ds.id,
            ds.shift_date,
            ds.truck_id,
            ds.worker_id,
            w.name AS worker_name,
            ds.shift_window_start,
            ds.shift_window_end,
            ds.shift_start,
            ds.shift_end,
            ds.source
        FROM driver_shifts AS ds
        LEFT JOIN workers AS w ON w.id = ds.worker_id
        ORDER BY ds.shift_date, w.name, ds.truck_id
        """
    ).fetchall()

    imported_filtered = [
        row
        for row in imported_rows
        if _date_in_range(row["shift_date"], start_date=start_date, end_date=end_date)
        or (not start_date and not end_date)
    ]

    def planned_keys(item: dict[str, Any]) -> set[tuple[str, int, str]]:
        date_value = (_parse_iso_datetime(item.get("plannedStart")) or _parse_iso_date(item.get("plannedStart")))
        if date_value is None:
            return set()
        return {
            (date_value.date().isoformat(), int(item["workerId"]), str(truck_id))
            for truck_id in item.get("truckIds", []) or [""]
        }

    imported_key_map: dict[tuple[str, int | None, str], list[sqlite3.Row]] = {}
    for row in imported_filtered:
        key = (str(row["shift_date"]), row["worker_id"], str(row["truck_id"] or ""))
        imported_key_map.setdefault(key, []).append(row)

    matched_import_ids: set[int] = set()
    reconciliation: list[dict[str, Any]] = []

    for item in planned_rows:
        keys = planned_keys(item)
        matched = False
        for key in keys:
            rows = imported_key_map.get(key, [])
            if rows:
                matched = True
                matched_import_ids.update(int(row["id"]) for row in rows)
        reconciliation.append(
            {
                "status": "matched" if matched else "planned_only",
                "workerId": item["workerId"],
                "workerName": item.get("workerName"),
                "truckIds": item.get("truckIds", []),
                "jobId": item.get("jobId"),
                "segmentId": item.get("segmentId"),
                "plannedStart": item.get("plannedStart"),
                "plannedEnd": item.get("plannedEnd"),
                "shiftDate": (
                    (_parse_iso_datetime(item.get("plannedStart")) or _parse_iso_date(item.get("plannedStart"))).date().isoformat()
                    if (_parse_iso_datetime(item.get("plannedStart")) or _parse_iso_date(item.get("plannedStart")))
                    else None
                ),
                "source": "planned_segments",
            }
        )

    for row in imported_filtered:
        if int(row["id"]) in matched_import_ids:
            continue
        reconciliation.append(
            {
                "status": "imported_only",
                "workerId": row["worker_id"],
                "workerName": row["worker_name"],
                "truckIds": [row["truck_id"]] if row["truck_id"] else [],
                "jobId": None,
                "segmentId": None,
                "plannedStart": row["shift_window_start"] or row["shift_start"],
                "plannedEnd": row["shift_window_end"] or row["shift_end"],
                "shiftDate": row["shift_date"],
                "source": row["source"],
            }
        )

    status_rank = {"planned_only": 0, "imported_only": 1, "matched": 2}
    return sorted(
        reconciliation,
        key=lambda item: (
            status_rank.get(str(item["status"]), 9),
            str(item.get("shiftDate") or ""),
            str(item.get("workerName") or ""),
        ),
    )


def list_job_operations_board(
    conn: sqlite3.Connection,
    *,
    job_id: int | None = None,
) -> list[dict[str, Any]]:
    segments = list_segment_readiness(conn, job_id=job_id)
    from analytics.db.inventory import list_segment_inventory_coordination

    inventory_by_segment = {
        int(item["segmentId"]): item
        for item in list_segment_inventory_coordination(conn, job_id=job_id)
    }
    jobs: dict[int, dict[str, Any]] = {}
    for segment in segments:
        current_job_id = int(segment["jobId"])
        record = jobs.setdefault(
            current_job_id,
            {
                "jobId": current_job_id,
                "jobClient": segment.get("jobClient"),
                "jobOrigin": segment.get("jobOrigin"),
                "jobDestination": segment.get("jobDestination"),
                "segmentCount": 0,
                "plannedStart": None,
                "plannedEnd": None,
                "statuses": [],
                "warningCount": 0,
                "blockingCount": 0,
                "overrideableCount": 0,
                "truckIds": set(),
                "workerNames": set(),
                "inventoryNames": set(),
                "supplierNames": set(),
                "requiredQuantity": 0.0,
                "allocatedQuantity": 0.0,
                "approvedSubstitutionQuantity": 0.0,
                "shortageQuantity": 0.0,
                "inventoryShortageCount": 0,
                "pendingSubstitutionCount": 0,
                "executionStages": set(),
                "segments": [],
            },
        )
        record["segmentCount"] += 1
        if segment.get("plannedStart") and (
            record["plannedStart"] is None or str(segment["plannedStart"]) < str(record["plannedStart"])
        ):
            record["plannedStart"] = segment["plannedStart"]
        if segment.get("plannedEnd") and (
            record["plannedEnd"] is None or str(segment["plannedEnd"]) > str(record["plannedEnd"])
        ):
            record["plannedEnd"] = segment["plannedEnd"]
        record["statuses"].append(segment["assignmentStatus"])
        record["warningCount"] += len(segment["warningFlags"])
        record["blockingCount"] += len(segment["blockingFlags"])
        record["overrideableCount"] += len(segment["overrideableFlags"])
        record["truckIds"].update(
            str(item["truckId"]) for item in segment["truckAssignments"] if item.get("truckId")
        )
        record["workerNames"].update(
            str(item["workerName"]) for item in segment["workerAssignments"] if item.get("workerName")
        )
        coordination = inventory_by_segment.get(int(segment["segmentId"]))
        if coordination:
            record["inventoryNames"].update(coordination.get("inventoryNames", []))
            record["supplierNames"].update(coordination.get("supplierNames", []))
            record["requiredQuantity"] += float(coordination.get("requiredQuantity", 0.0) or 0.0)
            record["allocatedQuantity"] += float(coordination.get("allocatedQuantity", 0.0) or 0.0)
            record["approvedSubstitutionQuantity"] += float(
                coordination.get("approvedSubstitutionQuantity", 0.0) or 0.0
            )
            record["shortageQuantity"] += float(coordination.get("shortageQuantity", 0.0) or 0.0)
            record["inventoryShortageCount"] += int(coordination.get("shortageCount", 0) or 0)
            record["pendingSubstitutionCount"] += int(
                coordination.get("pendingSubstitutionCount", 0) or 0
            )
            record["executionStages"].update(coordination.get("executionStages", []) or [])
        record["segments"].append(
            {
                "segmentId": segment["segmentId"],
                "segmentSequence": segment["segmentSequence"],
                "fromLocation": segment.get("fromLocation") or segment.get("jobOrigin"),
                "toLocation": segment.get("toLocation") or segment.get("jobDestination"),
                "plannedStart": segment.get("plannedStart"),
                "plannedEnd": segment.get("plannedEnd"),
                "assignmentStatus": segment.get("assignmentStatus"),
                "warningCount": len(segment["warningFlags"]),
                "blockingCount": len(segment["blockingFlags"]),
                "overrideableCount": len(segment["overrideableFlags"]),
                "truckIds": [item["truckId"] for item in segment["truckAssignments"] if item.get("truckId")],
                "workerNames": [item["workerName"] for item in segment["workerAssignments"] if item.get("workerName")],
                "inventoryNames": coordination.get("inventoryNames", []) if coordination else [],
                "supplierNames": coordination.get("supplierNames", []) if coordination else [],
                "shipmentCount": coordination.get("shipmentCount", 0) if coordination else 0,
                "requirementCount": coordination.get("requirementCount", 0) if coordination else 0,
                "requiredQuantity": coordination.get("requiredQuantity", 0.0) if coordination else 0.0,
                "allocatedQuantity": coordination.get("allocatedQuantity", 0.0) if coordination else 0.0,
                "approvedSubstitutionQuantity": coordination.get("approvedSubstitutionQuantity", 0.0) if coordination else 0.0,
                "shortageQuantity": coordination.get("shortageQuantity", 0.0) if coordination else 0.0,
                "blockingShortageQuantity": coordination.get("blockingShortageQuantity", 0.0) if coordination else 0.0,
                "warningShortageQuantity": coordination.get("warningShortageQuantity", 0.0) if coordination else 0.0,
                "shortageCount": coordination.get("shortageCount", 0) if coordination else 0,
                "pendingSubstitutionCount": coordination.get("pendingSubstitutionCount", 0) if coordination else 0,
                "executionStages": coordination.get("executionStages", []) if coordination else [],
                "architectures": coordination.get("architectures", []) if coordination else [],
            }
        )

    def _job_status(statuses: list[str]) -> str:
        if any(status == "blocked" for status in statuses):
            return "blocked"
        if any(status == "override_required" for status in statuses):
            return "override_required"
        if any(status == "overridden" for status in statuses):
            return "overridden"
        if all(status == "planned" for status in statuses if statuses):
            return "planned"
        return "draft"

    payload: list[dict[str, Any]] = []
    for record in jobs.values():
        payload.append(
            {
                "jobId": record["jobId"],
                "jobClient": record["jobClient"],
                "jobOrigin": record["jobOrigin"],
                "jobDestination": record["jobDestination"],
                "segmentCount": record["segmentCount"],
                "plannedStart": record["plannedStart"],
                "plannedEnd": record["plannedEnd"],
                "jobStatus": _job_status(record["statuses"]),
                "warningCount": record["warningCount"],
                "blockingCount": record["blockingCount"],
                "overrideableCount": record["overrideableCount"],
                "truckIds": sorted(record["truckIds"]),
                "workerNames": sorted(record["workerNames"]),
                "inventoryNames": sorted(record["inventoryNames"]),
                "supplierNames": sorted(record["supplierNames"]),
                "requiredQuantity": round(record["requiredQuantity"], 2),
                "allocatedQuantity": round(record["allocatedQuantity"], 2),
                "approvedSubstitutionQuantity": round(record["approvedSubstitutionQuantity"], 2),
                "shortageQuantity": round(record["shortageQuantity"], 2),
                "inventoryShortageCount": record["inventoryShortageCount"],
                "pendingSubstitutionCount": record["pendingSubstitutionCount"],
                "executionStages": sorted(record["executionStages"]),
                "segments": sorted(record["segments"], key=lambda item: int(item["segmentSequence"])),
            }
        )

    status_rank = {"blocked": 0, "override_required": 1, "overridden": 2, "planned": 3, "draft": 4}
    return sorted(
        payload,
        key=lambda item: (
            status_rank.get(str(item["jobStatus"]), 9),
            str(item["plannedStart"] or ""),
            str(item["jobId"]),
        ),
    )


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
    "approve_operations_cutover_promotion",
    "assign_segment_resources",
    "assign_worker_compliance",
    "assign_worker_role",
    "ensure_segment",
    "ensure_worker_compliance",
    "ensure_worker_role",
    "evaluate_segment_readiness",
    "get_operations_policy",
    "list_job_operations_board",
    "list_labor_reconciliation",
    "list_operations_cutover_events",
    "list_operations_cutover_rollout",
    "list_operations_cutover_workflows",
    "list_operational_readiness_items",
    "list_operational_conflicts",
    "list_planned_labor_assignments",
    "list_segments_for_truck",
    "list_segments_for_worker",
    "list_segment_readiness",
    "list_truck_assignment_summary",
    "list_worker_assignment_summary",
    "apply_operations_cutover_recommendation",
    "reject_operations_cutover_promotion",
    "record_operations_cutover_event",
    "request_operations_cutover_promotion",
    "upsert_operations_cutover_workflow",
    "update_operations_policy",
]
