"""Manager-facing operations diary helpers."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import UTC, date, datetime, timedelta
from typing import Any

from analytics.db.shipments import fetch_driver_shifts
from analytics.operations_assignment import (
    list_job_operations_board,
    list_labor_reconciliation,
    list_planned_labor_assignments,
    list_segment_readiness,
)
from analytics.db.schema import ensure_dashboard_tables

DIARY_TASK_SCOPES = ("day", "week", "job", "segment")
DIARY_TASK_STATUSES = ("open", "in_progress", "blocked", "done")
CUSTOMER_INVOICE_STATUSES = (
    "not_ready",
    "ready_to_invoice",
    "partially_invoiced",
    "invoiced",
    "reconciliation_warning",
)
SUBCONTRACTOR_BILL_STATUSES = (
    "no_bill_expected",
    "awaiting_bill",
    "bill_received",
    "bill_reconciled",
    "bill_exception",
)
OBSERVER_AUTHORITY_CLASSES = (
    "operational_truth",
    "reviewed_summary",
    "observed_actual",
    "downstream_projection",
)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _date_only(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        try:
            return date.fromisoformat(text[:10]).isoformat()
        except ValueError:
            return None


def _parse_dateish(value: Any) -> date | None:
    text = _date_only(value)
    return date.fromisoformat(text) if text else None


def _date_range(mode: str, anchor_date: str) -> tuple[str, str]:
    current = date.fromisoformat(anchor_date)
    if mode == "week":
        start = current - timedelta(days=current.weekday())
        end = start + timedelta(days=6)
        return start.isoformat(), end.isoformat()
    return current.isoformat(), current.isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _observer_row_to_event(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    record = dict(row)
    event_time = record["event_time"]
    return {
        "id": int(record["id"]),
        "surface": "observer",
        "eventId": record["event_id"],
        "sourceComponent": record["source_system"],
        "sourceSystem": record["source_system"],
        "sourceEntityId": record["source_entity_id"],
        "eventType": record["event_type"],
        "eventFamily": record["event_family"],
        "idempotencyKey": record["idempotency_key"],
        "correlationId": record["correlation_key"],
        "correlationKey": record["correlation_key"],
        "causationId": None,
        "actorRef": record["actor_ref"],
        "authorityClass": record["authority_class"],
        "summary": record["summary"],
        "status": record["status"],
        "objectRefs": json.loads(record["object_refs_json"]),
        "provenanceRefs": json.loads(record["provenance_refs_json"]),
        "evidenceRefs": json.loads(record["evidence_refs_json"]),
        "payload": json.loads(record["payload_json"]),
        "payloadHash": record["payload_hash"],
        "occurredAt": event_time,
        "eventTime": event_time,
        "ingestedAt": record["recorded_at"],
        "recordedAt": record["recorded_at"],
        "cursor": f"{event_time}|observer|{record['event_id']}",
    }


def _emit_observer_event(
    conn: sqlite3.Connection,
    *,
    source_entity_id: str,
    event_family: str,
    event_type: str,
    actor_ref: str | None,
    authority_class: str,
    summary: str,
    status: str | None,
    object_refs: dict[str, Any],
    provenance_refs: list[dict[str, Any]],
    evidence_refs: list[dict[str, Any]],
    payload: dict[str, Any],
    correlation_key: str | None = None,
    event_time: str | None = None,
) -> dict[str, Any]:
    ensure_dashboard_tables(conn)
    safe_authority = authority_class if authority_class in OBSERVER_AUTHORITY_CLASSES else "reviewed_summary"
    effective_event_time = event_time or _utc_now_iso()
    recorded_at = _utc_now_iso()
    object_refs_json = _json_dumps(object_refs)
    provenance_refs_json = _json_dumps(provenance_refs)
    evidence_refs_json = _json_dumps(evidence_refs)
    payload_json = _json_dumps(payload)
    payload_hash = hashlib.sha256(
        _json_dumps(
            {
                "eventFamily": event_family,
                "eventType": event_type,
                "summary": summary,
                "status": status,
                "objectRefs": object_refs,
                "payload": payload,
            }
        ).encode("utf-8")
    ).hexdigest()
    idempotency_key = f"corkysoft:observer:{event_family}:{source_entity_id}:{payload_hash}"
    event_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT OR IGNORE INTO observer_outbox_events (
            event_id,
            source_system,
            source_entity_id,
            event_family,
            event_type,
            idempotency_key,
            correlation_key,
            actor_ref,
            authority_class,
            summary,
            status,
            object_refs_json,
            provenance_refs_json,
            evidence_refs_json,
            payload_json,
            payload_hash,
            event_time,
            recorded_at
        ) VALUES (?, 'corkysoft', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_id,
            source_entity_id,
            event_family,
            event_type,
            idempotency_key,
            correlation_key,
            actor_ref,
            safe_authority,
            summary,
            status,
            object_refs_json,
            provenance_refs_json,
            evidence_refs_json,
            payload_json,
            payload_hash,
            effective_event_time,
            recorded_at,
        ),
    )
    row = conn.execute(
        "SELECT * FROM observer_outbox_events WHERE idempotency_key = ?",
        (idempotency_key,),
    ).fetchone()
    return _observer_row_to_event(row)


def list_observer_outbox_events(
    conn: sqlite3.Connection,
    *,
    limit: int = 100,
    event_family: str | None = None,
    job_id: int | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
) -> list[dict[str, Any]]:
    ensure_dashboard_tables(conn)
    filters: list[str] = []
    params: list[Any] = []
    if event_family:
        filters.append("event_family = ?")
        params.append(event_family)
    if start_time:
        filters.append("event_time >= ?")
        params.append(start_time)
    if end_time:
        filters.append("event_time <= ?")
        params.append(end_time)
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    fetch_limit = max(int(limit) * 4, int(limit))
    rows = conn.execute(
        f"""
        SELECT *
        FROM observer_outbox_events
        {where}
        ORDER BY recorded_at DESC, id DESC
        LIMIT ?
        """,
        [*params, fetch_limit],
    ).fetchall()
    payload = [_observer_row_to_event(row) for row in rows]
    if job_id is not None:
        target = int(job_id)
        payload = [
            row
            for row in payload
            if int(row["objectRefs"].get("job_id") or row["payload"].get("jobId") or -1) == target
            or target in [int(value) for value in row["objectRefs"].get("job_ids", [])]
        ]
    return payload[:limit]


def list_operations_diary_tasks(
    conn: sqlite3.Connection,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    job_id: int | None = None,
) -> list[dict[str, Any]]:
    ensure_dashboard_tables(conn)
    filters: list[str] = []
    params: list[Any] = []
    if start_date:
        filters.append("t.task_date >= ?")
        params.append(start_date)
    if end_date:
        filters.append("t.task_date <= ?")
        params.append(end_date)
    if job_id is not None:
        filters.append("t.job_id = ?")
        params.append(int(job_id))
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    rows = conn.execute(
        f"""
        SELECT
            t.*,
            j.client AS job_client,
            w.name AS worker_name,
            tr.name AS truck_name
        FROM operations_diary_tasks AS t
        LEFT JOIN jobs AS j ON j.id = t.job_id
        LEFT JOIN workers AS w ON w.id = t.assigned_worker_id
        LEFT JOIN trucks AS tr ON tr.truck_id = t.assigned_truck_id
        {where}
        ORDER BY t.task_date, COALESCE(t.planned_start, ''), t.id
        """,
        params,
    ).fetchall()
    return [dict(row) for row in rows]


def upsert_operations_diary_task(
    conn: sqlite3.Connection,
    *,
    task_id: int | None = None,
    task_date: str,
    title: str,
    task_scope: str = "day",
    task_type: str = "follow_up",
    status: str = "open",
    job_id: int | None = None,
    segment_id: int | None = None,
    assigned_worker_id: int | None = None,
    assigned_truck_id: str | None = None,
    planned_start: str | None = None,
    planned_end: str | None = None,
    notes: str | None = None,
    actor_ref: str | None = None,
) -> dict[str, Any]:
    ensure_dashboard_tables(conn)
    if task_scope not in DIARY_TASK_SCOPES:
        raise ValueError(f"Unsupported task scope: {task_scope}")
    if status not in DIARY_TASK_STATUSES:
        raise ValueError(f"Unsupported task status: {status}")
    now = _utc_now_iso()
    is_create = task_id is None
    if task_id is None:
        row_id = conn.execute(
            """
            INSERT INTO operations_diary_tasks (
                job_id, segment_id, task_date, task_scope, task_type, title, status,
                assigned_worker_id, assigned_truck_id, planned_start, planned_end,
                notes, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                segment_id,
                task_date,
                task_scope,
                task_type,
                title.strip(),
                status,
                assigned_worker_id,
                assigned_truck_id.strip() if assigned_truck_id else None,
                planned_start,
                planned_end,
                notes,
                now,
                now,
            ),
        ).lastrowid
    else:
        conn.execute(
            """
            UPDATE operations_diary_tasks
            SET
                job_id = ?,
                segment_id = ?,
                task_date = ?,
                task_scope = ?,
                task_type = ?,
                title = ?,
                status = ?,
                assigned_worker_id = ?,
                assigned_truck_id = ?,
                planned_start = ?,
                planned_end = ?,
                notes = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                job_id,
                segment_id,
                task_date,
                task_scope,
                task_type,
                title.strip(),
                status,
                assigned_worker_id,
                assigned_truck_id.strip() if assigned_truck_id else None,
                planned_start,
                planned_end,
                notes,
                now,
                int(task_id),
            ),
        )
        row_id = int(task_id)
    row = next(row for row in list_operations_diary_tasks(conn) if int(row["id"]) == int(row_id))
    _emit_observer_event(
        conn,
        source_entity_id=f"diary_task:{row_id}",
        event_family="diary_task_event",
        event_type="diary_task_event",
        actor_ref=actor_ref,
        authority_class="operational_truth",
        summary=f"Diary task {'created' if is_create else 'updated'}: {row['title']}",
        status=row.get("status"),
        object_refs={
            "task_id": int(row_id),
            "job_id": int(row["job_id"]) if row.get("job_id") is not None else None,
            "segment_id": int(row["segment_id"]) if row.get("segment_id") is not None else None,
            "worker_id": int(row["assigned_worker_id"]) if row.get("assigned_worker_id") is not None else None,
            "truck_id": row.get("assigned_truck_id"),
        },
        provenance_refs=[{"kind": "table_row", "table": "operations_diary_tasks", "rowId": int(row_id)}],
        evidence_refs=[],
        payload={
            "action": "created" if is_create else "updated",
            "taskId": int(row_id),
            "jobId": int(row["job_id"]) if row.get("job_id") is not None else None,
            "segmentId": int(row["segment_id"]) if row.get("segment_id") is not None else None,
            "taskDate": row.get("task_date"),
            "taskScope": row.get("task_scope"),
            "taskType": row.get("task_type"),
            "title": row.get("title"),
            "status": row.get("status"),
            "plannedStart": row.get("planned_start"),
            "plannedEnd": row.get("planned_end"),
            "notes": row.get("notes"),
        },
        correlation_key=f"job:{int(row['job_id'])}" if row.get("job_id") is not None else f"task:{int(row_id)}",
        event_time=row.get("updated_at") or now,
    )
    conn.commit()
    return row


def delete_operations_diary_task(conn: sqlite3.Connection, *, task_id: int, actor_ref: str | None = None) -> None:
    ensure_dashboard_tables(conn)
    existing = conn.execute("SELECT * FROM operations_diary_tasks WHERE id = ?", (int(task_id),)).fetchone()
    conn.execute("DELETE FROM operations_diary_tasks WHERE id = ?", (int(task_id),))
    if existing is not None:
        row = dict(existing)
        _emit_observer_event(
            conn,
            source_entity_id=f"diary_task:{int(task_id)}",
            event_family="diary_task_event",
            event_type="diary_task_event",
            actor_ref=actor_ref,
            authority_class="operational_truth",
            summary=f"Diary task deleted: {row.get('title')}",
            status="deleted",
            object_refs={
                "task_id": int(task_id),
                "job_id": int(row["job_id"]) if row.get("job_id") is not None else None,
                "segment_id": int(row["segment_id"]) if row.get("segment_id") is not None else None,
            },
            provenance_refs=[{"kind": "table_row", "table": "operations_diary_tasks", "rowId": int(task_id)}],
            evidence_refs=[],
            payload={
                "action": "deleted",
                "taskId": int(task_id),
                "jobId": int(row["job_id"]) if row.get("job_id") is not None else None,
                "segmentId": int(row["segment_id"]) if row.get("segment_id") is not None else None,
                "taskDate": row.get("task_date"),
                "taskScope": row.get("task_scope"),
                "taskType": row.get("task_type"),
                "title": row.get("title"),
                "status": row.get("status"),
                "notes": row.get("notes"),
            },
            correlation_key=f"job:{int(row['job_id'])}" if row.get("job_id") is not None else f"task:{int(task_id)}",
        )
    conn.commit()


def list_customer_invoice_reviews(conn: sqlite3.Connection, *, job_id: int | None = None) -> list[dict[str, Any]]:
    ensure_dashboard_tables(conn)
    params: list[Any] = []
    where = ""
    if job_id is not None:
        where = "WHERE cir.job_id = ?"
        params.append(int(job_id))
    rows = conn.execute(
        f"""
        SELECT cir.*, j.client AS job_client, j.origin, j.destination
        FROM customer_invoice_reviews AS cir
        JOIN jobs AS j ON j.id = cir.job_id
        {where}
        ORDER BY COALESCE(cir.invoice_date, ''), cir.job_id
        """,
        params,
    ).fetchall()
    return [dict(row) for row in rows]


def upsert_customer_invoice_review(
    conn: sqlite3.Connection,
    *,
    job_id: int,
    invoice_status: str,
    invoice_reference: str | None = None,
    invoice_date: str | None = None,
    invoice_amount: float | None = None,
    note: str | None = None,
    reviewed_by: str | None = None,
) -> dict[str, Any]:
    ensure_dashboard_tables(conn)
    if invoice_status not in CUSTOMER_INVOICE_STATUSES:
        raise ValueError(f"Unsupported customer invoice status: {invoice_status}")
    now = _utc_now_iso()
    resolved_at = (invoice_date or now) if invoice_status == "invoiced" else None
    conn.execute(
        """
        INSERT INTO customer_invoice_reviews (
            job_id, invoice_status, invoice_reference, invoice_date, invoice_amount,
            resolved_at, note, reviewed_by, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(job_id) DO UPDATE SET
            invoice_status = excluded.invoice_status,
            invoice_reference = excluded.invoice_reference,
            invoice_date = excluded.invoice_date,
            invoice_amount = excluded.invoice_amount,
            resolved_at = excluded.resolved_at,
            note = excluded.note,
            reviewed_by = excluded.reviewed_by,
            updated_at = excluded.updated_at
        """,
        (
            int(job_id),
            invoice_status,
            invoice_reference,
            invoice_date,
            invoice_amount,
            resolved_at,
            note,
            reviewed_by,
            now,
            now,
        ),
    )
    row = next(row for row in list_customer_invoice_reviews(conn, job_id=int(job_id)))
    _emit_observer_event(
        conn,
        source_entity_id=f"invoice_review:{int(row['id'])}",
        event_family="customer_invoice_review",
        event_type="customer_invoice_review",
        actor_ref=reviewed_by,
        authority_class="reviewed_summary",
        summary=f"Customer invoice review {invoice_status} for job {int(job_id)}",
        status=invoice_status,
        object_refs={"invoice_review_id": int(row["id"]), "job_id": int(job_id)},
        provenance_refs=[{"kind": "table_row", "table": "customer_invoice_reviews", "rowId": int(row["id"])}],
        evidence_refs=[{"kind": "table_row", "table": "jobs", "rowId": int(job_id)}],
        payload={
            "jobId": int(job_id),
            "invoiceReviewId": int(row["id"]),
            "invoiceStatus": invoice_status,
            "invoiceReference": invoice_reference,
            "invoiceDate": invoice_date,
            "invoiceAmount": invoice_amount,
            "note": note,
        },
        correlation_key=f"job:{int(job_id)}",
        event_time=row.get("updated_at") or now,
    )
    conn.commit()
    return row


def list_subcontractor_bill_reviews(
    conn: sqlite3.Connection,
    *,
    job_id: int | None = None,
) -> list[dict[str, Any]]:
    ensure_dashboard_tables(conn)
    params: list[Any] = []
    where = ""
    if job_id is not None:
        where = "WHERE sbr.job_id = ?"
        params.append(int(job_id))
    rows = conn.execute(
        f"""
        SELECT
            sbr.*,
            j.client AS job_client,
            s.company_name AS supplier_name
        FROM subcontractor_bill_reviews AS sbr
        JOIN jobs AS j ON j.id = sbr.job_id
        LEFT JOIN suppliers AS s ON s.id = sbr.supplier_id
        {where}
        ORDER BY COALESCE(sbr.bill_date, ''), sbr.job_id, sbr.id
        """,
        params,
    ).fetchall()
    return [dict(row) for row in rows]


def upsert_subcontractor_bill_review(
    conn: sqlite3.Connection,
    *,
    bill_id: int | None = None,
    job_id: int,
    bill_status: str,
    segment_id: int | None = None,
    supplier_id: int | None = None,
    bill_reference: str | None = None,
    bill_date: str | None = None,
    amount: float | None = None,
    note: str | None = None,
    reviewed_by: str | None = None,
) -> dict[str, Any]:
    ensure_dashboard_tables(conn)
    if bill_status not in SUBCONTRACTOR_BILL_STATUSES:
        raise ValueError(f"Unsupported subcontractor bill status: {bill_status}")
    now = _utc_now_iso()
    resolved_at = now if bill_status == "bill_reconciled" else None
    if bill_id is None:
        row_id = conn.execute(
            """
            INSERT INTO subcontractor_bill_reviews (
                job_id, segment_id, supplier_id, bill_status, bill_reference,
                bill_date, amount, resolved_at, note, reviewed_by, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(job_id),
                segment_id,
                supplier_id,
                bill_status,
                bill_reference,
                bill_date,
                amount,
                resolved_at,
                note,
                reviewed_by,
                now,
                now,
            ),
        ).lastrowid
    else:
        conn.execute(
            """
            UPDATE subcontractor_bill_reviews
            SET
                job_id = ?,
                segment_id = ?,
                supplier_id = ?,
                bill_status = ?,
                bill_reference = ?,
                bill_date = ?,
                amount = ?,
                resolved_at = ?,
                note = ?,
                reviewed_by = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                int(job_id),
                segment_id,
                supplier_id,
                bill_status,
                bill_reference,
                bill_date,
                amount,
                resolved_at,
                note,
                reviewed_by,
                now,
                int(bill_id),
            ),
        )
        row_id = int(bill_id)
    row = next(row for row in list_subcontractor_bill_reviews(conn) if int(row["id"]) == int(row_id))
    _emit_observer_event(
        conn,
        source_entity_id=f"bill_review:{row_id}",
        event_family="subcontractor_bill_review",
        event_type="subcontractor_bill_review",
        actor_ref=reviewed_by,
        authority_class="reviewed_summary",
        summary=f"Subcontractor bill review {bill_status} for job {int(job_id)}",
        status=bill_status,
        object_refs={
            "bill_review_id": int(row_id),
            "job_id": int(job_id),
            "segment_id": int(segment_id) if segment_id is not None else None,
            "supplier_id": int(supplier_id) if supplier_id is not None else None,
        },
        provenance_refs=[{"kind": "table_row", "table": "subcontractor_bill_reviews", "rowId": int(row_id)}],
        evidence_refs=[{"kind": "table_row", "table": "jobs", "rowId": int(job_id)}],
        payload={
            "jobId": int(job_id),
            "billReviewId": int(row_id),
            "segmentId": int(segment_id) if segment_id is not None else None,
            "supplierId": int(supplier_id) if supplier_id is not None else None,
            "billStatus": bill_status,
            "billReference": bill_reference,
            "billDate": bill_date,
            "amount": amount,
            "note": note,
        },
        correlation_key=f"job:{int(job_id)}",
        event_time=row.get("updated_at") or now,
    )
    conn.commit()
    return row


def export_operations_diary_observer_events(
    conn: sqlite3.Connection,
    *,
    anchor_date: str,
    view_mode: str = "day",
    focus_job_id: int | None = None,
    actor_ref: str | None = None,
    include_planning_snapshot: bool = True,
    include_reconciliation_exceptions: bool = True,
) -> dict[str, Any]:
    ensure_dashboard_tables(conn)
    diary = build_operations_diary(conn, anchor_date=anchor_date, view_mode=view_mode, focus_job_id=focus_job_id)
    emitted: list[dict[str, Any]] = []
    job_rows = diary["jobs"]
    scoped_job_ids = [int(row["jobId"]) for row in job_rows]
    invoice_reviews = {
        int(row["job_id"]): row
        for row in list_customer_invoice_reviews(conn)
        if int(row["job_id"]) in scoped_job_ids
    }
    bill_reviews = [
        row
        for row in list_subcontractor_bill_reviews(conn)
        if int(row["job_id"]) in scoped_job_ids
    ]
    exposure = diary["reconciliationExposure"]
    active_supplier_rows = {
        (
            int(row["jobId"]),
            int(row["supplierId"]) if row.get("supplierId") is not None else None,
            str(row.get("reference") or ""),
            str(row.get("status") or ""),
        ): row
        for row in exposure["activeSupplierRows"]
    }

    if include_planning_snapshot:
        emitted.append(
            _emit_observer_event(
                conn,
                source_entity_id=f"planning_snapshot:{view_mode}:{anchor_date}:{focus_job_id or 'all'}",
                event_family="planning_snapshot",
                event_type="planning_snapshot",
                actor_ref=actor_ref,
                authority_class="reviewed_summary",
                summary=f"{view_mode.title()} planning snapshot for {anchor_date}",
                status="current",
                object_refs={
                    "job_id": int(focus_job_id) if focus_job_id is not None else None,
                    "job_ids": scoped_job_ids,
                },
                provenance_refs=[],
                evidence_refs=[{"kind": "job", "rowId": int(job_id)} for job_id in scoped_job_ids],
                payload={
                    "anchorDate": diary["anchorDate"],
                    "viewMode": diary["viewMode"],
                    "startDate": diary["startDate"],
                    "endDate": diary["endDate"],
                    "focusJobId": int(focus_job_id) if focus_job_id is not None else None,
                    "summary": diary["summary"],
                    "jobs": [
                        {
                            "jobId": int(row["jobId"]),
                            "jobClient": row.get("jobClient"),
                            "invoiceStatus": row.get("invoiceStatus"),
                            "billStatus": row.get("billStatus"),
                            "taskCount": int(row.get("taskCount") or 0),
                        }
                        for row in job_rows
                    ],
                    "reconciliationExposure": {
                        "supplierUnresolvedTotal": exposure["supplierUnresolvedTotal"],
                        "customerOpenTotal": exposure["customerOpenTotal"],
                        "oldestSupplierAgeDays": exposure["oldestSupplierAgeDays"],
                        "longestSupplierLatencyDays": exposure["longestSupplierLatencyDays"],
                    },
                },
                correlation_key=f"operations_diary:{view_mode}:{anchor_date}",
                event_time=f"{diary['endDate']}T23:59:59+00:00",
            )
        )

    if include_reconciliation_exceptions:
        for job in job_rows:
            job_id = int(job["jobId"])
            invoice_status = str(job.get("invoiceStatus") or "")
            if invoice_status in {"not_ready", "reconciliation_warning"}:
                review = invoice_reviews.get(job_id)
                emitted.append(
                    _emit_observer_event(
                        conn,
                        source_entity_id=f"reconciliation_exception:invoice:{job_id}:{invoice_status}",
                        event_family="reconciliation_exception",
                        event_type="reconciliation_exception",
                        actor_ref=actor_ref or (review.get("reviewed_by") if review else None),
                        authority_class="reviewed_summary",
                        summary=f"Invoice exception for job {job_id}: {invoice_status}",
                        status="open",
                        object_refs={
                            "job_id": job_id,
                            "invoice_review_id": int(review["id"]) if review is not None else None,
                        },
                        provenance_refs=(
                            [{"kind": "table_row", "table": "customer_invoice_reviews", "rowId": int(review["id"])}]
                            if review is not None
                            else []
                        ),
                        evidence_refs=[{"kind": "table_row", "table": "jobs", "rowId": job_id}],
                        payload={
                            "exceptionKind": "customer_invoice",
                            "jobId": job_id,
                            "jobClient": job.get("jobClient"),
                            "invoiceStatus": invoice_status,
                            "taskCount": int(job.get("taskCount") or 0),
                            "note": review.get("note") if review else None,
                            "anchorDate": diary["anchorDate"],
                            "viewMode": diary["viewMode"],
                        },
                        correlation_key=f"job:{job_id}",
                    )
                )
        for review in bill_reviews:
            bill_status = str(review.get("bill_status") or "")
            if bill_status not in {"bill_received", "bill_exception"}:
                continue
            key = (
                int(review["job_id"]),
                int(review["supplier_id"]) if review.get("supplier_id") is not None else None,
                str(review.get("bill_reference") or ""),
                bill_status,
            )
            exposure_row = active_supplier_rows.get(key)
            emitted.append(
                _emit_observer_event(
                    conn,
                    source_entity_id=f"reconciliation_exception:bill:{int(review['id'])}:{bill_status}",
                    event_family="reconciliation_exception",
                    event_type="reconciliation_exception",
                    actor_ref=actor_ref or review.get("reviewed_by"),
                    authority_class="reviewed_summary",
                    summary=f"Supplier bill exception for job {int(review['job_id'])}: {bill_status}",
                    status="open",
                    object_refs={
                        "job_id": int(review["job_id"]),
                        "bill_review_id": int(review["id"]),
                        "segment_id": int(review["segment_id"]) if review.get("segment_id") is not None else None,
                        "supplier_id": int(review["supplier_id"]) if review.get("supplier_id") is not None else None,
                    },
                    provenance_refs=[{"kind": "table_row", "table": "subcontractor_bill_reviews", "rowId": int(review["id"])}],
                    evidence_refs=[{"kind": "table_row", "table": "jobs", "rowId": int(review["job_id"])}],
                    payload={
                        "exceptionKind": "supplier_bill",
                        "jobId": int(review["job_id"]),
                        "supplierId": int(review["supplier_id"]) if review.get("supplier_id") is not None else None,
                        "supplierName": review.get("supplier_name"),
                        "billReviewId": int(review["id"]),
                        "billStatus": bill_status,
                        "billReference": review.get("bill_reference"),
                        "billDate": review.get("bill_date"),
                        "amount": review.get("amount"),
                        "unresolvedAgeDays": exposure_row.get("unresolvedAgeDays") if exposure_row else None,
                        "latencyDays": exposure_row.get("latencyDays") if exposure_row else None,
                        "note": review.get("note"),
                        "anchorDate": diary["anchorDate"],
                        "viewMode": diary["viewMode"],
                    },
                    correlation_key=f"job:{int(review['job_id'])}",
                    event_time=review.get("updated_at"),
                )
            )
    conn.commit()
    counts: dict[str, int] = {}
    for row in emitted:
        family = str(row["eventFamily"])
        counts[family] = counts.get(family, 0) + 1
    return {
        "anchorDate": diary["anchorDate"],
        "viewMode": diary["viewMode"],
        "emittedCount": len(emitted),
        "byFamily": counts,
        "events": emitted,
    }


def _derive_invoice_status(record: dict[str, Any] | None, board_row: dict[str, Any]) -> str:
    if record:
        return str(record["invoice_status"])
    if int(board_row.get("blockingCount") or 0) > 0:
        return "not_ready"
    return "ready_to_invoice" if int(board_row.get("segmentCount") or 0) > 0 else "not_ready"


def _derive_bill_status(records: list[dict[str, Any]], board_row: dict[str, Any]) -> str:
    if records:
        statuses = [str(row["bill_status"]) for row in records]
        if "bill_exception" in statuses:
            return "bill_exception"
        if "bill_received" in statuses:
            return "bill_received"
        if "bill_reconciled" in statuses:
            return "bill_reconciled"
        return statuses[0]
    return "awaiting_bill" if board_row.get("supplierNames") else "no_bill_expected"


def _derive_job_execution_date(board_row: dict[str, Any]) -> str | None:
    segments = board_row.get("segments", [])
    candidate_dates = sorted(
        value
        for value in (_date_only(segment.get("plannedStart")) for segment in segments)
        if value
    )
    if candidate_dates:
        return candidate_dates[0]
    return _date_only(board_row.get("jobDate"))


def _customer_exposure_active(record: dict[str, Any]) -> bool:
    status = str(record.get("invoice_status") or "")
    return status in {"ready_to_invoice", "partially_invoiced", "reconciliation_warning"}


def _supplier_exposure_active(record: dict[str, Any]) -> bool:
    status = str(record.get("bill_status") or "")
    return status in {"bill_received", "bill_exception"}


def _days_between(start_value: Any, end_value: Any) -> int | None:
    start = _parse_dateish(start_value)
    end = _parse_dateish(end_value)
    if start is None or end is None:
        return None
    return (end - start).days


def build_reconciliation_exposure_summary(
    conn: sqlite3.Connection,
    *,
    as_of_date: str,
    job_id: int | None = None,
) -> dict[str, Any]:
    ensure_dashboard_tables(conn)
    board_rows = list_job_operations_board(conn, job_id=int(job_id)) if job_id is not None else list_job_operations_board(conn)
    board_by_job = {int(row["jobId"]): row for row in board_rows}

    customer_rows: list[dict[str, Any]] = []
    for record in list_customer_invoice_reviews(conn, job_id=job_id):
        row = board_by_job.get(int(record["job_id"]))
        if row is None:
            continue
        amount = float(record.get("invoice_amount") or 0.0)
        execution_date = _derive_job_execution_date(row)
        received_date = _date_only(record.get("invoice_date"))
        customer_rows.append(
            {
                "lane": "customer",
                "direction": "receivable",
                "jobId": int(record["job_id"]),
                "jobClient": row.get("jobClient") or record.get("job_client"),
                "reference": record.get("invoice_reference"),
                "status": record.get("invoice_status"),
                "amount": round(amount, 2),
                "signedAmount": round(amount, 2),
                "jobExecutionDate": execution_date,
                "receivedDate": received_date,
                "resolvedAt": record.get("resolved_at"),
                "latencyDays": _days_between(execution_date, received_date),
                "unresolvedAgeDays": _days_between(received_date, as_of_date) if _customer_exposure_active(record) else None,
                "note": record.get("note"),
                "isActive": _customer_exposure_active(record),
            }
        )

    supplier_rows: list[dict[str, Any]] = []
    for record in list_subcontractor_bill_reviews(conn, job_id=job_id):
        row = board_by_job.get(int(record["job_id"]))
        if row is None:
            continue
        amount = float(record.get("amount") or 0.0)
        execution_date = _derive_job_execution_date(row)
        received_date = _date_only(record.get("bill_date"))
        supplier_rows.append(
            {
                "lane": "supplier",
                "direction": "liability",
                "jobId": int(record["job_id"]),
                "jobClient": row.get("jobClient") or record.get("job_client"),
                "supplierId": record.get("supplier_id"),
                "supplierName": record.get("supplier_name"),
                "reference": record.get("bill_reference"),
                "status": record.get("bill_status"),
                "amount": round(amount, 2),
                "signedAmount": round(-amount, 2),
                "jobExecutionDate": execution_date,
                "receivedDate": received_date,
                "resolvedAt": record.get("resolved_at"),
                "latencyDays": _days_between(execution_date, received_date),
                "unresolvedAgeDays": _days_between(received_date, as_of_date) if _supplier_exposure_active(record) else None,
                "note": record.get("note"),
                "isActive": _supplier_exposure_active(record),
            }
        )

    active_supplier = [row for row in supplier_rows if row["isActive"]]
    active_customer = [row for row in customer_rows if row["isActive"]]
    oldest_supplier_age = [row["unresolvedAgeDays"] for row in active_supplier if row["unresolvedAgeDays"] is not None]
    supplier_latency = [row["latencyDays"] for row in supplier_rows if row["latencyDays"] is not None]
    active_supplier.sort(key=lambda row: (-abs(float(row["amount"])), -(row["unresolvedAgeDays"] or -1), str(row["jobId"])))
    active_customer.sort(key=lambda row: (-abs(float(row["amount"])), str(row["jobId"])))
    return {
        "asOfDate": as_of_date,
        "supplierRows": supplier_rows,
        "customerRows": customer_rows,
        "activeSupplierRows": active_supplier,
        "activeCustomerRows": active_customer,
        "supplierUnresolvedTotal": round(sum(float(row["amount"]) for row in active_supplier), 2),
        "customerOpenTotal": round(sum(float(row["amount"]) for row in active_customer), 2),
        "oldestSupplierAgeDays": max(oldest_supplier_age) if oldest_supplier_age else None,
        "longestSupplierLatencyDays": max(supplier_latency) if supplier_latency else None,
    }


def build_operations_diary(
    conn: sqlite3.Connection,
    *,
    anchor_date: str,
    view_mode: str = "day",
    focus_job_id: int | None = None,
) -> dict[str, Any]:
    ensure_dashboard_tables(conn)
    start_date, end_date = _date_range(view_mode, anchor_date)
    board_rows = list_job_operations_board(conn)
    tasks = list_operations_diary_tasks(conn, start_date=start_date, end_date=end_date)
    invoice_reviews = {int(row["job_id"]): row for row in list_customer_invoice_reviews(conn)}
    bill_rows = list_subcontractor_bill_reviews(conn)
    bills_by_job: dict[int, list[dict[str, Any]]] = {}
    for row in bill_rows:
        bills_by_job.setdefault(int(row["job_id"]), []).append(row)

    planned_labor = list_planned_labor_assignments(conn, start_date=start_date, end_date=end_date)
    imported_shifts = [dict(row) for row in fetch_driver_shifts(conn, start_date=start_date, end_date=end_date)]
    imported_by_job: dict[int, list[dict[str, Any]]] = {}
    for row in imported_shifts:
        linked = row.get("linked_job_id")
        if linked is None:
            continue
        imported_by_job.setdefault(int(linked), []).append(row)

    task_counts_by_job: dict[int, int] = {}
    for row in tasks:
        if row.get("job_id") is not None:
            task_counts_by_job[int(row["job_id"])] = task_counts_by_job.get(int(row["job_id"]), 0) + 1

    filtered_jobs: list[dict[str, Any]] = []
    for row in board_rows:
        segments = row.get("segments", [])
        in_range = any(
            start_date <= str(_date_only(segment.get("plannedStart")) or "") <= end_date
            for segment in segments
            if _date_only(segment.get("plannedStart"))
        )
        if not in_range and int(row["jobId"]) not in task_counts_by_job:
            continue
        invoice_status = _derive_invoice_status(invoice_reviews.get(int(row["jobId"])), row)
        bill_status = _derive_bill_status(bills_by_job.get(int(row["jobId"]), []), row)
        linked_imported = imported_by_job.get(int(row["jobId"]), [])
        filtered_jobs.append(
            {
                **row,
                "taskCount": task_counts_by_job.get(int(row["jobId"]), 0),
                "invoiceStatus": invoice_status,
                "billStatus": bill_status,
                "importedShiftCount": len(linked_imported),
                "importedShiftCost": round(
                    sum(float(item.get("cost_total") or 0.0) for item in linked_imported), 2
                ),
                "isFocusJob": bool(focus_job_id is not None and int(row["jobId"]) == int(focus_job_id)),
            }
        )
    filtered_jobs.sort(
        key=lambda item: (
            0 if item["isFocusJob"] else 1,
            str(item.get("plannedStart") or ""),
            str(item.get("jobId") or ""),
        )
    )
    exposure_summary = build_reconciliation_exposure_summary(conn, as_of_date=end_date)
    return {
        "viewMode": view_mode,
        "anchorDate": anchor_date,
        "startDate": start_date,
        "endDate": end_date,
        "jobs": filtered_jobs,
        "tasks": tasks,
        "reconciliationExposure": exposure_summary,
        "summary": {
            "jobCount": len(filtered_jobs),
            "taskCount": len(tasks),
            "openTaskCount": sum(1 for row in tasks if row.get("status") != "done"),
            "invoiceExceptionCount": sum(
                1 for row in filtered_jobs if row["invoiceStatus"] in {"not_ready", "reconciliation_warning"}
            ),
            "billExceptionCount": sum(
                1 for row in filtered_jobs if row["billStatus"] in {"awaiting_bill", "bill_exception"}
            ),
            "plannedLaborCount": len(planned_labor),
            "supplierUnresolvedTotal": exposure_summary["supplierUnresolvedTotal"],
            "customerOpenTotal": exposure_summary["customerOpenTotal"],
        },
    }


def build_job_usage_details(conn: sqlite3.Connection, *, job_id: int) -> dict[str, Any]:
    ensure_dashboard_tables(conn)
    board_row = next(
        (row for row in list_job_operations_board(conn, job_id=int(job_id)) if int(row["jobId"]) == int(job_id)),
        None,
    )
    if board_row is None:
        raise ValueError(f"Job {job_id} not found in operations board")
    tasks = list_operations_diary_tasks(conn, job_id=int(job_id))
    invoice_review = next(iter(list_customer_invoice_reviews(conn, job_id=int(job_id))), None)
    bill_reviews = list_subcontractor_bill_reviews(conn, job_id=int(job_id))
    vehicle_usage = build_vehicle_usage_for_job(conn, job_id=int(job_id))
    staff_usage = build_staff_usage_for_job(conn, job_id=int(job_id))
    return {
        "job": board_row,
        "tasks": tasks,
        "invoiceReview": invoice_review,
        "billReviews": bill_reviews,
        "vehicleUsage": vehicle_usage,
        "staffUsage": staff_usage,
        "reconciliationExposure": build_reconciliation_exposure_summary(conn, as_of_date=_date_only(_utc_now_iso()) or date.today().isoformat(), job_id=int(job_id)),
    }


def build_vehicle_usage_for_job(conn: sqlite3.Connection, *, job_id: int) -> list[dict[str, Any]]:
    ensure_dashboard_tables(conn)
    segments = list_segment_readiness(conn, job_id=int(job_id))
    imported = [
        dict(row)
        for row in fetch_driver_shifts(conn)
        if dict(row).get("linked_job_id") == int(job_id)
    ]
    rows: dict[str, dict[str, Any]] = {}
    for segment in segments:
        for assignment in segment.get("truckAssignments", []):
            key = str(assignment["truckId"])
            row = rows.setdefault(
                key,
                {
                    "truckId": key,
                    "truckName": assignment.get("truckName"),
                    "plannedSegmentCount": 0,
                    "plannedStarts": [],
                    "plannedEnds": [],
                    "actualShiftCount": 0,
                    "actualHours": 0.0,
                },
            )
            row["plannedSegmentCount"] += 1
            if segment.get("plannedStart"):
                row["plannedStarts"].append(str(segment["plannedStart"]))
            if segment.get("plannedEnd"):
                row["plannedEnds"].append(str(segment["plannedEnd"]))
    for item in imported:
        key = str(item.get("truck_id") or "")
        if not key:
            continue
        row = rows.setdefault(
            key,
            {
                "truckId": key,
                "truckName": item.get("truck_name"),
                "plannedSegmentCount": 0,
                "plannedStarts": [],
                "plannedEnds": [],
                "actualShiftCount": 0,
                "actualHours": 0.0,
            },
        )
        row["actualShiftCount"] += 1
        row["actualHours"] += float(item.get("hours") or 0.0)
    payload: list[dict[str, Any]] = []
    for row in rows.values():
        payload.append(
            {
                "truckId": row["truckId"],
                "truckName": row["truckName"] or row["truckId"],
                "plannedSegmentCount": row["plannedSegmentCount"],
                "plannedStart": min(row["plannedStarts"]) if row["plannedStarts"] else None,
                "plannedEnd": max(row["plannedEnds"]) if row["plannedEnds"] else None,
                "actualShiftCount": row["actualShiftCount"],
                "actualHours": round(float(row["actualHours"]), 2),
            }
        )
    return sorted(payload, key=lambda item: (str(item["truckId"]), str(item.get("plannedStart") or "")))


def build_staff_usage_for_job(conn: sqlite3.Connection, *, job_id: int) -> list[dict[str, Any]]:
    ensure_dashboard_tables(conn)
    planned = [row for row in list_planned_labor_assignments(conn) if int(row.get("jobId") or 0) == int(job_id)]
    imported = [
        dict(row)
        for row in fetch_driver_shifts(conn)
        if dict(row).get("linked_job_id") == int(job_id)
    ]
    rows: dict[int, dict[str, Any]] = {}
    for row in planned:
        worker_id = int(row["workerId"])
        record = rows.setdefault(
            worker_id,
            {
                "workerId": worker_id,
                "workerName": row.get("workerName"),
                "plannedAssignmentCount": 0,
                "plannedTruckIds": set(),
                "plannedStarts": [],
                "plannedEnds": [],
                "actualShiftCount": 0,
                "actualHours": 0.0,
            },
        )
        record["plannedAssignmentCount"] += 1
        record["plannedTruckIds"].update(str(item) for item in row.get("truckIds", []))
        if row.get("plannedStart"):
            record["plannedStarts"].append(str(row["plannedStart"]))
        if row.get("plannedEnd"):
            record["plannedEnds"].append(str(row["plannedEnd"]))
    for row in imported:
        worker_id = row.get("worker_id")
        if worker_id is None:
            continue
        record = rows.setdefault(
            int(worker_id),
            {
                "workerId": int(worker_id),
                "workerName": row.get("worker_name"),
                "plannedAssignmentCount": 0,
                "plannedTruckIds": set(),
                "plannedStarts": [],
                "plannedEnds": [],
                "actualShiftCount": 0,
                "actualHours": 0.0,
            },
        )
        record["actualShiftCount"] += 1
        record["actualHours"] += float(row.get("hours") or 0.0)
    payload: list[dict[str, Any]] = []
    for row in rows.values():
        payload.append(
            {
                "workerId": row["workerId"],
                "workerName": row["workerName"] or f"Worker {row['workerId']}",
                "plannedAssignmentCount": row["plannedAssignmentCount"],
                "plannedTruckIds": sorted(row["plannedTruckIds"]),
                "plannedStart": min(row["plannedStarts"]) if row["plannedStarts"] else None,
                "plannedEnd": max(row["plannedEnds"]) if row["plannedEnds"] else None,
                "actualShiftCount": row["actualShiftCount"],
                "actualHours": round(float(row["actualHours"]), 2),
            }
        )
    return sorted(payload, key=lambda item: (str(item["workerName"]), int(item["workerId"])))


__all__ = [
    "CUSTOMER_INVOICE_STATUSES",
    "DIARY_TASK_SCOPES",
    "DIARY_TASK_STATUSES",
    "SUBCONTRACTOR_BILL_STATUSES",
    "build_job_usage_details",
    "build_operations_diary",
    "build_reconciliation_exposure_summary",
    "build_staff_usage_for_job",
    "build_vehicle_usage_for_job",
    "delete_operations_diary_task",
    "export_operations_diary_observer_events",
    "list_observer_outbox_events",
    "list_customer_invoice_reviews",
    "list_operations_diary_tasks",
    "list_subcontractor_bill_reviews",
    "upsert_customer_invoice_review",
    "upsert_operations_diary_task",
    "upsert_subcontractor_bill_review",
]
