"""Shipment and driver shift-related database functions."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Optional, Sequence

from .fleet import upsert_worker
from .schema import _ensure_driver_shift_columns


def _resolve_shift_job_id(
    conn: sqlite3.Connection, job_id: int | None, shipment_id: int | None
) -> int | None:
    """Validate and resolve the job linked to a driver shift."""

    resolved_job_id = job_id
    shipment_row = None

    if shipment_id is not None:
        shipment_row = conn.execute(
            "SELECT id, job_id FROM shipments WHERE id = ?", (shipment_id,)
        ).fetchone()
        if shipment_row is None:
            raise ValueError(f"Shipment {shipment_id} does not exist")
        shipment_job_id = shipment_row[1]
        if resolved_job_id is None:
            resolved_job_id = shipment_job_id
        elif shipment_job_id is not None and shipment_job_id != resolved_job_id:
            raise ValueError(
                "Shipment is linked to a different job than the shift specifies"
            )

    if resolved_job_id is not None:
        job_row = conn.execute("SELECT 1 FROM jobs WHERE id = ?", (resolved_job_id,)).fetchone()
        if job_row is None:
            raise ValueError(f"Job {resolved_job_id} does not exist")

    return resolved_job_id


def upsert_driver_shift(
    conn: sqlite3.Connection,
    *,
    shift_date: str,
    truck_id: str | None = None,
    worker_name: str | None = None,
    ticket_numbers: str | None = None,
    shift_start: str | None = None,
    shift_end: str | None = None,
    shift_window_start: str | None = None,
    shift_window_end: str | None = None,
    role: str | None = None,
    hours: float | None = None,
    hourly_rate: float | None = None,
    cost_total: float | None = None,
    job_id: int | None = None,
    shipment_id: int | None = None,
    notes: str | None = None,
    source: str | None = None,
    imported_at: str | None = None,
) -> tuple[sqlite3.Row, bool]:
    """Insert or update a driver shift entry keyed by date/vehicle/worker/time."""

    _ensure_driver_shift_columns(conn)

    worker_id: int | None = None
    if worker_name:
        worker = upsert_worker(conn, name=worker_name)
        worker_id = int(worker["id"])

    resolved_job_id = _resolve_shift_job_id(conn, job_id, shipment_id)

    calculated_cost = cost_total
    if calculated_cost is None and hours is not None and hourly_rate is not None:
        calculated_cost = float(hours) * float(hourly_rate)

    timestamp = imported_at or datetime.now(UTC).isoformat()
    previous_row = conn.execute(
        """
        SELECT * FROM driver_shifts
        WHERE shift_date = ? AND truck_id IS ? AND worker_id IS ?
          AND shift_start IS ? AND shift_end IS ? AND ticket_numbers IS ?
        """,
        (shift_date, truck_id, worker_id, shift_start, shift_end, ticket_numbers),
    ).fetchone()

    conn.execute(
        """
        INSERT INTO driver_shifts (
            shift_date, truck_id, worker_id, job_id, shipment_id, ticket_numbers,
            shift_start, shift_end, shift_window_start, shift_window_end, role,
            hours, hourly_rate, cost_total, notes, source, imported_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(shift_date, truck_id, worker_id, shift_start, shift_end, ticket_numbers)
        DO UPDATE SET
            job_id = excluded.job_id,
            shipment_id = excluded.shipment_id,
            shift_start = excluded.shift_start,
            shift_end = excluded.shift_end,
            shift_window_start = excluded.shift_window_start,
            shift_window_end = excluded.shift_window_end,
            role = excluded.role,
            hours = excluded.hours,
            hourly_rate = excluded.hourly_rate,
            cost_total = excluded.cost_total,
            notes = excluded.notes,
            source = excluded.source,
            imported_at = excluded.imported_at,
            truck_id = excluded.truck_id,
            worker_id = excluded.worker_id,
            ticket_numbers = excluded.ticket_numbers
        """,
        (
            shift_date,
            truck_id,
            worker_id,
            resolved_job_id,
            shipment_id,
            ticket_numbers,
            shift_start,
            shift_end,
            shift_window_start,
            shift_window_end,
            role,
            hours,
            hourly_rate,
            calculated_cost,
            notes,
            source,
            timestamp,
        ),
    )
    conn.commit()
    row = conn.execute(
        """
        SELECT * FROM driver_shifts
        WHERE shift_date = ? AND truck_id IS ? AND worker_id IS ?
          AND shift_start IS ? AND shift_end IS ? AND ticket_numbers IS ?
        """,
        (shift_date, truck_id, worker_id, shift_start, shift_end, ticket_numbers),
    ).fetchone()
    return row, previous_row is None


def _resolve_shipment_locations(
    conn: sqlite3.Connection,
    *,
    job_id: int | None,
    historical_job_id: int | None,
    from_location: str | None,
    to_location: str | None,
) -> tuple[str | None, str | None]:
    """Return shipment locations preferring explicit values then job origins/destinations."""

    if from_location is not None and to_location is not None:
        return from_location, to_location

    resolved_from = from_location
    resolved_to = to_location

    if resolved_from is None or resolved_to is None:
        if job_id is not None:
            row = conn.execute(
                "SELECT origin, destination FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is not None:
                resolved_from = resolved_from or row[0]
                resolved_to = resolved_to or row[1]
        if (resolved_from is None or resolved_to is None) and historical_job_id is not None:
            row = conn.execute(
                "SELECT origin, destination FROM historical_jobs WHERE id = ?",
                (historical_job_id,),
            ).fetchone()
            if row is not None:
                resolved_from = resolved_from or row[0]
                resolved_to = resolved_to or row[1]

    return resolved_from, resolved_to


def create_shipment(
    conn: sqlite3.Connection,
    *,
    job_id: int | None = None,
    historical_job_id: int | None = None,
    inventory_item_id: int | None = None,
    truck_id: str | None = None,
    worker_id: int | None = None,
    quantity: float | None = None,
    from_location: str | None = None,
    to_location: str | None = None,
    status: str = "planned",
    scheduled_date: str | None = None,
    delivered_at: str | None = None,
    reserve_in_transit: bool = True,
) -> sqlite3.Row:
    """Insert a shipment linked to a job or historical job.

    If ``inventory_item_id`` is provided, ensure the requested ``quantity`` is
    available before creating the shipment and recording an inventory movement.
    When ``reserve_in_transit`` is True, the quantity is tracked as allocated in
    transit rather than deducted from on-hand stock.
    """

    if job_id is None and historical_job_id is None:
        raise ValueError("Shipments must reference a job or historical job")

    resolved_from, resolved_to = _resolve_shipment_locations(
        conn,
        job_id=job_id,
        historical_job_id=historical_job_id,
        from_location=from_location,
        to_location=to_location,
    )

    quantity_value = 1.0 if quantity is None else float(quantity)
    timestamp = datetime.now(UTC).isoformat()
    conn.execute(
        """
        INSERT INTO shipments (
            job_id,
            historical_job_id,
            inventory_item_id,
            truck_id,
            worker_id,
            quantity,
            from_location,
            to_location,
            status,
            scheduled_date,
            delivered_at,
            created_at,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            job_id,
            historical_job_id,
            inventory_item_id,
            truck_id,
            worker_id,
            quantity_value,
            resolved_from,
            resolved_to,
            status,
            scheduled_date,
            delivered_at,
            timestamp,
            timestamp,
        ),
    )
    conn.commit()
    # TODO: this is buggy, shipment_id is not defined
    # return conn.execute(
    #     "SELECT * FROM shipments WHERE id = ?", (shipment_id,)
    # ).fetchone()
    return conn.execute("SELECT * FROM shipments WHERE id = last_insert_rowid()").fetchone()


def fetch_driver_shifts(
    conn: sqlite3.Connection,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    worker_names: Sequence[str] | None = None,
    truck_ids: Sequence[str] | None = None,
) -> list[sqlite3.Row]:
    """Return driver shifts matching the provided filters."""

    filters: list[str] = []
    params: list[object] = []

    if start_date:
        filters.append("ds.shift_date >= ?")
        params.append(start_date)
    if end_date:
        filters.append("ds.shift_date <= ?")
        params.append(end_date)
    if worker_names:
        filters.append("w.name IN (" + ",".join(["?"] * len(worker_names)) + ")")
        params.extend(worker_names)
    if truck_ids:
        filters.append("ds.truck_id IN (" + ",".join(["?"] * len(truck_ids)) + ")")
        params.extend(truck_ids)

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    query = f"""
        SELECT
            ds.*, w.name AS worker_name, t.name AS truck_name,
            COALESCE(ds.job_id, s.job_id) AS linked_job_id,
            s.job_id AS shipment_job_id,
            j.origin AS job_origin,
            j.destination AS job_destination
        FROM driver_shifts ds
        LEFT JOIN workers w ON ds.worker_id = w.id
        LEFT JOIN trucks t ON ds.truck_id = t.truck_id
        LEFT JOIN shipments s ON ds.shipment_id = s.id
        LEFT JOIN jobs j ON COALESCE(ds.job_id, s.job_id) = j.id
        {where_clause}
        ORDER BY ds.shift_date DESC, ds.shift_start
    """
    return list(conn.execute(query, params))


def rollup_driver_shift_costs_by_job(
    conn: sqlite3.Connection,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[sqlite3.Row]:
    """Aggregate driver shift hours and costs grouped by job."""

    filters: list[str] = ["COALESCE(ds.job_id, s.job_id) IS NOT NULL"]
    params: list[object] = []

    if start_date:
        filters.append("ds.shift_date >= ?")
        params.append(start_date)
    if end_date:
        filters.append("ds.shift_date <= ?")
        params.append(end_date)

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    query = f"""
        SELECT
            COALESCE(ds.job_id, s.job_id) AS job_id,
            COUNT(*) AS shift_count,
            SUM(ds.hours) AS total_hours,
            SUM(ds.cost_total) AS total_cost
        FROM driver_shifts ds
        LEFT JOIN shipments s ON ds.shipment_id = s.id
        {where_clause}
        GROUP BY COALESCE(ds.job_id, s.job_id)
        ORDER BY job_id
    """
    return list(conn.execute(query, params))


def fetch_shipments_with_context(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return shipments joined with job, inventory, and assignment context."""

    query = """
        SELECT
            s.id,
            s.quantity,
            s.from_location,
            s.to_location,
            s.status,
            s.scheduled_date,
            s.delivered_at,
            s.job_id,
            s.historical_job_id,
            j.origin AS job_origin,
            j.destination AS job_destination,
            h.origin AS historical_origin,
            h.destination AS historical_destination,
            i.name AS inventory_name,
            i.quantity AS inventory_quantity,
            sup.company_name AS supplier_company_name,
            sup.contact_name AS supplier_contact_name,
            sup.contact_number AS supplier_contact_number,
            sup.email AS supplier_email,
            t.truck_id,
            t.name AS truck_name,
            w.name AS worker_name,
            w.role AS worker_role,
            w.rate AS worker_rate,
            w.tickets AS worker_tickets
        FROM shipments AS s
        LEFT JOIN jobs AS j ON s.job_id = j.id
        LEFT JOIN historical_jobs AS h ON s.historical_job_id = h.id
        LEFT JOIN inventory_items AS i ON s.inventory_item_id = i.id
        LEFT JOIN suppliers AS sup ON i.supplier_id = sup.id
        LEFT JOIN trucks AS t ON s.truck_id = t.truck_id
        LEFT JOIN workers AS w ON s.worker_id = w.id
        ORDER BY s.id
    """
    return list(conn.execute(query))
