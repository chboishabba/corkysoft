"""Inventory and supplier-related database functions."""
from __future__ import annotations

import os
import sqlite3
from datetime import UTC, datetime
from typing import Sequence

import pandas as pd

from analytics.google_sheets import (
    build_google_sheet_csv_url,
    resolve_google_sheet_reference,
)
from .schema import ensure_dashboard_tables, ensure_suppliers_table

INVENTORY_STATES: Sequence[str] = (
    "created",
    "staged",
    "loaded",
    "in_transit",
    "delivered",
    "exception",
)
INVENTORY_ARCHITECTURES: Sequence[str] = (
    "container",
    "consumable",
    "reusable_asset",
    "serialized_asset",
    "job_specific",
    "general",
)
INVENTORY_CUSTODY_TYPES: Sequence[str] = (
    "depot",
    "truck",
    "container",
    "in_transit",
    "site",
    "returned_storage",
    "exception",
)
INVENTORY_EXECUTION_STAGES: Sequence[str] = (
    "required",
    "picked",
    "packed",
    "loaded",
    "in_transit",
    "unloaded",
    "returned_storage",
    "exception",
)
INVENTORY_SUBSTITUTION_STATUSES: Sequence[str] = (
    "requested",
    "approved",
    "rejected",
)
INVENTORY_SUBSTITUTION_APPROVER_ROLES: Sequence[str] = (
    "dispatcher",
    "operations_manager",
)

_DEFAULT_SUBSTITUTION_REASON_CODES: tuple[tuple[str, str, str], ...] = (
    ("stock_shortage", "Stock shortage", "Planned stock is unavailable in the required quantity."),
    ("damaged_stock", "Damaged stock", "Planned stock is damaged or quarantined and cannot be used."),
    ("wrong_item_picked", "Wrong item picked", "Picked stock does not match the planned requirement."),
    ("container_unavailable", "Container unavailable", "Required container is unavailable for the leg."),
    ("late_return", "Late return", "Reusable stock or container has not returned in time."),
    ("site_constraint", "Site constraint", "Site conditions require a materially different equivalent item."),
)

_EXECUTION_STAGE_TRANSITIONS: dict[str, tuple[str, ...]] = {
    "required": ("picked", "exception"),
    "picked": ("packed", "exception"),
    "packed": ("loaded", "exception"),
    "loaded": ("in_transit", "unloaded", "exception"),
    "in_transit": ("unloaded", "exception"),
    "unloaded": ("returned_storage", "exception"),
    "returned_storage": (),
    "exception": (),
}


def upsert_inventory_item(
    conn: sqlite3.Connection,
    *,
    name: str,
    description: str = "",
    quantity: int = 0,
    unit: str = "unit",
    supplier_id: int | None = None,
    job_id: int | None = None,
    state: str = "created",
    item_id: str | None = None,
    asset_tag: str | None = None,
    architecture: str = "general",
    custody_location_type: str | None = None,
    custody_location_ref: str | None = None,
    custody_location_label: str | None = None,
) -> sqlite3.Row:
    """Create or update an inventory item and return the stored row."""

    timestamp = datetime.now(UTC).isoformat()
    state_value = state if state in INVENTORY_STATES else "created"
    architecture_value = (
        architecture if architecture in INVENTORY_ARCHITECTURES else "general"
    )
    custody_type_value = (
        custody_location_type
        if custody_location_type in INVENTORY_CUSTODY_TYPES
        else None
    )
    conn.execute(
        """
        INSERT INTO inventory_items (
            name,
            description,
            quantity,
            unit,
            supplier_id,
            job_id,
            state,
            item_id,
            asset_tag,
            architecture,
            custody_location_type,
            custody_location_ref,
            custody_location_label,
            custody_updated_at,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            description = excluded.description,
            quantity = excluded.quantity,
            unit = excluded.unit,
            supplier_id = excluded.supplier_id,
            job_id = COALESCE(excluded.job_id, inventory_items.job_id),
            state = excluded.state,
            item_id = COALESCE(excluded.item_id, inventory_items.item_id),
            asset_tag = COALESCE(excluded.asset_tag, inventory_items.asset_tag),
            architecture = excluded.architecture,
            custody_location_type = COALESCE(
                excluded.custody_location_type,
                inventory_items.custody_location_type
            ),
            custody_location_ref = COALESCE(
                excluded.custody_location_ref,
                inventory_items.custody_location_ref
            ),
            custody_location_label = COALESCE(
                excluded.custody_location_label,
                inventory_items.custody_location_label
            ),
            custody_updated_at = COALESCE(
                excluded.custody_updated_at,
                inventory_items.custody_updated_at
            ),
            updated_at = excluded.updated_at
        """,
        (
            name,
            description,
            int(quantity),
            unit,
            supplier_id,
            job_id,
            state_value,
            item_id,
            asset_tag,
            architecture_value,
            custody_type_value,
            _clean_optional_str(custody_location_ref),
            _clean_optional_str(custody_location_label),
            timestamp if custody_type_value or custody_location_ref or custody_location_label else None,
            timestamp,
        ),
    )
    conn.commit()
    return conn.execute("SELECT * FROM inventory_items WHERE name = ?", (name,)).fetchone()


def _inventory_balance_query(where_clause: str = "") -> str:
    balances_cte = """
        WITH movement_totals AS (
            SELECT
                inventory_item_id,
                COALESCE(SUM(change_on_hand), 0) AS delta_on_hand,
                COALESCE(SUM(change_allocated), 0) AS delta_allocated
            FROM inventory_movements
            GROUP BY inventory_item_id
        )
    """
    select_sql = f"""
        SELECT
            i.*,
            i.quantity + COALESCE(m.delta_on_hand, 0) AS on_hand_quantity,
            COALESCE(m.delta_allocated, 0) AS allocated_quantity,
            i.quantity + COALESCE(m.delta_on_hand, 0) - COALESCE(m.delta_allocated, 0)
                AS available_quantity
        FROM inventory_items AS i
        LEFT JOIN movement_totals AS m ON m.inventory_item_id = i.id
        {where_clause}
        ORDER BY i.name
    """
    return balances_cte + select_sql


def list_inventory_balances(
    conn: sqlite3.Connection,
    *,
    job_id: int | None = None,
    states: Sequence[str] | None = None,
) -> list[sqlite3.Row]:
    """Return inventory items with on-hand, allocated, and available totals."""

    where_clauses: list[str] = []
    params: list[object] = []
    if job_id is not None:
        where_clauses.append("i.job_id = ?")
        params.append(job_id)
    if states:
        placeholders = ",".join("?" for _ in states)
        where_clauses.append(f"i.state IN ({placeholders})")
        params.extend(states)

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    return list(conn.execute(_inventory_balance_query(where_sql), params))
    return list(conn.execute(_inventory_balance_query()))


def get_inventory_balance(
    conn: sqlite3.Connection, inventory_item_id: int
) -> sqlite3.Row | None:
    """Return a single inventory balance row by item id."""

    rows = conn.execute(
        _inventory_balance_query("WHERE i.id = ?"), (inventory_item_id,)
    ).fetchall()
    return rows[0] if rows else None


def _seed_inventory_substitution_reason_codes(conn: sqlite3.Connection) -> None:
    timestamp = datetime.now(UTC).isoformat()
    conn.executemany(
        """
        INSERT OR IGNORE INTO inventory_substitution_reason_codes (
            code,
            label,
            description,
            active,
            system_seeded,
            created_at,
            updated_at
        ) VALUES (?, ?, ?, 1, 1, ?, ?)
        """,
        [
            (code, label, description, timestamp, timestamp)
            for code, label, description in _DEFAULT_SUBSTITUTION_REASON_CODES
        ],
    )
    conn.commit()


def list_inventory_substitution_reason_codes(
    conn: sqlite3.Connection,
    *,
    active_only: bool = False,
) -> list[dict[str, object]]:
    ensure_dashboard_tables(conn)
    _seed_inventory_substitution_reason_codes(conn)
    where = "WHERE active = 1" if active_only else ""
    rows = conn.execute(
        f"""
        SELECT code, label, description, active, system_seeded, created_at, updated_at
        FROM inventory_substitution_reason_codes
        {where}
        ORDER BY active DESC, label, code
        """
    ).fetchall()
    return [
        {
            "code": row["code"],
            "label": row["label"],
            "description": row["description"],
            "active": bool(row["active"]),
            "systemSeeded": bool(row["system_seeded"]),
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
        }
        for row in rows
    ]


def upsert_inventory_substitution_reason_code(
    conn: sqlite3.Connection,
    *,
    code: str,
    label: str,
    description: str | None = None,
    active: bool = True,
) -> sqlite3.Row:
    ensure_dashboard_tables(conn)
    _seed_inventory_substitution_reason_codes(conn)
    normalized_code = _clean_optional_str(code)
    normalized_label = _clean_optional_str(label)
    if not normalized_code or not normalized_label:
        raise ValueError("Inventory substitution reason code and label are required")
    timestamp = datetime.now(UTC).isoformat()
    conn.execute(
        """
        INSERT INTO inventory_substitution_reason_codes (
            code,
            label,
            description,
            active,
            system_seeded,
            created_at,
            updated_at
        ) VALUES (?, ?, ?, ?, 0, ?, ?)
        ON CONFLICT(code) DO UPDATE SET
            label = excluded.label,
            description = excluded.description,
            active = excluded.active,
            updated_at = excluded.updated_at
        """,
        (
            normalized_code,
            normalized_label,
            _clean_optional_str(description),
            1 if active else 0,
            timestamp,
            timestamp,
        ),
    )
    conn.commit()
    return conn.execute(
        "SELECT * FROM inventory_substitution_reason_codes WHERE code = ?",
        (normalized_code,),
    ).fetchone()


def get_allowed_inventory_execution_stages(
    current_stage: str | None,
    *,
    architecture: str | None = None,
) -> list[str]:
    normalized_stage = (
        current_stage if current_stage in INVENTORY_EXECUTION_STAGES else "required"
    )
    allowed = list(_EXECUTION_STAGE_TRANSITIONS.get(normalized_stage, ()))
    normalized_architecture = architecture or "general"
    if "returned_storage" in allowed and normalized_architecture not in {
        "container",
        "reusable_asset",
        "serialized_asset",
    }:
        allowed = [stage for stage in allowed if stage != "returned_storage"]
    return allowed


def record_inventory_movement(
    conn: sqlite3.Connection,
    *,
    inventory_item_id: int,
    shipment_id: int | None = None,
    change_on_hand: int = 0,
    change_allocated: int = 0,
    reason: str | None = None,
    commit: bool = True,
    state: str | None = None,
    job_id: int | None = None,
    sequence_no: int | None = None,
    location_type: str | None = None,
    location_ref: str | None = None,
    location_label: str | None = None,
) -> sqlite3.Row:
    """Insert an inventory movement entry and return the stored row."""

    timestamp = datetime.now(UTC).isoformat()
    state_value = state if state in INVENTORY_STATES else None
    location_type_value = location_type if location_type in INVENTORY_CUSTODY_TYPES else None
    conn.execute(
        """
        INSERT INTO inventory_movements (
            inventory_item_id,
            shipment_id,
            job_id,
            change_on_hand,
            change_allocated,
            reason,
            state,
            sequence_no,
            location_type,
            location_ref,
            location_label,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            inventory_item_id,
            shipment_id,
            job_id,
            int(change_on_hand),
            int(change_allocated),
            reason or "",
            state_value,
            sequence_no,
            location_type_value,
            _clean_optional_str(location_ref),
            _clean_optional_str(location_label),
            timestamp,
        ),
    )

    if state_value or job_id is not None or location_type_value or location_ref or location_label:
        conn.execute(
            """
            UPDATE inventory_items
            SET state = COALESCE(?, state),
                job_id = COALESCE(?, job_id),
                custody_location_type = COALESCE(?, custody_location_type),
                custody_location_ref = COALESCE(?, custody_location_ref),
                custody_location_label = COALESCE(?, custody_location_label),
                custody_updated_at = CASE
                    WHEN ? IS NOT NULL OR ? IS NOT NULL OR ? IS NOT NULL THEN ?
                    ELSE custody_updated_at
                END,
                updated_at = ?
            WHERE id = ?
            """,
            (
                state_value,
                job_id,
                location_type_value,
                _clean_optional_str(location_ref),
                _clean_optional_str(location_label),
                location_type_value,
                _clean_optional_str(location_ref),
                _clean_optional_str(location_label),
                timestamp,
                timestamp,
                inventory_item_id,
            ),
        )

    if commit:
        conn.commit()
    return conn.execute(
        "SELECT * FROM inventory_movements WHERE id = last_insert_rowid()"
    ).fetchone()


def list_inventory(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return all inventory items ordered by name."""

    return list(
        conn.execute(
            """
            SELECT
                i.*,
                s.company_name AS supplier_company_name,
                s.contact_name AS supplier_contact_name,
                s.contact_number AS supplier_contact_number,
                s.email AS supplier_email,
                s.notes AS supplier_notes
            FROM inventory_items AS i
            LEFT JOIN suppliers AS s ON i.supplier_id = s.id
            ORDER BY i.name
            """
        )
    )


def list_inventory_movements(
    conn: sqlite3.Connection,
    *,
    limit: int = 50,
    job_id: int | None = None,
    states: Sequence[str] | None = None,
) -> list[sqlite3.Row]:
    """Return recent inventory movements with item context."""

    where_clauses: list[str] = []
    params: list[object] = []
    if job_id is not None:
        where_clauses.append("m.job_id = ?")
        params.append(job_id)
    if states:
        placeholders = ",".join("?" for _ in states)
        where_clauses.append(f"COALESCE(m.state, i.state) IN ({placeholders})")
        params.extend(states)
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    return list(
        conn.execute(
            f"""
            SELECT
                m.*, i.name AS inventory_name, i.unit,
                COALESCE(m.state, i.state) AS movement_state,
                i.job_id AS item_job_id,
                i.architecture AS item_architecture,
                COALESCE(m.location_type, i.custody_location_type) AS location_type_value,
                COALESCE(m.location_ref, i.custody_location_ref) AS location_ref_value,
                COALESCE(m.location_label, i.custody_location_label) AS location_label_value
            FROM inventory_movements AS m
            JOIN inventory_items AS i ON i.id = m.inventory_item_id
            {where_sql}
            ORDER BY m.created_at DESC
            LIMIT ?
            """,
            (*params, limit),
        )
    )


def upsert_supplier(
    conn: sqlite3.Connection,
    *,
    company_name: str,
    contact_name: str | None = None,
    contact_number: str | None = None,
    email: str | None = None,
    notes: str | None = None,
    source_system: str | None = None,
    source_sheet: str | None = None,
    source_imported_at: str | None = None,
) -> sqlite3.Row:
    """Create or update a supplier record by company name."""

    if not company_name.strip():
        raise ValueError("Supplier company name is required")

    timestamp = datetime.now(UTC).isoformat()
    conn.execute(
        """
        INSERT INTO suppliers (
            company_name, contact_name, contact_number, email, notes,
            source_system, source_sheet, source_imported_at, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(company_name) DO UPDATE SET
            contact_name = excluded.contact_name,
            contact_number = excluded.contact_number,
            email = excluded.email,
            notes = excluded.notes,
            source_system = COALESCE(excluded.source_system, suppliers.source_system),
            source_sheet = COALESCE(excluded.source_sheet, suppliers.source_sheet),
            source_imported_at = COALESCE(excluded.source_imported_at, suppliers.source_imported_at),
            updated_at = excluded.updated_at
        """,
        (
            company_name.strip(),
            _clean_optional_str(contact_name),
            _clean_optional_str(contact_number),
            _clean_optional_str(email),
            _clean_optional_str(notes),
            _clean_optional_str(source_system),
            _clean_optional_str(source_sheet),
            _clean_optional_str(source_imported_at),
            timestamp,
            timestamp,
        ),
    )
    conn.commit()
    return conn.execute(
        "SELECT * FROM suppliers WHERE company_name = ?", (company_name.strip(),)
    ).fetchone()


def list_suppliers(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return suppliers ordered by company name."""

    return list(
        conn.execute("SELECT * FROM suppliers ORDER BY LOWER(company_name), company_name")
    )


def import_suppliers_from_google_sheet(
    conn: sqlite3.Connection,
    *,
    sheet_id: str | None = None,
    sheet_name: str = "SUPPLIERS",
    csv_url: str | None = None,
    dataframe: pd.DataFrame | None = None,
) -> int:
    """Import suppliers from a Google Sheet export or DataFrame.

    Returns the number of supplier rows inserted or updated.
    """

    ensure_suppliers_table(conn)

    df = dataframe
    if df is None:
        resolved_url = csv_url or _build_suppliers_sheet_url(sheet_id, sheet_name)
        if resolved_url is None:
            raise ValueError("Provide a dataframe, csv_url, or sheet_id for import")
        df = pd.read_csv(resolved_url)

    if df.empty:
        return 0

    normalized = df.rename(columns=_normalize_supplier_column)
    required_field = "company_name"
    if required_field not in normalized.columns:
        raise ValueError("Suppliers sheet must include a company name column")

    imported = 0
    for _, row in normalized.iterrows():
        company = _clean_optional_str(row.get("company_name"))
        if not company:
            continue
        contact_name = _clean_optional_str(row.get("contact_name"))
        contact_number = _clean_optional_str(row.get("contact_number"))
        email = _clean_optional_str(row.get("email"))
        notes = _clean_optional_str(row.get("notes"))
        upsert_supplier(
            conn,
            company_name=company,
            contact_name=contact_name,
            contact_number=contact_number,
            email=email,
            notes=notes,
            source_system="google_sheets",
            source_sheet=sheet_name,
            source_imported_at=datetime.now(UTC).isoformat(),
        )
        imported += 1
    return imported


def import_inventory_items_from_dataframe(
    conn: sqlite3.Connection, dataframe: pd.DataFrame
) -> int:
    """Bulk import inventory items from a DataFrame."""

    if dataframe.empty:
        return 0
    normalized = dataframe.rename(columns=_normalize_inventory_column)
    if "name" not in normalized.columns:
        raise ValueError("Inventory imports require a 'name' column")

    imported = 0
    for _, row in normalized.iterrows():
        name = _clean_optional_str(row.get("name"))
        if not name:
            continue
        description = row.get("description") or ""
        unit = row.get("unit") or "unit"
        quantity = int(row.get("quantity") or 0)
        supplier_id = row.get("supplier_id")
        job_id = row.get("job_id")
        state = row.get("state") or "created"
        item_id = _clean_optional_str(row.get("item_id"))
        asset_tag = _clean_optional_str(row.get("asset_tag"))
        architecture = _clean_optional_str(row.get("architecture")) or "general"
        custody_location_type = _clean_optional_str(row.get("custody_location_type"))
        custody_location_ref = _clean_optional_str(row.get("custody_location_ref"))
        custody_location_label = _clean_optional_str(row.get("custody_location_label"))
        upsert_inventory_item(
            conn,
            name=name,
            description=str(description),
            quantity=quantity,
            unit=str(unit),
            supplier_id=int(supplier_id) if pd.notna(supplier_id) else None,
            job_id=int(job_id) if pd.notna(job_id) else None,
            state=str(state),
            item_id=item_id,
            asset_tag=asset_tag,
            architecture=architecture,
            custody_location_type=custody_location_type,
            custody_location_ref=custody_location_ref,
            custody_location_label=custody_location_label,
        )
        imported += 1
    return imported


def import_inventory_movements_from_dataframe(
    conn: sqlite3.Connection,
    dataframe: pd.DataFrame,
    *,
    default_reason: str | None = None,
) -> int:
    """Bulk import movement events from a DataFrame."""

    if dataframe.empty:
        return 0

    normalized = dataframe.rename(columns=_normalize_inventory_column)
    required_field = None
    for candidate in ("inventory_item_id", "name"):
        if candidate in normalized.columns:
            required_field = candidate
            break
    if required_field is None:
        raise ValueError("Movement imports require an 'inventory_item_id' or 'name' column")

    name_to_id = {
        row["name"]: row["id"] for row in list_inventory(conn)
    }

    imported = 0
    for _, row in normalized.iterrows():
        inventory_item_id = row.get("inventory_item_id")
        if pd.isna(inventory_item_id) or inventory_item_id is None:
            name = _clean_optional_str(row.get("name"))
            inventory_item_id = name_to_id.get(name) if name else None
        if inventory_item_id is None:
            continue

        job_id = row.get("job_id")
        state = row.get("state")
        reason = row.get("reason") or default_reason
        sequence_no = row.get("sequence_no")
        location_type = _clean_optional_str(row.get("location_type"))
        location_ref = _clean_optional_str(row.get("location_ref"))
        location_label = _clean_optional_str(row.get("location_label"))
        record_inventory_movement(
            conn,
            inventory_item_id=int(inventory_item_id),
            job_id=int(job_id) if pd.notna(job_id) else None,
            change_on_hand=int(row.get("change_on_hand") or 0),
            change_allocated=int(row.get("change_allocated") or 0),
            state=str(state) if pd.notna(state) and state else None,
            sequence_no=int(sequence_no) if pd.notna(sequence_no) else None,
            reason=str(reason) if reason is not None else None,
            location_type=location_type,
            location_ref=location_ref,
            location_label=location_label,
        )
        imported += 1
    return imported


def upsert_inventory_requirement(
    conn: sqlite3.Connection,
    *,
    job_id: int,
    segment_id: int,
    inventory_item_id: int | None = None,
    requirement_name: str,
    required_quantity: float,
    substitution_allowed: bool = False,
    architecture: str = "general",
    notes: str | None = None,
) -> sqlite3.Row:
    """Create or update an inventory requirement line for a job segment."""

    ensure_dashboard_tables(conn)
    requirement_name_value = _clean_optional_str(requirement_name)
    if not requirement_name_value:
        raise ValueError("Requirement name is required")
    if required_quantity <= 0:
        raise ValueError("Required quantity must be positive")
    if conn.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone() is None:
        raise ValueError(f"Job {job_id} not found")
    segment = conn.execute(
        "SELECT id, job_id FROM job_segments WHERE id = ?",
        (segment_id,),
    ).fetchone()
    if segment is None:
        raise ValueError(f"Segment {segment_id} not found")
    if int(segment["job_id"]) != int(job_id):
        raise ValueError("Inventory requirement job/segment mismatch")
    if inventory_item_id is not None and conn.execute(
        "SELECT 1 FROM inventory_items WHERE id = ?",
        (inventory_item_id,),
    ).fetchone() is None:
        raise ValueError(f"Inventory item {inventory_item_id} not found")

    timestamp = datetime.now(UTC).isoformat()
    architecture_value = (
        architecture if architecture in INVENTORY_ARCHITECTURES else "general"
    )
    conn.execute(
        """
        INSERT INTO inventory_requirements (
            job_id,
            segment_id,
            inventory_item_id,
            requirement_name,
            required_quantity,
            substitution_allowed,
            architecture,
            notes,
            created_at,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(job_id, segment_id, inventory_item_id, requirement_name) DO UPDATE SET
            required_quantity = excluded.required_quantity,
            substitution_allowed = excluded.substitution_allowed,
            architecture = excluded.architecture,
            notes = excluded.notes,
            updated_at = excluded.updated_at
        """,
        (
            job_id,
            segment_id,
            inventory_item_id,
            requirement_name_value,
            float(required_quantity),
            int(bool(substitution_allowed)),
            architecture_value,
            _clean_optional_str(notes),
            timestamp,
            timestamp,
        ),
    )
    conn.commit()
    return conn.execute(
        """
        SELECT *
        FROM inventory_requirements
        WHERE job_id = ? AND segment_id = ? AND requirement_name = ? AND (
            (inventory_item_id = ?) OR (inventory_item_id IS NULL AND ? IS NULL)
        )
        """,
        (job_id, segment_id, requirement_name_value, inventory_item_id, inventory_item_id),
    ).fetchone()


def record_inventory_execution_event(
    conn: sqlite3.Connection,
    *,
    job_id: int,
    segment_id: int,
    stage: str,
    requirement_id: int | None = None,
    inventory_item_id: int | None = None,
    quantity: float | None = None,
    actor: str | None = None,
    note: str | None = None,
    container_ref: str | None = None,
    truck_id: str | None = None,
    location_type: str | None = None,
    location_ref: str | None = None,
    location_label: str | None = None,
) -> sqlite3.Row:
    """Record a warehouse execution event above the persisted inventory state model."""

    ensure_dashboard_tables(conn)
    if stage not in INVENTORY_EXECUTION_STAGES:
        raise ValueError(f"Unsupported inventory execution stage: {stage}")
    current_stage = "required"
    requirement_architecture = "general"
    if conn.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone() is None:
        raise ValueError(f"Job {job_id} not found")
    segment = conn.execute(
        "SELECT id, job_id FROM job_segments WHERE id = ?",
        (segment_id,),
    ).fetchone()
    if segment is None:
        raise ValueError(f"Segment {segment_id} not found")
    if int(segment["job_id"]) != int(job_id):
        raise ValueError("Inventory execution event job/segment mismatch")
    if requirement_id is not None and conn.execute(
        "SELECT 1 FROM inventory_requirements WHERE id = ? AND segment_id = ?",
        (requirement_id, segment_id),
    ).fetchone() is None:
        raise ValueError(f"Inventory requirement {requirement_id} not found for segment {segment_id}")
    if requirement_id is not None:
        current_requirement = conn.execute(
            "SELECT architecture FROM inventory_requirements WHERE id = ?",
            (requirement_id,),
        ).fetchone()
        if current_requirement is not None:
            requirement_architecture = current_requirement["architecture"] or "general"
        latest = conn.execute(
            """
            SELECT stage
            FROM inventory_execution_events
            WHERE requirement_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (requirement_id,),
        ).fetchone()
        current_stage = latest["stage"] if latest and latest["stage"] else "required"
        allowed_next = get_allowed_inventory_execution_stages(
            current_stage,
            architecture=requirement_architecture,
        )
        if stage not in allowed_next:
            raise ValueError(
                f"Execution stage '{stage}' is not allowed from '{current_stage}'. "
                f"Allowed next stages: {', '.join(allowed_next) or 'none'}"
            )
    if inventory_item_id is not None and conn.execute(
        "SELECT 1 FROM inventory_items WHERE id = ?",
        (inventory_item_id,),
    ).fetchone() is None:
        raise ValueError(f"Inventory item {inventory_item_id} not found")

    timestamp = datetime.now(UTC).isoformat()
    conn.execute(
        """
        INSERT INTO inventory_execution_events (
            job_id,
            segment_id,
            requirement_id,
            inventory_item_id,
            stage,
            quantity,
            actor,
            note,
            container_ref,
            truck_id,
            location_type,
            location_ref,
            location_label,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            job_id,
            segment_id,
            requirement_id,
            inventory_item_id,
            stage,
            float(quantity) if quantity is not None else None,
            _clean_optional_str(actor),
            _clean_optional_str(note),
            _clean_optional_str(container_ref),
            _clean_optional_str(truck_id),
            location_type if location_type in INVENTORY_CUSTODY_TYPES else None,
            _clean_optional_str(location_ref),
            _clean_optional_str(location_label),
            timestamp,
        ),
    )

    if inventory_item_id is not None:
        state_mapping = {
            "picked": "staged",
            "packed": "staged",
            "loaded": "loaded",
            "in_transit": "in_transit",
            "unloaded": "delivered",
            "returned_storage": "created",
            "exception": "exception",
        }
        mapped_state = state_mapping.get(stage)
        if mapped_state or location_type or location_ref or location_label:
            record_inventory_movement(
                conn,
                inventory_item_id=inventory_item_id,
                job_id=job_id,
                change_on_hand=0,
                change_allocated=0,
                reason=f"execution_stage:{stage}",
                state=mapped_state,
                commit=False,
                location_type=location_type,
                location_ref=location_ref,
                location_label=location_label,
            )

    conn.commit()
    return conn.execute(
        "SELECT * FROM inventory_execution_events WHERE id = last_insert_rowid()"
    ).fetchone()


def list_inventory_execution_events(
    conn: sqlite3.Connection,
    *,
    job_id: int | None = None,
    segment_id: int | None = None,
    limit: int = 100,
) -> list[dict[str, object]]:
    """Return recent warehouse execution events with requirement and item context."""

    ensure_dashboard_tables(conn)
    filters: list[str] = []
    params: list[object] = []
    if job_id is not None:
        filters.append("e.job_id = ?")
        params.append(job_id)
    if segment_id is not None:
        filters.append("e.segment_id = ?")
        params.append(segment_id)
    where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""
    rows = conn.execute(
        f"""
        SELECT
            e.*,
            ir.requirement_name,
            i.name AS inventory_name
        FROM inventory_execution_events AS e
        LEFT JOIN inventory_requirements AS ir ON ir.id = e.requirement_id
        LEFT JOIN inventory_items AS i ON i.id = e.inventory_item_id
        {where_sql}
        ORDER BY e.created_at DESC, e.id DESC
        LIMIT ?
        """,
        (*params, limit),
    ).fetchall()
    return [
        {
            "eventId": int(row["id"]),
            "jobId": int(row["job_id"]),
            "segmentId": int(row["segment_id"]),
            "requirementId": int(row["requirement_id"]) if row["requirement_id"] is not None else None,
            "inventoryItemId": int(row["inventory_item_id"]) if row["inventory_item_id"] is not None else None,
            "stage": row["stage"],
            "quantity": float(row["quantity"]) if row["quantity"] is not None else None,
            "actor": row["actor"],
            "note": row["note"],
            "containerRef": row["container_ref"],
            "truckId": row["truck_id"],
            "locationType": row["location_type"],
            "locationRef": row["location_ref"],
            "locationLabel": row["location_label"],
            "requirementName": row["requirement_name"],
            "inventoryName": row["inventory_name"],
            "createdAt": row["created_at"],
        }
        for row in rows
    ]


def request_inventory_substitution(
    conn: sqlite3.Connection,
    *,
    requirement_id: int,
    requested_quantity: float,
    requested_by: str | None,
    reason_code: str,
    note: str | None = None,
    substitute_inventory_item_id: int | None = None,
) -> sqlite3.Row:
    """Request a substitution for a requirement line."""

    ensure_dashboard_tables(conn)
    _seed_inventory_substitution_reason_codes(conn)
    requirement = conn.execute(
        """
        SELECT id, job_id, segment_id, inventory_item_id, substitution_allowed
        FROM inventory_requirements
        WHERE id = ?
        """,
        (requirement_id,),
    ).fetchone()
    if requirement is None:
        raise ValueError(f"Inventory requirement {requirement_id} not found")
    if requested_quantity <= 0:
        raise ValueError("Requested quantity must be positive")
    if not bool(requirement["substitution_allowed"]):
        raise ValueError("Substitution is not allowed for this requirement")
    reason_row = conn.execute(
        """
        SELECT 1
        FROM inventory_substitution_reason_codes
        WHERE code = ? AND active = 1
        """,
        (_clean_optional_str(reason_code),),
    ).fetchone()
    if reason_row is None:
        raise ValueError(f"Inventory substitution reason '{reason_code}' is not active")
    if substitute_inventory_item_id is not None and conn.execute(
        "SELECT 1 FROM inventory_items WHERE id = ?",
        (substitute_inventory_item_id,),
    ).fetchone() is None:
        raise ValueError(f"Inventory item {substitute_inventory_item_id} not found")

    timestamp = datetime.now(UTC).isoformat()
    conn.execute(
        """
        INSERT INTO inventory_substitution_requests (
            job_id,
            segment_id,
            requirement_id,
            inventory_item_id,
            substitute_inventory_item_id,
            requested_quantity,
            status,
            requested_by,
            reason_code,
            note,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, 'requested', ?, ?, ?, ?)
        """,
        (
            int(requirement["job_id"]),
            int(requirement["segment_id"]),
            int(requirement["id"]),
            int(requirement["inventory_item_id"]) if requirement["inventory_item_id"] is not None else None,
            substitute_inventory_item_id,
            float(requested_quantity),
            _clean_optional_str(requested_by),
            _clean_optional_str(reason_code),
            _clean_optional_str(note),
            timestamp,
        ),
    )
    conn.commit()
    return conn.execute(
        "SELECT * FROM inventory_substitution_requests WHERE id = last_insert_rowid()"
    ).fetchone()


def decide_inventory_substitution(
    conn: sqlite3.Connection,
    *,
    substitution_id: int,
    status: str,
    approved_by: str | None,
    approved_role: str | None,
    approved_quantity: float | None = None,
    note: str | None = None,
    substitute_inventory_item_id: int | None = None,
) -> sqlite3.Row:
    """Approve or reject a substitution request."""

    ensure_dashboard_tables(conn)
    _seed_inventory_substitution_reason_codes(conn)
    if status not in {"approved", "rejected"}:
        raise ValueError("Substitution decision status must be 'approved' or 'rejected'")
    normalized_role = _clean_optional_str(approved_role)
    if normalized_role not in INVENTORY_SUBSTITUTION_APPROVER_ROLES:
        raise ValueError(
            "Inventory substitution approval role must be one of: "
            + ", ".join(INVENTORY_SUBSTITUTION_APPROVER_ROLES)
        )
    if not _clean_optional_str(approved_by):
        raise ValueError("Approver identifier is required for inventory substitution decisions")
    existing = conn.execute(
        "SELECT * FROM inventory_substitution_requests WHERE id = ?",
        (substitution_id,),
    ).fetchone()
    if existing is None:
        raise ValueError(f"Substitution request {substitution_id} not found")
    if substitute_inventory_item_id is not None and conn.execute(
        "SELECT 1 FROM inventory_items WHERE id = ?",
        (substitute_inventory_item_id,),
    ).fetchone() is None:
        raise ValueError(f"Inventory item {substitute_inventory_item_id} not found")
    quantity_value = (
        float(approved_quantity)
        if approved_quantity is not None
        else float(existing["requested_quantity"] or 0)
    )
    decided_at = datetime.now(UTC).isoformat()
    conn.execute(
        """
        UPDATE inventory_substitution_requests
        SET status = ?,
            approved_by = ?,
            approved_role = ?,
            approved_quantity = ?,
            substitute_inventory_item_id = COALESCE(?, substitute_inventory_item_id),
            note = COALESCE(?, note),
            decided_at = ?
        WHERE id = ?
        """,
        (
            status,
            _clean_optional_str(approved_by),
            normalized_role,
            quantity_value if status == "approved" else None,
            substitute_inventory_item_id,
            _clean_optional_str(note),
            decided_at,
            substitution_id,
        ),
    )
    conn.commit()
    return conn.execute(
        "SELECT * FROM inventory_substitution_requests WHERE id = ?",
        (substitution_id,),
    ).fetchone()


def list_inventory_substitutions(
    conn: sqlite3.Connection,
    *,
    job_id: int | None = None,
    segment_id: int | None = None,
    status: str | None = None,
) -> list[dict[str, object]]:
    """Return substitution requests and decisions with requirement context."""

    ensure_dashboard_tables(conn)
    filters: list[str] = []
    params: list[object] = []
    if job_id is not None:
        filters.append("s.job_id = ?")
        params.append(job_id)
    if segment_id is not None:
        filters.append("s.segment_id = ?")
        params.append(segment_id)
    if status is not None:
        filters.append("s.status = ?")
        params.append(status)
    where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""
    rows = conn.execute(
        f"""
        SELECT
            s.*,
            ir.requirement_name,
            i.name AS inventory_name,
            si.name AS substitute_inventory_name
        FROM inventory_substitution_requests AS s
        JOIN inventory_requirements AS ir ON ir.id = s.requirement_id
        LEFT JOIN inventory_items AS i ON i.id = s.inventory_item_id
        LEFT JOIN inventory_items AS si ON si.id = s.substitute_inventory_item_id
        {where_sql}
        ORDER BY s.created_at DESC, s.id DESC
        """,
        params,
    ).fetchall()
    return [
        {
            "substitutionId": int(row["id"]),
            "jobId": int(row["job_id"]),
            "segmentId": int(row["segment_id"]),
            "requirementId": int(row["requirement_id"]),
            "requirementName": row["requirement_name"],
            "inventoryName": row["inventory_name"],
            "substituteInventoryItemId": int(row["substitute_inventory_item_id"]) if row["substitute_inventory_item_id"] is not None else None,
            "substituteInventoryName": row["substitute_inventory_name"],
            "requestedQuantity": float(row["requested_quantity"] or 0.0),
            "approvedQuantity": float(row["approved_quantity"]) if row["approved_quantity"] is not None else None,
            "status": row["status"],
            "requestedBy": row["requested_by"],
            "approvedBy": row["approved_by"],
            "approvedRole": row["approved_role"],
            "reasonCode": row["reason_code"],
            "note": row["note"],
            "createdAt": row["created_at"],
            "decidedAt": row["decided_at"],
        }
        for row in rows
    ]


def list_inventory_requirements(
    conn: sqlite3.Connection,
    *,
    job_id: int | None = None,
    segment_id: int | None = None,
) -> list[dict[str, object]]:
    """Return inventory requirement lines with allocated and shortage quantities."""

    ensure_dashboard_tables(conn)
    filters: list[str] = []
    params: list[object] = []
    if job_id is not None:
        filters.append("ir.job_id = ?")
        params.append(job_id)
    if segment_id is not None:
        filters.append("ir.segment_id = ?")
        params.append(segment_id)
    where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""

    rows = conn.execute(
        f"""
        WITH substitution_summary AS (
            SELECT
                requirement_id,
                COALESCE(SUM(CASE
                    WHEN status = 'approved' THEN COALESCE(approved_quantity, requested_quantity)
                    ELSE 0
                END), 0) AS approved_substitution_quantity,
                COALESCE(SUM(CASE
                    WHEN status = 'requested' THEN requested_quantity
                    ELSE 0
                END), 0) AS requested_substitution_quantity,
                MAX(CASE WHEN status = 'requested' THEN 1 ELSE 0 END) AS has_pending_substitution
            FROM inventory_substitution_requests
            GROUP BY requirement_id
        ),
        latest_execution AS (
            SELECT requirement_id, MAX(id) AS latest_event_id
            FROM inventory_execution_events
            WHERE requirement_id IS NOT NULL
            GROUP BY requirement_id
        )
        SELECT
            ir.*,
            js.segment_sequence,
            js.from_location,
            js.to_location,
            i.name AS inventory_name,
            i.unit,
            i.architecture AS inventory_architecture,
            COALESCE(SUM(s.quantity), 0) AS allocated_quantity,
            COALESCE(ss.approved_substitution_quantity, 0) AS approved_substitution_quantity,
            COALESCE(ss.requested_substitution_quantity, 0) AS requested_substitution_quantity,
            COALESCE(ss.has_pending_substitution, 0) AS has_pending_substitution,
            ie.stage AS execution_stage,
            ie.actor AS execution_actor,
            ie.note AS execution_note,
            ie.container_ref AS execution_container_ref,
            ie.truck_id AS execution_truck_id,
            ie.created_at AS execution_recorded_at
        FROM inventory_requirements AS ir
        JOIN job_segments AS js ON js.id = ir.segment_id
        LEFT JOIN inventory_items AS i ON i.id = ir.inventory_item_id
        LEFT JOIN substitution_summary AS ss ON ss.requirement_id = ir.id
        LEFT JOIN latest_execution AS le ON le.requirement_id = ir.id
        LEFT JOIN inventory_execution_events AS ie ON ie.id = le.latest_event_id
        LEFT JOIN shipments AS s
            ON s.segment_id = ir.segment_id
            AND (
                (ir.inventory_item_id IS NOT NULL AND s.inventory_item_id = ir.inventory_item_id)
                OR (
                    ir.inventory_item_id IS NULL
                    AND s.inventory_item_id IN (
                        SELECT id FROM inventory_items WHERE name = ir.requirement_name
                    )
                )
            )
        {where_sql}
        GROUP BY
            ir.id,
            ir.job_id,
            ir.segment_id,
            ir.inventory_item_id,
            ir.requirement_name,
            ir.required_quantity,
            ir.substitution_allowed,
            ir.architecture,
            ir.notes,
            ir.created_at,
            ir.updated_at,
            js.segment_sequence,
            js.from_location,
            js.to_location,
            i.name,
            i.unit,
            i.architecture,
            ss.approved_substitution_quantity,
            ss.requested_substitution_quantity,
            ss.has_pending_substitution,
            ie.stage,
            ie.actor,
            ie.note,
            ie.container_ref,
            ie.truck_id,
            ie.created_at
        ORDER BY ir.job_id, js.segment_sequence, ir.requirement_name
        """,
        params,
    ).fetchall()

    payload: list[dict[str, object]] = []
    for row in rows:
        required_quantity = float(row["required_quantity"] or 0)
        allocated_quantity = float(row["allocated_quantity"] or 0)
        approved_substitution_quantity = float(row["approved_substitution_quantity"] or 0.0)
        requested_substitution_quantity = float(row["requested_substitution_quantity"] or 0.0)
        effective_fulfilled_quantity = allocated_quantity + approved_substitution_quantity
        shortage_quantity = max(required_quantity - effective_fulfilled_quantity, 0.0)
        payload.append(
            {
                "requirementId": int(row["id"]),
                "jobId": int(row["job_id"]),
                "segmentId": int(row["segment_id"]),
                "segmentSequence": int(row["segment_sequence"]),
                "fromLocation": row["from_location"],
                "toLocation": row["to_location"],
                "inventoryItemId": int(row["inventory_item_id"]) if row["inventory_item_id"] is not None else None,
                "inventoryName": row["inventory_name"],
                "requirementName": row["requirement_name"],
                "requiredQuantity": required_quantity,
                "allocatedQuantity": allocated_quantity,
                "approvedSubstitutionQuantity": approved_substitution_quantity,
                "requestedSubstitutionQuantity": requested_substitution_quantity,
                "effectiveFulfilledQuantity": effective_fulfilled_quantity,
                "shortageQuantity": shortage_quantity,
                "substitutionAllowed": bool(row["substitution_allowed"]),
                "hasPendingSubstitution": bool(row["has_pending_substitution"]),
                "architecture": row["architecture"] or row["inventory_architecture"] or "general",
                "unit": row["unit"] or "unit",
                "notes": row["notes"],
                "executionStage": row["execution_stage"] or "required",
                "executionActor": row["execution_actor"],
                "executionNote": row["execution_note"],
                "executionContainerRef": row["execution_container_ref"],
                "executionTruckId": row["execution_truck_id"],
                "executionRecordedAt": row["execution_recorded_at"],
            }
        )
    return payload


def list_inventory_exceptions(
    conn: sqlite3.Connection, *, resolved: bool | None = None
) -> list[sqlite3.Row]:
    """Return inventory exceptions, optionally filtering by resolved status."""

    where = ""
    params: list[object] = []
    if resolved is True:
        where = "WHERE resolved_at IS NOT NULL"
    elif resolved is False:
        where = "WHERE resolved_at IS NULL"

    return list(
        conn.execute(
            f"""
            SELECT e.*, i.name AS inventory_name, i.job_id AS inventory_job_id
            FROM inventory_exceptions AS e
            LEFT JOIN inventory_items AS i ON i.id = e.inventory_item_id
            {where}
            ORDER BY e.noted_at DESC
            """,
            params,
        )
    )


def resolve_inventory_exception(
    conn: sqlite3.Connection, exception_id: int, *, note: str | None = None
) -> None:
    """Mark an inventory exception as resolved."""

    timestamp = datetime.now(UTC).isoformat()
    conn.execute(
        """
        UPDATE inventory_exceptions
        SET resolved_at = ?, resolution_note = COALESCE(?, resolution_note)
        WHERE id = ?
        """,
        (timestamp, note, exception_id),
    )
    conn.commit()


def list_segment_inventory_coordination(
    conn: sqlite3.Connection,
    *,
    job_id: int | None = None,
) -> list[dict[str, object]]:
    """Return segment-level inventory and supplier coordination summaries."""

    ensure_dashboard_tables(conn)
    requirements = list_inventory_requirements(conn, job_id=job_id)
    requirements_by_segment: dict[int, list[dict[str, object]]] = {}
    for item in requirements:
        requirements_by_segment.setdefault(int(item["segmentId"]), []).append(item)
    params: list[object] = []
    where = ""
    if job_id is not None:
        where = "WHERE js.job_id = ?"
        params.append(job_id)
    segment_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(job_segments)").fetchall()
    }
    assignment_status_sql = (
        "js.assignment_status"
        if "assignment_status" in segment_columns
        else "'draft'"
    )

    rows = conn.execute(
        f"""
        SELECT
            js.id AS segment_id,
            js.job_id,
            js.segment_sequence,
            js.from_location,
            js.to_location,
            js.planned_start,
            js.planned_end,
            {assignment_status_sql} AS assignment_status,
            COUNT(DISTINCT s.id) AS shipment_count,
            COALESCE(SUM(s.quantity), 0) AS allocated_quantity,
            GROUP_CONCAT(DISTINCT i.name) AS inventory_names,
            GROUP_CONCAT(DISTINCT sup.company_name) AS supplier_names
        FROM job_segments AS js
        LEFT JOIN shipments AS s ON s.segment_id = js.id
        LEFT JOIN inventory_items AS i ON i.id = s.inventory_item_id
        LEFT JOIN suppliers AS sup ON sup.id = i.supplier_id
        {where}
        GROUP BY
            js.id,
            js.job_id,
            js.segment_sequence,
            js.from_location,
            js.to_location,
            js.planned_start,
            js.planned_end,
            {assignment_status_sql}
        ORDER BY js.job_id, js.segment_sequence
        """,
        params,
    ).fetchall()

    payload: list[dict[str, object]] = []
    for row in rows:
        inventory_names = sorted(
            {
                item.strip()
                for item in str(row["inventory_names"] or "").split(",")
                if item and item.strip()
            }
        )
        supplier_names = sorted(
            {
                item.strip()
                for item in str(row["supplier_names"] or "").split(",")
                if item and item.strip()
            }
        )
        payload.append(
            {
                "segmentId": int(row["segment_id"]),
                "jobId": int(row["job_id"]),
                "segmentSequence": int(row["segment_sequence"]),
                "fromLocation": row["from_location"],
                "toLocation": row["to_location"],
                "plannedStart": row["planned_start"],
                "plannedEnd": row["planned_end"],
                "assignmentStatus": row["assignment_status"],
                "shipmentCount": int(row["shipment_count"] or 0),
                "allocatedQuantity": float(row["allocated_quantity"] or 0),
                "inventoryNames": inventory_names,
                "supplierNames": supplier_names,
            }
        )
    for entry in payload:
        segment_requirements = requirements_by_segment.get(int(entry["segmentId"]), [])
        non_sub_shortage = sum(
            float(item["shortageQuantity"])
            for item in segment_requirements
            if not bool(item["substitutionAllowed"])
        )
        substitutable_shortage = sum(
            float(item["shortageQuantity"])
            for item in segment_requirements
            if bool(item["substitutionAllowed"])
        )
        entry["requirementCount"] = len(segment_requirements)
        entry["requiredQuantity"] = sum(float(item["requiredQuantity"]) for item in segment_requirements)
        entry["shortageQuantity"] = sum(float(item["shortageQuantity"]) for item in segment_requirements)
        entry["blockingShortageQuantity"] = non_sub_shortage
        entry["warningShortageQuantity"] = substitutable_shortage
        entry["shortageCount"] = sum(1 for item in segment_requirements if float(item["shortageQuantity"]) > 0)
        entry["requirementNames"] = [
            str(item["requirementName"]) for item in segment_requirements
        ]
        entry["executionStages"] = sorted(
            {
                str(item["executionStage"])
                for item in segment_requirements
                if item.get("executionStage")
            }
        )
        entry["pendingSubstitutionCount"] = sum(
            1 for item in segment_requirements if bool(item.get("hasPendingSubstitution"))
        )
        entry["approvedSubstitutionQuantity"] = sum(
            float(item.get("approvedSubstitutionQuantity") or 0.0)
            for item in segment_requirements
        )
        entry["containerRequirementCount"] = sum(
            1
            for item in segment_requirements
            if str(item.get("architecture") or "") == "container"
        )
        entry["containerRequiredQuantity"] = sum(
            float(item.get("requiredQuantity") or 0.0)
            for item in segment_requirements
            if str(item.get("architecture") or "") == "container"
        )
        entry["containerAllocatedQuantity"] = sum(
            float(item.get("allocatedQuantity") or 0.0)
            for item in segment_requirements
            if str(item.get("architecture") or "") == "container"
        )
        entry["containerShortageQuantity"] = sum(
            float(item.get("shortageQuantity") or 0.0)
            for item in segment_requirements
            if str(item.get("architecture") or "") == "container"
        )
        entry["architectures"] = sorted(
            {
                str(item["architecture"])
                for item in segment_requirements
                if item.get("architecture")
            }
        )
    return payload


def allocate_inventory_to_segment(
    conn: sqlite3.Connection,
    *,
    segment_id: int,
    inventory_item_id: int,
    quantity: float,
    status: str = "planned",
) -> sqlite3.Row:
    """Allocate stock to a job segment using a segment-linked shipment."""

    ensure_dashboard_tables(conn)
    segment = conn.execute(
        """
        SELECT id, job_id, from_location, to_location, planned_start
        FROM job_segments
        WHERE id = ?
        """,
        (segment_id,),
    ).fetchone()
    if segment is None:
        raise ValueError(f"Segment {segment_id} not found")
    item = conn.execute(
        "SELECT id FROM inventory_items WHERE id = ?",
        (inventory_item_id,),
    ).fetchone()
    if item is None:
        raise ValueError(f"Inventory item {inventory_item_id} not found")
    if quantity <= 0:
        raise ValueError("Quantity must be positive")

    from .shipments import create_shipment

    return create_shipment(
        conn,
        job_id=int(segment["job_id"]),
        inventory_item_id=inventory_item_id,
        segment_id=int(segment["id"]),
        quantity=quantity,
        from_location=segment["from_location"],
        to_location=segment["to_location"],
        scheduled_date=segment["planned_start"],
        status=status,
    )


def _clean_optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    value_str = str(value).strip()
    return value_str or None


def _normalize_supplier_column(column_name: str) -> str:
    normalized = column_name.strip().lower().replace(" ", "_")
    if normalized in {"company", "supplier"}:
        return "company_name"
    if normalized in {"contact", "contact_person"}:
        return "contact_name"
    if normalized in {"phone", "phone_number"}:
        return "contact_number"
    return normalized


def _normalize_inventory_column(column_name: str) -> str:
    normalized = column_name.strip().lower().replace(" ", "_")
    aliases = {
        "item": "name",
        "inventory": "name",
        "item_name": "name",
        "item_id": "item_id",
        "job": "job_id",
        "state": "state",
        "type": "architecture",
        "inventory_type": "architecture",
        "asset": "asset_tag",
        "barcode": "asset_tag",
    }
    return aliases.get(normalized, normalized)


def _build_suppliers_sheet_url(
    sheet_id: str | None,
    sheet_name: str,
    *,
    env_var: str = "SUPPLIERS_SHEET_ID",
) -> str | None:
    resolved_reference = resolve_google_sheet_reference(
        sheet_id,
        env_keys=(
            env_var,
            "SUPPLIERS_SHEET_URL",
            "OPERATIONS_WORKBOOK_SHEET_ID",
            "OPERATIONS_WORKBOOK_URL",
        ),
    )
    if not resolved_reference:
        return None
    if resolved_reference.startswith("http") and "gviz/tq" in resolved_reference:
        return resolved_reference
    return build_google_sheet_csv_url(resolved_reference, sheet_name)


__all__ = [
    "INVENTORY_ARCHITECTURES",
    "INVENTORY_CUSTODY_TYPES",
    "INVENTORY_EXECUTION_STAGES",
    "INVENTORY_STATES",
    "INVENTORY_SUBSTITUTION_APPROVER_ROLES",
    "INVENTORY_SUBSTITUTION_STATUSES",
    "ensure_suppliers_table",
    "get_allowed_inventory_execution_stages",
    "get_inventory_balance",
    "import_inventory_items_from_dataframe",
    "import_inventory_movements_from_dataframe",
    "import_suppliers_from_google_sheet",
    "list_inventory_execution_events",
    "list_inventory_requirements",
    "list_segment_inventory_coordination",
    "list_inventory",
    "list_inventory_balances",
    "list_inventory_movements",
    "list_inventory_substitution_reason_codes",
    "list_inventory_substitutions",
    "list_inventory_exceptions",
    "list_suppliers",
    "record_inventory_movement",
    "record_inventory_execution_event",
    "allocate_inventory_to_segment",
    "decide_inventory_substitution",
    "resolve_inventory_exception",
    "request_inventory_substitution",
    "upsert_inventory_substitution_reason_code",
    "upsert_inventory_requirement",
    "upsert_inventory_item",
    "upsert_supplier",
]
