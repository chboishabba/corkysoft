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
from .schema import ensure_suppliers_table

INVENTORY_STATES: Sequence[str] = (
    "created",
    "staged",
    "loaded",
    "in_transit",
    "delivered",
    "exception",
)


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
) -> sqlite3.Row:
    """Create or update an inventory item and return the stored row."""

    timestamp = datetime.now(UTC).isoformat()
    state_value = state if state in INVENTORY_STATES else "created"
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
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            description = excluded.description,
            quantity = excluded.quantity,
            unit = excluded.unit,
            supplier_id = excluded.supplier_id,
            job_id = COALESCE(excluded.job_id, inventory_items.job_id),
            state = excluded.state,
            item_id = COALESCE(excluded.item_id, inventory_items.item_id),
            asset_tag = COALESCE(excluded.asset_tag, inventory_items.asset_tag),
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
) -> sqlite3.Row:
    """Insert an inventory movement entry and return the stored row."""

    timestamp = datetime.now(UTC).isoformat()
    state_value = state if state in INVENTORY_STATES else None
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
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            timestamp,
        ),
    )

    if state_value or job_id is not None:
        conn.execute(
            """
            UPDATE inventory_items
            SET state = COALESCE(?, state),
                job_id = COALESCE(?, job_id),
                updated_at = ?
            WHERE id = ?
            """,
            (state_value, job_id, timestamp, inventory_item_id),
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
                i.job_id AS item_job_id
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
        record_inventory_movement(
            conn,
            inventory_item_id=int(inventory_item_id),
            job_id=int(job_id) if pd.notna(job_id) else None,
            change_on_hand=int(row.get("change_on_hand") or 0),
            change_allocated=int(row.get("change_allocated") or 0),
            state=str(state) if pd.notna(state) and state else None,
            sequence_no=int(sequence_no) if pd.notna(sequence_no) else None,
            reason=str(reason) if reason is not None else None,
        )
        imported += 1
    return imported


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
    "INVENTORY_STATES",
    "ensure_suppliers_table",
    "get_inventory_balance",
    "import_inventory_items_from_dataframe",
    "import_inventory_movements_from_dataframe",
    "import_suppliers_from_google_sheet",
    "list_inventory",
    "list_inventory_balances",
    "list_inventory_movements",
    "list_inventory_exceptions",
    "list_suppliers",
    "record_inventory_movement",
    "resolve_inventory_exception",
    "upsert_inventory_item",
    "upsert_supplier",
]
