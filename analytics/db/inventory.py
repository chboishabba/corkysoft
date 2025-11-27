"""Inventory and supplier-related database functions."""
from __future__ import annotations

import os
import sqlite3
from datetime import UTC, datetime
from urllib.parse import quote_plus
from typing import IO, Iterable, Optional, Sequence

import pandas as pd

from .connection import _table_columns
from .schema import ensure_suppliers_table


def upsert_inventory_item(
    conn: sqlite3.Connection,
    *,
    name: str,
    description: str = "",
    quantity: int = 0,
    unit: str = "unit",
    supplier_id: int | None = None,
) -> sqlite3.Row:
    """Create or update an inventory item and return the stored row."""

    timestamp = datetime.now(UTC).isoformat()
    conn.execute(
        """
        INSERT INTO inventory_items (name, description, quantity, unit, supplier_id, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            description = excluded.description,
            quantity = excluded.quantity,
            unit = excluded.unit,
            supplier_id = excluded.supplier_id,
            updated_at = excluded.updated_at
        """,
        (name, description, int(quantity), unit, supplier_id, timestamp),
    )
    conn.commit()
    return conn.execute(
        "SELECT * FROM inventory_items WHERE name = ?", (name,)
    ).fetchone()


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


def list_inventory_balances(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return inventory items with on-hand, allocated, and available totals."""

    # _ensure_inventory_movements_table(conn)
    return list(conn.execute(_inventory_balance_query()))


def get_inventory_balance(
    conn: sqlite3.Connection,
    inventory_item_id: int
) -> sqlite3.Row | None:
    """Return a single inventory balance row by item id."""

    # _ensure_inventory_movements_table(conn)
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
) -> sqlite3.Row:
    """Insert an inventory movement entry and return the stored row."""

    # _ensure_inventory_movements_table(conn)
    timestamp = datetime.now(UTC).isoformat()
    conn.execute(
        """
        INSERT INTO inventory_movements (
            inventory_item_id,
            shipment_id,
            change_on_hand,
            change_allocated,
            reason,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            inventory_item_id,
            shipment_id,
            int(change_on_hand),
            int(change_allocated),
            reason or "",
            timestamp,
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

def upsert_supplier(
    conn: sqlite3.Connection,
    *,
    company_name: str,
    contact_name: str | None = None,
    contact_number: str | None = None,
    email: str | None = None,
    notes: str | None = None,
) -> sqlite3.Row:
    """Create or update a supplier record by company name."""

    if not company_name.strip():
        raise ValueError("Supplier company name is required")

    timestamp = datetime.now(UTC).isoformat()
    conn.execute(
        """
        INSERT INTO suppliers (
            company_name, contact_name, contact_number, email, notes, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(company_name) DO UPDATE SET
            contact_name = excluded.contact_name,
            contact_number = excluded.contact_number,
            email = excluded.email,
            notes = excluded.notes,
            updated_at = excluded.updated_at
        """,
        (
            company_name.strip(),
            _clean_optional_str(contact_name),
            _clean_optional_str(contact_number),
            _clean_optional_str(email),
            _clean_optional_str(notes),
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
        conn.execute(
            "SELECT * FROM suppliers ORDER BY LOWER(company_name), company_name"
        )
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
        )
        imported += 1
    return imported

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

def _build_suppliers_sheet_url(
    sheet_id: str | None,
    sheet_name: str,
    *, 
    env_var: str = "SUPPLIERS_SHEET_ID"
) -> str | None:
    resolved_id = sheet_id or os.environ.get(env_var)
    if not resolved_id:
        explicit_url = os.environ.get("SUPPLIERS_SHEET_URL")
        return explicit_url
    return (
        f"https://docs.google.com/spreadsheets/d/{resolved_id}/gviz/tq?tqx=out:csv"
        f"&sheet={quote_plus(sheet_name)}"
    )
