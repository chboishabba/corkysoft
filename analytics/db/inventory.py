"""Inventory and supplier helpers extracted from the legacy database module."""
from __future__ import annotations

import sqlite3
from typing import IO, Iterable

from .legacy import (
    ensure_suppliers_table,
    get_inventory_balance,
    import_suppliers_from_google_sheet,
    list_inventory,
    list_inventory_balances,
    list_suppliers,
    record_inventory_movement,
    upsert_inventory_item,
    upsert_supplier,
)

__all__ = [
    "ensure_suppliers_table",
    "get_inventory_balance",
    "import_suppliers_from_google_sheet",
    "list_inventory",
    "list_inventory_balances",
    "list_suppliers",
    "record_inventory_movement",
    "upsert_inventory_item",
    "upsert_supplier",
]
