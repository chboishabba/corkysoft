import sqlite3

from analytics.db import ensure_dashboard_tables
from analytics.db.connection import _table_exists


def test_ensure_dashboard_tables_creates_core_tables() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    ensure_dashboard_tables(conn)

    assert _table_exists(conn, "jobs")
    assert _table_exists(conn, "shipments")
    assert _table_exists(conn, "inventory_items")
