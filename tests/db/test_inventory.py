import sqlite3

from analytics.db import (
    ensure_dashboard_tables,
    get_inventory_balance,
    record_inventory_movement,
    upsert_inventory_item,
    upsert_supplier,
)


def test_inventory_flow_tracks_on_hand_quantity() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    ensure_dashboard_tables(conn)
    supplier = upsert_supplier(conn, company_name="ACME Supplies")
    item = upsert_inventory_item(
        conn, name="Gravel", description="Fine gravel", quantity=5, supplier_id=supplier["id"]
    )

    record_inventory_movement(conn, inventory_item_id=item["id"], change_on_hand=3)
    balance = get_inventory_balance(conn, item["id"])

    assert balance is not None
    assert balance["on_hand_quantity"] == 8
