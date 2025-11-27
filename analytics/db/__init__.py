"""Database facade module re-exporting domain helpers."""
from __future__ import annotations

from .connection import (
    DEFAULT_DB_PATH,
    _create_table_if_missing,
    _table_columns,
    _table_exists,
    _unique_index_columns,
    connection_scope,
    get_connection,
    initialize_database,
)
from .fleet import upsert_truck, upsert_vehicle_details
from .inventory import (
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
from .parameters import (
    bootstrap_parameters,
    ensure_global_parameters_table,
    get_parameter_value,
    set_parameter_value,
)
from .shifts import (
    fetch_driver_shifts,
    rollup_driver_shift_costs_by_job,
    upsert_driver_shift,
)
from .shipments import (
    create_shipment,
    fetch_shipments_with_context,
    upsert_container,
    upsert_container_allocation,
    upsert_container_booking,
    upsert_job_by_number,
    upsert_job_segment,
)
from .workers import import_workers_from_staff_sheet, upsert_worker
from .legacy import (
    ensure_dashboard_tables,
    ensure_historical_job_routes_table,
    migrate_geojson_to_routes,
)

__all__ = [
    "DEFAULT_DB_PATH",
    "_create_table_if_missing",
    "_table_columns",
    "_table_exists",
    "_unique_index_columns",
    "bootstrap_parameters",
    "connection_scope",
    "create_shipment",
    "ensure_dashboard_tables",
    "ensure_global_parameters_table",
    "ensure_historical_job_routes_table",
    "ensure_suppliers_table",
    "fetch_driver_shifts",
    "fetch_shipments_with_context",
    "get_connection",
    "get_inventory_balance",
    "get_parameter_value",
    "import_suppliers_from_google_sheet",
    "import_workers_from_staff_sheet",
    "initialize_database",
    "list_inventory",
    "list_inventory_balances",
    "list_suppliers",
    "migrate_geojson_to_routes",
    "record_inventory_movement",
    "rollup_driver_shift_costs_by_job",
    "set_parameter_value",
    "upsert_container",
    "upsert_container_allocation",
    "upsert_container_booking",
    "upsert_driver_shift",
    "upsert_inventory_item",
    "upsert_job_by_number",
    "upsert_job_segment",
    "upsert_supplier",
    "upsert_truck",
    "upsert_vehicle_details",
    "upsert_worker",
]
