"""Database facade module re-exporting domain helpers."""
from __future__ import annotations

from .connection import DEFAULT_DB_PATH, connection_scope, get_connection
from .fleet import import_workers_from_staff_sheet, upsert_truck, upsert_vehicle_details, upsert_worker
from .inventory import (
    get_inventory_balance,
    import_suppliers_from_google_sheet,
    list_inventory,
    list_inventory_balances,
    list_suppliers,
    record_inventory_movement,
    upsert_inventory_item,
    upsert_supplier,
)
from .parameters import bootstrap_parameters, ensure_global_parameters_table, get_parameter_value, set_parameter_value
from .schema import _DASHBOARD_SCHEMA_SQL, ensure_dashboard_tables, ensure_historical_job_routes_table, migrate_geojson_to_routes
from .shipments import (
    create_shipment,
    fetch_driver_shifts,
    fetch_shipments_with_context,
    rollup_driver_shift_costs_by_job,
    upsert_driver_shift,
)

__all__ = [
    "DEFAULT_DB_PATH",
    "_DASHBOARD_SCHEMA_SQL",
    "bootstrap_parameters",
    "connection_scope",
    "create_shipment",
    "ensure_dashboard_tables",
    "ensure_global_parameters_table",
    "ensure_historical_job_routes_table",
    "fetch_driver_shifts",
    "fetch_shipments_with_context",
    "get_connection",
    "get_inventory_balance",
    "get_parameter_value",
    "import_suppliers_from_google_sheet",
    "import_workers_from_staff_sheet",
    "list_inventory",
    "list_inventory_balances",
    "list_suppliers",
    "migrate_geojson_to_routes",
    "record_inventory_movement",
    "rollup_driver_shift_costs_by_job",
    "set_parameter_value",
    "upsert_driver_shift",
    "upsert_inventory_item",
    "upsert_supplier",
    "upsert_truck",
    "upsert_vehicle_details",
    "upsert_worker",
]
