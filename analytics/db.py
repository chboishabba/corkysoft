"""Database helpers for analytics features."""
from __future__ import annotations

from .db.connection import (
    DEFAULT_DB_PATH,
    get_connection,
    connection_scope,
)
from .db.parameters import (
    ensure_global_parameters_table,
    get_parameter_value,
    set_parameter_value,
    bootstrap_parameters,
)
from .db.inventory import (
    upsert_inventory_item,
    list_inventory_balances,
    get_inventory_balance,
    record_inventory_movement,
    list_inventory,
    upsert_supplier,
    list_suppliers,
    import_suppliers_from_google_sheet,
)
from .db.fleet import (
    import_workers_from_dataframe,
    import_workers_from_google_sheet,
    upsert_truck,
    upsert_vehicle_details,
    upsert_worker,
    import_workers_from_staff_sheet,
)
from .db.shipments import (
    upsert_driver_shift,
    create_shipment,
    fetch_driver_shifts,
    rollup_driver_shift_costs_by_job,
    fetch_shipments_with_context,
)
from .db.schema import (
    _DASHBOARD_SCHEMA_SQL,
    ensure_dashboard_tables,
    migrate_geojson_to_routes,
    ensure_historical_job_routes_table,
)


__all__ = [
    "DEFAULT_DB_PATH",
    "get_connection",
    "connection_scope",
    "ensure_global_parameters_table",
    "get_parameter_value",
    "set_parameter_value",
    "bootstrap_parameters",
    "ensure_dashboard_tables",
    "migrate_geojson_to_routes",
    "upsert_inventory_item",
    "list_inventory_balances",
    "get_inventory_balance",
    "record_inventory_movement",
    "list_inventory",
    "upsert_supplier",
    "list_suppliers",
    "import_suppliers_from_google_sheet",
    "import_workers_from_dataframe",
    "import_workers_from_google_sheet",
    "upsert_truck",
    "upsert_vehicle_details",
    "upsert_worker",
    "import_workers_from_staff_sheet",
    "upsert_driver_shift",
    "create_shipment",
    "fetch_driver_shifts",
    "rollup_driver_shift_costs_by_job",
    "fetch_shipments_with_context",
    "_DASHBOARD_SCHEMA_SQL",
    "ensure_historical_job_routes_table",
]
