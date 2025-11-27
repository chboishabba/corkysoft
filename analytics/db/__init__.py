<<<<<<< HEAD
"""Database facade module re-exporting domain helpers."""
=======
"""A refactored database access layer for analytics features."""
>>>>>>> c3ed293 (Remove tracked pycache)
from __future__ import annotations

from .connection import (
    DEFAULT_DB_PATH,
<<<<<<< HEAD
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
    upsert_container_booking,
    upsert_job_by_number,
    upsert_job_container_allocation,
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
    "upsert_container_booking",
    "upsert_driver_shift",
    "upsert_inventory_item",
    "upsert_job_by_number",
    "upsert_job_container_allocation",
    "upsert_job_segment",
    "upsert_supplier",
    "upsert_truck",
    "upsert_vehicle_details",
    "upsert_worker",
]
=======
    get_connection,
    connection_scope,
)
from .parameters import (
    ensure_global_parameters_table,
    get_parameter_value,
    set_parameter_value,
    bootstrap_parameters,
)
from .inventory import (
    upsert_inventory_item,
    list_inventory_balances,
    get_inventory_balance,
    record_inventory_movement,
    list_inventory,
    upsert_supplier,
    list_suppliers,
    import_suppliers_from_google_sheet,
)
from .fleet import (
    upsert_truck,
    upsert_vehicle_details,
    upsert_worker,
    import_workers_from_staff_sheet,
)
from .shipments import (
    upsert_driver_shift,
    create_shipment,
    fetch_driver_shifts,
    rollup_driver_shift_costs_by_job,
    fetch_shipments_with_context,
)
from .schema import (
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
>>>>>>> c3ed293 (Remove tracked pycache)
