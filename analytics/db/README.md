# `analytics.db` package overview

The original `analytics/db.py` module has been split into domain-focused modules to reduce coupling and make tests easier to target. This package groups those responsibilities while keeping a facade import layer in `__init__.py` for backward compatibility.

## Module responsibilities
- `connection.py` – database plumbing, table helpers, and bootstrap utilities such as `get_connection`, `_create_table_if_missing`, and migration helpers.
- `parameters.py` – global parameter table creation plus `get_parameter_value`/`set_parameter_value` helpers and parameter bootstrap routines.
- `inventory.py` – supplier/inventory tables, CRUD helpers, balance queries, and supplier import utilities.
- `fleet.py` and `workers.py` – vehicle/truck metadata upserts alongside worker import/upsert helpers.
- `shipments.py` and `shifts.py` – shipment creation/update helpers, driver shift import/upsert logic, and reporting rollups for shipments and shifts.
- `legacy.py` – transitional utilities for dashboard-specific tables and historical route migrations that have not yet been retired.

## Using the facade
Public APIs remain importable from `analytics.db` to avoid churn while downstream code is migrated:

```python
from analytics import db

conn = db.get_connection()
db.bootstrap_parameters(conn)
db.upsert_inventory_item(conn, item)
```

Import from submodules when you want tighter scoping or more explicit ownership of a domain:

```python
from analytics.db import inventory, shipments

shipments.create_shipment(conn, job_number, payload)
```
