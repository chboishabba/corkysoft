# `analytics.db` package overview

The original `analytics/db.py` module was split into domain-focused modules to reduce coupling and make tests and ownership boundaries clearer. This package keeps a facade import layer in `__init__.py` so existing imports from `analytics.db` continue to work while newer code can target narrower submodules.

## Module responsibilities

- `connection.py` - database connection plumbing and bootstrap helpers such as `get_connection`, `connection_scope`, and shared table utilities.
- `schema.py` - dashboard schema creation and migration helpers such as `ensure_dashboard_tables`, `ensure_historical_job_routes_table`, and route migration helpers.
- `parameters.py` - global parameter table creation plus parameter bootstrap and get/set helpers.
- `inventory.py` - supplier and inventory CRUD, balances, movements, requirements, substitutions, and supplier import utilities.
- `fleet.py` and `workers.py` - truck, vehicle, and worker import and upsert helpers.
- `shipments.py` and `shifts.py` - shipment creation and update helpers, driver shift imports, and reporting rollups.
- `absence.py` - worker absence records and related query helpers.
- `site_media.py` - site assessment and uploaded media persistence helpers.
- `legacy.py` - transitional utilities for older dashboard tables and historical compatibility paths that have not yet been retired.

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
inventory.list_inventory_requirements(conn)
```
