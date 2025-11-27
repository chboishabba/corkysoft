# Fleet vehicle, supplier, and driver tables

This note documents the operational tables added for fleet maintenance and labour tracking. It covers the column layouts, relationships, and ingestion helpers that keep vehicles, suppliers, and drivers aligned in the shared SQLite datastore.

## Vehicle metadata: `vehicle_details`

`vehicle_details` extends the legacy `trucks` table with registration, insurance, and pre-start data keyed by `truck_id`. Each row is a one-to-one extension of `trucks` and inherits its lifecycle through a foreign-key cascade. Key columns include registration (`rego`, `rego_expiry`), make/model/year, NHVR/insurance fields, odometer and last/next service markers, the currently assigned driver, and whether the daily check is complete.【F:analytics/db.py†L118-L138】【F:analytics/db.py†L367-L421】

## Repair ledger with suppliers: `vehicle_repairs`

`vehicle_repairs` captures workshop history per vehicle with a foreign key back to `trucks`. The ledger tracks the job item, optional description, spend, supplier/vendor name, service and next-service dates, and arbitrary notes. Inserts and updates are idempotent per truck/job/date/price combination, with a covering index on `(truck_id, service_date)` to support dashboard rollups.【F:analytics/db.py†L140-L156】【F:analytics/vehicle_repairs.py†L15-L225】 Import helpers in `analytics.vehicle_repairs` normalise column aliases (e.g., `vendor` or `workshop` map to `supplier`), coerce dates to ISO strings, and upsert rows while preserving created timestamps for reporting.【F:analytics/vehicle_repairs.py†L15-L243】

## Driver coverage and costs: `driver_shifts`

Driver shifts are stored in `driver_shifts`, keyed by shift date and optionally linked to `trucks`, `workers`, `jobs`, and `shipments`. The table records role assignments, ticket numbers, shift start/end, planned shift windows, hours, hourly rate, total cost, free-form notes, and the source sheet name used for import; a uniqueness constraint across date/truck/worker/time/tickets guards against accidental duplicates.【F:analytics/db.py†L177-L205】 The `analytics.driver_shifts` module imports the `VEHICLE_DRIVER` worksheet, resolves aliases for driver, vehicle, job/shipment references, and time fields, backfills `cost_total` when only hours and rate are present, and writes through to the shared tables via `upsert_driver_shift`.【F:analytics/driver_shifts.py†L1-L196】 Queries that join worker, truck, shipment, and job context for auditing live in `fetch_driver_shifts`, enabling downstream dashboards to filter by date range, worker, or vehicle while preserving the imported metadata. Per-job labour rollups can be fetched through `rollup_driver_shift_costs_by_job` to attach shift hours and costs to job analytics.【F:analytics/db.py†L959-L1010】
