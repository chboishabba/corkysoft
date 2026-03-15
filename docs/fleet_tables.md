# Fleet vehicle, supplier, and driver tables

This note documents the operational tables added for fleet maintenance and labour tracking. It covers the column layouts, relationships, and ingestion helpers that keep vehicles, suppliers, and drivers aligned in the shared SQLite datastore.

## Vehicle metadata: `vehicle_details`

`vehicle_details` extends the legacy `trucks` table with registration, insurance, and pre-start data keyed by `truck_id`. Each row is a one-to-one extension of `trucks` and inherits its lifecycle through a foreign-key cascade. Key columns include registration (`rego`, `rego_expiry`), make/model/year, NHVR/insurance fields, odometer and last/next service markers, the currently assigned driver, and whether the daily check is complete.

## Repair ledger with suppliers: `vehicle_repairs`

`vehicle_repairs` captures workshop history per vehicle with a foreign key back to `trucks`. The ledger tracks the job item, optional description, spend, supplier/vendor name, service and next-service dates, and arbitrary notes. Inserts and updates are idempotent per truck/job/date/price combination, with a covering index on `(truck_id, service_date)` to support dashboard rollups. Import helpers in `analytics.vehicle_repairs` normalise column aliases (e.g., `vendor` or `workshop` map to `supplier`), coerce dates to ISO strings, and upsert rows while preserving created timestamps for reporting.

## Driver coverage and costs: `driver_shifts`

Driver shifts are stored in `driver_shifts`, keyed by shift date and optionally linked to `trucks`, `workers`, `jobs`, and `shipments`. The table records role assignments, ticket numbers, shift start/end, planned shift windows, hours, hourly rate, total cost, free-form notes, and the source sheet name used for import; a uniqueness constraint across date/truck/worker/time/tickets guards against accidental duplicates. The `analytics.driver_shifts` module imports the `VEHICLE_DRIVER` worksheet, resolves aliases for driver, vehicle, job/shipment references, and time fields, backfills `cost_total` when only hours and rate are present, and writes through to the shared tables via `upsert_driver_shift`. Queries that join worker, truck, shipment, and job context for auditing live in `fetch_driver_shifts`, enabling downstream dashboards to filter by date range, worker, or vehicle while preserving the imported metadata. Per-job labour rollups can be fetched through `rollup_driver_shift_costs_by_job` to attach shift hours and costs to job analytics.

## Planning truth and assignments

Operational spreadsheets remain import inputs, not assignment truth. Corkysoft is the
internal source of truth for:

- truck-to-segment assignments
- worker-to-segment assignments
- readiness/blocking state
- override capture when assignment policy is bypassed

The canonical planning unit is `job_segments`. This allows:

- multiple trucks on one job
- multiple workers on one segment
- multiple segments on one job
- segment-level readiness checks before work is confirmed

Imported spreadsheet fields such as `present_driver` are treated as contextual inputs
and reconciliation hints, not as final assignment state.

Current UI implication:

- Staff tab shows planned segment assignments separately from imported sheet truck context
- Fleet tab shows planned job-segment assignments separately from `present_driver`
- operators should treat `present_driver` as imported context only unless it is reconciled into segment assignments
- Fleet also acts as the maintenance/compliance cockpit for blocked and due-soon rego, COI, service, and worker compliance items
- Staff provides lightweight native role/compliance assignment actions so readiness data can be maintained in-app

## Assignment readiness

Assignment readiness combines imported fleet/staff state with Corkysoft planning state.
The minimum readiness model covers:

- truck registration expiry
- COI expiry
- next service due / overdue state
- worker role eligibility
- worker compliance expiry
- assignment conflicts against existing segment plans

Policy semantics:

- due soon -> warning
- expired / invalid -> hard block by default
- configurable rules may be overrideable
- non-overrideable legal/compliance failures remain hard blocks

Readiness is evaluated per segment so operators can understand exactly why a truck,
worker, or segment is blocked or at risk.
