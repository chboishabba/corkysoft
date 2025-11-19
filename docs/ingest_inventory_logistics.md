# Ingest, Inventory, and Logistics Architecture

This note documents how operational data flows into Corkysoft, how inventory and evidence remain traceable, and how downstream logistics tooling consumes the same sources. Use it as the contract for API payloads, database linkage, and dashboard overlays.

## Ingest surfaces

- **REST payloads**: Ingest endpoints accept JSON with envelope fields (`source`, `payload_type`, `timestamp`, `version`) plus the domain payload (e.g., movement events, inventory scans, telemetry). All payloads must carry a `correlation_id` so cross-system retries can be deduplicated.
- **Batch uploads**: CSV/NDJSON imports flow through the same validation layer as REST submissions. The CLI wrappers in `routes_to_sqlite.py` reuse these schemas for offline seeding.
- **Hash & integrity**: Binary uploads (PEC stills, bodycam clips) follow the manifest rules in `docs/media_ingest.md`, computing hashes on device and verifying server-side before writing database references.
- **Idempotency & retries**: Ingest requests return a stable `ingest_id` and store the original payload with checksum. Clients retry with the same `correlation_id` to avoid duplicate rows.

## Inventory capture and reconciliation

- **Item identity**: Every item on a job carries an `item_id` (UUID) plus optional `asset_tag`/`barcode`. The ingest layer enforces uniqueness per job and rejects unknown tags unless a `create_missing=true` flag is present for supervised runs.
- **State model**: Inventory lines progress through `created` → `staged` → `loaded` → `in_transit` → `delivered` → `exception`. Each transition is a movement event that can be replayed to rebuild history.
- **Location hints**: Scans capture `lat`, `lon`, and `geofence_id` (depot/warehouse) so dashboards can reason about dwell time and custody. When GPS is absent, the depot geofence becomes the fallback.
- **Reconciliation jobs**: Nightly workers compare expected inventory states against the latest movement events. Drifts (items still `staged` after a route departure) are written to an `inventory_exceptions` table for dashboard surfacing.
- **Audit trail**: Each update stores `captured_by`, `role`, device metadata, and a monotonic `sequence_no` so out-of-order uploads can be reassembled.

## Logistics & movement events

- **Timeline contract**: Movement events represent every operational handoff (pickup arrival, dock exit, depot arrival, proof-of-delivery). They anchor media evidence and inventory state transitions through foreign keys.
- **Route context**: Events include `job_id`, `route_id`, and optional encoded polyline for the leg. A missing route falls back to the origin/destination coordinates stored on the job record.
- **Scheduling signals**: When an event indicates `delay_reason` or `eta_delta_minutes`, the scheduling service updates corridor risk metrics and triggers alerts in the live network view.
- **Vehicle telemetry**: GPS snapshots append to `truck_positions`; periodic rollups populate `active_routes` for the live map overlay described in `docs/live_network_overview.md`.
- **Data retention**: Movement events and inventory transitions are immutable; corrections are additive (new event rows) with references to the superseded `sequence_no`.

## Data model alignment

- **Jobs** remain the top-level grouping for routes, inventory, and media. Movement events use `job_id` to join against cost and profitability tables already consumed by the dashboard.
- **Inventory** attaches to jobs via `job_id` and `item_id`, with optional `asset_tag` to integrate warehouse systems. Media uses `movement_event_id` + `item_id` to bind PEC evidence to the same record.
- **Telemetry** provides live overlays but also backfills `lane_summary` inputs when paired with historical jobs, keeping profitability analytics consistent between live and batch views.

## Operational playbook

- Validate ingest schemas in lower environments before enabling new payload types; incompatible changes must bump the `version` envelope.
- Ship dashboards with clear badges when reconciliation detects drifts or missing media so crews know which jobs require re-scans.
- Keep nightly reconciliation timing close to depot close-of-business to minimise false positives while still catching next-day departures.
- When adding new movement events, document the state transition and routing context so downstream analytics can maintain referential integrity.
