# Integration Staging Schema

This document defines the minimal staging schema Corkysoft uses to ingest data
from external systems (CRM, dispatch, accounting, fleet tracking) before it is
normalized into the core analytics tables.

## Goals

- Accept CSV and API payloads with the smallest viable field set.
- Preserve source identifiers for traceability and idempotent updates.
- Normalize into core job, cost, and lane analytics without forcing full CRM data.

## Data Flow

1. Ingest CSV/API payloads into staging tables.
2. Validate required fields and coerce types.
3. Normalize addresses and resolve routing data.
4. Merge into core tables (`jobs`, cost ledger, corridor rollups).
5. Capture ingest outcomes for audit and backfills.

## Required Fields (Minimum)

These are the minimum inputs required to power pricing intelligence.

- `job_date`
- `origin`
- `destination`
- `volume_m3`
- `quoted_price`
- `crew_cost`
- `truck_cost`

## Optional Fields (Recommended)

- `distance_km`
- `duration_hr`
- `client_name`
- `job_status`
- `crew_count`
- `truck_id`
- `source_system`

## Staging Tables

### `staging_jobs`

Purpose: capture raw job records from external sources.

Fields:
- `staging_job_id` (pk)
- `source_system` (text)
- `source_job_id` (text)
- `job_date` (date)
- `origin` (text)
- `destination` (text)
- `volume_m3` (real)
- `quoted_price` (real)
- `crew_cost` (real)
- `truck_cost` (real)
- `distance_km` (real, nullable)
- `duration_hr` (real, nullable)
- `client_name` (text, nullable)
- `job_status` (text, nullable)
- `ingested_at` (timestamp)

### `staging_costs`

Purpose: capture granular cost line items when available.

Fields:
- `staging_cost_id` (pk)
- `source_system` (text)
- `source_job_id` (text)
- `cost_type` (text)
- `cost_amount` (real)
- `cost_notes` (text, nullable)
- `ingested_at` (timestamp)

### `staging_ingest_runs`

Purpose: audit and retry ingest operations.

Fields:
- `ingest_run_id` (pk)
- `source_system` (text)
- `source_file` (text, nullable)
- `row_count` (integer)
- `success_count` (integer)
- `error_count` (integer)
- `started_at` (timestamp)
- `finished_at` (timestamp)
- `notes` (text, nullable)

## Idempotency and Matching

- Use `(source_system, source_job_id)` as the unique match key.
- Re-ingests should upsert on the same key rather than create duplicates.
- Failed rows should be captured with error notes in `staging_ingest_runs`.

## Normalization Targets

After validation and enrichment, data should populate:
- `jobs` (core job table with normalized addresses).
- Cost ledger tables in `analytics/db/`.
- Corridor rollups for pricing and benchmarking.

## Notes

- This staging schema is intentionally small to keep integration friction low.
- A future extension can add `staging_clients` if CRM data needs to be stored.
