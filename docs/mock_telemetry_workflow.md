# Mock Telemetry Workflow

## Overview
The simulated GPS pipeline mirrors the same structure the real harness expects: a generator feeds truck positions into SQLite, the ingest harness applies metadata and timing logic, and the dashboard layers those tables onto the map. The sections below summarise the key responsibilities.

## 1. Route selection & telemetry synthesis
1. **Seed candidate routes** – `_pick_candidate_routes` queries `historical_jobs` joined to geocoded addresses so only routes with coordinates are eligible. When historical data is absent it falls back to depot waypoints to keep the simulation running.【F:analytics/live_data.py†L205-L268】
2. **Cycle through trucks** – `MockTelemetryIngestor.run_cycle` assigns each configured truck to a route, reuses prior assignments when possible, and introduces jitter so trucks are staggered in time.【F:analytics/live_data.py†L480-L556】
3. **Derive positions** – For routes with stored polylines, `_route_points_from_geojson` and `_position_along_route` interpolate the latitude/longitude and heading. Otherwise the code linearly interpolates between origin and destination while also synthesising reasonable speed and ETA values.【F:analytics/live_data.py†L305-L341】【F:analytics/live_data.py†L566-L595】
4. **Persist mock rows** – The ingestor writes to `truck_positions` and `active_routes`, keeping previous geometry when new runs lack it and cleaning up completed routes for reassignment on the next iteration.【F:analytics/live_data.py†L597-L685】

## 2. Parsing real (or replayed) telemetry
1. **Snapshot schema** – `TruckGpsSnapshot` models a single reading from external feeds, capturing the coordinates, job association, optional polyline, and timing metadata the harness can use.【F:analytics/live_data.py†L687-L708】
2. **Metadata enrichment** – `TruckTelemetryHarness.ingest` back-fills missing context by querying the historical job catalogue, including origin/destination coordinates, route geometry, and planned travel time.【F:analytics/live_data.py†L757-L779】
3. **Progress estimation** – If the snapshot omits progress, the harness either projects the point along stored route geometry or falls back to a straight-line haversine fraction, ensuring downstream timing calculations remain consistent.【F:analytics/live_data.py†L781-L795】
4. **Temporal fields** – Using the inferred progress and travel seconds, the harness normalises timestamps (`started_at`, `eta`) so the dashboard can display coherent timelines even when progress is derived on the fly.【F:analytics/live_data.py†L797-L808】
5. **Upserts into live tables** – The same tables as the mock ingestor are updated, keeping route geometry sticky and clamping progress between 0 and 1 before persistence.【F:analytics/live_data.py†L810-L899】

## 3. Visualisation in the dashboard
1. **Data loaders** – `load_truck_positions` and `load_active_routes` provide pandas DataFrames that power map layers and analytics panels.【F:analytics/live_data.py†L44-L104】
2. **Heatmap assembly** – `build_live_heatmap_source` merges historical lane endpoints, active route endpoints, and current truck points into a weighted point cloud so the UI can emphasise real-time activity.【F:analytics/live_data.py†L106-L198】
3. **Live loop helper** – `run_mock_ingestor` offers a CLI-friendly loop that keeps the simulation running, sleeping between iterations while committing new rows for the dashboard to pick up.【F:analytics/live_data.py†L902-L921】

Together these components let us plug either synthetic or real GPS feeds into the same storage contract, ensuring the Streamlit visualisations behave identically regardless of the upstream data source.
