## 2026-03-04

Source of truth update based on current repository state and roadmap review.

Current status summary:
- Core routing + costing are stable.
- Analytics and Streamlit dashboard are partially implemented.
- Major blockers: historical job ingestion, corridor/lane data model, full dashboard wiring.

High-leverage next features:
- Historical import pipeline.
- Corridor/lane detection + rollups.
- $/m³ benchmarking overlays.
- Quote recommendation engine.
- Backhaul detection.
- Profitability scoring.
- Corridor heatmap layer.
- Automated corridor pricing adjustments.

Docs updated:
- README.md and ROADMAP.md updated to reflect current status and priorities.

Implementation changes:
- None. Documentation-only updates.

## 2026-03-04 (Positioning)

Added a positioning and competitive landscape summary to align product focus as a pricing intelligence layer that integrates with incumbent systems rather than replacing them.

Docs updated:
- README.md
- ROADMAP.md
- docs/positioning.md

Implementation changes:
- None. Documentation-only updates.

## 2026-03-04 (Integration Staging Schema)

Added a minimal integration staging schema document to define required fields,
staging tables, and ingest flow for external system data.

Docs updated:
- README.md
- docs/integration_staging_schema.md

Implementation changes:
- None. Documentation-only updates.

## 2026-03-04 (Roadmap Status Recheck)

Reclassified roadmap items based on current code:
- Marked partial completion for historical import, $/m3 metrics, heatmaps, and dashboard wiring.
- Marked modifier tables and base-rate schedule as implemented.
- Updated blockers to focus on data validation, corridor formalization, quote benchmarking, backhaul detection, and end-to-end dashboard wiring.

Docs updated:
- ROADMAP.md

Implementation changes:
- None. Documentation-only updates.

## 2026-03-04 (Corridor Detection)

Added a corridor detection design doc covering baseline clustering, metrics,
and backhaul implications.

Docs updated:
- README.md
- ROADMAP.md
- docs/corridor_detection.md

Implementation changes:
- None. Documentation-only updates.

## 2026-03-04 (Directional + Grouped Corridors)

Updated corridor detection docs to define directional corridors grouped into
bidirectional corridor groups, with time-bucket stats.

Docs updated:
- docs/corridor_detection.md
- ROADMAP.md

Implementation changes:
- None. Documentation-only updates.

## 2026-03-04 (Corridor Schema Plan)

Added a corridor schema plan with proposed tables, metrics, and time buckets.

Docs updated:
- README.md
- ROADMAP.md
- docs/corridor_schema_plan.md

Implementation changes:
- None. Documentation-only updates.

## 2026-03-04 ($/m3 Distribution + Phantom Corridors)

Extended corridor docs to include $/m3 distribution buckets and phantom corridor
signals for opportunity detection.

Docs updated:
- docs/corridor_detection.md
- docs/corridor_schema_plan.md
- ROADMAP.md

Implementation changes:
- None. Documentation-only updates.

## 2026-03-04 (Corridor Decisions + Break-Even)

Added recommended decisions (manual clusters + geohash fallback, threshold=6,
denormalized corridor keys), break-even overlay guidance, phantom corridor
scoring, and gravity model note.

Docs updated:
- docs/corridor_detection.md
- docs/corridor_schema_plan.md
- ROADMAP.md

Implementation changes:
- None. Documentation-only updates.

## 2026-03-04 (Corridor Defaults + AU Clusters)

Added default thresholds, buckets, and break-even constants, plus an AU manual
cluster template.

Docs updated:
- README.md
- ROADMAP.md
- docs/corridor_defaults.md
- docs/cluster_template_au.md

Implementation changes:
- None. Documentation-only updates.

## 2026-03-04 (Gravity + Opportunity Scoring)

Added gravity model formula defaults, opportunity scoring, and geohash corridor
automation notes.

Docs updated:
- docs/corridor_detection.md
- docs/corridor_schema_plan.md
- docs/corridor_defaults.md
- ROADMAP.md

Implementation changes:
- None. Documentation-only updates.

## 2026-03-04 (Corridor Opportunity Report)

Added a report spec for corridor opportunity ranking.

Docs updated:
- README.md
- ROADMAP.md
- docs/corridor_opportunity_report.md

Implementation changes:
- None. Documentation-only updates.

## 2026-03-05 (Corridor Opportunity View)

Added a docs-only SQL view definition for the corridor opportunity report.

Docs updated:
- README.md
- ROADMAP.md
- docs/corridor_opportunity_view.md

Implementation changes:
- None. Documentation-only updates.

## 2026-03-11 (AMS Backend Docs Context Sync)

Resolved archived thread metadata and used it to refresh backend documentation
intent before making doc/TODO updates.

Resolved thread:
- title: Kent Removals AMS Info
- online UUID: 69b137e9-862c-83a0-95ff-90b12cdb7751
- canonical thread ID: 3a4e2a1bf2bd1571afb73c915a4edf99953ad3a7
- source used: db (local canonical archive at `~/chat_archive.sqlite`)

Main topics pulled from the thread:
- Backend documentation structure that reduces onboarding time.
- Typical relocation backend entity model (`assignees`, `moves`, `shipments`, `services`, `vendors`, `inventory_items`, `documents`, `tasks`).
- Reverse-engineering checklist for undocumented systems.
- Importance of documenting lifecycle state transitions and pricing-engine inputs.

Docs updated:
- README.md
- ROADMAP.md
- docs/ams_backend_docs_playbook.md

Implementation changes:
- None. Documentation/TODO alignment only.

## 2026-03-11 (Kent AMS Interface Spec)

Added an explicit Corkysoft-to-Kent AMS integration document so the interface
boundary and payload expectations are not implied.

Docs updated:
- README.md
- ROADMAP.md
- docs/kent_ams_integration.md

Implementation changes:
- None. Documentation-only update.

## 2026-03-11 (Kent AMS Mapping + Roadmap)

Expanded Kent integration documentation with an explicit field mapping table
from expected Kent AMS entities to current Corkysoft schema fields and added a
phased integration roadmap with milestones and acceptance gates.

Docs updated:
- README.md
- ROADMAP.md
- docs/kent_ams_integration.md
- docs/kent_ams_integration_roadmap.md

Implementation changes:
- None. Documentation-only update.

## 2026-03-12 (Kent AMS Importer Scaffold)

Brought Kent AMS integration interface closer to MoveWare by adding an internal
Kent importer endpoint and resource dispatcher for adapter-fed payloads.

Code updated:
- corkysoft/api.py
- analytics/kent_ams_import.py
- tests/test_api.py

Docs updated:
- docs/kent_ams_integration.md

Implementation changes:
- Added `POST /importers/kent-ams/{resource}` using the same request/summary
  shape as MoveWare importer.
- Added Kent resources: `jobs`, `subcontractors`/`vendors`, `bids`, `awards`.
- Added Kent bid/award persistence tables created on demand by importer.

## 2026-03-12 (Kent Tender Pre-Scoring)

Implemented tender pre-scoring so incoming Kent AMS tenders are ranked for
operator focus based on profitability, urgency, seasonality, and capacity fit.

Code updated:
- analytics/kent_ams_import.py
- corkysoft/api.py
- tests/test_api.py

Docs updated:
- docs/kent_ams_integration.md
- ROADMAP.md

Implementation changes:
- Added Kent `tenders` import resource on `POST /importers/kent-ams/{resource}`.
- Added tender storage table `kent_job_tenders` with persisted score components.
- Added read endpoint `GET /kent-ams/tenders/prioritized` for ranked tender queues.
- Added score outputs and recommended action labels (`pursue_now`, `review_today`,
  `review_if_capacity`, `defer`).

## 2026-03-12 (Kent Tender Calibration Metrics)

Implemented calibration metrics to quantify whether tender scores predict win
rate and realized margin outcomes.

Code updated:
- analytics/kent_ams_import.py
- corkysoft/api.py
- tests/test_api.py

Docs updated:
- docs/kent_ams_integration.md
- ROADMAP.md

Implementation changes:
- Added `GET /kent-ams/tenders/calibration?lookback_days=<n>` endpoint.
- Added score-band metrics: tender count, wins, win rate, predicted vs realized
  margin, and mean absolute margin error.
- Added documented recommendations for weight tuning with peak-season guardrails.

## 2026-03-12 (Route/Location Scoring Added)

Expanded tender scoring and docs to include route/location fit, covering lane
familiarity and historical lane margin effects.

Code updated:
- analytics/kent_ams_import.py
- corkysoft/api.py
- tests/test_api.py

Docs updated:
- docs/kent_ams_integration.md
- ROADMAP.md

## 2026-03-12 (En-Route Spare Capacity Signal Integration)

Added a shared operational signal module and integrated it into:
- Kent tender scoring
- quote creation summaries/persistence
- MoveWare and Kent job ingest paths

Code updated:
- analytics/operational_signals.py
- analytics/kent_ams_import.py
- analytics/moveware_import.py
- corkysoft/quote_service.py
- corkysoft/repo.py
- corkysoft/api.py
- tests/test_api.py

Docs updated:
- docs/kent_ams_integration.md

## 2026-03-12 (Multi-Truck Optimization Spec)

Documented the multi-truck pickup/drop sequencing and transfer optimization
problem explicitly, including the "many jobs per truck / many trucks per job"
model and phased optimization approach.

Docs updated:
- README.md
- ROADMAP.md
- docs/multi_truck_route_load_optimization.md

Implementation changes:
- None. Documentation-only update.

## 2026-03-12 (Kent AMS Policy Workflow)

Implemented the operator-facing Kent tender policy workflow and aligned code,
API, dashboard, and docs around the same behavior.

Code updated:
- analytics/db/parameters.py
- analytics/kent_ams_import.py
- corkysoft/api.py
- dashboard/app.py
- tests/test_api.py

Docs updated:
- README.md
- ROADMAP.md
- docs/kent_ams_integration.md

Implementation changes:
- Added profitability rule-mode defaults in `global_parameters`.
- Changed tender queue ordering to prioritize policy matches while keeping
  non-matching tenders visible with fail reasons and loss alerts.
- Added seeded/admin-managed override reason codes and tender override audit
  logging.
- Added API endpoints for Kent policy config, reason-code management, override
  creation, and override history.
- Added a Streamlit `Kent tenders` tab for queue review, config updates, reason
  management, and override capture/history.

## 2026-03-12 (Kent Validation + Shared Policy Signals)

Completed three follow-up actions after the policy workflow landed:

- ran the Kent API suite successfully under the existing project `venv`
- added fixture-backed Kent tender payload validation for local smoke coverage
- pushed profitability policy semantics into quote creation and job ingest
  operational signals so pricing/dispatch share the same pass/fail/loss model

## 2026-03-12 (Docs/Governance/Kent Reliability Remediation)

Executed the remediation milestone from the repo audit to align project truth,
complete missing product/governance docs, and harden brittle Kent behavior.

Planning state added:
- spec.md
- plan.md
- status.json
- devlog.md

Docs aligned:
- README.md
- ROADMAP.md
- docs/positioning.md
- docs/kent_ams_integration.md
- docs/kent_ams_integration_roadmap.md
- docs/live_network_overview.md
- docs/multi_truck_route_load_optimization.md
- docs/ingest_inventory_logistics.md
- docs/fleet_tables.md
- docs/heatmap_logic.md
- docs/operator_user_stories.md
- docs/commercial_workflow_lifecycle.md

Implementation changes:
- added internal mutating API auth via `X-Corkysoft-Api-Key` validated against
  `CORKYSOFT_API_TOKEN`
- made API `dry_run` execution side-effect free by running imports against an
  in-memory shadow database
- fixed Kent prioritized queue correctness so top-N is selected after final
  ranking rather than before it
- restricted hard-block handling to governed safety/legal/compliance categories
- split Streamlit Kent operator workflow from Kent admin/config workflow
- added UX handling for the "no active override reasons" case
- relabeled quote workflow output as a quote-policy preview rather than silent
  equivalence with tender policy

Validation:
- `py_compile` passed on touched Python files
- targeted tests passed under local `venv`:
  `tests/test_dashboard_app.py`, `tests/test_api.py`,
  `tests/test_kent_ams_fixtures.py`

## 2026-03-12 (Crusader Workbook Fleet Import Hardening)

Closed a local-data reliability gap around the `Crusader.xlsx` placeholder
workbook so it can actually seed fleet state used by pricing and Kent triage.

Implementation changes:
- fixed Fleet tab uploaded-XLSX handling to use the workbook-aware importer
  instead of blindly reading only the first worksheet
- fixed `analytics.vehicle_workbook` so sheet-per-vehicle workbooks without an
  explicit `REGO` column inherit the sheet name as the truck identifier using a
  canonical column path
- added regression coverage against the real repo workbook

Code updated:
- analytics/vehicle_workbook.py
- dashboard/components/maintenance.py
- tests/test_vehicle_workbook.py

Validation:
- targeted local `venv` tests passed:
  `tests/test_vehicle_workbook.py`, `tests/test_dashboard_app.py`,
  `tests/test_api.py`, `tests/test_kent_ams_fixtures.py`

## 2026-03-12 (Google Sheets-First Operations Workbook Migration Start)

Started moving operational data refresh away from the local `Crusader.xlsx`
placeholder and toward the live Google Sheets setup already used by the
business. The private sheet URLs were used for inspection only and were not
recorded in repo docs.

Observed live workbook shape:
- one shared operations workbook contains fleet/staff/supplier tabs
- a separate workbook contains sheet-per-vehicle maintenance history plus
  repairs history/index tabs

Docs updated:
- README.md
- ROADMAP.md

Implementation changes:
- added shared Google Sheets ID/URL resolution helpers
- added Google Sheets staff import using the shared operations workbook
- added supplier import UI for Google Sheets-backed `SUPPLIERS`
- updated fleet import UI defaults to prefer the shared operations workbook env
- kept local `.xlsx` upload paths as fallback rather than the primary workflow

Code updated:
- analytics/google_sheets.py
- analytics/db/fleet.py
- analytics/db/inventory.py
- analytics/driver_shifts.py
- analytics/vehicle_workbook.py
- dashboard/app.py
- dashboard/components/maintenance.py
- tests/test_google_sheets_imports.py

Validation:
- targeted local `venv` tests passed:
  `tests/test_google_sheets_imports.py`, `tests/test_vehicle_workbook.py`,
  `tests/test_dashboard_app.py`, `tests/test_api.py`,
  `tests/test_kent_ams_fixtures.py`

## 2026-03-12 (Shared Operations Workbook Sync)

Implemented a coordinated sync path for the shared operations workbook so fleet,
staff, and suppliers can be refreshed together from one workbook reference.

Docs updated:
- README.md
- ROADMAP.md

Implementation changes:
- added `analytics.operations_workbook.sync_operations_workbook`
- added a Fleet-tab dashboard action to sync the shared operations workbook in
  one step
- added optional env-based per-tab overrides for shared-workbook `STAFF` and
  `SUPPLIERS` sheet names while keeping defaults simple

Code updated:
- analytics/operations_workbook.py
- dashboard/components/maintenance.py
- tests/test_operations_workbook.py

Validation:
- targeted local `venv` tests passed:
  `tests/test_operations_workbook.py`, `tests/test_google_sheets_imports.py`,
  `tests/test_vehicle_workbook.py`, `tests/test_dashboard_app.py`,
  `tests/test_api.py`, `tests/test_kent_ams_fixtures.py`

## 2026-03-12 (Spreadsheet-First Operations Planning)

Implemented the next integration layer for spreadsheet cooperation so
Corkysoft can plan trucks/workers/jobs internally while continuing to ingest
Google Sheets as operational inputs.

Decisions locked:
- spreadsheets are import-only for now; no write-back
- Corkysoft internal state is the planning truth
- `job_segments` are the canonical assignment unit
- maintenance/rego/COI/compliance readiness is evaluated as part of assignment,
  not as a detached dashboard-only concern

Docs updated:
- README.md
- ROADMAP.md
- docs/fleet_tables.md
- docs/commercial_workflow_lifecycle.md

Implementation changes:
- added source provenance fields for workers, suppliers, and vehicle details
- added `analytics.operations_assignment` for policy, readiness, assignment,
  conflicts, and segment bootstrap/update behavior
- added `/operations/*` API endpoints for policy, sync, segment creation,
  readiness listing, assignment, and conflicts
- added dashboard `Operations` tab for segment planning and assignment
- added readiness policy controls in Fleet/admin workflow
- updated shipments wrapper to use the richer legacy segment-aware shipment path
- fixed `ensure_segment` so backfilled default segments can actually be updated
  with planned windows, which restores overlap/conflict detection
- normalized vehicle workbook provenance timestamps to UTC

Tests added/updated:
- tests/test_operations_assignment.py
- tests/test_api.py
- tests/test_dashboard_app.py
- tests/test_google_sheets_imports.py
- tests/test_vehicle_workbook.py

Validation:
- targeted local `venv` tests passed:
  `tests/test_operations_assignment.py`, `tests/test_google_sheets_imports.py`,
  `tests/test_vehicle_workbook.py`, `tests/test_dashboard_app.py`,
  `tests/test_api.py`, `tests/test_kent_ams_fixtures.py`,
  `tests/test_operations_workbook.py`

Follow-up implementation completed the next workflow step:
- Staff tab now shows planned segment assignments, planned trucks/jobs, and next
  planned work separately from imported sheet truck context and recent shifts
- Fleet tab now shows planned segment/job/worker context for each truck and a
  per-truck planned segment detail view
- added `docs/spreadsheet_replacement_plan.md` to define full spreadsheet
  replacement by workflow rather than by raw table

Validation:
- local `venv` tests passed after Staff/Fleet integration updates:
  `tests/test_operations_assignment.py`, `tests/test_dashboard_app.py`,
  `tests/test_api.py`, `tests/test_google_sheets_imports.py`,
  `tests/test_vehicle_workbook.py`, `tests/test_operations_workbook.py`
