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
