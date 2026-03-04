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
