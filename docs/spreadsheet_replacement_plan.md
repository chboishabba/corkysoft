# Spreadsheet Replacement Plan

This document defines the path from spreadsheet-assisted operations to
Corkysoft-native operational workflows.

## Goal

Replace spreadsheets as day-to-day operating surfaces without losing the speed,
flexibility, or tacit operational context that made them useful.

End state:

- Corkysoft owns operational planning, assignments, readiness, and audit
- Google Sheets become optional import/export bridges only
- truck, worker, maintenance, and job/segment workflows cooperate in one system

## Current State

Spreadsheets still provide live operational input for:

- `FLEET`
- `STAFF`
- `SUPPLIERS`
- `VEHICLE_DRIVER`
- vehicle maintenance / repair history

Corkysoft now owns:

- segment-based planning state
- truck and worker assignment state
- readiness/blocking evaluation
- override-aware assignment workflow

## Replacement Principles

1. Replace by workflow, not by table.
2. Keep imports until the native workflow is faster than the sheet.
3. Separate imported context from internal planning truth at every step.
4. Never silently overwrite internal assignments from inbound sheets.
5. Remove spreadsheet dependence only after operator validation and fallback drills.

## Phases

### Phase 1: Assignment Truth

Objective:
- make Corkysoft the default place to assign trucks and workers to jobs

Deliverables:
- segment planning surface
- staff/fleet views that show planned assignments
- conflict detection
- readiness checks

Exit criteria:
- operators can plan a day’s work without relying on sheet-side assignment columns

### Phase 2: Maintenance and Compliance Cockpit

Objective:
- move rego/COI/service/compliance monitoring out of spreadsheet-first workflows

Deliverables:
- due-soon and blocked-resource views
- maintenance scheduling actions
- compliance assignment management
- policy-controlled warning/block windows

Exit criteria:
- spreadsheets no longer needed to determine whether a truck/worker is assignable

### Phase 3: Native Driver and Labor Planning

Objective:
- replace `VEHICLE_DRIVER` as the active roster surface

Deliverables:
- native shift/availability planning
- worker-to-segment scheduling
- reconciliation against imported historical shift data
- variance detection between planned work and recorded shifts

Exit criteria:
- daily driver/worker planning happens in Corkysoft first

### Phase 4: Inventory and Supplier Coordination

Objective:
- integrate supplier and stock workflows into planned jobs/segments

Deliverables:
- segment-linked stock allocation
- supplier-linked maintenance/inventory flows
- exception handling and reconciliation inside Corkysoft

Exit criteria:
- operational stock and supplier workflows no longer require sheet-side tracking

### Phase 5: Controlled Spreadsheet Decommissioning

Objective:
- reduce sheets from primary system to fallback bridge

Deliverables:
- per-workflow cutover checklist
- read-only import fallback mode
- optional export snapshots for stakeholders still outside the app
- operator training and rollback instructions

Exit criteria:
- daily operations continue if spreadsheet imports are paused for a day

## Priority Order

1. Truck/worker assignment and dispatch
2. Maintenance/rego/COI/compliance readiness
3. Driver/worker shift planning
4. Inventory/supplier coordination
5. Sheet decommissioning and optional exports

## Main Risks

- operators still trust sheet columns more than native planning views
- spreadsheet edge cases contain tacit rules not yet modeled
- maintenance/compliance data quality may be too weak for hard-block use
- dual-running can create drift if reconciliation views are weak

## Required Near-Term Work

- add job-centric dispatch views built directly on `job_segments`
- replace `present_driver` workflow dependence with native assignment/reconciliation
- add maintenance/compliance cockpit and action flows
- add native driver planning surface before deprecating `VEHICLE_DRIVER`
- define cutover metrics for each spreadsheet-backed workflow
