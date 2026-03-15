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
- planner UX that derives draft operational legs from route/corridor/site context instead of expecting manual segment authoring as the normal operator flow
- dedicated `Planner` tab as the primary planning surface
- existing `Operations` segment form retained as advanced/manual fallback

Exit criteria:
- operators can plan a day’s work without relying on sheet-side assignment columns

Planner direction to lock before deeper implementation:
- canonical interaction spec: [Planner Interaction Model](planner_interaction_model.md)
- operators should not normally hand-author `job_segments` through a raw form
- the planning surface should be map/corridor/site-first
- roadway extents, site context, and historical overlap should suggest candidate legs
- route, traffic, and resource-fit considerations should shape the draft plan before assignment
- manual segment editing should remain an advanced/admin fallback only

### Phase 2: Maintenance and Compliance Cockpit

Objective:
- move rego/COI/service/compliance monitoring out of spreadsheet-first workflows

Deliverables:
- due-soon and blocked-resource views
- maintenance scheduling actions
- compliance assignment management
- policy-controlled warning/block windows

Current implementation:
- due-soon and blocked-resource cockpit exists in Fleet
- worker role/compliance assignment management exists in Staff
- policy-controlled warning/block windows exist for rego, COI, service, and worker compliance

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

Current implementation:
- `job_segments` assignments drive a native planned labor roster
- driver shifts tab shows reconciliation between planned labor and imported `VEHICLE_DRIVER` rows
- imported shifts remain for audit/history rather than primary planning

Exit criteria:
- daily driver/worker planning happens in Corkysoft first

### Phase 4: Inventory and Supplier Coordination

Objective:
- integrate supplier and stock workflows into planned jobs/segments

Deliverables:
- segment-linked stock allocation
- supplier-linked maintenance/inventory flows
- exception handling and reconciliation inside Corkysoft

Current implementation:
- Inventory tab exposes segment-linked stock and supplier coordination
- inventory can be allocated directly to planned job segments
- supplier context flows through segment-linked inventory shipments

Current implementation detail:
- per-job / per-segment inventory requirement lines now exist
- required, allocated, and shortage quantities are tracked and surfaced
- non-substitutable shortages block readiness; substitutable shortages create explicit override-required flags
- Dispatch and Planner now surface shortage state before work is confirmed
- custody/location truth now supports depot, truck, container, in transit, site, returned/storage, and exception contexts
- container-heavy operations are supported as a first-class inventory architecture alongside consumables, reusable assets, serialized/tagged gear, job-specific lines, and general stock

Next milestone:
- deepen this from planning truth into richer warehouse execution and substitution workflow support
- validate the requirement/custody model against real container-heavy operating patterns
- use [Inventory Execution Workflow](inventory_execution_workflow.md) as the canonical warehouse-facing workflow spec

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

Current implementation:
- Dispatch tab provides a native job-centric execution board across segments,
  trucks, workers, stock, suppliers, and readiness flags
- dispatch snapshot CSV export exists for external stakeholders who still need a
  lightweight operational extract
- spreadsheet-backed imports remain available as fallback inputs rather than the
  primary execution surface
- Fleet admin tracks per-workflow cutover status, fallback mode, checklist
  completion, snapshot requirements, and rollback instructions
- Fleet admin also tracks rollout metrics per workflow: native usage %, target
  %, fallback-use count, open issues, snapshot consumers, and review timestamp
- review, fallback-drill, fallback-use, and snapshot-issued events are logged so
  rollout metrics come from actual operational activity where possible
- Fleet admin surfaces guarded recommended transitions so workflows can move
  from `dual_run` -> `native_primary` -> `fallback_only` only when the current
  gates and derived metrics justify it
- rollout promotions now follow an explicit approval chain:
  - operations manager requests promotion
  - commercial owner approves or rejects it
  - admin applies the status transition only after approval is present
- rollout execution behavior is defined in
  [Rollout Execution Stories](rollout_execution_user_stories.md)

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

- finish cutover checklists and fallback drills for each spreadsheet-backed workflow
- validate which teams still need CSV snapshots and keep only the required field sets stable
- run rollback drills for import outages or cutover regressions and keep the event log current in the cutover admin surface
- keep cutover metrics current for each spreadsheet-backed workflow and use
  them to decide when a sheet can move from dual-run to fallback-only
- keep review, request, approval, rejection, drill, fallback-use, and
  transition history current so rollout evidence stays trustworthy
