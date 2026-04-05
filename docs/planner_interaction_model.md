# Planner Interaction Model

This document defines the intended operator workflow for turning route, corridor,
site, and resource context into a draft operational plan.

It sits above the internal `job_segments` model.
`job_segments` remain the planning truth in storage and downstream execution, but
operators should not normally hand-author them through a low-level record form.

## Goal

Provide a click-heavy planning surface that lets operators:

- select roadway, corridor, and site extents visually
- surface historical route overlap and lane familiarity
- see route-level and segment-level profitability context during planning
- factor in traffic/routing/site/resource considerations before assignment
- confirm or adjust a proposed operational plan before it becomes the internal
  `job_segments` representation

## Core Planning Principles

1. Plan visually first, confirm structurally second.
2. Treat `job_segments` as an internal artifact, not the primary operator UX.
3. Keep profitability and route familiarity visible during planning, not only in
   downstream analytics.
4. Surface operational constraints before assignment, not after.
5. Preserve explainability so operators can understand why a draft plan was
   proposed.

## Intended Workflow

### 1. Visual Selection

The planner starts from visual interaction rather than freeform data entry.

Primary interactions:
- select roadway extents
- select corridor or region bands
- inspect site location context around origins, destinations, depots, or
  transfer nodes

Selection outcomes should identify:
- the route or route band under consideration
- nearby historical moves or route fragments that materially overlap
- known site-specific constraints that could change planning outcomes

### 2. Historical Overlap and Corridor Context

Once the user selects a roadway or corridor region, the planner should surface:

- historical route overlap candidates
- corridor familiarity signals
- route/lane profitability context
- nearby or adjacent corridor alternatives when the current selection is weakly
  supported

The key idea is not just “show a map”, but “show why this route shape is or is
not commercially and operationally familiar”.

### 3. Draft Operational Leg Generation

The planner should then propose draft operational legs.

Draft legs may represent:
- pickup
- staging/storage
- linehaul
- delivery
- transfer/cross-dock handoff where allowed

These draft legs become the precursor to internal `job_segments`.
The operator should confirm, split, merge, reorder, or reject them through GUI
interactions rather than manual sequence/data entry as the normal path.

### 4. Profitability and Routing Context During Planning

The planner must surface commercial and routing context before assignment.

Required planning-time signals:
- per-route profitability context
- per-segment profitability implications where inferable
- corridor/lane familiarity
- traffic and routing considerations
- site location constraints
- spare-capacity or resource-fit signals when known

These signals are planning inputs, not merely post-hoc analytics.

### 5. Resource Allocation Handoff

After the operator confirms the draft leg structure, the planner hands off into
resource allocation.

That handoff should support:
- truck assignment
- worker assignment
- readiness evaluation
- stock/supplier coordination where segment-linked inventory is required

The assignment model remains segment-based internally, but it should inherit the
context created in the visual planning step.

## Explainability Requirements

Any planner-generated proposal should expose at least:

- why these legs were proposed
- which historical overlaps influenced the suggestion
- what route/corridor profitability context was used
- which site/routing/resource considerations were relevant
- what remains uncertain and may still require operator judgment

## Manual Editing Policy

Manual segment editing remains valid only as:
- an advanced/admin fallback
- a debugging/data-repair tool
- an exception path when the planner cannot model a case well enough yet

It should not be treated as the desired daily planning workflow.

## Relationship To Existing Docs

This document complements, rather than replaces:

- [Spreadsheet Replacement Plan](spreadsheet_replacement_plan.md)
- [Quote to Award Lifecycle](commercial_workflow_lifecycle.md)
- [Multi-Truck Route and Load Optimization](multi_truck_route_load_optimization.md)
- [Corridor Detection](corridor_detection.md)
- [Kent AMS Integration Spec](kent_ams_integration.md)

Those docs define the planning truth, commercial signals, and optimization
constraints. This document defines the missing interaction model that sits above
those concepts.


## First Implementation Slice

The first implementation should ship as a scaffold rather than a full solver.

It should include:
- a dedicated `Planner` workflow inside the `Operations` shell
- lightweight role-based layout defaults that can make `Operations` the landing shell for the relevant roles while keeping `Planner` as the primary planning workflow
- corridor/roadway-driven candidate surfacing using existing route/profitability history
- draft leg proposal generation that becomes internal `job_segments` only after confirmation
- the current manual segment editor retained as advanced/manual fallback in `Operations`

Current status:
- the dedicated `Planner` workflow inside `Operations` and the supporting role-based layout defaults are implemented
- current planner behavior supports both job-first and corridor-first planning over historical route/profitability context
- current planner should expose a day view anchored on the selected move date so proposed legs can be compared against already-planned segments, trucks, and workers for that day
- planner day view should link into the manager-facing `Operations diary` workflow once the operator needs job usage, vehicle usage, staff usage, invoice state, or subcontractor-bill review rather than only route/leg shaping
- current proposals surface routing context and resource-fit alongside corridor familiarity before confirmation
- current proposals also surface weak-confidence warnings before confirmation into `job_segments`
- current planner site context includes first-pass street-level imagery when Google Maps is the active provider and imagery is available
- current planner now stores and surfaces accepted site-risk assessments, linked site media, reviewed advisory media outputs (for example site-feature or volume-estimate scaffolding), and first derived planning constraints against jobs
- current saved-route overlay parity is aligned with the active provider for Google/ORS tile selection
- deeper interpreted site constraints, walkaround ingestion workflows, and actual model-backed media inference remain the next expansion step

## Next Expansion Thoughts

### 1. Richer interpreted site constraints

The next useful planner step is not more imagery. It is converting accepted
site evidence into operationally meaningful constraints that change planning
behavior.

That means Planner should move from:
- showing imagery and accepted risk labels

to:
- surfacing explicit planning consequences such as:
  - large-truck unsuitability
  - likely shuttle / smaller-vehicle requirement
  - likely labor uplift
  - likely access-time uplift
  - likely parking / loading delay risk
  - likely claims-risk / manual-handling uplift

The important distinction is:
- imagery is evidence
- site assessments are interpreted operator truth
- interpreted site constraints are the planning consequences consumed by quote,
  planner, dispatch, and later optimization

This should remain additive and reviewable.
The system should not silently derive hard operational truth from raw imagery
without a human-reviewed assessment path.

Priority note:
- this is the correct planner direction, but it is probably not the immediate
  next implementation priority
- the current heuristic slice is enough to prove the shape of the planner
  contract
- deeper interpreted-constraint logic should wait until the media/CV side is
  more real, because otherwise the planner ends up over-investing in derived
  logic on top of placeholder evidence
- in practice, richer constraint derivation should advance in step with
  production-grade walkaround ingestion, reviewed site-feature extraction, and
  reviewed volume estimation

In other words:
- accepted imagery/assessment -> first heuristic constraints is already useful
- deeper constraint policy should follow real media evidence quality, not run
  far ahead of it

### 2. Replace advisory scaffolding with real media/CV services

The current advisory/manual inference scaffolding is the right bridge, but it is
not the desired end state.

The replacement path should be:
1. ingest walkaround video/image media cleanly
2. run service-backed CV/object-detection / volume-estimation jobs
3. store raw model outputs as advisory artifacts
4. require operator acceptance/correction for high-impact outputs
5. expose accepted outputs in Planner/Quote/Dispatch as operational context

This matters because:
- raw model outputs are noisy
- last-mile constraints are commercially and operationally expensive when wrong
- accepted/corrected outputs need a stable audit trail

In practice, the first production-grade model-backed outputs should probably be:
- site-feature extraction
  - stairs
  - narrow access
  - clearance concern
  - driveway/loading-zone concern
- volume estimate

Object detection should support those outcomes, but the planner should consume
reviewed feature/volume results rather than raw bounding-box lists.
Those reviewed outputs should attach to quote/job records as durable evidence
and only become planner-consumable constraints after explicit acceptance or
correction.

Priority note:
- this is the more important next milestone
- once walkaround/media ingestion and reviewed model outputs are real, the next
  round of interpreted site-constraint work becomes worth doing in earnest
- until then, the current heuristic constraint layer should be treated as a
  bridge rather than a large standalone optimization project

It should not yet require:
- bespoke planner-only auth or identity logic beyond the dashboard auth shell
- a full spatial freehand editing system
- exact global optimization
- removal of the existing `Operations` assignment workflow
