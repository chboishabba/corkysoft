# Inventory Execution Workflow

This document defines the operational inventory workflow that sits between
inventory planning and day-of-execution work. It is the canonical workflow spec
for pick / pack / load, shortage response, substitution, and container-heavy
execution.

Use this document as the source of truth for future UI, API, and readiness
changes.

## Purpose

Corkysoft already supports:

- per-job / per-segment inventory requirement lines
- required vs allocated vs shortage quantities
- readiness impact from shortages
- custody/location truth

The remaining gap is warehouse execution. This document defines how planned
inventory moves through real work and how operators respond when stock is short
or operationally different from the plan.

## Core Model

Inventory execution is a dual-layer model:

- planning demand stays in `m3` and requirement lines
- operational custody is container-first where applicable

That means:

- planners still reason about required stock, volume, and segment fit
- warehouse and crew users reason about what was picked, packed, loaded, and
  handed off
- container-heavy work uses containers as the primary execution/custody unit
- non-container workflows still work through item lines, assets, or consumables

## Roles

### Warehouse / Crew

Primary surfaces:

- `Inventory`

Responsibilities:

- record normal pick / pack / load transitions
- record custody handoffs
- flag shortages, substitutions needed, or execution exceptions

Normal authority:

- advance routine execution states
- request substitution
- record location/custody changes

Not allowed to:

- approve substitutions
- silently clear blocking shortages

### Dispatcher / Operations Manager

Primary surfaces:

- `Dispatch`
- `Inventory`

Responsibilities:

- decide whether work is still executable
- approve or reject substitution requests
- correct, escalate, or re-plan when execution diverges from inventory plan

Authority:

- approve substitutions
- override substitutable shortage paths
- escalate blocked work to manager review when necessary

### Inventory / Supplier Coordinator

Primary surfaces:

- `Inventory`

Responsibilities:

- maintain requirement fulfillment visibility
- coordinate stock, replenishment context, and supplier follow-up
- reconcile execution exceptions and late/missing stock

### Operations Manager

Primary surfaces:

- `Dispatch`
- `Inventory`
- `Fleet`

Responsibilities:

- review exception cases
- arbitrate unresolved warehouse/dispatch disagreement
- review repeated substitution or shortage drift

Normal role:

- exception owner, not routine transition owner

## Workflow Stages

The operational workflow below is the canonical sequence.

### 1. Required

Meaning:

- inventory demand is known for a segment or job
- requirement lines express what the work needs before execution begins

Recorded by:

- planner / inventory coordinator during planning

Effects:

- shortage can already be visible in Planner, Inventory, and Dispatch
- non-substitutable shortages block readiness
- substitutable shortages remain visible as override-required

### 2. Picked

Meaning:

- stock has been selected from depot/storage against a segment or container plan
- the system can now distinguish planned need from physically claimed stock

Recorded by:

- warehouse / crew

Expected evidence:

- item or container selected
- quantity or unit count confirmed
- custody set to depot pick area, truck staging area, or container preparation

Effects:

- requirement is moving toward execution
- unresolved shortage after pick remains visible

### 3. Packed

Meaning:

- stock is physically packed into a container, grouped for a leg, or prepared as
  a ready-to-load unit

Recorded by:

- warehouse / crew

Expected evidence:

- packed unit identified
- container or packed grouping known where relevant
- custody updated to the correct packed/staged context

Effects:

- for container-heavy jobs, this is where container identity becomes the main
  operational handle
- for generic stock, this may simply mean staged and physically ready

### 4. Loaded

Meaning:

- stock or container has been loaded onto the assigned truck or execution unit

Recorded by:

- warehouse / crew

Expected evidence:

- truck reference
- container reference where applicable
- loaded state recorded in inventory movement/custody

Effects:

- Dispatch should be able to see what is actually loaded, not just allocated
- any remaining shortage after load is an active operational exception

### 5. In Transit

Meaning:

- stock is no longer at depot/staging and is now travelling on the active leg

Recorded by:

- warehouse / crew or dispatch through normal operational event flow

Effects:

- custody transfers to truck / in-transit context
- item/container is no longer considered depot-available

### 6. Unloaded / Delivered

Meaning:

- stock has reached the destination site or the leg’s completion point

Recorded by:

- crew

Expected evidence:

- destination or site context
- delivered/unloaded confirmation

Effects:

- execution state for that leg is complete
- custody transfers from truck/in-transit to site/delivered context

### 7. Returned Storage

Meaning:

- stock or equipment returns to storage/depot rather than staying with the job

Recorded by:

- warehouse / crew

Effects:

- reusable assets and containers become available for future work

### 8. Exception

Meaning:

- normal execution flow broke

Examples:

- shortage remains unresolved
- wrong stock was loaded
- item/container is missing
- damaged or quarantined stock
- substitution request required

Recorded by:

- warehouse / crew initially
- dispatcher / operations manager may add escalation context

Effects:

- visible in Inventory and Dispatch
- may block the segment or require override/replanning depending on the cause

## Substitution Workflow

Substitution is an explicit governed path. It is not a silent data correction.

### When substitution is allowed

- the requirement line is marked substitutable
- the substitute preserves operational viability for the segment
- dispatch / operations can still deliver the committed service safely

### When substitution is not allowed

- the requirement line is non-substitutable
- the substitute would materially alter service, safety, handling, or customer
  expectations
- the operator cannot verify that the substitute is functionally acceptable

### Who can do what

- warehouse / crew:
  - request substitution
  - identify the proposed substitute
  - record why the planned stock is not available
- dispatcher / operations manager:
  - approve or reject substitution
  - own the final operational decision
- operations manager:
  - reviews escalations or repeated drift

### Required audit fields

Every substitution decision should log:

- job and segment
- original requirement
- requested substitute item or container
- quantity delta
- reason code
- free-text note
- request actor
- approval actor
- timestamp

### Readiness effect

- unresolved non-substitutable shortage remains blocking
- unresolved substitutable shortage remains override-required
- approved substitution clears the shortage for execution, but the segment still
  shows substitution history

### Dispatch effect

Dispatch must be able to tell:

- whether the segment is fulfilled as planned
- whether it is fulfilled by approved substitution
- whether it is still short and blocked / override-required

## Container-Heavy vs Generic Execution

### Container-Heavy Execution

This is the default high-priority model for Crusader-style work.

Rules:

- containers are the primary custody and movement unit
- requirement lines still express what the segment needs
- packed/loaded state is tracked against the container execution path
- custody handoff should be expressible as:
  - depot -> container
  - container -> truck
  - truck -> site
  - site -> returned storage or exception

Container-heavy flow should support:

- full container movement
- partial container fulfillment where only part of demand is loaded
- container plus loose stock in the same segment

### Generic Stock Execution

This covers:

- consumables
- reusable assets
- serialized/tagged equipment
- job-specific packed lines

Rules:

- execution may be item-led instead of container-led
- custody still uses the same location model
- pick / pack / load still applies even when no container object is involved

### Common Convergence

Both models must converge into the same operational truth for:

- readiness
- Dispatch visibility
- exception handling
- shortage handling
- substitution governance

## Custody Handoff Model

The custody/location model is operational, not just descriptive.

Supported custody contexts:

- depot
- truck
- container
- in_transit
- site
- returned_storage
- exception

Every meaningful handoff should be recordable as an additive movement/state
change. Corrections should never erase prior operational history.

## UI Ownership

### Inventory

Primary workflow surface for:

- pick / pack / load progression
- custody updates
- shortage review
- substitution request / approval path
- execution exceptions

Intended interaction model:

- select job
- select segment
- select requirement line or container
- perform explicit allowed next action

Normal warehouse work should use constrained actions rather than free-form stage
entry.

### Dispatch

Operational visibility surface for:

- whether a segment is ready, blocked, or override-required because of
  inventory execution state
- whether substitution was used
- whether actual loaded state diverges from plan

### Planner

Planning visibility surface only:

- inventory fit
- requirement risk
- shortage awareness before confirmation

Planner should not be the primary warehouse execution surface.

### Fleet / Admin

Not the primary execution surface.

Relevant only for:

- policy/governance settings
- role/permission configuration when those controls exist

## Required Scenarios

Future implementation must support at least:

1. container-heavy segment with normal pick / pack / load progression
2. container-heavy segment with unresolved shortage before load
3. approved substitution on a substitutable requirement
4. non-substitutable shortage that blocks execution
5. warehouse/crew records routine transitions without manager involvement
6. dispatcher corrects or escalates an execution exception
7. generic consumable workflow without containers
8. reusable asset return to storage
9. mixed container + loose-stock segment

## UI Direction

Warehouse execution should move toward a requirement-first workflow:

- job picklist
- segment picklist
- requirement or container picklist
- explicit action buttons for the next allowed action

Routine actions should be:

- `Pick`
- `Pack`
- `Load`
- `Mark in transit`
- `Mark unloaded`
- `Return to storage`
- `Flag exception`

## Scan Support Direction

Warehouse execution should support both:

- barcode
- QR

Primary scan targets:

- item identifiers
- asset tags
- container references
- handoff references where useful

Scan support should accelerate the same execution workflow, not create a second
parallel process.

## Implementation Notes

This document is intentionally decision-complete enough to guide implementation.
The current codebase now implements:

- constrained warehouse progression for routine `pick -> pack -> load` actions
- substitution requests backed by explicit reason catalogs
- dispatcher / operations approval-role checks for substitution decisions

Remaining work should deepen warehouse ergonomics and execution tooling rather
than reverting to free-form state entry or free-text-only substitution
governance.
