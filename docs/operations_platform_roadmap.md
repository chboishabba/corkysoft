# Operations Platform Roadmap

Last updated: **2026-07-17**

## Purpose

Corkysoft already has a pricing, profitability, planning, dispatch, inventory,
and telemetry substrate. This roadmap consolidates the planned work that turns
those separate capabilities into a coherent daily operations flow:

**Quote -> accepted work -> plan -> assign -> execute -> update -> complete -> review**

It is not a claim that Corkysoft is already a complete moving-company operating
system. It is the bounded bridge from the current decision-intelligence product
to a usable operational execution product.

## Product Boundary

### Current and partly implemented

- quote, margin, corridor, and profitability intelligence
- accepted-job, segment, truck, worker, readiness, and conflict state
- native job-centric Dispatch board with inventory and supplier context
- planned-labor roster, shift reconciliation, worker-time review, and payroll
  preparation analytics
- live telemetry and ETA primitives for internal Network views
- inventory requirements, custody, shortages, substitutions, and execution
  stages
- operations diary, job-cost, invoice, and subcontractor-bill review

### Next operational bridge

1. Calendar-first dispatch over the existing assignment and conflict engine.
2. A lightweight crew-facing workflow for assignment acknowledgement, job
   context, routine state changes, inventory events, evidence, and time
   capture.
3. A promoted job-state and customer-communication contract for booking,
   day-before, delay, milestone, completion, receipt, and support updates.
4. A quote-to-job handoff that preserves commercial assumptions, operational
   fit, and ownership without re-entry.
5. Reviewed evidence and closure gates that connect execution, customer proof,
   and commercial reconciliation.

### Deliberately not committed as near-term parity work

- general-purpose CRM, web-lead capture, lead attribution, and sales-pipeline
  automation
- payment processing, payment links, and accounting-system synchronisation
- reputation/review-request automation
- storage-business billing and warehouse-occupancy management
- an end-to-end AI video-survey/cube-sheet product
- a broad marketplace of third-party connectors

These may be integrated with rather than replaced by Corkysoft. Any decision
to internalize one needs a separate product case, source-of-truth boundary,
privacy model, and delivery milestone.

## Milestones

### 0. Protect the control plane

Keep the active P0/P1 work in the progress board ahead of customer-facing or
high-authority automation: scoped credentials, import/migration correctness,
decision-signal provenance, CI reproducibility, and bad-case closure.

**Promotion gate:** no crew or customer action may be introduced without an
authenticated actor where required, explicit authority, audit receipt, and
safe stale/unknown behavior.

### 1. Calendar-first dispatch

Build a daily/weekly dispatcher view over `job_segments`, planned workers,
trucks, readiness, and conflict policy. The calendar must make conflicts,
leave/compliance constraints, travel-time assumptions, capacity, and manual
overrides understandable before confirming an assignment.

**Completion evidence:** a dispatcher can assign, reschedule, and explain a
day's work without spreadsheet-side assignment columns; conflict and override
coverage remains test-backed.

### 2. Crew execution workflow

Provide a small mobile-web-first surface rather than prematurely committing to
native iOS/Android apps. It should present approved job details, address,
access instructions, inventory, required evidence, and a limited set of
role-scoped actions. It must tolerate offline capture/queueing where evidence
or routine execution events are recorded.

**Completion evidence:** workers can acknowledge work and submit routine
events; higher-authority substitutions, exceptions, and closure decisions are
routed to the responsible operator with a receipt.

### 3. Job state, customer updates, and tracking

Promote a reviewed state flow such as:

`assigned -> acknowledged -> travelling -> arrived -> loading -> in_transit -> unloading -> completed -> exception`

Project only approved, customer-safe information into the separate tracking,
notification, and receipt contracts. ETA, delay, and milestone messages must
carry freshness and uncertainty rules; customer links must be tokenized,
scoped, expiring, revocable, and auditable.

**Completion evidence:** a bounded customer page renders approved status and
ETA without internal leakage, and delivery/completion can issue or deliberately
withhold a receipt/POD with a recorded reason.

### 4. Quote-to-job lifecycle

Make quote acceptance create or link operational work with the accepted scope,
commercial assumptions, policy result, operational-fit signals, owner, and
next action intact. The lifecycle should hand off cleanly into planning,
dispatch, execution, completion, and reconciliation.

**Completion evidence:** an operator can trace one accepted quote through job,
segment, assignment, completion, invoice readiness, and realized-margin
review without re-keying the core commercial context.

### 5. Evidence and closure

Implement reviewed PEC/media, inventory custody, time, exception, and
completion evidence only after the capture, storage, consent, integrity, and
authority boundaries are real. The media/RFID/bodycam concepts are roadmap
items, not current end-to-end product claims.

**Completion evidence:** an auditable job-close decision can verify required
operational evidence and customer-safe proof, while unresolved exceptions
remain visible to an owner.

## Competitive Positioning

Broad removals suites set the baseline for calendar dispatch, crew access,
customer updates, and workflow continuity. Corkysoft should not claim full
CRM-to-payment-to-review parity today.

Its near-term proposition is:

> A moving-company operations platform with unusually deep pricing,
> profitability, corridor, and decision intelligence.

The operational board should answer both whether an assignment is feasible and
whether it is commercially sensible: expected margin, lane history, capacity,
overtime, dead-running, backload opportunity, readiness, and evidence risk.

## Related Authority Documents

- [Positioning](positioning.md) for system-of-decision and integration
  boundaries.
- [Spreadsheet Replacement Plan](spreadsheet_replacement_plan.md) for current
  native dispatch, planning, and cutover work.
- [Service Blueprint](service_blueprint.md) for lifecycle, crew, customer,
  and notification requirements.
- [Customer tracking and receipt roadmap](../ROADMAP.md#5-customer-facing-tracking-and-receipt-surfaces)
  for public-safe access controls.
- [Media ingest workflow](media_ingest.md) for evidence-capture requirements.
