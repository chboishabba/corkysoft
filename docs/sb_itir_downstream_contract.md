# Corkysoft -> SB / ITIR Downstream Contract

This document defines the intended downstream contract for diary, planning, and
reconciliation outputs from Corkysoft into StatiBaker (SB) / ITIR.

It is a design contract, not an implemented integration.

## Boundary

- Corkysoft remains authoritative for jobs, segments, diary tasks, assignments,
  invoice review, subcontractor bill review, and operational exceptions.
- SB receives append-only downstream envelopes derived from Corkysoft truth and
  reviewed operational decisions.
- ITIR owns orchestration, context routing, and cross-project contract hygiene.
- SB must not become a second mutable workflow database for removals
  operations.

## Event Families

The following family names are the intended stable downstream categories:

- `planning_snapshot`
- `diary_task_event`
- `job_usage_review`
- `vehicle_usage_review`
- `staff_usage_review`
- `customer_invoice_review`
- `subcontractor_bill_review`
- `reconciliation_exception`
- `compliance_gap_flag`

## Minimum Envelope Contract

Every downstream envelope should contain:

- `event_id`
- `event_family`
- `event_time`
- `source_system`
- `actor_ref`
- `authority_class`
- `correlation_key`
- `summary`
- `object_refs`
- `status`
- `provenance_refs`
- `evidence_refs`
- `payload`

## Object References

`object_refs` should use Corkysoft-native identifiers when available:

- `job_id`
- `segment_id`
- `task_id`
- `invoice_review_id`
- `bill_review_id`
- `worker_id`
- `truck_id`
- `supplier_id`

## Authority Classes

Downstream consumers must not assume all Corkysoft envelopes are equally strong.

Use these authority classes:

- `operational_truth`
- `reviewed_summary`
- `observed_actual`
- `downstream_projection`

Rules:

- Corkysoft may emit `operational_truth`, `reviewed_summary`, or
  `observed_actual`.
- SB may derive `downstream_projection` views.
- SB must not upgrade a weaker class into stronger truth without an explicit
  upstream contract.

## Family-Specific Intent

### `planning_snapshot`

Represents a timeboxed day/week view of planned work and key exceptions.

### `diary_task_event`

Represents create/update/close/re-scope events for diary tasks.

### `job_usage_review`

Represents required-vs-utilized review at the job level.

### `vehicle_usage_review`

Represents vehicle-focused job review.

### `staff_usage_review`

Represents staff-focused job review.

### `customer_invoice_review`

Represents operator-reviewed customer invoice readiness or release state.

### `subcontractor_bill_review`

Represents operator-reviewed third-party or subcontractor bill state.

### `reconciliation_exception`

Represents explicit exception state rather than generic alert noise.

### `compliance_gap_flag`

Represents missing requirement/proposal/governance state for compliance-heavy
jobs.

## What Is Not Emitted

This contract does not assume downstream emission of:

- raw accounting-ledger truth
- autonomous pricing or scheduling decisions
- transcript text as authoritative workflow truth
- unreviewed extracted claims promoted directly into operational fact
- any event that requires SB to infer business state Corkysoft does not store

## Transport Posture

The contract is transport-agnostic.

The intended operational shape is compatible with Corkysoft's existing
append-only outbox pattern used for call-intelligence downstream preparation:

- append-only local event creation
- idempotent/resumable delivery
- explicit delivery receipts
- no synchronous UI-path delivery requirement

## Current Implementation Status

- Call-intelligence downstream preparation exists as an analogous pattern.
- Diary/planner/reconciliation downstream export is not implemented yet.
- This document exists to keep future export work bounded and authority-safe.
