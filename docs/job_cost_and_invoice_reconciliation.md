# Job Cost and Invoice Reconciliation

This document defines how Corkysoft should help operators reconcile completed
jobs against customer invoices, subcontractor bills, and actual resource usage.

## Goal

Give operations and finance-facing managers one job-level review flow that
answers:

- what was required for this job?
- what was actually utilized?
- what should the customer be invoiced for?
- what third-party or subcontractor cost must be accepted, challenged, or
  clarified?

## Exposure Timing Model

Reconciliation should distinguish two different clocks:

- economic exposure starts when the job was executed
- operational exposure starts when the bill or invoice actually became
  actionable

For subcontractor / third-party bills, Corkysoft should preserve both:

- `job_execution_date`: derived from job / segment execution truth
- `bill_date`: the bill-received / bill-actionable marker used for aging

This creates two useful measures:

- latent liability window: `bill_date - job_execution_date`
- active unresolved exposure age: `as_of_date - bill_date` until reconciliation

The active unresolved age is the default aging clock for manager review.
The latent window remains visible so delayed billing does not hide true margin
distortion.

## Core Reconciliation Inputs

Operational context:

- job record
- `job_segments`
- assigned trucks
- assigned workers
- inventory and supplier coordination
- diary tasks linked to the job

Actual/reconciled context:

- imported driver shifts
- reviewed worker-time capture
- subcontractor / supplier bill records
- vehicle or labor exceptions captured during execution

## Review Surfaces

### Customer invoicing

Customer-side review should answer:

- is the job complete enough to invoice?
- are there unresolved operational exceptions that make invoicing unsafe?
- does the invoice reflect the known route, staffing, and supplier context?

### Subcontractor bills

Subcontractor-side review should answer:

- which supplier/subcontractor was actually used?
- which job and segment(s) does the bill relate to?
- do the bill dates line up with the execution period?
- do planned vs utilized truck/staff/supplier signals support the claim?
- is this a normal bill, partial bill, duplicate bill, or exception case?

## Exception Categories

The system should surface explicit exceptions such as:

- missing customer invoice
- invoice not ready because operational truth is incomplete
- subcontractor bill received with no linked job
- subcontractor bill linked to a job but not to any known segment/supplier
  context
- late holiday / peak-period bill needing manager review
- planned truck/staff usage does not match actual/reconciled usage
- third-party cost exists without a matching task or operational note
- received-but-unreconciled bill carrying exposure for weeks or months
- job appears profitable only because delayed supplier cost has not been
  processed yet

## Aging / exposure review

The reconciliation layer should help managers answer:

- which received-but-unreconciled third-party bills have been sitting too long?
- which job or account carries the largest unresolved supplier exposure?
- which supplier bills arrived much later than the work occurred?

The lightweight Corkysoft surface should therefore provide:

- top unresolved supplier-liability rows
- unresolved supplier-liability total
- open customer-side review total where known
- oldest active unresolved supplier age
- longest supplier billing latency

This is a review summary, not a full accounting ledger.

## Status Model

Customer invoice statuses:

- `not_ready`
- `ready_to_invoice`
- `partially_invoiced`
- `invoiced`
- `reconciliation_warning`

Subcontractor bill statuses:

- `no_bill_expected`
- `awaiting_bill`
- `bill_received`
- `bill_reconciled`
- `bill_exception`

## V1 Boundary

V1 should support:

- persistent customer invoice review records per job
- persistent subcontractor bill records linked to jobs and optionally segments
- explicit resolution timestamps when a bill or invoice leaves unresolved state
- diary drill-through into a reconciliation view
- required-vs-utilized summaries for trucks, staff, inventory/suppliers, and
  known financial review records
- manager-facing unresolved-exposure summary using job execution date plus bill
  / invoice date semantics

V1 should not assume:

- perfect actual-capture coverage
- external accounting integration
- automatic approval of supplier or subcontractor bills
