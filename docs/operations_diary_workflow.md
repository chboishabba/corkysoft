# Operations Diary Workflow

This document defines the manager-facing `Operations diary` workflow inside the
`Operations` shell.

The diary is a separate operational cockpit that sits above `Planner`,
`Dispatch`, labor reconciliation, supplier coordination, and invoicing.

## Goal

Give an operations manager one day/week surface where they can:

- see every job, segment, task, truck, worker, supplier, and exception that
  matters for the selected period
- move from planning into execution without losing the financial follow-through
- compare required vs utilized resources for each job
- move directly into customer invoicing and subcontractor-cost reconciliation
- add or remove operational tasks for the day or week without forcing every
  managerial activity into `job_segments`

## Why This Exists

The current repo already has most of the underlying truth:

- `job_segments` for operational planning
- truck and worker assignments
- native planned labor roster and imported/reviewed actuals
- inventory and supplier coordination
- job-centric dispatch rollups

The missing layer is an operator workflow that ties those signals together into
one daily/weekly manager view.

## Primary Audience

Primary operator:
- operations manager

Secondary operators:
- dispatcher
- labor planner / staff coordinator
- finance-facing manager reviewing invoice and bill exceptions

## Core Views

### 1. Day View

The day view is the daily execution and review cockpit.

It should show:

- jobs active on the selected date
- planned segments and times
- diary tasks for the date
- planned truck usage
- planned staff usage
- actual/reconciled labor signals where available
- inventory and supplier/subcontractor dependencies
- customer invoice readiness
- subcontractor bill / reconciliation status
- explicit exceptions and unresolved mismatches
- unresolved supplier-liability summary for received-but-unreconciled bills

### 2. Week View

The week view is the planning and workload-balancing cockpit.

It should show:

- jobs and diary tasks grouped by day
- truck and worker pressure across the week
- pending invoice or subcontractor-bill issues that need escalation
- jobs with unresolved plan-vs-actual gaps
- manager-level ability to add/remove tasks and decide the next operational
  actions

## Navigation Contract

The diary is a separate workflow inside `Operations`, not a Planner subpanel.

Entry points:

- `Planner` day view should link to the diary for the selected date/job
- `Dispatch` job inspection should link to the diary for follow-through
- finance/reconciliation drill-down should return to the same diary context

Drill-through destinations from the diary:

- job usage
- vehicle usage
- staff usage
- invoicing / reconciliation

## Plan vs Actual

The diary defaults to a plan-vs-actual comparison model.

Plan sources:

- `job_segments`
- truck assignments
- worker assignments
- segment-linked inventory/supplier coordination

Actual/reconciled sources:

- imported driver shifts
- reviewed worker-time events
- accepted supplier/subcontractor records
- invoice/bill review state

The diary must show uncertainty explicitly when actuals are incomplete rather
than pretend the data is complete.

## Diary Tasks

Diary tasks are a distinct persisted object.

They are not just free-text notes and they are not merely a UI projection of
`job_segments`.

Tasks may be linked to:

- a job
- a segment
- a day
- a week

Examples:

- confirm third-party Christmas/New Year invoice against actual job usage
- chase missing truck allocation evidence for a completed job
- review customer invoice before release
- add an extra depot/load-prep action for tomorrow

## Financial Review Expectations

The diary must support both:

- outbound customer invoicing follow-through
- inbound subcontractor bill reconciliation

The current urgent pain point is subcontracted-job bill review, especially when
third-party invoices arrive after the operational work is complete.

The lightweight Corkysoft diary should therefore surface:

- unresolved supplier-liability total
- oldest unresolved supplier age
- supplier billing latency markers using job execution date vs bill-received
  date
- top overdue third-party rows so the manager can open the underlying job and
  reconcile it

## V1 Boundary

V1 should provide:

- the new diary screen
- day and week filters
- diary task create/update/delete
- job/vehicle/staff/invoicing drill-through
- plan-vs-actual usage summaries
- customer invoice status and subcontractor bill status with explicit exception
  categories
- unresolved supplier-exposure summary rows and metrics inside the diary

V1 does not need:

- full accounting ledger behavior
- external accounting sync
- automatic invoice generation across every source system
- replacement of the existing Planner or Dispatch workflows
- the full long-horizon timeline/lens view intended for downstream SB analysis
