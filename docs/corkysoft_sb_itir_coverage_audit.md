# Corkysoft / SB / ITIR Coverage Audit

This document audits what the client story asked for, what Corkysoft now
documents and implements, and what remains only planned or conceptual in
StatiBaker (SB) / ITIR.

Locked boundary:

- Corkysoft is the operational truth and decision surface.
- StatiBaker is a downstream interpretible log / compiled-state consumer.
- ITIR is the orchestration and contract/context layer across systems.

## Coverage Matrix

| Capability | Client need | Current Corkysoft docs | Current Corkysoft implementation | SB / ITIR status | Next missing step | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| Planner day view | Compare proposed work against the live day plan | `docs/planner_interaction_model.md` | Implemented in Planner with day/date focus | No direct SB surface | Keep planner/day wording aligned with actual UI behavior | Corkysoft |
| Planner -> diary navigation | Move from planning into day/week operational review | `docs/planner_interaction_model.md`, `docs/operations_diary_workflow.md` | Implemented via Planner link into `Operations diary` | No direct SB surface | Preserve selected job/date context when expanding drill-through | Corkysoft |
| Operations diary day view | Daily cockpit for jobs, tasks, usage, and invoice/bill issues | `docs/operations_diary_workflow.md` | Implemented | Conceptual only as future downstream summary | Emit downstream day-review events only after contract is defined | Corkysoft now, SB later |
| Operations diary week view | Weekly workload and escalation planning | `docs/operations_diary_workflow.md` | Implemented | Conceptual only as future downstream summary | Decide if weekly summaries are emitted as snapshots or derived by SB | Corkysoft now, cross-project later |
| Job usage drill-through | See what was required vs utilized per job | `docs/operations_diary_workflow.md`, `docs/job_cost_and_invoice_reconciliation.md` | Implemented | No direct SB surface | Define downstream `job_usage_review` envelope if needed | Corkysoft now, cross-project later |
| Vehicle usage drill-through | Inspect vehicle usage for a job | `docs/operations_diary_workflow.md`, `docs/operator_user_stories.md` | Implemented | No direct SB surface | Define whether SB should render vehicle-usage exceptions as evidence cards | Corkysoft now, SB pattern later |
| Staff usage drill-through | Inspect staff usage and mismatches for a job | `docs/operations_diary_workflow.md`, `docs/operator_user_stories.md` | Implemented | No direct SB surface | Define downstream `staff_usage_review` event family | Corkysoft now, cross-project later |
| Diary task add/remove/update | Track day/week follow-up work separate from segments | `docs/operations_diary_workflow.md` | Implemented | Conceptual only | Define downstream `diary_task_event` family for reviewed task changes | Corkysoft now, cross-project later |
| Customer invoice readiness review | Decide whether completed jobs are ready to invoice | `docs/job_cost_and_invoice_reconciliation.md` | Implemented as review/status records | No direct SB surface | Define downstream `customer_invoice_review` envelope | Corkysoft now, cross-project later |
| Subcontractor / third-party bill reconciliation | Reconcile inbound third-party bills against operational truth | `docs/job_cost_and_invoice_reconciliation.md`, `docs/operations_diary_workflow.md` | Implemented as review/status records | No direct SB surface | Define downstream `subcontractor_bill_review` and exception envelopes | Corkysoft now, cross-project later |
| Late holiday-period bill handling | Review Christmas / New Year subcontractor bills after the fact | `docs/operations_diary_workflow.md`, `docs/job_cost_and_invoice_reconciliation.md` | Covered by implemented bill review, exception categories, and unresolved-exposure summary semantics | No direct SB surface yet | Add fuller time-oriented downstream trace semantics after local aging fields are in place | Corkysoft now, cross-project later |
| Paperwork / compliance-heavy workflow | Make international / paperwork-heavy jobs explicit rather than implicit | `docs/operator_user_stories.md`, `docs/commercial_workflow_lifecycle.md`, `docs/positioning.md` | Documented only; not implemented as a full workflow | Conceptual strategy only | Add a dedicated Corkysoft workflow spec before any code or SB export | Corkysoft |
| Downstream SB / ITIR evidence export for diary/reconciliation | Preserve reviewed operational truth in a cross-source interpretible log | This document plus `docs/sb_itir_downstream_contract.md` | Not implemented yet, except analogous call-intelligence outbox pattern | Conceptual, partially informed by call-intelligence design | Define transport-agnostic envelope contract and later add outbox worker | Cross-project |

## What Is Already Solid In Corkysoft

- The original client story about moving from day view into invoicing, vehicle
  usage, staff usage, and job-level required-vs-utilized review is covered in
  both docs and implementation.
- The urgent pain point around late subcontractor / third-party holiday bills
  is documented explicitly and supported by the current diary/reconciliation
  slice.
- Corkysoft now has the right object types for this workflow:
  `job_segments`, diary tasks, invoice review records, and subcontractor bill
  review records.

## What Is Only Planned Or Conceptual

- International/compliance-heavy paperwork workflow remains product intent,
  backlog, and user-story material. It is not yet a concrete implementation
  slice.
- SB does not yet receive diary/reconciliation exports from Corkysoft.
- ITIR does not yet define a removals-specific diary/reconciliation interface
  beyond the general orchestration boundary.

## Current Cross-Project Truth

- Corkysoft is where job planning, task handling, invoice review, and bill
  reconciliation happen.
- SB should receive reviewed downstream summaries and provenance-bearing events,
  not replace Corkysoft as the operational cockpit.
- ITIR should coordinate contracts and context, not own Corkysoft business
  semantics.
