# Usage Onboarding Guide

This guide is the practical entry point for new Corkysoft users. It is written
as a wiki-style operational manual rather than an implementation note.

## Purpose

Use Corkysoft as the internal operational and commercial workspace for:

- quoting and commercial review
- job dispatch and segment planning
- truck and worker assignment
- maintenance and compliance readiness
- inventory and supplier coordination
- Kent tender triage
- spreadsheet cutover governance

Google Sheets remain import inputs where needed, but daily planning should
happen inside Corkysoft.

## Role Map

| Role | Primary tabs | Secondary tabs |
| --- | --- | --- |
| Estimator | `Quote builder` | analytics tabs |
| Dispatcher | `Dispatch`, `Kent tenders` | `Operations` |
| Fleet / Operations Manager | `Operations`, `Dispatch`, `Fleet` | `Vehicle maintenance` |
| Commercial Owner | `Quote builder`, `Kent tenders` | analytics tabs, `Kent admin` review |
| Labor Planner / Staff Coordinator | `Staff`, `Driver shifts`, `Operations` | `Dispatch` |
| Maintenance / Compliance Coordinator | `Fleet`, `Vehicle maintenance` | `Operations`, `Dispatch` |
| Warehouse / Crew | `Inventory` | `Dispatch` |
| Inventory / Supplier Coordinator | `Inventory` | `Dispatch` |
| Workforce Time Capture Coordinator | `Staff`, `Driver shifts` | future time-capture review surface |
| System / Rollout Admin | `Fleet`, `Kent admin` | `Staff`, `Inventory`, `Driver shifts` |

See [UI Role Coverage Matrix](ui_role_coverage_matrix.md) for the authoritative role-to-surface mapping.

## Roles and Starting Surfaces

### Estimator

Primary tab:
- `Quote builder`

Use it to:
- create or review a quote
- inspect profitability-policy output
- judge whether the route and operational signals support the work

What to look for:
- price and cost summary
- profitability pass/fail state
- operational fit context

Do not use it for:
- assigning trucks or workers
- spreadsheet cutover administration

### Dispatcher

Primary tabs:
- `Dispatch`
- `Kent tenders`

Use them to:
- review the native job-centric queue
- inspect segment readiness
- see truck, worker, inventory, and supplier context in one place
- prioritize tenders and booked work by urgency, flags, and operational fit
- export a dispatch snapshot when an external party still needs a lightweight extract

What to look for:
- job status
- segment detail
- warning, block, and override counts
- current rollout state for dispatch
- tender priority and override tags

Do not use them for:
- changing rollout status
- changing cutover governance
- editing Kent policy defaults

### Fleet / Operations Manager

Primary tabs:
- `Operations`
- `Dispatch`
- `Fleet`

Secondary tab:
- `Vehicle maintenance`

Use them to:
- create and assign segments
- inspect assignment conflicts and readiness state
- manage native planning rather than relying on spreadsheet-side assignment truth
- review whether current execution still matches plan

What to look for:
- assignment conflicts
- blocked and due-soon items
- provenance and freshness of imported operational data
- cross-job truck and worker pressure

Do not use them for:
- staff roster administration as the primary workflow
- supplier coordination as the primary workflow
- commercial policy governance unless acting in an approved admin role

### Commercial Owner

Primary tabs:
- `Quote builder`
- `Kent tenders`

Secondary / governed-review tab:
- `Kent admin`

Use them to:
- review quote quality and tender prioritization
- inspect profitability policy behavior
- review override and governance separately from the operator queue
- approve or reject governed rollout promotions when required

What to look for:
- quote profitability outputs
- tender prioritization order
- policy behavior, override history, and governed approval state
- promotion approval state and risk

Do not use them for:
- truck or worker assignment
- daily dispatch execution
- source-sheet syncing

Current access note:
- `Kent admin` is currently writable only for `System / Rollout Admin`. Treat it
  as a review surface unless you are acting in that admin role.

### Labor Planner / Staff Coordinator

Primary tabs:
- `Staff`
- `Driver shifts`
- `Operations`

Use them to:
- maintain the worker roster
- review planned assignments by worker
- reconcile imported shifts against the native labor plan
- maintain roles and compliances for upcoming work

What to look for:
- active workers and recent shift history
- planned segment assignments per worker
- plan-vs-imported reconciliation state
- missing roles or expiring compliances

Do not use them for:
- fleet cutover administration
- vehicle maintenance control
- supplier coordination

### Maintenance / Compliance Coordinator

Primary tabs:
- `Fleet`
- `Vehicle maintenance`

Secondary tabs:
- `Operations`
- `Dispatch`

Use them to:
- review due-soon and blocked readiness items
- inspect the trucks impacted by rego, COI, service, or worker compliance issues
- check maintenance history and current assignment impact

What to look for:
- blocked items
- due-soon items
- planned workers and upcoming segments tied to affected trucks
- repair history and service context

Do not use them for:
- roster administration beyond what readiness requires
- rollout promotion approvals unless acting as the designated admin

### Warehouse / Crew

Primary tab:
- `Inventory`

Secondary tab:
- `Dispatch`

Use them to:
- move planned stock through pick / pack / load
- record custody/location changes
- surface shortages and execution exceptions
- request substitutions when the planned stock is not available

What to look for:
- requirement lines and allocated stock
- current custody/location state
- execution-stage progress
- shortage or mismatch conditions that need dispatcher review

Do not use them for:
- approving substitutions
- changing rollout or admin policy
- supplier governance or workbook syncing

### Inventory / Supplier Coordinator

Primary tab:
- `Inventory`

Secondary tab:
- `Dispatch`

Use them to:
- review stock balances and supplier context
- allocate inventory to planned segments
- record movements and reconcile exceptions
- monitor whether warehouse execution still matches the plan
- follow up on shortages, substitutions, and supplier-side risk
- follow up on remote/overnight operational support constraints where they affect execution

What to look for:
- segment-linked allocations
- warehouse execution state
- supplier list and sheet provenance
- outstanding exceptions
- reserved vs available stock
- corridor or job contexts where external support availability may become operationally tight

Do not use them for:
- truck or worker assignment
- routine pick / pack / load progression
- approving substitutions
- Kent policy administration
- rollout promotion governance

### Workforce Time Capture Coordinator

Primary tabs:
- `Staff`
- `Driver shifts`

Use them to:
- reconcile captured clock-on / clock-off events against the labor plan
- review low-confidence events from app, WhatsApp, or voice/landline capture
- correct worker/time matches before labor actuals are treated as settled

What to look for:
- worker identity match
- event-time confidence
- source channel
- mismatch against planned shift or assignment

Do not use them for:
- truck maintenance governance
- supplier coordination
- warehouse inventory execution

### Owner / Commercial / Finance-facing Manager

Primary future tab:
- `Payroll / Labor analytics`

Supporting tabs:
- `Staff`
- `Driver shifts`

Use them to:
- forecast labor/pay exposure over a selected date range
- inspect overtime, hours-worked, and labor-cost distributions
- review absence and anomaly trends
- prepare export-ready labor summaries for external payroll/accounting tools

What to look for:
- aggregate labor cost and hours trends
- payroll-prep confidence
- unresolved anomaly and review counts
- outlier workers, teams, jobs, clients, or corridors when a drill-down is justified

Do not use them for:
- full payroll execution
- tax/super/deduction handling
- surveillance-style monitoring
- direct bookkeeping replacement

### System / Rollout Admin

Primary tab:
- `Fleet`

Secondary tabs:
- `Staff`
- `Inventory`
- `Driver shifts`

Use them to:
- sync the shared operations workbook
- manage spreadsheet cutover state
- record reviews, drills, fallback use, and snapshot issuance
- request, approve, reject, and apply rollout promotions
- run source-sheet imports for staff, suppliers, and driver shifts

What to look for:
- recommendation reason
- approval state
- derived native-usage metrics
- rollback instructions
- import results and source-sheet health

Do not use it for:
- day-to-day dispatching
- daily tender queue work
- quote creation

## Recommended Daily Flow

1. Estimator reviews or creates quotes in `Quote builder`.
2. Dispatcher runs booked work from `Dispatch` and tender triage from `Kent tenders`.
3. Fleet / operations management maintains segments and assignments in `Operations`.
4. Labor planning is maintained in `Staff` and reconciled in `Driver shifts`.
5. Maintenance and compliance are monitored in `Fleet` and `Vehicle maintenance`.
6. Warehouse / crew runs daily inventory execution in `Inventory`, with execution context from `Dispatch`.
7. Inventory and supplier coordination follow the same work in `Inventory`, focusing on shortages, substitutions, supply-side exceptions, and support constraints around the work.
8. Workforce time capture should reconcile into `Staff` and `Driver shifts`, even when the source channel is WhatsApp or a voice/landline path.
9. Kent policy and override governance are maintained in `Kent admin`.
10. Spreadsheet cutover reviews and promotions are handled in `Fleet`, not in operator tabs.
11. Payroll and labor analytics should stay aggregate-first by default; person-level drill-down is for anomaly resolution, payroll prep, or planning justification rather than general worker monitoring.

## Spreadsheet Cutover Rules

- `Dispatch` is read-only for rollout governance.
- `Fleet` is the cutover administration surface.
- Workflow promotion is governed:
  - operations manager requests promotion
  - commercial owner approves or rejects it
  - admin applies the transition only after approval exists

## Common Mistakes To Avoid

- Do not treat imported spreadsheet values as the final planning truth once segments have been assigned in Corkysoft.
- Do not use `Dispatch` to change rollout state.
- Do not use `Kent admin` as the day-to-day tender queue.
- Do not bypass readiness warnings without recording the correct override path.
- Do not treat `Staff`, `Inventory`, or `Fleet` admin sections as the primary daily execution surface unless your role owns that admin workflow.

## When To Export Or Fall Back

Export a snapshot only when:
- an external team still needs a CSV extract
- a rollback drill requires one
- native execution is temporarily unavailable

Record fallback use when:
- the team bypasses the native surface and continues from a snapshot or source sheet

## Related Docs

- [Operator User Stories](operator_user_stories.md)
- [UI Role Coverage Matrix](ui_role_coverage_matrix.md)
- [Rollout Execution Stories](rollout_execution_user_stories.md)
- [Quote to Award Lifecycle](commercial_workflow_lifecycle.md)
- [Spreadsheet Replacement Plan](spreadsheet_replacement_plan.md)
- [Payroll and Labor Analytics](payroll_and_labor_analytics.md)
- [Naive User Tester Notes](naive_user_tester_notes.md)
