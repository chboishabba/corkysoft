# Payroll and Labor Analytics

This document defines the product layer between Corkysoft's labor operations
surfaces and any downstream payroll or accounting system.

Use it as the canonical planning document for payroll preparation and workforce
analytics.

## Purpose

Corkysoft already captures and reconciles operational labor truth:

- planned labor assignments from `job_segments`
- imported `VEHICLE_DRIVER` shift rows
- reviewed worker-time capture events from app, WhatsApp, and voice/landline

What is missing is a management and finance-facing layer that answers:

- what labor cost should we expect this pay period or date range?
- where are overtime, absence, or labor-cost patterns emerging?
- what still needs review before labor actuals are trustworthy for payroll prep?

This layer should support:

- payroll preparation
- labor forecasting
- workforce statistics
- labor-cost modeling

It should not become a full payroll or accounting suite.

## Product Posture

### Default stance

This product layer should be useful to owners and managers without turning into
a surveillance tool.

Default presentation should favor:

- aggregates
- distributions
- trends
- exception counts
- review backlog

Person-level drill-down is allowed when justified by:

- payroll preparation
- anomaly resolution
- forecasting
- unusual overtime or absence patterns
- cost/risk investigation

It should not default to ranking or monitoring workers for its own sake.

### Truth model

- reviewed labor actuals inside Corkysoft can become **payroll-prep truth**
- unresolved or ambiguous items remain visible as confidence blockers
- final payroll calculation, statutory handling, and accounting posting stay
  outside Corkysoft in v1

## Scope for v1

V1 should cover:

- payroll preparation
- labor forecasting
- labor distributions and trend analysis
- anomaly and confidence reporting
- export-ready summaries for external payroll/accounting tools

V1 should not cover:

- full pay-run execution
- tax, super, deduction, or leave-entitlement/accrual engines
- award interpretation engine
- full bookkeeping/accounting replacement
- payroll-provider-specific execution workflows

## Core Questions

The layer should answer questions such as:

- how much should we budget for a worker, team, or period between selected dates?
- what does likely payroll exposure look like this week, fortnight, month, or year?
- who or which teams are accumulating unusual overtime?
- where do actual labor hours diverge from planned labor?
- which jobs, clients, or corridors drive unusual labor intensity?
- how much reviewed truth exists versus unresolved labor anomalies?

## Primary Surface Groups

The first implemented shape is a `Payroll / Labor analytics` area with views
modeled more like `Histogram` and `Optimizer` than like `Staff` admin forms.

### 1. Pay Forecast

Use for:

- estimating labor/pay exposure over a date range
- comparing planned labor with reviewed actuals

Expected outputs:

- estimated pay by worker
- estimated pay by team
- planned versus reviewed-actual exposure
- confidence indicator based on unresolved anomalies

### 2. Overtime Distribution

Use for:

- overtime patterns by worker, team, client, corridor, or period

Expected outputs:

- overtime distributions
- outlier visibility
- trend charts over time

### 3. Hours and Cost Distribution

Use for:

- understanding labor intensity and variance

Expected outputs:

- hours-worked distributions
- labor cost distributions
- planned versus actual variance
- cost-per-job / cost-per-corridor summaries

### 4. Absence and Sick-Day Patterns

Use for:

- staffing-risk visibility
- planning support

Expected outputs:

- recorded absence/leave counts
- sick-day trend summaries built from explicit recorded absence rows
- aggregate versus person-level drill-down where justified

The wording and presentation should stay practical and non-punitive.

Current implementation note:

- Corkysoft now has a basic explicit `worker_absence_records` model for recorded
  leave/absence rows.
- This is intentionally better than inferring absence from missing shifts, but it
  is still not a full leave-entitlement or payroll-award engine.

### 5. Review and Confidence Panel

Use for:

- understanding whether payroll-prep truth is trustworthy yet

Expected outputs:

- pending worker-time review counts
- duplicate-event counts
- missing prior clock-on counts
- roster mismatch counts
- accepted versus pending versus rejected review states

### 6. Labor Cost Drivers

Use for:

- seeing what operational work is expensive in labor terms

Expected outputs:

- labor cost by worker/team
- labor cost by job/client/corridor
- labor variance against plan
- concentration views for unusual labor-heavy work

## User Stories

### Owner / Commercial / Finance-facing Manager

Primary surfaces:

- `Payroll / Labor analytics`

Secondary surfaces:

- `Staff`
- `Driver shifts`

Trigger:

- needs labor-cost insight, payroll-prep visibility, or forward budgeting

Primary decisions:

- what should labor/pay likely cost in the selected period?
- which patterns deserve intervention or follow-up?
- where are unresolved reviews or anomalies reducing confidence?

Expected outputs:

- pay forecasts
- overtime distributions
- hours and cost distributions
- absence summaries
- review backlog and confidence status
- drill-down from aggregate to individual where justified

Operator actions:

- review labor trends
- inspect outliers
- forecast payroll exposure
- prepare export-ready summaries
- follow up on anomalies or planning risk

### Labor Planner / Staff Coordinator

Additional responsibilities beyond current labor-planning scope:

- ensure reviewed labor actuals are usable for payroll preparation
- resolve mismatches between plan, imported shifts, and accepted worker-time
- review variance patterns that affect staffing or cost

### Workforce Time Capture Coordinator

Additional responsibilities beyond current capture-review scope:

- ensure accepted events are payroll-prep usable
- reduce unresolved anomaly backlog before period close
- monitor which source channels create the most ambiguity

### System / Rollout Admin

Additional responsibilities:

- maintain export boundaries and mapping assumptions
- ensure Corkysoft remains operational truth for reviewed labor actuals without
  pretending to be the payroll engine

## Data and Interface Concepts

The implementation should eventually distinguish:

- `labor_actual`
  - reviewed operational labor truth suitable for payroll prep
- `payroll_prep_period`
  - a date-bounded forecasting/review/export scope
- `labor_anomaly`
  - an item reducing confidence in payroll-prep truth
- `labor_statistics_view`
  - an aggregate or drill-down analytical slice
- `export_ready_labor_summary`
  - prepared handoff data for payroll/accounting tools

The product should support export boundaries for tools such as:

- payroll providers
- accounting/bookkeeping tools
- QuickBooks / Quicken-class financial systems

But those exports are future integration work, not part of v1 execution.

## Suggested Implementation Order

1. labor analytics and payroll-prep spec alignment
2. aggregate labor statistics surfaces
3. pay-period/date-range forecasting views
4. anomaly/confidence reporting
5. export-ready summaries
6. basic explicit absence/leave recording
7. only later, external payroll/accounting connectors

## Notes

- This is a management and finance insight layer, not a worker surveillance layer.
- Aggregate-first presentation is the default.
- Person-level drill-down is justified only when it supports planning, payroll
  prep, anomaly review, or cost investigation.
- Corkysoft should prepare and explain labor/pay truth, not replace payroll or
  accounting products.
