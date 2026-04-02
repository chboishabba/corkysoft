# UI Role Coverage Matrix

This matrix maps current Corkysoft roles to the main dashboard surfaces.
Use it as the canonical reference for who should start where.

Current access-boundary note:
- `Kent admin` is visible as a governed review surface for `Commercial Owner`,
  but write access is currently restricted to `System / Rollout Admin`.

| Role | Quote builder | Dispatch | Operations | Fleet | Vehicle maintenance | Staff | Driver shifts | Inventory | Kent tenders | Kent admin | Payroll / Labor analytics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Estimator | primary | not used | not used | not used | not used | not used | not used | not used | secondary | not used | not used |
| Dispatcher | not used | primary | secondary | secondary | not used | not used | not used | secondary | primary | not used | not used |
| Fleet / Operations Manager | not used | primary | primary | primary | secondary | secondary | secondary | secondary | secondary | not used | secondary |
| Commercial Owner | primary | not used | not used | not used | not used | not used | secondary | not used | primary | secondary | primary |
| Labor Planner / Staff Coordinator | not used | secondary | primary | secondary | not used | primary | primary | not used | not used | not used | secondary |
| Maintenance / Compliance Coordinator | not used | secondary | secondary | primary | primary | secondary | not used | not used | not used | not used | not used |
| Warehouse / Crew | not used | secondary | not used | not used | not used | not used | not used | primary | not used | not used | not used |
| Inventory / Supplier Coordinator | not used | secondary | secondary | not used | not used | not used | not used | primary | not used | not used | not used |
| Workforce Time Capture Coordinator | not used | secondary | not used | not used | not used | primary | primary | not used | not used | not used | secondary |
| System / Rollout Admin | not used | not used | secondary | primary | secondary | admin-only | admin-only | admin-only | not used | primary | admin-only |

## Interpretation

- `primary`: the role should normally start here for its main task.
- `secondary`: the role uses this surface for context or follow-up work.
- `admin-only`: the role uses only governance, import, or maintenance sections of the surface.
- `not used`: the surface is outside the role's normal workflow.

## Mixed Surfaces

### Fleet

`Fleet` currently mixes:
- workbook sync
- readiness policy
- rollout admin
- maintenance/compliance cockpit
- vehicle register and import/export

This is operationally valid today, but roles should treat it as a mixed surface rather than a pure day-to-day board.

### Staff

`Staff` currently mixes:
- roster editing
- linked-shift review
- role/compliance admin
- worker readiness review

This is the correct home for labor coordination, but some sections are coordination-focused while others are admin/governance.

### Payroll / Labor analytics

This is a future management and finance-facing surface for:
- payroll preparation
- labor forecasting
- overtime/hours/cost distributions
- absence and anomaly summaries

It should default to aggregate and exception views rather than person-by-person monitoring.

### Inventory

`Inventory` currently mixes:
- warehouse execution workflow
- segment-linked stock coordination
- supplier import
- movement event import
- reserve/release actions
- exception reconciliation

This is acceptable for the current scale, but users should distinguish daily warehouse execution and stock coordination from source-data administration.
