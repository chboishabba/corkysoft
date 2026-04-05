# UI Role Coverage Matrix

This matrix maps current Corkysoft roles to the revised top-level dashboard
shell. Treat the shell views as entrypoints, then use the noted leaf workflows
inside each shell.

Current shell taxonomy:

- `Quote`
- `Pricing Intelligence`
- `Network`
- `Operations`
- `Admin`

Current access-boundary note:

- `Admin` is the governed system and Kent-policy surface.
- `Commercial Owner` may review admin/governance state where exposed, but live
  write access remains restricted to `System / Rollout Admin`.

| Role | Quote | Pricing Intelligence | Network | Operations | Admin |
| --- | --- | --- | --- | --- | --- |
| Estimator | primary | primary | secondary | not used | not used |
| Dispatcher | secondary | not used | primary | primary | not used |
| Fleet / Operations Manager | not used | secondary | primary | primary | not used |
| Commercial Owner | primary | primary | not used | secondary | secondary-review |
| Labor Planner / Staff Coordinator | not used | not used | secondary | primary | not used |
| Maintenance / Compliance Coordinator | not used | not used | secondary | primary | not used |
| Warehouse / Crew | not used | not used | not used | primary | not used |
| Inventory / Supplier Coordinator | not used | not used | not used | primary | not used |
| Workforce Time Capture Coordinator | not used | not used | not used | primary | not used |
| System / Rollout Admin | secondary | secondary | secondary | secondary | primary |

## Interpretation

- `primary`: normal entrypoint for the role.
- `secondary`: supporting context or follow-up work.
- `secondary-review`: visible for governed review, but not the main operating surface.
- `not used`: outside the role's normal workflow.

## Leaf Workflow Mapping

### Quote

Primary leaf workflows:

- `Quote builder`
- `Calls`
- `Kent tenders`

Typical owners:

- Estimator
- Commercial Owner
- Dispatcher for tender follow-up context

### Pricing Intelligence

Primary leaf workflows:

- optimizer
- histogram / distribution review
- price history
- profitability insights

Typical owners:

- Estimator
- Commercial Owner
- Fleet / Operations Manager for review, not daily entry

### Network

Primary leaf workflows:

- live network overview
- telemetry and route-map review
- corridor and historic route overlays

Typical owners:

- Dispatcher
- Fleet / Operations Manager

### Operations

Primary leaf workflows:

- `Dispatch`
- `Planner`
- `Operations diary`
- `Operations`
- `Fleet`
- `Vehicle Maintenance`
- `Inventory`
- `Staff`
- `Driver Shifts`
- `Payroll / Labor`

Typical owners:

- Dispatcher
- Fleet / Operations Manager
- Labor Planner / Staff Coordinator
- Maintenance / Compliance Coordinator
- Inventory / Supplier Coordinator
- Workforce Time Capture Coordinator
- Warehouse / Crew

### Admin

Primary leaf workflows:

- dashboard user administration
- Kent governance/admin controls
- system-log and sync actions

Typical owners:

- System / Rollout Admin
- Commercial Owner for governed review only where exposed

## Governance Notes

- The revised shell improves operator focus, but mixed surfaces still exist
  inside `Operations`; keep execution, review, and admin subsections visibly
  separated there.
- KPI strips and alert banners in the new views should be treated as
  presentation scaffolding until the values are sourced from live metrics with
  freshness and ownership.
