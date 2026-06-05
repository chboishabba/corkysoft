# UI Role Coverage Matrix

This matrix maps current Corkysoft roles to the revised top-level dashboard
shell. Treat the shell views as entrypoints, then use the noted leaf workflows
inside each shell.

Authority:

- This file owns internal role-to-shell and role-to-leaf workflow mapping.
- [Operator User Stories](operator_user_stories.md) derives actor intent from
  this matrix; it should not create a second role/surface truth table.
- [Usage Onboarding Guide](usage_onboarding_guide.md) derives practical
  instructions from this matrix.
- [Service Blueprint](service_blueprint.md) owns end-to-end lifecycle,
  customer, notification, worker, and completion matrices.

Row schema:

- `role`: internal user class or external actor class.
- `primary shell`: normal starting shell for daily work.
- `primary leaf workflows`: nested workflows that role normally owns.
- `secondary shell / leaf workflows`: supporting context or follow-through.
- `governed-review surfaces`: visible for review but not routine write action.
- `not used / denied`: outside normal workflow or must fail closed.

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
| Customer | not used | not used | not used | not used | not used |
| Support / Customer Service | secondary | not used | secondary | secondary | not used |

## Interpretation

- `primary`: normal entrypoint for the role.
- `secondary`: supporting context or follow-up work.
- `secondary-review`: visible for governed review, but not the main operating surface.
- `not used`: outside the role's normal workflow.

External actor note:

- `Customer` is not an internal dashboard role. Customer-visible status,
  notifications, tracking, and receipts are defined as public-safe projections
  in [Service Blueprint](service_blueprint.md), not as direct dashboard access.
- `Support / Customer Service` is a future or deployment-specific internal
  role for customer-safe inquiry handling. It should not expose internal notes,
  worker data, admin controls, or unreviewed transcript/model output.

## Role Surface Derivation Table

| Role | Primary shell | Primary leaf workflows | Secondary / review workflows | Story section |
| --- | --- | --- | --- | --- |
| Estimator | `Quote` | `Quote builder`, `Calls` | `Pricing Intelligence`, `Network` | [Estimator](operator_user_stories.md#estimator) |
| Dispatcher | `Operations` | `Dispatch` | `Quote` -> `Kent tenders`, `Network`, `Calls` | [Dispatcher](operator_user_stories.md#dispatcher) |
| Fleet / Operations Manager | `Operations` | `Operations diary`, `Planner`, `Dispatch`, `Fleet` | `Network`, `Pricing Intelligence` | [Fleet / Operations Manager](operator_user_stories.md#fleet--operations-manager) |
| Commercial Owner | `Quote`, `Pricing Intelligence` | `Quote builder`, `Kent tenders`, price/performance review | `Admin` review where exposed | [Commercial Owner](operator_user_stories.md#commercial-owner) |
| Compliance-Heavy / International Workflow Owner | `Quote`, `Operations` | `Quote builder`, `Kent tenders`, `Operations diary` | compliance/audit review surfaces | [Compliance-Heavy / International Workflow Owner](operator_user_stories.md#compliance-heavy--international-workflow-owner) |
| Labor Planner / Staff Coordinator | `Operations` | `Staff`, `Driver shifts`, `Payroll / Labor` | `Network` | [Labor Planner / Staff Coordinator](operator_user_stories.md#labor-planner--staff-coordinator) |
| Maintenance / Compliance Coordinator | `Operations` | `Fleet`, `Vehicle Maintenance` | `Network`, `Dispatch` | [Maintenance / Compliance Coordinator](operator_user_stories.md#maintenance--compliance-coordinator) |
| Warehouse / Crew | `Operations` | `Inventory` | `Dispatch`, `Operations diary` context | [Warehouse / Crew](operator_user_stories.md#warehouse--crew) |
| Inventory / Supplier Coordinator | `Operations` | `Inventory` | `Dispatch` context | [Inventory / Supplier Coordinator](operator_user_stories.md#inventory--supplier-coordinator) |
| Workforce Time Capture Coordinator | `Operations` | `Staff`, `Driver shifts` | future time-capture review surface | [Workforce Time Capture Coordinator](operator_user_stories.md#workforce-time-capture-coordinator) |
| Owner / Commercial / Finance-facing Manager | `Operations` | `Operations diary`, `Payroll / Labor analytics` | `Quote`, `Staff`, `Driver shifts` | [Owner / Commercial / Finance-facing Manager](operator_user_stories.md#owner--commercial--finance-facing-manager) |
| System / Rollout Admin | `Admin` | user admin, Kent/system governance, sync/cutover controls | `Operations`, `Network`, `Pricing Intelligence`, `Quote` | [System / Rollout Admin](operator_user_stories.md#system--rollout-admin) |
| Customer | no dashboard access | public-safe status projection only | tracking/receipt/support contract | [Service Blueprint](service_blueprint.md#customer-communication-matrix) |
| Support / Customer Service | deployment-specific | `Calls`, customer-safe job/status timeline | `Operations diary` customer-safe context | [Service Blueprint](service_blueprint.md#call-and-follow-up-matrix) |

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
