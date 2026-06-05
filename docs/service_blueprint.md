# Service Blueprint And Workflow Matrices

Last updated: 2026-06-04.

This is the canonical end-to-end service blueprint for Corkysoft. It ties
customer progress, internal user workflows, worker/crew execution, call
follow-up, and job completion into one matrix surface.

Related authority surfaces:

- [UI Role Coverage Matrix](ui_role_coverage_matrix.md) owns internal
  role-to-shell and role-to-leaf workflow mapping.
- [Operator User Stories](operator_user_stories.md) owns actor decisions,
  triggers, inputs, outputs, and actions.
- [Quote to Award Lifecycle](commercial_workflow_lifecycle.md) owns the
  commercial flow from quote to accepted work.
- [Operations Diary Workflow](operations_diary_workflow.md) owns the
  manager-facing day/week follow-through cockpit.
- [Known Bugs And Bad Cases](known_bad_cases.md) owns confirmed defects and
  promotion blockers.

## Current State

Corkysoft already has strong internal workflow fragments:

- quote and tender intake
- planner and `job_segments`
- dispatch and readiness review
- operations diary, tasks, invoice review, and subcontractor-bill review
- inventory pick / pack / load / custody execution
- worker-time capture and review
- call sessions, transcripts, notes, accepted extracted actions, and local
  append-only egress events

The major missing product model is not another dashboard tab. It is a clear
service contract that says:

- who owns each stage
- what customer-visible status exists
- what internal state is authoritative
- what notifications or receipts should exist
- what remains advisory until reviewed
- what closes a job

Customer notifications, public tracking pages, SMS/email reminders, and
proof-of-delivery receipts are **planned contract work**, not implemented
delivery behavior yet.

## Story Path Index

Use these story path IDs when attributing diagrams, shells, workflow
interactions, and acceptance stories. The matching flow diagram is
[diagrams/service_blueprint_flows.puml](diagrams/service_blueprint_flows.puml).

Rendered views:
[SVG](diagrams/service_blueprint_flows.svg) |
[PNG](diagrams/service_blueprint_flows.png)

![Service blueprint story paths](diagrams/service_blueprint_flows.svg)

| ID | Primary actor | User need / path | Shells and interactions | Gate |
| --- | --- | --- | --- | --- |
| SP-CUST-01 | Customer, Estimator / Call Operator | Inquiry is acknowledged and routed to a quote or missing-info follow-up. | `Quote` -> `Calls`, `Quote builder`; `call_follow_up` task | Contact preference, acknowledgement receipt, transcript remains advisory. |
| SP-CUST-02 | Customer, Estimator, Commercial Owner | Quote scope, assumptions, price, expiry, and approval next step are clear. | `Quote` -> `Quote builder`; `Pricing Intelligence` review | Reviewed quote state; no internal margin or admin notes exposed. |
| SP-CUST-03 | Customer, Dispatcher, Operations Manager | Accepted work becomes a booking with preparation expectations and a day-before reminder. | `Operations` -> `Dispatch`, `Planner`, `Operations diary`; `Quote` -> `Kent tenders` | Booking confirmation, consented reminder channel, reviewed plan. |
| SP-CUST-04 | Customer, Dispatcher / Network Operator | Customer can ask where the job is up to and receive a public-safe ETA/progress answer. | `Network`; `Operations` -> `Dispatch`; public-safe status projection | Freshness check, public-safe field classification, stale ETA downgrade. |
| SP-CUST-05 | Customer, Operations Manager / Support | Delivery, completion, receipt/POD, invoice timing, and support path are clear. | `Operations` -> `Operations diary`, `Inventory`, reconciliation; support timeline | Completion gate, proof data classification, receipt/audit event. |
| SP-CALL-01 | Caller, Call Operator, Support | Call transcript or model output becomes an accepted follow-up action, not automatic truth. | `Quote` -> `Calls`; `call_follow_up`; support/status response task | Operator acceptance, actor-bound write, API read authz. |
| SP-WORK-01 | Worker / Crew, Dispatcher, Labor Planner | Assignment is received, acknowledged, and tied to a segment/truck/role. | `Operations` -> `Dispatch`, `Staff`, `Driver Shifts` | Readiness checks, worker identity, acknowledgement receipt. |
| SP-WORK-02 | Warehouse / Crew, Inventory Coordinator, Dispatcher | Pick/pack/load, shortages, substitutions, and custody changes are captured. | `Operations` -> `Inventory`, `Dispatch`, `Operations diary` | Crew can flag; dispatcher/ops approves high-authority changes. |
| SP-WORK-03 | Worker, Time Capture Coordinator, Finance-facing Manager | Clock events and labor actuals become reviewed payroll-prep inputs. | `Operations` -> `Staff`, `Driver Shifts`, `Payroll / Labor` | Confidence threshold, anomaly review, period-close approval. |
| SP-MGR-01 | Fleet / Operations Manager | Day/week control moves work from planned to completed or exception-owned. | `Operations` -> `Operations diary`, `Planner`, `Dispatch`, `Fleet` | Segment, inventory, labor, invoice, bill, and task closure gate. |
| SP-ADMIN-01 | System / Rollout Admin | Auth, cutover, sync, and notification/customer-status governance are controlled. | `Admin`; governed review from other shells | Scoped credentials, receipts, no inert admin actions, secret-like payload rejection. |

## Lifecycle Path Crosswalk

Use these lifecycle IDs for service-flow, swimlane, and sequence diagrams that
need finer granularity than the story path IDs.

| Lifecycle ID | Stage | Primary story IDs | Primary actors | Canonical state / object gap |
| --- | --- | --- | --- | --- |
| SB-LC-01-INQUIRY-CALL | Inquiry / inbound call | SP-CUST-01, SP-CALL-01 | Customer, Estimator, Call Operator, Support | Needs contact preference, consent, acknowledgement receipt, and inquiry SLA. |
| SB-LC-02-QUOTE-PREP | Quote creation | SP-CUST-02 | Estimator, Commercial Owner | Quote record exists; customer-safe quote-send/expiry/acceptance messaging is not modeled. |
| SB-LC-03-QUOTE-FOLLOWUP | Quote sent / missing info | SP-CUST-01, SP-CUST-02, SP-CALL-01 | Estimator, Support, Customer | Needs `call_follow_up` with owner, due date, channel, and receipt. |
| SB-LC-04-BOOKING-AWARD | Booking / award | SP-CUST-03 | Customer, Dispatcher, Operations Manager, Commercial Owner | Needs booking confirmation and accepted-scope receipt. |
| SB-LC-05-PLAN-PREP | Planning / prep | SP-CUST-03, SP-MGR-01 | Operations Manager, Planner, Dispatcher, Customer | Needs customer prep checklist and day-before contract. |
| SB-LC-06-ASSIGN-READY | Assignment / readiness | SP-WORK-01, SP-ADMIN-01 | Dispatcher, Fleet/Ops, Labor, Inventory, Maintenance | Needs assignee notification, acknowledgement, and customer-safe assignment projection. |
| SB-LC-07-DAY-BEFORE | Day-before notification | SP-CUST-03, SP-CALL-01 | Dispatcher, Operations Manager, Customer, Call Operator | Needs reminder transport, consent gate, schedule, failure receipt, and escalation. |
| SB-LC-08-PICKUP-LOAD | Pickup / load | SP-WORK-02, SP-CUST-04 | Crew, Warehouse, Dispatcher, Customer | Needs customer-safe pickup/load milestone publishing. |
| SB-LC-09-TRANSIT-ETA | Transit / delay / progress | SP-CUST-04 | Dispatcher, Network Operator, Customer, Support | Needs tracking link, ETA freshness rules, and public-safe status projection. |
| SB-LC-10-DELIVERY-UNLOAD | Delivery / unload | SP-WORK-02, SP-CUST-05 | Crew, Dispatcher, Operations Manager, Customer | Needs delivery milestone and exception acknowledgement contract. |
| SB-LC-11-COMPLETE-CLOSE | Completion / closure | SP-CUST-05, SP-WORK-03, SP-MGR-01 | Operations Manager, Finance-facing Manager, Crew, Dispatcher | Needs promoted completion state, receipt/POD, and receipt-withheld reason. |
| SB-LC-12-SUPPORT-CASE | Support / complaint | SP-CUST-05, SP-CALL-01 | Customer, Support, Operations Manager, Dispatcher | Needs support case, severity, SLA, support replay, response receipt, and public-safe timeline. |

## Lifecycle Matrix

| Stage | Customer expectation | Internal owner | Primary UI / workflow | Authoritative internal state | Customer-visible output | Current gap |
| --- | --- | --- | --- | --- | --- | --- |
| Inquiry | Request is received and someone will respond. | Estimator / call operator | `Quote` -> `Calls`, `Quote builder` | Call session, client/contact record, inquiry note | Acknowledgement and preferred channel confirmation | No notification preference, acknowledgement, or inquiry SLA model. |
| Quote | Price, scope, assumptions, expiry, and next step are clear. | Estimator / Commercial Owner | `Quote` -> `Quote builder`; `Pricing Intelligence` | Quote record, pricing policy result, route/cost context | Quote sent/accepted/expired state | Quote lifecycle exists internally, but sent/accepted/expired customer messaging is not modeled. |
| Booking / Award | Move window and accepted scope are confirmed. | Dispatcher / Operations Manager | `Quote` -> `Kent tenders`; `Operations` -> `Dispatch` | Accepted job/tender, override audit, policy context | Booking confirmation and preparation checklist | No customer booking confirmation contract. |
| Planning | Customer can prepare access, inventory, site constraints, and contact availability. | Operations Manager / Planner | `Operations` -> `Planner`, `Operations diary` | Confirmed `job_segments`, site evidence, resource plan, diary tasks | Planned date/window and preparation instructions | No customer-facing prep checklist or day-before reminder model. |
| Assignment | Workers, trucks, inventory, and suppliers are assigned and ready. | Dispatcher / Fleet / Labor / Inventory coordinators | `Operations` -> `Dispatch`, `Fleet`, `Staff`, `Inventory` | Readiness checks, assignments, shortage/substitution state | Internal-only except high-level customer reassurance | No assignee acknowledgement or customer-safe assignment projection. |
| Day-before | Customer expects reminder, access confirmation, arrival window, and exception warning. | Dispatcher / Operations Manager | `Operations` -> `Operations diary`, `Dispatch` | Reviewed plan, open tasks, known exceptions | SMS/email/call reminder and preparation checklist | No transport, consent, reminder schedule, or failure receipt. |
| Pickup / Load | Customer expects crew arrival/load progress and delay notice if needed. | Dispatcher / Crew / Warehouse | `Operations` -> `Dispatch`, `Inventory` | Segment status, truck/crew state, inventory stages | Pickup/load milestone updates | Inventory stages exist internally; customer-safe milestone projection is missing. |
| Transit | Customer expects ETA and progress where visibility is allowed. | Dispatcher / Network operator | `Network`, `Operations` -> `Dispatch` | Reviewed telemetry, route progress, ETA/disruption state | Tokenized tracking link or ETA update | Public tracking contract is roadmap only; Live Network is internal MVP. |
| Delivery / Unload | Customer expects arrival/unload/completion state and exception capture. | Crew / Dispatcher / Operations Manager | `Operations` -> `Dispatch`, `Inventory`, `Operations diary` | Segment completion, custody handoff, exceptions, diary tasks | Delivery milestone and exception acknowledgement | No customer-facing delivery milestone or exception acknowledgement contract. |
| Completion | Customer expects receipt, proof/evidence where appropriate, invoice timing, and support path. | Operations Manager / Finance-facing Manager | `Operations` -> `Operations diary`, reconciliation | Closed tasks, invoice readiness, bill/subcontractor state, accepted notes/actions | Receipt / proof-of-delivery / invoice-ready summary | No explicit completed-job closure gate or customer receipt/POD model. |
| Support / Complaint | Customer expects a clear answer without internal leakage. | Support / Operations Manager | `Quote` -> `Calls`; `Operations` -> `Operations diary` | Customer-safe timeline, accepted notes/actions, reviewed exceptions | Support response and status timeline | No customer-safe support replay, complaint state, SLA, or response receipt. |

## Customer Communication Matrix

This matrix defines expected communication products. It does not mean those
products are implemented.

| Trigger | Default channel expectation | Required source state | Consent/privacy gate | Receipt/audit requirement | Failure behavior |
| --- | --- | --- | --- | --- | --- |
| Inquiry received | SMS/email/call acknowledgement based on preference | Call/inquiry record and contact preference | Customer contact consent and correct channel | Message queued/sent receipt | Task owner alerted if acknowledgement fails. |
| Quote ready | Email/SMS link or manual call follow-up | Reviewed quote summary, scope, assumptions, expiry | No internal margin, worker, or admin notes exposed | Quote sent receipt and expiry timer | Estimator follow-up task if unsent or unopened. |
| Booking accepted | Email/SMS booking confirmation | Accepted job/tender and move date/window | Public-safe booking fields only | Confirmation sent/accepted receipt | Dispatcher follow-up task. |
| Day-before reminder | SMS/email, with optional phone call for high-risk jobs | Reviewed plan, arrival window, access/prep checklist | Customer channel consent; no internal readiness detail leakage | Reminder sent receipt and delivery failure state | Dispatcher/operations alert before job day. |
| Delay / disruption | SMS/email/tracking-page alert | Reviewed ETA/disruption state | ETA uncertainty and freshness disclosed | Alert sent receipt and operator override note if manual | Downgrade to “contact office” when freshness is stale. |
| Pickup/load milestone | Tracking page, optional SMS for key milestones | Customer-safe segment/inventory milestone | No truck/worker private data unless approved | Milestone published receipt | Hold update if milestone is advisory/unreviewed. |
| Delivery/completion | SMS/email and receipt link | Reviewed delivery/unload/completion evidence | Proof data classified before exposure | Receipt/POD issued receipt | Manager task if completion cannot be published. |
| Complaint/support | Phone/email response and timeline note | Accepted support note/action, linked job/call | Customer-safe timeline only | Response receipt and owner/SLA | Escalate unresolved cases by SLA. |

## Internal Actor Stage Matrix

| Actor | Inquiry / quote | Planning | Day-before / execution | Completion / reconciliation | UI/workflow needs |
| --- | --- | --- | --- | --- | --- |
| Estimator | Capture inquiry, build quote, explain policy result. | Surface operational fit or missing site/compliance inputs. | Handoff to dispatcher after booking. | Review quote-vs-realized margin later. | `Quote builder`, quote send/expiry states, approval request state. |
| Dispatcher | Triage tender/booked work, prioritize, override with reason. | Confirm queue, handoff to Planner/Dispatch. | Own day-before/customer-facing operational updates, exceptions, delay notices. | Confirm dispatch outcome and support customer status questions. | `Dispatch`, customer milestone projection, exception owner queue, empty-state safe UI. |
| Fleet / Operations Manager | Review commercial/operational pressure when capacity tightens. | Own diary/plan coherence, segment readiness, truck/worker conflicts. | Own day/week control, escalations, reassignment, diary follow-through. | Promote job to completed/auditable when closure gates pass or exceptions are owned. | `Operations diary` as cockpit, closure gate, owner-routed exceptions. |
| Labor Planner / Staff Coordinator | Provide labor availability context where quote/booking needs it. | Maintain roster, worker roles, compliance, planned assignments. | Review worker-time captures and anomalies. | Prepare labor actuals for reconciliation/payroll prep. | Staff/Driver Shifts, period-close workflow, accepted actuals confidence. |
| Warehouse / Crew | Not primary. | Review requirements and prepare inventory. | Pick/pack/load, custody handoffs, substitution requests, exception flags. | Return/reconcile reusable assets or containers. | Worker-first task queue, mobile/simple acknowledgement, role-scoped routine actions. |
| Inventory / Supplier Coordinator | Provide stock/supplier feasibility where needed. | Allocate inventory and supplier context by segment. | Resolve shortages, substitution requests, supplier delays. | Reconcile supplier-side exceptions. | Inventory exception inbox, supplier follow-up tasks. |
| Maintenance / Compliance Coordinator | Provide readiness constraints for quotes/plans when needed. | Review due-soon/blocked truck and worker compliance. | Escalate blocked readiness before dispatch. | Close maintenance/compliance exceptions. | Fleet/maintenance cockpit, scheduling/actions that are wired or disabled. |
| Commercial Owner / Finance-facing Manager | Review quote policy, tender calibration, commercial overrides. | Review risk and approval where commercial impact is high. | Monitor major exception exposure. | Review invoice readiness, supplier bills, labor exposure, margin distortion. | Pricing Intelligence, Operations diary, approval/receipt surfaces. |
| System / Rollout Admin | Not primary. | Maintain user/source/system governance. | Keep imports/cutover/sync safe. | Preserve audit trail and rollout status. | Admin, authz, scoped credentials, no inert admin buttons. |

## Worker And Crew Execution Matrix

| Stage | Worker/crew input | Internal owner | Authority rule | Notification/ack gap |
| --- | --- | --- | --- | --- |
| Assignment published | Worker/truck/segment assignment acknowledged. | Dispatcher / Labor Planner | Assignment must pass readiness checks. | No assignee notification or acknowledgement workflow. |
| Pick / pack / load | Inventory stage and custody update. | Warehouse / Crew | Crew can advance routine execution states. | No worker-first simple task queue. |
| Substitution / shortage | Request substitution or flag exception. | Warehouse / Crew initially; Dispatcher/Ops approves | Crew cannot approve substitution; dispatcher/ops owns final decision. | Approval role must bind to authenticated identity. |
| Clock on/off | App, WhatsApp, voice/landline, or supervisor entry. | Worker / Time Capture Coordinator | Low-confidence or anomalous events require review. | No worker receipt or supervisor alert policy. |
| Exception during work | Report damage, delay, missing item, access issue, no-show. | Dispatcher / Operations Manager | Accepted notes/actions become operational truth. | Unified owner/SLA exception inbox is missing. |
| End of job | Confirm unload/delivery/return/storage state. | Crew / Dispatcher / Operations Manager | Completion needs reviewed segment, inventory, task, and exception state. | No completed-job closure gate or customer receipt. |
| Period close | Confirm reviewed actuals for payroll prep. | Labor Planner / Finance-facing Manager | Payroll analytics remain prep, not payroll execution. | No formal period-close approval/export governance. |

## Call And Follow-Up Matrix

| Flow | Current authority | Needed follow-up object | Customer/worker output | Gate |
| --- | --- | --- | --- | --- |
| Inbound quote call | Call session/routing is compiled state; transcript advisory. | `call_follow_up` linked to client/quote/job with owner and due time. | Inquiry acknowledgement or quote task. | API read authz and actor-bound writes. |
| Missing quote information | Pending action/advisory transcript until accepted. | Missing-info task with required fields and channel. | Customer request for missing information. | Accepted action or operator note. |
| Day-before confirmation call | Operator note is authoritative; transcript is advisory. | Reminder/confirmation task and receipt. | Confirmation, access/prep checklist. | Contact consent and customer-safe fields. |
| Worker time call-in | High-confidence event may be accepted; anomalies pending review. | Review task for low-confidence worker time. | Worker receipt and supervisor alert where needed. | Worker identity/confidence threshold. |
| Progress query | Internal job/telemetry state is not automatically public. | Support/status response task. | Customer-safe status response. | Public-safe projection and freshness check. |
| Complaint/support | Accepted note/action is authoritative. | Complaint case with severity, owner, SLA, response receipt. | Response and support timeline. | No raw transcript as authority. |

## Job Completion Gate

A job should not be treated as completed/auditable just because the route day
has passed. Completion should be a promoted state with evidence:

- all planned segments are delivered, cancelled, or explicitly exceptioned
- inventory/custody stages are delivered, returned, or exceptioned
- worker-time and shift actuals are reviewed enough for payroll-prep confidence
- customer invoice status is `ready_to_invoice`, `invoiced`, or explicitly
  blocked with a named owner
- subcontractor/supplier bill status is reconciled, not expected, or
  exceptioned with aging visibility
- diary tasks are closed or carried forward with owner and due date
- customer receipt/POD/support summary is issued or intentionally withheld with
  reason
- observer/outbox or downstream summary is emitted where required

## Story Backlog Derived From The Matrices

Customer-side:

- As a customer, I want acknowledgement after inquiry so I know the request was
  received and which channel will be used.
- As a customer, I want quote scope, assumptions, price, and expiry in one
  message so I can approve or ask questions.
- As a customer, I want booking confirmation and a day-before reminder so I can
  prepare access, inventory, and contact availability.
- As a customer, I want ETA/progress updates from a public-safe tracking
  projection so I can understand where things are without seeing internal notes.
- As a customer, I want a completion receipt or proof-of-delivery summary so I
  know what was completed and how to contact support.

Internal users:

- As a dispatcher, I want customer-visible milestones generated only from
  reviewed operational state so updates do not leak internal or advisory data.
- As an operations manager, I want one closure gate for completed jobs so
  dispatch, diary, inventory, labor, invoice, and supplier states converge.
- As a warehouse/crew user, I want a simple task queue for pick/pack/load and
  exceptions so I do not need to navigate admin-heavy inventory screens.
- As a labor planner, I want low-confidence worker-time events routed to a
  review queue before they affect payroll-prep truth.
- As support staff, I want a customer-safe timeline separate from internal
  dashboard replay so I can answer “where is my job up to?” without leaking
  private notes or worker data.

Governance:

- As a system admin, I want notification sends and customer-visible status
  changes to produce receipts so delivery failures and manual overrides are
  auditable.
- As a commercial owner, I want customer communications to preserve scope,
  assumptions, and compliance caveats so quote and delivery promises remain
  defensible.

## Promotion Gates

Before customer-side communications or tracking can be promoted:

- sensitive API reads must be authenticated and scoped
- customer-visible fields must be classified as public-safe,
  customer-confidential, internal-only, or admin-only
- notification channel preference and consent must be recorded
- message/task delivery receipts must be stored
- stale telemetry must downgrade ETA/progress claims
- transcript/model outputs must remain advisory until accepted
- role-bound actor identity must replace free-text actor fields for
  high-authority writes
