# Operator User Stories

This document defines the actor-based workflows Corkysoft must support today.
It is intentionally decision-focused rather than feature-focused.

## Estimator

Primary surfaces:
- `Quote builder`

Trigger:
- A customer or incumbent system requests a quote or tender response.

Primary decisions:
- Is this work commercially acceptable?
- Does the current quote clear policy thresholds?
- Does the route look operationally workable?

Required inputs:
- origin and destination
- move date
- volume or size estimate
- estimated sale price or target margin
- known modifiers and site constraints
- paperwork/compliance requirements when the job is international,
  insurance-heavy, tender-driven, or otherwise governance-sensitive

Expected system outputs:
- quoted amount and cost breakdown
- profitability policy pass/fail state
- reasons when policy does not pass
- operational fit signals that may affect confidence
- visibility into missing requirement/proposal/governance state when paperwork
  or compliance completeness materially changes quote quality

Operator actions:
- accept the recommendation
- adjust margin/price inputs
- record a manual override when commercial context justifies it
- flag missing paperwork/compliance structure for follow-up rather than letting
  it stay implicit

## Dispatcher

Primary surfaces:
- `Dispatch`
- `Kent tenders`

Trigger:
- A tender or booked job enters the day-to-day operating queue.

Primary decisions:
- Which tenders/jobs deserve immediate attention?
- Which items are safe to defer?
- Which exceptions require an explicit override?
- When does an external handoff still require a dispatch snapshot?

Required inputs:
- ranked tender/job queue
- policy status
- hard-block vs overrideable flags
- route fit and spare-capacity context
- segment readiness, assigned trucks/workers, and inventory context

Expected system outputs:
- ordered queue
- clear flag types
- explicit override reason capture
- audit history for prior overrides
- dispatch snapshot export when an external handoff is still required

Operator actions:
- pursue
- review later
- defer
- override with reason and note when needed
- export a dispatch snapshot when native execution must be shared externally

## Fleet / Operations Manager

Primary surfaces:
- `Operations diary`
- `Operations`
- `Dispatch`
- `Fleet`

Trigger:
- Capacity tightens, SLA risk rises, or peak-season conditions require intervention.

Primary decisions:
- Are assignments still operationally coherent?
- Are conflict and readiness policies still appropriate?
- Does the current plan need truck/worker reassignment or an override?

Required inputs:
- assignment conflicts
- readiness warnings and blocks
- observed capacity pressure
- current job/segment plan
- diary tasks and unresolved operational follow-ups
- customer invoice state and subcontractor-bill exceptions
- plan-vs-actual truck/staff/supplier usage signals
- override history and cutover status

Expected system outputs:
- segment-based planning truth
- a day/week manager cockpit that links jobs, tasks, usage, and reconciliation
- visibility into blocked and due-soon items
- operator/admin separation where governance is required
- auditable policy and override changes

Operator actions:
- review the day/week diary to decide what needs action now
- add, remove, or re-scope operational diary tasks
- create or update segments
- assign trucks and workers
- drill into job usage, vehicle usage, staff usage, and invoicing/reconciliation
- approve or escalate operational overrides
- review capacity pressure and planning drift

## Operations Manager Diary / Reconciliation

Primary surfaces:
- `Operations diary`

Secondary surfaces:
- `Planner`
- `Dispatch`
- `Payroll / Labor analytics`

Trigger:
- A manager needs one day/week view of work in motion, required vs utilized
  resources, and invoice/bill follow-through.

Primary decisions:
- Which jobs need attention today or this week?
- Which jobs are missing required trucks, workers, supplier coverage, or
  follow-up tasks?
- Which completed jobs are ready for customer invoicing?
- Which subcontractor or third-party bills match the known operational truth,
  and which need escalation?
- Which received third-party bills have been sitting unresolved for too long,
  and which job/account is carrying that exposure?

Required inputs:
- jobs and `job_segments`
- diary tasks linked to a job, segment, day, or week
- truck and worker assignments
- imported/reviewed actual labor signals
- supplier and subcontractor context
- customer invoice review state
- subcontractor bill review state

Expected system outputs:
- day and week diary views
- job-level required vs utilized summaries
- vehicle and staff usage drill-through
- clear invoice and bill exception categories
- operational follow-up task list that survives beyond raw segment planning
- unresolved supplier-exposure summary with aging and latency cues

Operator actions:
- open the diary from Planner or Dispatch context
- review jobs by day or week
- add/remove/update diary tasks
- inspect vehicle usage for a job
- inspect staff usage for a job
- review customer invoice readiness
- reconcile subcontractor or supplier bills against job truth
- open the most over-aged unresolved supplier rows and resolve or escalate them

## Commercial Owner

Primary surfaces:
- `Quote builder`
- `Kent tenders`
- `Kent admin`
- analytics tabs

Trigger:
- Periodic review of pricing performance, tender conversion, and margin quality.

Primary decisions:
- Are quotes and tenders being prioritized correctly?
- Are current thresholds too strict or too loose?
- Which corridors or customer segments need intervention?
- Should a rollout promotion be approved from a commercial-risk perspective?

Required inputs:
- calibration metrics
- margin quality by score band
- override trends
- lane and route performance summaries
- rollout approval state when promotions are requested
- requirement/proposal/governance gaps for compliance-heavy work

Expected system outputs:
- explainable policy framework
- confidence that operators can override within governance
- documented workflow from quote to awarded work
- auditable approval history for governed promotions
- visibility into where the current product still lacks explicit paperwork,
  insurance, tender, customs, or audit-state modeling

Operator actions:
- approve policy changes
- approve or reject rollout promotions
- request calibration/tuning work
- decide where to invest operational attention
- prioritize requirement/proposal/governance formalization where external
  paperwork quality is currently driving margin or risk

## Compliance-Heavy / International Workflow Owner

Primary surfaces:
- `Quote builder`
- `Kent tenders`
- `Operations diary`

Trigger:
- A job, tender, or customer workflow depends on international shipping,
  customs, insurance, tender compliance, or audit-heavy paperwork.

Primary decisions:
- Are the job's requirement states explicit enough to price and accept safely?
- Is the proposal/document package complete enough to submit or execute?
- Is the governance/evidence trail sufficient for later dispute, insurer,
  subcontractor, or customer review?

Required inputs:
- commercial quote/tender context
- paperwork and compliance checklist state
- known external authority requirements
- job/segment operational truth
- invoice/bill review history when work is already complete

Expected system outputs:
- explicit requirement completeness state
- proposal/document-package gaps
- governance/evidence gaps tied to the job
- clear handoff into operations diary and reconciliation when execution starts

Operator actions:
- halt or escalate work that lacks required paperwork/governance coverage
- request missing requirements/proposal data
- review late supplier/subcontractor paperwork against operational truth
- hand complete work forward into quoting, dispatch, diary review, or invoicing

## Labor Planner / Staff Coordinator

Primary surfaces:
- `Staff`
- `Driver shifts`
- `Operations`

Trigger:
- Workers need to be rostered, reviewed, or reconciled against planned work.

Primary decisions:
- Which workers should be assigned to which segments?
- Does the planned roster match the imported shift feed?
- Are role and compliance assignments sufficient for upcoming work?

Required inputs:
- planned segment assignments
- worker roster and active status
- imported `VEHICLE_DRIVER` shift feed
- worker roles, compliances, and readiness alerts
- operations diary context when a manager wants staff usage reviewed from a
  specific job or day

Expected system outputs:
- native planned labor roster
- reconciliation between planned and imported shifts
- worker-level role/compliance visibility
- planned truck and segment context per worker

Operator actions:
- maintain roster details
- review planned segment assignments per worker
- open staff-usage drill-down from the diary when job-level mismatches exist
- reconcile imported shifts against the native plan
- assign or update worker roles and compliances
- review accepted labor actuals quality before payroll-prep truth is trusted
- resolve worker/job/truck/time mismatches that affect both operations and pay confidence

## Owner / Commercial / Finance-facing Manager

Primary surfaces:
- `Operations diary`
- future `Payroll / Labor analytics`

Secondary surfaces:
- `Staff`
- `Driver shifts`
- `Quote builder`

Trigger:
- Labor cost patterns, payroll exposure, or workforce exceptions need review.
- Customer invoice or subcontractor-bill discrepancies need job-level review.

Primary decisions:
- How much should labor/pay likely cost over a selected period or date range?
- Which workers, teams, jobs, clients, or corridors drive unusual labor cost?
- Are overtime, absence, or unresolved review patterns becoming a financial or operational risk?

Required inputs:
- reviewed labor actuals
- planned labor roster
- imported shift reconciliation
- accepted worker-time events
- anomaly and review backlog
- labor cost rollups by worker/team/job/client/corridor
- job-level invoice and subcontractor-bill review state

Expected system outputs:
- pay forecasting by worker/team/date range
- overtime and hours-worked distributions
- labor cost distributions and variance from plan
- absence / sick-day summaries
- payroll-prep confidence signals
- aggregate-first insight with justified drill-down to individual workers
- job-level visibility into whether invoice and bill reviews match operational
  truth

Operator actions:
- review labor trends and outliers
- forecast payroll exposure
- open diary-led reconciliation when a job-level exception needs explanation
- inspect unresolved anomalies that reduce confidence
- prepare export-ready labor summaries for external payroll/accounting tools
- follow up only where patterns imply cost, staffing, or review action

## Maintenance / Compliance Coordinator

Primary surfaces:
- `Fleet`
- `Vehicle maintenance`

Trigger:
- A truck or worker approaches expiry, service due date, or blocked readiness state.

Primary decisions:
- Which vehicles or workers are due soon versus blocked now?
- Which items require immediate intervention before assignment?
- Are readiness warning and blocking policies still appropriate?

Required inputs:
- rego expiry
- COI due dates
- service due dates
- worker compliance expiry
- current planned assignments and affected segments

Expected system outputs:
- maintenance/compliance cockpit with due-soon and blocked items
- vehicle-level planning context
- repair and maintenance history
- clear separation between warning, blocked, and overrideable states

Operator actions:
- review due-soon and blocked items
- coordinate repairs or compliance remediation
- adjust readiness policy only when governance requires it
- inform operations when assignments are impacted

## Warehouse / Crew

Primary surfaces:
- `Inventory`

Secondary surfaces:
- `Dispatch`
- `Operations diary`

Trigger:
- Planned work is moving from requirement/allocation into physical execution.

Primary decisions:
- Has the required stock actually been picked, packed, and loaded?
- Is the current custody/location state accurate?
- Has a shortage or mismatch been discovered that needs dispatcher review?

Required inputs:
- segment requirement lines
- allocated inventory and container context
- current custody/location state
- active truck / site context for the leg
- shortage and exception state

Expected system outputs:
- routine pick / pack / load progression
- current custody handoff state
- visible shortage and execution exceptions
- explicit substitution requests when planned stock is not available

Operator actions:
- record pick / pack / load progression
- record custody/location changes
- flag execution exceptions
- request substitution when the planned stock cannot be fulfilled as-is
- use barcode or QR-assisted capture where available rather than retyping item/container identifiers

## Inventory / Supplier Coordinator

Primary surfaces:
- `Inventory`
- `Dispatch`

Trigger:
- Inventory, supplier coordination, or segment allocations need review.

Primary decisions:
- Is enough stock allocated and executable for planned segments?
- Which suppliers or stock movements are missing or late?
- Are there unresolved inventory exceptions that block execution?
- Does a warehouse-raised shortage or substitution request need follow-up?

Required inputs:
- segment-linked inventory coordination
- supplier list and provenance
- recent movement events
- current stock balances and exceptions
- warehouse execution state from `Inventory`
- job/segment execution state from Dispatch
- inventory architecture context: container-based, consumable, reusable asset, or serialized/tagged workflow

Expected system outputs:
- segment-level inventory allocation visibility
- supplier context linked to planned work
- movement history and exception queue
- clear view of stock reserved, allocated, and execution-ready by segment
- clear view of required vs allocated vs shortage state
- location/custody truth that can distinguish depot, truck/container, in-transit, site, returned/storage, and exception contexts
- explicit substitution requests raised by warehouse/crew and dispatcher/ops decisions against those requests

Operator actions:
- import or review supplier data
- record movement events
- reserve or release stock
- allocate inventory to planned segments
- review warehouse execution exceptions
- review shortages and substitution requests
- reconcile inventory exceptions
- review shortages and architecture-specific stock risks before work is confirmed
- follow accommodation/availability pressure when it affects operational feasibility on remote or peak-period work

## Workforce Time Capture Coordinator

Primary surfaces:
- `Staff`
- `Driver shifts`

Secondary surfaces:
- future time-capture intake/review surface

Trigger:
- workers need clock-on or clock-off capture across mixed communication channels

Primary decisions:
- was the event captured cleanly enough to accept automatically?
- does the event need supervisor review?
- does the source channel affect confidence or audit requirements?

Required inputs:
- worker roster
- known phone numbers or employee numbers
- planned assignments and shift windows
- incoming app, WhatsApp, or voice/landline events

Expected system outputs:
- auditable clock-on / clock-off events
- source-channel visibility
- confidence or review status where capture was ambiguous
- linkage back to worker, shift, and operational context

Operator actions:
- review or correct low-confidence events
- confirm worker identity and event time
- reconcile captured events against planned labor
- reduce unresolved anomalies before period close
- ensure accepted events are suitable for payroll-prep use

Substitution authority:
- warehouse / crew requests substitution
- dispatcher / operations approves or rejects substitution
- manager handles escalation or repeated drift, not routine approval

## System / Rollout Admin

Primary surfaces:
- `Fleet`
- admin/import areas in `Staff`, `Inventory`, and `Driver shifts`

Trigger:
- Operational source data must be refreshed, or spreadsheet cutover state must be governed.

Primary decisions:
- When should shared workbook data be resynced?
- Is a workflow still sheet-primary, dual-run, native-primary, or fallback-only?
- Has the approval chain completed for a promotion?

Required inputs:
- shared workbook references and import status
- cutover metrics and checklist state
- review, drill, fallback-use, and snapshot events
- approval state for promotion requests

Expected system outputs:
- successful sync/import feedback
- explicit cutover status per workflow
- event-backed audit trail for reviews, drills, approvals, and transitions
- rollback instructions and notes tied to each workflow

Operator actions:
- sync the shared operations workbook
- import supplier, staff, and driver-shift source sheets
- record reviews, drills, fallback use, and snapshot issuance
- request, approve, reject, or apply workflow promotions according to governance
