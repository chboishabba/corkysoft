# Quote to Award Lifecycle

This document defines the current Corkysoft commercial workflow across quoting,
Kent tender triage, overrides, and downstream operational execution.

## 1. Quote Creation

Input:
- origin, destination, date, volume, modifiers, target margin

System behavior:
- calculate route and cost estimate
- generate a proposed sell price
- evaluate quote profitability policy against the proposed quote
- attach operational fit signals such as spare-capacity context
- surface missing requirements/proposal/governance state when the work is
  international, insurance-heavy, tender-driven, or otherwise compliance-heavy

Operator interpretation:
- policy pass means the current quote clears the configured commercial rule
- policy fail does not block quoting, but requires deliberate review
- loss alert means the quote is below the configured loss floor

## 2. Tender / Queue Triage

Input:
- Kent tender payload or other queued commercial opportunity

System behavior:
- score and rank tenders
- separate hard-block conditions from overrideable flags
- keep non-matching tenders visible but lower priority

Operator interpretation:
- hard-block means safety/legal/compliance stop
- overrideable flags mean commercial or operational concerns that can be
  accepted with explicit reason logging

## 3. Override Decision

When to trust the system:
- profitability passes
- no hard block exists
- operational signals are neutral/favorable

When to override:
- customer retention
- backhaul or positioning value
- known route/site context not represented in the system
- temporary capacity strategy during peak periods

Override requirements:
- operator identity
- reason code
- optional free-text note
- resulting audit event

## 4. Awarded / Accepted Work

System behavior:
- persist job/tender state
- carry profitability and operational signals into downstream job context
- preserve the audit trail that explains why work was accepted
- expose the accepted work to the `Operations diary` so managers can review the
  day/week plan, resource usage, and invoice/bill follow-through in one place
- keep requirement/proposal/governance completeness visible for jobs whose
  paperwork or compliance state will matter later during execution,
  subcontractor review, insurer review, or customer invoicing

## 5. Operational Execution

Downstream use:
- dispatch can see policy context and operational fit
- dispatch should operate from the native job-centric board rather than from the
  source spreadsheets directly
- dispatch/planning should assign trucks and staff at the `job_segments` level
- `job_segments` are an internal planning artifact; the intended operator workflow is a higher-level planning surface that proposes legs from route, site, and corridor context rather than requiring manual segment record entry
- operational spreadsheets inform availability/readiness but do not own final assignment truth
- stock and supplier coordination should follow the same `job_segments` so segment-level execution is coherent
- future optimization work must respect the original governance boundaries
- future planner UX should expose route/site context, traffic/routing considerations, and corridor familiarity before resource allocation is confirmed
- per-route and per-segment profitability should be visible during planning, not only in downstream analytics or tender triage
- post-hoc calibration can compare predicted vs realized margin outcomes
- customer invoicing and subcontractor bill review should both happen against
  the same job/segment operational truth rather than through disconnected
  spreadsheets or inbox-driven review
- future international/compliance-heavy workflows should not treat paperwork as
  detached admin clutter; requirement state, proposal/document completeness, and
  governance evidence should sit beside quote/award/dispatch decisions

Planner interaction reference:
- [Planner Interaction Model](planner_interaction_model.md)

## 6. Review Loop

Required periodic checks:
- override frequency and reason distribution
- loss-alert acceptance rate
- calibration by score band
- whether quote and tender thresholds still match operator reality
- whether completed jobs are invoice-ready or stuck behind operational gaps
- whether subcontractor or third-party bills reconcile cleanly against known
  usage, task history, and supplier context
- whether compliance-heavy jobs are blocked by missing requirement,
  proposal/document, or governance evidence states
