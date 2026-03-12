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

## 5. Operational Execution

Downstream use:
- dispatch can see policy context and operational fit
- dispatch/planning should assign trucks and staff at the `job_segments` level
- operational spreadsheets inform availability/readiness but do not own final assignment truth
- future optimization work must respect the original governance boundaries
- post-hoc calibration can compare predicted vs realized margin outcomes

## 6. Review Loop

Required periodic checks:
- override frequency and reason distribution
- loss-alert acceptance rate
- calibration by score band
- whether quote and tender thresholds still match operator reality
