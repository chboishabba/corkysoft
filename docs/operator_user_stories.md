# Operator User Stories

This document defines the minimum actor-based workflows Corkysoft must support
today. It is intentionally decision-focused rather than feature-focused.

## Estimator

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

Expected system outputs:
- quoted amount and cost breakdown
- profitability policy pass/fail state
- reasons when policy does not pass
- operational fit signals that may affect confidence

Operator actions:
- accept the recommendation
- adjust margin/price inputs
- record a manual override when commercial context justifies it

## Dispatcher

Trigger:
- A tender or booked job enters the day-to-day operating queue.

Primary decisions:
- Which tenders/jobs deserve immediate attention?
- Which items are safe to defer?
- Which exceptions require an explicit override?

Required inputs:
- ranked tender/job queue
- policy status
- hard-block vs overrideable flags
- route fit and spare-capacity context

Expected system outputs:
- ordered queue
- clear flag types
- explicit override reason capture
- audit history for prior overrides

Operator actions:
- pursue
- review later
- defer
- override with reason and note when needed

## Fleet / Operations Manager

Trigger:
- Capacity tightens, SLA risk rises, or peak-season conditions require
  intervention.

Primary decisions:
- Are thresholds or reason-code governance still appropriate?
- Are operators overriding too often or for the wrong reasons?
- Should temporary peak-period policy changes be approved?

Required inputs:
- override history
- policy defaults
- loss-alert frequency
- observed capacity pressure

Expected system outputs:
- operator/admin separation
- auditable policy changes
- visibility into override drift and review candidates

Operator actions:
- adjust policy defaults
- manage active override reasons
- review override patterns and policy misuse

## Commercial Owner

Trigger:
- Periodic review of pricing performance, tender conversion, and margin quality.

Primary decisions:
- Are quotes and tenders being prioritized correctly?
- Are current thresholds too strict or too loose?
- Which corridors or customer segments need intervention?

Required inputs:
- calibration metrics
- margin quality by score band
- override trends
- lane and route performance summaries

Expected system outputs:
- explainable policy framework
- confidence that operators can override within governance
- documented workflow from quote to awarded work

Operator actions:
- approve policy changes
- request calibration/tuning work
- decide where to invest operational attention
