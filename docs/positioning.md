# Corkysoft Positioning and Competitive Landscape

## Market Categories

### Traditional removals software (examples)
- MoveWare
- SmartMoving
- MoveitPro
- Elromco
- Movegistics

Typical strengths:
- CRM, job scheduling, dispatch, contracts, invoicing, and claims workflows.

Typical gaps:
- Route profitability analytics.
- Statistical pricing and market benchmarking.
- Lane performance analysis.
- Dynamic quoting tied to margin outcomes.

### Fleet management platforms (examples)
- Samsara
- Verizon Connect
- Fleetio
- Teletrac Navman

Typical strengths:
- GPS tracking, driver behavior, maintenance, fuel analytics.

Typical gaps:
- Move-specific pricing context (m3, packing modifiers, lane profitability).

### Route optimization platforms (examples)
- Routific
- OptimoRoute
- Onfleet
- Locus

Typical strengths:
- Route planning, sequencing, delivery windows, dispatch.

Typical gaps:
- Profit optimization and pricing intelligence.

## Corkysoft Differentiation

Core focus:
- Decision support for removals operators, with pricing intelligence and
  profitability analytics as the current strongest surface.

Unique capabilities in this domain:
- Route profitability analytics.
- $/m3 statistical pricing.
- Lane performance benchmarking.
- Break-even visualization and margin overlays.
- Backhaul detection and discount recommendations.
- Pricing recommendation engine tied to corridor outcomes.
- Growing operational evidence layer linking planning, usage review, and
  invoice/bill reconciliation around the same job truth.

Strategic framing:
- MoveWare-class systems are typically systems of record.
- Corkysoft should become a system of decision.
- That means the value is not just storing jobs, but helping operators decide
  what to quote, accept, plan, staff, escalate, and reconcile.

## Strategy: Integrate, Do Not Replace

The product should sit on top of existing operational systems rather than replace them.

Corkysoft should own:
- Pricing intelligence and profitability analytics.
- Route profitability and corridor/lane rollups.
- Recommendation logic (pricing, backhaul, margin risk).
- Decision workflows where the commercial/operational question matters more
  than raw record storage.

Corkysoft should integrate:
- CRM and job management (e.g., MoveWare, SmartMoving, HubSpot, Zoho).
- Dispatch and route optimization (e.g., OptimoRoute, Routific, Onfleet).
- Accounting systems (e.g., Xero, MYOB, QuickBooks).
- Fleet tracking (e.g., Samsara, Teletrac, Verizon Connect).

Integration paths:
- CSV import/export.
- API sync where available.
- Webhooks for incremental updates.

Selective internalization:
- Where incumbent systems are weak on reasoning, Corkysoft should gradually
  formalize requirements, proposal, and governance state instead of leaving it
  as undocumented operator memory.
- The clearest current gap is international/compliance-heavy work: paperwork,
  insurance, tender, customs, and audit obligations need explicit structure
  even if source-of-record documents still live in external systems for now.

## Primary Users

The current product should be understood through the decisions it supports:

- **Estimator**
  - decides whether a quote is commercially acceptable
  - uses quote pricing outputs plus profitability policy state
- **Dispatcher**
  - decides which tenders and jobs deserve immediate effort
  - uses ranked queues, hard-blocks, overrideable flags, and route-fit signals
- **Fleet / operations manager**
  - decides whether operator policy defaults and override governance remain appropriate
  - uses override history, loss-alert volume, and capacity pressure
- **Commercial owner**
  - decides whether thresholds, calibration, and corridor strategy are working
  - uses margin quality, conversion, and calibration outcomes
- **Operations / finance-facing manager**
  - decides whether completed work is operationally and financially coherent
  - uses the operations diary, usage review, and invoice/bill reconciliation
- **Compliance-heavy / international workflow owner** (future emphasis)
  - decides whether paperwork and governance state are complete enough to quote,
    accept, or invoice sensitive work
  - uses requirements/proposal/governance status rather than free-text memory

See `docs/operator_user_stories.md` for the actor-level workflows and
`docs/commercial_workflow_lifecycle.md` for the end-to-end lifecycle.

## Minimum Data Needed for Core Intelligence

Required fields:
- job_date
- origin
- destination
- volume_m3
- quoted_price
- crew_cost
- truck_cost
- distance_km
- duration_hr

Derived metrics:
- price_per_m3
- price_per_km
- margin_percent

## Product Positioning Statement

"Decision intelligence for removals companies."
