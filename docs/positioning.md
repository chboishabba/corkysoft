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
- Pricing intelligence and profitability analytics for removals operators.

Unique capabilities in this domain:
- Route profitability analytics.
- $/m3 statistical pricing.
- Lane performance benchmarking.
- Break-even visualization and margin overlays.
- Backhaul detection and discount recommendations.
- Pricing recommendation engine tied to corridor outcomes.

## Strategy: Integrate, Do Not Replace

The product should sit on top of existing operational systems rather than replace them.

Corkysoft should own:
- Pricing intelligence and profitability analytics.
- Route profitability and corridor/lane rollups.
- Recommendation logic (pricing, backhaul, margin risk).

Corkysoft should integrate:
- CRM and job management (e.g., MoveWare, SmartMoving, HubSpot, Zoho).
- Dispatch and route optimization (e.g., OptimoRoute, Routific, Onfleet).
- Accounting systems (e.g., Xero, MYOB, QuickBooks).
- Fleet tracking (e.g., Samsara, Teletrac, Verizon Connect).

Integration paths:
- CSV import/export.
- API sync where available.
- Webhooks for incremental updates.

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

"Profit intelligence for removals companies."
