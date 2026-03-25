# Adaptive Learning Loop

This document defines the intended first-stage design for Corkysoft's bounded
operational learning loop.

## Purpose

Corkysoft should treat realised jobs, live route disruption signals, and market
conditions as inputs to a small learned policy state.

The goal is not to rewrite pricing after one unusual move. The goal is to make
slow, reviewable updates toward observed reality.

## Core Rule

The system should prefer:

- many realised jobs over one anecdotal job
- bounded parameter nudges over wholesale model rewrites
- explicit stored policy state over hidden spreadsheet drift
- operator-reviewed updates over silent automation

## Situational-Awareness Inputs

The intended upstream signals include:

- historical jobs and realised margin outcomes
- live and recent road-closure or traffic disruption feeds
- weather events and route-affecting conditions
- truck and driver execution outcomes
- market and seasonal pricing pressure

These inputs should inform pricing, ETA, and risk posture only when they can be
mapped to explicit, stored parameters.

## Learned Policy State

The first learned parameter set should stay intentionally small:

- `adaptive.lane_rate_per_m3`
- `adaptive.lane_eta_multiplier`
- `adaptive.weather_risk_multiplier`
- `adaptive.closure_delay_factor`
- `adaptive.truck_efficiency_score`
- `adaptive.driver_efficiency_score`
- `adaptive.seasonal_margin_uplift`

These values are policy inputs, not direct decisions. Quoting, dispatch, and
analytics layers may consume them once each integration path is documented and
tested.

## Update Model

Each learning cycle should follow this pattern:

1. Collect realised job outcomes and situational-awareness inputs.
2. Compare expected vs realised price, ETA, and margin behavior.
3. Propose bounded parameter deltas.
4. Clamp updates to safe per-cycle limits.
5. Store the accepted parameter state with audit-friendly descriptions.

This keeps the system contractive in practice: repeated updates should get
smaller as quoted behavior approaches realised behavior.

## First Implementation Boundary

The current implementation target is intentionally narrow:

- store and bootstrap the adaptive parameter defaults
- expose a typed helper for reading the current policy state
- expose a bounded numeric update helper for a single parameter

The current implementation does not yet guarantee:

- automatic ingestion from state/national road-closure feeds
- automatic weather-feed integration
- autonomous quote updates
- dashboard controls for approving or rejecting proposals
- historical learning jobs or scheduler infrastructure

## Next Delivery Steps

- map external closure/weather feeds to a common disruption schema
- calculate proposal deltas from realised jobs and route exceptions
- surface adaptive-policy review in dashboard/admin workflows
- add audit history for accepted and rejected policy proposals
- wire approved parameters into quoting, ETA, and lane analytics paths

## Situational-Awareness Implementation Status

- `analytics/situational_awareness.DisruptionEvent` records weather, traffic, and closure severity events plus optional source/location metadata.
- `analytics/situational_awareness.insert_disruption_event` populates the new `disruption_events` table defined in `analytics/db/schema.py`; the helper also normalizes timestamps and clamps severity to non-negative values.
- `analytics.situational_awareness.update_adaptive_policy_from_disruptions` summarizes recent severity totals, computes bounded targets for the weather, closure, and lane-ETA multipliers, and runs `apply_bounded_parameter_target` so policy state nudges remain auditable.
- Tests (`tests/test_situational_awareness.py`) verify severity aggregation, table persistence, and parameter updates.

Future work now focuses on exposing the proposed updates for operator review before affecting quotes or ETA guidance.
