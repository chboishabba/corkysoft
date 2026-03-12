# Kent AMS Integration Roadmap

Date baseline: 2026-03-12

This roadmap reflects the current implemented Kent AMS surface and the
remaining work needed before the adapter should be treated as production-ready.

## Current State

Implemented now:

- internal importer route: `POST /importers/kent-ams/{resource}`
- supported resources: `jobs`, `subcontractors`/`vendors`, `tenders`, `bids`,
  `awards`
- ranked tender queue: `GET /kent-ams/tenders/prioritized`
- calibration endpoint: `GET /kent-ams/tenders/calibration`
- policy config and override/audit endpoints for internal use
- Streamlit Kent operator queue with ranking and override capture
- fixture-backed contract smoke tests

Still provisional:

- field names and enums are based on expected Kent payloads, not a locked live
  export corpus
- current workflow is internal and governance-led, not production-rolled-out
- admin/operator separation, auth hardening, and live-payload validation remain
  incomplete

## Remaining Milestones

### Milestone 1: Contract Lock

Deliverables:
- 20-30 anonymized Kent payloads covering key lifecycle states
- final field dictionary and enum catalog
- documented auth and deployment contract

Acceptance gate:
- `docs/kent_ams_integration.md` is validated against live payloads rather than
  only fixtures
- unknown fields and unsupported enums have explicit handling rules

### Milestone 2: Governance and Workflow Hardening

Deliverables:
- operator/admin workflow split
- override governance and review cadence
- governed hard-block categories
- side-effect-free `dry_run` semantics

Acceptance gate:
- operator flow is decision-focused and admin settings are separated
- hard-block behavior is restricted to approved categories
- `dry_run` matches the documented contract

### Milestone 3: Entity and State Hardening

Deliverables:
- external ID bridge/link strategy
- enum translation tables
- replay-safe entity updates
- actionable validation failures

Acceptance gate:
- repeat ingest of identical payloads produces no duplicate logical entities
- validation failures are explicit and logged for review

### Milestone 4: Outbound and Operational Readiness

Deliverables:
- outbound quote recommendation contract
- dead-letter/replay runbook
- freshness and calibration review workflow
- deployment auth model

Acceptance gate:
- inbound change -> outbound recommendation path is testable
- operators and reviewers can explain why work was accepted or overridden

## Workstreams

1. Data contracts
- field mappings
- enum mappings
- payload versioning

2. Governance
- override policy
- review cadence
- admin/operator boundaries

3. Runtime hardening
- auth
- validation
- replay safety
- `dry_run` correctness

4. Quality
- fixture coverage
- contract tests
- dashboard regression coverage

## Immediate Next Actions

1. Validate the current provisional mapping against live Kent payloads.
2. Separate Kent admin controls from the operator queue in both docs and UI.
3. Finish auth and hard-block governance for mutating/internal endpoints.
4. Review calibration and override history with real operator usage before any
   deeper solver work.
