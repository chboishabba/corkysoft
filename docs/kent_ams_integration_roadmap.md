# Kent AMS Integration Roadmap

Date baseline: 2026-03-11

This roadmap plans Corkysoft integration with Kent AMS in delivery phases with
clear acceptance gates.

## Objectives

- Build a reliable inbound Kent -> Corkysoft sync for moves, assignees, and
  shipment context.
- Produce stable outbound quote recommendation payloads with auditability.
- Ensure idempotency, replay safety, and testable contracts.

## Phase Plan

### Phase 0: Discovery and Contract Lock (target: 2026-03-11 to 2026-03-18)

Deliverables:
- Real Kent sample payload corpus (happy path + edge cases).
- Final field dictionary and enum catalog.
- Auth method and environment contract.

Acceptance gate:
- `docs/kent_ams_integration.md` mapping table validated against real payloads.
- Unknown/optional fields explicitly marked.

### Phase 1: Inbound Foundation (target: 2026-03-18 to 2026-04-01)

Deliverables:
- Ingest adapter for Kent payload normalization.
- Idempotent upsert by external identifiers.
- External link mapping table design and migration plan.

Acceptance gate:
- Repeat ingest of the same payload produces zero duplicate logical entities.
- Schema validation failures are actionable and logged.

### Phase 2: Core Entity Sync (target: 2026-04-01 to 2026-04-15)

Deliverables:
- Client/assignee sync into `clients`.
- Move/job sync into `jobs` and/or `historical_jobs`.
- Shipment and vendor sync with status enum mapping.

Acceptance gate:
- Contract tests pass for entity creation/update/deletion-safe behavior.
- Enum translation table documented and versioned.

### Phase 3: Pricing and Outbound Contract (target: 2026-04-15 to 2026-04-29)

Deliverables:
- Quote recommendation payload builder from Corkysoft outputs.
- Reason code and component breakdown support.
- Outbound retry + dead-letter behavior.

Acceptance gate:
- Golden payload tests for outbound format and required fields.
- End-to-end scenario from inbound move update to outbound quote recommendation.

### Phase 4: Operational Hardening (target: 2026-04-29 to 2026-05-13)

Deliverables:
- Sync observability (success/error counts, lag, replay count).
- Replay and backfill runbook.
- On-call troubleshooting guide.

Acceptance gate:
- Simulated failure drills demonstrate safe recovery and replay.
- SLA dashboard indicates freshness and failure rates.

### Phase 5: Pilot and Rollout (target: 2026-05-13 to 2026-05-27)

Deliverables:
- Controlled pilot with selected Kent workflows.
- Production rollout checklist and go/no-go criteria.
- Post-launch monitoring and issue triage plan.

Acceptance gate:
- Pilot sign-off from business and technical stakeholders.
- No unresolved P1 data-integrity defects.

## Workstreams

1. Data contracts: field mappings, enum mappings, payload versioning.
2. Integration runtime: adapter, scheduler/webhook path, retries, dead-letter.
3. Persistence and audit: idempotent upsert, external ID links, change logs.
4. Quality: contract tests, fixture library, replay tests.
5. Operations: dashboards, alerting, runbooks.

## Risks and Mitigations

- Risk: Kent payload shape drift.
  - Mitigation: versioned schemas + strict validator with compatibility layer.
- Risk: enum mismatch creates workflow regressions.
  - Mitigation: explicit translation table + unknown-status quarantine.
- Risk: duplicate writes from retries or replay.
  - Mitigation: external-id idempotency keys + deterministic upsert rules.
- Risk: stale quotes after upstream changes.
  - Mitigation: freshness SLA monitor + forced reprice trigger on key fields.

## Test Strategy Gates

1. Unit: field transforms and enum mapping logic.
2. Contract: inbound/outbound schema conformance using fixture payloads.
3. Integration: local DB upsert idempotency and replay tests.
4. E2E: simulated Kent move update -> Corkysoft processing -> outbound payload.

## Immediate Next Actions

1. Acquire 20-30 anonymized Kent payload samples spanning move lifecycle states.
2. Finalize external ID bridge schema and migration.
3. Implement mapping validator to fail fast on unknown required fields.
4. Add CI job that runs contract and replay tests on every integration change.
