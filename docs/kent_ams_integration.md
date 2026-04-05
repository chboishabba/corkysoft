# Kent AMS Integration Spec (Corkysoft)

This document defines how Corkysoft should interface with Kent AMS for data
exchange and pricing analytics handoff.

## Scope

- Inbound from Kent AMS into Corkysoft:
  - assignee/job metadata
  - origin/destination and service details
  - shipment and vendor context (when available)
- Outbound from Corkysoft back to Kent AMS:
  - calculated quote recommendations
  - margin diagnostics
  - pricing rationale/audit fields

## Integration Boundary

- Kent AMS remains system of record for operational relocation workflow.
- Corkysoft is a pricing intelligence layer and does not replace AMS workflow
  ownership.
- Corkysoft persists derived analytics in its own SQLite-backed model.
- Current status: internal/provisional workflow implemented, but live payload
  validation and deployment hardening are still required before production use.

## Canonical Entity Mapping

- `assignee` (Kent) -> `client/customer` concepts in Corkysoft quote flows
- `move` (Kent) -> `job` rows used by routing and costing
- `shipment` (Kent) -> optional logistics context for pricing modifiers
- `service/vendor` (Kent) -> service and cost adjustment factors

## Field Mapping Table (`kent_field` -> `corkysoft_field`)

This is the first-pass mapping based on current repo schema and API models.
Replace placeholder Kent names with real payload keys once sample payloads are
available.

| Kent AMS field (expected) | Corkysoft target | Transform / rule | Status |
| --- | --- | --- | --- |
| `move.id` | `jobs.job_number` or `jobs.client_reference` | Keep source identifier for idempotent upsert and traceability. | provisional |
| `move.created_at` | `jobs.created_at` | Parse ISO-8601; preserve timezone. | provisional |
| `move.updated_at` | `jobs.updated_at` | Parse ISO-8601; overwrite only if newer. | provisional |
| `move.origin.address` | `jobs.origin` | Store as free text; normalize in routing pass. | provisional |
| `move.destination.address` | `jobs.destination` | Store as free text; normalize in routing pass. | provisional |
| `move.customer_reference` | `jobs.client_reference` | Direct copy. | provisional |
| `move.country` | `jobs.country` | Default to `Australia` when missing. | provisional |
| `move.distance_km` (if provided) | `jobs.distance_km` | Prefer Corkysoft routing output if recomputed. | provisional |
| `move.duration_hr` (if provided) | `jobs.duration_hr` | Prefer Corkysoft routing output if recomputed. | provisional |
| `move.quoted_price` | `historical_jobs.quoted_price` | Store as numeric quote history input. | provisional |
| `move.volume_m3` | `historical_jobs.m3` or `quotes.cubic_m` | Route to historical/job quote context by ingest mode. | provisional |
| `move.client_name` | `historical_jobs.client` or `clients.company_name` | Split person/company where possible. | provisional |
| `assignee.first_name` | `clients.first_name` | Trim whitespace; title case optional. | provisional |
| `assignee.last_name` | `clients.last_name` | Trim whitespace; title case optional. | provisional |
| `assignee.email` | `clients.email` | Lowercase for matching; keep original for display if needed. | provisional |
| `assignee.phone` | `clients.phone` | Keep raw; normalize digits for dedupe logic. | provisional |
| `assignee.address.line1` | `clients.address_line1` | Direct copy. | provisional |
| `assignee.address.line2` | `clients.address_line2` | Direct copy. | provisional |
| `assignee.address.city` | `clients.city` | Direct copy. | provisional |
| `assignee.address.state` | `clients.state` | Normalize to AU state code where possible. | provisional |
| `assignee.address.postcode` | `clients.postcode` | 4-digit AU validation rule. | provisional |
| `assignee.address.country` | `clients.country` | Default `Australia` if missing. | provisional |
| `shipment.id` | `shipments.id` (external map table required) | Do not force primary key collision; keep mapping in bridge table. | gap |
| `shipment.status` | `shipments.status` | Map Kent enum -> local enum (`planned`, `in_transit`, `delivered`, etc.). | gap |
| `shipment.scheduled_date` | `shipments.scheduled_date` | Parse ISO date/time. | provisional |
| `shipment.delivered_at` | `shipments.delivered_at` | Parse ISO date/time. | provisional |
| `shipment.quantity` | `shipments.quantity` | Numeric parse, default `1`. | provisional |
| `shipment.from_location` | `shipments.from_location` | Direct copy. | provisional |
| `shipment.to_location` | `shipments.to_location` | Direct copy. | provisional |
| `shipment.truck_ref` | `shipments.truck_id` | Resolve to `trucks.truck_id`; create truck if missing and policy allows. | gap |
| `shipment.worker_ref` | `shipments.worker_id` | Resolve to `workers.id`; create worker if missing and policy allows. | gap |
| `vendor.name` | `suppliers.company_name` | Upsert supplier by unique company name. | provisional |
| `vendor.phone` | `suppliers.contact_number` | Direct copy. | provisional |
| `vendor.email` | `suppliers.email` | Lowercase normalization. | provisional |
| `inventory.item_name` | `inventory_items.name` | Upsert item by name. | provisional |
| `inventory.quantity` | `inventory_items.quantity` | Aggregate or overwrite by ingest mode. | gap |
| `service.cost_components[]` | `job_cost_components` | Expand array to category/rate/quantity/total rows. | provisional |
| `quote.manual_override` | `quotes.manual_quote` | Preserve manual override separately from computed final quote. | provisional |
| `quote.final_amount` | `quotes.final_quote` | Store as authoritative outward recommendation value. | provisional |
| `quote.margin_percent` | `quotes.margin_percent` | Numeric percentage. | provisional |

Legend:
- `provisional`: mapped to existing fields but still needs real Kent payload validation.
- `gap`: needs bridge table, enum mapping, or explicit policy before production use.

## Interface Contract (Initial)

1. Inbound payload (minimum):
   - external IDs (`assignee_id`, `move_id`, `shipment_id` where present)
   - addresses, service date windows, move type
   - volume/weight estimates
   - service requirements (packing, storage, special handling)
2. Corkysoft processing:
   - routing distance/duration calculation
   - cost breakdown and margin estimate
   - corridor and historical benchmarking
3. Outbound payload (minimum):
   - recommended quote total
   - confidence/benchmark context
   - cost component breakdown
   - reason codes for major adjustments

## Internal API Contract (Implemented)

Current internal ingestion endpoint:

- `POST /importers/kent-ams/{resource}`

Supported resources:

- `jobs`
- `subcontractors` (alias: `vendors`)
- `tenders`
- `bids`
- `awards`

Payload shape (same wrapper used by MoveWare importer):

```json
{
  "records": [{ "...": "resource-specific fields" }],
  "dry_run": true
}
```

The route returns:

- `resource`
- `imported` (record count received)
- `dry_run`

Internal API auth:

- mutating internal endpoints are expected to use `X-Corkysoft-Api-Key`
- the server validates this against `CORKYSOFT_API_TOKEN`
- `dry_run` is intended to validate without persisting business/config changes

Current internal API surface:

- `GET /kent-ams/tenders/prioritized?status=open&limit=50`
- `GET /kent-ams/config`
- `PUT /kent-ams/config`
- `GET /kent-ams/override-reasons`
- `PUT /kent-ams/override-reasons/{code}`
- `POST /kent-ams/tenders/{tender_external_id}/override`
- `GET /kent-ams/tenders/{tender_external_id}/overrides`

Ranking factors currently applied:

- profitability policy match (rule mode + thresholds)
- capacity fit (required trucks/workers vs currently active fleet/staff pressure)
- urgency (days until tender due time)
- seasonality (peak/shoulder/base uplift context)
- route/location fit (origin/destination completeness, lane familiarity, historic lane margin)
- en-route spare capacity fit (active trucks already on matching routes with spare load headroom)

Policy behavior now implemented:

- default profitability config is stored in `global_parameters`
- supported rule modes:
  - `ABS_ONLY`
  - `PCT_ONLY`
  - `EITHER`
  - `BOTH`
- profitability policy changes queue priority, not visibility
- policy-pass tenders rank above policy-fail tenders
- policy-fail tenders remain visible with explicit fail reasons
- loss-making tenders remain selectable but raise a high-visibility alert
- hard block is reserved for explicit safety/legal/compliance flags
- transfer/SLA/commercial issues are stored as overrideable flags
- operator overrides require:
  - reason code
  - operator id
  - optional note
  - full score/policy snapshot in audit history

Admin/operator split target:

- operators should only see queue review, flags, and override capture
- admins/managers should own threshold changes, reason-code management, and
  override governance review
- the current implementation is still transitioning toward that split

Kent admin write behavior is guarded by the `KENT_ADMIN_WRITE_ROLES` set so
only `system_rollout_admin` may mutate policy defaults or override pairings.
The regression test `tests/test_kent_admin.py` concretely keeps that write gate
explicit as the surface evolves, preventing unintended admin drift.

Pipeline integration note:

- The same route spare-capacity signal is now written into:
  - quote workflow (`quote_operational_signals`)
  - job ingest workflow (`job_operational_signals`) for Kent and MoveWare imports
- the quote workflow currently uses the same profitability rule vocabulary as
  Kent triage, but should be interpreted as a quote-policy preview rather than
  as proof of tender economics

Current ranking model:

- first tier: hard-block vs non-hard-block
- second tier: profitability policy match vs fail
- third tier: loss alert vs non-loss alert
- fourth tier: heuristic score

Current heuristic weights:

- profitability secondary score: 42%
- capacity fit: 16%
- urgency: 14%
- seasonality: 8%
- route/location fit: 12%
- en-route spare capacity fit: 8%

Dashboard workflow now implemented:

- Streamlit includes a `Quote` -> `Kent tenders` workflow
- operators can:
  - review ranked tenders
  - inspect policy fail reasons and flags
  - record override events and view tender-level audit history
- current admin controls exist but should move out of the high-frequency
  operator queue

Calibration endpoint:

- `GET /kent-ams/tenders/calibration?lookback_days=180`
- returns win rate and margin calibration by score band.

## Weight-Tuning Thoughts (Step 1)

Recommended tuning process before changing production defaults:

1. Lock rule mode and thresholds per branch or operating context before adjusting heuristic weights.
2. Use 6-12 months of tenders with outcomes (`won/lost`) and realized margins.
3. Compare policy-pass vs policy-fail performance, then compare high-band vs low-band win rates inside each tier.
4. Adjust one heuristic weight at a time; keep policy behavior unchanged.
5. Re-run calibration and accept only changes that improve:
   - monotonic win-rate by score band
   - monotonic realized-margin by score band
   - lower mean absolute margin error

Practical guidance for Kent's peak-season context:

- In peak months, capacity pressure is often the hard constraint, so capacity
  weight can be increased temporarily (for example +5 to +10 points) only when
  fleet utilization is consistently high.
- Route/location weight should increase when operators report repeated losses on
  unfamiliar lanes despite high headline revenue.
- En-route spare capacity should be favored when direct-route trucks have enough
  headroom to absorb additional tender volume without degrading current service.
- Keep profitability policy as the dominant queue gate so the heuristic layer
  does not drag low-value work above clearly profitable tenders.
- Urgency should not dominate without profitability support, otherwise operators
  can be pulled into low-value rush work.

## Sync Model

- Preferred: scheduled pull + idempotent upsert by Kent external IDs.
- Optional: webhook/event triggers for high-priority job updates.
- Conflict handling:
  - Kent operational fields win.
  - Corkysoft analytics fields are recomputed from latest inbound state.

## Auth and Security

- Use environment-managed secrets only (`.env`, never committed).
- Current API surface is internal and should be treated as non-public.
- Deployment expectation:
  - mutating endpoints require service credentials or internal admin tokens
  - network exposure should be restricted until that is enforced consistently
- Audit all quote recommendation pushes with timestamp and source version.

## Failure Handling

- Retry transient transport/provider errors with exponential backoff.
- Dead-letter irrecoverable payloads for operator review.
- Preserve last successful analytics snapshot to avoid null pricing responses.

## Open Decisions

- Confirm transport protocol (REST pull, webhook, file drop, or hybrid).
- Confirm Kent AMS field names and enum values for move/shipment status.
- Confirm required SLA for quote turnaround and reprice frequency.
- Confirm authoritative timezone and currency precision rules.
- Decide bridge-table schema for external IDs (recommended: `kent_entity_links`).
- Decide whether shipment/vendor reference creation is automatic or operator-approved.
- Finalize operator/admin separation in the dashboard workflow.
- Finalize override governance:
  - who may override
  - which reasons require review
  - what constitutes override misuse/drift

## Override Governance (v1)

Default v1 governance:

- hard-block is reserved for safety/legal/compliance only
- hard-blocks are not overrideable through the normal operator path
- overrideable flags cover commercial, transfer, SLA, and contextual judgments
- operators may override only when:
  - an active reason code exists
  - operator identity is captured
  - the override is auditable
- managers/admins should review:
  - repeated loss-alert overrides
  - repeated use of `other`
  - overrides on the same corridor/customer pattern

Review cadence:

- weekly during active Kent pilot use
- monthly once workflow stabilizes and calibration is reliable

## Implementation Checklist

1. Capture Kent AMS sample payloads and status enums.
2. Add explicit field mapping table (`kent_field` -> `corkysoft_field`).
3. Implement ingest validator and schema conformance tests.
4. Add outbound contract tests for quote recommendation payloads.
5. Add runbook for sync failures and replay.

Current fixture coverage:

- synthetic tender fixture coverage exists at
  `tests/fixtures/kent_ams/tenders_sample.json`
- fixture smoke validation exists in `tests/test_kent_ams_fixtures.py`
- live Kent AMS exports are still required to lock field names, enum mappings,
  auth behavior, and web-adapter extraction rules

## Next Reference

See [Kent AMS Integration Roadmap](kent_ams_integration_roadmap.md) for phased
delivery, milestone gates, and sequencing.
