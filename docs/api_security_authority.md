# API Security And Authority Contract

Last updated: 2026-07-18.

This document owns Corkysoft's internal API security and authority contract.
Dashboard user auth remains documented separately in
[Authentication And Users](authentication_and_users.md).

## Current Implemented Slice

Sensitive internal reads now require the configured internal API token across
the current API surface, including:

- `GET /jobs/{jobId}`
- `GET /driver-shifts`
- calls/transcripts and state-egress routes
- operations readiness, jobs board, inventory, labor, and cutover routes
- labor/payroll/absence analytics routes
- Kent/commercial tender routes

These routes expose client, billing, job, worker, truck, shift, transcript,
state-egress, payroll, tender, rate, and cost data, so unauthenticated requests
fail closed with `401`.

Current header:

- `X-Corkysoft-Api-Key`

Current environment variable:

- `CORKYSOFT_API_TOKEN`

Scoped credentials are now supported for migrated routes through:

- `CORKYSOFT_SERVICE_CREDENTIALS_JSON`
- optional `X-Corkysoft-Request-Id`

Each credential entry supplies an id, token, actor, and scopes. Optional
timezone-aware `not_before`, `expires_at`, and `revoked_at` timestamps define
its lifecycle; inactive, expired, not-yet-active, and revoked credentials fail
closed. Migrated routes
derive authority from the resulting API auth context instead of trusting actor
fields in request bodies.

Operations planning, operations cutover, calls/transcripts, worker-time review,
Kent tender policy and override, labor absence, and MoveWare/Kent importer
writes are now migrated to scoped service credentials. Migrated routes bind
persisted actor/operator identity to the auth context for scoped callers and
write `api_write_receipts` rows after successful mutations. Receipts include
the authenticated actor, credential, scopes, request id, action/resource,
route/method, timestamp, and `succeeded` outcome.

`CORKYSOFT_API_TOKEN` remains valid for protected reads. It is no longer an
implicit write credential: legacy write compatibility is disabled by default
and must be explicitly enabled with
`CORKYSOFT_ALLOW_LEGACY_API_WRITE_TOKEN=1` during a bounded migration window.

## Credential Rotation And Revocation

Rotate a service by publishing its replacement credential with a distinct id,
actor, token, and bounded `not_before`/`expires_at` overlap. Deploy consumers
onto the replacement, then set `revoked_at` on the old credential (or remove
it) when the overlap ends. The API evaluates lifecycle state on every request;
revocation takes effect immediately without a server restart because the
credential configuration is read for each request.

Do not create unbounded credential pairs. Every production service credential
should have an `expires_at`; missing expiry is tolerated only for existing
compatibility configuration while it is migrated. Record the planned legacy
token removal date in deployment configuration, remove
`CORKYSOFT_ALLOW_LEGACY_API_WRITE_TOKEN`, and then remove the shared token from
write clients.

## Remaining P0 Scope

BAD-001 is corrected for authenticated sensitive reads. Read-scope granularity
still belongs to BAD-002 because the current token is not a scoped service
credential.

BAD-002 is implemented for current high-authority API families.
Scoped service credentials, actor-bound writes, and API write receipts are in
place for operations planning and cutover, calls/transcripts, worker-time
review, Kent tenders/config, labor absence, and importer writes. BAD-002
now has explicit rotation/revocation/deprecation behavior and regression
coverage for wrong-scope, expired, revoked, legacy-disabled, actor-binding, and
receipt-outcome behavior.

BAD-003 is narrowed to promotion governance. Transcript/audio uploads now
enforce size/content checks, strict base64 decoding, extension allowlists, and
safer adapter JSON/error handling. Transcript artifacts now persist explicit
data classification, `observer_capture_ref` authority class, sanitized failed
task errors, and failed-artifact metadata. Transcript/model output remains
advisory until an authorized actor explicitly promotes reviewed content into an
operational record.

## Next Security Priorities

1. Finish BAD-002 governance cleanup:
   credential rotation/deprecation docs, explicit legacy-token sunset guidance,
   and exhaustive denial/receipt coverage for every write family.
2. Implement BAD-003 reviewed promotion:
   scoped actor decisions that accept, reject, or hold transcript/browser/PNF
   evidence before it can influence operational or customer-safe state.
3. Build customer-safe projection foundations only after reviewed internal
   state exists.

## Target Authority Model

Future internal API credentials should produce an auth context with:

- credential id
- actor or service identity
- granted scopes
- request or correlation id

Initial supported scopes:

- `api:read`
- `import:write`
- `calls:write`
- `worker_time:write`
- `evidence:review`
- `kent:write`
- `labor:write`
- `operations:write`
- `operations.cutover:write`
- `operations.cutover:approve`

High-authority writes should derive actor identity from the auth context, not
from request-body fields such as `actor`, `operatorId`, or `recordedBy`.

Customer-visible status, tracking, receipts, notifications, and support replay
must consume only reviewed public-safe projections. They must not expose raw
dashboard state, internal notes, margin/cost data, worker private data, raw
telemetry, or advisory transcript/model output.

## Reviewed Evidence Promotion

Transcript and model evidence is never an authority-bearing command. A caller
with `evidence:review` may create a durable, `held` proposal naming its source
artifact, target, proposed action, and public/internal payload. A scoped actor
can later record `accepted`, `rejected`, or `held`; the decision stores actor,
credential, scopes, request id, reason, and immutable history. Neither
proposal nor acceptance itself applies a note, worker-time decision, job-state
change, or customer projection. Those effects remain separate governed writes,
so raw evidence cannot silently cross into operations.

## Acceptance Criteria

- sensitive REST reads require authentication and authorized read scope
- unauthenticated and wrong-scope requests fail closed with tests
- shared write token is replaced or explicitly deprecated behind scoped service
  credentials
- high-authority writes derive actor/service identity from auth context
- spoofed actor/operator request-body fields are rejected or ignored
- high-authority writes persist receipt metadata for actor, scope, action,
  resource id, timestamp, and request/correlation id
- transcript/audio uploads enforce size, content, and adapter-boundary limits
- transcript/model outputs remain advisory until accepted by an authorized actor
- MCP tools stay read-only and resource paths stay inside configured roots
- security/API tests pass through the repo virtualenv without network
  dependency
