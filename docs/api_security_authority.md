# API Security And Authority Contract

Last updated: 2026-06-06.

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

Each credential entry supplies an id, token, actor, and scopes. Migrated routes
derive authority from the resulting API auth context instead of trusting actor
fields in request bodies.

Operations planning, operations cutover, calls/transcripts, worker-time review,
Kent tender policy and override, labor absence, and MoveWare/Kent importer
writes are now migrated to scoped service credentials. Migrated routes bind
persisted actor/operator identity to the auth context for scoped callers and
write `api_write_receipts` rows after successful mutations. The legacy
`CORKYSOFT_API_TOKEN` still works as a temporary compatibility credential for
current internal clients while rotation/deprecation guidance is completed.

## Remaining P0 Scope

BAD-001 is corrected for authenticated sensitive reads. Read-scope granularity
still belongs to BAD-002 because the current token is not a scoped service
credential.

BAD-002 is substantially implemented for current high-authority API families.
Scoped service credentials, actor-bound writes, and API write receipts are in
place for operations planning and cutover, calls/transcripts, worker-time
review, Kent tenders/config, labor absence, and importer writes. BAD-002
remains open until credential rotation/deprecation docs are explicit and
denial/receipt coverage spans every write family.

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
