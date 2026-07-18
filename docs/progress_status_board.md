# Progress Status Board

Last updated: **2026-07-18**

This page is the operational tracker for implementation-to-docs alignment. Use
it with [ROADMAP.md](../ROADMAP.md), [README.md](../README.md), and
[COMPACTIFIED_CONTEXT.md](../COMPACTIFIED_CONTEXT.md). Confirmed bugs, bad
cases, accepted risks, owner lanes, and promotion gates live in
[Known Bugs And Bad Cases](known_bad_cases.md).

## Progress Snapshot

- Core routing and costing: **🟢 implemented**
- Streamlit dashboard baseline: **🟢 implemented**
- Five-view workflow shell (`Quote`, `Pricing Intelligence`, `Network`, `Operations`, `Admin`): **🟢 implemented**
- Shared UI design system (`dashboard/theme.py`, KPI strip, alert banner): **🟢 implemented**
- Role-layout defaults and hidden-tab protections for the new shell: **🟢 implemented**
- Dashboard shell docs/onboarding alignment: **🟡 in progress**
  Canonical shell surfaces now describe only the five top-level views; deeper
  workflow docs still need periodic drift sweeps.
- Static KPI/alert placeholder replacement with sourced operational truth: **🟡 in progress**
- Shareable workspace-state and support-grade shell reconstruction: **🟡 in progress**
  Canonical `ws` state now supports shell and Operations child-workflow replay;
  persisted snapshot IDs and richer form/state reconstruction are still pending.
- Operations diary surface: **🟢 implemented**
- Auth and role-boundary hardening: **🟢 implemented**
- Customer-facing tracking / receipt surfaces: **⚪ roadmap**
- Historical job corridor/lane model: **🟢 implemented**
- Situation-awareness/disruption ingestion and bounded auto-policy use: **🟢 implemented**
- Historical ingest coverage governance: **🟢 implemented**
- Observer export for diary/reconciliation: **🟢 implemented**
- MCP adapter for cross-project/tooling access: **🟡 in progress**
  Current read-only tools now constrain caller-supplied `db_path` values to
  configured roots; mutable tool governance remains deferred.
- Kent payload governance and operator workflow validation: **🟡 in progress**
- API-wide authz and scoped integration credentials: **🟡 in progress**
  Sensitive REST reads now require the internal API token across current API
  routers. Scoped service credentials, actor-bound writes, and API write
  receipts now cover operations planning/cutover, calls/transcripts,
  worker-time review, Kent tender/config writes, labor absence, and importer
  writes; credential rotation/deprecation docs and exhaustive denial/receipt
  coverage remain blockers for customer-facing automation.
- Canonical migration/import contract: **🔴 blocker**
- CI/dev workflow reproducibility: **🟡 in progress**

GitHub execution tracking (2026-07-18): issues [#221](https://github.com/chboishabba/corkysoft/issues/221)
through [#225](https://github.com/chboishabba/corkysoft/issues/225) now track the
remaining implementation programme. PR [#226](https://github.com/chboishabba/corkysoft/pull/226)
contains the BAD-005 Dispatch empty-filter guard; it remains open until merged
and promoted with behavioral coverage.

## Current Audit Conclusions

- The major UI revision is real in code and reflected in dashboard-specific
  tests.
- The main repo-level gap is not shell implementation but stale documentation
  and governance language that still refer to the old flat tab model.
- The canonical control docs now describe `Quote`, `Pricing Intelligence`,
  `Network`, `Operations`, and `Admin` as the only top-level shell entrypoints;
  deeper workflow docs still require drift checks so nested workflows are not
  mistaken for top-level navigation.
- The deeper workflow-spec sweep is now removing stale direct-tab and legacy
  `view=` examples from planner, diary, auth, and cutover docs so test and
  operator guidance stay aligned with the implemented shell contract.
- The current deeper-doc sweep has already corrected planner and Kent workflow
  specs so they describe child workflows under the owning shell instead of
  free-standing top-level tabs.
- A real provider regression was found and corrected: Google-selected flows
  were still vulnerable to implicit ORS/OSM fallback behavior in some map and
  isochrone paths.
- The new view-level KPI strips and alert banners currently use hard-coded
  placeholder values, so they are presentation scaffolding rather than
  decision-grade operational telemetry; the shell now shows an explicit
  placeholder-governance notice in each top-level view until sourced metrics
  replace those values.
- The shell currently supports landing via `view=` and related query params,
  and now also supports a normalized `ws` payload for canonical shell and
  Operations child-workflow replay. That is real progress, but it is not yet
  full support-grade reconstruction: persisted snapshot IDs and richer
  workflow/form replay still remain.
- There is no customer-safe/public tracking or printable receipt surface yet,
  but the telemetry/ETA substrate is reusable: `analytics/live_data.py`
  already provides `truck_positions`, `active_routes`, route progress, ETA,
  and geometry that can anchor a later customer-facing tracker.
- BAD-001 is corrected for authenticated sensitive reads across the current API
  surface. Read-scope granularity remains part of the broader BAD-002 scoped
  credential and actor-binding work.
- BAD-002 now covers current high-authority API families: `ApiAuthContext`,
  scoped write credentials, spoofed body actor override for scoped callers,
  request IDs, and persisted `api_write_receipts` are test-backed for
  operations planning/cutover, calls/transcripts, worker-time review, Kent,
  labor absence, and importer writes. Rotation/deprecation docs and exhaustive
  write-family denial coverage still need completion.
- BAD-003 is narrowed to reviewed promotion governance: upload size/content
  checks, strict base64 decoding, extension allowlists, safer WhisperX adapter
  JSON/error handling, transcript classification, and failed-artifact metadata
  are now test-backed.
- BAD-016 is corrected for v1 read-only MCP tools by constraining `db_path` to
  configured roots and denying out-of-root paths in the result envelope.
- Validation note: targeted provider and shell regression suites pass in the
  project virtualenv, including a 133-test focused pass across dashboard shell,
  layout, provider, isochrone, and quote-state coverage; repo guidance is
  being tightened so agent/user execution stays inside the repo venv.
- 2026-06-04 audit note: six read-only lanes found that the highest current
  risk is not the five-view shell itself, but security/API authority,
  persistence/import drift, analytics decision-quality semantics, CI/runtime
  reproducibility, and the absence of a single bad-case register. The register
  is now [Known Bugs And Bad Cases](known_bad_cases.md).
- Security note: `../zkSEC` was used as supporting context for governance
  expectations. Corkysoft should treat public/uncertain signals as
  proposal-only, require actor/scope/receipt for high-authority actions, bound
  adapter resources to declared roots, and block secret-like payloads.

## Current Wave

- Priority 0: secure internal API and integration authority ([#221](https://github.com/chboishabba/corkysoft/issues/221))
- current write routers are migrated to scoped credentials with actor binding
  and receipts; finish credential rotation/deprecation docs and exhaustive
  denial/receipt coverage

- Priority 1: implement reviewed promotion for advisory evidence ([#222](https://github.com/chboishabba/corkysoft/issues/222))
- add scoped actor decisions that accept, reject, or hold transcript,
  browser/OpenRecall, and PNF-derived evidence before it can influence
  operational or customer-safe state

- Priority 2: establish canonical migrations and imports ([#223](https://github.com/chboishabba/corkysoft/issues/223))
- unify DDL authority, old-DB upgrades, MoveWare schema alignment, and
  persisted import issue reporting before downstream workflows depend on them

- Priority 3: finish operator-breaking dashboard fixes ([#224](https://github.com/chboishabba/corkysoft/issues/224))
- close hidden-tab reveal and promote the Dispatch empty-filter guard in PR
  [#226](https://github.com/chboishabba/corkysoft/pull/226) with behavioral tests

- Priority 4: correct analytics decision quality ([#225](https://github.com/chboishabba/corkysoft/issues/225))
- address margin semantics, deterministic history windows, cost/revenue
  collisions, and telemetry freshness

- Priority 5: harden decision-signal governance in the new views
- replace static KPI and alert content with sourced metrics, freshness stamps,
  and explicit unknown/fallback states

- Priority 4: keep the shell reviewable
- expand regression coverage around shared UI primitives, role-layout reset,
  deep-link landing, support-grade workspace sharing, and mixed-surface
  composition boundaries

- Priority 5: normalize the architecture surface
- keep one reviewed metasystem view and child UML/C4 drill-down set aligned
  with the implemented shell and control boundaries

- Priority 6: formalize operational data contracts
- require source, owner, freshness, stale-threshold, and fallback semantics for
  decision-adjacent shell data before it is treated as operational truth

- Future lane: customer-facing tracking and receipt surfaces
- build a separate public/customer-safe status and receipt contract on top of
  live telemetry and job-status primitives without reusing internal shell state

- Product-bridge roadmap: [Operations Platform Roadmap](operations_platform_roadmap.md)
- consolidate calendar-first dispatch, a crew-facing execution workflow,
  promoted job/customer state, quote-to-job handoff, and reviewed closure
  evidence without claiming full CRM-to-payment-suite parity

## Worker Lanes

- Worker 1 lane: security/API authority
- Worker 2 lane: persistence/import contracts
- Worker 3 lane: dashboard shell and role/workspace behavior
- Worker 4 lane: analytics decision quality
- Worker 5 lane: CI/dev workflow reproducibility
- Worker 6 lane: docs/architecture governance

## Orchestrator Control Map

- Worker 1 control objective: gate internal REST/MCP/transcript surfaces with
  scoped actor/scope/receipt controls.
- Worker 2 control objective: collapse schema/import drift behind a canonical
  migration and importer contract.
- Worker 3 control objective: keep role visibility, workspace state, and
  operator selection flows behaviorally safe.
- Worker 4 control objective: remove ambiguous denominators, date windows,
  import inference, and telemetry freshness semantics from decision views.
- Worker 5 control objective: make repo-venv pytest and Playwright smoke the
  repeatable control plane.
- Worker 6 control objective: keep roadmap/progress/known-bad-case/UML surfaces
  synchronized.
- Previous signal lane control objective: replace shell scaffolding with contract-backed
  KPI/alert signals that expose source, owner, freshness, and fallback.
- Previous workspace lane control objective: make workspace state reproducible, shareable, and
  fail-closed under role and query-param constraints.
- Previous architecture lane control objective: keep the reviewed metasystem UML/C4 entrypoint
  and child shell diagrams aligned with current control boundaries.
- Previous operational-data lane control objective: make operational data contracts explicit before
  decision-looking signals are promoted as operational truth.
- Future customer-tracking objective: add a tokenized, public-safe status and
  receipt surface with expiring scoped links, explicit data classification, and
  fail-closed origin/access behavior.

## Exit Criteria For The Current Wave

- docs and code use the same five-view shell taxonomy
- targeted provider-selection regressions pass in the repo virtualenv
- shell/deep-link/layout/UML artifacts remain aligned with the implemented
  workflow shell
- workspace-state sharing is deterministic and support-safe for the supported
  workflows
- placeholder signals are either sourced and bounded or clearly labeled as
  non-decision-grade scaffolding
- decision-adjacent shell data declares owner/freshness/fallback semantics
- all open P0/P1 items in [Known Bugs And Bad Cases](known_bad_cases.md) are
  either fixed with evidence or explicitly accepted with owner and review date
- any future customer-facing tracking link remains tokenized, expiring,
  least-privilege, and auditable rather than mirroring internal shell state

## Non-blocking Documentation Tasks

- Keep [ROADMAP.md](../ROADMAP.md), [README.md](../README.md), and role-facing
  docs aligned whenever shell labels or role entrypoints change.
- Keep [Known Bugs And Bad Cases](known_bad_cases.md) aligned whenever audits
  find, fix, downgrade, or accept bad cases.
- Keep [Service Blueprint](service_blueprint.md) aligned whenever lifecycle,
  customer notification, worker execution, support, or job-completion
  expectations change.
- Update [UI Role Coverage Matrix](ui_role_coverage_matrix.md) first when
  role/shell ownership changes, then refresh stories and onboarding from it.
- Update the C4/PlantUML dashboard-shell diagrams when the five-view shell
  structure changes materially.

## Update Protocol

1. Update roadmap wording when feature scope or shell ownership changes.
2. Update this board in the same pass when status or blockers move.
3. Keep [COMPACTIFIED_CONTEXT.md](../COMPACTIFIED_CONTEXT.md) aligned with the
   current audit conclusions.
4. Record externally visible workflow changes in [CHANGELOG.md](../CHANGELOG.md).
