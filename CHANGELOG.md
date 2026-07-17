# Changelog

## 2026-07-17

- consolidated the existing dispatch, worker-execution, customer-communication,
  quote-to-job, and evidence/closure plans in
  `docs/operations_platform_roadmap.md`
- clarified product positioning: Corkysoft integrates first but selectively
  internalizes decision-critical operational workflows; it does not yet claim
  general CRM, payments, reputation-management, storage-billing, or broad
  connector parity
- aligned README, ROADMAP, and the progress board with the consolidated
  operations-platform bridge and its unchanged security/import/decision-quality
  promotion gates

## 2026-06-04

- added `docs/service_blueprint.md` as the canonical lifecycle matrix for
  inquiry-to-completion, customer communications, worker execution, call
  follow-up, and job-completion gates
- added `docs/service_blueprint_flows.puml` and a story path index so diagrams
  can attribute users, shells, interactions, and gates back to canonical story
  paths
- rendered the service blueprint story-path diagram to SVG and PNG, linked the
  rendered views from README, architecture docs, the service blueprint, and the
  generated UML index
- added `docs/api_security_authority.md`, gated sensitive API reads across the
  current API surface behind the internal API token, removed duplicated
  Kent/Labor token helpers, and added transcript/audio upload plus WhisperX
  adapter boundary hardening
- constrained MCP `db_path` inputs to configured roots and added out-of-root
  denial coverage for read-only MCP tools
- added the first BAD-002 scoped authority slice with API auth contexts,
  operations cutover write/approval scopes, auth-context actor binding,
  persisted API write receipts, and a Google Sheets export-URL guard so
  workbook connectors remain read/import-only
- expanded BAD-002 scoped-write coverage to calls/transcripts, worker-time
  review, Kent tender/config writes, labor absence records, and MoveWare/Kent
  importer routes, with scoped actor-binding and API write receipt regressions
- migrated residual operations-planning mutations to `operations:write`
  credentials with API write receipts for policy, workbook sync, segment
  planning, worker role/compliance, and inventory allocation writes
- added explicit transcript artifact classification, observer-capture authority
  metadata, sanitized failed-task persistence, and regression coverage for
  failed call transcript polling
- strengthened `docs/ui_role_coverage_matrix.md` as the source of truth for
  internal role-to-shell and role-to-leaf mapping, with user stories and
  onboarding deriving from it
- integrated six read-only audit lanes into the canonical planning surfaces
- added `docs/known_bad_cases.md` as the central register for confirmed bugs,
  bad cases, accepted risks, owner lanes, and promotion gates
- reprioritized the active roadmap around security/API authority, import/schema
  correctness, dashboard operator safety, analytics decision quality, CI/dev
  reproducibility, and docs/architecture governance
- folded `../zkSEC` security expectations into Corkysoft promotion gates:
  proposal-only public signals, actor/scope/receipt for high-authority actions,
  resource-root bounds, and secret-like payload rejection
- updated the progress board, README, UML index, and status metadata so roadmap
  and audit state no longer rely on stale or competing status surfaces

## 2026-04-02

- normalized the PlantUML supermega/dashboard-shell generator around the
  implemented five-view `dashboard/views/*` layer, assigned `dashboard.theme`
  to the shell domain, and refreshed focused UML coverage so the architecture
  root no longer summarizes the UI through retired component-era shell labels
- corrected the remaining canonical-doc shell drift so README, roadmap, and
  architecture/progress control surfaces no longer describe `Planner`,
  `Dispatch`, or `Operations diary` as pending or top-level shell tabs
- removed the next deeper shell-doc drift in planner/diary/auth workflow specs,
  including stale direct-tab and legacy hidden-query-param examples that no
  longer matched the five-view shell contract
- corrected deeper workflow specs so planner, Kent tender, spreadsheet-cutover,
  and operations-diary docs now describe those surfaces as nested workflows
  under the current shell rather than as standalone top-level tabs
- removed the remaining stale `Operations diary` deep-link target from the
  diary refresh flow, refreshed the shell-layout regression guard, and updated
  user-testing notes so the old `experimental_rerun` crash is recorded as
  historical evidence rather than current behavior
- added an explicit placeholder-governance notice to all five top-level shell
  views so scaffold KPI strips and alerts are visibly marked as
  non-decision-grade until sourced metrics with freshness/ownership replace
  them
- hardened the shared KPI-strip and alert-banner primitives so telemetry-backed
  labels, values, and messages are HTML-escaped before rendering, and added
  focused regression coverage for those components
- validated the current shell/provider remediation wave with a focused
  repo-venv suite covering dashboard shell, layout, provider strictness,
  isochrones, and quote-state flows: 133 passing tests
- formalized the current remediation wave into four explicit worker control
  lanes, each with standards-aligned scope, completion evidence, and promotion
  gates in the roadmap, progress board, and architecture docs
- aligned the dashboard shell, role docs, roadmap, and progress board around
  the revised five-view top level: `Quote`, `Pricing Intelligence`, `Network`,
  `Operations`, and `Admin`
- repaired query-param deep links from quote, planner, dispatch, and
  operations-diary flows so they return to owning top-level views instead of
  retired flat-tab names
- updated the Fleet-admin role-layout editor to use the live five-view shell
  taxonomy and refreshed focused regression coverage accordingly
- replaced the one-off dashboard architecture diagram with a generated UML suite driven by the internal Python import graph
- added a supermega UML entrypoint plus child-view PlantUML and rendered SVG artifacts under `docs/rendered/`
- added render/check ergonomics to `scripts/build_supermega_uml.py`, including `--render` and `--check --render`
- corrected child UML diagrams so cross-domain dotted arrows attach to the real source modules instead of the alphabetically first module in each domain
- added focused UML builder coverage to lock domain assignment, supermega cross-domain links, child-link anchoring, render target mapping, and missing-PlantUML failure behavior

## 2026-03-31

- expanded MCP bridge governance coverage to assert blank-tool-name and unknown-tool-name behavior
- increased the focused integrated validation suite to 98 passing tests when including bridge-level MCP coverage
- audited rerun handling and confirmed the shared dashboard state layer is now the sole rerun owner
- added a Kent UI regression proving non-admin roles see admin write controls disabled, not just helper-level write gating
- increased the focused integrated validation suite to 89 passing tests
- completed the narrow cleanup wave with 88 focused tests passing
- moved quote suggestion manual-override reset into the authoritative shared state helper
- consolidated planner, maintenance, and operations rerun handling onto the shared dashboard state helper
- expanded MCP scenario coverage to verify execution-error envelopes preserve code and details
- corrected the rerun-wrapper backlog to the real remaining paths in planner, maintenance, and operations, then assigned the next narrow cleanup wave around quote helper boundaries, real rerun cleanup, and deeper governance scenarios
- partially completed the next cleanup wave with 87 focused tests passing
- moved quote suggestion application onto the shared dashboard state helper
- expanded MCP tests to cover success and error envelope payload shapes
- confirmed that the calls/maps rerun-wrapper backlog was partly stale: calls already reused shared rerun handling and maps does not currently rerun
- assigned the next three non-blocking cleanup lanes after the 85-test integrated pass: quote decision-control cleanup, remaining rerun-wrapper consolidation, and broader governance scenario validation
- completed the boundary-cleanup and live-control-validation wave with 85 focused tests passing
- moved quote-builder route-label formatting onto the shared dashboard state helper
- removed duplicate rerun helpers from planner and route maps in favor of the shared dashboard state helper
- made the Kent admin write-role set explicit and added regression coverage for both the guard logic and allowed-role set
- tightened MCP contract coverage so documented tool names and response-version invariants stay locked by tests
- updated Kent and MCP docs to reflect the enforced governance and contract invariants
- assigned the next three non-blocking worker lanes after the integrated 46-test validation pass: quote-builder boundary cleanup, shared rerun/state helper consolidation, and Kent/MCP live-control validation coverage
- assigned the next three non-blocking worker lanes after the previous wave landed: operator reconciliation, outbound contract hardening, and quote-builder/shared-state consolidation
- assessed the remaining roadmap and reordered the active wave into three
  priority bands: operator execution completion, governance and contract
  hardening, and decision-quality/planner-intelligence work
- assigned one non-blocking worker lane per band so the next execution wave is
  parallelized against disjoint ownership areas
- split the dashboard sidebar control layer into smaller internal control
  points for database initialization, dataset/provider selection, historical
  ingest, dataset loading, filter-state resolution, and break-even updates
- added focused regression coverage for the new data-control helper paths
- removed remaining helper duplication between `dashboard/app.py` and the
  shared dashboard state/shell layers
- updated the README, module guide, architecture guide, progress board, plan,
  and compactified context to reflect the extracted dashboard control layers
- added a PlantUML dashboard-shell architecture view for contributor-facing C4
  style documentation

## 2026-03-30

- aligned the role/onboarding docs to the current Kent governance boundary, so
  `Commercial Owner` now treats `Kent admin` as a governed review surface while
  `System / Rollout Admin` remains the current write owner
- documented the next UI remediation wave around analytics-first shell chrome,
  role-layout reset/deep-link hardening, rerun-compatibility cleanup, and mixed
  surface sectioning before the next browser-led testing pass
- recorded the next `dashboard/app.py` decomposition wave, splitting the work
  into non-blocking lanes for auth/query-param flow, shell/layout-state logic,
  and data-controls/tab-composition extraction

## 2026-03-27

- fixed the Kent admin tab crash caused by calling the injected
  dashboard-user admin renderer positionally instead of by keyword
- added a regression test covering the Kent admin renderer call shape
- deduplicated duplicate DataFrame columns before passing map data into pydeck
  layers, preventing repeated `DataFrame columns are not unique` warnings from
  the network overlay map path
- added a focused map helper regression test for pydeck column deduplication

## 2026-03-20

- documented the adaptive learning loop as a bounded, reviewable policy-update
  system informed by realised jobs and situational-awareness inputs
- added a first implementation slice for adaptive policy parameters on top of
  the existing global parameter store, with focused tests
- added an `Operations diary` workflow with day/week job review, diary tasks,
  vehicle/staff usage drill-down, and customer/subcontractor invoice-review
  records
- added a Planner day view tied to existing planned segments so route shaping
  can be compared against the live daily schedule

## 2026-03-21

- synced four archived strategy threads into project context and tightened the
  docs to frame Corkysoft as a system of decision rather than only a record or
  analytics surface
- added explicit roadmap and user-story coverage for the
  international/paperwork/compliance gap as future
  requirements/proposal/governance workflow work
- added a Corkysoft/SB/ITIR coverage audit plus a downstream contract note for
  planner/diary/reconciliation outputs, keeping Corkysoft as workflow truth and
  StatiBaker as downstream interpretible state
- extracted reusable planner/diary workflow patterns for possible future
  SB/ITIR lens design without implying SB should own removals operations
- added a dual-marker reconciliation-aging model for delayed supplier bills,
  including unresolved-exposure summaries in Corkysoft and explicit latency
  semantics for later SB timeline work
- added Google-backed dashboard auth using Streamlit OIDC plus a local
  `dashboard_users` allowlist, with shared-environment auth required and
  explicit development-only anonymous access
- made auth state visible in the UI so authenticated Google sessions and
  anonymous local-development runs are clearly distinguishable
- added a focused auth red-team plan, hardened hidden-tab handling so query
  params cannot re-expose role-hidden admin tabs, and made bootstrap-admin
  seeding one-shot once dashboard users exist
## 2026-03-24

- added `docs/progress_status_board.md` to keep feature progress visibility and TODO alignment centralized
- linked `README.md` and `ROADMAP.md` to the new progress board and added explicit documentation alignment guidance
- updated `COMPACTIFIED_CONTEXT.md` with the tracking decision context
- added `disruption_events` persistence plus `analytics/situational_awareness.py` to ingest closure/weather/traffic severity signals and nudge adaptive-policy multipliers inside bounded updates, with tests covering aggregation and parameter rewrites

## 2026-03-25

- hardened historical ingest with durable run summaries, row-level issue capture, readiness classification, and Fleet-admin ingest-health visibility
- formalized corridor/lane handling with canonical location clusters, directional lanes, corridor groups, and persisted assignment status on historical/live rows
- added lane-assignment health visibility plus promotion governance in Fleet admin, including proposal, approval/rejection, and apply flows
- updated planner consumption so corridor suggestions default to canonically assigned lane history and only include ambiguous/unassigned rows when operators opt in
- implemented a Corkysoft-native observer outbox for diary/reconciliation review state, plus explicit export for planning snapshots and reconciliation exceptions
- synced README, roadmap, progress board, and compactified context to reflect the new ingest and lane-governance baseline
- documented the remaining observer-export follow-ups: delivery receipt / watermark semantics and dashboard visibility of emitted envelopes

## 2026-03-26

- implemented commercial quote guidance overlays with benchmark context and backhaul-aware discount guidance in the live quote workflow
- decomposed the main dashboard shell, API root, pricing surface, and call-ops surface into smaller bounded modules while preserving stable import and test monkeypatch surfaces
- added `docs/contributor_docs_sync.md` and synced README, roadmap, progress board, module guide, and positioning docs to reflect the current architecture and product state
- added observer-outbox visibility to the Operations diary so emitted planning/reconciliation envelopes can be filtered and inspected inside the dashboard
- added job-scoped labor reconciliation to the Operations diary detail so managers can compare planned labor against imported shifts inside the same usage/reconciliation cockpit

## 2026-03-27

- gated Kent admin policy-default and override-reason writes to `system_rollout_admin`, keeping operator review/tender workflow visible while tightening dashboard governance boundaries
- surfaced persisted spare-capacity signals and container-pressure rollups in the Dispatch board so operations can review backhaul/container-fit context after award, not only during quoting
- added operator-facing share/reallocation recommendations in Dispatch based on spare-capacity and container-pressure signals, without yet introducing automatic reassignment
- added explicit Dispatch response actions for share/reallocation and utilisation handling, including persisted action/status logging on recommendation candidates
- added a first regression/modeling baseline for profitability: historical margin-per-m³ regression over distance and season, surfaced inside profitability insights with fit metrics and scenario previews
- extended the profitability model with corridor-aware margin prediction, baseline-vs-corridor fit deltas, and explicit operator-safe interpretation guidance in the UI
- added chronological holdout validation and trust labels for the corridor-aware model so operators can distinguish reviewable output from low-support or weak-fit cases
- added a first Corkysoft MCP adapter scaffold with a local JSON bridge, stable result envelopes, and four read-only tools over profitability, dispatch recommendations, operations-diary summary, and quote-guidance preview
- hardened the MCP entrypoint so `python -m corkysoft.mcp` now defaults to the supported JSON bridge, with explicit `--bridge` / `--server` selection and clearer optional-FastMCP failure behavior
- added a development-only auth-state harness plus focused tests so browser automation can exercise anonymous, misconfigured, unauthorized, inactive, and low-privilege hidden-tab auth paths without live Google login
- added Playwright auth harness coverage for the core denial/banner flows and hardened remote-origin / `redirect_uri` mismatch handling so tunneled deployments fail clearly instead of looking partially authenticated
- added an explicit temporary auth shortcut so successful Google logins can auto-provision as local `system_rollout_admin` users behind `CORKYSOFT_AUTO_PROVISION_GOOGLE_ADMIN=1`, with a visible warning banner to keep the posture obvious during owner/testing sharing
