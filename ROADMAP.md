Unified deliverables map for Corkysoft, aligning current implementation status with the planned roadmap.

---

## Active Audit Update (2026-07-18)

Current canonical status surface:

- [docs/progress_status_board.md](docs/progress_status_board.md) is the
  current implementation/status tracker.
- [docs/known_bad_cases.md](docs/known_bad_cases.md) is the canonical register
  for confirmed bugs, bad cases, accepted risks, owner lanes, and promotion
  gates.
- Historical roadmap tables below remain useful context, but they are lower
  authority than the active wave, progress board, and known-bad-case register.

Priority order for the next execution wave:

1. **P0 security and authority boundary (GitHub issue [#221](https://github.com/chboishabba/corkysoft/issues/221)).**
   Gate sensitive REST reads, replace the single shared write token with scoped
   service credentials, bind high-authority actions to actor/scope/receipt, and
   add payload/resource limits for transcript, WhisperX, and MCP paths.
   Sensitive REST reads now require the internal API token across the current
   API routers, and transcript/audio uploads now enforce first-pass payload and
   adapter-boundary limits. MCP `db_path` access is now constrained to
   configured roots. Scoped credentials, actor-bound writes, and API write
   receipts now cover operations planning/cutover, calls/transcripts,
   worker-time review, Kent tender/config writes, labor absence, and
   MoveWare/Kent importer writes. Remaining P0 work is credential
   rotation/deprecation docs, exhaustive denial/receipt coverage, and reviewed
   transcript-promotion governance.
2. **P1 reviewed evidence promotion (GitHub issue [#222](https://github.com/chboishabba/corkysoft/issues/222)).**
   Keep transcript and model-derived signals proposal-only until an authorised
   actor accepts, rejects, or holds the evidence.
3. **P1 canonical migrations and imports (GitHub issue [#223](https://github.com/chboishabba/corkysoft/issues/223)).**
   Unify schema authority and make legacy upgrades/import issue persistence
   explicit before relying on downstream operational data.
4. **P1 operator-breaking and decision-misleading bugs (GitHub issues [#224](https://github.com/chboishabba/corkysoft/issues/224) and [#225](https://github.com/chboishabba/corkysoft/issues/225)).**
   Fix authenticated hidden-tab reveal, Dispatch empty-filter crashes, broken
   MoveWare/non-dry-run import paths, CLI history import column drift, and
   ambiguous margin/date-window/cost-inference analytics.
   The direct BAD-005 fix is currently in [PR #226](https://github.com/chboishabba/corkysoft/pull/226).
5. **P1 control-plane and CI reproducibility.**
   Make CI run repo-venv pytest plus root Playwright smoke, separate screenshot
   artifact generation, and remove implicit `git pull`/system-tool startup
   behavior from scripts.
6. **P2 roadmap/data-contract hardening.**
   Finish support-safe workspace-state replay, source shell KPI/alert signals,
   add operational data contracts, and keep generated UML freshness visible.
7. **Future product lanes.**
   Customer tracking/receipts, richer planner/media/CV workflows,
   international/compliance-heavy work, and deeper solver work should wait
   until the security, import, and decision-quality gates are under control.
   The [Operations Platform Roadmap](docs/operations_platform_roadmap.md)
   now consolidates the already-planned calendar dispatch, crew execution,
   customer-update, quote-to-job, and evidence/closure bridge. It does not
   expand near-term scope to general CRM, payments, reputation management,
   storage billing, or broad connector parity.

zkSEC-informed security gates:

- treat public or uncertain signals as proposal-only
- require verified actor identity, authorized scope, plan/receipt metadata, and
  explicit approval for high-authority actions
- keep adapter resources inside declared roots and reject secret-like payloads
- prevent advisory/model/transcript output from silently becoming operational
  authority without review

Worker lanes for the audit remediation wave:

| Worker | Lane | Primary outcome | Promotion evidence |
| --- | --- | --- | --- |
| Worker 1 | Security/API authority | API-wide authz, scoped service credentials, transcript/WhisperX/MCP boundary controls | unauthenticated denial, scoped access, actor receipt, payload/resource-boundary tests |
| Worker 2 | Persistence/imports | canonical migrations and working non-dry-run imports | old-DB upgrade tests, MoveWare/Kent importer contract tests, bad-row issue capture |
| Worker 3 | Dashboard shell | hidden-tab enforcement, Dispatch empty-state guard, role-aware child routing, action semantics | behavioral AppTest/Playwright coverage |
| Worker 4 | Analytics decision quality | finance semantics, history windows, cost inference, telemetry freshness | tests proving denominators, date windows, coverage metadata, stale GPS behavior |
| Worker 5 | CI/dev workflow | repo-venv CI matrix and safe startup modes | pytest collection/run in CI, Playwright smoke, separated screenshots, venv-only scripts |
| Worker 6 | Docs/architecture governance | status-surface cleanup, bad-case register, UML freshness, command guidance | README/ROADMAP/progress/status sync and reviewed doc links |

## UI Revision Audit (2026-04-02)

The major dashboard-shell revision is materially landed in code.
The top-level user-facing navigation is now organized around five workflow
views:

- `Quote`
- `Pricing Intelligence`
- `Network`
- `Operations`
- `Admin`

Observed implementation state:

- shared design-system injection is now wired through `dashboard/theme.py`
- reusable KPI and alert primitives now exist in `dashboard/components/kpi_strip.py`
  and `dashboard/components/alert_banner.py`
- workflow views now compose leaf surfaces under `dashboard/views/` rather than
  exposing the old flat top-level tab set directly
- role-layout defaults and tests now target the new shell taxonomy

Audit conclusions:

- the UI revision itself is real and test-backed; the main drift is now
  documentation, governance wording, and placeholder-decision data inside the
  new shells
- one concrete provider regression was present in the route-map enrichment path:
  Google provider selection could be bypassed or diluted by string/provider
  mismatch plus implicit ORS/OSM fallback behavior
- several docs still instruct operators to start in old leaf tabs such as
  `Quote builder`, `Dispatch`, `Kent tenders`, and `Kent admin` instead of
  explaining the new shell entrypoints and the leaf workflows inside them
- the new KPI strips and alert banners currently use static/demo values in the
  new workflow views, so they should not yet be treated as operational truth
  without provenance, freshness, and ownership
- repo automation guidance was too loose about interpreter choice; agent/user
  instructions now need to enforce repo-venv execution as a control boundary

Control objectives for the next wave:

- `ITIL`: treat the new shell as a service transition; document ownership,
  supported workflows, incident paths, and change gates per view
- `ISO 9001`: make view purpose, inputs, outputs, and acceptance criteria
  explicit so shell changes remain reviewable rather than taste-driven
- `ISO 42001` / `NIST AI RMF` / `ISO 23894`: keep advisory/model-backed
  outputs bounded, explainable, and human-reviewed; do not let placeholder risk
  banners or inferred signals masquerade as governed truth
- `ISO 27001` / `ISO 27701`: keep role visibility, admin actions, audit
  attribution, and labor/person data exposure least-privilege and documented
- `Six Sigma`: remove workflow variance caused by stale labels, mixed role
  entrypoints, and duplicate shell semantics across docs and UI
- `C4` / `PlantUML`: refresh the dashboard-shell diagrams so the five-view
  workflow shell is the documented current-state architecture

Worker lanes for the remediation wave:

- Worker 1: docs and service-governance alignment
  Scope: README, onboarding, role matrix, roadmap/progress board, ownership and
  acceptance criteria for the five shell views.
- Worker 2: provider-parity and mapping-control hardening
  Scope: enforce strict Google-versus-ORS/OSM provider behavior across route
  enrichment, route maps, isochrones, and map rendering so selected provider
  matches actual behavior without cross-provider fallback.
- Worker 3: composition and quality hardening
  Scope: expand tests around shared theme/components, provider selection, view
  composition, role layout reset/deep-link behavior, and responsive regressions
  in the new shell.
- Worker 4: security, privacy, and AI-risk controls
  Scope: admin-action auditability, role-hidden surface verification, PII/labor
  data minimization, governance for advisory/media/model-backed outputs, and
  repo-venv execution discipline in contributor/agent instructions.

Orchestrator execution contract:

| Worker | Primary lane | Standards focus | Required inputs | Completion evidence |
| --- | --- | --- | --- | --- |
| Worker 1 | docs and service-governance alignment | ITIL, ISO 9001, Six Sigma | README, onboarding, role matrix, roadmap, progress board, operator stories | shell labels and role entrypoints match code; owner and acceptance language is explicit; changelog/docs sync complete |
| Worker 2 | provider-parity and mapping-control hardening | ISO 9001, Six Sigma, ISO 27001 | routing provider selection, map rendering, route enrichment, isochrone paths, provider tests | Google selection remains Google-only; no silent ORS/OSM fallback; targeted regression suite passes in repo venv |
| Worker 3 | shell composition and regression hardening | ITIL, ISO 9001, C4/PlantUML | dashboard shell/views/components, role layout logic, deep links, UML sources | shell/view composition stays deterministic; reset/deep-link regressions are covered; architecture diagrams still describe the current shell |
| Worker 4 | security, privacy, AI-risk, and repo-venv discipline | ISO 27001, ISO 27701, ISO 42001, ISO 23894, NIST AI RMF | authz boundaries, admin actions, labor/person data surfaces, advisory/model-backed output handling, AGENTS/README execution rules | least-privilege visibility holds; advisory outputs remain reviewable and bounded; contributor/agent execution stays inside repo venv |

Promotion gate for this wave:

- no worker lane closes on prose alone; each lane needs matching code/docs/tests evidence
- do not mark placeholder KPI or alert content as decision-grade until provenance, freshness, and owner are explicit
- do not accept provider parity work while any Google-selected path can silently render or compute through ORS/OSM
- keep the generated PlantUML/C4 suite aligned with the implemented shell before declaring the UI revision governance-complete

## Targeted Advancement Wave (2026-04-02)

This wave should prioritize four bounded advances rather than reopening the
already-landed shell refactor.

### 1. Sourced shell signals and KPI provenance

Problem:

- the shell-level KPI strips and alert banners still use scaffold values in the
  five top-level views
- the UI now tells the truth that these are placeholders, but it still does not
  expose real source, freshness, owner, or unknown-state semantics

Roadmap:

- define a normalized signal contract for shell metrics and alerts covering
  `signal_id`, source module, owner, refresh cadence, stale threshold,
  confidence/review state, and fallback behavior
- replace per-view hard-coded KPI/alert payloads with shared signal builders so
  shell views render from one reviewed contract rather than free-form literals
- show freshness, owner, and advisory-versus-decision-grade state in the
  shared rendering layer instead of burying those rules in each view
- fail closed to explicit unknown/stale states when data is unavailable or
  outside freshness policy

Completion evidence:

- each top-level shell view renders from a shared signal contract
- no hard-coded decision-looking KPI/alert payloads remain in shell views
- tests cover fresh, stale, unknown, and advisory-only signal rendering

### 2. State-addressable shell and regression hardening

Problem:

- the current shell supports basic `view=` landing, but not a support-grade,
  shareable workspace-state contract
- session state, query params, and role layout are still only loosely coupled,
  which is enough for landing but not enough for reproducible support links or
  logged incident snapshots

Roadmap:

- separate simple navigation params from durable workspace state:
  `view=` remains the landing selector, while heavier workflow state moves into
  a normalized workspace-state layer
- define a support-safe share model that can reconstruct a user's shell,
  child workflow, filters, and selected records without leaking secrets or
  high-risk person data directly into the URL
- prefer compact URL-safe state for low-risk filters and navigation, and use
  persisted snapshot IDs for heavier or sensitive workspace state that should
  be reopened for support/audit
- add explicit state canonicalization so role changes, stale session keys, and
  hidden-tab constraints cannot resurrect invalid or unauthorized surfaces

Current progress:

- phase 1 is now landed:
  a normalized `ws` workspace-state payload exists for supported shell and
  operations child-workflow contexts, and canonical query-state writes are now
  emitted from quote, planner, dispatch, and operations-diary navigation paths
- the remaining gap is phase 2:
  persisted snapshot IDs for heavier or sensitive support replay, plus richer
  form/filter reconstruction for quote and other workflow-heavy surfaces

Completion evidence:

- users can open a reproducible workspace state for supported workflows instead
  of only a coarse `view=` landing
- support/audit flows can record and reopen workspace snapshots deterministically
- tests cover query-param normalization, role/layout reset, hidden-tab
  rejection, and snapshot rehydration

### 3. Architecture surface normalization

Problem:

- the UML suite exists, but the next value is governance, not more diagrams
- the repo needs one whole-system metasystem view plus a small reviewed set of
  child diagrams that stay aligned with the five-view shell and the new
  workspace-state/data-contract layers

Roadmap:

- treat `docs/rendered/plantuml/supermega_01.puml` as the metasystem entry
  surface and `docs/rendered/plantuml/dashboard_shell.puml` plus the existing
  child views as the reviewed drill-down set
- extend architecture guidance so any material shell-state or data-contract
  boundary change triggers a `scripts/build_supermega_uml.py --check`
  validation pass
- avoid creating extra diagram families unless they explain a real new control
  boundary; keep the whole-system view plus child views small and stable

Completion evidence:

- architecture docs name the metasystem view and reviewed child diagrams
- UML freshness is checked when shell topology or control boundaries move
- no competing hand-curated architecture surface drifts away from generated UML

### 4. Operational data contracts

Problem:

- the UI increasingly depends on decision-adjacent metrics, but the contract
  around source, owner, freshness, and fallback is still implicit

Roadmap:

- define one normalized operational-data contract for shell signals and similar
  review surfaces covering source, owner, refresh cadence, freshness SLA,
  stale threshold, fallback/unknown behavior, and advisory-versus-decision-grade
  classification
- keep the contract close to producer code and reference it from architecture
  and roadmap surfaces rather than duplicating field semantics in many docs
- require all new shell metrics, alerts, and review panels to declare the
  contract before they can be promoted as operational truth

Completion evidence:

- a reviewed contract exists and is referenced by shell-signal producers
- new decision-looking shell data cannot ship without owner/freshness/fallback
  semantics
- tests verify contract-driven stale/unknown-state rendering and auditability

### 5. Customer-facing tracking and receipt surfaces

Problem:

- the repo already contains live telemetry, ETA, dispatch-share, observer-outbox,
  and delivery-receipt-adjacent primitives, but there is no public-safe or
  customer-safe surface for a shareable tracking page, printable status receipt,
  or low-friction delivery-status update flow
- customers increasingly expect Domino's/Uber/Taxibox-style visibility:
  live vehicle state where allowed, ETA, status progression, proof/receipt
  artifacts, and a support-safe page they can reopen without joining the
  internal operator shell

Roadmap:

- define a dedicated customer-tracking contract rather than reusing internal
  shell or support-state payloads:
  one public-safe page model for status timeline, ETA, current stage, and
  optional vehicle-map visibility; one printable/shareable receipt model for
  post-delivery evidence and summary
- anchor the backend on existing primitives where possible:
  `analytics/live_data.py` for vehicle/ETA inputs,
  `analytics/adaptive_policy.py` for governed ETA modifiers,
  dispatch-share and observer/outbox patterns for issuance/audit/receipt
  delivery semantics
- require tokenized, scoped, expiring access rather than raw query-param or
  session-state mirroring:
  customer links and internal support replay links must be separate products
  with different scopes, TTLs, and audit expectations
- classify every exposed field as public-safe, customer-confidential,
  internal-only, or admin-only, and do not expose worker/labor/admin/internal
  notes on customer pages
- make inferred values explicit:
  ETA, delay risk, disruption, or advisory status must declare freshness,
  uncertainty class, and fallback behavior so the page does not imply a legal
  or operational guarantee it cannot support

Completion evidence:

- a reviewed customer-tracking/receipt contract exists with scope, expiry,
  revocation, and field-classification rules
- one bounded customer page can render status timeline plus ETA from approved
  sources without exposing internal-only state
- printable/shareable receipt output is generated from reviewed delivery/event
  evidence rather than ad hoc UI text
- tests cover token expiry/revocation, hidden-surface denial, stale ETA
  downgrade, and public-origin misconfiguration fail-closed behavior

### Worker assignments for this wave

| Worker | Lane | Primary outcome | Standards focus |
| --- | --- | --- | --- |
| Worker 1 | sourced shell signals | replace scaffold KPI/alert payloads with contract-backed signals | ITIL, ISO 9001, ISO 42001, Six Sigma |
| Worker 2 | state-addressable shell and regression hardening | make workspace state reproducible and shareable without unsafe URL drift | ITIL, ISO 9001, ISO 27001, ISO 27701 |
| Worker 3 | architecture surface normalization | keep one metasystem UML/C4 view and the reviewed child diagrams aligned | C4/PlantUML, ISO 9001 |
| Worker 4 | operational data contracts | formalize freshness/owner/fallback/advisory semantics for decision-adjacent data | ISO 9001, ISO 42001, ISO 23894, NIST AI RMF |

Adjacent future lane:

- Worker 1 or a later customer-experience lane can own customer-facing tracking
  and receipt surfaces once the state and data-contract layers are stable

Sidecars:

- docs sidecar: active for this planning wave because the next control surfaces
  needed explicit normalization
- UML sidecar: inactive until shell topology or control boundaries actually
  change
- commit sidecar: inactive until a publish/checkpoint request exists

## Documentation TODOs

- Keep `spec.md`, `plan.md`, `status.json`, and `devlog.md` current during the remediation milestone.
- Keep `docs/progress_status_board.md` and this roadmap synchronized after any feature status changes.
- Keep `docs/known_bad_cases.md` current when audits find, fix, downgrade, or accept bugs and bad cases.
- Keep `docs/service_blueprint.md` current when lifecycle, customer,
  notification, worker, support, or job-completion expectations change.
- When role/shell ownership changes, update `docs/ui_role_coverage_matrix.md`
  first, then refresh `docs/operator_user_stories.md` and
  `docs/usage_onboarding_guide.md` from it.
- Validate the deliverable status tables against the current code and tests.
- Decide whether `corkysoft/src/dashboard` remains a packaging stub or should be wired to the main Streamlit entry point.
- Maintain `docs/contributor_docs_sync.md` as the contributor-facing rule for README/ROADMAP/progress-board/docs alignment after feature or refactor changes.
- Keep `docs/modules.md` updated when module responsibilities or entry points change.
- Maintain the generated UML control surface in `docs/UML_INDEX.md`, `docs/rendered/plantuml/`, and `docs/rendered/svg/`, with `scripts/build_supermega_uml.py` as the authoritative generation and validation path.
- Add analytics documentation for historical ingest readiness, lane governance, and planner consumption defaults.
- Add a system architecture diagram (truck ↔ server ↔ cloud).
- Keep `docs/positioning.md` aligned with supported integrations and product focus.
- Keep product framing consistent: Corkysoft is a system of decision layered
  over removals operations, not merely a dashboard or passive system of record.
- Maintain `docs/operator_user_stories.md` as the actor-based product truth, including role coverage for labor, maintenance/compliance, inventory/suppliers, and system admin.
- Maintain `docs/usage_onboarding_guide.md` as the formal onboarding/help truth for daily usage.
- Maintain `docs/ui_role_coverage_matrix.md` as the canonical role-to-surface ownership map.
- Maintain `docs/payroll_and_labor_analytics.md` as the canonical payroll-preparation and labor-statistics truth.
- Maintain `docs/naive_user_tester_notes.md` as the plain-language user-testing log and convert repeated friction into backlog items.
- Maintain `docs/commercial_workflow_lifecycle.md` as the quote -> tender -> awarded-work lifecycle truth.
- Add a dedicated workflow spec for international/compliance-heavy work so
  paperwork, insurance, tender, customs, and audit requirements are modeled as
  explicit requirements/proposal/governance states rather than left implicit.
- Maintain the Corkysoft/SB/ITIR boundary explicitly: Corkysoft is workflow
  truth, SB is downstream interpretible state, and ITIR is orchestration/context.
- Maintain `docs/corkysoft_mcp_v1.md` as the canonical read-only-first MCP
  adapter contract above Corkysoft-owned producer logic, and keep it aligned
  with the implemented local registry/bridge in `corkysoft/mcp/`.
- Maintain `docs/corridor_detection.md` alongside corridor model updates.
- Keep `docs/corridor_schema_plan.md` aligned with corridor table changes.
- Maintain `docs/corridor_defaults.md` and `docs/cluster_template_au.md` when thresholds or clusters change.
- Maintain `docs/corridor_opportunity_report.md` alongside opportunity scoring changes.
- Maintain `docs/corridor_opportunity_view.md` alongside the report schema.
- Maintain `docs/ams_backend_docs_playbook.md` as the onboarding baseline for backend docs.
- Add a "Move/Job Lifecycle" doc that maps current CLI + dashboard flow states end-to-end.
- Add a data-model glossary mapping relocation terms (assignee/move/shipment) to Corkysoft table and field names.
- Add a pricing-engine documentation page that lists margin inputs, formulas, and data dependencies.
- Maintain `docs/adaptive_learning_loop.md` as the canonical bounded-learning spec for situational-awareness inputs, learned policy state, and staged rollout boundaries.
- Validate and maintain `docs/kent_ams_integration.md` against real Kent AMS payloads and auth constraints.
- Execute `docs/kent_ams_integration_roadmap.md` Phase 0 contract lock with sample payload coverage and enum catalog.
- Expand Kent adapter from import-only to operator triage with tender pre-scoring calibration and peak-season weighting validation.
- Review `GET /kent-ams/tenders/calibration` weekly and tune score weights only when band monotonicity and margin-error metrics improve.
- Validate route/location scoring against historical lane outcomes to avoid over-prioritizing unfamiliar but risky tenders.
- Integrate en-route spare capacity signals more broadly into ingestion triage and operational review; quote creation already surfaces benchmark overlays plus backhaul-aware discount guidance.
- Normalize state/national closure, traffic, and weather signals into a common disruption input layer before using them in quote or ETA policy updates.
- Keep adaptive policy learning bounded and reviewable: bootstrap explicit parameters first, then add proposal generation, then approval/audit flows.
- Maintain `docs/multi_truck_route_load_optimization.md` and align implementation milestones to its transfer/split/sequence constraints.
- Maintain `docs/planner_interaction_model.md` as the canonical planner-UX spec above `job_segments`.
- Maintain `docs/operations_diary_workflow.md` as the canonical manager-facing day/week cockpit spec across planning, usage review, and financial follow-through.
- Maintain `docs/job_cost_and_invoice_reconciliation.md` as the canonical invoice/bill review spec tied to job and segment truth.
- Maintain `docs/corkysoft_sb_itir_coverage_audit.md` as the truth table for
  what Corkysoft already covers versus what is only conceptual or planned in
  SB/ITIR.
- Maintain `docs/sb_itir_downstream_contract.md` as the transport-agnostic
  downstream contract for Corkysoft diary/planner/reconciliation outputs.
- Maintain `docs/planner_diary_patterns_for_sb_itir.md` as the pattern-extraction
  note for future SB/ITIR lens design.
- Maintain the implemented hybrid planner: job-first and map/corridor-first selection, historical overlap surfacing, and profitability-aware draft leg generation before assignment.
- Replace the anonymous/manual role switcher with Google-authenticated local users while keeping role-layout policy simple and reviewable.
- Follow the first Google-auth/dashboard-user slice with per-action audit attribution and tighter admin/user governance.
- Maintain a focused auth red-team plan now that executable browser-based auth checks exist for the local harnessed states.
- Keep the implemented browser-testable auth-state harness current so Playwright continues to cover anonymous, misconfigured, unauthorized, inactive, and hidden-tab-denial flows without needing live Google automation.
- Keep tunneled/public-origin auth behavior explicit so `redirect_uri` / origin mismatches fail clearly and remain documented as deployment/configuration errors.
- Support a temporary, explicit owner/testing shortcut where successful Google logins auto-provision as local admins, while preserving a clean path back to proper per-user roles.
- Treat role-hidden admin tabs as part of the authz boundary; do not allow query-param navigation or stale session state to re-expose them.
- Keep bootstrap-admin seeding one-shot and explicit; do not let lingering env vars silently reassert admin access after user setup exists.
- Deepen the implemented Planner workflow inside `Operations` from the current hybrid scaffold toward richer site-aware and more interactive visual planning.
- Deepen the implemented `Operations diary` workflow inside `Operations` so managers can review day/week workload, usage, tasks, and invoice/bill exceptions with stronger sourcing, task follow-through, and reconciliation state.
- Google/ORS parity is now aligned across Planner preview and saved-route Folium overlays; continue auditing remaining route/map surfaces and fallback behavior for strict parity.
- Planner now supports accepted site-risk interpretation plus Google-first 360/media attachment, advisory CV/volume scaffolding, and first derived site constraints. The current priority should favor real media/CV ingestion and reviewed outputs; deeper interpreted constraint logic should follow once that evidence pipeline is more real.
- Route geometry should be enriched automatically during ingest/load/seed flows; manual population is no longer the intended operator path.
- The quote-save historical job path now attempts route-geometry enrichment automatically; continue closing any remaining non-test creation/sync seams that bypass shared geometry enrichment helpers.
- Model-backed walkaround CV and volume-estimation services should attach to quote/job records as reviewed evidence first, then feed planner constraints only after acceptance/correction.
- Make accepted site evidence produce explicit planning consequences, not just labels: truck suitability, shuttle need, labor uplift, access-time risk, and loading-delay risk should become planner-consumable constraints.
- Continue to keep the raw `Operations` segment editor as an advanced/manual fallback rather than the primary workflow.
- Validate the new inventory requirement / shortage model against real operating patterns, especially Crusader's container-heavy workflows.
- Replace generic warehouse stage entry with requirement/container picklists, explicit next-action buttons, and barcode/QR-assisted capture.
- Extend inventory execution from the newly-implemented constrained pick/pack/load and substitution governance into richer warehouse controls such as picklists, scanning, and container-specific execution tooling.
- Keep custody/location state current in operations so depot, truck/container, in-transit, site, returned/storage, and exception locations stay trustworthy in practice.
- Extend inventory architecture handling beyond the current baseline so containers, consumables, reusable assets, serialized/tagged gear, and other workflows each get fit-for-purpose behavior.
- Maintain `docs/inventory_execution_workflow.md` as the canonical workflow spec for pick / pack / load, substitution, and container-heavy execution.
- Implemented the routed call-intelligence foundation: `Calls` tab, call sessions, call legs, routing-event history, ambient office transcript sessions, transcript artifacts, extracted-action review, worker time capture events, and append-only downstream egress preparation.
- Keep fake transcript generation as the default workflow-testing ingest path for both phone and ambient-session workflows until live telephony/IVR is implemented.
- Treat WhisperX-WebUI, or a directly compatible successor, as the default external ASR/call-intelligence backend for Corkysoft rather than embedding transcription logic locally.
- Implement and harden the Corkysoft send/receive interface to WhisperX-WebUI: audio submission, async task polling, transcript/artifact finalization, service separation, and failure/retry handling.
- Fold newer livestreaming transcription-completion concepts back into the WhisperX-WebUI seam where practical so Corkysoft only has one reviewed transcription ingress contract.
- Continue worker time capture rollout across app, WhatsApp, and voice/landline call-in paths with transcription/review for low-confidence events.
- Fixed the live rerun-compatibility blocker in `Calls` and `Inventory`; continue checking other operator surfaces for any remaining direct `st.experimental_rerun()` usage under the current Streamlit build.
- Bring dispatcher role-layout defaults into line with the live role story; the current default focus tabs still omit `Calls` even though routed call handling is now part of dispatcher work.
- Dispatcher code defaults already include `Calls`; the remaining gap is stale stored layout state, which should be repaired via the new dispatcher-layout repair prompt.
- Expose reviewed call-derived worker-time state more clearly in `Staff` / `Driver shifts`; review is now available in labor surfaces, and `Driver shifts` now compares accepted call-derived worker-time against imported VEHICLE_DRIVER rows with explicit mismatch classes and timing-drift visibility. Keep start/end tolerance rules deferred until worker-time capture becomes richer than simple `clock_on` / `clock_off`.
- Implemented the first payroll-preparation and labor-analytics layer above labor operations: aggregate-first pay forecasting, overtime/hours/cost distributions, plan-vs-actual views, confidence/anomaly panels, labor-cost drivers, export-ready labor summaries, read-only API summaries, and a basic recorded absence/leave model.
- Keep absence analytics grounded in explicit recorded leave/absence rows; do not regress to inferring sick days from missing shifts or weak worker-time signals.
- Add operational accommodation/availability support signals so Planner and Dispatch can surface remote/peak-period lodging pressure early enough to book.
- Rerun the MCP/UI role-completion wave for dispatcher, warehouse/crew, labor planner, and system admin after the rerun-compatibility fix lands; current persistence is good, but uninterrupted operator flow is not yet good enough.
- Add outbox delivery retry/idempotency tests once the StatiBaker delivery worker exists.
- Strengthen inventory custody conflict handling beyond current latest-location assertions.
- Add barcode/QR execution paths on top of the constrained warehouse inventory flow.
- Turn the new site media / assessment / advisory inference scaffolding into real service-backed ingestion and review flows for street-view, walkaround, and visual last-mile optimisation.
- Add accommodation/provider-side operational support logic after the base support-signal model is wired into Planner and Dispatch.
- Validate Kent tender policy defaults and override workflow against live operator usage before adding any deeper async solver.
- Separate operator Kent workflows from admin policy/reason-code management in both docs and dashboard UX; the dashboard now gates Kent policy/reason-code writes to `system_rollout_admin`, with broader live-usage validation still pending.
- Govern hard-block semantics so only safety/legal/compliance categories can block work.
- Make importer `dry_run` behavior side-effect free and documented consistently.
- Migrate `FLEET`, `STAFF`, and `SUPPLIERS` operational data flow to Google Sheets-first connectors and treat local `.xlsx` workbooks as fallback only.
- Add a shared operations-workbook sync path so fleet/staff/supplier state can be refreshed together from one Google Sheets source.
- Add per-tab override configuration for shared operations-workbook sync only if real workbook naming diverges from `STAFF` / `SUPPLIERS` defaults.
- Make `job_segments` the canonical operational planning unit for assigning trucks and workers.
- Build readiness evaluation for assignments covering rego, COI, service due dates, worker roles, and worker compliances.
- Keep spreadsheet imports import-only in v1; do not write assignments back to production sheets.
- Split operator assignment workflow from admin sync/policy workflow for operations planning.
- Replace remaining Staff/Fleet reliance on `present_driver` as assignment truth with segment-based planning views and reconciliation.
- Add a maintenance/compliance cockpit for due-soon and blocked rego, COI, service, and worker compliance items.
- Add native role/compliance assignment flows so readiness governance can be maintained inside Corkysoft rather than only through sheets.
- Replace `VEHICLE_DRIVER` as the primary roster/planning surface with native labor planning and imported-shift reconciliation.
- Link inventory allocation and supplier coordination directly to `job_segments` so stock planning cooperates with truck/worker planning.
- Keep the Dispatch workflow inside `Operations` as the primary job-centric execution surface across trucks, workers, inventory, suppliers, and readiness flags.
- Deepen the implemented manager-facing day/week diary so it links jobs, tasks, vehicle usage, staff usage, labor reconciliation, customer invoice readiness, and subcontractor-bill reconciliation with reviewable operational truth.
- Add persistent diary tasks for day/week/job/segment follow-through that do not need to masquerade as `job_segments`.
- Add invoice and subcontractor-bill review records tied to jobs, with explicit exception states when operational truth is incomplete.
- Implemented a Corkysoft-native observer outbox for diary/planner/reconciliation
  review state, including persisted review/task families and explicit export
  for planning snapshots and reconciliation exceptions.
- Operations diary now exposes observer-outbox rows directly so managers/admins
  can inspect emitted envelopes, payloads, and provenance without leaving the
  dashboard.
- Add delivery receipt / watermark semantics for the observer outbox once
  deployment posture matters.
- Reuse Corkysoft planner/diary workflow patterns in SB/ITIR only as downstream
  summary/review lenses, not as a second operational cockpit.
- Extend the implemented Corkysoft MCP scaffold and local bridge for bounded
  read-only tools such as profitability summary, dispatch recommendations,
  operations-diary summary, and quote-guidance preview; do not expose mutable
  dispatch/admin tools until auth, audit, and operator-policy governance are
  stronger.
- Keep the JSON bridge as the supported default MCP entrypoint until the
  optional FastMCP server earns separate transport-level support.
- Add unresolved supplier-exposure aging inside Corkysoft so managers can see
  received-but-unreconciled liabilities, billing latency, and top overdue jobs
  before the fuller SB time surface exists.
- Add cutover metrics, rollback instructions, and CSV snapshot/export rules for controlled spreadsheet decommissioning.
- Track spreadsheet decommissioning workflow-by-workflow with explicit cutover status, fallback mode, checklist completion, and last-drill timestamps.
- Use workflow-level native-usage, fallback-use, open-issue, and snapshot-consumer metrics to decide when each sheet can move to fallback-only.
- Derive cutover metrics from live operations state and logged review/drill/fallback events instead of relying on manual metric entry.
- Add guarded rollout recommendations so workflow status transitions are applied from current evidence rather than by manual convention.
- Add approval-backed rollout promotions so ops requests, commercial approval, and final status transitions are all visible in one audit trail.
- Suppress irrelevant `historical_jobs` warnings on operational tabs so Dispatch, Operations, Fleet, and Kent surfaces do not inherit analytics-only noise.
- Review whether the default landing surface should remain analytics-first or become role-aware / operational-first for day-to-day users.
- Record that the major five-view shell revision is now implemented and keep all
  user-facing docs aligned with `Quote` / `Pricing Intelligence` / `Network` /
  `Operations` / `Admin`.
- Replace static/demo KPI and alert content in the new workflow shells with
  sourced operational metrics, explicit freshness, and unknown-state handling
  before treating those banners as decision-grade signals.
- Fix the current role-layout reset/repair flow so session-backed layout widgets do not throw `StreamlitAPIException` when operators repair or reset a role layout.
- Make role-aware deep-linking deterministic in development and authenticated runs so `view=` routes can land users in the owning role instead of silently inheriting a stale session layout.
- Sweep remaining operator surfaces for direct `st.experimental_rerun()` usage before the next live testing wave.
- Re-section mixed surfaces such as `Fleet`, `Inventory`, and `Staff` so execution, review, and admin/governance work are visually separated before further usability testing.
- Clarify planner UX before implementing it: docs now need to say how roadway extents, corridor overlap, site context, traffic/routing considerations, and resource allocation should combine into a click-heavy planning flow.
- Land the docs-only role coverage pass before changing UI copy or tab structure, so role ownership is explicit before any surface reshuffle.

---

## 🧱 1. Core Routing & Cost Engine

| Deliverable                           | Status | Description                                        |
| ------------------------------------- | :----: | -------------------------------------------------- |
| **OpenRouteService integration**      |    ✅   | Working API calls for distance/duration.           |
| **CLI tool (`routes_to_sqlite.py`)**  |    ✅   | Add, run, list, and import jobs.                   |
| **SQLite persistence (`routes.db`)**  |    ✅   | Stores all job and geocode data.                   |
| **Geocode caching**                   |    ✅   | Prevents repeated lookups.                         |
| **Cost calculator (hourly + per-km)** |    ✅   | Calculates total cost and components.              |
| **Error handling / back-off**         |    ✅   | Handles rate limits and errors gracefully.         |
| **Address normalisation (AU)**        |    ✅   | Expands street abbreviations (e.g., cr → Circuit). |
| **CSV import/export**                 |    ✅   | Batch job ingestion.                               |
| **Folium route visualisation**        |    ✅   | Produces full-route HTML maps.                     |
| **README.md**                         |    ✅   | Complete and public-ready.                         |
| **Unit tests**                        |   🧩   | Core suites exist; expand DB/API coverage.         |

---

## 🗺️ 2. Mapping & Visualisation

| Deliverable                           | Status | Description                                 |
| ------------------------------------- | :----: | ------------------------------------------- |
| **Multi-route Folium map**            |    ✅   | Working map output.                         |
| **CustomIcon fix**                    |    ✅   | Bug resolved.                               |
| **Break-even / margin overlays**      |   🧩   | Present in analytics; confirm full dashboard wiring. |
| **Interactive dashboard (Streamlit)** |   🧩   | Implemented and usable; workflow polish, governance, and some specs still lag reality. |
| **Profit & volume heatmaps**          |   🧩   | Implemented analytics surface; validate lane profitability overlays and current-state docs. |

---

## 📦 3. Data Model & Integration

| Deliverable                  | Status | Description                       |
| ---------------------------- | :----: | --------------------------------- |
| **Jobs + geocode tables**    |    ✅   | Implemented.                      |
| **Schema migration support** |    ✅   | Handles new columns.              |
| **Client registry & dedupe** |    ✅   | Quote builder stores clients, flags duplicates, and allows quotes without forcing a client record. |
| **Historical job import**    |    ✅   | CSV/history ingest now records run coverage, row issues, readiness status, and Fleet-admin visibility; continue expanding source coverage. |
| **$ per m³ calculation**     |   🧩   | Derived metrics exist and now sit on a stronger ingest-governance base; continue validating source consistency. |
| **Corridor / lane table**    |    ✅   | Canonical clusters, directional lanes, corridor groups, assignment status, promotion governance, and planner-safe consumption are implemented. |
| **Modifier tables**          |    ✅   | Access, packing, seasonal rules in schema. |
| **Integration staging schema** |   🧩 | Contract exists; operational workflow and failure handling still need alignment. |
| **CSV/API connectors**       |   🧩   | Internal API endpoints and importers exist; external-system hardening remains. |

---

## 🚛 4. Operational Business Logic

| Deliverable                                              | Status | Description                            |
| -------------------------------------------------------- | :----: | -------------------------------------- |
| **Metro vs regional logic (≤100 km)**                    |    ✅   | Rule defined.                          |
| **Base-rate schedule (Sunshine Coast 120 → Cairns 185)** |    ✅   | Encoded in lane base rate table. |
| **Packing / bad-access fees**                            |    ✅   | Modifier tables include packing + access fees. |
| **Seasonal margin uplift (20–80 %)**                     |   🧩   | Seasonal uplifts table exists; validate usage. |
| **Backhaul / container sharing**                         |   🧩   | Quote backhaul detection and discount guidance are implemented, and Dispatch now surfaces persisted spare-capacity, container-pressure context, share/reallocation recommendations, and explicit operator response actions; broader operational container-sharing and under-/over-utilisation handling still pending. |
| **Truck / driver cost baselines**                        |    ✅   | Base fuel/driver/maintenance + overhead parameters drive break-even engine. |
| **Private cost component ledger**                        |    ✅   | Record crew, truck, fuel and other cost inputs per job inside SQLite.       |

---

## 🧮 5. Analytics & Statistical Modelling

| Deliverable                            | Status | Description                                  |
| -------------------------------------- | :----: | -------------------------------------------- |
| **Price distribution $/m³ histogram**  |    ✅   | Jobs sorted left→right by $/m³; bar = count. |
| **Break-even + margin bands**          |    ✅   | Visual overlay on histogram.                 |
| **Loss-leader detection**              |    ✅   | Identify sub-margin jobs.                    |
| **Regression / corridor model**        |   🧩   | Profitability insights now include baseline and corridor-aware margin-per-m³ regression over distance/season, fit-improvement reporting, and holdout trust signals; next steps are suppression/promotion rules and rolling backtests before holiday interactions, uncertainty ranges, or other broader model features. |
| **Terrain & temperature factors**      |   🔜   | Weight costs for harsh routes.               |
| **Driver / truck performance metrics** |   🔜   | Wear, reliability, fuel efficiency.          |

---

## 📸 6. RFID / Camera / Audit System

| Deliverable                            | Status | Description                             |
| -------------------------------------- | :----: | --------------------------------------- |
| **Technical architecture spec**        |    ✅   | Complete multi-layer doc.               |
| **Media ingest doc (PEC/bodycam)**     |    ✅   | Capture→upload pipeline documented.     |
| **Data-model integration (PEC/media)** |   🧩   | Fields drafted; implementation pending. |
| **Pre-Existing-Condition capture**     |   🔜   | Two-photo workflow + customer sign-off. |
| **Event-based bodycam clips**          |   🔜   | Short triggered recordings.             |
| **Video/call processing (Frigate/Whisper + SFM)** |   🔜   | Stub: use Frigate for video detection, Whisper for call transcripts, and SFM to assess access constraints. |
| **Claim-risk scoring**                 |   🔜   | Use dispute data to adjust pricing.     |
| **Hash-verified storage**              |   🔜   | SHA-256 for insurer integrity.          |
| **Privacy & consent controls**         |   🔜   | Face-blur + role-based access.          |

---

## 📊 7. Dashboards & Reporting

| Deliverable                              | Status | Description                                |
| ---------------------------------------- | :----: | ------------------------------------------ |
| **CLI report (`list`)**                  |    ✅   | Clean console output.                      |
| **Streamlit dashboard (MVP)**            |   🧩   | Route map + distribution + quote + Kent triage exist; lane-governance admin surfaces, grouped proposal review, Kent review summaries, and lane-status trust-boundary controls are now present, but broader admin/operator separation and spec cleanup remain. |
| **Insurance / audit bundles (PDF)**      |   🔜   | One-click job evidence packs.              |
| **Automated CSV / Google Sheets export** |   🧩   | Helpers produce CSV-ready profitability summaries. |
| **API endpoints**                        |   🧩   | Internal JSON/REST endpoints exist; auth, governance, and external contracts remain incomplete. |

---

## 🔐 8. Security & Compliance

| Deliverable                       | Status | Description                           |
| --------------------------------- | :----: | ------------------------------------- |
| **Immutable manifests / hashing** |   🧩   | Partially in design.                  |
| **Privacy safeguards**            |   🔜   | Implement face-blur, RBAC, retention. |
| **PIA / Ethics review**           |   🔜   | Required for video + RFID data.       |

---

## 🪜 9. Documentation & Governance

| Deliverable                        | Status | Description                         |
| ---------------------------------- | :----: | ----------------------------------- |
| **README.md**                      |   🧩   | Current after remediation; keep product-centered and aligned with code reality. |
| **GM summary**                     |    ✅   | Delivered (non-technical overview). |
| **Architecture diagram**           |   🔜   | Truck ↔ server ↔ cloud schematic.   |
| **Analytics README / docs folder** |   🧩   | Core ingest/lane governance work has shipped; consolidate the analytics/governance docs into a cleaner operator-facing summary. |

---

## 📈 Overall Progress Snapshot

| Domain                     | Progress |
| -------------------------- | -------- |
| Core Routing & Costing     | ✅ 90 %   |
| Mapping & Visualisation    | 🧩 60 %  |
| Business Logic & Rates     | 🧩 50 %  |
| Analytics & Stats          | 🔜 30 %  |
| RFID / Camera Audit        | 🧩 40 %  |
| Dashboards & Reports       | 🔜 25 %  |
| Documentation & Governance | 🧩 70 %  |

---

### Current Truth

- Routing, costing, caching, database schema, and core CLI are in place.
- Streamlit dashboard is implemented and used as the main operator surface.
- Kent tender triage exists internally with provisional governance and contract assumptions.
- Adaptive policy learning now ingests closure, weather, and traffic severity feeds, keeps the bounded parameter defaults, and auto-nudges weather, closure, and ETA multipliers while an approval workflow for those nudges is still being formalized.

### Highest-Priority Remaining Work

1. Align docs, roadmap, and operator stories to one source of truth.
2. Validate Kent against real payloads and real operator behavior.
3. Tighten governance for overrides, hard-blocks, and policy review.
4. Formalize corridor/lane and multi-truck policy before deeper optimization work.
5. Define the adaptive policy review and approval workflow before tying parameter changes directly to quote or ETA recomputation.

---

## Next Phase Blockers (2026-03-04)

- Historical job ingestion coverage and validation (analytics need reliable data).
- Corridor/lane model formalization (directional + bidirectional grouping).
- Quote recommendation + benchmarking logic (market medians, confidence, $/m3 distribution buckets, break-even overlays).
- Backhaul detection logic.
- End-to-end dashboard workflow clarity (ensure docs and UX use the same operator story).
- Phantom corridor detection + gravity model exploration.
- Opportunity scoring that combines gravity demand with $/m3 distributions.
- Adaptive-policy review and audit workflow that governs automatic nudges for pricing and ETA parameters.

## High-Leverage Near-Term Features

- Historical job import pipeline (CSV + MoveWare exports).
- Corridor / lane detection and rollups.
- $/m³ market benchmarking overlays (median, percentile, break-even lines).
- Quote recommendation engine (corridor median + modifiers).
- Backhaul detection and discount suggestions.
- Job profitability scoring and risk tags.
- Corridor profitability heatmap layer.
- Automated corridor pricing adjustments.
- Kent ranking correctness, admin/operator separation, and governed hard-block handling.
- Adaptive policy review and approval flows for recorded disruption analytics.
