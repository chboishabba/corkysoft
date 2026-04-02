## 2026-03-31 (Bridge Governance Regression Expansion)

Deeper local governance pass after the 89-test state:

- MCP bridge tests now assert blank tool names fail with `input_error`
- MCP bridge tests now assert unknown tool names fail with `tool_error`
- the focused integrated suite passed at 98 tests with the bridge test file
  included

State implication:
- rerun cleanup is complete
- governance validation is materially deeper across helper, UI, envelope, and
  bridge behavior
- the main remaining work is the quote helper boundary judgment rather than an
  obvious defect backlog

## 2026-03-31 (Rerun Audit And Kent UI Governance Regression)

Follow-up local pass after the 88-test wave:

- repo audit confirms `dashboard/state.py` is now the only rerun helper owner
- there are no remaining direct rerun calls or local rerun wrappers outside the
  shared state layer
- Kent UI coverage now proves non-admin roles see admin write controls disabled,
  not only helper-level write gating

Validation:
- focused integrated suite passed at 89 tests

## 2026-03-31 (Narrow Cleanup Wave Landed)

The narrow cleanup wave completed and the integrated focused suite passed at 88
tests.

What landed:
- quote suggestion application now resets manual-override state inside the
  authoritative shared helper in `dashboard/state.py`
- planner, maintenance, and operations now route rerun behavior through the
  shared helper instead of keeping local rerun logic
- MCP tests now cover execution-error envelopes in addition to success and
  input-error envelopes

What remains:
- the remaining quote helper split is now mostly about boundary judgment rather
  than obvious duplication
- rerun cleanup should be verified by audit before another implementation lane
- deeper Kent and MCP scenario/payload validation is still the clearest open
  governance task

## 2026-03-31 (Narrow Cleanup Wave Assigned)

The rerun-wrapper backlog was re-scoped from a stale assumption to the actual
remaining local paths:

- planner fallback rerun logic
- maintenance local rerun helper
- operations direct rerun calls

The next narrow split is therefore:
- Worker 1: quote UI versus shared-helper boundary decision
- Worker 2: real rerun-wrapper backlog cleanup
- Worker 3: deeper Kent and MCP scenario validation

Why this split now:
- ITIL: these are the last small workflow-consistency risks in the dashboard
- ISO 9001: shared helper ownership now needs a final boundary decision, not
  another broad extraction
- Six Sigma: the remaining variation is localized and measurable

## 2026-03-31 (Final Boundary And Validation Cleanup Wave Partially Landed)

The next cleanup wave produced two real changes and one validated no-op, and
the integrated focused suite passed at 87 tests.

What landed:
- quote suggestion application now reuses `_apply_quote_suggestion` from
  `dashboard/state.py`
- MCP tests now cover `success_payload` and `error_payload` envelope helpers
- MCP docs now state that those envelope shapes are test-enforced

What the no-op lane established:
- `dashboard/components/calls.py` already reuses the shared rerun helper
- `dashboard/components/maps.py` does not currently own rerun behavior
- the rerun-wrapper backlog needs re-scoping to real remaining local wrappers

What remains:
- quote decision helpers still mix reusable logic with UI-local concerns
- broader Kent and MCP scenario validation remains beyond the current envelope
  and static contract checks

## 2026-03-31 (Final Boundary And Validation Cleanup Wave Assigned)

After the 85-test integrated pass, the remaining work is small enough to keep
parallel but narrow:

- Worker 1: quote decision-control cleanup
- Worker 2: remaining rerun-wrapper consolidation
- Worker 3: broader governance scenario validation

Why this split now:
- ITIL: the remaining service risk is mostly local workflow-control consistency
- ISO 9001: the best next move is to finish consolidating authoritative helper
  ownership and test the governed contract paths more deeply
- Six Sigma: the remaining variation comes from a few last duplicate helpers and
  under-specified governance scenarios rather than broad architectural issues

## 2026-03-31 (Boundary Cleanup And Live-Control Validation Wave Landed)

The three-lane cleanup wave completed and the integrated focused suite passed at
85 tests.

What landed:
- quote builder now reuses the shared route-label helper from `dashboard/state.py`
- planner and route maps now reuse the shared rerun helper from `dashboard/state.py`
- Kent governance now exposes the explicit `KENT_ADMIN_WRITE_ROLES` set
- MCP contract coverage now locks documented tool names and response version
- Kent/MCP docs now describe those enforced invariants

What remains:
- quote decision-control helpers still live mostly inside
  `dashboard/components/quote_builder.py`
- other components still keep local rerun wrappers, especially calls/maps
- live payload/scenario validation remains broader than the current static
  contract tests

## 2026-03-31 (Boundary Cleanup And Live-Control Validation Wave Assigned)

After the integrated focused suite passed at 46 tests, the next wave is narrower
and remains suitable for parallel non-blocking work:

- Worker 1: quote-builder boundary cleanup
- Worker 2: shared rerun/state helper consolidation
- Worker 3: Kent and MCP live-control validation coverage

Why this split now:
- ITIL: the operator shell is stable enough that the next service risk is local
  workflow-control drift rather than route landing or shell failures
- ISO 9001: the remaining defects are mostly ownership and control-boundary
  issues, so they should be split by authoritative module
- Six Sigma: duplicated helper paths are still a variation source and can be
  reduced independently from governance validation work

## 2026-03-31 (Next Parallel Wave Assigned)

After the prior three-lane wave landed, the next non-blocking split remains
appropriate and is narrower:

- Worker 1: operator reconciliation and execution completion
- Worker 2: outbound contract and governance hardening
- Worker 3: quote-builder and shared-state consolidation

Why this split still holds:
- ITIL: operator reconciliation remains the highest service-value unfinished path
- ISO 9001: contract and governance controls remain separable from workflow UI
- Six Sigma: quote-builder duplicate helper/state paths are a local variation source and can be reduced independently

## 2026-03-31 (Prioritized Remaining Roadmap And Worker Lanes)

Assessed the full remaining roadmap against the current status board and
architecture docs, then collapsed the open work into three priority bands.

Priority order:
- Priority 1: operator execution completion
- Priority 2: governance and contract hardening
- Priority 3: decision-quality and planner intelligence upgrades

Why this order:
- ITIL: stabilize the service path used by operators before extending adjacent
  governance and advisory layers
- ISO 9001: tighten controlled evidence, approval, and ownership boundaries
  before adding more scope
- Six Sigma: attack the defect-heavy execution and governance paths before
  model sophistication

Assigned non-blocking worker lanes:
- Worker 1: operator execution completion
- Worker 2: governance and contract hardening
- Worker 3: dashboard boundary and decision-quality polish

Primary docs updated:
- `plan.md`
- `docs/progress_status_board.md`
- `COMPACTIFIED_CONTEXT.md`
- `CHANGELOG.md`

## 2026-03-31 (Second-Pass Dashboard Boundary Polish)

Read the local planning and contributor docs before continuing the dashboard
decomposition wave, then aligned the next changes to explicit quality-control
goals:

- ITIL-style service objective:
  keep the dashboard shell supportable as role-aware workflows expand
- ISO 9001 style control objective:
  make module ownership explicit and auditable
- Six Sigma style defect objective:
  reduce variation caused by one mixed-responsibility control layer

What changed:
- split `dashboard/data_controls.py` into smaller internal control points for:
  - database initialization
  - dataset selection
  - routing-provider selection
  - historical ingest feedback
  - dataset loading
  - filter-state resolution
  - break-even updates
- added focused helper coverage in `tests/test_data_controls.py`
- updated `README.md`, `docs/modules.md`, and `docs/architecture.md` so the
  extracted dashboard control layers are now documented as real ownership
  boundaries
- added `docs/architecture_dashboard_shell.puml` as the C4-style PlantUML
  control view for the dashboard shell

What remains:
- `dashboard/data_controls.py` is cleaner but still large enough to justify a
  future file split if more sidebar logic accumulates
- `dashboard/components/quote_builder.py` still owns duplicate pin and
  route-label helpers that should eventually consolidate on the shared state
  path
- `dashboard/app.py` is slimmer but still the main orchestration hotspot

## 2026-03-30 (App Shell Decomposition Lanes)

After the shell and deep-link fixes landed, the next bottleneck is structural:
`dashboard/app.py` still owns too much orchestration even though the acute UI
bugs are fixed.

Current decomposition target:
- auth flow
- query-param routing
- shell selection
- dataset/filter setup
- tab composition
- cross-surface state hydration

Parallel non-blocking dev lanes assigned:
- Worker 1: extract auth flow + query-param routing
- Worker 2: extract shell selection + cross-surface state hydration
- Worker 3: extract dataset/filter setup + tab composition

Governance note:
- use `gpt-5.1-codex-mini` only for child lanes in this wave
- keep lane ownership disjoint except for final `dashboard/app.py` integration
  seams, which should stay minimal and explicit

## 2026-03-30 (UI Assessment And Remediation Lanes)

Docs-first alignment pass after a live UI/flow review of the dashboard shell and
role-specific surfaces.

Observed product truth:
- the live shell still frames most workflows as analytics-first via the global
  `Price distribution analytics` title/caption and left sidebar filters
- role-layout reset/repair can currently throw `StreamlitAPIException` because
  widget-backed session keys are mutated after instantiation
- `Kent admin` is still writable only for `system_rollout_admin`, so commercial
  owners should currently treat it as a governed review surface rather than a
  primary write surface
- remaining direct `st.experimental_rerun()` usage in live operator surfaces
  should be cleaned up before the next browser-led testing wave

Docs updated:
- README.md
- ROADMAP.md
- plan.md
- docs/operator_user_stories.md
- docs/ui_role_coverage_matrix.md
- docs/usage_onboarding_guide.md
- docs/progress_status_board.md
- CHANGELOG.md
- COMPACTIFIED_CONTEXT.md

Parallel non-blocking dev lanes assigned:
- Worker 1: shell chrome + role landing/reset hardening
- Worker 2: rerun-compatibility sweep for live operator surfaces
- Worker 3: mixed-surface sectioning improvements focused on labor/admin flow

Implementation changes:
- None yet in this pass. This entry records the docs/TODO alignment before the
  next implementation/testing wave.

## 2026-03-27 (Temporary Google Admin Auto-Provision Mode)

Added the safer temporary shortcut for the current sharing/testing phase.

What changed:
- successful Google logins can now auto-provision into `dashboard_users` as
  `system_rollout_admin` when `CORKYSOFT_AUTO_PROVISION_GOOGLE_ADMIN=1`
- the local user table remains populated, so this is easier to unwind later
  than a pure auth bypass
- the dashboard now shows a warning banner while this mode is active

Why this path:
- it matches the current need: business-owner sharing and testing without
  managing proper roles yet
- it preserves the local authorization surface so later migration back to
  explicit per-user roles is simpler and safer

## 2026-03-27 (Auth Harness And Redirect Hardening Implemented)

Promoted the next auth slice from plan to implementation.

What landed:
- `analytics/auth.py`
  - added a development-only auth test harness controlled by
    `CORKYSOFT_ENABLE_TEST_AUTH` and `CORKYSOFT_TEST_AUTH_MODE`
  - supports deterministic browser-checkable states:
    - anonymous
    - misconfigured
    - login_required
    - unauthorized
    - inactive
    - authenticated
- `dashboard/app.py`
  - resolves the test-auth harness before normal Streamlit OIDC flow
  - surfaces explicit remote-origin / `redirect_uri` mismatch errors via
    `CORKYSOFT_PUBLIC_BASE_URL`
- `tests/test_auth.py`
  - added focused policy and test-harness coverage
- `e2e/auth-harness.spec.ts`
  - added Playwright coverage for:
    - anonymous local-development banner
    - auth-required misconfigured gate
    - unauthorized denial
    - inactive denial
    - low-privilege hidden-tab/query-param denial
- `e2e/support/streamlitHarness.ts`
  - added per-scenario local Streamlit launcher for deterministic browser tests

Validation:
- `venv/bin/python -m pytest tests/test_auth.py tests/test_dashboard_app.py tests/test_dashboard_layouts.py -q`
- `npm run test:e2e:auth`

Docs updated:
- docs/auth_red_team_plan.md
- docs/authentication_and_users.md
- docs/progress_status_board.md
- ROADMAP.md
- COMPACTIFIED_CONTEXT.md

Remaining auth backlog:
- real Google-backed browser login/logout automation
- deployed/tunneled origin verification against real OIDC config
- deeper audit attribution and admin governance follow-on

## 2026-03-27 (Corkysoft MCP v1 Implemented Scaffold)

Implemented the first Corkysoft MCP slice using the ITIR-style adapter pattern.

What landed:
- `corkysoft/mcp/registry.py`
  - namespaced in-process registry
- `corkysoft/mcp/tools.py`
  - four bounded read-only tools:
    - `corkysoft.profitability_summary`
    - `corkysoft.dispatch_recommendations`
    - `corkysoft.operations_diary_summary`
    - `corkysoft.quote_guidance_preview`
- `corkysoft/mcp/bridge.py`
  - local JSON-line bridge with `health`, `info`, `list`, and `call`
- `corkysoft/mcp/server.py`
  - optional FastMCP server when the Python MCP SDK is installed
- `tests/test_corkysoft_mcp.py`
  - registry, quote-guidance, dispatch/diary, and profitability coverage

Policy held:
- read-only first
- stable success/error envelopes
- bridge-default CLI, with FastMCP remaining optional
- producer ownership preserved in analytics and existing workflow helpers
- mutable dispatch/admin/policy tools still deferred

Validation:
- `venv/bin/python -m pytest tests/test_corkysoft_mcp.py tests/test_operations_assignment.py tests/test_operations_diary.py tests/test_quote_service.py tests/test_price_distribution.py -q`
- `116 passed`

Docs updated:
- README.md
- ROADMAP.md
- docs/progress_status_board.md
- docs/architecture.md
- docs/modules.md
- docs/corkysoft_mcp_v1.md
- CHANGELOG.md
- COMPACTIFIED_CONTEXT.md

## 2026-03-27 (Corkysoft MCP v1 Direction)

Checked `ITIR-suite` MCP integration as the reference pattern and locked the
transferable architectural rule for Corkysoft.

Decisions captured:
- copy the architectural posture, not the exact package layout
- Corkysoft MCP should be an adapter layer over existing producer-owned logic
- v1 should be read-only and deterministic
- v1 should use stable result envelopes:
  - success: `{\"ok\": true, \"result\": ...}`
  - failure: `{\"ok\": false, \"error\": {\"code\": ..., \"message\": ..., \"details\": ...}}`
- the first tool family should stay bounded to:
  - `corkysoft.profitability_summary`
  - `corkysoft.dispatch_recommendations`
  - `corkysoft.operations_diary_summary`
  - `corkysoft.quote_guidance_preview`

Boundary locked:
- Corkysoft remains workflow and operational-truth owner
- SB remains the downstream reviewed-state consumer
- ITIR remains orchestration/context and contract hygiene
- MCP must not become a second mutable workflow owner

Deferred from MCP v1:
- mutable dispatch actions
- Kent admin/policy writes
- rollout approval/control actions
- other auth-sensitive governance mutations

Docs updated:
- README.md
- ROADMAP.md
- docs/progress_status_board.md
- docs/architecture.md
- docs/modules.md
- docs/corkysoft_mcp_v1.md
- COMPACTIFIED_CONTEXT.md

Implementation changes:
- None. Documentation/TODO-only alignment.

## 2026-03-27 (Model Governance Direction)

Aligned the next modeling phase around governance-first expansion of the current
corridor-aware profitability model.

Decisions captured:
- The implemented model now includes:
  - baseline distance/season regression
  - corridor-aware adjustments
  - baseline-vs-corridor fit reporting
  - chronological holdout trust signals
- Current product stance:
  - model output is advisory only
  - current trust labels are useful guidance, not a full promotion policy
- Next modeling priority order:
  1. stricter governance thresholds and suppression rules for weak or rare corridors
  2. explicit promotion states for when corridor effects may be shown, guarded, or suppressed
  3. rolling chronological backtesting windows over time
  4. corridor-season and holiday/day-type interaction features
  5. operator-facing uncertainty ranges, with prediction intervals preferred over raw statistical jargon
- Deferred:
  - customer/site class effects stay out of current committed scope until data quality is strong enough

Disambiguations recorded:
- "corridor-season interaction" means route-specific seasonal or holiday/day-type effects, not just one global seasonal coefficient
- "proper backtesting" means repeated train/test windows over time, not one static holdout split
- "uncertainty intervals" should be framed to operators as expected ranges/reliability, with prediction intervals more relevant than confidence intervals for future jobs

Docs updated:
- ROADMAP.md
- docs/progress_status_board.md
- COMPACTIFIED_CONTEXT.md

Implementation changes:
- None. Documentation/TODO-only alignment.

## 2026-03-04

Source of truth update based on current repository state and roadmap review.

Current status summary:
- Core routing + costing are stable.
- Analytics and Streamlit dashboard are partially implemented.
- Major blockers: historical job ingestion, corridor/lane data model, full dashboard wiring.

High-leverage next features:
- Historical import pipeline.
- Corridor/lane detection + rollups.
- $/m³ benchmarking overlays.
- Quote recommendation engine.
- Backhaul detection.
- Profitability scoring.
- Corridor heatmap layer.
- Automated corridor pricing adjustments.

Docs updated:
- README.md and ROADMAP.md updated to reflect current status and priorities.

Implementation changes:
- None. Documentation-only updates.

## 2026-03-04 (Positioning)

Added a positioning and competitive landscape summary to align product focus as a pricing intelligence layer that integrates with incumbent systems rather than replacing them.

Docs updated:
- README.md
- ROADMAP.md
- docs/positioning.md

Implementation changes:
- None. Documentation-only updates.

## 2026-03-04 (Integration Staging Schema)

Added a minimal integration staging schema document to define required fields,
staging tables, and ingest flow for external system data.

Docs updated:
- README.md
- docs/integration_staging_schema.md

Implementation changes:
- None. Documentation-only updates.

## 2026-03-04 (Roadmap Status Recheck)

Reclassified roadmap items based on current code:
- Marked partial completion for historical import, $/m3 metrics, heatmaps, and dashboard wiring.
- Marked modifier tables and base-rate schedule as implemented.
- Updated blockers to focus on data validation, corridor formalization, quote benchmarking, backhaul detection, and end-to-end dashboard wiring.

Docs updated:
- ROADMAP.md

Implementation changes:
- None. Documentation-only updates.

## 2026-03-04 (Corridor Detection)

Added a corridor detection design doc covering baseline clustering, metrics,
and backhaul implications.

Docs updated:
- README.md
- ROADMAP.md
- docs/corridor_detection.md

Implementation changes:
- None. Documentation-only updates.

## 2026-03-04 (Directional + Grouped Corridors)

Updated corridor detection docs to define directional corridors grouped into
bidirectional corridor groups, with time-bucket stats.

Docs updated:
- docs/corridor_detection.md
- ROADMAP.md

Implementation changes:
- None. Documentation-only updates.

## 2026-03-04 (Corridor Schema Plan)

Added a corridor schema plan with proposed tables, metrics, and time buckets.

Docs updated:
- README.md
- ROADMAP.md
- docs/corridor_schema_plan.md

Implementation changes:
- None. Documentation-only updates.

## 2026-03-04 ($/m3 Distribution + Phantom Corridors)

Extended corridor docs to include $/m3 distribution buckets and phantom corridor
signals for opportunity detection.

Docs updated:
- docs/corridor_detection.md
- docs/corridor_schema_plan.md
- ROADMAP.md

Implementation changes:
- None. Documentation-only updates.

## 2026-03-04 (Corridor Decisions + Break-Even)

Added recommended decisions (manual clusters + geohash fallback, threshold=6,
denormalized corridor keys), break-even overlay guidance, phantom corridor
scoring, and gravity model note.

Docs updated:
- docs/corridor_detection.md
- docs/corridor_schema_plan.md
- ROADMAP.md

Implementation changes:
- None. Documentation-only updates.

## 2026-03-04 (Corridor Defaults + AU Clusters)

Added default thresholds, buckets, and break-even constants, plus an AU manual
cluster template.

Docs updated:
- README.md
- ROADMAP.md
- docs/corridor_defaults.md
- docs/cluster_template_au.md

Implementation changes:
- None. Documentation-only updates.

## 2026-03-04 (Gravity + Opportunity Scoring)

Added gravity model formula defaults, opportunity scoring, and geohash corridor
automation notes.

Docs updated:
- docs/corridor_detection.md
- docs/corridor_schema_plan.md
- docs/corridor_defaults.md
- ROADMAP.md

Implementation changes:
- None. Documentation-only updates.

## 2026-03-04 (Corridor Opportunity Report)

Added a report spec for corridor opportunity ranking.

Docs updated:
- README.md
- ROADMAP.md
- docs/corridor_opportunity_report.md

Implementation changes:
- None. Documentation-only updates.

## 2026-03-05 (Corridor Opportunity View)

Added a docs-only SQL view definition for the corridor opportunity report.

Docs updated:
- README.md
- ROADMAP.md
- docs/corridor_opportunity_view.md

Implementation changes:
- None. Documentation-only updates.

## 2026-03-11 (AMS Backend Docs Context Sync)

Resolved archived thread metadata and used it to refresh backend documentation
intent before making doc/TODO updates.

Resolved thread:
- title: Kent Removals AMS Info
- online UUID: 69b137e9-862c-83a0-95ff-90b12cdb7751
- canonical thread ID: 3a4e2a1bf2bd1571afb73c915a4edf99953ad3a7
- source used: db (local canonical archive at `~/chat_archive.sqlite`)

Main topics pulled from the thread:
- Backend documentation structure that reduces onboarding time.
- Typical relocation backend entity model (`assignees`, `moves`, `shipments`, `services`, `vendors`, `inventory_items`, `documents`, `tasks`).
- Reverse-engineering checklist for undocumented systems.
- Importance of documenting lifecycle state transitions and pricing-engine inputs.

Docs updated:
- README.md
- ROADMAP.md
- docs/ams_backend_docs_playbook.md

Implementation changes:
- None. Documentation/TODO alignment only.

## 2026-03-11 (Kent AMS Interface Spec)

Added an explicit Corkysoft-to-Kent AMS integration document so the interface
boundary and payload expectations are not implied.

Docs updated:
- README.md
- ROADMAP.md
- docs/kent_ams_integration.md

Implementation changes:
- None. Documentation-only update.

## 2026-03-20 (Adaptive Learning Loop Context Sync)

Resolved archived thread metadata and used it to sharpen repo-facing intent
before making docs/TODO/code updates.

## 2026-03-25 (Observer Export Follow-up Documentation)

Current status summary:
- Corkysoft now emits a local observer outbox for diary/reconciliation review
  families and supports explicit export for `planning_snapshot` and
  `reconciliation_exception`.
- JMD remains out of scope.
- Remaining observer-export follow-up work is operational, not modeling:
  delivery receipt / watermark semantics and dashboard visibility of emitted
  envelopes.

Docs updated:
- ROADMAP.md
- docs/progress_status_board.md
- CHANGELOG.md

Implementation changes:
- None. Documentation/status alignment only.

Resolved thread:
- title: Weather Handling Docs
- online UUID: 69b7de6e-56dc-839e-b322-80af1804d40e
- canonical thread ID: 269d31b5e8d7045653828e225297af0d1f235c59
- source used: db (after live ingest into `~/chat_archive.sqlite`)

Main topics pulled from the thread:
- Corkysoft should ingest situational-awareness signals such as road closures,
  weather events, and market/route disruption metadata where practical.
- Those signals should feed a bounded operational learning loop rather than
  trigger ad hoc pricing rewrites.
- Learned policy state should be a small, explicit parameter set updated
  slowly from realised jobs, with guardrails and operator review.

Docs updated:
- README.md
- ROADMAP.md

## 2026-03-25 (Historical Ingest + Lane Governance)

Collapsed the remaining trust gap around corridor/lane formalization into three implemented slices:
- historical ingest coverage hardening
- canonical lane assignment
- lane promotion governance plus planner-safe consumption

Implemented state:
- historical ingest now records run-level coverage, row-level issues, and readiness status
- Fleet admin now shows ingest-health summaries plus lane-assignment health and gap visibility
- canonical lane entities now exist: `location_clusters`, `directional_lanes`, `corridor_groups`
- historical and live jobs now persist lane assignment metadata:
  - `origin_cluster_key`
  - `destination_cluster_key`
  - `lane_key`
  - `corridor_group_key`
  - `lane_assignment_status`
  - `lane_assignment_source`
  - `lane_assignment_note`
- lane governance now includes proposal, approval/rejection, and apply flows before promoting new directional lanes
- grouped lane proposals now exist for repeated candidate cluster pairs
- Planner plus analytics tabs now default to `assigned` lane history and require explicit operator opt-in to include `ambiguous` or `unassigned` rows
- Kent admin now exposes a compact review summary for hard-block, policy-fail, loss-alert, and overrideable tender counts

Docs updated:
- README.md
- ROADMAP.md
- docs/progress_status_board.md
- CHANGELOG.md

Implementation changes:
- analytics/price_distribution.py
- analytics/lane_assignment.py
- analytics/planner.py
- analytics/db/legacy.py
- dashboard/components/maintenance.py
- dashboard/components/planner.py
- tests/test_lane_assignment.py
- tests/test_planner.py
- docs/adaptive_learning_loop.md
- docs/modules.md
- docs/architecture.md

Implementation changes:
- Added minimal adaptive-policy helpers on top of the existing
  `global_parameters` table plus focused tests.

## 2026-03-20 (Operations Diary + Reconciliation)

Added a manager-facing workflow definition and implementation slice for day/week
operations review, job usage drill-down, and invoice/bill follow-through.

Main topics pulled from the client story:
- move from a diary/day view into invoicing, vehicle usage, staff usage, and
  job-level requirement/utilization review
- support planning across the day/week rather than treating Planner as only a
  route-shaping surface
- support late subcontractor and third-party bill review against completed jobs
  and known operational truth

Docs updated:
- README.md
- ROADMAP.md
- docs/operator_user_stories.md
- docs/planner_interaction_model.md
- docs/commercial_workflow_lifecycle.md
- docs/operations_diary_workflow.md
- docs/job_cost_and_invoice_reconciliation.md
- docs/modules.md
- docs/architecture.md

Implementation changes:
- Added diary tasks plus customer-invoice and subcontractor-bill review tables.
- Added operations-diary service helpers and a new `Operations diary` dashboard
  tab.
- Linked Planner and Dispatch into the diary and added focused service tests.

## 2026-03-21 (Strategy / Positioning Context Sync)

Resolved four archived threads and used them to tighten the repo-facing product
framing without changing implementation scope.

Resolved threads:
- title: MoveWare vs Corkysoft Gaps
- online UUID: 69bcd9fd-7fe4-83a0-8e8b-608d9ad2d54f
- canonical thread ID: 05ed68f07441b7873a750ea1215b9554a4370ae8
- source used: db
- main topics: international/shipping paperwork gap; MoveWare's relative edge is
  encoded requirements/proposal/governance (`R -> P -> G`) rather than better
  operational compute/state/structure

- title: Corkysoft ITIR Merge
- online UUID: 69bcda66-0564-83a1-b2fe-965ea7ab8700
- canonical thread ID: ddc7f5f5a101b41e04b9b1f71549abcf091dad0b
- source used: db
- main topics: Corkysoft is converging toward a provable logistics-state
  compiler; pricing, routing, telemetry, evidence, and auditability belong in
  one decision stack

- title: Corkysoft Parity Strategy
- online UUID: 69bcda38-1a74-839f-9415-f94de6f0169a
- canonical thread ID: 956747b9f3d9214a3877b9c3f98f2385aeb91553
- source used: db
- main topics: Corkysoft should be framed as a system of decision rather than
  only a system of record; parity matters, but superiority comes from better
  decisions, not CRUD duplication

- title: Weather Handling Docs
- online UUID: 69b7de6e-56dc-839e-b322-80af1804d40e
- canonical thread ID: 269d31b5e8d7045653828e225297af0d1f235c59
- source used: db
- main topics: weather/disruption inputs should remain bounded,
  situational-awareness signals that feed explicit policy state and decision
  proofs rather than opaque autonomous behavior

Main decisions pulled into repo intent:
- Corkysoft should be described as a system of decision over removals work,
  not just a route-profitability dashboard or generic operational record store.
- The current implementation remains deliberately staged: decision support,
  planning, operations diary, usage review, and reconciliation come first.
- A major documented gap is international/compliance-heavy work, where
  requirements capture, proposal assembly, and governance evidence are not yet
  formalized enough.
- That gap should be treated as future `R -> P -> G` workflow work tied to
  paperwork, insurance, tender, customs, and audit-heavy jobs rather than as a
  vague "feature parity" request.

Docs updated:
- README.md
- ROADMAP.md
- docs/positioning.md
- docs/operator_user_stories.md
- docs/commercial_workflow_lifecycle.md

Implementation changes:
- None. Documentation/TODO/changelog alignment only.

## 2026-03-21 (Corkysoft / SB / ITIR Boundary Audit)

Locked the cross-project boundary more explicitly after checking the current
diary/reconciliation implementation and the existing ITIR/StatiBaker docs.

Main decisions:
- Corkysoft is where removals workflow state changes happen: planner, diary,
  tasks, assignments, invoice review, and subcontractor-bill review.
- StatiBaker is downstream-only for this domain: interpretible logs, compiled
  summaries, provenance, and review lenses across many sources.
- ITIR remains the orchestration/context and contract layer across projects.
- Corkysoft workflow learnings may later inform SB/ITIR lens design, but SB
  must not become a second operational cockpit for removals execution.

Docs updated:
- README.md
- ROADMAP.md
- docs/architecture.md
- docs/modules.md
- docs/corkysoft_sb_itir_coverage_audit.md
- docs/sb_itir_downstream_contract.md
- docs/planner_diary_patterns_for_sb_itir.md

Implementation changes:
- None. This pass only added audit/contract/pattern documentation.

## 2026-03-21 (Holiday Bill Aging / Exposure Model)

Clarified the received-but-unprocessed Christmas / New Year bill story into a
concrete reconciliation-aging model.

Main decisions:
- unresolved supplier liability should age from the bill-received / bill-action
  date, not only from job execution
- job execution date must still remain visible so delayed billing latency and
  hidden-margin distortion are explicit
- Corkysoft should get a thin manager-facing unresolved-exposure summary in the
  diary
- the heavier long-horizon timeline/lens treatment remains a better downstream
  SB fit later

Docs updated:
- README.md
- ROADMAP.md
- docs/job_cost_and_invoice_reconciliation.md
- docs/operations_diary_workflow.md
- docs/operator_user_stories.md
- docs/corkysoft_sb_itir_coverage_audit.md

## 2026-03-11 (Kent AMS Mapping + Roadmap)

Expanded Kent integration documentation with an explicit field mapping table
from expected Kent AMS entities to current Corkysoft schema fields and added a
phased integration roadmap with milestones and acceptance gates.

Docs updated:
- README.md
- ROADMAP.md
- docs/kent_ams_integration.md
- docs/kent_ams_integration_roadmap.md

Implementation changes:
- None. Documentation-only update.

## 2026-03-12 (Kent AMS Importer Scaffold)

Brought Kent AMS integration interface closer to MoveWare by adding an internal
Kent importer endpoint and resource dispatcher for adapter-fed payloads.

Code updated:
- corkysoft/api.py
- analytics/kent_ams_import.py
- tests/test_api.py

Docs updated:
- docs/kent_ams_integration.md

Implementation changes:
- Added `POST /importers/kent-ams/{resource}` using the same request/summary
  shape as MoveWare importer.
- Added Kent resources: `jobs`, `subcontractors`/`vendors`, `bids`, `awards`.
- Added Kent bid/award persistence tables created on demand by importer.

## 2026-03-12 (Kent Tender Pre-Scoring)

Implemented tender pre-scoring so incoming Kent AMS tenders are ranked for
operator focus based on profitability, urgency, seasonality, and capacity fit.

Code updated:
- analytics/kent_ams_import.py
- corkysoft/api.py
- tests/test_api.py

Docs updated:
- docs/kent_ams_integration.md
- ROADMAP.md

Implementation changes:
- Added Kent `tenders` import resource on `POST /importers/kent-ams/{resource}`.
- Added tender storage table `kent_job_tenders` with persisted score components.
- Added read endpoint `GET /kent-ams/tenders/prioritized` for ranked tender queues.
- Added score outputs and recommended action labels (`pursue_now`, `review_today`,
  `review_if_capacity`, `defer`).

## 2026-03-12 (Kent Tender Calibration Metrics)

Implemented calibration metrics to quantify whether tender scores predict win
rate and realized margin outcomes.

Code updated:
- analytics/kent_ams_import.py
- corkysoft/api.py
- tests/test_api.py

Docs updated:
- docs/kent_ams_integration.md
- ROADMAP.md

Implementation changes:
- Added `GET /kent-ams/tenders/calibration?lookback_days=<n>` endpoint.
- Added score-band metrics: tender count, wins, win rate, predicted vs realized
  margin, and mean absolute margin error.
- Added documented recommendations for weight tuning with peak-season guardrails.

## 2026-03-12 (Route/Location Scoring Added)

Expanded tender scoring and docs to include route/location fit, covering lane
familiarity and historical lane margin effects.

Code updated:
- analytics/kent_ams_import.py
- corkysoft/api.py
- tests/test_api.py

Docs updated:
- docs/kent_ams_integration.md
- ROADMAP.md

## 2026-03-12 (En-Route Spare Capacity Signal Integration)

Added a shared operational signal module and integrated it into:
- Kent tender scoring
- quote creation summaries/persistence
- MoveWare and Kent job ingest paths

Code updated:
- analytics/operational_signals.py
- analytics/kent_ams_import.py
- analytics/moveware_import.py
- corkysoft/quote_service.py
- corkysoft/repo.py
- corkysoft/api.py
- tests/test_api.py

Docs updated:
- docs/kent_ams_integration.md

## 2026-03-12 (Multi-Truck Optimization Spec)

Documented the multi-truck pickup/drop sequencing and transfer optimization
problem explicitly, including the "many jobs per truck / many trucks per job"
model and phased optimization approach.

Docs updated:
- README.md
- ROADMAP.md
- docs/multi_truck_route_load_optimization.md

Implementation changes:
- None. Documentation-only update.

## 2026-03-12 (Kent AMS Policy Workflow)

Implemented the operator-facing Kent tender policy workflow and aligned code,
API, dashboard, and docs around the same behavior.

Code updated:
- analytics/db/parameters.py
- analytics/kent_ams_import.py
- corkysoft/api.py
- dashboard/app.py
- tests/test_api.py

Docs updated:
- README.md
- ROADMAP.md
- docs/kent_ams_integration.md

Implementation changes:
- Added profitability rule-mode defaults in `global_parameters`.
- Changed tender queue ordering to prioritize policy matches while keeping
  non-matching tenders visible with fail reasons and loss alerts.
- Added seeded/admin-managed override reason codes and tender override audit
  logging.
- Added API endpoints for Kent policy config, reason-code management, override
  creation, and override history.
- Added a Streamlit `Kent tenders` tab for queue review, config updates, reason
  management, and override capture/history.

## 2026-03-12 (Kent Validation + Shared Policy Signals)

Completed three follow-up actions after the policy workflow landed:

- ran the Kent API suite successfully under the existing project `venv`
- added fixture-backed Kent tender payload validation for local smoke coverage
- pushed profitability policy semantics into quote creation and job ingest
  operational signals so pricing/dispatch share the same pass/fail/loss model

## 2026-03-12 (Docs/Governance/Kent Reliability Remediation)

Executed the remediation milestone from the repo audit to align project truth,
complete missing product/governance docs, and harden brittle Kent behavior.

Planning state added:
- spec.md
- plan.md
- status.json
- devlog.md

Docs aligned:
- README.md
- ROADMAP.md
- docs/positioning.md
- docs/kent_ams_integration.md
- docs/kent_ams_integration_roadmap.md
- docs/live_network_overview.md
- docs/multi_truck_route_load_optimization.md
- docs/ingest_inventory_logistics.md
- docs/fleet_tables.md
- docs/heatmap_logic.md
- docs/operator_user_stories.md
- docs/commercial_workflow_lifecycle.md

Implementation changes:
- added internal mutating API auth via `X-Corkysoft-Api-Key` validated against
  `CORKYSOFT_API_TOKEN`
- made API `dry_run` execution side-effect free by running imports against an
  in-memory shadow database
- fixed Kent prioritized queue correctness so top-N is selected after final
  ranking rather than before it
- restricted hard-block handling to governed safety/legal/compliance categories
- split Streamlit Kent operator workflow from Kent admin/config workflow
- added UX handling for the "no active override reasons" case
- relabeled quote workflow output as a quote-policy preview rather than silent
  equivalence with tender policy

Validation:
- `py_compile` passed on touched Python files
- targeted tests passed under local `venv`:
  `tests/test_dashboard_app.py`, `tests/test_api.py`,
  `tests/test_kent_ams_fixtures.py`

## 2026-03-12 (Crusader Workbook Fleet Import Hardening)

Closed a local-data reliability gap around the `Crusader.xlsx` placeholder
workbook so it can actually seed fleet state used by pricing and Kent triage.

Implementation changes:
- fixed Fleet tab uploaded-XLSX handling to use the workbook-aware importer
  instead of blindly reading only the first worksheet
- fixed `analytics.vehicle_workbook` so sheet-per-vehicle workbooks without an
  explicit `REGO` column inherit the sheet name as the truck identifier using a
  canonical column path
- added regression coverage against the real repo workbook

Code updated:
- analytics/vehicle_workbook.py
- dashboard/components/maintenance.py
- tests/test_vehicle_workbook.py

Validation:
- targeted local `venv` tests passed:
  `tests/test_vehicle_workbook.py`, `tests/test_dashboard_app.py`,
  `tests/test_api.py`, `tests/test_kent_ams_fixtures.py`

## 2026-03-12 (Google Sheets-First Operations Workbook Migration Start)

Started moving operational data refresh away from the local `Crusader.xlsx`
placeholder and toward the live Google Sheets setup already used by the
business. The private sheet URLs were used for inspection only and were not
recorded in repo docs.

Observed live workbook shape:
- one shared operations workbook contains fleet/staff/supplier tabs
- a separate workbook contains sheet-per-vehicle maintenance history plus
  repairs history/index tabs

Docs updated:
- README.md
- ROADMAP.md

Implementation changes:
- added shared Google Sheets ID/URL resolution helpers
- added Google Sheets staff import using the shared operations workbook
- added supplier import UI for Google Sheets-backed `SUPPLIERS`
- updated fleet import UI defaults to prefer the shared operations workbook env
- kept local `.xlsx` upload paths as fallback rather than the primary workflow

Code updated:
- analytics/google_sheets.py
- analytics/db/fleet.py
- analytics/db/inventory.py
- analytics/driver_shifts.py
- analytics/vehicle_workbook.py
- dashboard/app.py
- dashboard/components/maintenance.py
- tests/test_google_sheets_imports.py

Validation:
- targeted local `venv` tests passed:
  `tests/test_google_sheets_imports.py`, `tests/test_vehicle_workbook.py`,
  `tests/test_dashboard_app.py`, `tests/test_api.py`,
  `tests/test_kent_ams_fixtures.py`

## 2026-03-12 (Shared Operations Workbook Sync)

Implemented a coordinated sync path for the shared operations workbook so fleet,
staff, and suppliers can be refreshed together from one workbook reference.

Docs updated:
- README.md
- ROADMAP.md

Implementation changes:
- added `analytics.operations_workbook.sync_operations_workbook`
- added a Fleet-tab dashboard action to sync the shared operations workbook in
  one step
- added optional env-based per-tab overrides for shared-workbook `STAFF` and
  `SUPPLIERS` sheet names while keeping defaults simple

Code updated:
- analytics/operations_workbook.py
- dashboard/components/maintenance.py
- tests/test_operations_workbook.py

Validation:
- targeted local `venv` tests passed:
  `tests/test_operations_workbook.py`, `tests/test_google_sheets_imports.py`,
  `tests/test_vehicle_workbook.py`, `tests/test_dashboard_app.py`,
  `tests/test_api.py`, `tests/test_kent_ams_fixtures.py`

## 2026-03-12 (Spreadsheet-First Operations Planning)

Implemented the next integration layer for spreadsheet cooperation so
Corkysoft can plan trucks/workers/jobs internally while continuing to ingest
Google Sheets as operational inputs.

Decisions locked:
- spreadsheets are import-only for now; no write-back
- Corkysoft internal state is the planning truth
- `job_segments` are the canonical assignment unit
- maintenance/rego/COI/compliance readiness is evaluated as part of assignment,
  not as a detached dashboard-only concern

Docs updated:
- README.md
- ROADMAP.md
- docs/fleet_tables.md
- docs/commercial_workflow_lifecycle.md

Implementation changes:
- added source provenance fields for workers, suppliers, and vehicle details
- added `analytics.operations_assignment` for policy, readiness, assignment,
  conflicts, and segment bootstrap/update behavior
- added `/operations/*` API endpoints for policy, sync, segment creation,
  readiness listing, assignment, and conflicts
- added dashboard `Operations` tab for segment planning and assignment
- added readiness policy controls in Fleet/admin workflow
- updated shipments wrapper to use the richer legacy segment-aware shipment path
- fixed `ensure_segment` so backfilled default segments can actually be updated
  with planned windows, which restores overlap/conflict detection
- normalized vehicle workbook provenance timestamps to UTC

Tests added/updated:
- tests/test_operations_assignment.py
- tests/test_api.py
- tests/test_dashboard_app.py
- tests/test_google_sheets_imports.py
- tests/test_vehicle_workbook.py

Validation:
- targeted local `venv` tests passed:
  `tests/test_operations_assignment.py`, `tests/test_google_sheets_imports.py`,
  `tests/test_vehicle_workbook.py`, `tests/test_dashboard_app.py`,
  `tests/test_api.py`, `tests/test_kent_ams_fixtures.py`,
  `tests/test_operations_workbook.py`

Follow-up implementation completed the next workflow step:
- Staff tab now shows planned segment assignments, planned trucks/jobs, and next
  planned work separately from imported sheet truck context and recent shifts
- Fleet tab now shows planned segment/job/worker context for each truck and a
  per-truck planned segment detail view
- added `docs/spreadsheet_replacement_plan.md` to define full spreadsheet
  replacement by workflow rather than by raw table

Validation:
- local `venv` tests passed after Staff/Fleet integration updates:
  `tests/test_operations_assignment.py`, `tests/test_dashboard_app.py`,
  `tests/test_api.py`, `tests/test_google_sheets_imports.py`,
  `tests/test_vehicle_workbook.py`, `tests/test_operations_workbook.py`

Phase 2 maintenance/compliance work implemented:
- Fleet now exposes a maintenance/compliance cockpit showing due-soon and blocked
  rego, COI, service, and worker compliance items
- Staff now supports native role assignment and worker compliance assignment with
  expiry dates
- added `/operations/readiness/resources` plus internal worker role/compliance
  mutation endpoints

Validation:
- local `venv` tests passed after cockpit/API changes:
  `tests/test_operations_assignment.py`, `tests/test_api.py`,
  `tests/test_dashboard_app.py`, `tests/test_google_sheets_imports.py`,
  `tests/test_vehicle_workbook.py`, `tests/test_operations_workbook.py`

Phase 3 native labor planning work implemented:
- native labor roster is now derived from `job_segments` assignments
- Driver shifts tab is reframed as labor planning + reconciliation rather than a
  raw spreadsheet-first planning surface
- added labor roster and plan-vs-imported reconciliation API endpoints

Validation:
- local `venv` tests passed after labor-planning changes:
  `tests/test_operations_assignment.py`, `tests/test_api.py`,
  `tests/test_dashboard_app.py`, `tests/test_google_sheets_imports.py`,
  `tests/test_vehicle_workbook.py`, `tests/test_operations_workbook.py`

Phase 4 inventory/supplier coordination work implemented:
- Inventory tab now shows segment-linked stock and supplier coordination
- stock can be allocated directly to planned `job_segments`
- added API routes for segment-linked inventory coordination and allocation

Validation:
- local `venv` tests passed after inventory coordination changes:
  `tests/test_inventory_and_shipments.py`, `tests/test_operations_assignment.py`,
  `tests/test_api.py`, `tests/test_dashboard_app.py`,
  `tests/test_google_sheets_imports.py`, `tests/test_vehicle_workbook.py`,
  `tests/test_operations_workbook.py`

Phase 5 dispatch-board and spreadsheet decommissioning work started:
- added a native `Dispatch` tab as the primary job-centric execution surface
- Dispatch aggregates segment readiness, truck assignments, worker assignments,
  stock allocations, and supplier context into one board
- added `/operations/jobs/board` API for the same unified job view
- added dispatch snapshot CSV export so external stakeholders can receive a
  lightweight operational extract without keeping spreadsheets as the live
  operating surface

Validation:
- local `venv` tests passed after dispatch-board changes:
  `tests/test_inventory_and_shipments.py`, `tests/test_operations_assignment.py`,
  `tests/test_api.py`, `tests/test_dashboard_app.py`,
  `tests/test_google_sheets_imports.py`, `tests/test_vehicle_workbook.py`,
  `tests/test_operations_workbook.py`

Phase 5 cutover-governance work implemented:
- added persistent workflow-level cutover tracking for spreadsheet replacement
- each workflow now carries cutover status, fallback mode, snapshot rules,
  checklist completion, last drill timestamp, and rollback instructions
- Dispatch shows a read-only cutover summary for operators
- Fleet admin exposes editable cutover controls for admins
- added `/operations/cutover/workflows` GET/PUT API endpoints

Validation:
- local `venv` tests passed after cutover-governance changes:
  `tests/test_operations_assignment.py`, `tests/test_api.py`,
  `tests/test_dashboard_app.py`, `tests/test_inventory_and_shipments.py`,
  `tests/test_google_sheets_imports.py`, `tests/test_vehicle_workbook.py`,
  `tests/test_operations_workbook.py`

Phase 5 cutover-metrics work implemented:
- each workflow now tracks native usage %, target %, fallback-use count, open
  issues, snapshot-consumer count, and last review timestamp
- Dispatch exposes whether cutover targets are met for each workflow
- Fleet admin can maintain the metrics used to decide when a workflow can move
  from dual-run to fallback-only

Validation:
- a cutover-table seed insert mismatch was caught by the local `venv` tests,
  fixed, and the same full operations slice passed afterward:
  `tests/test_operations_assignment.py`, `tests/test_api.py`,
  `tests/test_dashboard_app.py`, `tests/test_inventory_and_shipments.py`,
  `tests/test_google_sheets_imports.py`, `tests/test_vehicle_workbook.py`,
  `tests/test_operations_workbook.py`

Phase 5 event-driven rollout tracking implemented:
- cutover metrics now derive from live operations state where practical
- review, fallback-drill, fallback-use, and snapshot-issued events are logged
  and exposed via API/UI
- Dispatch logs snapshot issuance from the operator surface
- Fleet admin exposes cutover actions and recent event history

MCP validation against the live Streamlit app:
- seeded a local SQLite scenario with one planned job segment, truck, worker,
  inventory allocation, cutover events, and one Kent tender
- confirmed Quote builder, Dispatch, Fleet, Kent tenders, and Kent admin load
  as separate role surfaces
- verified Dispatch snapshot export logs an event
- verified Fleet cutover admin logs a review event
- discovered and fixed an Inventory tab crash (`KeyError: 'name'`) in the
  reserve/release selector before completing the walkthrough
- cleaned up the remaining `use_container_width` deprecation warnings in the
  active Streamlit surfaces by migrating to `width='stretch'`
- added guarded rollout recommendations and an apply-transition path so admins
  can promote workflows only when current checklist gates and derived metrics
  justify the move
- added `docs/rollout_execution_user_stories.md` to define the remaining live
  rollout stories separately from the core product stories
- rollout promotion now requires an approval chain on top of the existing
  evidence gate:
  - ops manager requests promotion
  - commercial owner approves or rejects it
  - admin applies the transition only after approval is present
- implementation reuses `operations_cutover_events`; no second audit table was
  added
- Fleet cutover admin now exposes request/approve/reject/apply controls and
  shows approval state beside the recommendation
- Dispatch remains read-only for rollout state, but now shows approval status
  in its cutover summary
- targeted local `venv` validation passed for the full operations/dashboard
  slice after the approval-chain changes:
  `tests/test_operations_assignment.py`, `tests/test_api.py`,
  `tests/test_dashboard_app.py`, `tests/test_inventory_and_shipments.py`,
  `tests/test_google_sheets_imports.py`, `tests/test_vehicle_workbook.py`,
  `tests/test_operations_workbook.py`
- MCP walkthrough against seeded `/tmp/corkysoft-mcp-rollout.db` and local
  Streamlit on `http://localhost:8501` confirmed:
  - Quote builder loaded for estimator flow
  - Dispatch exported a snapshot and logged `snapshot_issued`
  - Fleet cutover admin showed blocked recommendation before request
  - Fleet cutover admin logged `promotion_requested`
  - Fleet cutover admin logged `promotion_approved`
  - Fleet cutover admin applied the guarded transition and logged
    `status_transition`
  - `dispatch_execution` persisted as `native_primary`
  - Kent tenders and Kent admin remained separate operator/admin surfaces
- Added two parallel documentation tracks for user-facing quality:
  - `docs/usage_onboarding_guide.md` for formal onboarding/help directions
  - `docs/naive_user_tester_notes.md` for out-loud first-time-user testing notes
- Reran the user stories through MCP against fresh seeded data in
  `/tmp/corkysoft-mcp-stories.db` and local Streamlit on `http://localhost:8502`
- Seeded scenario included:
  - three jobs (planned, warning-state, and blocked/readiness-affected)
  - two open Kent tenders
  - one blocked vehicle
  - one eligible rollout promotion path
- Confirmed through the live UI:
  - Quote builder still loads as the estimator surface
  - Dispatch exports snapshot CSV and logs `snapshot_issued`
  - Operations shows segment-based planning clearly
  - Fleet shows blocked maintenance/readiness state and supports governed
    request -> approve -> apply rollout flow
  - Kent tenders and Kent admin remain separated by operator/admin purpose
- Persisted SQLite verification after the walkthrough confirmed:
  - `dispatch_execution` moved to `native_primary`
  - `snapshot_issued`, `promotion_requested`, `promotion_approved`, and
    `status_transition` were logged in order
- Main UX debt recorded from this pass:
  - the global `historical_jobs` warning appears on operational tabs where it
    is irrelevant and noisy
  - analytics-first landing remains suboptimal for many day-to-day operators
- This pass required no code changes; docs and roadmap were updated instead
- Fleet shared-workbook sync bug reproduced from the UI:
  - error: `Failed to sync operations workbook: Error binding parameter 5: type 'NaTType' is not supported`
- Root cause:
  - pandas `NaT` values from the vehicle workbook were being passed through as
    optional text fields to SQLite instead of being normalized to `None`
- Fix implemented:
  - vehicle workbook importer now sanitizes optional text fields before upsert
  - vehicle workbook date parsing now respects ISO first and day-first sheet
    dates second
  - explicit SQLite ignore patterns were added to `.gitignore`
- Validation:
  - local `venv` tests passed for `tests/test_vehicle_workbook.py`,
    `tests/test_operations_workbook.py`, and
    `tests/test_google_sheets_imports.py`
- Imports performed into local `routes.db` without recording the private URLs in
  repo docs:
  - shared operations workbook refreshed fleet/staff/suppliers
  - second provided workbook was identified as a multi-tab vehicle workbook and
    imported via its index tab plus per-vehicle sheets
- Post-import `routes.db` counts:
  - trucks: 29
  - vehicle_details: 29
  - workers: 7
  - suppliers: 14

Validation:
- after the MCP-found inventory fix, the full operations slice passed again:
  `tests/test_operations_assignment.py`, `tests/test_api.py`,
  `tests/test_dashboard_app.py`, `tests/test_inventory_and_shipments.py`,
  `tests/test_google_sheets_imports.py`, `tests/test_vehicle_workbook.py`,
  `tests/test_operations_workbook.py`


## 2026-03-13 (Role Coverage Docs Review)
- Reviewed the current imported operational footprint in local `routes.db`:
  - trucks: 29
  - vehicle_details: 29
  - workers: 7
  - suppliers: 14
- Confirmed the current UI already covers the required operational functions for this dataset:
  - `Dispatch` covers jobs, segments, trucks, workers, inventory, suppliers, and rollout visibility
  - `Operations` covers segment creation, assignment, conflicts, and override-aware planning
  - `Fleet` covers workbook sync, readiness policy, cutover admin, maintenance/compliance, and vehicle imports
  - `Staff` covers roster editing, planned assignments, linked shifts, roles, compliances, and worker readiness
  - `Driver shifts` covers native labor roster plus imported-shift reconciliation
  - `Inventory` covers segment-linked stock coordination, supplier import, movements, allocation, and exception reconciliation
- Main gap identified:
  - docs and onboarding under-described the full role stack and left several surfaces without explicit primary-role ownership
- Decision:
  - perform a docs-only alignment pass before any UI copy or surface changes
  - expand actor stories and onboarding to include labor planning, maintenance/compliance, inventory/suppliers, and system/rollout admin as first-class roles
  - add a dedicated `docs/ui_role_coverage_matrix.md` as the canonical role-to-surface map
- This pass changes documentation and roadmap/context only; no code or API changes are intended.

## 2026-03-13 (Planner UX Documentation Gap)
- Cross-check against current docs confirmed that Corkysoft already documents:
  - `job_segments` as the planning truth for execution, readiness, and stock coordination
  - workflow-first spreadsheet replacement with segment-based cooperation between trucks, workers, and inventory
  - corridor/lane profitability as a core commercial and operational signal
  - multi-leg, sequencing, transfer, and explainability as valid planning directions
- Main documentation gap identified:
  - current docs describe the data model and optimization goals more clearly than the intended planner interaction model
  - they do not yet state clearly that the target planner should let users select roadway/corridor/site extents visually, surface historical route overlap, expose per-route/per-segment profitability, and shape draft operational legs before assignment
- Decision for this round:
  - update docs and roadmap only
  - keep `job_segments` as the internal planning artifact
  - treat manual segment editing as an advanced/admin fallback rather than the desired operator workflow
  - return to implementation planning after the documentation reflects this interaction model

- new canonical doc added: `docs/planner_interaction_model.md` to define the operator interaction model that sits above `job_segments`, corridor analytics, and multi-leg optimization constraints

- implementation direction clarified after planner-doc work:
  - no auth/profile system work for role-aware layouts
  - role layout defaults should be lightweight config with manual role switching
  - first planner milestone should ship as a dedicated `Planner` tab
  - existing `Operations` segment form should remain as advanced/manual fallback rather than primary planner UX
- implementation follow-up:
  - confirmed that the role-layout defaults module, planner helper module, planner component, Fleet role-layout admin, and Planner tab definition were already on disk
  - found the concrete missing behavior: `Planner` existed in the tab list but was not rendered in `dashboard/app.py`
  - fixed the live dashboard flow to render `Planner`
  - fixed the role-layout integration bug where hidden tabs could still raise `KeyError` because the app rendered bodies for tabs that were no longer visible
  - updated top-level docs/roadmap language from planned to implemented for the current scaffold, and added a regression test that asserts the dashboard render path includes `render_planner_tab(...)`
- planner milestone 2 implemented:
  - `Planner` now supports dual entrypoints: `job_first` and `corridor_first`
  - planner proposals now carry routing context, resource-fit context, grouped explainability, and warnings
  - current milestone keeps site/location planning deferred, uses traffic/routing as summary context rather than live traffic optimization, and still confirms into internal `job_segments`
- inventory direction clarified after planner milestone 2:
  - current inventory is good enough as a stock ledger and segment-linked coordination surface, but not yet a strong operational inventory management system
  - next milestone should focus on requirement planning, shortage detection, and custody/location truth rather than generic inventory CRUD
  - Crusader's container-heavy operating model should be treated as first-class while still supporting consumables, reusable assets, serialized/tagged gear, and other architectures

## 2026-03-13 (Inventory Planning, Shortage, and Custody Implementation)
- Implemented the first planning-aware inventory layer on top of the existing stock/movement system.
- Schema/model changes:
  - `inventory_items` now carries `architecture`, `custody_location_type`, `custody_location_ref`, `custody_location_label`, and `custody_updated_at`
  - `inventory_movements` now carries `location_type`, `location_ref`, and `location_label`
  - new `inventory_requirements` table stores per-job / per-segment requirement lines
- Inventory semantics now support multiple architectures:
  - container
  - consumable
  - reusable_asset
  - serialized_asset
  - job_specific
  - general
- Custody/location truth now supports:
  - depot
  - truck
  - container
  - in_transit
  - site
  - returned_storage
  - exception
- Requirement and shortage behavior implemented:
  - required vs allocated vs shortage quantities are computed per segment requirement
  - non-substitutable shortages block readiness
  - substitutable shortages create explicit override-required flags
- Readiness / operations integration:
  - `evaluate_segment_readiness(...)` now includes inventory shortages
  - job/segment operations board now rolls up required, allocated, and shortage totals
  - Dispatch now shows shortage state in the job board and segment detail
  - Planner now shows inventory-fit context and shortage lines for attached jobs
  - Inventory tab now supports:
    - requirement planning
    - custody/location updates
    - richer segment coordination and shortage visibility
- Validation:
  - `py_compile` passed for the changed inventory, operations, and dashboard modules
  - local `venv` tests passed for:
    - `tests/test_inventory_and_shipments.py`
    - `tests/test_operations_assignment.py`
    - `tests/test_planner.py`
    - `tests/test_dashboard_app.py`
    - `tests/test_api.py`
    - `tests/test_operations_workbook.py`
    - `tests/test_google_sheets_imports.py`
    - `tests/test_vehicle_workbook.py`

## 2026-03-13 (Inventory Execution Workflow Spec)
- Added new canonical doc:
  - `docs/inventory_execution_workflow.md`
- Purpose:
  - close the gap between inventory planning semantics and real warehouse/day-of-execution workflow
- Locked workflow decisions:
  - planning remains requirement-line / `m3` based
  - custody/execution is container-first where applicable
  - warehouse/crew records normal pick / pack / load transitions
  - dispatcher / operations manager approves substitutions
  - manager review is reserved for escalations and repeated drift, not routine execution
- Adjacent docs were aligned to point back to the new canonical workflow rather than restating warehouse policy ad hoc:
  - `README.md`
  - `ROADMAP.md`
  - `docs/ingest_inventory_logistics.md`
  - `docs/spreadsheet_replacement_plan.md`
  - `docs/operator_user_stories.md`

## 2026-03-13 (Inventory Role Model Alignment)
- Reviewed the new inventory execution workflow against the existing role docs and found a real actor-model mismatch:
  - `Warehouse / Crew` existed in the canonical workflow spec but not in user stories, onboarding, or the role matrix
- Documentation correction applied:
  - `Warehouse / Crew` is now a first-class actor in:
    - `docs/operator_user_stories.md`
    - `docs/usage_onboarding_guide.md`
    - `docs/ui_role_coverage_matrix.md`
- Ownership is now explicit:
  - warehouse / crew owns routine pick / pack / load and custody progression
  - dispatcher / operations owns substitution approval
  - manager handles escalation and repeated drift rather than routine approvals
- State-model ambiguity resolved in docs:
  - persisted inventory states remain the lower-level movement/state model
  - operator-facing workflow stages in `docs/inventory_execution_workflow.md` are documented as a layer above that persisted model, not a silent replacement for it

## 2026-03-13 (Inventory Execution Workflow Implementation)
- Implemented the first concrete code slice of `docs/inventory_execution_workflow.md` after checking it against the current user stories and onboarding docs.
- Inventory execution is now constrained in code and UI:
  - warehouse progression follows allowed next actions (`required -> picked -> packed -> loaded -> ...`) rather than arbitrary free-form stage entry
  - execution transitions are validated in `analytics/db/inventory.py`
- Substitution governance is now explicit in code rather than free-text-only:
  - seeded/admin-manageable inventory substitution reason codes
  - active reason-code validation on request
  - dispatcher / operations approval-role validation on decision
- Inventory and Dispatch surfaces now show execution-aware inventory state:
  - latest execution stage
  - approved/requested substitution quantities
  - pending substitution count
  - recent execution history
- Targeted validation passed in the local `venv`:
  - `tests/test_inventory_and_shipments.py`
  - `tests/test_operations_assignment.py`
  - `tests/test_dashboard_app.py`
  - `tests/test_planner.py`

## 2026-03-13 (Planner/Inventory Adjacent Ops Documentation)
- Documented the next warehouse UX step after constrained pick/pack/load implementation:
  - requirement/container picklists
  - explicit next-action buttons
  - barcode and QR-assisted capture
- Added workforce time capture planning:
  - canonical doc: `docs/worker_time_capture_workflow.md`
  - channels explicitly in scope: app, WhatsApp, and voice/landline call-in with transcription/review
- Added accommodation availability operations planning:
  - canonical doc: `docs/accommodation_availability_operations.md`
  - treated as an operational support signal for remote/peak-period work, not a travel product
- Ran local DB-first context fetch for booking.com/accommodation prior discussion:
  - source used: `db`
  - no direct canonical booking.com thread was resolved
  - availability hits were noisy and not a usable prior spec
  - result: document the idea as a new product direction rather than pretending archive-backed prior intent

## 2026-03-13 (Call Intelligence Foundation)
- Added fake transcript generation as the current recommended ingest surface for call workflow testing; no timestamps or telephony dependency required.
- Documented the preferred StatiBaker delivery-worker design: restartable outbox worker, explicit receipts, no synchronous UI-path delivery, and preserved authority-class distinctions.

- Implemented the first call-intelligence slice after the ITIR/WhisperX/StatiBaker planning pass.
- New Corkysoft operational substrate now exists for:
  - `call_events`
  - transcript artifacts
  - authoritative operator notes
  - extracted actions with accept/reject
  - worker time capture events sharing the same event pipe
  - append-only downstream egress rows for StatiBaker-style consumers
- Integration approach:
  - WhisperX-WebUI is treated as an external async transcription backend via `/transcription/` and `/task/{identifier}`
  - StatiBaker is treated as downstream append-only consumer; Corkysoft now stores outbox-style rows locally but does not yet push them
- UI/API implemented:
  - new `Calls` dashboard tab
  - API routes for call events, notes, extracted actions, transcript artifacts, worker time events, and state egress inspection
- Role-layout defaults were extended to include `Calls` so the new surface participates in the documented operational role model rather than existing as an orphan tab.
- Validation passed in local `venv` for:
  - `tests/test_call_ops.py`
  - `tests/test_dashboard_layouts.py`
  - `tests/test_dashboard_app.py`
  - `tests/test_api.py`
  - `tests/test_operations_assignment.py`
  - `tests/test_inventory_and_shipments.py`
  - `tests/test_planner.py`
- 2026-03-13: Reworked the call-intelligence model from flat `call_event` records into routed `call_session` + `call_leg` handling, with legacy call-event creation preserved as a compatibility wrapper.
- Added explicit routing events so the system can represent direct calls, phone-tree routing, operator->manager consults, operator->worker follow-up, and timesheet branch calls without losing thread continuity.
- Added `ambient_session` support for always-on office transcription; this is now modeled separately from telephony rather than forced into fake phone-call lifecycles.
- Calls Console now centers on sessions/legs while still using accepted notes/actions as authoritative state and raw transcript artifacts as advisory state.
- Validation passed in local `venv` for `tests/test_call_ops.py`, `tests/test_api.py`, and `tests/test_dashboard_app.py` after fixing a migration-order issue around new transcript indexes.
- 2026-03-14: Added the first risk-driven red-team test wave covering multi-leg call contradictions, link correction audit preservation, worker-time duplicate/missing-clock-on anomalies, inventory substitution/readiness outcomes, planner blocked-resource warnings, and Google Sheets worker-import reconciliation.
- Worker time capture now tags duplicate events and missing-prior-clock-on cases in `rawPayload["anomalyFlags"]` and keeps them in `pending_review` instead of auto-accepting high-confidence but suspicious events.
- Focused validation passed in local `venv` for `tests/test_call_ops.py`, `tests/test_api.py`, `tests/test_inventory_and_shipments.py`, `tests/test_operations_assignment.py`, `tests/test_planner.py`, and `tests/test_google_sheets_imports.py`.
- 2026-03-14: Ran the MCP/UI role-completion wave on the already-running local Streamlit server (`http://localhost:8501`) using `routes.db` after seeding one minimal live scenario for dispatcher, warehouse/crew, labor planner, and system admin.
- Persisted via live UI and verified in SQLite:
  - dispatch snapshot export -> `operations_cutover_events`
  - extracted action acceptance -> `call_extracted_actions` + `state_egress_events`
  - warehouse `picked` event -> `inventory_execution_events`
  - worker-time review acceptance -> `worker_time_capture_events`
  - Fleet review logging -> `operations_cutover_events`
- Live-walkthrough blocker discovered:
  - `Calls` and `Inventory` still invoke `st.experimental_rerun()` directly
  - current Streamlit build lacks that attribute
  - operator actions persist, then the page crashes with `AttributeError`
- Fleet remains the model for the fix because it already uses a rerun compatibility helper (`st.rerun()` fallback pattern).
- Additional UX/role findings from the same walkthrough:
  - dispatcher role-layout defaults still omit `Calls`, despite routed call handling now being part of the dispatcher story
  - labor-planner review still effectively terminates in `Calls`; `Staff` / `Driver shifts` do not yet surface reviewed call-derived time capture strongly enough as the downstream ownership surface
- Remaining path explicitly recorded from the walkthrough:
  - rerun-compatibility fix, then rerun the MCP role wave
  - outbox delivery retry/idempotency once StatiBaker worker exists
  - richer inventory custody conflicts
  - barcode/QR execution paths
  - accommodation/provider-side operational support logic
- Added a reusable planning seed harness (`analytics/seed_harness.py`, `scripts/seed_planning_harness.py`) for local Planner/Dispatch/Inventory testing.
- Used the harness to insert 10 new clustered mainland-Australia jobs into `routes.db` on 2026-03-14 (Australia/Brisbane context), with one segment and one container requirement per job.
- Baseline container stock seeded as 30 `Standard Container Pod` units; the initial 10 seeded jobs consume exactly 30 allocated container units, creating dense but non-overflowing planning state.
- Seeded corridor mix now includes: Brisbane->Sunshine Coast (4), Brisbane->Gold Coast (2), Brisbane->Toowoomba (2), Sydney->Newcastle (2), Melbourne->Geelong (1).
- 2026-03-14: Completed the next visual last-mile/planner slice after the initial street-level preview work.
- Map/provider parity:
  - saved-route Folium overlays now use provider-aware Google tile configuration when Google is the active provider, aligning them with the existing Plotly/PyDeck map surfaces.
- Added durable planner site-context persistence:
  - `site_media_assets`
  - `site_assessments`
  - `media_inference_results`
- Planner now consumes accepted site assessments plus reviewed advisory media outputs as planning context for jobs.
- Implemented Google-first 360 URL generation as part of the site-context helper layer; this is still imagery/reference support, not full 360 capture or CV.
- Advisory media/CV outputs are scaffold-only for now:
  - manual/advisory creation
  - accept/reject/correct review path
  - accepted volume estimates and site-feature outputs now surface in Planner warnings/explainability
- Remaining visual last-mile path after this slice:
  - richer interpreted site constraints on top of accepted assessments/media
  - real walkaround ingestion workflow polish
  - actual model-backed CV/object detection/volume estimation
- 2026-03-14: Clarified planner docs that future model-backed walkaround CV / volume-estimation services should attach reviewed outputs to quote/job records first, then feed planner constraints only after acceptance/correction.
- 2026-03-15: Clarified priority that richer interpreted site constraints are the correct planner direction, but should not become the immediate implementation focus before the media/CV evidence pipeline is more real. Current heuristic constraints are a bridge; deeper constraint logic should follow production-grade walkaround ingestion and reviewed model outputs.
- 2026-03-14: Seed harness updated to backfill deterministic synthetic jobs into both live jobs and historical_jobs/historical_job_routes so default historical-first analysis surfaces can see the seeded route geometry without manual populate actions.
- 2026-03-14: Planner routing preview changed to prefer stored route geometry only; removed the normal straight-line fallback path from planner preview.
- 2026-03-15: Implemented the practical non-CV hardening slice:
  - replaced direct rerun calls in `Calls` and `Inventory` with the compatibility helper, removing the confirmed live crash path under the current Streamlit build
  - updated `upsert_job_by_number(...)` to attempt live route-geometry enrichment automatically after job creation/update
  - made Planner site-summary empty states more explicit that accepted manual site assessments are what unlock planning consequences like truck fit, shuttle need, labor uplift, and access/load uplift
- 2026-03-15: Live MCP recheck confirmed:
  - fake transcript generation in `Calls` now succeeds without rerun crash
  - custody/location update in `Inventory` now succeeds without rerun crash
  - app remained at 0 console errors during the check
- 2026-03-15: Implemented the next hardening pass:
  - `persist_quote(...)` now attempts historical-job route-geometry enrichment automatically after quote save
  - added dispatcher stale-layout detection/repair so older stored layouts missing `Calls` can be fixed from the main dashboard layout controls
  - exposed worker-time capture state much more clearly in `Staff` and `Driver shifts`, with labor-surface review controls and reviewed-event visibility
- 2026-03-15: Follow-up clarification and refinement:
  - audited remaining non-test job/historical creation seams and did not find another production writer outside the already-covered shared helpers that needed new geometry-enrichment code
  - strengthened `Driver shifts` so accepted call-derived worker-time and imported VEHICLE_DRIVER rows are compared explicitly via `matched / imported_only / call_only` reconciliation output
- 2026-03-15: Focused validation after that slice:
  - `52 passed` across `tests/test_quote_service.py`, `tests/test_dashboard_layouts.py`, `tests/test_dashboard_app.py`, and `tests/test_call_ops.py`
- 2026-03-15: Tightened `Driver shifts` reconciliation again:
  - matching is no longer only date/worker/truck exact-key based
  - same-worker/same-day accepted events are now checked against imported shift windows; out-of-window accepted events classify as `time_mismatch`
  - mismatch taxonomy is now `truck_mismatch`, `job_mismatch`, `assignment_mismatch`, `time_mismatch`, plus `imported_only` / `call_only`
- 2026-03-15: Live verification notes for that refinement:
  - seeded a deterministic `Riley Worker` mismatch case into `routes.db`
  - the live `Driver shifts` tab rendered with `Truck/job mismatch = 1`
  - the seeded imported shift + accepted event classified correctly as `time_mismatch` when queried through the same helper the UI uses
- 2026-03-15: Focused validation after the refinement:
  - `tests/test_dashboard_app.py`: `5 passed`
- 2026-03-15: Polished `Driver shifts` reconciliation UI:
  - added operator-facing display labels/explanations for reconciliation classes instead of surfacing only raw status codes
  - renamed the aggregate metric to `Mismatch / timing drift` because the bucket includes timing drift as well as truck/job assignment mismatches
- 2026-03-15: Explicitly deferred start/end tolerance rules:
  - current worker-time capture remains simple `clock_on` / `clock_off`
  - exact shift-window containment is the active rule for now
  - finer start/end proximity logic should wait until the worker-time event model becomes richer or current reconciliation proves too blunt
- 2026-03-15: Added `docs/payroll_and_labor_analytics.md` as the canonical payroll-preparation and labor-statistics spec.
- 2026-03-15: Locked the new layer as analytics + payroll prep rather than full payroll execution.
- 2026-03-15: Locked the privacy posture as low-surveillance by default: aggregate/trend/exception views first, with person-level drill-down only when justified.
- 2026-03-15: Updated operator stories, onboarding, UI role matrix, README, and roadmap so payroll/labor analytics is treated as a future product layer rather than being muddled into `Staff` / `Driver shifts`.
- 2026-03-15: Implemented the first `Payroll / Labor analytics` surface.
- 2026-03-15: Added a shared derived analytics layer over planned labor, imported shifts, and reviewed worker-time events, reused by both the dashboard tab and read-only API endpoints.
- 2026-03-15: Implemented v1 sections for pay forecast, overtime/hours/cost distributions, plan-vs-actual, confidence/anomaly summaries, and labor cost drivers.
- 2026-03-15: Initially kept absence/sick-day analytics deferred rather than building weak inference from missing events; later in the same milestone replaced that gap with a basic explicit recorded absence/leave model.
- 2026-03-15: Extended payroll analytics with export-ready labor summaries and a basic explicit `worker_absence_records` model.
- 2026-03-15: Added read-only labor-analytics absence/export endpoints plus mutating absence-record API creation, keeping Corkysoft at payroll-prep truth rather than payroll execution.
- 2026-03-15: Replaced deferred absence status in the payroll cockpit with basic recorded absence/leave analytics grounded in explicit rows instead of inferred missing shifts.
- 2026-03-21: Locked dashboard auth direction:
  - use Streamlit's native Google OIDC support rather than inventing a separate cookie/session stack
  - keep Corkysoft authorization local through a `dashboard_users` allowlist keyed by email
  - shared/deployed environments should fail closed into auth-required mode
  - anonymous UI access remains only for explicit local development runs via `CORKYSOFT_ENV=development` and `CORKYSOFT_ALLOW_ANONYMOUS_UI=1`
  - bootstrap the first admin explicitly from env instead of allowing permissive first-login account creation
- 2026-03-22: Auth red-team hardening direction:
  - role-hidden tabs are part of the authz boundary for new auth-sensitive surfaces, not just cosmetic layout state
  - query-param tab requests should not re-expose hidden admin tabs like `Kent admin`
  - bootstrap-admin env seeding should behave as a first-user bootstrap only, not as a perpetual admin reassertion path if env vars linger
- 2026-03-22: Implemented the first auth red-team hardening slice:
  - added explicit auth red-team documentation and TODO alignment
  - fixed hidden-tab query-param escalation in dashboard layout resolution
  - made bootstrap-admin seeding no-op once dashboard users already exist
  - added targeted auth/layout regression tests around those paths
- 2026-03-24: Added a dedicated progress-tracking board in `docs/progress_status_board.md` and linked it from README and ROADMAP so documentation/TODO/changelog updates can be coordinated around one source of truth.

## 2026-03-24 (Situational-Awareness Ingestion)

- Added the `disruption_events` table plus `analytics/situational_awareness.py` so closure, weather, and traffic severity events can be persisted together with location/source metadata and normalized timestamps.
- `update_adaptive_policy_from_disruptions` now summarizes recent severity totals and nudges the weather, closure, and lane-ETA multipliers via `apply_bounded_parameter_target`; severity-agnostic defaults and bounded deltas keep the policy state contractive.
- Tests (`tests/test_situational_awareness.py`) cover event aggregation, severity filtering, and parameter updates against the shadow global-parameter store.
- Docs updated: `README.md`, `ROADMAP.md`, `docs/adaptive_learning_loop.md`, `docs/progress_status_board.md`, and `CHANGELOG.md` now describe the ingestion + adaptive-policy step.
