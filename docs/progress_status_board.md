# Progress Status Board

Last updated: **2026-04-02**

This page is the operational tracker for implementation-to-docs alignment. Use
it with [ROADMAP.md](../ROADMAP.md), [README.md](../README.md), and
[COMPACTIFIED_CONTEXT.md](../COMPACTIFIED_CONTEXT.md).

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
- Kent payload governance and operator workflow validation: **🟡 in progress**

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
- Validation note: targeted provider and shell regression suites pass in the
  project virtualenv, including a 133-test focused pass across dashboard shell,
  layout, provider, isochrone, and quote-state coverage; repo guidance is
  being tightened so agent/user execution stays inside the repo venv.

## Current Wave

- Priority 1: document the five-view shell as the canonical user entry model
- update onboarding, role coverage, roadmap, and README copy so they explain
  shell entrypoints first and leaf workflows second

- Priority 2: harden decision-signal governance in the new views
- replace static KPI and alert content with sourced metrics, freshness stamps,
  and explicit unknown/fallback states

- Priority 3: keep the shell reviewable
- expand regression coverage around shared UI primitives, role-layout reset,
  deep-link landing, support-grade workspace sharing, and mixed-surface
  composition boundaries

- Priority 4: normalize the architecture surface
- keep one reviewed metasystem view and child UML/C4 drill-down set aligned
  with the implemented shell and control boundaries

- Priority 5: formalize operational data contracts
- require source, owner, freshness, stale-threshold, and fallback semantics for
  decision-adjacent shell data before it is treated as operational truth

- Future lane: customer-facing tracking and receipt surfaces
- build a separate public/customer-safe status and receipt contract on top of
  live telemetry and job-status primitives without reusing internal shell state

## Worker Lanes

- Worker 1 lane: sourced shell signals
- Worker 2 lane: state-addressable shell and regression hardening
- Worker 3 lane: architecture surface normalization
- Worker 4 lane: operational data contracts

## Orchestrator Control Map

- Worker 1 control objective: replace shell scaffolding with contract-backed
  KPI/alert signals that expose source, owner, freshness, and fallback.
- Worker 2 control objective: make workspace state reproducible, shareable, and
  fail-closed under role and query-param constraints.
- Worker 3 control objective: keep the reviewed metasystem UML/C4 entrypoint
  and child shell diagrams aligned with current control boundaries.
- Worker 4 control objective: make operational data contracts explicit before
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
- any future customer-facing tracking link remains tokenized, expiring,
  least-privilege, and auditable rather than mirroring internal shell state

## Non-blocking Documentation Tasks

- Keep [ROADMAP.md](../ROADMAP.md), [README.md](../README.md), and role-facing
  docs aligned whenever shell labels or role entrypoints change.
- Update the C4/PlantUML dashboard-shell diagrams when the five-view shell
  structure changes materially.

## Update Protocol

1. Update roadmap wording when feature scope or shell ownership changes.
2. Update this board in the same pass when status or blockers move.
3. Keep [COMPACTIFIED_CONTEXT.md](../COMPACTIFIED_CONTEXT.md) aligned with the
   current audit conclusions.
4. Record externally visible workflow changes in [CHANGELOG.md](../CHANGELOG.md).
