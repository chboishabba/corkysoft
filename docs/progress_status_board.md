# Progress Status Board

Last updated: **2026-03-31**

This page is the operational tracker for implementation-to-docs alignment.  
Use it with [ROADMAP.md](../ROADMAP.md), [README.md](../README.md), and [COMPACTIFIED_CONTEXT.md](../COMPACTIFIED_CONTEXT.md).

## Progress Snapshot

- Core routing and costing: **🟢 implemented**
- Streamlit dashboard baseline: **🟢 implemented**
- Planner core interaction flow: **🟢 implemented** (hybrid planner scaffold and site-risk data model)
- Operations diary surface: **🟢 implemented** (core day/week/task/invoice-bill views plus job-scoped labor reconciliation detail)
- Auth and role-boundary hardening: **🟢 implemented** (Google OIDC + local allowlist + hidden-tab guards now include a browser-testable auth harness, Playwright coverage for the core denial/banner paths, explicit remote-origin / `redirect_uri` mismatch handling, and an explicit temporary auto-provision-admin mode for tightly controlled owner/testing sharing)
- Historical job corridor/lane model: **🟢 implemented** (canonical clusters/lanes/groups, assignment status, promotion governance, and planner-safe consumption)
- Inventory execution maturity beyond baseline: **🟡 in progress**
- Adaptive policy: **🟡 bounded proposal/approval workflow implemented, broader operational rollout still pending**
- Kent payload governance and operator workflow validation: **🟡 in progress** (Kent admin policy and reason-code writes are now gated to `system_rollout_admin`; broader live-payload and operator validation still pending)
- Situation-awareness/disruption ingestion and bounded auto-policy use: **🟢 implemented**
- Historical ingest coverage governance: **🟢 implemented** (run summaries, row issues, readiness status, Fleet visibility)
- Lane trust-boundary controls across analytics: **🟢 implemented** (Planner, route maps, live network, profitability, price history, optimizer, summary/histogram)
- Grouped lane promotion review: **🟢 implemented** (repeated candidate cluster pairs can collapse into one governance proposal)
- Observer export for diary/reconciliation: **🟢 implemented** (observer outbox for task/review families plus explicit planning-snapshot and reconciliation-exception export)
- Observer-outbox dashboard visibility: **🟢 implemented** (Operations diary now exposes filtered observer envelopes with payload/provenance inspection)
- Quote recommendation and benchmarking overlays: **🟢 implemented** (live quote workflow now shows benchmark guidance and recommendation overlays)
- Backhaul-aware quote discount guidance: **🟢 implemented** (quote workflow now surfaces backhaul discount headroom and recommendation support)
- Broader backhaul / container-sharing operational visibility: **🟡 in progress** (Dispatch now surfaces persisted spare-capacity signals, container pressure, share/reallocation recommendations, and explicit operator response actions; deeper execution handling still pending)
- Regression / corridor modeling: **🟡 in progress** (profitability insights now include baseline and corridor-aware margin-per-m³ regression over distance and season, fit-improvement reporting, and holdout trust signals; the next work is governance-first, with suppression/promotion rules and rolling backtests before broader feature expansion)
- MCP adapter for cross-project/tooling access: **🟡 in progress** (read-only-first v1 contract, local registry, bridge-default CLI, JSON bridge, and four bounded read-only tools are implemented; FastMCP remains optional and broader rollout remains pending)
- App/API/pricing decomposition pass: **🟢 implemented** (major shell/router/pricing hotspots collapsed into bounded modules with compatibility facades preserved)
- Dashboard shell control-layer decomposition: **🟢 implemented** (auth, query-param, shell, layout-state, data-controls, and tab-registry helpers now exist; current follow-up is second-pass boundary polish)

## Active TODO Alignment (Current Wave)

- Priority 1: operator execution completion
- finish the operations-diary path toward fuller job, staff, and invoice reconciliation workflow
- extend dispatch and inventory handling from explicit response actions into fuller execution flows for backhaul, container-sharing, and under/over-utilisation response

- Priority 2: governance and contract hardening
- validate Kent operator/admin workflows against real payloads and repeated override patterns now that admin-only governance controls are enforced
- harden the implemented Corkysoft MCP v1 adapter with stronger scenario, transport, and result-envelope coverage while keeping mutable tools out of scope
- extend auth browser coverage from the implemented local harness into real Google-backed login/logout and deployed tunnel-origin checks when deployment posture matters
- add delivery receipt and watermark semantics to the observer outbox when deployment posture matters

- Priority 3: decision quality and planner intelligence
- extend the corridor-aware model from basic holdout validation into suppression thresholds, promotion rules, and rolling backtests
- add corridor-season and holiday/day-type interaction features only after governance and backtesting are stable
- add operator-facing uncertainty ranges, with prediction intervals preferred over mean-only point estimates
- add explicit evidence-backed site/media/CV ingestion before deeper constraint automation
- continue second-pass dashboard decomposition polish, especially helper consolidation around `dashboard/components/quote_builder.py`

## Active TODO Alignment (Next Parallel Wave)

- Worker 1 lane: deepen operator reconciliation in the operations diary and adjacent inventory flow without broad workflow redesign
- Worker 2 lane: harden outbound governance and contract surfaces, especially MCP and Kent-facing guarantees
- Worker 3 lane: finish quote-builder and shared-state consolidation so dashboard control logic continues to move out of local duplicates

## Active TODO Alignment (Current Wave)

Status: completed on 2026-03-31 with focused integrated validation green.

- Priority 1: quote-builder boundary cleanup
- remove the remaining local helper ownership in the quote workflow where shared dashboard state or query-param behavior should be authoritative
- keep quote decision-support behavior unchanged while reducing local control-surface duplication

- Priority 2: shared rerun/state helper consolidation
- remove remaining duplicated rerun/state helper definitions from planner and route-map surfaces
- keep rerun semantics identical while collapsing ownership onto the shared dashboard state layer

- Priority 3: Kent and MCP live-control validation coverage
- add stronger regression coverage around Kent write-gate behavior and MCP tool result/control invariants
- keep the posture read-only and governed, with docs reflecting the validated control boundary

## Active TODO Alignment (Next Cleanup Wave)

- Priority 1: finish quote decision-control boundary cleanup
- move remaining quote-only decision helpers onto the right shared ownership layer where reuse or policy authority is warranted

- Priority 2: continue rerun/state helper consolidation
- remove the remaining local rerun wrappers from other dashboard components such as calls and maps without changing behavior

- Priority 3: live governance validation beyond static contract tests
- add broader payload and scenario validation for Kent and MCP result envelopes where deployment posture matters

## Active TODO Alignment (Current Cleanup Wave)

Status: partially completed on 2026-03-31 with focused integrated validation green.

- Priority 1: quote decision-control cleanup
- move quote-only decision helpers onto the shared quote-service or state layer only where ownership is truly reusable and policy-bearing

- Priority 2: remaining rerun-wrapper consolidation
- remove local rerun wrappers from the real remaining backlog while preserving behavior

- Priority 3: broader governance scenario validation
- extend Kent and MCP tests from static contract assertions into richer scenario and result-envelope validation without widening mutable scope

## Active TODO Alignment (Next Cleanup Wave)

- Priority 1: finish quote decision-control boundary decisions
- decide which remaining quote UI concerns should stay local versus move into shared helpers, especially manual-override and modal/query-param coordination

- Priority 2: verify and prune the rerun-wrapper backlog
- remove stale cleanup assumptions from the roadmap where consolidation has already happened, and target only real remaining local wrappers

- Priority 3: broaden governance validation further
- extend Kent and MCP coverage from envelope invariants into richer scenario and payload-behavior checks where deployment posture matters

## Active TODO Alignment (Current Narrow Cleanup Wave)

Status: completed on 2026-03-31 with focused integrated validation green.

- Priority 1: quote UI-versus-shared helper boundary decision
- keep only genuinely reusable or policy-bearing quote logic in shared helpers, leaving UI-only reset and widget concerns local

- Priority 2: real rerun-wrapper backlog cleanup
- consolidate the actual remaining rerun duplication in planner, maintenance, and operations rather than the stale calls/maps assumption

- Priority 3: deeper Kent and MCP scenario validation
- extend governance coverage from envelope invariants into richer scenario and payload-behavior checks without widening mutable scope

## Active TODO Alignment (Next Narrow Cleanup Wave)

- Priority 1: finish quote helper boundary decisions
- decide whether any remaining local quote helpers are truly shared-policy logic or should remain component-local UI orchestration

- Priority 2: audit rerun cleanup completion
- completed: repo audit confirms there are no other local rerun helpers or direct rerun calls outside the shared state layer

- Priority 3: deepen governance scenario coverage
- extend Kent and MCP validation from the current envelope and execution-error checks into richer payload and scenario behavior where deployment posture matters
- completed in this pass: non-admin Kent admin rendering now has regression coverage proving write controls are disabled in the UI, not only gated by helper logic
- completed in this pass: MCP bridge coverage now includes blank-tool-name and unknown-tool-name behavior, in addition to envelope and execution-error assertions

## Non-blocking documentation tasks

- Keep [ROADMAP.md](../ROADMAP.md), [README.md](../README.md), and user-stories/docs aligned whenever behavior changes.
- Run this board update whenever any item moves between statuses (complete/in-progress/blocked).

## Update protocol

1. Update ROADMAP item wording when feature scope changes.
2. Update this board with status and blockers in one pass.
3. Keep [COMPACTIFIED_CONTEXT.md](../COMPACTIFIED_CONTEXT.md) evidence entries current for decisions.
4. Record the change in [CHANGELOG.md](../CHANGELOG.md).
