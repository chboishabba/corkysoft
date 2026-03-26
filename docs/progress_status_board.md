# Progress Status Board

Last updated: **2026-03-27**

This page is the operational tracker for implementation-to-docs alignment.  
Use it with [ROADMAP.md](../ROADMAP.md), [README.md](../README.md), and [COMPACTIFIED_CONTEXT.md](../COMPACTIFIED_CONTEXT.md).

## Progress Snapshot

- Core routing and costing: **🟢 implemented**
- Streamlit dashboard baseline: **🟢 implemented**
- Planner core interaction flow: **🟢 implemented** (hybrid planner scaffold and site-risk data model)
- Operations diary surface: **🟢 implemented** (core day/week/task/invoice-bill views plus job-scoped labor reconciliation detail)
- Auth and role-boundary hardening: **🟢 implemented** (Google OIDC + local allowlist + hidden-tab guards)
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
- Regression / corridor modeling: **🟡 in progress** (profitability insights now include baseline and corridor-aware margin-per-m³ regression over distance and season, with fit-improvement reporting and operator-safe interpretation notes; fuller prediction/validation still pending)
- App/API/pricing decomposition pass: **🟢 implemented** (major shell/router/pricing hotspots collapsed into bounded modules with compatibility facades preserved)

## Active TODO Alignment (Current Wave)

- Continue operations-diary expansion toward fuller job/staff/invoice reconciliation workflow.
- Extend dispatch/inventory handling from explicit response actions into fuller execution flows for backhaul/container-sharing and under-/over-utilisation response.
- Extend the current corridor-aware model into stronger holdout-style validation, corridor-level prediction governance, and safer operator guidance around when not to trust the model.
- Add delivery receipt / watermark semantics to the observer outbox when deployment posture matters.
- Extend operations inventory with barcode/QR and container-specific execution controls.
- Add explicit evidence-backed site/media/CV ingestion before deeper constraint automation.
- Add and maintain explicit policy/override governance for hard-block categories.
- Extend adaptive policy governance from proposal/approval into broader operational rollout and override controls.
- Validate Kent operator/admin workflows against real payloads and repeated override patterns now that admin-only governance controls are enforced in the dashboard.

## Non-blocking documentation tasks

- Keep [ROADMAP.md](../ROADMAP.md), [README.md](../README.md), and user-stories/docs aligned whenever behavior changes.
- Run this board update whenever any item moves between statuses (complete/in-progress/blocked).

## Update protocol

1. Update ROADMAP item wording when feature scope changes.
2. Update this board with status and blockers in one pass.
3. Keep [COMPACTIFIED_CONTEXT.md](../COMPACTIFIED_CONTEXT.md) evidence entries current for decisions.
4. Record the change in [CHANGELOG.md](../CHANGELOG.md).
