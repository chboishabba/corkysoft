# Changelog

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
