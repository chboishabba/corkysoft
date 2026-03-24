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
