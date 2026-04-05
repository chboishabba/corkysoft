# Naive User Tester Notes

This document records plain-language user-testing observations from the point of
view of an ideal but naive first-time operator. It is intentionally verbal and
direct. The goal is to surface friction, confusion, and trust issues that a
spec-only review will miss.

## What I Am Looking For

- Can I tell where to start for my role?
- Can I tell which tabs are operational versus administrative?
- Can I understand what the system wants me to do next?
- Can I recover when a workflow is blocked?
- Do the cutover controls feel governed rather than improvised?

## What I Want As A New User

- a clear start point for each role
- visible separation between execution work and admin/config work
- obvious explanations when the system blocks or warns
- confidence that actions are logged and reviewable
- no need to infer workflow state from tribal knowledge

## What I Generally Dislike

- tabs that mix frequent operator actions with low-frequency admin settings
- wording that assumes I already know internal rollout jargon
- actions that appear clickable before prerequisites are actually satisfied
- surfaces that make me guess whether imported spreadsheet data or Corkysoft
  data is the real planning truth

## Current Walkthrough Notes

Date:
- 2026-03-12

Seeded scenario used:
- three jobs:
  - one fully planned dispatchable move
  - one unassigned move with warning-state planning
  - one move affected by an expired-truck readiness issue
- two open Kent tenders
- one blocked vehicle in Fleet readiness
- one rollout workflow promoted from `dual_run` to `native_primary`

Role flows exercised through the live app:
- estimator in `Quote builder`
- dispatcher in `Dispatch`
- operations / fleet manager in `Operations` and `Fleet`
- commercial owner in `Kent admin` and rollout approval path
- rollout coordinator / admin in Fleet cutover admin

What felt smooth:
- the role tabs are now meaningfully separated
- `Dispatch` reads like an execution surface, not a config surface
- `Operations` makes it obvious that `job_segments` are the planning unit
- Fleet cutover admin shows the promotion story clearly once I am inside it:
  request, approval, then apply
- Kent operator and Kent admin surfaces are separated in a way that makes sense

What I was looking for while testing:
- whether I could infer the right tab without repo knowledge
- whether the promotion path felt governed
- whether operational tabs felt native-first rather than spreadsheet-first
- whether blocked/readiness state was visible enough to trust

What I liked:
- the approval trail is explicit and visible in one place
- snapshot export is straightforward from Dispatch
- blocked vehicle state is visible in Fleet without digging into spreadsheets
- the native planning language is stronger than it was before

What I disliked:
- the global `historical_jobs` warning appears on operational tabs where it is
  irrelevant and noisy
- the app still opens on the analytics/histogram surface, which is not the best
  default for most operators
- Fleet cutover admin is powerful, but it is still dense; a new user has to
  parse a lot of fields before understanding the next action

Areas for improvement:
- suppress historical analytics warnings on tabs that are operational rather
  than historical-analysis-driven
- consider a role-aware landing flow or at least a more operational default tab
- make the next required rollout action more visually dominant than the
  surrounding config fields

What should change next:
- clean up non-role-specific analytics warnings from operational tabs
- keep tightening onboarding so a first-time operator lands on the right
  workflow faster

## Red-Team Role Walkthrough

Date:
- 2026-03-14

Live server used:
- existing local Streamlit server on `http://localhost:8501`

Live DB used:
- `routes.db`

Minimal live scenario added for the walkthrough:
- one planned job/segment with truck + worker assignment
- one container-heavy inventory requirement with shortage and pending substitution
- one routed call session with manager consult and pending extracted action
- one ambiguous worker time capture event pending review
- one dispatch cutover workflow in `dual_run`

Roles exercised through the live UI:
- dispatcher
- warehouse / crew
- labor planner
- system admin

What completed successfully:
- `Dispatcher`
  - exported a dispatch snapshot from `Dispatch`
  - event persisted to cutover history
- `Dispatcher`
  - accepted a pending extracted action in `Calls`
  - decision persisted with audit trail and outbox event
- `Warehouse / Crew`
  - recorded a constrained `picked` execution update in `Inventory`
  - event persisted with actor, container ref, and truck ref
- `Labor Planner`
  - reviewed an ambiguous worker time event in `Calls`
  - accepted linkage persisted to worker/job/segment with review note
- `System Admin`
  - recorded a cutover review in `Fleet`
  - event persisted and `last review` updated live

What broke at the time:
- `Calls` action flows were still calling `st.experimental_rerun()` after persisting changes
- `Inventory` execution/substitution flows were still calling `st.experimental_rerun()` after persisting changes
- under the tested Streamlit build, those actions succeeded in the DB and then crashed the page with:
  - `AttributeError: module 'streamlit' has no attribute 'experimental_rerun'`

Current status:
- those flows have since been moved onto the shared rerun compatibility helper
- continue treating this note as historical evidence for why rerun usage must stay centralized in `dashboard/state.py`

What I liked:
- the persisted operational truth was correct even when the UI crashed afterward
- `Dispatch` already gives a useful execution summary for shortages and substitution pressure
- `Fleet` cutover admin is the cleanest governed surface in the current app

What I disliked:
- dispatcher role defaults still emphasize `Dispatch`, `Planner`, `Kent tenders`, and `Operations`, but not `Calls`
- that does not match the real dispatcher workflow now that routed call handling is first-class
- labor-planner review currently happens in `Calls`; `Staff` and `Driver shifts` do not yet feel like the natural downstream review surface for call-derived time capture
- warehouse flows still succeed-then-crash, which is a trust problem even though the database state is correct

Remaining path from this walkthrough:
- keep remaining operator surfaces on the shared rerun compatibility helper and avoid reintroducing direct `st.experimental_rerun()` calls
- rerun the same MCP/UI role wave after that fix so success is measured in both persistence and uninterrupted operator flow
- add explicit outbox delivery retry/idempotency coverage once the StatiBaker delivery worker exists
- deepen inventory custody conflict handling beyond current latest-location assertions
- add barcode/QR execution paths on top of the constrained warehouse workflow
- add accommodation/provider-side operational support logic once Planner/Dispatch availability signals are ready to consume live provider data
