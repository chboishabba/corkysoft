Unified deliverables map for Corkysoft, aligning current implementation status with the planned roadmap.

---

## Documentation TODOs

- Keep `spec.md`, `plan.md`, `status.json`, and `devlog.md` current during the remediation milestone.
- Keep `docs/progress_status_board.md` and this roadmap synchronized after any feature status changes.
- Validate the deliverable status tables against the current code and tests.
- Decide whether `corkysoft/src/dashboard` remains a packaging stub or should be wired to the main Streamlit entry point.
- Maintain `docs/contributor_docs_sync.md` as the contributor-facing rule for README/ROADMAP/progress-board/docs alignment after feature or refactor changes.
- Keep `docs/modules.md` updated when module responsibilities or entry points change.
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
- Maintain a focused auth red-team plan and expand it into executable browser-based checks after the current unit/static hardening pass.
- Treat role-hidden admin tabs as part of the authz boundary; do not allow query-param navigation or stale session state to re-expose them.
- Keep bootstrap-admin seeding one-shot and explicit; do not let lingering env vars silently reassert admin access after user setup exists.
- Deepen the `Planner` tab from the current hybrid scaffold toward richer site-aware and more interactive visual planning.
- Add a separate `Operations diary` above Planner and Dispatch so managers can review day/week workload, usage, tasks, and invoice/bill exceptions without collapsing those concerns back into the route-planning UI.
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
- Separate operator Kent workflows from admin policy/reason-code management in both docs and dashboard UX.
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
- Make the Dispatch tab the primary job-centric execution surface across trucks, workers, inventory, suppliers, and readiness flags.
- Add a manager-facing day/week diary that links jobs, tasks, vehicle usage, staff usage, labor reconciliation, customer invoice readiness, and subcontractor-bill reconciliation.
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
| **Backhaul / container sharing**                         |   🧩   | Quote backhaul detection and discount guidance are implemented; broader operational container-sharing and under-/over-utilisation handling still pending. |
| **Truck / driver cost baselines**                        |    ✅   | Base fuel/driver/maintenance + overhead parameters drive break-even engine. |
| **Private cost component ledger**                        |    ✅   | Record crew, truck, fuel and other cost inputs per job inside SQLite.       |

---

## 🧮 5. Analytics & Statistical Modelling

| Deliverable                            | Status | Description                                  |
| -------------------------------------- | :----: | -------------------------------------------- |
| **Airbnb-style $/m³ histogram**        |    ✅   | Jobs sorted left→right by $/m³; bar = count. |
| **Break-even + margin bands**          |    ✅   | Visual overlay on histogram.                 |
| **Loss-leader detection**              |    ✅   | Identify sub-margin jobs.                    |
| **Regression / corridor model**        |   🔜   | Predict margins vs distance/season.          |
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
