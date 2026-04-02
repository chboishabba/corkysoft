# Remediation Plan

## Phase 1: Truth Alignment

- Create canonical planning state.
- Align README, ROADMAP, and Kent docs to one current-state narrative.
- Add product-centered navigation and current-state wording.

## Phase 2: Product and Governance

- Add operator user stories.
- Add quote -> tender -> override -> awarded job lifecycle doc.
- Tighten Kent override governance.
- Rewrite Live Network and ingest contracts to MVP/current-state truth.
- Complete multi-truck policy definitions needed for v1.

## Phase 3: Code and UX Hardening

- Fix Kent top-N prioritization correctness.
- Govern hard-block categories.
- Make `dry_run` side-effect free.
- Separate operator and admin Kent dashboard flows.
- Add tests for corrected Kent behavior.

## Phase 4: Shell And Operator-Flow Remediation

- Split analytics-first shell chrome from operational role chrome.
- Fix role-layout reset/repair and deep-link behavior so role-aware landing is deterministic.
- Remove remaining direct `st.experimental_rerun()` usage from live operator surfaces.
- Re-section mixed surfaces (`Fleet`, `Inventory`, `Staff`) so execution, review, and admin work are more obviously separated.

## Parallel Dev Lanes Before The Next Testing Wave

- Worker 1: dashboard shell + role landing/reset hardening (`dashboard/app.py`, `analytics/dashboard_layouts.py`, layout tests).
- Worker 2: rerun-compatibility sweep across live operator surfaces (`dashboard/components/dispatch.py`, `dashboard/components/operations_diary.py`, related tests/helpers).
- Worker 3: mixed-surface information architecture improvements for labor/inventory/admin flows (`dashboard/components/staff.py`, and follow-on sectioning in mixed operator/admin tabs as needed).

## Phase 5: App Shell Decomposition

- Extract auth flow and query-param routing out of `dashboard/app.py`.
- Extract shell selection and cross-surface layout/session hydration out of `dashboard/app.py`.
- Extract dataset/filter setup and tab composition out of `dashboard/app.py`.

## Phase 6: Second-Pass Boundary Polish

- Keep `dashboard/app.py` as a composition-only layer.
- Split `dashboard/data_controls.py` along stable control boundaries so provider, ingest, dataset-load, and filter logic are easier to verify.
- Consolidate duplicated pin and route-label helper ownership where it still exists outside the shared state module.
- Keep architecture and module docs aligned with every boundary change.

## Phase 7: Prioritized Remaining Roadmap Wave

- Priority 1: operator execution completion
- Priority 2: governance and contract hardening
- Priority 3: decision-quality and planner intelligence upgrades

Rationale:
- ITIL: stabilize the highest-frequency operator service paths before expanding supporting intelligence layers.
- ISO 9001: reduce uncontrolled workflow variation by tightening ownership, evidence, and approval paths.
- Six Sigma: address the most likely defect-producing surfaces first: execution flow gaps, governance ambiguity, then advisory-model drift.

Current non-blocking worker lanes:

- Worker 1: operator execution completion lane
  files: `dashboard/components/dispatch.py`, `dashboard/components/operations_diary.py`, `dashboard/components/inventory.py`, `analytics/operations_diary.py`, `analytics/operations_assignment.py`, related focused tests
- Worker 2: governance and contract hardening lane
  files: `dashboard/components/kent.py`, `analytics/auth.py`, `corkysoft/mcp/`, `docs/kent_ams_integration.md`, `docs/corkysoft_mcp_v1.md`, related focused tests
- Worker 3: dashboard boundary and decision-quality lane
  files: `dashboard/components/quote_builder.py`, `dashboard/state.py`, `analytics/price_distribution.py`, `analytics/profitability_*.py`, related focused tests and docs

## Phase 8: Next Parallel Wave

- Priority 1: operator reconciliation and execution completion
- Priority 2: outbound contract and governance hardening
- Priority 3: dashboard boundary consolidation and decision-support polish

Current non-blocking worker lanes:

- Worker 1: operator reconciliation lane
  files: `dashboard/components/operations_diary.py`, `analytics/operations_diary.py`, `dashboard/components/inventory.py`, related focused tests
- Worker 2: outbound contract and governance lane
  files: `corkysoft/mcp/`, `docs/corkysoft_mcp_v1.md`, `docs/kent_ams_integration.md`, `dashboard/components/kent.py`, related focused tests
- Worker 3: quote-builder and shared-state consolidation lane
  files: `dashboard/components/quote_builder.py`, `dashboard/state.py`, `dashboard/query_params.py`, related focused tests

## Phase 9: Boundary Cleanup And Live-Control Validation

- Priority 1: quote-builder boundary cleanup
- Priority 2: shared rerun/state helper consolidation
- Priority 3: Kent and MCP live-control validation coverage

Current non-blocking worker lanes:

- Worker 1: quote-builder boundary cleanup lane
  files: `dashboard/components/quote_builder.py`, `dashboard/state.py`, `dashboard/query_params.py`, related focused tests
- Worker 2: shared rerun/state cleanup lane
  files: `dashboard/components/planner.py`, `dashboard/components/route_maps.py`, `dashboard/state.py`, related focused tests
- Worker 3: governance live-validation lane
  files: `dashboard/components/kent.py`, `corkysoft/mcp/`, `docs/kent_ams_integration.md`, `docs/corkysoft_mcp_v1.md`, related focused tests

## Phase 10: Final Boundary And Validation Cleanup

- Priority 1: quote decision-control cleanup
- Priority 2: remaining rerun-wrapper consolidation
- Priority 3: broader governance scenario validation

Current non-blocking worker lanes:

- Worker 1: quote decision-control cleanup lane
  files: `dashboard/components/quote_builder.py`, `corkysoft/quote_service.py`, `dashboard/state.py`, related focused tests
- Worker 2: remaining rerun-wrapper consolidation lane
  files: `dashboard/components/planner.py`, `dashboard/components/maintenance.py`, `dashboard/components/operations.py`, `dashboard/state.py`, related focused tests
- Worker 3: broader governance scenario-validation lane
  files: `dashboard/components/kent.py`, `corkysoft/mcp/`, `docs/kent_ams_integration.md`, `docs/corkysoft_mcp_v1.md`, related focused tests

## Phase 11: Narrow Cleanup Wave

- Priority 1: quote UI-versus-shared helper boundary decision
- Priority 2: real rerun-wrapper backlog cleanup
- Priority 3: deeper Kent and MCP scenario validation

Current non-blocking worker lanes:

- Worker 1: quote UI/shared-helper boundary lane
  files: `dashboard/components/quote_builder.py`, `dashboard/state.py`, `corkysoft/quote_service.py`, related focused tests
- Worker 2: rerun-wrapper cleanup lane
  files: `dashboard/components/planner.py`, `dashboard/components/maintenance.py`, `dashboard/components/operations.py`, `dashboard/state.py`, related focused tests
- Worker 3: deeper governance scenario-validation lane
  files: `dashboard/components/kent.py`, `corkysoft/mcp/`, `docs/kent_ams_integration.md`, `docs/corkysoft_mcp_v1.md`, related focused tests

## Phase 12: UML Architecture Control Surface

- Replace manual architecture drift with generated PlantUML sources plus rendered review artifacts.
- Keep a supermega view that links superdomains and representative child-module feeds.
- Validate UML coverage against the repo import graph and render pipeline before future architecture changes land.

Current non-blocking worker lanes:

- Worker 1: UML coverage audit lane
  files: `docs/architecture.md`, `docs/UML_INDEX.md`, `docs/rendered/plantuml/`, `scripts/build_supermega_uml.py`
- Worker 2: UML render/index automation lane
  files: `scripts/build_supermega_uml.py`, `docs/UML_INDEX.md`, `docs/rendered/svg/`, related focused tests
- Worker 3: supermega semantic-link quality lane
  files: `scripts/build_supermega_uml.py`, `tests/test_uml_builder.py`, `docs/rendered/plantuml/supermega_01.puml`

## Parallel Dev Lanes For The Decomposition Wave

- Worker 1: auth flow + query-param routing (`dashboard/auth_ui.py`, `dashboard/query_params.py`, `analytics/auth.py`, `dashboard/app.py`, auth/dashboard tests).
- Worker 2: shell selection + cross-surface state hydration (`dashboard/shell.py`, `dashboard/layout_state.py`, `analytics/dashboard_layouts.py`, `dashboard/app.py`, layout/dashboard tests).
- Worker 3: dataset/filter setup + tab composition (`dashboard/data_controls.py`, `dashboard/tab_registry.py`, `dashboard/app.py`, focused dashboard tests).

## Validation

- `py_compile` on changed Python files.
- Kent API and fixture tests in the existing `venv`.
- Streamlit smoke tests if dashboard changes affect startup.
- Re-run focused live UI role walkthroughs only after the three non-blocking lanes land.
