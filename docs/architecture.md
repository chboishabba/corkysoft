# Architecture

This project blends a routing and costing CLI with a Streamlit dashboard. Core
business logic lives under `corkysoft/`, analytics and data prep live under
`analytics/`, and the UI is under `dashboard/`.

Quality-control framing for the current dashboard refactor:

- Service management objective:
  keep the dashboard shell operable and reviewable as role-aware workflows expand.
- ISO 9001 style control objective:
  make module ownership explicit so contributors can change one control layer without destabilizing unrelated surfaces.
- Six Sigma style defect focus:
  reduce variation caused by one large composition file owning auth, query params, shell copy, layout hydration, dataset controls, and tab order together.

## C4 and PlantUML

The generated UML suite is documented in:

- [architecture_dashboard_shell.puml](architecture_dashboard_shell.puml)
- [UML_INDEX.md](UML_INDEX.md)

The UML sources are generated from the internal Python import graph by:

- `python scripts/build_supermega_uml.py`

The integrated whole-system entrypoint is:

- [rendered/plantuml/supermega_01.puml](rendered/plantuml/supermega_01.puml)

Current dashboard-shell control view:

- [rendered/plantuml/dashboard_shell.puml](rendered/plantuml/dashboard_shell.puml)
- [rendered/svg/dashboard_shell.svg](rendered/svg/dashboard_shell.svg)

## Control lanes

Treat the current remediation wave as four bounded control lanes rather than
one undifferentiated refactor:

- Lane 1, service-governance/docs: README, roadmap, onboarding, role matrix,
  and operator stories must describe the same five-view shell and role entry
  model that the code renders.
- Lane 2, provider parity: routing, isochrones, route enrichment, and map
  rendering must honor the selected provider without silent cross-provider
  fallback.
- Lane 3, shell composition: layout reset, deep-link landing, shared shell
  primitives, and generated architecture views must remain deterministic as the
  shell grows.
- Lane 4, security/privacy/AI-risk: admin actions, role-hidden surfaces,
  person/labor data exposure, and advisory/model-backed outputs must remain
  bounded, auditable, and reviewable.

Execution control:

- use the repo virtualenv as the execution boundary for Python, Streamlit, and
  pytest commands
- treat `Quote`, `Pricing Intelligence`, `Network`, `Operations`, and `Admin`
  as the only top-level shell boundaries in architecture and contributor docs;
  `Planner`, `Dispatch`, `Operations diary`, `Kent tenders`, and `Kent admin`
  remain child workflows inside those owning shells
- treat `view=` as navigation only; support-grade workspace reconstruction
  should move through a normalized workspace-state contract rather than raw,
  ever-growing query-param sprawl
- update PlantUML/C4 artifacts when the shell composition materially changes
- treat placeholder KPI/alert content as scaffolding until provenance,
  freshness, and ownership are explicit

## Next advancement focus

The next architecture-relevant advances should be:

- sourced shell signals:
  move KPI/alert rendering behind shared producer contracts that carry source,
  owner, freshness, and fallback semantics
- state-addressable workspace support:
  distinguish simple navigation params from reproducible workspace-state
  snapshots so support and audit flows can reopen the same shell/workflow state
- customer-facing tracking and receipt surfaces:
  treat customer status pages and printable receipts as a separate boundary from
  both the internal operator shell and internal support replay; reuse live
  telemetry/job-state producers, but do not expose internal shell state or
  workflow-only fields directly
- metasystem view discipline:
  keep [rendered/plantuml/supermega_01.puml](rendered/plantuml/supermega_01.puml)
  as the whole-system entrypoint and the existing child diagrams as the only
  reviewed drill-down surfaces
- operational data contracts:
  keep decision-adjacent shell data explicitly classified as advisory,
  review-backed, or decision-grade

## Entry points

- `routes_to_sqlite.py`: Primary CLI for geocoding, routing, and persisting jobs.
- `dashboard/app.py`: Streamlit dashboard entry point.
- `analytics/ingest_live_data.py`: Mock telemetry streamer for map demos.
- `analytics/ingest_real_gps.py`: Load real telemetry snapshots from files or stdin.
- `map_jobs.py`: Render HTML route maps from stored routes.
- `quick_quote.py`: Lightweight quote calculator script.
- `profit_optimizer.py`: Simplex-based profitability optimizer.

## Core modules

- `corkysoft/`: Routing, pricing, and quoting primitives.
  - `corkysoft/routing/providers.py`: Provider-specific clients and error handling.
  - `corkysoft/routing/__init__.py`: Provider selection, caching, and compatibility shims.
  - `corkysoft/pricing.py`: Pricing modifiers and defaults.
  - `corkysoft/quote_service.py`: Quote calculation and formatting helpers.
  - `corkysoft/repo.py`: Quote persistence and client matching.
  - `corkysoft/schema.py`: Core schema bootstrap for CLI workflows.
  - `corkysoft/mcp/`: Read-only-first MCP adapter registry, bridge, and optional transport server.
- `analytics/`: Data prep, analytics, and telemetry helpers.
  - `analytics/db.py`: Public exports for the analytics database layer.
  - `analytics/db/`: Inventory, fleet, shifts, and schema definitions.
  - `analytics/adaptive_policy.py`: Adaptive policy parameter defaults, snapshots, and bounded update helpers.
  - `analytics/operations_diary.py`: Day/week diary rollups, diary-task CRUD,
    and invoice/bill review helpers.
  - `analytics/price_distribution.py`: Aggregations and chart helpers for the dashboard.
  - `analytics/live_data.py`: Live map helpers for trucks and routes.
- `dashboard/`: Streamlit UI and reusable widgets.
  - `dashboard/app.py`: Composition layer for auth, shell selection, sidebar controls, and tab rendering.
  - `dashboard/auth_ui.py`: Auth gate, identity resolution, and local dashboard-user admin controls.
  - `dashboard/query_params.py`: Shared query-param compatibility helpers used across the shell and workflow surfaces.
  - `dashboard/layout_state.py`: Role-layout hydration and pending-reset handling.
  - `dashboard/shell.py`: Role-aware title, caption, and sidebar framing selection.
  - `dashboard/data_controls.py`: Dataset, provider, ingest, filter, and break-even sidebar controls.
  - `dashboard/tab_registry.py`: Landing-tab and visible-tab composition.
  - `dashboard/components/`: Tab renderers and shared UI pieces.
  - `dashboard/views/`: Higher-level role/workflow view composites that group component surfaces.
  - `dashboard/components/operations_diary.py`: Manager-facing diary screen for
    job usage, tasks, and invoice/bill follow-through.
  - `dashboard/map_provider.py`: Mapbox, PyDeck, and Folium configuration.

Note: `corkysoft/src/dashboard/` is a placeholder package stub for packaging.
The main Streamlit entry point remains `dashboard/app.py`.

## Data flow

1. CLI workflows write to `routes.db` via `routes_to_sqlite.py`.
2. Schema creation and migrations are handled by `corkysoft/schema.py` and
   `analytics/db/schema.py`.
3. The dashboard composition layer resolves auth, query params, role layout,
   shell chrome, dataset controls, and tab order before handing work to leaf
   surfaces.
4. The dashboard reads from SQLite through `analytics/db_connection.py` and
   uses aggregation helpers in `analytics/price_distribution.py`.
5. Telemetry ingestors populate `truck_positions` and `active_routes`, which
   `analytics/live_data.py` loads for the live map.
6. Adaptive-policy helpers store small learned pricing/ETA/risk parameters in
   `global_parameters` so later ingestion and review workflows can update them
   without hidden spreadsheet drift.
7. Operations diary helpers combine jobs, segments, tasks, labor actuals, and
   invoice/bill review state into a manager-facing day/week workflow.
8. Any later customer-tracking or printable-receipt surface should consume a
   dedicated public-safe contract over reviewed job status, telemetry progress,
   ETA, and receipt evidence rather than reopening internal dashboard state.

## Cross-project boundary

- Corkysoft is the workflow and operational-truth owner for removals planning,
  diary tasks, invoice review, and subcontractor-bill reconciliation.
- StatiBaker should consume append-only downstream summaries and reviewed-state
  envelopes where useful, but should not become a second mutable workflow
  database for these operations.
- ITIR remains the orchestration/context layer that coordinates cross-project
  contracts without taking ownership of Corkysoft business semantics.
- Corkysoft's MCP posture should follow the same producer-ownership rule:
  expose bounded read-only adapters over existing Corkysoft logic rather than
  making MCP a second workflow owner.
- Diary/planner/reconciliation export now exists as a native observer-outbox
  surface, while a broader Corkysoft MCP adapter now has an implemented local
  registry plus JSON bridge for bounded read-only tools.

## Routing providers and configuration

Routing providers are selected via `ROUTING_PROVIDER`:

- `ors`: OpenRouteService via `ORS_API_KEY`
- `google`: Google Maps via `GOOGLE_MAPS_API_KEY`

Provider switching occurs in `corkysoft/routing/__init__.py` and delegates to
`corkysoft/routing/providers.py` for client setup and API calls.

## Tests

Pytest suites live under `tests/` and mirror analytics, routing, and dashboard
helpers. Use targeted runs such as `pytest tests/test_price_distribution.py`
when iterating on a specific area.
