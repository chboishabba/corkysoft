# Architecture

This project blends a routing and costing CLI with a Streamlit dashboard. Core
business logic lives under `corkysoft/`, analytics and data prep live under
`analytics/`, and the UI is under `dashboard/`.

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
- `analytics/`: Data prep, analytics, and telemetry helpers.
  - `analytics/db.py`: Public exports for the analytics database layer.
  - `analytics/db/`: Inventory, fleet, shifts, and schema definitions.
  - `analytics/price_distribution.py`: Aggregations and chart helpers for the dashboard.
  - `analytics/live_data.py`: Live map helpers for trucks and routes.
- `dashboard/`: Streamlit UI and reusable widgets.
  - `dashboard/components/`: Tab renderers and shared UI pieces.
  - `dashboard/map_provider.py`: Mapbox, PyDeck, and Folium configuration.

Note: `corkysoft/src/dashboard/` is a placeholder package stub for packaging.
The main Streamlit entry point remains `dashboard/app.py`.

## Data flow

1. CLI workflows write to `routes.db` via `routes_to_sqlite.py`.
2. Schema creation and migrations are handled by `corkysoft/schema.py` and
   `analytics/db/schema.py`.
3. The dashboard reads from SQLite through `analytics/db_connection.py` and
   uses aggregation helpers in `analytics/price_distribution.py`.
4. Telemetry ingestors populate `truck_positions` and `active_routes`, which
   `analytics/live_data.py` loads for the live map.

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
