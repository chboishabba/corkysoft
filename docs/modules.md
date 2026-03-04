# Module Guide

This guide summarizes the responsibilities and key entry points for each
major module in the repository.

## corkysoft

Core routing, pricing, and quote services.

- Routing providers and adapters: `corkysoft/routing/`
- Pricing modifiers: `corkysoft/pricing.py`
- Quote calculation and formatting: `corkysoft/quote_service.py`
- Quote persistence and client matching: `corkysoft/repo.py`
- Core schema bootstrap: `corkysoft/schema.py`

## analytics

Analytics helpers, data preparation, and telemetry processing.

- Public DB helpers: `analytics/db.py`
- Inventory, fleet, shifts, schema: `analytics/db/`
- Price distribution and chart helpers: `analytics/price_distribution.py`
- Live data loaders: `analytics/live_data.py`
- Telemetry ingestion: `analytics/ingest_live_data.py`, `analytics/ingest_real_gps.py`
- Route map aggregation: `analytics/routes_map.py`

## dashboard

Streamlit UI entry point and reusable components.

- Main Streamlit app: `dashboard/app.py`
- UI tabs and widgets: `dashboard/components/`
- Map provider configuration: `dashboard/map_provider.py`

Note: `corkysoft/src/dashboard/` is a packaging stub and does not host the
Streamlit entry point.

## tests

Pytest suites mirroring analytics, routing, and dashboard helpers.

- Core tests live under `tests/`
- Database-focused tests live under `tests/db/`

## e2e

Playwright smoke tests and visual snapshot checks for the dashboard UI.

- Specs live under `e2e/`

## Top-level scripts

Entry points and utilities intended to be run directly.

- `routes_to_sqlite.py`: CLI for routing and job persistence
- `map_jobs.py`: HTML map renderer
- `quick_quote.py`: lightweight quote calculator
- `profit_optimizer.py`: simplex-based optimizer
- `start_app.sh` / `start_app.bat`: convenience launchers
