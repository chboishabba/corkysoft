# Repository Guidelines

## Project Structure & Module Organization
- Core application code lives under `dashboard/` (`app.py` as the Streamlit entry point, `components/` for reusable widgets) and `analytics/` for data prep, pricing insights, and live telemetry processing.
- Persistence and schema helpers reside in `analytics/db.py`, while command-line utilities such as `routes_to_sqlite.py` sit at the repo root.
- Tests are collected in `tests/`, mirroring feature areas (`test_price_distribution.py`, `test_live_data.py`, etc.); keep new tests alongside their target modules.
- SQLite databases (`routes.db`, numbered snapshots) stay in the project root—avoid committing new binaries.

## Build, Test, and Development Commands
- `python3 -m venv venv && source venv/bin/activate` bootstraps the virtualenv used by the project.
- `pip install -r requirements.txt` installs runtime and dev dependencies.
- `streamlit run dashboard/app.py` launches the dashboard UI; use `streamlit_price_distribution.py` for legacy single-file demos.
- `pytest` runs the full automated test suite; prefer `pytest tests/<module>` for targeted debugging.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation and descriptive, lowercase_with_underscores names for modules, functions, and variables.
- Keep Streamlit UI logic inside `dashboard/`; business logic belongs in pure-Python modules under `analytics/`, `services/`, or `utils/`.
- Use type hints throughout new code and prefer `dataclasses` or Pydantic models in `models/` when introducing structured payloads.
- Stick to ASCII characters unless extending existing non-ASCII text (e.g., currency symbols already present).

## Testing Guidelines
- Tests use `pytest`; name files `test_<area>.py` and ensure each new feature or bugfix includes coverage.
- Reproduce existing patterns: fixture-heavy integration tests in `tests/test_price_distribution.py`, telemetry harness checks in `tests/test_live_data.py`.
- Always run `pytest` (or the focused subset) before opening a PR; ensure new tests pass without relying on network calls.

## Commit & Pull Request Guidelines
- Write imperative commit messages (e.g., “Add symmetric colour scaling to map”) and keep them scoped to one logical change.
- Pull requests should include: a concise summary, references to ROADMAP or issues touched, screenshots/GIFs for UI tweaks, and confirmation of `pytest` results.
- Update `README.md` and `ROADMAP.md` when behavior, setup steps, or milestone status changes.

## Security & Configuration Tips
- Keep `.env` secrets out of version control; use `.env.example` as the template for required keys like `ORS_API_KEY`.
- When handling database migrations, update `db/migrations.sql` and document manual steps in the PR description.
