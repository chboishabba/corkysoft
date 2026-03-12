# Devlog

## 2026-03-12

- Initialized canonical planning files for the docs/governance/Kent remediation milestone.
- Scope locked to three phases: truth alignment, product/governance completion, and Kent code/UX hardening.
- Existing durable decision log remains in `COMPACTIFIED_CONTEXT.md`.
- Aligned README, ROADMAP, positioning, Kent docs, Live Network docs, and ingest contracts to the implemented current state.
- Added actor-based user stories and a quote -> tender -> awarded-work lifecycle doc.
- Hardened Kent runtime behavior: internal mutating API auth, governed hard-block categories, side-effect-free `dry_run`, true top-N prioritization, and operator/admin dashboard separation.
- Validated with targeted local `venv` tests: `tests/test_dashboard_app.py`, `tests/test_api.py`, and `tests/test_kent_ams_fixtures.py`.
- Shifted the spreadsheet integration plan toward a Google Sheets-first operations model where spreadsheets remain import-only and Corkysoft holds the internal planning truth.
- Added segment-based operations planning/readiness services and API routes for segment creation, resource assignment, conflict visibility, policy reads/updates, and shared-workbook sync.
- Added an `Operations` dashboard tab for segment planning and kept spreadsheet sync/policy controls on the fleet/admin side rather than the core assignment flow.
- Added import provenance on fleet/staff/supplier records plus readiness state on job segments so spreadsheet freshness and assignment trust can be surfaced together.
- Fixed `ensure_segment` so sequence-1 backfilled segments are actually updated when operators set planned windows; this restores conflict detection for overlapping assignments.
- Normalized vehicle workbook provenance timestamps to UTC and validated the spreadsheet/operations path with local `venv` tests: `tests/test_operations_assignment.py`, `tests/test_google_sheets_imports.py`, `tests/test_vehicle_workbook.py`, `tests/test_dashboard_app.py`, `tests/test_api.py`, `tests/test_kent_ams_fixtures.py`, and `tests/test_operations_workbook.py`.
- Threaded segment-based planning into existing Staff and Fleet views so imported sheet context no longer masquerades as assignment truth.
- Added worker/truck assignment summary helpers and segment detail views for staff/truck review screens.
- Added a workflow-based spreadsheet replacement roadmap covering assignment, compliance/maintenance, native driver planning, inventory coordination, and final sheet decommissioning.
