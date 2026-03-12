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

## Validation

- `py_compile` on changed Python files.
- Kent API and fixture tests in the existing `venv`.
- Streamlit smoke tests if dashboard changes affect startup.
