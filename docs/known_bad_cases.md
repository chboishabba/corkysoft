# Known Bugs And Bad Cases

Last updated: 2026-07-18.

This register is the canonical audit surface for confirmed bugs, bad cases,
accepted risks, and governance gaps. Use it with [ROADMAP.md](../ROADMAP.md)
and [progress_status_board.md](progress_status_board.md). Items stay here until
the linked promotion gate has code/docs/tests evidence.

Severity key:

- `P0`: security, privacy, or authority failure that can expose sensitive data
  or permit unsafe action.
- `P1`: operator-facing crash, materially misleading decision signal, broken
  import path, or CI/control failure.
- `P2`: important workflow friction, stale behavior, or test/documentation gap.

## Open Register

| ID | Severity | Area | Case | Evidence | Owner lane | Promotion gate |
| --- | --- | --- | --- | --- | --- | --- |
| BAD-002 | P0 | REST/API security | Corrected for current high-authority REST writes. Scoped credentials bind actors and receipts; credential lifecycle now fails closed for not-yet-active, expired, and revoked credentials; the legacy shared write token is disabled unless explicitly enabled for a bounded migration window. | `corkysoft/api_shared.py`, `docs/api_security_authority.md`, `tests/test_api.py`. | Security/API governance | Keep a service-credential inventory current as new write families are added; do not re-enable legacy writes by default. |
| BAD-003 | P0 | Transcript/audio boundary | Transcript/audio payloads and adapter errors are bounded, and transcript artifacts now carry advisory classification plus failed-artifact metadata. Remaining risk is promotion governance: raw transcript/model output must not become operational authority without reviewed actor acceptance. | `corkysoft/api_calls.py` validates base64/audio uploads; `corkysoft/whisperx_adapter.py` normalizes invalid JSON and HTTP failures; `corkysoft/call_ops_transcripts.py` persists `data_classification`, `authority_class`, sanitized failed-task errors, and `failure_kind`. | Security/API governance | Add explicit reviewed-promotion path tied to BAD-002 scoped credentials; tests prove raw transcript output cannot authorize notes, worker-time, or customer-visible projections. |
| BAD-004 | P1 | Dashboard authz | Authenticated users can reveal role-hidden top-level tabs through the session-level show-all control. | `dashboard/app.py`, `analytics/dashboard_layouts.py`, `docs/authentication_and_users.md`. | Dashboard shell/security | Admin-gate or remove show-all for authenticated restricted roles; behavioral tests prove hidden-tab denial. |
| BAD-005 | P1 | Dashboard operations | Dispatch can crash after filters remove all rows because selection code indexes an empty options map. | `dashboard/components/dispatch.py`; implementation is in PR [#226](https://github.com/chboishabba/corkysoft/pull/226), which guards the empty post-filter result before selection. | Dashboard shell | Merge PR #226 and add/retain behavioral coverage with zero filtered rows. |
| BAD-006 | P1 | Analytics finance semantics | `margin_total_pct` and `margin_per_m3_pct` use cost as denominator while UI copy can imply gross margin. | `analytics/job_loading.py`. | Analytics/model correctness | Define and test canonical `gross_margin_pct` versus `cost_roi_pct`; rename or label existing fields. |
| BAD-007 | P1 | Analytics history | Price-history current/previous windows can include prior-year data when no explicit date window is supplied. | `analytics/price_history_analysis.py`, partial coverage in `tests/test_price_distribution.py`. | Analytics/model correctness | Explicit no-date multi-year tests and deterministic current-period derivation. |
| BAD-008 | P1 | Historical ingest | Legacy cost inference can classify sell/price columns as final cost, corrupting profitability. | `analytics/historical_ingest.py`. | Analytics/imports | Import-schema confidence checks for cost/revenue collisions and bad-row issue recording. |
| BAD-009 | P1 | MoveWare imports | Worker import expects `workers.employee_code` and incompatible `upsert_worker` args on the current schema. | `analytics/moveware_import.py`, `analytics/db/schema.py`, `analytics/db/fleet.py`. | Persistence/imports | Non-dry-run importer tests and schema alignment for worker natural keys. |
| BAD-010 | P1 | MoveWare imports | Bookings/containers/allocation imports depend on tables only created by legacy bootstrap, not current dashboard schema. | `analytics/moveware_import.py`, `analytics/db/schema.py`, `analytics/db/legacy.py`. | Persistence/imports | Canonical migration path includes those resources or importer rejects unsupported resources explicitly. |
| BAD-011 | P1 | CLI import | `routes_to_sqlite.py import-history` writes `client_name` into a table defined with `client`, then continues after row errors. | `routes_to_sqlite.py`. | Persistence/imports | Focused CLI import regression with persisted row issues and non-silent failure semantics. |
| BAD-012 | P1 | CI/dev workflow | GitHub Playwright workflow starts Streamlit manually with system tooling while Playwright config defines a managed `venv/bin/python` server; CI also omits pytest. | `.github/workflows/playwright.yml`, `playwright.config.ts`, `requirements.txt`. | Tests/CI | CI matrix runs repo-venv pytest, root Playwright smoke, and separates screenshot artifacts. |
| BAD-013 | P1 | Startup/security | Startup scripts mutate state with `git pull`/install and public tunnel scripts do not enforce the auth/public-origin posture. | `start_app.sh`, `start_app.bat`, `start_app_public.sh`, `localhost_run_insecure_remote.sh`. | Tests/CI/security | Scripts use repo venv, avoid implicit pulls, and separate local anonymous/auth/public tunnel modes. |
| BAD-014 | P2 | Workspace state | `ws` support influences shell copy/role inference but top-level tab landing can still revert to raw `view` or defaults. | `dashboard/app.py`, `dashboard/workspace_state.py`. | Dashboard shell | `ws`-only link tests for shell and child workflow landing. |
| BAD-015 | P2 | UI action semantics | Several prominent top-level buttons render as real actions without handlers, disabled state, or explicit scaffold notices. | `dashboard/views/quote_view.py`, `pricing_intelligence_view.py`, `network_view.py`, `operations_view.py`, `admin_view.py`. | Dashboard shell | Either wire handlers or render disabled/scaffold controls with tests. |
| BAD-017 | P2 | Schema governance | DDL is split across `routes_to_sqlite.py`, `analytics/db/schema.py`, and `analytics/db/legacy.py`, with drifting columns and constraints. | `routes_to_sqlite.py`, `analytics/db/schema.py`, `analytics/db/legacy.py`. | Persistence/imports | Versioned canonical migration module plus upgrade tests from old DB shapes. |
| BAD-018 | P2 | SQLite migrations | Some `ALTER TABLE ... ADD COLUMN ... NOT NULL` migrations lack defaults or rebuild paths for non-empty old tables. | `analytics/db/schema.py`. | Persistence/imports | Migration tests from non-empty legacy snapshots. |
| BAD-019 | P2 | Test quality | Some UI governance tests inspect source strings instead of exercising rendered behavior. | `tests/test_dashboard_app.py`, `tests/test_dashboard_view_governance.py`. | Tests/CI | Replace with behavioral AppTest/Playwright/unit checks for auth/view governance. |
| BAD-020 | P2 | Fixture/workbook governance | `Crusader.xlsx` is a tracked binary placeholder and tests depend on it directly, coupling draft fallback data to test truth. | `Crusader.xlsx`, `AGENTS.md`, `README.md`, `tests/test_inventory_and_shipments.py`. | Persistence/imports | Move to governed fixture path or generate fixture data; surface workbook provenance/freshness. |

## zkSEC-Informed Security Gates

The security audit used `../zkSEC` as supporting context, not as a Corkysoft
runtime dependency. Corkysoft security work should inherit these gates:

- Public or uncertain signals are proposal-only and cannot authorize writes.
- High-authority actions require verified actor identity, authorized scope,
  explicit plan/receipt metadata, and confirmation where risk is high.
- Adapter/resource access must stay inside declared roots and reject obvious
  secret-like payloads.
- No silent authority crossing: advisory/model/transcript outputs require
  human review before they become operational authority.

## Recently Corrected Or Historical Cases

| Case | Status | Evidence |
| --- | --- | --- |
| Google-selected route/map flows could silently fall back to ORS/OSM in some paths. | Corrected, keep regression coverage. | `ROADMAP.md`, `docs/progress_status_board.md`, provider tests. |
| Legacy `st.experimental_rerun` crash evidence in operator flows. | Historical, continue sweep for direct rerun calls. | `CHANGELOG.md`, `docs/naive_user_tester_notes.md`. |
| Top-level shell drift between code and docs after the five-view revision. | Partially corrected; deeper docs still need drift sweeps. | `README.md`, `ROADMAP.md`, `docs/progress_status_board.md`. |
| Sensitive internal REST reads were unauthenticated. | Corrected for current API routes; scoped credential granularity remains under BAD-002. | `corkysoft/api.py`, router-level dependencies in `corkysoft/api_calls.py`, `corkysoft/api_labor.py`, `corkysoft/api_operations.py`, `corkysoft/api_kent.py`, `tests/test_api.py`. |
| MCP tools accepted caller-supplied DB paths without root scoping. | Corrected for v1 read-only tools. | `corkysoft/mcp/tools.py`, `tests/test_corkysoft_mcp.py`, `docs/corkysoft_mcp_v1.md`. |
