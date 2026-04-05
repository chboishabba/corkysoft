"""Smoke tests for the Streamlit dashboard package."""
from __future__ import annotations

import importlib
import inspect
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
from streamlit.testing.v1 import AppTest
from typing import Any

import dashboard.app as dashboard_app
import dashboard.auth_ui as dashboard_auth_ui
from dashboard.tab_registry import build_tab_map
from dashboard.components.worker_time import _build_worker_time_shift_comparison

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("CORKYSOFT_ENV", "development")
os.environ.setdefault("CORKYSOFT_ALLOW_ANONYMOUS_UI", "1")


def test_dashboard_app_module_importable() -> None:
    module = importlib.import_module("dashboard.app")
    assert hasattr(module, "render_price_distribution_dashboard")
    tabs = getattr(module, "PRICE_DASHBOARD_TABS", [])
    assert "Quote" in tabs
    assert "Pricing Intelligence" in tabs
    assert "Network" in tabs
    assert "Operations" in tabs
    assert "Admin" in tabs


def test_streamlit_entrypoint_exposed() -> None:
    module = importlib.import_module("dashboard.app")
    render = getattr(module, "render_price_distribution_dashboard", None)
    assert callable(render)
    assert hasattr(module, "_render_authenticated_user_banner")
    assert hasattr(module, "_render_anonymous_dev_banner")
    assert hasattr(module, "_auth_redirect_config_issue")


def test_planner_tab_is_rendered_in_dashboard_flow() -> None:
    module = importlib.import_module("dashboard.app")
    render = getattr(module, "render_price_distribution_dashboard")
    source = inspect.getsource(render)
    assert "_render_authenticated_user_banner(auth_state)" in source
    assert "_render_anonymous_dev_banner(auth_state)" in source
    assert 'with tab_map["Operations"]' in source
    assert "render_operations_view(" in source
    assert 'with tab_map["Quote"]' in source
    assert "render_quote_view(" in source
    assert 'with tab_map["Admin"]' in source
    assert "render_admin_view(" in source
    assert "Repair dispatcher layout" in source
    assert "find_layout_by_tab(role_layouts, view_param)" in source
    assert '_resolve_dashboard_shell(shell_tab)' in source
    assert '_canonical_role_layout(str(selected_role_layout["roleKey"]))' in source
    assert "show_analytics_overview = requested_tab in _ANALYTICS_SHELL_TABS" in source
    assert "show_overview=show_analytics_overview" in source
    assert 'active_tab_index = st.session_state.get("dashboard_active_tab")' in source
    assert 'requested_tab if "view" in params or requested_tab_from_session else None' in source
    assert 'Analytics filters and pricing controls' in source
    assert 'st.session_state[_LAYOUT_PENDING_KEY] = _layout_defaults_from_layout(selected_role_layout)' in source
    assert 'if "dashboard_active_role" not in st.session_state:' in source

    auth_banner_source = inspect.getsource(getattr(module, "_render_authenticated_user_banner"))
    assert "_render_authenticated_user_sidebar_card(auth_state)" in auth_banner_source
    auth_sidebar_source = inspect.getsource(
        getattr(dashboard_auth_ui, "_render_authenticated_user_sidebar_card")
    )
    assert "Authenticated via Google" in auth_sidebar_source
    assert "Temporary auth mode is active" in auth_sidebar_source
    assert "with st.sidebar" in auth_sidebar_source
    assert '"Settings"' in auth_sidebar_source

    anon_banner_source = inspect.getsource(getattr(module, "_render_anonymous_dev_banner"))
    assert "Anonymous development mode is active" in anon_banner_source

    auth_gate_source = inspect.getsource(getattr(module, "_render_auth_gate"))
    assert "OIDC is not configured correctly" in auth_gate_source

    redirect_issue_source = inspect.getsource(getattr(module, "_auth_redirect_config_issue"))
    assert "OIDC redirect URI does not match the configured public origin" in redirect_issue_source

    payroll_source = inspect.getsource(getattr(module, "render_payroll_labor_analytics_tab"))
    assert "Export-ready Labor Summary" in payroll_source
    assert "Absence / Leave" in payroll_source

    operations_diary_source = inspect.getsource(getattr(module, "render_operations_diary_tab"))
    assert "_set_operations_diary_workspace_params(" in operations_diary_source
    assert 'view="Operations diary"' not in operations_diary_source

    staff_source = inspect.getsource(getattr(module, "render_staff_tab"))
    assert "Reviewed worker-time events" in staff_source

    shifts_source = inspect.getsource(getattr(module, "render_driver_shifts_tab"))
    assert "Worker-time events in selected range" in shifts_source
    assert "Imported shifts vs accepted call-derived worker time" in shifts_source
    assert "Mismatch / timing drift" in shifts_source
    assert "_display_worker_time_shift_comparison" in shifts_source


def test_worker_time_shift_comparison_flags_assignment_and_timing_mismatches() -> None:
    imported = pd.DataFrame(
        [
            {
                "shift_date": pd.Timestamp("2026-03-15").date(),
                "worker_name": "A Worker",
                "truck_id": "T1",
                "linked_job_id": "101",
            },
            {
                "shift_date": pd.Timestamp("2026-03-15").date(),
                "worker_name": "B Worker",
                "truck_id": "T2",
                "linked_job_id": "102",
                "shift_window_start": "08:00",
                "shift_window_end": "12:00",
            },
            {
                "shift_date": pd.Timestamp("2026-03-15").date(),
                "worker_name": "C Worker",
                "truck_id": "T3",
                "linked_job_id": "103",
                "shift_window_start": "08:00",
                "shift_window_end": "10:00",
            },
        ]
    )
    worker_time = pd.DataFrame(
        [
            {
                "effective_date": pd.Timestamp("2026-03-15").date(),
                "workerName": "A Worker",
                "truckId": "T9",
                "jobId": "101",
                "reviewStatus": "accepted",
                "effectiveTimestamp": "2026-03-15T09:15:00+10:00",
            },
            {
                "effective_date": pd.Timestamp("2026-03-15").date(),
                "workerName": "B Worker",
                "truckId": "T2",
                "jobId": "999",
                "reviewStatus": "accepted",
                "effectiveTimestamp": "2026-03-15T09:30:00+10:00",
            },
            {
                "effective_date": pd.Timestamp("2026-03-15").date(),
                "workerName": "C Worker",
                "truckId": "T3",
                "jobId": "103",
                "reviewStatus": "accepted",
                "effectiveTimestamp": "2026-03-15T07:20:00+10:00",
            },
        ]
    )

    comparison = _build_worker_time_shift_comparison(
        imported_shifts=imported,
        worker_time_events=worker_time,
    )

    assert set(comparison["Status"]) == {
        "truck_mismatch",
        "job_mismatch",
        "time_mismatch",
    }
    truck_row = comparison[comparison["Status"] == "truck_mismatch"].iloc[0]
    assert truck_row["Imported truck"] == "T1"
    assert truck_row["Call truck"] == "T9"
    assert truck_row["Imported job"] == "101"
    assert truck_row["Call job"] == "101"

    job_row = comparison[comparison["Status"] == "job_mismatch"].iloc[0]
    assert job_row["Imported truck"] == "T2"
    assert job_row["Call truck"] == "T2"
    assert job_row["Imported job"] == "102"
    assert job_row["Call job"] == "999"

    time_row = comparison[comparison["Status"] == "time_mismatch"].iloc[0]
    assert time_row["Imported truck"] == "T3"
    assert time_row["Call truck"] == "T3"
    assert time_row["Imported job"] == "103"
    assert time_row["Call job"] == "103"
    assert time_row["Imported window"] == "08:00 - 10:00"
    assert "07:20:00" in time_row["Call time"]


def test_streamlit_app_runs_cleanly(capsys) -> None:
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
    )
    try:
        app.run(timeout=30)
        captured = capsys.readouterr()
        log = (captured.out + captured.err).strip()
        if not log:
            log = "<no output captured>"
        assert not app.exception, (
            "Streamlit app raised an exception during execution:\n"
            + (log if log else repr(app.exception))
        )
        assert "Traceback" not in log, "Streamlit app emitted traceback output:\n" + log
        assert (
            "StreamlitDuplicateElementId" not in log
        ), "Streamlit app encountered duplicate element IDs:\n" + log
    finally:
        if hasattr(app, "_tree") and getattr(app._tree, "_runner", None) is not None:
            app._tree._runner = None


def _fake_layout(label: str) -> dict[str, Any]:
    return {
        "label": label,
        "roleKey": label.lower().replace(" ", "_"),
        "primaryTabs": ["Quote"],
        "hiddenTabs": ["Admin"],
        "defaultLandingTab": "Quote",
    }


def test_hydrate_role_layout_session_applies_pending_choice() -> None:
    dashboard_app.st.session_state.clear()
    layout = _fake_layout("Estimator")
    pending = {
        "primaryTabs": ["Operations"],
        "hiddenTabs": ["Pricing Intelligence"],
        "landingTab": "Operations",
        "showAll": True,
    }
    dashboard_app.st.session_state[dashboard_app._LAYOUT_PENDING_KEY] = pending
    dashboard_app._hydrate_role_layout_session(layout)
    assert dashboard_app._LAYOUT_PENDING_KEY not in dashboard_app.st.session_state
    assert dashboard_app.st.session_state["dashboard_session_primary_tabs"] == ["Operations"]
    assert dashboard_app.st.session_state["dashboard_session_hidden_tabs"] == ["Pricing Intelligence"]
    assert dashboard_app.st.session_state["dashboard_session_landing_tab"] == "Operations"
    assert dashboard_app.st.session_state["dashboard_show_all_tabs"]
    assert dashboard_app.st.session_state["dashboard_active_role_last"] == "Estimator"


def test_hydrate_role_layout_session_resets_on_role_change() -> None:
    dashboard_app.st.session_state.clear()
    layout_a = _fake_layout("Estimator")
    layout_b = _fake_layout("Dispatcher")
    dashboard_app._hydrate_role_layout_session(layout_a)
    assert dashboard_app.st.session_state["dashboard_session_primary_tabs"] == ["Quote"]
    dashboard_app._hydrate_role_layout_session(layout_b)
    assert dashboard_app.st.session_state["dashboard_session_primary_tabs"] == ["Quote"]
    assert dashboard_app.st.session_state["dashboard_active_role_last"] == "Dispatcher"


def test_hydrate_role_layout_session_force_resets_same_role() -> None:
    dashboard_app.st.session_state.clear()
    layout = _fake_layout("Dispatcher")
    dashboard_app.st.session_state["dashboard_session_primary_tabs"] = ["Operations"]
    dashboard_app.st.session_state["dashboard_session_hidden_tabs"] = []
    dashboard_app.st.session_state["dashboard_session_landing_tab"] = "Operations"
    dashboard_app.st.session_state["dashboard_show_all_tabs"] = True
    dashboard_app.st.session_state["dashboard_active_role_last"] = "Dispatcher"

    dashboard_app._hydrate_role_layout_session(layout, force_reset=True)

    assert dashboard_app.st.session_state["dashboard_session_primary_tabs"] == ["Quote"]
    assert dashboard_app.st.session_state["dashboard_session_hidden_tabs"] == ["Admin"]
    assert dashboard_app.st.session_state["dashboard_session_landing_tab"] == "Quote"
    assert dashboard_app.st.session_state["dashboard_show_all_tabs"] is False


def test_anonymous_view_layout_primes_active_role_for_deep_link() -> None:
    dashboard_app.st.session_state.clear()
    dashboard_app.st.session_state["dashboard_active_role"] = "Estimator"
    role_label = "Dispatcher"

    if dashboard_app.st.session_state.get("dashboard_active_role") != role_label:
        dashboard_app.st.session_state["dashboard_active_role"] = role_label

    assert dashboard_app.st.session_state["dashboard_active_role"] == "Dispatcher"


def test_resolve_dashboard_shell_distinguishes_operator_and_analytics_tabs() -> None:
    analytics_shell = dashboard_app._resolve_dashboard_shell("Pricing Intelligence")
    assert analytics_shell["title"] == "Pricing Intelligence"
    assert analytics_shell["sidebar_heading"] == "Filters"
    assert not analytics_shell["collapse_analytics_sidebar"]

    operator_shell = dashboard_app._resolve_dashboard_shell("Operations")
    assert operator_shell["title"] == "Operations & Network Control"
    assert operator_shell["sidebar_heading"] == "Workflow support"
    assert operator_shell["collapse_analytics_sidebar"]

    commercial_shell = dashboard_app._resolve_dashboard_shell("Quote")
    assert commercial_shell["title"] == "Quote Workspace"
    assert commercial_shell["collapse_analytics_sidebar"]


def test_canonical_role_layout_uses_role_defaults() -> None:
    dispatcher_layout = dashboard_app._canonical_role_layout("dispatcher")
    assert dispatcher_layout["defaultLandingTab"] == "Operations"
    assert "Network" in dispatcher_layout["primaryTabs"]
    assert "Admin" in dispatcher_layout["hiddenTabs"]


def test_main_sets_static_corkysoft_page_title() -> None:
    source = inspect.getsource(dashboard_app.main)
    assert 'page_title="Corkysoft"' in source


def test_distribution_overview_has_optional_non_tab_summary_gate() -> None:
    source = inspect.getsource(dashboard_app.render_distribution_analytics_surface)
    assert "show_overview: bool = True" in source
    assert "if show_overview and has_filtered_data:" in source


def test_build_tab_map_keeps_order_stable_without_keyed_tabs(monkeypatch) -> None:
    monkeypatch.setattr(
        "dashboard.tab_registry.inspect.signature",
        lambda _: (_ for _ in ()).throw(ValueError("signature unavailable")),
    )
    fake_tabs = [MagicMock(), MagicMock(), MagicMock()]
    tabs_placeholder = MagicMock()
    tabs_placeholder.__enter__ = MagicMock(return_value=tabs_placeholder)
    tabs_placeholder.__exit__ = MagicMock(return_value=None)
    monkeypatch.setattr("dashboard.tab_registry.st.tabs", lambda labels: fake_tabs)

    result = build_tab_map(
        tab_labels=["Quote", "Operations", "Admin"],
        requested_tab="Operations",
        params={"view": ["Operations"]},
        tabs_placeholder=tabs_placeholder,
    )

    assert result.tab_order == ["Quote", "Operations", "Admin"]
    assert list(result.tab_map.keys()) == ["Quote", "Operations", "Admin"]
