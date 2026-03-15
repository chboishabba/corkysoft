"""Smoke tests for the Streamlit dashboard package."""
from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_dashboard_app_module_importable() -> None:
    module = importlib.import_module("dashboard.app")
    assert hasattr(module, "render_price_distribution_dashboard")
    tabs = getattr(module, "PRICE_DASHBOARD_TABS", [])
    assert "Price history" in tabs
    assert "Dispatch" in tabs
    assert "Planner" in tabs
    assert "Operations" in tabs
    assert "Calls" in tabs
    assert "Kent tenders" in tabs
    assert "Kent admin" in tabs
    assert "Calls" in tabs
    assert "Payroll / Labor analytics" in tabs


def test_streamlit_entrypoint_exposed() -> None:
    module = importlib.import_module("dashboard.app")
    render = getattr(module, "render_price_distribution_dashboard", None)
    assert callable(render)


def test_planner_tab_is_rendered_in_dashboard_flow() -> None:
    module = importlib.import_module("dashboard.app")
    render = getattr(module, "render_price_distribution_dashboard")
    source = inspect.getsource(render)
    assert 'with tab_map["Planner"]' in source
    assert "render_planner_tab(" in source
    assert 'with tab_map["Calls"]' in source
    assert "render_calls_tab(" in source
    assert 'with tab_map["Payroll / Labor analytics"]' in source
    assert "render_payroll_labor_analytics_tab(conn)" in source
    assert "Repair dispatcher layout" in source

    payroll_source = inspect.getsource(getattr(module, "render_payroll_labor_analytics_tab"))
    assert "Export-ready Labor Summary" in payroll_source
    assert "Absence / Leave" in payroll_source

    staff_source = inspect.getsource(getattr(module, "render_staff_tab"))
    assert "Reviewed worker-time events" in staff_source

    shifts_source = inspect.getsource(getattr(module, "render_driver_shifts_tab"))
    assert "Worker-time events in selected range" in shifts_source
    assert "Imported shifts vs accepted call-derived worker time" in shifts_source
    assert "Mismatch / timing drift" in shifts_source
    assert "_display_worker_time_shift_comparison" in inspect.getsource(module)


def test_worker_time_shift_comparison_flags_assignment_and_timing_mismatches() -> None:
    module = importlib.import_module("dashboard.app")
    build_comparison = getattr(module, "_build_worker_time_shift_comparison")

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

    comparison = build_comparison(
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
