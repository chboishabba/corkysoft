"""Smoke tests for the Streamlit dashboard package."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_dashboard_app_module_importable() -> None:
    module = importlib.import_module("dashboard.app")
    assert hasattr(module, "render_price_distribution_dashboard")
    tabs = getattr(module, "PRICE_DASHBOARD_TABS", [])
    assert "Price history" in tabs


def test_streamlit_entrypoint_exposed() -> None:
    module = importlib.import_module("dashboard.app")
    render = getattr(module, "render_price_distribution_dashboard", None)
    assert callable(render)


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
