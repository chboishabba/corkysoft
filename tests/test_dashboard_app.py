"""Smoke tests for the Streamlit dashboard package."""
from __future__ import annotations

import importlib


def test_dashboard_app_module_importable() -> None:
    module = importlib.import_module("dashboard.app")
    assert hasattr(module, "render_price_distribution_dashboard")
    tabs = getattr(module, "PRICE_DASHBOARD_TABS", [])
    assert "Price history" in tabs


def test_streamlit_entrypoint_exposed() -> None:
    module = importlib.import_module("dashboard.app")
    render = getattr(module, "render_price_distribution_dashboard", None)
    assert callable(render)
