"""
Test that the main app starts without error.
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_runs_without_error():
    """
    Tests that the app starts without raising an exception.
    """
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")
    )
    app.run()
    assert not app.exception