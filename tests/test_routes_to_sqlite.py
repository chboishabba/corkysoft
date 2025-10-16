import importlib
import sqlite3
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest


def _import_routes_to_sqlite(
    monkeypatch: pytest.MonkeyPatch, *, set_dummy_key: bool = True
):
    """Import the module, optionally configuring a dummy ORS key."""
    if set_dummy_key:
        monkeypatch.setenv("ORS_API_KEY", "dummy-key")
    else:
        monkeypatch.delenv("ORS_API_KEY", raising=False)
    # Ensure a clean import so env vars are read freshly.
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if "openrouteservice" not in sys.modules:
        mock_ors = ModuleType("openrouteservice")

        class _DummyClient:  # pragma: no cover - simple stub to satisfy import
            def __init__(self, *args, **kwargs):
                pass

        mock_ors.Client = _DummyClient
        sys.modules["openrouteservice"] = mock_ors
    sys.modules.pop("routes_to_sqlite", None)
    module = importlib.import_module("routes_to_sqlite")
    monkeypatch.setattr(module, "_ors_client", None, raising=False)
    return module


def test_pelias_geocode_uses_iterable_filters(monkeypatch: pytest.MonkeyPatch):
    module = _import_routes_to_sqlite(monkeypatch)

    mock_client = MagicMock()
    monkeypatch.setattr(module, "get_ors_client", lambda: mock_client)

    mock_client.pelias_search.side_effect = [
        {"features": []},
        {
            "features": [
                {
                    "geometry": {"coordinates": [153.0, -27.0]},
                    "properties": {"label": "Test Address"},
                }
            ]
        },
    ]

    result = module.pelias_geocode("123 Test St", "Australia")

    assert mock_client.pelias_search.call_count == 2

    strict_call = mock_client.pelias_search.call_args_list[0]
    strict_kwargs = strict_call.kwargs

    assert isinstance(strict_kwargs["layers"], (list, tuple))
    assert list(strict_kwargs["layers"]) == module.STRICT_PELIAS_LAYERS

    assert isinstance(strict_kwargs["sources"], (list, tuple))
    assert list(strict_kwargs["sources"]) == module.STRICT_PELIAS_SOURCES

    fallback_kwargs = mock_client.pelias_search.call_args_list[1].kwargs
    assert "layers" not in fallback_kwargs
    assert "sources" not in fallback_kwargs

    assert result.lon == 153.0
    assert result.lat == -27.0
    assert result.label == "Test Address"


def test_ensure_schema_creates_historical_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _import_routes_to_sqlite(monkeypatch)

    db_path = tmp_path / "routes.db"
    conn = sqlite3.connect(db_path)
    module.ensure_schema(conn)

    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }

    conn.close()

    assert "historical_jobs" in tables


def test_cli_add_does_not_require_ors_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    db_path = tmp_path / "cli.db"
    monkeypatch.setenv("ROUTES_DB", str(db_path))
    module = _import_routes_to_sqlite(monkeypatch, set_dummy_key=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["routes_to_sqlite.py", "add", "Brisbane", "Sydney"],
        raising=False,
    )

    module.cli()

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT origin, destination FROM jobs ORDER BY id"
        ).fetchall()
    finally:
        conn.close()

    assert rows == [("Brisbane", "Sydney")]
