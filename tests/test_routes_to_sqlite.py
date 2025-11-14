import importlib
import json
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
    monkeypatch.setenv("ROUTING_PROVIDER", "ors")
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
        mock_ors.exceptions = ModuleType("openrouteservice.exceptions")
        sys.modules["openrouteservice"] = mock_ors
        sys.modules["openrouteservice.exceptions"] = mock_ors.exceptions
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


def test_cli_run_with_google_provider(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    db_path = tmp_path / "cli-google.db"
    monkeypatch.setenv("ROUTES_DB", str(db_path))

    module = _import_routes_to_sqlite(monkeypatch, set_dummy_key=False)
    from analytics.routing_provider import RouteGeometryResult
    from corkysoft.au_address import GeocodeResult

    stub_provider_calls: list[tuple[tuple[float, float], tuple[float, float], str]] = []

    def encode_polyline(points: list[tuple[float, float]]) -> str:
        result_chars: list[str] = []
        prev_lat = 0
        prev_lng = 0
        for lat, lng in points:
            lat_e5 = int(round(lat * 1e5))
            lng_e5 = int(round(lng * 1e5))
            for value in (lat_e5 - prev_lat, lng_e5 - prev_lng):
                shifted = value << 1
                if value < 0:
                    shifted = ~shifted
                while shifted >= 0x20:
                    result_chars.append(chr((0x20 | (shifted & 0x1F)) + 63))
                    shifted >>= 5
                result_chars.append(chr(shifted + 63))
            prev_lat, prev_lng = lat_e5, lng_e5
        return "".join(result_chars)

    encoded_polyline = encode_polyline([(-27.0, 153.0), (-33.0, 151.0)])

    class StubProvider:
        def route_geometry(self, *, origin, destination, profile):
            stub_provider_calls.append((origin, destination, profile))
            return RouteGeometryResult(
                distance_km=123.4,
                duration_hr=2.5,
                encoded_polyline=encoded_polyline,
            )

    stub_provider = StubProvider()

    monkeypatch.setattr(
        module,
        "create_routing_provider",
        lambda provider_name=None, *, client=None, provider=None: stub_provider,
    )

    def fake_geocode(_conn, place: str, _country: str) -> GeocodeResult:
        if place == "Origin":
            return GeocodeResult(
                lon=153.0,
                lat=-27.0,
                label="Origin Label",
                normalization=None,
                search_candidates=[place],
            )
        return GeocodeResult(
            lon=151.0,
            lat=-33.0,
            label="Destination Label",
            normalization=None,
            search_candidates=[place],
        )

    monkeypatch.setattr(module, "geocode_cached", fake_geocode)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "routes_to_sqlite.py",
            "--routing-provider",
            "google",
            "add",
            "Origin",
            "Destination",
        ],
        raising=False,
    )
    module.cli()

    monkeypatch.setattr(
        sys,
        "argv",
        ["routes_to_sqlite.py", "--routing-provider", "google", "run"],
        raising=False,
    )
    module.cli()

    conn = sqlite3.connect(db_path)
    try:
        provider, distance_km, duration_hr, route_geojson = conn.execute(
            """
            SELECT provider, distance_km, duration_hr, route_geojson
            FROM jobs
            WHERE origin = ? AND destination = ?
            """,
            ("Origin", "Destination"),
        ).fetchone()
    finally:
        conn.close()

    assert provider == "google"
    assert distance_km == pytest.approx(123.4)
    assert duration_hr == pytest.approx(2.5)
    geojson = json.loads(route_geojson)
    assert geojson["type"] == "FeatureCollection"
    assert stub_provider_calls
