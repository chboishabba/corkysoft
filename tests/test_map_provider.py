import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard import map_provider


@pytest.fixture(autouse=True)
def reset_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROUTING_PROVIDER", raising=False)
    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)


def test_pydeck_map_kwargs_uses_google_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTING_PROVIDER", "google")
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "abc123")

    kwargs = map_provider.pydeck_map_kwargs("mapbox-style")

    assert kwargs["map_provider"] == "google_maps"
    assert kwargs["api_keys"]["google_maps"] == "abc123"
    assert kwargs["map_style"] is None


def test_pydeck_map_kwargs_falls_back_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTING_PROVIDER", "google")

    kwargs = map_provider.pydeck_map_kwargs("mapbox-style")

    assert kwargs == {"map_style": "mapbox-style"}


def test_plotly_map_layout_uses_google_tiles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTING_PROVIDER", "google")
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "abc123")

    layout = map_provider.plotly_map_layout({"lat": -25.0, "lon": 133.0}, zoom=4, engine="map")

    assert "map" in layout
    config = layout["map"]
    assert config["style"] == "white-bg"
    assert config["layers"]
    tile_url = config["layers"][0]["source"][0]
    assert "google.com" in tile_url
    assert "abc123" in tile_url


def test_plotly_map_layout_defaults_when_not_google(monkeypatch: pytest.MonkeyPatch) -> None:
    layout = map_provider.plotly_map_layout({"lat": -25.0, "lon": 133.0}, zoom=3, engine="mapbox")

    assert "mapbox" in layout
    config = layout["mapbox"]
    assert config["style"] == "carto-positron"
    assert "layers" not in config
