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


def test_pydeck_map_kwargs_ignores_key_when_not_google(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "abc123")

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
    assert config["layers"][0]["below"] == "traces"


def test_plotly_map_layout_defaults_when_not_google(monkeypatch: pytest.MonkeyPatch) -> None:
    layout = map_provider.plotly_map_layout({"lat": -25.0, "lon": 133.0}, zoom=3, engine="mapbox")

    assert "mapbox" in layout
    config = layout["mapbox"]
    assert config["style"] == "carto-positron"
    assert "layers" not in config


def test_plotly_map_layout_ignores_key_when_not_google(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "abc123")

    layout = map_provider.plotly_map_layout({"lat": -25.0, "lon": 133.0}, zoom=3, engine="mapbox")

    assert "mapbox" in layout
    config = layout["mapbox"]
    assert config["style"] == "carto-positron"
    assert "layers" not in config


def test_google_street_view_static_url_uses_google_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTING_PROVIDER", "google")
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "abc123")

    url = map_provider.google_street_view_static_url(
        lat=-27.47,
        lon=153.02,
        heading=45.0,
    )

    assert url is not None
    assert "maps.googleapis.com/maps/api/streetview" in url
    assert "location=-27.470000%2C153.020000" in url
    assert "heading=45.0" in url
    assert "key=abc123" in url


def test_google_street_view_static_url_returns_none_when_not_google(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "abc123")
    assert map_provider.google_street_view_static_url(lat=-27.47, lon=153.02) is None


def test_google_street_view_360_url_uses_google_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTING_PROVIDER", "google")
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "abc123")

    url = map_provider.google_street_view_360_url(
        lat=-27.47,
        lon=153.02,
        heading=180.0,
    )

    assert url is not None
    assert "google.com/maps/@" in url
    assert "map_action=pano" in url
    assert "viewpoint=-27.470000%2C153.020000" in url
    assert "heading=180.0" in url


def test_google_street_view_embed_url_uses_google_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTING_PROVIDER", "google")
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "abc123")

    url = map_provider.google_street_view_embed_url(
        lat=-27.47,
        lon=153.02,
        heading=135.0,
    )

    assert url is not None
    assert "google.com/maps/embed/v1/streetview" in url
    assert "location=-27.470000%2C153.020000" in url
    assert "heading=135.0" in url
    assert "key=abc123" in url
