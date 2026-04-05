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
    assert kwargs["map_style"] == "roadmap"


def test_session_selected_provider_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTING_PROVIDER", "ors")
    monkeypatch.setattr(map_provider, "_session_selected_provider", lambda: "google")

    assert map_provider.using_google_maps() is True


def test_session_selected_provider_accepts_canonical_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        map_provider.st,
        "session_state",
        {map_provider.ROUTING_PROVIDER_SESSION_KEY: "google"},
    )

    assert map_provider._session_selected_provider() == "google"


def test_session_selected_provider_handles_ui_label(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        map_provider.st,
        "session_state",
        {map_provider.ROUTING_PROVIDER_SESSION_KEY: "Google Maps"},
    )

    assert map_provider._session_selected_provider() == "google"


def test_resolved_provider_normalises_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTING_PROVIDER", "Google Maps")

    assert map_provider._resolved_provider() == "google"


def test_pydeck_map_kwargs_falls_back_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTING_PROVIDER", "google")

    kwargs = map_provider.pydeck_map_kwargs("mapbox-style")

    assert kwargs == {"map_style": None}


def test_pydeck_map_kwargs_ignores_key_when_not_google(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "abc123")

    kwargs = map_provider.pydeck_map_kwargs("mapbox-style")

    assert kwargs == {"map_style": "mapbox-style"}


def test_pydeck_map_kwargs_defaults_to_light_when_not_google() -> None:
    kwargs = map_provider.pydeck_map_kwargs(None)

    assert kwargs == {"map_style": "light"}


def test_pydeck_map_kwargs_accepts_explicit_google_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "abc123")

    kwargs = map_provider.pydeck_map_kwargs(None, provider="google")

    assert kwargs["map_provider"] == "google_maps"
    assert kwargs["map_style"] == "roadmap"


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


def test_plotly_map_layout_accepts_explicit_google_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "abc123")

    layout = map_provider.plotly_map_layout(
        {"lat": -25.0, "lon": 133.0},
        zoom=4,
        engine="map",
        provider="Google Maps",
    )

    config = layout["map"]
    assert config["style"] == "white-bg"
    assert config["layers"]


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


def test_plotly_map_layout_stays_blank_without_key_when_google_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROUTING_PROVIDER", "google")

    layout = map_provider.plotly_map_layout({"lat": -25.0, "lon": 133.0}, zoom=3, engine="map")

    assert "map" in layout
    config = layout["map"]
    assert config["style"] == "white-bg"
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


def test_folium_map_configuration_stays_provider_strict_without_google_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROUTING_PROVIDER", "google")

    map_kwargs, tile_layer_kwargs = map_provider.folium_map_configuration()

    assert map_kwargs["tiles"] is None
    assert tile_layer_kwargs is None
