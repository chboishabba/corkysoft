import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analytics import routing_provider as analytics_rp
from corkysoft import routing as routing_module
from corkysoft.routing import providers as provider_module


def test_get_google_maps_client_uses_env_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    created_keys: list[str] = []

    class DummyClient:
        def __init__(self, *, key: str) -> None:
            created_keys.append(key)

    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "test-key")
    monkeypatch.setattr(provider_module, "googlemaps", SimpleNamespace(Client=DummyClient), raising=False)
    monkeypatch.setattr(provider_module, "_GOOGLE_CLIENT", None, raising=False)

    client = provider_module.get_google_maps_client()

    assert isinstance(client, DummyClient)
    assert created_keys == ["test-key"]


def test_get_google_maps_client_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyClient:
        def __init__(self, *, key: str) -> None:
            pass

    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)
    monkeypatch.setattr(provider_module, "googlemaps", SimpleNamespace(Client=DummyClient), raising=False)
    monkeypatch.setattr(provider_module, "_GOOGLE_CLIENT", None, raising=False)

    with pytest.raises(RuntimeError, match="GOOGLE_MAPS_API_KEY"):
        provider_module.get_google_maps_client()


def test_google_routes_provider_invokes_directions(monkeypatch: pytest.MonkeyPatch) -> None:
    class RecordingClient:
        def __init__(self) -> None:
            self.calls: list[tuple[Any, Any, str]] = []

        def directions(self, *, origin, destination, mode):
            self.calls.append((origin, destination, mode))
            return {
                "routes": [
                    {
                        "legs": [
                            {
                                "distance": {"value": 2400},
                                "duration": {"value": 900},
                            }
                        ],
                        "overview_polyline": {"points": "encoded-polyline"},
                    }
                ]
            }

    client = RecordingClient()
    provider = analytics_rp.GoogleRoutesProvider(client)

    result = provider.route_geometry(
        origin=(153.0, -27.5),
        destination=(151.0, -33.0),
        profile="driving-car",
    )

    assert client.calls == [((-27.5, 153.0), (-33.0, 151.0), "driving")]
    assert result.distance_km == pytest.approx(2.4)
    assert result.duration_hr == pytest.approx(0.25)
    assert result.encoded_polyline == "encoded-polyline"


def test_google_routes_provider_accepts_sequence_payload() -> None:
    class ListClient:
        def directions(self, *, origin, destination, mode):
            return [
                {
                    "legs": [
                        {
                            "distance": {"value": 3100},
                            "duration": {"value": 700},
                        }
                    ],
                    "overview_polyline": {"points": "sequence-polyline"},
                }
            ]

    provider = analytics_rp.GoogleRoutesProvider(ListClient())

    result = provider.route_geometry(
        origin=(153.0, -27.5),
        destination=(151.0, -33.0),
        profile="driving-car",
    )

    assert result.distance_km == pytest.approx(3.1)
    assert result.duration_hr == pytest.approx(700 / 3600.0)
    assert result.encoded_polyline == "sequence-polyline"


def test_google_routes_provider_falls_back_to_compute_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    class ComputeClient:
        def __init__(self) -> None:
            self.calls: list[tuple[Any, Any, str]] = []

        def compute_routes(self, *, origin, destination, profile):
            self.calls.append((origin, destination, profile))
            return {
                "routes": [
                    {
                        "distanceMeters": 5000,
                        "duration": {"value": 1200},
                        "polyline": {"encodedPolyline": "from-compute"},
                    }
                ]
            }

    client = ComputeClient()
    provider = analytics_rp.GoogleRoutesProvider(client)

    result = provider.route_geometry(
        origin=(150.0, -33.0),
        destination=(151.0, -34.0),
        profile="driving-car",
    )

    assert client.calls == [((-33.0, 150.0), (-34.0, 151.0), "driving-car")]
    assert result.distance_km == pytest.approx(5.0)
    assert result.duration_hr == pytest.approx(1200 / 3600.0)
    assert result.encoded_polyline == "from-compute"


def test_get_routing_provider_uses_env_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTING_PROVIDER", "google")
    fake_client = object()

    provider = analytics_rp.get_routing_provider(client=fake_client)

    assert isinstance(provider, analytics_rp.GoogleRoutesProvider)
    assert provider._client is fake_client  # type: ignore[attr-defined]


def test_get_routing_provider_accepts_string_provider_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = object()

    provider = analytics_rp.get_routing_provider(provider="google", client=fake_client)

    assert isinstance(provider, analytics_rp.GoogleRoutesProvider)
    assert provider._client is fake_client  # type: ignore[attr-defined]


def test_get_routing_provider_accepts_google_maps_label() -> None:
    fake_client = object()

    provider = analytics_rp.get_routing_provider(provider="Google Maps", client=fake_client)

    assert isinstance(provider, analytics_rp.GoogleRoutesProvider)
    assert provider._client is fake_client  # type: ignore[attr-defined]


def test_snap_coordinates_to_road_respects_routing_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTING_PROVIDER", "google")
    calls: dict[str, Any] = {}

    class DummyGoogleProvider:
        def __init__(self, client=None) -> None:
            calls["client"] = client

        def snap_to_road(self, origin, destination, *, profile="driving-car", radii=None):
            calls["args"] = (origin, destination, profile, tuple(radii or ()))
            return provider_module.SnapResult(
                coordinates=[
                    [origin[0] + 0.01, origin[1] + 0.02],
                    [destination[0] + 0.02, destination[1] + 0.03],
                ],
                notes={
                    "origin": "Snapped to nearest routable road",
                    "destination": "Snapped to nearest routable road",
                },
            )

    monkeypatch.setattr(routing_module, "GoogleMapsRoutingProvider", DummyGoogleProvider)

    result = routing_module.snap_coordinates_to_road(
        (153.0, -27.5),
        (151.0, -33.0),
    )

    assert calls["args"][0] == (153.0, -27.5)
    assert calls["args"][1] == (151.0, -33.0)
    assert calls["args"][2] == "driving-car"
    assert calls["args"][3] == routing_module.SNAP_SEARCH_RADII
    assert result.changed is True
    assert result.notes == {
        "origin": "Snapped to nearest routable road",
        "destination": "Snapped to nearest routable road",
    }


def test_google_routing_request_denied_error_is_human_readable(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyApiError(Exception):
        def __init__(self, status: str, message: str) -> None:
            super().__init__(message)
            self.status = status
            self.message = message

    class FailingClient:
        def geocode(self, *args, **kwargs):
            raise DummyApiError("REQUEST_DENIED", "Not authorized for this API")

    monkeypatch.setattr(provider_module, "_GoogleMapsApiError", DummyApiError, raising=False)

    provider = provider_module.GoogleMapsRoutingProvider(client=FailingClient())

    with pytest.raises(provider_module.RoutingError) as excinfo:
        provider.geocode("Origin", "Australia")

    message = str(excinfo.value)
    assert "request denied" in message.lower().replace("_", " ")
    assert "ROUTING_PROVIDER=ors" in message
