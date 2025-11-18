import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corkysoft.au_address import GeocodeResult
from corkysoft.routing import (
    GoogleMapsRoutingProvider,
    OpenRouteServiceProvider,
    SNAP_SEARCH_RADII,
)
from corkysoft.routing.providers import IncompleteRouteError


class _FakeORSClient:
    def __init__(self) -> None:
        self.snap_calls: list[dict] = []
        self.nearest_calls: list[list[list[float]]] = []

    def directions(self, *, coordinates, profile, format):
        assert profile == "driving-car"
        assert format == "json"
        return {
            "routes": [
                {
                    "summary": {"distance": 1500.0, "duration": 1800.0},
                    "segments": [],
                }
            ]
        }

    def snap(self, *, profile, locations, radius, format):
        assert profile == "driving-car"
        assert format == "json"
        payload = {
            "profile": profile,
            "locations": locations,
            "radius": radius,
        }
        self.snap_calls.append(payload)
        lon, lat = locations[0]
        return {"locations": [{"location": [lon + 0.01, lat + 0.02]}]}

    def nearest(self, *, coordinates, number):
        self.nearest_calls.append(coordinates)
        lon, lat = coordinates[0]
        return {
            "features": [
                {"geometry": {"coordinates": [lon + 0.05, lat + 0.03]}},
            ]
        }

    def isochrones(self, *, locations, profile, range):
        return {
            "features": [
                {"geometry": {"type": "Polygon", "coordinates": locations}}
            ],
            "type": "FeatureCollection",
        }


def test_openrouteservice_provider_wraps_client_calls() -> None:
    client = _FakeORSClient()
    provider = OpenRouteServiceProvider(client=client)

    route = provider.directions(
        coordinates=[[150.0, -33.0], [151.0, -34.0]], profile="driving-car"
    )
    assert route.distance_km == pytest.approx(1.5)
    assert route.duration_hr == pytest.approx(0.5)

    snap = provider.snap_to_road(
        (150.0, -33.0),
        (151.0, -34.0),
        profile="driving-car",
        radii=SNAP_SEARCH_RADII,
    )
    assert snap is not None
    assert client.snap_calls
    assert snap.notes["origin"].startswith("Snapped")

    iso = provider.isochrone(
        centre=(150.0, -33.0), profile="driving-car", range_seconds=[300]
    )
    assert iso is not None


def test_openrouteservice_provider_reraises_missing_metrics() -> None:
    class _Client:
        def directions(self, *_, **__):
            return {"routes": [{"summary": {}, "segments": []}]}

    provider = OpenRouteServiceProvider(client=_Client())
    with pytest.raises(IncompleteRouteError):
        provider.directions(coordinates=[[0.0, 0.0], [1.0, 1.0]])


class _FakeGoogleClient:
    def __init__(self) -> None:
        self.snap_calls: list[list[tuple[float, float]]] = []
        self.iso_calls: list[dict[str, object]] = []

    def geocode(self, place, *, components):
        assert components["country"] == "Australia"
        return [
            {
                "formatted_address": "123 Example St, Brisbane QLD",
                "geometry": {"location": {"lng": 153.02, "lat": -27.47}},
                "address_components": [
                    {
                        "types": ["postal_code"],
                        "long_name": "4000",
                        "short_name": "4000",
                    },
                    {
                        "types": ["administrative_area_level_1"],
                        "long_name": "Queensland",
                        "short_name": "QLD",
                    },
                ],
            }
        ]

    def directions(self, *, origin, destination, mode):
        assert mode == "driving"
        assert origin == (-27.5, 153.0)
        assert destination == (-27.6, 153.1)
        return [
            {
                "legs": [
                    {
                        "distance": {"value": 12000},
                        "duration": {"value": 900},
                    }
                ],
                "overview_polyline": {"points": "abc"},
            }
        ]

    def isochrones(self, *, centre, profile, range_seconds):
        self.iso_calls.append(
            {"centre": centre, "profile": profile, "range_seconds": range_seconds}
        )
        return {
            "paths": [
                [
                    {"lat": centre[1] + 0.1, "lng": centre[0] + 0.1},
                    {"lat": centre[1] + 0.1, "lng": centre[0] + 0.2},
                    {"lat": centre[1] + 0.2, "lng": centre[0] + 0.2},
                ]
            ]
        }

    def snap_to_roads(self, *, path, interpolate):
        self.snap_calls.append(path)
        return [
            {"location": {"longitude": path[0][1] + 0.01, "latitude": path[0][0] + 0.01}},
            {"location": {"longitude": path[1][1] + 0.01, "latitude": path[1][0] + 0.01}},
        ]


def test_google_maps_provider_normalises_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = GoogleMapsRoutingProvider(client=_FakeGoogleClient())

    geocode = provider.geocode("123 Example St", "Australia")
    assert isinstance(geocode, GeocodeResult)
    assert geocode.postalcode == "4000"
    assert geocode.region_code == "QLD"

    route = provider.directions(
        coordinates=[[153.0, -27.5], [153.1, -27.6]], profile="driving-car"
    )
    assert route.distance_km == pytest.approx(12.0)
    assert route.duration_hr == pytest.approx(0.25)

    snap = provider.snap_to_road((153.0, -27.5), (153.1, -27.6))
    assert snap is not None
    assert provider._client.snap_calls
    assert provider._client.snap_calls[0] == [(-27.5, 153.0), (-27.6, 153.1)]

    iso = provider.isochrone(
        centre=(153.0, -27.5), profile="driving-car", range_seconds=[300]
    )
    assert iso is not None
    assert provider._client.iso_calls
    assert iso.raw["paths"]
