from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corkysoft.au_address import GeocodeResult, geocode_with_normalization, normalize_au_address


class FakePeliasClient:
    def __init__(self, responses: Dict[str, Dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[Dict[str, Any]] = []

    def pelias_search(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(kwargs)
        key = kwargs.get("text")
        return self.responses.get(
            key,
            {
                "features": [
                    {
                        "geometry": {"coordinates": [153.02, -27.47]},
                        "properties": {
                            "label": "Kangaroo Point, QLD",
                            "name": "Kangaroo Point",
                            "locality": "Kangaroo Point",
                        },
                    }
                ]
            },
        )


def test_normalize_au_address_preserves_state_and_lists_alternatives() -> None:
    result = normalize_au_address("15 cr southbank 4101")
    assert result.canonical == "15 Cr Southbank 4101, QLD"
    assert "15 Crescent Southbank 4101, QLD" in result.alternatives
    assert "15 Circuit Southbank 4101, QLD" in result.alternatives
    assert result.ambiguous_tokens["Cr"] == ["Crescent", "Court", "Circuit"]


def test_normalize_au_address_respects_state_capitalisation() -> None:
    result = normalize_au_address("10 main st nsw 2000")
    assert result.canonical == "10 Main Street NSW 2000"


def test_geocode_with_normalization_surfaces_suggestions() -> None:
    fake_client = FakePeliasClient({})
    result = geocode_with_normalization(
        fake_client,
        "25 cr kangaroo point 4169",
        "Australia",
    )

    assert isinstance(result, GeocodeResult)
    assert result.normalization is not None
    assert result.normalization.alternatives
    assert result.normalization.autocorrections
    assert any("Kangaroo" in suggestion for suggestion in result.suggestions)
    assert fake_client.calls, "Expected the Pelias client to be called"
    first_call = fake_client.calls[0]
    assert first_call["layers"] == ["address", "street", "locality"]


def test_geocode_with_normalization_normalises_string_parameters() -> None:
    fake_client = FakePeliasClient({})

    geocode_with_normalization(
        fake_client,
        "1 queen st",
        "Australia",
        strict_layers="address , street ,locality ",
        strict_sources="osm, wof",
    )

    assert fake_client.calls, "Expected the Pelias client to be called"
    first_call = fake_client.calls[0]
    assert first_call["layers"] == ["address", "street", "locality"]
    assert first_call["sources"] == ["osm", "wof"]


def test_geocode_prefers_feature_matching_input_tokens() -> None:
    fake_client = FakePeliasClient(
        {
            "Alice Springs, Australia": {
                "features": [
                    {
                        "geometry": {"coordinates": [146.66, -19.29]},
                        "properties": {
                            "label": "Alice River, QLD, Australia",
                            "name": "Alice River",
                            "locality": "Alice River",
                        },
                    },
                    {
                        "geometry": {"coordinates": [133.88, -23.7]},
                        "properties": {
                            "label": "Alice Springs NT, Australia",
                            "name": "Alice Springs",
                            "locality": "Alice Springs",
                            "region": "Northern Territory",
                        },
                    },
                ]
            }
        }
    )

    result = geocode_with_normalization(fake_client, "Alice Springs", "Australia")

    assert result.label == "Alice Springs NT, Australia"
    assert abs(result.lon - 133.88) < 1e-6
    assert abs(result.lat - (-23.7)) < 1e-6
