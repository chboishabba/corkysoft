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


def test_geocode_prioritises_specific_address_in_autocorrect() -> None:
    fake_client = FakePeliasClient(
        {
            "12 Carlton Street Toowoomba, Australia": {
                "features": [
                    {
                        "geometry": {"coordinates": [151.95, -27.56]},
                        "properties": {
                            "label": "Toowoomba, QLD, Australia",
                            "name": "Toowoomba",
                            "locality": "Toowoomba",
                        },
                    }
                ]
            }
        }
    )

    result = geocode_with_normalization(
        fake_client,
        "12 carlton st toowoomba",
        "Australia",
    )

    assert result.normalization is not None
    assert result.normalization.autocorrections
    first_suggestion = result.normalization.autocorrections[0]
    assert first_suggestion.startswith("12 Carlton Street Toowoomba")


def test_geocode_prefers_address_with_misspelt_query() -> None:
    fake_client = FakePeliasClient(
        {
            "26/100 Champtions Crescent, Brookwater, Australia": {
                "features": [
                    {
                        "geometry": {"coordinates": [152.91, -27.66]},
                        "properties": {
                            "label": "Brookwater QLD 4300, Australia",
                            "name": "Brookwater",
                            "locality": "Brookwater",
                            "layer": "locality",
                            "confidence": 0.6,
                        },
                    },
                    {
                        "geometry": {"coordinates": [152.92, -27.65]},
                        "properties": {
                            "label": "Champions Cresent, Brookwater QLD 4300, Australia",
                            "name": "Champions Cresent",
                            "street": "Champions Cresent",
                            "locality": "Brookwater",
                            "layer": "address",
                            "housenumber": "26",
                            "confidence": 0.9,
                        },
                    },
                ]
            }
        }
    )

    result = geocode_with_normalization(
        fake_client,
        "26/100 CHamptions Crescent, Brookwater",
        "Australia",
    )

    assert result.label == "Champions Cresent, Brookwater QLD 4300, Australia"


def test_geocode_rejects_high_confidence_address_without_token_overlap() -> None:
    fake_client = FakePeliasClient(
        {
            "Riverview QLD, Australia": {
                "features": [
                    {
                        "geometry": {"coordinates": [151.2, -33.9]},
                        "properties": {
                            "label": "123 Example Road, Sydney NSW 2000",
                            "name": "123 Example Road",
                            "street": "Example Road",
                            "locality": "Sydney",
                            "region": "New South Wales",
                            "layer": "address",
                            "housenumber": "123",
                            "confidence": 0.99,
                        },
                    },
                    {
                        "geometry": {"coordinates": [152.88, -27.6]},
                        "properties": {
                            "label": "Riverview",
                            "name": "Riverview",
                            "locality": "Riverview",
                            "layer": "locality",
                            "confidence": 0.6,
                        },
                    },
                ]
            }
        }
    )

    result = geocode_with_normalization(fake_client, "Riverview QLD", "Australia")

    assert result.label == "Riverview"
    assert result.locality == "Riverview"
