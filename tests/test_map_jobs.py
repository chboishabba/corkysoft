"""Tests for the map generation helpers in :mod:`map_jobs`."""

from __future__ import annotations

import json

import pytest

from map_jobs import combine_route_geojson, compute_map_center


def test_combine_route_geojson_merges_features():
    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"id": 1}, "geometry": {"type": "LineString", "coordinates": []}},
            {"type": "Feature", "properties": {"id": 2}, "geometry": {"type": "LineString", "coordinates": []}},
        ],
    }
    single_feature = {
        "type": "Feature",
        "properties": {"id": 3},
        "geometry": {"type": "LineString", "coordinates": []},
    }

    combined = combine_route_geojson([
        json.dumps(feature_collection),
        json.dumps(single_feature),
    ])

    ids = [feature["properties"]["id"] for feature in combined["features"]]
    assert sorted(ids) == [1, 2, 3]


def test_combine_route_geojson_skips_invalid_entries():
    combined = combine_route_geojson(["not json", json.dumps({"foo": "bar"}), ""])  # type: ignore[list-item]
    assert combined == {"type": "FeatureCollection", "features": []}


@pytest.mark.parametrize(
    "rows, expected",
    [
        (
            [
                {
                    "origin_lat": -33.0,
                    "origin_lon": 150.0,
                    "dest_lat": -32.0,
                    "dest_lon": 151.0,
                },
                {
                    "origin_lat": -31.0,
                    "origin_lon": 149.0,
                    "dest_lat": -30.0,
                    "dest_lon": 148.0,
                },
            ],
            [(-33.0 + -32.0 + -31.0 + -30.0) / 4, (150.0 + 151.0 + 149.0 + 148.0) / 4],
        ),
        (
            [
                {"origin_lat": None, "origin_lon": None, "dest_lat": None, "dest_lon": None},
            ],
            [-25.0, 135.0],
        ),
    ],
)
def test_compute_map_center(rows, expected):
    assert compute_map_center(rows) == expected
