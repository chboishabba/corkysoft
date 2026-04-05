import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.components import maps
from dashboard.components.maps import (
    _base_map_layer_for_network,
    _coerce_position_sequence,
    _filter_valid_geometry_rows,
    _geojson_to_path,
    _is_valid_position_sequence,
    _pydeck_frame,
    _valid_xy_row,
)


def test_pydeck_frame_deduplicates_duplicate_columns() -> None:
    df = pd.DataFrame([[1, 2, 3]], columns=["lat", "lon", "lat"])
    result = _pydeck_frame(df)

    assert list(result.columns) == ["lat", "lon"]
    assert result.iloc[0].to_dict() == {"lat": 1, "lon": 2}


def test_base_map_layer_is_suppressed_for_google_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(maps, "using_google_maps", lambda: True)
    monkeypatch.setattr(maps, "google_maps_requested_without_key", lambda: False)

    assert _base_map_layer_for_network({}) is None


def test_base_map_layer_uses_osm_for_non_google_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_layer(layer_type: str, **kwargs: object) -> dict[str, object]:
        captured["layer_type"] = layer_type
        captured.update(kwargs)
        return captured

    monkeypatch.setattr(maps, "using_google_maps", lambda: False)
    monkeypatch.setattr(maps, "google_maps_requested_without_key", lambda: False)
    monkeypatch.setattr(maps.pdk, "Layer", fake_layer)

    layer = _base_map_layer_for_network({})

    assert layer is captured
    assert captured["layer_type"] == "TileLayer"
    assert captured["data"] == "https://tile.openstreetmap.org/{z}/{x}/{y}.png"


def test_position_sequence_validation_rejects_invalid_geometry() -> None:
    assert _is_valid_position_sequence([[151.0, -33.0], [152.0, -32.0]], minimum_points=2)
    assert not _is_valid_position_sequence([[151.0, -33.0]], minimum_points=2)
    assert not _is_valid_position_sequence([[151.0, float("nan")], [152.0, -32.0]], minimum_points=2)
    assert not _is_valid_position_sequence("not-a-sequence", minimum_points=2)


def test_valid_xy_row_requires_numeric_coordinates() -> None:
    row = pd.Series({"origin_lon": 151.0, "origin_lat": -33.0})
    assert _valid_xy_row(row, lon_key="origin_lon", lat_key="origin_lat")

    bad_row = pd.Series({"origin_lon": "bad", "origin_lat": -33.0})
    assert not _valid_xy_row(bad_row, lon_key="origin_lon", lat_key="origin_lat")


def test_filter_valid_geometry_rows_drops_invalid_paths() -> None:
    df = pd.DataFrame(
        {
            "route_path": [
                [[151.0, -33.0], [152.0, -32.0]],
                "bad-geometry",
            ]
        }
    )

    filtered = _filter_valid_geometry_rows(df, column="route_path", minimum_points=2)

    assert len(filtered) == 1
    assert filtered.iloc[0]["route_path"] == [[151.0, -33.0], [152.0, -32.0]]


def test_coerce_position_sequence_parses_json_payloads_and_rejects_html() -> None:
    payload = '[["151.2", "-33.8"], ["151.5", "-34.1"], ["151.7", "-34.3"]]'
    assert _coerce_position_sequence(payload, minimum_points=2) == [
        [151.2, -33.8],
        [151.5, -34.1],
        [151.7, -34.3],
    ]
    assert _coerce_position_sequence("<!-- Copy something -->", minimum_points=2) is None


def test_geojson_to_path_rejects_html_payload() -> None:
    assert _geojson_to_path("<!-- Copy something -->") is None


def test_render_network_map_uses_shared_pydeck_renderer_for_google_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(maps, "using_google_maps", lambda: True)
    monkeypatch.setattr(maps, "google_maps_requested_without_key", lambda: False)
    monkeypatch.setattr(maps.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(maps.st, "radio", lambda *args, **kwargs: "Overlay")
    monkeypatch.setattr(maps.st, "checkbox", lambda *args, **kwargs: False)
    monkeypatch.setattr(maps.st, "toggle", lambda *args, **kwargs: False)
    monkeypatch.setattr(maps, "_base_map_layer_for_network", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(maps, "_initial_view_state", lambda _df: object())
    monkeypatch.setattr(maps.pdk, "Deck", lambda *args, **kwargs: {"args": args, "kwargs": kwargs})
    monkeypatch.setattr(maps.pdk, "ViewState", lambda *args, **kwargs: object())

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_pydeck_chart(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr(maps.st, "pydeck_chart", fake_pydeck_chart)

    maps.render_network_map(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    assert calls


def test_render_network_map_overlay_adds_history_polygon_layer_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_layers: list[str] = []

    monkeypatch.setattr(maps, "using_google_maps", lambda: False)
    monkeypatch.setattr(maps, "google_maps_requested_without_key", lambda: False)
    monkeypatch.setattr(maps.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(maps.st, "radio", lambda *args, **kwargs: "Overlay")
    monkeypatch.setattr(maps.st, "checkbox", lambda *args, **kwargs: False)
    monkeypatch.setattr(maps.st, "toggle", lambda *args, **kwargs: True)
    monkeypatch.setattr(maps, "_base_map_layer_for_network", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(maps.st, "pydeck_chart", lambda *args, **kwargs: None)
    monkeypatch.setattr(maps.pdk, "Deck", lambda *args, **kwargs: None)
    monkeypatch.setattr(maps.pdk, "ViewState", lambda *args, **kwargs: None)

    def fake_layer(layer_type: str, **kwargs: object) -> dict[str, object]:
        created_layers.append(layer_type)
        return {"layer_type": layer_type, **kwargs}

    monkeypatch.setattr(maps.pdk, "Layer", fake_layer)

    historical_routes = pd.DataFrame(
        [
            {
                "id": 1,
                "lane_key": "a->b",
                "origin_lon": 153.0,
                "origin_lat": -27.0,
                "dest_lon": 153.4,
                "dest_lat": -28.0,
                "route_polygon": [[153.0, -27.0], [153.2, -27.5], [153.4, -28.0]],
                "fill_colour": [255, 0, 0, 80],
                "colour": [255, 0, 0],
                "line_width": 2,
                "tooltip": "Historical lane",
                "job_count": 1,
                "route_geojson": None,
            }
        ]
    )

    maps.render_network_map(historical_routes, pd.DataFrame(), pd.DataFrame())

    assert created_layers.count("PolygonLayer") == 1


def test_render_network_map_skips_malformed_active_route_geojson_before_pydeck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_layers: list[dict[str, object]] = []
    warnings: list[str] = []

    monkeypatch.setattr(maps, "using_google_maps", lambda: False)
    monkeypatch.setattr(maps, "google_maps_requested_without_key", lambda: False)
    monkeypatch.setattr(maps.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(maps.st, "radio", lambda *args, **kwargs: "Overlay")
    monkeypatch.setattr(maps.st, "checkbox", lambda *args, **kwargs: True)
    monkeypatch.setattr(maps.st, "toggle", lambda *args, **kwargs: True)
    monkeypatch.setattr(maps, "_base_map_layer_for_network", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(maps.st, "pydeck_chart", lambda *args, **kwargs: None)
    monkeypatch.setattr(maps.pdk, "Deck", lambda *args, **kwargs: None)
    monkeypatch.setattr(maps.pdk, "ViewState", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        maps.logger,
        "warning",
        lambda message, skipped, label: warnings.append(message % (skipped, label)),
    )

    def fake_layer(layer_type: str, **kwargs: object) -> dict[str, object]:
        payload = {"layer_type": layer_type, **kwargs}
        created_layers.append(payload)
        return payload

    monkeypatch.setattr(maps.pdk, "Layer", fake_layer)

    active_routes = pd.DataFrame(
        [
            {
                "truck_id": "T-1",
                "origin_lon": 153.0,
                "origin_lat": -27.0,
                "dest_lon": 153.4,
                "dest_lat": -28.0,
                "line_width": 2,
                "route_geometry": "{\"coordinates\":[[153.0,-27.0],[153.4,-28.0]]}",
            }
        ]
    )

    maps.render_network_map(pd.DataFrame(), pd.DataFrame(), active_routes)

    path_layers = [layer for layer in created_layers if layer["layer_type"] == "PathLayer"]
    assert len(path_layers) == 0
    assert warnings == ["Skipping 1 malformed active route GeoJSON payload(s) before pydeck render."]
