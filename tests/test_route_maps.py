from __future__ import annotations

import pandas as pd

from dashboard.components import route_maps


def test_normalise_provider_labels_uses_routing_env_for_bogus_names() -> None:
    labels = route_maps._normalise_provider_labels(  # type: ignore[attr-defined]
        ["str"],
        routing_provider_env="google",
    )

    assert labels == ["Google Maps"]


def test_merge_saved_job_overlay_rows_adds_saved_jobs_and_marks_source(
    monkeypatch,
) -> None:
    base_df = pd.DataFrame(
        [
            {
                "id": 1,
                "origin": "Brisbane",
                "destination": "Gold Coast",
                "origin_lat": -27.4698,
                "origin_lon": 153.0251,
                "dest_lat": -28.0167,
                "dest_lon": 153.4,
            }
        ]
    )
    saved_rows = [
        {
            "id": 2,
            "origin": "Sydney",
            "destination": "Newcastle",
            "origin_resolved": "Sydney NSW",
            "destination_resolved": "Newcastle NSW",
            "origin_lat": -33.8688,
            "origin_lon": 151.2093,
            "dest_lat": -32.9283,
            "dest_lon": 151.7817,
            "route_geojson": '{"type":"FeatureCollection","features":[]}',
        }
    ]
    captions: list[str] = []

    monkeypatch.setattr(route_maps, "fetch_job_route_rows", lambda conn, include_actual: saved_rows)
    monkeypatch.setattr(route_maps.st, "caption", lambda text: captions.append(str(text)))

    merged = route_maps._merge_saved_job_overlay_rows(base_df, conn=None)  # type: ignore[arg-type]

    assert list(merged["route_dataset_source"]) == ["Current selection", "Saved jobs"]
    saved_row = merged.loc[merged["id"] == 2].iloc[0]
    assert saved_row["origin_city"] == "Sydney NSW"
    assert saved_row["destination_city"] == "Newcastle NSW"
    assert saved_row["client_display"] == "Saved jobs"
    assert any("Added 1 saved job route rows" in text for text in captions)


def test_merge_saved_job_overlay_rows_skips_duplicate_ids(monkeypatch) -> None:
    base_df = pd.DataFrame(
        [
            {
                "id": 7,
                "origin_lat": -27.0,
                "origin_lon": 153.0,
                "dest_lat": -28.0,
                "dest_lon": 153.4,
            }
        ]
    )
    saved_rows = [
        {
            "id": 7,
            "origin": "Same",
            "destination": "Same",
            "origin_lat": -27.0,
            "origin_lon": 153.0,
            "dest_lat": -28.0,
            "dest_lon": 153.4,
        }
    ]
    captions: list[str] = []

    monkeypatch.setattr(route_maps, "fetch_job_route_rows", lambda conn, include_actual: saved_rows)
    monkeypatch.setattr(route_maps.st, "caption", lambda text: captions.append(str(text)))

    merged = route_maps._merge_saved_job_overlay_rows(base_df, conn=None)  # type: ignore[arg-type]

    assert len(merged) == 1
    assert list(merged["route_dataset_source"]) == ["Current selection"]
    assert any("already covered by the current selection" in text for text in captions)


def test_merge_saved_job_overlay_rows_skips_duplicate_route_signature(monkeypatch) -> None:
    base_df = pd.DataFrame(
        [
            {
                "id": 1,
                "origin_lat": -27.4698,
                "origin_lon": 153.0251,
                "dest_lat": -28.0167,
                "dest_lon": 153.4,
                "route_geojson": '{"type":"FeatureCollection","features":[]}',
            }
        ]
    )
    saved_rows = [
        {
            "id": 99,
            "origin": "Different id",
            "destination": "Different id",
            "origin_lat": -27.4698,
            "origin_lon": 153.0251,
            "dest_lat": -28.0167,
            "dest_lon": 153.4,
            "route_geojson": '{"type":"FeatureCollection","features":[]}',
        }
    ]
    captions: list[str] = []

    monkeypatch.setattr(route_maps, "fetch_job_route_rows", lambda conn, include_actual: saved_rows)
    monkeypatch.setattr(route_maps.st, "caption", lambda text: captions.append(str(text)))

    merged = route_maps._merge_saved_job_overlay_rows(base_df, conn=None)  # type: ignore[arg-type]

    assert len(merged) == 1
    assert list(merged["route_dataset_source"]) == ["Current selection"]
    assert any("already covered by the current selection" in text for text in captions)


def test_merge_saved_job_overlay_rows_warns_and_returns_base_on_failure(monkeypatch) -> None:
    base_df = pd.DataFrame(
        [
            {
                "id": 1,
                "origin_lat": -27.0,
                "origin_lon": 153.0,
                "dest_lat": -28.0,
                "dest_lon": 153.4,
            }
        ]
    )
    warnings: list[str] = []

    def _raise(conn, include_actual):
        raise RuntimeError("boom")

    monkeypatch.setattr(route_maps, "fetch_job_route_rows", _raise)
    monkeypatch.setattr(route_maps.st, "warning", lambda text: warnings.append(str(text)))

    merged = route_maps._merge_saved_job_overlay_rows(base_df, conn=None)  # type: ignore[arg-type]

    assert len(merged) == 1
    assert list(merged["route_dataset_source"]) == ["Current selection"]
    assert warnings == ["Unable to load saved jobs overlay: boom"]
