from __future__ import annotations

from datetime import date

import pandas as pd

from dashboard.data import blank_column_mapping
from dashboard.data_controls import (
    _client_options,
    _corridor_options,
    _date_bounds_for_dataset,
    _empty_dataset_message_for,
    _load_dataset_snapshot,
)


def test_empty_dataset_message_matches_dataset_key() -> None:
    assert "historical_jobs" in _empty_dataset_message_for("historical")
    assert "quotes table" in _empty_dataset_message_for("quotes")
    assert "jobs table" in _empty_dataset_message_for("live")
    assert _empty_dataset_message_for("unknown") == "No rows available for the selected dataset."


def test_date_bounds_for_dataset_uses_job_date_when_present() -> None:
    df = pd.DataFrame(
        {
            "job_date": ["2026-03-01", "2026-03-09"],
            "client_display": ["A", "B"],
        }
    )
    start_date, end_date = _date_bounds_for_dataset(
        df,
        blank_column_mapping(),
        today_value=date(2026, 3, 31),
    )
    assert start_date == date(2026, 3, 1)
    assert end_date == date(2026, 3, 9)


def test_corridor_and_client_options_are_sorted_and_unique() -> None:
    df = pd.DataFrame(
        {
            "corridor_display": ["B lane", "A lane", "B lane", None],
            "client_display": ["Zulu", "Alpha", "Zulu", None],
        }
    )
    assert _corridor_options(df) == ["A lane", "B lane"]
    assert _client_options(df) == ["Alpha", "Zulu"]


def test_load_dataset_snapshot_returns_error_for_runtime_failure() -> None:
    def _broken_loader(_conn: object) -> tuple[pd.DataFrame, object]:
        raise RuntimeError("load failed")

    df, mapping, dataset_error, data_available = _load_dataset_snapshot(
        object(),
        _broken_loader,
        "Historical quotes",
    )
    assert df.empty
    assert dataset_error == "load failed"
    assert not data_available
    assert mapping.date is None


def test_load_dataset_snapshot_returns_loaded_frame() -> None:
    expected_df = pd.DataFrame({"job_date": ["2026-03-01"]})
    expected_mapping = blank_column_mapping()

    def _loader(_conn: object) -> tuple[pd.DataFrame, object]:
        return expected_df, expected_mapping

    df, mapping, dataset_error, data_available = _load_dataset_snapshot(
        object(),
        _loader,
        "Historical quotes",
    )
    assert df.equals(expected_df)
    assert mapping is expected_mapping
    assert dataset_error is None
    assert data_available
