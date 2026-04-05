from __future__ import annotations

from dashboard.workspace_state import (
    WORKSPACE_STATE_PARAM,
    decode_workspace_state,
    normalize_workspace_state,
    workspace_state_from_query_params,
    workspace_state_to_query_params,
)


_TABS = ["Quote", "Pricing Intelligence", "Network", "Operations", "Admin"]


def test_workspace_state_round_trips_operations_diary_context() -> None:
    params = workspace_state_to_query_params(
        {
            "view": "Operations",
            "workflow": "operations_diary",
            "diary_view": "week",
            "diary_date": "2026-04-02",
            "diary_job": "42",
        },
        available_tabs=_TABS,
    )

    assert params["view"] == "Operations"
    assert params["workflow"] == "operations_diary"
    assert params["diary_view"] == "week"
    assert params["diary_date"] == "2026-04-02"
    assert params["diary_job"] == "42"
    assert WORKSPACE_STATE_PARAM in params

    restored = workspace_state_from_query_params(
        {key: [value] for key, value in params.items()},
        available_tabs=_TABS,
    )

    assert restored == {
        "v": 1,
        "view": "Operations",
        "workflow": "operations_diary",
        "operations_tab": "Operations Diary",
        "diary_view": "week",
        "diary_date": "2026-04-02",
        "diary_job": "42",
    }


def test_workspace_state_ignores_invalid_view_and_workflow() -> None:
    normalized = normalize_workspace_state(
        {
            "view": "Not A Tab",
            "workflow": "operations_diary",
            "diary_date": "2026-04-02",
        },
        available_tabs=_TABS,
    )
    assert normalized == {"v": 1}

    normalized = normalize_workspace_state(
        {
            "view": "Operations",
            "workflow": "not_real",
            "diary_date": "2026-04-02",
        },
        available_tabs=_TABS,
    )
    assert normalized == {"v": 1, "view": "Operations"}


def test_workspace_state_strips_operations_details_for_non_operations_views() -> None:
    normalized = normalize_workspace_state(
        {
            "view": "Quote",
            "workflow": "operations_diary",
            "diary_date": "2026-04-02",
        },
        available_tabs=_TABS,
    )

    assert normalized == {"v": 1, "view": "Quote"}


def test_workspace_state_legacy_params_override_invalid_token() -> None:
    invalid_token = "not-valid"
    restored = workspace_state_from_query_params(
        {
            WORKSPACE_STATE_PARAM: [invalid_token],
            "view": ["Operations"],
            "workflow": ["planner"],
        },
        available_tabs=_TABS,
    )

    assert restored == {
        "v": 1,
        "view": "Operations",
        "workflow": "planner",
        "operations_tab": "Planner",
    }


def test_workspace_state_decode_rejects_non_mapping_payload() -> None:
    assert decode_workspace_state("") is None
    assert decode_workspace_state("WzEsMiwzXQ") is None


def test_workspace_state_supports_dispatch_workflow() -> None:
    params = workspace_state_to_query_params(
        {"view": "Operations", "workflow": "dispatch"},
        available_tabs=_TABS,
    )

    restored = workspace_state_from_query_params(
        {key: [value] for key, value in params.items()},
        available_tabs=_TABS,
    )

    assert restored == {
        "v": 1,
        "view": "Operations",
        "workflow": "dispatch",
        "operations_tab": "Dispatch",
    }


def test_workspace_state_drops_operations_details_when_view_changes() -> None:
    params = workspace_state_to_query_params(
        {
            "view": "Operations",
            "workflow": "operations_diary",
            "diary_view": "day",
            "diary_date": "2026-04-04",
            "diary_job": "99",
        },
        available_tabs=_TABS,
    )

    restored = workspace_state_from_query_params(
        {**{key: [value] for key, value in params.items()}, "view": ["Quote"]},
        available_tabs=_TABS,
    )

    assert restored == {"v": 1, "view": "Quote"}
