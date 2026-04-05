from __future__ import annotations

import base64
import json
from datetime import date
from typing import Any, Iterable, Mapping

WORKSPACE_STATE_VERSION = 1
WORKSPACE_STATE_PARAM = "ws"
_SUPPORTED_OPERATIONS_WORKFLOWS = {
    "dispatch": "Dispatch",
    "planner": "Planner",
    "operations_diary": "Operations Diary",
}
_SUPPORTED_DIARY_VIEWS = {"day", "week"}


def encode_workspace_state(state: Mapping[str, Any]) -> str:
    payload = json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
    token = base64.urlsafe_b64encode(payload).decode("ascii")
    return token.rstrip("=")


def decode_workspace_state(token: str) -> dict[str, Any] | None:
    if not token:
        return None
    try:
        padding = "=" * (-len(token) % 4)
        payload = base64.urlsafe_b64decode(f"{token}{padding}".encode("ascii"))
        decoded = json.loads(payload.decode("utf-8"))
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(decoded, dict):
        return None
    return decoded


def normalize_workspace_state(
    raw_state: Mapping[str, Any] | None,
    *,
    available_tabs: Iterable[str],
) -> dict[str, Any]:
    normalized: dict[str, Any] = {"v": WORKSPACE_STATE_VERSION}
    if not raw_state:
        return normalized

    available_tab_set = {str(tab) for tab in available_tabs}

    view = raw_state.get("view")
    if isinstance(view, str) and view in available_tab_set:
        normalized["view"] = view
    else:
        return normalized

    if normalized["view"] != "Operations":
        return normalized

    workflow = raw_state.get("workflow")
    if isinstance(workflow, str) and workflow in _SUPPORTED_OPERATIONS_WORKFLOWS:
        normalized["workflow"] = workflow
        normalized["operations_tab"] = _SUPPORTED_OPERATIONS_WORKFLOWS[workflow]
    else:
        return normalized

    if workflow != "operations_diary":
        return normalized

    diary_view = raw_state.get("diary_view")
    if isinstance(diary_view, str) and diary_view in _SUPPORTED_DIARY_VIEWS:
        normalized["diary_view"] = diary_view

    diary_date = raw_state.get("diary_date")
    if isinstance(diary_date, str):
        try:
            normalized["diary_date"] = date.fromisoformat(diary_date).isoformat()
        except ValueError:
            pass

    diary_job = raw_state.get("diary_job")
    if isinstance(diary_job, int) and diary_job > 0:
        normalized["diary_job"] = str(diary_job)
    elif isinstance(diary_job, str) and diary_job.isdigit() and int(diary_job) > 0:
        normalized["diary_job"] = diary_job

    return normalized


def workspace_state_from_query_params(
    params: Mapping[str, list[str]],
    *,
    available_tabs: Iterable[str],
) -> dict[str, Any]:
    token = params.get(WORKSPACE_STATE_PARAM, [None])[0]
    token_state = decode_workspace_state(str(token)) if token else None
    legacy_state: dict[str, Any] = {}

    view = params.get("view", [None])[0]
    if isinstance(view, str):
        legacy_state["view"] = view

    workflow = params.get("workflow", [None])[0]
    if isinstance(workflow, str):
        legacy_state["workflow"] = workflow

    diary_view = params.get("diary_view", [None])[0]
    if isinstance(diary_view, str):
        legacy_state["diary_view"] = diary_view

    diary_date = params.get("diary_date", [None])[0]
    if isinstance(diary_date, str):
        legacy_state["diary_date"] = diary_date

    diary_job = params.get("diary_job", [None])[0]
    if isinstance(diary_job, str):
        legacy_state["diary_job"] = diary_job

    merged: dict[str, Any] = {}
    if isinstance(token_state, dict):
        merged.update(token_state)
    merged.update({key: value for key, value in legacy_state.items() if value not in (None, "")})
    return normalize_workspace_state(merged, available_tabs=available_tabs)


def workspace_state_to_query_params(
    state: Mapping[str, Any],
    *,
    available_tabs: Iterable[str],
) -> dict[str, str]:
    normalized = normalize_workspace_state(state, available_tabs=available_tabs)
    params: dict[str, str] = {}
    view = normalized.get("view")
    if not isinstance(view, str):
        return params

    params["view"] = view
    if normalized.get("workflow"):
        params["workflow"] = str(normalized["workflow"])
    if normalized.get("diary_view"):
        params["diary_view"] = str(normalized["diary_view"])
    if normalized.get("diary_date"):
        params["diary_date"] = str(normalized["diary_date"])
    if normalized.get("diary_job"):
        params["diary_job"] = str(normalized["diary_job"])

    params[WORKSPACE_STATE_PARAM] = encode_workspace_state(normalized)
    return params
