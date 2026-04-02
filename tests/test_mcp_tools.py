from __future__ import annotations

from corkysoft.mcp.contracts import ToolExecutionError, ToolInputError, error_payload, success_payload
from corkysoft.mcp.tools import get_corkysoft_tools


def test_all_mcp_tools_are_read_only() -> None:
    specs = [spec for spec, _ in get_corkysoft_tools()]
    assert all(spec.read_only for spec in specs)


def test_mcp_tool_names_and_response_version_are_stable() -> None:
    specs = [spec for spec, _ in get_corkysoft_tools()]
    expected_names = {
        "corkysoft.profitability_summary",
        "corkysoft.dispatch_recommendations",
        "corkysoft.operations_diary_summary",
        "corkysoft.quote_guidance_preview",
    }
    assert {spec.name for spec in specs} == expected_names
    assert all(spec.response_version == "result.v1" for spec in specs)


def test_mcp_result_envelope_helpers_match_contract() -> None:
    payload = {"jobCount": 0}
    response = success_payload(payload)
    assert response["ok"] is True
    assert response["result"] == payload


def test_mcp_error_payload_includes_code_and_details() -> None:
    exc = ToolInputError("bad payload", details={"field": "start_date"})
    response = error_payload(exc)
    assert response["ok"] is False
    assert response["error"]["code"] == "input_error"
    assert response["error"]["details"]["field"] == "start_date"


def test_mcp_execution_error_payload_keeps_code() -> None:
    exc = ToolExecutionError("execution failed", details={"stage": "fetch"})
    response = error_payload(exc)
    assert response["ok"] is False
    assert response["error"]["code"] == "execution_error"
    assert response["error"]["details"]["stage"] == "fetch"
