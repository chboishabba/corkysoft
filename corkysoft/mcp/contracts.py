from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


JsonDict = dict[str, Any]
ToolHandler = Callable[[Mapping[str, Any]], JsonDict]
BridgeResponse = dict[str, Any]


CORKYSOFT_MCP_VERSION = "0.1.0"
CORKYSOFT_MCP_PROTOCOL_VERSION = "2026-03-27"


class ToolError(Exception):
    """Base adapter-level tool failure."""

    code: str = "tool_error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: JsonDict | None = None,
    ) -> None:
        super().__init__(message)
        if code:
            self.code = code
        self.details = details or {}
        self.message = message


class ToolInputError(ToolError):
    """Raised when a tool payload is malformed."""

    code = "input_error"


class ToolExecutionError(ToolError):
    """Raised when a tool handler fails during execution."""

    code = "execution_error"


def success_payload(payload: JsonDict) -> BridgeResponse:
    return {
        "ok": True,
        "result": payload,
    }


def error_payload(exc: ToolError) -> BridgeResponse:
    return {
        "ok": False,
        "error": {
            "code": getattr(exc, "code", "tool_error"),
            "message": str(exc),
            "details": getattr(exc, "details", {}),
        },
    }


@dataclass(frozen=True)
class ToolSpec:
    """Static description of a Corkysoft-exposed tool."""

    name: str
    title: str
    description: str
    input_schema: JsonDict
    response_version: str = "result.v1"
    read_only: bool = True
