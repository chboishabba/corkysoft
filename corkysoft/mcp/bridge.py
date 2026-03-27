from __future__ import annotations

import json
import sys
from typing import Any

from . import CORKYSOFT_MCP_PROTOCOL_VERSION, CORKYSOFT_MCP_VERSION, build_default_registry


def _to_json(message: dict[str, Any]) -> str:
    return json.dumps(message, ensure_ascii=False, sort_keys=True)


def _registry_envelope() -> dict[str, Any]:
    registry = build_default_registry()
    return {
        "ok": True,
        "tools": [
            {
                "name": spec.name,
                "title": spec.title,
                "description": spec.description,
                "input_schema": spec.input_schema,
                "response_version": spec.response_version,
                "read_only": spec.read_only,
            }
            for spec in registry.list_tools()
        ],
    }


def _call_tool(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(name, str) or not name.strip():
        return {
            "ok": False,
            "error": {"code": "input_error", "message": "tool name is required", "details": {}},
        }

    registry = build_default_registry()
    return registry.invoke(name, payload)


def _health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "corkysoft-mcp-bridge",
        "version": CORKYSOFT_MCP_VERSION,
        "protocol": CORKYSOFT_MCP_PROTOCOL_VERSION,
    }


def run() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            op = request.get("op")
        except (json.JSONDecodeError, AttributeError):
            print(
                _to_json(
                    {
                        "ok": False,
                        "error": {"code": "protocol_error", "message": "invalid json", "details": {}},
                    }
                ),
                flush=True,
            )
            continue

        if op == "health":
            print(_to_json(_health()), flush=True)
            continue
        if op == "list":
            print(_to_json(_registry_envelope()), flush=True)
            continue
        if op == "info":
            print(
                _to_json(
                    {
                        "ok": True,
                        "version": CORKYSOFT_MCP_VERSION,
                        "protocol": CORKYSOFT_MCP_PROTOCOL_VERSION,
                        "tools": len(build_default_registry().list_tools()),
                        "ready": True,
                    }
                ),
                flush=True,
            )
            continue
        if op == "call":
            payload = request.get("payload", {})
            if not isinstance(payload, dict):
                payload = {}
            name = request.get("name", "")
            print(_to_json(_call_tool(name, payload)), flush=True)
            continue

        print(
            _to_json(
                {
                    "ok": False,
                    "error": {"code": "invalid_operation", "message": f"unknown op: {op}", "details": {}},
                }
            ),
            flush=True,
        )

    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(run())
