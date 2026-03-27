from __future__ import annotations

import sys

from .bridge import run as run_bridge
from .server import main as run_fastmcp


def _usage() -> str:
    return (
        "Usage: python -m corkysoft.mcp [--bridge|--server|--help]\n"
        "\n"
        "Default: run the supported local JSON bridge.\n"
        "  --bridge  Run the JSON-line bridge explicitly.\n"
        "  --server  Run the optional FastMCP stdio server.\n"
        "  --help    Show this help text.\n"
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--help" in args or "-h" in args:
        print(_usage(), end="")
        return 0
    if "--bridge" in args and "--server" in args:
        print("Choose only one transport: --bridge or --server.", file=sys.stderr)
        return 2
    if "--server" in args:
        run_fastmcp()
        return 0
    if "--bridge" in args:
        return run_bridge()
    return run_bridge()


if __name__ == "__main__":
    raise SystemExit(main())
