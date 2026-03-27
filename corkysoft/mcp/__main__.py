from __future__ import annotations

import sys

from .bridge import run as run_bridge
from .server import main as run_fastmcp


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--bridge" in args:
        return run_bridge()
    run_fastmcp()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
