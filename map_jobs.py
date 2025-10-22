"""Generate a Folium map showing saved jobs and their routes.

This script previously only rendered straight-line connections between the
origin and destination points.  It now optionally overlays the actual routed
paths that were retrieved from OpenRouteService.
"""

from __future__ import annotations

import argparse
import sqlite3
from collections.abc import Sequence
from typing import Any, Mapping

from analytics.routes_map import (
    build_job_route_map,
    combine_route_geojson,
    compute_map_center,
    fetch_job_route_rows,
)


DB_PATH = "routes.db"


def build_map(rows: Sequence[Mapping[str, Any]], *, include_actual: bool) -> "folium.Map":
    """Create the Folium map with straight-line and optional actual routes."""

    return build_job_route_map(rows, include_actual=include_actual)


def load_rows(conn: sqlite3.Connection, *, include_actual: bool) -> list[Mapping[str, Any]]:
    """Fetch job rows, optionally including route geometry."""

    return fetch_job_route_rows(conn, include_actual=include_actual)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Folium map of jobs")
    parser.add_argument("--db", default=DB_PATH, help="Path to the SQLite database")
    parser.add_argument("--out", default="routes_map.html", help="Output HTML map path")
    parser.add_argument(
        "--show-actual",
        action="store_true",
        help="Overlay the actual routed paths when geometry is available",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    try:
        rows = load_rows(conn, include_actual=args.show_actual)
    finally:
        conn.close()

    if not rows:
        print("No routes found. Run routes_to_sqlite.py run to calculate them first.")
        return 1

    fmap = build_map(rows, include_actual=args.show_actual)
    fmap.save(args.out)
    print(f"Map saved to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution entry point
    raise SystemExit(main())
