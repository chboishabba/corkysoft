"""Command-line entrypoint for updating live telemetry tables."""
from __future__ import annotations

import argparse

from analytics.live_data import run_mock_ingestor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest mock truck telemetry into the Corkysoft database.",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Path to the SQLite database (defaults to CORKYSOFT_DB / ROUTES_DB / routes.db).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Seconds between telemetry updates.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of update cycles to run (defaults to looping forever).",
    )
    parser.add_argument(
        "--trucks",
        nargs="*",
        default=None,
        help="Optional list of truck IDs to simulate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_mock_ingestor(
        db_path=args.db_path or None,
        truck_ids=args.trucks,
        interval_seconds=args.interval,
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
