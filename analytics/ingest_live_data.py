"""Command-line entrypoint for updating live telemetry tables."""
from __future__ import annotations

import argparse

from analytics.live_data import run_mock_ingestor


def _parse_route_speed(value: str) -> tuple[int, float]:
    try:
        job_id_str, speed_str = value.split("=", maxsplit=1)
        return int(job_id_str), float(speed_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Route speeds must follow JOB_ID=SPEED_KPH (e.g. 101=72.5)."
        ) from exc


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
    parser.add_argument(
        "--start-date",
        dest="start_date",
        default=None,
        help="Earliest historical job date (YYYY-MM-DD) to include.",
    )
    parser.add_argument(
        "--end-date",
        dest="end_date",
        default=None,
        help="Latest historical job date (YYYY-MM-DD) to include.",
        "--route-speed",
        dest="route_speeds",
        action="append",
        type=_parse_route_speed,
        help="Override a route's average speed using JOB_ID=SPEED_KPH (e.g. 42=68).",
    )
    parser.add_argument(
        "--metro-speed",
        type=float,
        default=50.0,
        help="Default km/h applied to metro-length routes when duration is missing.",
    )
    parser.add_argument(
        "--highway-speed",
        type=float,
        default=90.0,
        help="Default km/h applied to long-haul routes when duration is missing.",
    )
    parser.add_argument(
        "--metro-threshold-km",
        type=float,
        default=100.0,
        help="Distance threshold in km for applying metro defaults.",
    )
    parser.add_argument(
        "--drive-hours",
        type=float,
        default=4.0,
        help="Hours of continuous driving allowed before a rest period is enforced.",
    )
    parser.add_argument(
        "--rest-minutes",
        type=float,
        default=30.0,
        help="Length of each rest period applied during the duty cycle, in minutes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    route_speed_map = {job_id: speed for job_id, speed in args.route_speeds} if args.route_speeds else None
    run_mock_ingestor(
        db_path=args.db_path or None,
        truck_ids=args.trucks,
        interval_seconds=args.interval,
        iterations=args.iterations,
        start_date=args.start_date,
        end_date=args.end_date,
        route_speeds=route_speed_map,
        metro_speed_kph=args.metro_speed,
        highway_speed_kph=args.highway_speed,
        metro_distance_km=args.metro_threshold_km,
        drive_hours=args.drive_hours,
        rest_minutes=args.rest_minutes,
    )


if __name__ == "__main__":
    main()
