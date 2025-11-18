"""Command-line entrypoint for ingesting real GPS telemetry."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from itertools import chain
from typing import IO, Iterable, Iterator, Optional

from analytics.db import connection_scope
from analytics.live_data import TruckGpsSnapshot, TruckTelemetryHarness


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        stringified = str(value).strip()
    except Exception:
        return None
    if not stringified:
        return None
    try:
        return float(stringified)
    except ValueError:
        return None


def _parse_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        stringified = str(value).strip()
    except Exception:
        return None
    if not stringified:
        return None
    try:
        return int(stringified)
    except ValueError:
        return None


def _detect_format(path: Optional[str], first_line: str, forced: Optional[str]) -> str:
    if forced:
        return forced
    if path and path.lower().endswith(".csv"):
        return "csv"
    stripped = first_line.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        return "json"
    return "csv"


def _iter_records(stream: IO[str], fmt: str, first_line: str) -> Iterator[dict]:
    iterator = chain([first_line], stream)
    if fmt == "csv":
        reader = csv.DictReader(iterator)
        for row in reader:
            yield row
    else:
        for line in iterator:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError:
                continue


def _to_snapshot(record: dict) -> Optional[TruckGpsSnapshot]:
    try:
        truck_id = record["truck_id"]
        lat = float(record["lat"])
        lon = float(record["lon"])
        status = record.get("status") or "en_route"
    except (KeyError, TypeError, ValueError):
        return None

    recorded_at = _parse_datetime(record.get("recorded_at")) or datetime.now(UTC)

    return TruckGpsSnapshot(
        truck_id=str(truck_id),
        lat=float(lat),
        lon=float(lon),
        status=str(status),
        recorded_at=recorded_at,
        heading=_parse_float(record.get("heading")),
        speed_kph=_parse_float(record.get("speed_kph")),
        job_id=_parse_int(record.get("job_id")),
        eta=_parse_datetime(record.get("eta")),
        notes=record.get("notes"),
        progress=_parse_float(record.get("progress")),
        travel_seconds=_parse_float(record.get("travel_seconds")),
        origin_lat=_parse_float(record.get("origin_lat")),
        origin_lon=_parse_float(record.get("origin_lon")),
        dest_lat=_parse_float(record.get("dest_lat")),
        dest_lon=_parse_float(record.get("dest_lon")),
        route_geometry=record.get("route_geometry"),
        started_at=_parse_datetime(record.get("started_at")),
    )


def _batched(items: Iterable[TruckGpsSnapshot], size: int) -> Iterator[list[TruckGpsSnapshot]]:
    batch: list[TruckGpsSnapshot] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest real GPS snapshots into the Corkysoft database.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Optional path to NDJSON or CSV input (defaults to stdin).",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Path to the SQLite database (defaults to CORKYSOFT_DB / ROUTES_DB / routes.db).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of batches to process before exiting.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of snapshots to ingest per batch.",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        dest="input_format",
        help="Force an input format instead of auto-detection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stream: IO[str]
    if args.path:
        stream = open(args.path, "r", encoding="utf-8")
    else:
        stream = sys.stdin

    with stream:
        first_line = stream.readline()
        if not first_line:
            return
        fmt = _detect_format(args.path, first_line, args.input_format)
        records = _iter_records(stream, fmt, first_line)
        snapshots = (snap for snap in (_to_snapshot(record) for record in records) if snap)

        with connection_scope(args.db_path or None) as conn:
            harness = TruckTelemetryHarness(conn)
            batches_processed = 0
            for batch in _batched(snapshots, max(1, args.batch_size)):
                harness.ingest(batch)
                batches_processed += 1
                if args.iterations is not None and batches_processed >= args.iterations:
                    break


if __name__ == "__main__":
    main()
