"""Utilities for importing MoveWare payloads into SQLite tables."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Iterable, Mapping, Sequence

from analytics.db import (
    ensure_dashboard_tables,
    upsert_container,
    upsert_container_booking,
    upsert_job_by_number,
    upsert_job_segment,
    upsert_job_container_allocation,
    upsert_worker,
)
from analytics.operational_signals import upsert_job_operational_signal


def _first_present(
    record: Mapping[str, object], *keys: str, default: object | None = None
) -> object | None:
    for key in keys:
        if key in record:
            return record[key]
    return default


def _clean_str(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    return str(value).strip() or None


def _coerce_float(value: object | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _coerce_int(value: object | None) -> int | None:
    number = _coerce_float(value)
    if number is None:
        return None
    try:
        return int(number)
    except (TypeError, ValueError):
        return None


def _coerce_timestamp(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    text = _clean_str(value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.isoformat()


def _job_id_from_number(conn, job_number: str) -> int:
    row = conn.execute(
        "SELECT id FROM jobs WHERE job_number = ?", (job_number,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Job with number {job_number} does not exist")
    return int(row[0])


def import_jobs(conn, records: Sequence[Mapping[str, object]], *, dry_run: bool = False) -> tuple[int, int]:
    inserted = 0
    updated = 0
    for record in records:
        job_number = _clean_str(
            _first_present(record, "job_number", "jobNumber", "id")
        )
        if not job_number:
            raise ValueError("MoveWare job record missing job_number")

        existing = conn.execute(
            "SELECT id FROM jobs WHERE job_number = ?", (job_number,)
        ).fetchone()
        if not dry_run:
            upsert_job_by_number(
                conn,
                job_number=job_number,
                job_date=_clean_str(_first_present(record, "job_date", "jobDate")),
                client=_clean_str(record.get("client")),
                client_reference=_clean_str(
                    _first_present(record, "client_reference", "clientReference")
                ),
                origin=_clean_str(record.get("origin")),
                destination=_clean_str(record.get("destination")),
                revenue_total=_coerce_float(_first_present(record, "revenue_total", "revenueTotal")),
                revenue=_coerce_float(record.get("revenue")),
                volume_m3=_coerce_float(_first_present(record, "volume_m3", "volumeM3")),
                volume=_coerce_float(record.get("volume")),
                distance_km=_coerce_float(_first_present(record, "distance_km", "distanceKm")),
                final_cost=_coerce_float(_first_present(record, "final_cost", "finalCost")),
                origin_postcode=_clean_str(record.get("origin_postcode")),
                destination_postcode=_clean_str(record.get("destination_postcode")),
                origin_lat=_coerce_float(record.get("origin_lat")),
                origin_lon=_coerce_float(record.get("origin_lon")),
                dest_lat=_coerce_float(record.get("dest_lat")),
                dest_lon=_coerce_float(record.get("dest_lon")),
                created_at=_coerce_timestamp(
                    _first_present(record, "created_at", "createdAt", "created")
                ),
                updated_at=_coerce_timestamp(
                    _first_present(
                        record, "updated_at", "updatedAt", "lastUpdated", "importedAt"
                    )
                ),
            )
            upsert_job_operational_signal(
                conn,
                job_number=job_number,
                origin=_clean_str(record.get("origin")),
                destination=_clean_str(record.get("destination")),
                estimated_volume_m3=_coerce_float(
                    _first_present(record, "volume_m3", "volumeM3")
                ),
                source="moveware_import",
            )
        if existing is None:
            inserted += 1
        else:
            updated += 1
    return inserted, updated


def import_workers(conn, records: Sequence[Mapping[str, object]], *, dry_run: bool = False) -> tuple[int, int]:
    inserted = 0
    updated = 0
    for record in records:
        name = _clean_str(record.get("name"))
        if not name:
            raise ValueError("Worker record requires a name")
        employee_code = _clean_str(
            _first_present(record, "employee_code", "employeeCode", "staffCode")
        )
        phone = _clean_str(_first_present(record, "phone", "mobile")) or ""
        if employee_code:
            existing = conn.execute(
                "SELECT id FROM workers WHERE employee_code = ?", (employee_code,)
            ).fetchone()
        else:
            existing = conn.execute(
                "SELECT id FROM workers WHERE name = ? AND phone IS ?",
                (name, phone),
            ).fetchone()
        if not dry_run:
            upsert_worker(
                conn,
                employee_code=employee_code,
                name=name,
                role=_clean_str(record.get("role")) or "",
                phone=phone,
                rate=_coerce_float(record.get("rate")),
                tickets=_coerce_int(record.get("tickets")),
                active=bool(record.get("active", True)),
                hired_at=_coerce_timestamp(_first_present(record, "hired_at", "hiredAt")),
                created_at=_coerce_timestamp(_first_present(record, "created_at", "createdAt")),
            )
        if existing is None:
            inserted += 1
        else:
            updated += 1
    return inserted, updated


def import_segments(conn, records: Sequence[Mapping[str, object]], *, dry_run: bool = False) -> tuple[int, int]:
    inserted = 0
    updated = 0
    for record in records:
        sequence = _coerce_int(
            _first_present(record, "segment_sequence", "segmentSequence", "sequence")
        )
        if sequence is None:
            raise ValueError("Segment record missing segment_sequence")
        job_number = _clean_str(_first_present(record, "job_number", "jobNumber"))
        job_id = _coerce_int(record.get("job_id"))
        if job_id is None:
            if not job_number:
                raise ValueError("Segment record must include job_number when job_id is absent")
            job_id = _job_id_from_number(conn, job_number)

        existing = conn.execute(
            "SELECT id FROM job_segments WHERE job_id = ? AND segment_sequence = ?",
            (job_id, sequence),
        ).fetchone()
        if not dry_run:
            upsert_job_segment(
                conn,
                job_id=job_id,
                segment_sequence=sequence,
                from_location=_clean_str(record.get("origin")),
                to_location=_clean_str(record.get("destination")),
                mode=_clean_str(record.get("mode")),
                status=_clean_str(record.get("status")),
                planned_start=_coerce_timestamp(
                    _first_present(record, "planned_start", "plannedStart")
                ),
                planned_end=_coerce_timestamp(
                    _first_present(record, "planned_end", "plannedEnd")
                ),
                actual_start=_coerce_timestamp(
                    _first_present(record, "actual_start", "actualStart")
                ),
                actual_end=_coerce_timestamp(
                    _first_present(record, "actual_end", "actualEnd")
                ),
                distance_km=_coerce_float(_first_present(record, "distance_km", "distanceKm")),
                client_reference=_clean_str(
                    _first_present(record, "client_reference", "clientReference")
                ),
                created_at=_coerce_timestamp(_first_present(record, "created_at", "createdAt")),
                updated_at=_coerce_timestamp(_first_present(record, "updated_at", "updatedAt")),
            )
        if existing is None:
            inserted += 1
        else:
            updated += 1
    return inserted, updated


def import_container_bookings(
    conn, records: Sequence[Mapping[str, object]], *, dry_run: bool = False
) -> tuple[int, int]:
    inserted = 0
    updated = 0
    for record in records:
        booking_reference = _clean_str(
            _first_present(record, "booking_reference", "bookingReference")
        )
        if not booking_reference:
            raise ValueError("Booking record missing booking_reference")
        job_number = _clean_str(_first_present(record, "job_number", "jobNumber"))
        job_id = _coerce_int(record.get("job_id"))
        if job_id is None and job_number:
            job_id = _job_id_from_number(conn, job_number)

        existing = conn.execute(
            "SELECT id FROM container_bookings WHERE booking_reference = ?",
            (booking_reference,),
        ).fetchone()
        if not dry_run:
            upsert_container_booking(
                conn,
                booking_reference=booking_reference,
                job_id=job_id,
                client_reference=_clean_str(
                    _first_present(record, "client_reference", "clientReference")
                ),
                status=_clean_str(record.get("status")),
                created_at=_coerce_timestamp(_first_present(record, "created_at", "createdAt")),
                updated_at=_coerce_timestamp(_first_present(record, "updated_at", "updatedAt")),
            )
        if existing is None:
            inserted += 1
        else:
            updated += 1
    return inserted, updated


def import_containers(
    conn, records: Sequence[Mapping[str, object]], *, dry_run: bool = False
) -> tuple[int, int]:
    inserted = 0
    updated = 0
    for record in records:
        container_number = _clean_str(
            _first_present(record, "container_number", "containerNumber")
        )
        if not container_number:
            raise ValueError("Container record missing container_number")

        booking_reference = _clean_str(
            _first_present(record, "booking_reference", "bookingReference")
        )
        booking_id = _coerce_int(record.get("booking_id"))
        if booking_id is None and booking_reference:
            booking_row = conn.execute(
                "SELECT id FROM container_bookings WHERE booking_reference = ?",
                (booking_reference,),
            ).fetchone()
            booking_id = booking_row[0] if booking_row else None

        job_number = _clean_str(_first_present(record, "job_number", "jobNumber"))
        job_id = _coerce_int(record.get("job_id"))
        if job_id is None and job_number:
            job_id = _job_id_from_number(conn, job_number)

        existing = conn.execute(
            "SELECT id FROM containers WHERE container_number = ?", (container_number,)
        ).fetchone()
        if not dry_run:
            upsert_container(
                conn,
                container_number=container_number,
                booking_id=booking_id,
                job_id=job_id,
                client_reference=_clean_str(
                    _first_present(record, "client_reference", "clientReference")
                ),
                status=_clean_str(record.get("status")),
                created_at=_coerce_timestamp(_first_present(record, "created_at", "createdAt")),
                updated_at=_coerce_timestamp(_first_present(record, "updated_at", "updatedAt")),
            )
        if existing is None:
            inserted += 1
        else:
            updated += 1
    return inserted, updated


def import_allocations(
    conn, records: Sequence[Mapping[str, object]], *, dry_run: bool = False
) -> tuple[int, int]:
    inserted = 0
    updated = 0
    for record in records:
        booking_reference = _clean_str(
            _first_present(record, "booking_reference", "bookingReference")
        )
        booking_id = _coerce_int(record.get("booking_id"))
        if booking_id is None and booking_reference:
            booking_row = conn.execute(
                "SELECT id FROM container_bookings WHERE booking_reference = ?",
                (booking_reference,),
            ).fetchone()
            booking_id = booking_row[0] if booking_row else None

        job_number = _clean_str(_first_present(record, "job_number", "jobNumber"))
        job_id = _coerce_int(record.get("job_id"))
        if job_id is None and job_number:
            job_id = _job_id_from_number(conn, job_number)

        if job_id is None:
            raise ValueError("Allocation record missing job linkage")
        if booking_id is None:
            raise ValueError("Allocation record missing booking linkage")

        segment_sequence = _coerce_int(
            _first_present(record, "segment_sequence", "segmentSequence")
        )
        segment_id = _coerce_int(
            _first_present(record, "segment_id", "segmentId", default=None)
        )
        if segment_id is None and segment_sequence is not None:
            segment_row = conn.execute(
                """
                SELECT id FROM job_segments
                WHERE job_id = ? AND segment_sequence = ?
                """,
                (job_id, segment_sequence),
            ).fetchone()
            segment_id = segment_row[0] if segment_row else None

        existing = conn.execute(
            """
            SELECT 1 FROM job_container_allocations
            WHERE job_id = ? AND booking_id = ? AND (
                (segment_id IS NULL AND ? IS NULL) OR segment_id = ?
            )
            """,
            (job_id, booking_id, segment_id, segment_id),
        ).fetchone()
        if not dry_run:
            upsert_job_container_allocation(
                conn,
                job_id=job_id,
                booking_id=booking_id,
                segment_id=segment_id,
                volume_share=_coerce_float(
                    _first_present(record, "volume_share", "volumeShare")
                ),
                weight_share=_coerce_float(
                    _first_present(record, "weight_share", "weightShare")
                ),
            )
        if existing is None:
            inserted += 1
        else:
            updated += 1
    return inserted, updated


def import_moveware_records(
    conn, resource: str, records: Iterable[Mapping[str, object]], *, dry_run: bool = False
) -> tuple[int, int]:
    """Dispatch MoveWare imports based on the resource path segment."""

    ensure_dashboard_tables(conn)
    normalized = resource.strip().lower()
    record_list = list(records)
    if normalized == "jobs":
        return import_jobs(conn, record_list, dry_run=dry_run)
    if normalized == "workers":
        return import_workers(conn, record_list, dry_run=dry_run)
    if normalized == "segments":
        return import_segments(conn, record_list, dry_run=dry_run)
    if normalized == "bookings":
        return import_container_bookings(conn, record_list, dry_run=dry_run)
    if normalized == "containers":
        return import_containers(conn, record_list, dry_run=dry_run)
    if normalized == "allocations":
        return import_allocations(conn, record_list, dry_run=dry_run)
    raise ValueError(f"Unsupported MoveWare resource '{resource}'")


__all__ = [
    "import_moveware_records",
    "import_jobs",
    "import_workers",
    "import_segments",
    "import_container_bookings",
    "import_containers",
    "import_allocations",
]
