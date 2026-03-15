from __future__ import annotations

import json
import math
import random
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Sequence

from analytics.db.inventory import (
    allocate_inventory_to_segment,
    upsert_inventory_item,
    upsert_inventory_requirement,
)
from analytics.db.schema import ensure_dashboard_tables, ensure_historical_job_routes_table
from analytics.operations_assignment import ensure_segment


@dataclass(frozen=True)
class NamedLocation:
    name: str
    lat: float
    lon: float
    postcode: str
    state: str


@dataclass(frozen=True)
class CorridorTemplate:
    origin: NamedLocation
    destination: NamedLocation
    corridor_label: str


@dataclass(frozen=True)
class SeededJob:
    client: str
    origin: str
    destination: str
    planned_start: str
    planned_end: str
    volume_m3: float
    required_containers: int
    allocated_containers: int
    shortage_containers: int
    revenue_total: float
    final_cost: float
    distance_km: float
    job_id: int
    segment_id: int


_LOCATIONS: tuple[NamedLocation, ...] = (
    NamedLocation("Brisbane Depot", -27.4705, 153.0260, "4000", "QLD"),
    NamedLocation("Sunshine Coast Site", -26.6500, 153.0667, "4551", "QLD"),
    NamedLocation("Gold Coast Storage", -28.0167, 153.4000, "4217", "QLD"),
    NamedLocation("Toowoomba Warehouse", -27.5598, 151.9507, "4350", "QLD"),
    NamedLocation("Ipswich Yard", -27.6146, 152.7609, "4305", "QLD"),
    NamedLocation("Sydney Inner West", -33.8898, 151.1570, "2040", "NSW"),
    NamedLocation("Newcastle Depot", -32.9283, 151.7817, "2300", "NSW"),
    NamedLocation("Wollongong Site", -34.4278, 150.8931, "2500", "NSW"),
    NamedLocation("Melbourne West Yard", -37.8136, 144.9631, "3000", "VIC"),
    NamedLocation("Geelong Storage", -38.1499, 144.3617, "3220", "VIC"),
)

_CORRIDORS: tuple[CorridorTemplate, ...] = (
    CorridorTemplate(_LOCATIONS[0], _LOCATIONS[1], "Brisbane -> Sunshine Coast"),
    CorridorTemplate(_LOCATIONS[0], _LOCATIONS[2], "Brisbane -> Gold Coast"),
    CorridorTemplate(_LOCATIONS[0], _LOCATIONS[3], "Brisbane -> Toowoomba"),
    CorridorTemplate(_LOCATIONS[5], _LOCATIONS[6], "Sydney -> Newcastle"),
    CorridorTemplate(_LOCATIONS[8], _LOCATIONS[9], "Melbourne -> Geelong"),
)

_CLIENTS: tuple[str, ...] = (
    "Acacia Projects",
    "Blue Wattle Estates",
    "Coastal Relocations",
    "Driftwood Hospitality",
    "Harbourline Fitouts",
    "Ironbark Medical",
    "Peak Season Logistics",
    "Redgum Education",
    "Sunline Property Group",
    "Vista Aged Care",
)

_NOTES: tuple[str, ...] = (
    "Container-first move with likely staged unload.",
    "Peak corridor move; monitor spare-capacity overlap.",
    "Remote-site setup requiring early last-mile planning.",
    "High-touch customer with tight delivery window.",
    "Container hire acceptable if stock is exhausted.",
)


def _utc_iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _haversine_km(origin: NamedLocation, destination: NamedLocation) -> float:
    radius_km = 6371.0
    lat1 = math.radians(origin.lat)
    lat2 = math.radians(destination.lat)
    delta_lat = math.radians(destination.lat - origin.lat)
    delta_lon = math.radians(destination.lon - origin.lon)
    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius_km * c


def _route_geojson(origin: NamedLocation, destination: NamedLocation) -> str:
    return json.dumps(
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [origin.lon, origin.lat],
                    [destination.lon, destination.lat],
                ],
            },
            "properties": {
                "origin": origin.name,
                "destination": destination.name,
            },
        }
    )


def _required_containers(volume_m3: float) -> int:
    return max(1, math.ceil(volume_m3 / 8.0))


def _corridor_for_index(index: int) -> CorridorTemplate:
    weighted: Sequence[CorridorTemplate] = (
        _CORRIDORS[0],
        _CORRIDORS[0],
        _CORRIDORS[0],
        _CORRIDORS[1],
        _CORRIDORS[1],
        _CORRIDORS[2],
        _CORRIDORS[2],
        _CORRIDORS[3],
        _CORRIDORS[3],
        _CORRIDORS[4],
    )
    return weighted[index % len(weighted)]


def _ensure_address(
    conn: sqlite3.Connection,
    location: NamedLocation,
) -> int:
    normalized = location.name.strip().lower()
    existing = conn.execute(
        """
        SELECT id
        FROM addresses
        WHERE normalized = ? AND COALESCE(country, 'AU') = 'AU'
        """,
        (normalized,),
    ).fetchone()
    if existing is not None:
        return int(existing[0])

    cursor = conn.execute(
        """
        INSERT INTO addresses (
            raw_input,
            normalized,
            city,
            state,
            postcode,
            country,
            lon,
            lat
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            location.name,
            normalized,
            location.name,
            location.state,
            location.postcode,
            "AU",
            location.lon,
            location.lat,
        ),
    )
    return int(cursor.lastrowid)


def _find_existing_live_job(
    conn: sqlite3.Connection,
    *,
    job_date: str,
    client: str,
    origin: str,
    destination: str,
    revenue_total: float,
    volume_m3: float,
) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT id, route_geojson
        FROM jobs
        WHERE job_date = ?
          AND client = ?
          AND origin = ?
          AND destination = ?
          AND ROUND(COALESCE(revenue_total, 0), 2) = ROUND(?, 2)
          AND ROUND(COALESCE(volume_m3, 0), 1) = ROUND(?, 1)
        LIMIT 1
        """,
        (job_date, client, origin, destination, revenue_total, volume_m3),
    ).fetchone()


def _ensure_historical_job(
    conn: sqlite3.Connection,
    *,
    corridor: CorridorTemplate,
    client: str,
    job_date: str,
    price_per_m3: float,
    revenue_total: float,
    volume_m3: float,
    distance_km: float,
    final_cost: float,
    route_geojson: str,
    updated_at: str,
) -> int:
    origin_address_id = _ensure_address(conn, corridor.origin)
    destination_address_id = _ensure_address(conn, corridor.destination)
    existing = conn.execute(
        """
        SELECT id
        FROM historical_jobs
        WHERE job_date = ?
          AND client = ?
          AND origin = ?
          AND destination = ?
          AND ROUND(COALESCE(revenue_total, 0), 2) = ROUND(?, 2)
          AND ROUND(COALESCE(volume_m3, 0), 1) = ROUND(?, 1)
        LIMIT 1
        """,
        (
            job_date,
            client,
            corridor.origin.name,
            corridor.destination.name,
            revenue_total,
            volume_m3,
        ),
    ).fetchone()

    if existing is None:
        cursor = conn.execute(
            """
            INSERT INTO historical_jobs (
                job_date,
                client,
                corridor_display,
                price_per_m3,
                revenue_total,
                revenue,
                volume_m3,
                volume,
                distance_km,
                final_cost,
                origin,
                destination,
                origin_postcode,
                destination_postcode,
                origin_address_id,
                destination_address_id,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_date,
                client,
                corridor.corridor_label,
                price_per_m3,
                revenue_total,
                revenue_total,
                volume_m3,
                volume_m3,
                distance_km,
                final_cost,
                corridor.origin.name,
                corridor.destination.name,
                corridor.origin.postcode,
                corridor.destination.postcode,
                origin_address_id,
                destination_address_id,
                updated_at,
                updated_at,
            ),
        )
        historical_job_id = int(cursor.lastrowid)
    else:
        historical_job_id = int(existing[0])

    ensure_historical_job_routes_table(conn)
    conn.execute(
        """
        INSERT INTO historical_job_routes (historical_job_id, geojson, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(historical_job_id) DO UPDATE SET
            geojson = excluded.geojson,
            updated_at = excluded.updated_at
        """,
        (historical_job_id, route_geojson, updated_at, updated_at),
    )
    return historical_job_id


def seed_mainland_jobs(
    conn: sqlite3.Connection,
    *,
    count: int = 10,
    seed: int = 20260314,
    baseline_containers: int = 30,
    start_at: datetime | None = None,
) -> list[SeededJob]:
    """Seed realistic mainland-Australia planning jobs plus container requirements."""

    ensure_dashboard_tables(conn)
    rng = random.Random(seed)
    now = (start_at or datetime.now(UTC)).astimezone(UTC)
    container_item = upsert_inventory_item(
        conn,
        name="Standard Container Pod",
        description="Reusable container pod for container-heavy planning scenarios.",
        quantity=baseline_containers,
        unit="container",
        architecture="container",
        state="created",
        custody_location_type="depot",
        custody_location_ref="main_depot",
        custody_location_label="Main depot",
    )
    available_containers = baseline_containers
    created: list[SeededJob] = []

    for index in range(count):
        corridor = _corridor_for_index(index)
        client = _CLIENTS[index % len(_CLIENTS)]
        volume_m3 = round(rng.uniform(8.0, 28.0), 1)
        required_containers = _required_containers(volume_m3)
        allocated_containers = min(required_containers, available_containers)
        available_containers -= allocated_containers
        shortage_containers = required_containers - allocated_containers

        start_offset_days = index % 7
        start_hour = 7 + (index % 4) * 2
        planned_start = now + timedelta(days=start_offset_days, hours=start_hour)
        planned_end = planned_start + timedelta(hours=3 + (index % 3))

        distance_km = round(_haversine_km(corridor.origin, corridor.destination), 1)
        base_cost = 420.0 + (distance_km * 2.05) + (volume_m3 * 31.0)
        margin_multiplier = 1.15 + (0.02 * (index % 5))
        revenue_total = round(base_cost * margin_multiplier, 2)
        price_per_m3 = round(revenue_total / volume_m3, 2)
        final_cost = round(base_cost, 2)
        route_geojson = _route_geojson(corridor.origin, corridor.destination)
        updated_at = _utc_iso(now)
        existing_job = _find_existing_live_job(
            conn,
            job_date=planned_start.date().isoformat(),
            client=client,
            origin=corridor.origin.name,
            destination=corridor.destination.name,
            revenue_total=revenue_total,
            volume_m3=volume_m3,
        )
        if existing_job is None:
            cursor = conn.execute(
                """
                INSERT INTO jobs (
                    job_date,
                    client,
                    origin,
                    destination,
                    origin_resolved,
                    destination_resolved,
                    price_per_m3,
                    revenue_total,
                    revenue,
                    volume_m3,
                    volume,
                    distance_km,
                    final_cost,
                    origin_postcode,
                    destination_postcode,
                    origin_lat,
                    origin_lon,
                    dest_lat,
                    dest_lon,
                    route_geojson,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    planned_start.date().isoformat(),
                    client,
                    corridor.origin.name,
                    corridor.destination.name,
                    corridor.origin.name,
                    corridor.destination.name,
                    price_per_m3,
                    revenue_total,
                    revenue_total,
                    volume_m3,
                    volume_m3,
                    distance_km,
                    final_cost,
                    corridor.origin.postcode,
                    corridor.destination.postcode,
                    corridor.origin.lat,
                    corridor.origin.lon,
                    corridor.destination.lat,
                    corridor.destination.lon,
                    route_geojson,
                    updated_at,
                ),
            )
            job_id = int(cursor.lastrowid)
        else:
            job_id = int(existing_job["id"])
            if not existing_job["route_geojson"]:
                conn.execute(
                    "UPDATE jobs SET route_geojson = ?, updated_at = ? WHERE id = ?",
                    (route_geojson, updated_at, job_id),
                )
        _ensure_historical_job(
            conn,
            corridor=corridor,
            client=client,
            job_date=planned_start.date().isoformat(),
            price_per_m3=price_per_m3,
            revenue_total=revenue_total,
            volume_m3=volume_m3,
            distance_km=distance_km,
            final_cost=final_cost,
            route_geojson=route_geojson,
            updated_at=updated_at,
        )
        conn.commit()

        segment = ensure_segment(
            conn,
            job_id=job_id,
            segment_sequence=1,
            from_location=corridor.origin.name,
            to_location=corridor.destination.name,
            planned_start=_utc_iso(planned_start),
            planned_end=_utc_iso(planned_end),
        )
        requirement = upsert_inventory_requirement(
            conn,
            job_id=job_id,
            segment_id=int(segment["id"]),
            inventory_item_id=int(container_item["id"]),
            requirement_name="Standard Container Pod",
            required_quantity=float(required_containers),
            substitution_allowed=True,
            architecture="container",
            notes=_NOTES[index % len(_NOTES)],
        )
        if allocated_containers > 0:
            allocate_inventory_to_segment(
                conn,
                segment_id=int(segment["id"]),
                inventory_item_id=int(container_item["id"]),
                quantity=float(allocated_containers),
                status="planned",
            )
        created.append(
            SeededJob(
                client=client,
                origin=corridor.origin.name,
                destination=corridor.destination.name,
                planned_start=_utc_iso(planned_start),
                planned_end=_utc_iso(planned_end),
                volume_m3=volume_m3,
                required_containers=required_containers,
                allocated_containers=allocated_containers,
                shortage_containers=shortage_containers,
                revenue_total=revenue_total,
                final_cost=final_cost,
                distance_km=distance_km,
                job_id=job_id,
                segment_id=int(segment["id"]),
            )
        )

    return created
