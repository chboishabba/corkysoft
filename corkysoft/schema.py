"""Database schema helpers for Corkysoft lane pricing."""
from __future__ import annotations

import sqlite3
from typing import Sequence

LANE_BASE_RATES: Sequence[tuple] = (
    ("BNE-METRO", "Brisbane Metro", "Brisbane Metro", 110.0, 190.0, "Metro hourly fallback"),
    ("BNE-SC", "Brisbane", "Sunshine Coast", 120.0, 195.0, "Sunshine Coast corridor"),
    ("BNE-BDB", "Brisbane", "Bundaberg", 145.0, 205.0, "Bundaberg corridor"),
    ("BNE-MKY", "Brisbane", "Mackay", 165.0, 215.0, "Mackay corridor"),
    ("BNE-CNS", "Brisbane", "Cairns", 185.0, 225.0, "Cairns corridor"),
)

MODIFIER_FEES: Sequence[tuple] = (
    ("DIFFICULT_ACCESS", "Difficult access", "access", "flat", 350.0, 1),
    ("PIANO", "Piano handling", "special_item", "flat", 200.0, 1),
    ("TV_CRATE", "TV crate", "special_item", "flat", 90.0, 1),
)

PACKING_RATE_TIERS: Sequence[tuple] = (
    ("PACKING", 0.0, 20.0, 55.0, "Packing up to 20 m³"),
    ("PACKING", 20.0, 40.0, 50.0, "Packing 20-40 m³"),
    ("PACKING", 40.0, None, 45.0, "Packing 40+ m³"),
    ("UNPACKING", 0.0, 20.0, 40.0, "Unpacking up to 20 m³"),
    ("UNPACKING", 20.0, 40.0, 35.0, "Unpacking 20-40 m³"),
    ("UNPACKING", 40.0, None, 30.0, "Unpacking 40+ m³"),
)

SEASONAL_UPLIFTS: Sequence[tuple] = (
    ("PEAK", "Peak season", 10, 1, 12, 31, 0.80, 1),
    ("SHOULDER", "Shoulder season", 1, 1, 5, 31, 0.20, 1),
)

DEPOT_METRO_ZONES: Sequence[tuple] = (
    ("BNE", "Brisbane Depot", -27.5598, 152.9715, 100.0, "QLD South East"),
    ("SC", "Sunshine Coast Depot", -26.6500, 153.0667, 100.0, "Sunshine Coast"),
    ("MKY", "Mackay Depot", -21.1411, 149.1860, 100.0, "Mackay"),
    ("CNS", "Cairns Depot", -16.8858, 145.7493, 100.0, "Far North QLD"),
)


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS lane_base_rates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            corridor_code TEXT NOT NULL UNIQUE,
            origin_region TEXT NOT NULL,
            destination_region TEXT NOT NULL,
            per_m3_rate REAL NOT NULL CHECK(per_m3_rate >= 0),
            metro_hourly_rate REAL NOT NULL CHECK(metro_hourly_rate >= 0),
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS modifier_fees (
            code TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            category TEXT NOT NULL,
            charge_type TEXT NOT NULL CHECK(charge_type IN ('flat', 'per_m3', 'percentage')),
            value REAL NOT NULL,
            active INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0,1))
        );

        CREATE TABLE IF NOT EXISTS packing_rate_tiers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            service_code TEXT NOT NULL,
            min_volume REAL NOT NULL,
            max_volume REAL,
            rate_per_m3 REAL NOT NULL CHECK(rate_per_m3 >= 0),
            description TEXT,
            UNIQUE(service_code, min_volume),
            CHECK(max_volume IS NULL OR max_volume > min_volume)
        );

        CREATE TABLE IF NOT EXISTS seasonal_uplifts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT NOT NULL UNIQUE,
            description TEXT NOT NULL,
            start_month INTEGER NOT NULL CHECK(start_month BETWEEN 1 AND 12),
            start_day INTEGER NOT NULL CHECK(start_day BETWEEN 1 AND 31),
            end_month INTEGER NOT NULL CHECK(end_month BETWEEN 1 AND 12),
            end_day INTEGER NOT NULL CHECK(end_day BETWEEN 1 AND 31),
            uplift_pct REAL NOT NULL,
            active INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0,1))
        );

        CREATE TABLE IF NOT EXISTS depot_metro_zones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            depot_code TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            radius_km REAL NOT NULL CHECK(radius_km > 0),
            region TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_packing_service ON packing_rate_tiers(service_code, min_volume);
        CREATE INDEX IF NOT EXISTS idx_seasonal_active ON seasonal_uplifts(active);
        CREATE INDEX IF NOT EXISTS idx_modifier_active ON modifier_fees(active);
        """
    )


def _seed_data(conn: sqlite3.Connection) -> None:
    conn.executemany(
        """
        INSERT OR IGNORE INTO lane_base_rates
            (corridor_code, origin_region, destination_region, per_m3_rate, metro_hourly_rate, notes)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        LANE_BASE_RATES,
    )

    conn.executemany(
        """
        INSERT OR IGNORE INTO modifier_fees
            (code, description, category, charge_type, value, active)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        MODIFIER_FEES,
    )

    conn.executemany(
        """
        INSERT OR IGNORE INTO packing_rate_tiers
            (service_code, min_volume, max_volume, rate_per_m3, description)
        VALUES (?, ?, ?, ?, ?)
        """,
        PACKING_RATE_TIERS,
    )

    conn.executemany(
        """
        INSERT OR IGNORE INTO seasonal_uplifts
            (code, description, start_month, start_day, end_month, end_day, uplift_pct, active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        SEASONAL_UPLIFTS,
    )

    conn.executemany(
        """
        INSERT OR IGNORE INTO depot_metro_zones
            (depot_code, name, latitude, longitude, radius_km, region)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        DEPOT_METRO_ZONES,
    )


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create tables required for lane pricing and seed default data."""
    _create_tables(conn)
    _seed_data(conn)
    conn.commit()


__all__ = ["ensure_schema"]
