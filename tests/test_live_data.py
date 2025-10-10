from __future__ import annotations

import sqlite3

import pandas as pd

from analytics.live_data import MockTelemetryIngestor, load_active_routes, load_truck_positions
from analytics.price_distribution import (
    PROFITABILITY_COLOURS,
    classify_profit_band,
    prepare_route_map_data,
)
from corkysoft.schema import ensure_schema


def _build_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS addresses (
            id INTEGER PRIMARY KEY,
            raw_input TEXT NOT NULL,
            normalized TEXT,
            city TEXT,
            state TEXT,
            postcode TEXT,
            country TEXT,
            lon REAL,
            lat REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS historical_jobs (
            id INTEGER PRIMARY KEY,
            job_date TEXT,
            client TEXT,
            origin_address_id INTEGER,
            destination_address_id INTEGER,
            volume_m3 REAL,
            revenue_total REAL,
            price_per_m3 REAL,
            FOREIGN KEY(origin_address_id) REFERENCES addresses(id),
            FOREIGN KEY(destination_address_id) REFERENCES addresses(id)
        )
        """
    )
    addresses = [
        (1, "Brisbane", "BRISBANE", "Brisbane", "QLD", "4000", "Australia", 153.0260, -27.4705),
        (2, "Sydney", "SYDNEY", "Sydney", "NSW", "2000", "Australia", 151.2093, -33.8688),
        (3, "Melbourne", "MELBOURNE", "Melbourne", "VIC", "3000", "Australia", 144.9631, -37.8136),
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO addresses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        addresses,
    )
    jobs = [
        (1, "2024-01-01", "Acme", 1, 2, 40.0, 12000.0, 300.0),
        (2, "2024-01-02", "Bravo", 1, 3, 30.0, 9000.0, 280.0),
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO historical_jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        jobs,
    )
    conn.commit()
    return conn


def test_mock_ingestor_populates_live_tables():
    conn = _build_conn()
    try:
        ingestor = MockTelemetryIngestor(conn, truck_ids=("TRK-1", "TRK-2"))
        ingestor.run_cycle()
        ingestor.run_cycle()

        trucks_df = load_truck_positions(conn)
        routes_df = load_active_routes(conn)

        assert not trucks_df.empty
        assert set(trucks_df["truck_id"]) == {"TRK-1", "TRK-2"}
        assert "lat" in trucks_df.columns and "lon" in trucks_df.columns
        assert not routes_df.empty
        assert set(routes_df["truck_id"]) <= {"TRK-1", "TRK-2"}

        base_df = routes_df.copy()
        base_df["id"] = base_df["job_id"]
        base_df["price_per_m3"] = 300.0
        base_df["corridor_display"] = "Test Corridor"
        mapped = prepare_route_map_data(base_df, break_even=250.0)

        assert "colour" in mapped.columns
        assert all(isinstance(colour, list) for colour in mapped["colour"])
    finally:
        conn.close()


def test_classify_profit_band_edges():
    assert classify_profit_band(None, 250.0) == "Unknown"
    assert classify_profit_band(200.0, 250.0) == "Below break-even"
    assert classify_profit_band(260.0, 250.0) == "0-50 above break-even"
    assert classify_profit_band(320.0, 250.0) == "50-100 above break-even"
    assert classify_profit_band(370.0, 250.0) == "100+ above break-even"
    assert PROFITABILITY_COLOURS["Below break-even"]
