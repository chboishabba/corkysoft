from __future__ import annotations

import sqlite3

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from analytics.db import ensure_global_parameters_table, set_parameter_value
from analytics.price_distribution import (
    BREAK_EVEN_KEY,
    create_histogram,
    load_historical_jobs,
    summarise_distribution,
)


def build_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    ensure_global_parameters_table(conn)
    conn.execute(
        """
        CREATE TABLE addresses (
            id INTEGER PRIMARY KEY,
            raw_input TEXT NOT NULL,
            normalized TEXT,
            street_number TEXT,
            street_name TEXT,
            street_type TEXT,
            unit_number TEXT,
            city TEXT,
            state TEXT,
            postcode TEXT,
            country TEXT,
            lon REAL,
            lat REAL,
            UNIQUE(normalized, country)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE historical_jobs (
            id INTEGER PRIMARY KEY,
            job_date TEXT,
            client TEXT,
            origin_address_id INTEGER NOT NULL,
            destination_address_id INTEGER NOT NULL,
            volume_m3 REAL,
            revenue_total REAL,
            FOREIGN KEY(origin_address_id) REFERENCES addresses(id),
            FOREIGN KEY(destination_address_id) REFERENCES addresses(id)
        )
        """
    )
    addresses = [
        (
            1,
            "Brisbane, QLD",
            "BRISBANE QLD",
            None,
            None,
            None,
            None,
            "Brisbane",
            "QLD",
            "4000",
            "Australia",
            153.0260,
            -27.4705,
        ),
        (
            2,
            "Sydney, NSW",
            "SYDNEY NSW",
            None,
            None,
            None,
            None,
            "Sydney",
            "NSW",
            "2000",
            "Australia",
            151.2093,
            -33.8688,
        ),
        (
            3,
            "Melbourne, VIC",
            "MELBOURNE VIC",
            None,
            None,
            None,
            None,
            "Melbourne",
            "VIC",
            "3000",
            "Australia",
            144.9631,
            -37.8136,
        ),
        (
            4,
            "Cairns, QLD",
            "CAIRNS QLD",
            None,
            None,
            None,
            None,
            "Cairns",
            "QLD",
            "4870",
            "Australia",
            145.7700,
            -16.9200,
        ),
    ]
    conn.executemany(
        """
        INSERT INTO addresses (
            id,
            raw_input,
            normalized,
            street_number,
            street_name,
            street_type,
            unit_number,
            city,
            state,
            postcode,
            country,
            lon,
            lat
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        addresses,
    )
    jobs = [
        (1, "2024-01-05", "Acme", 1, 2, 50, 15000),
        (2, "2024-01-18", "Acme", 1, 3, 40, 9000),
        (3, "2024-02-01", "Beta", 4, 2, 30, 6000),
        (4, "2024-02-10", "Gamma", 4, 3, 20, 3200),
    ]
    conn.executemany(
        "INSERT INTO historical_jobs VALUES (?, ?, ?, ?, ?, ?, ?)",
        jobs,
    )
    conn.commit()
    set_parameter_value(conn, BREAK_EVEN_KEY, 250.0, "Test break-even")
    return conn


def test_load_historical_jobs_filters_by_client_and_corridor():
    conn = build_conn()
    try:
        df, mapping = load_historical_jobs(
            conn,
            start_date=pd.Timestamp("2024-01-01"),
            end_date=pd.Timestamp("2024-01-31"),
            clients=["Acme"],
        )
        assert len(df) == 2
        assert mapping.volume == "volume_m3"
        assert np.isclose(df["price_per_m3"].iloc[0], 300.0)

        corridor = df["corridor_display"].iloc[0]
        df_filtered, _ = load_historical_jobs(
            conn,
            corridor=corridor,
        )
        assert df_filtered["corridor_display"].unique().tolist() == [corridor]

        df_postcode, _ = load_historical_jobs(conn, postcode_prefix="48")
        assert set(df_postcode["origin"].unique()) == {"Cairns"}
    finally:
        conn.close()


def test_summarise_distribution_and_histogram():
    conn = build_conn()
    try:
        df, _ = load_historical_jobs(conn)
        summary = summarise_distribution(df, 250.0)
        assert summary.job_count == 4
        assert summary.priced_job_count == 4
        assert round(summary.median, 2) == 262.5
        assert summary.below_break_even_count == 2

        fig = create_histogram(df, 250.0, bins=10)
        # Expect one trace for the histogram and vertical lines as shapes
        assert fig.data[0].type == "histogram"
        band_labels = {ann.get("text") for ann in fig.layout.annotations}
        assert {"Break-even", "+10%", "-10%"}.issubset(band_labels)
    finally:
        conn.close()
