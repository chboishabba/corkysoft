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
        CREATE TABLE historical_jobs (
            id INTEGER PRIMARY KEY,
            job_date TEXT,
            client TEXT,
            origin TEXT,
            destination TEXT,
            origin_postcode TEXT,
            destination_postcode TEXT,
            volume_m3 REAL,
            revenue_total REAL
        )
        """
    )
    jobs = [
        (1, "2024-01-05", "Acme", "Brisbane", "Sydney", "4000", "2000", 50, 15000),
        (2, "2024-01-18", "Acme", "Brisbane", "Melbourne", "4000", "3000", 40, 9000),
        (3, "2024-02-01", "Beta", "Cairns", "Sydney", "4870", "2000", 30, 6000),
        (4, "2024-02-10", "Gamma", "Cairns", "Melbourne", "4870", "3000", 20, 3200),
    ]
    conn.executemany(
        "INSERT INTO historical_jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
