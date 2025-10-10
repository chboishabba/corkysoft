from __future__ import annotations

import sqlite3

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from analytics.db import ensure_global_parameters_table, set_parameter_value
from analytics.price_distribution import (
    BREAK_EVEN_KEY,
    create_histogram,
    create_m3_margin_figure,
    create_m3_vs_km_figure,
    load_historical_jobs,
    prepare_route_map_data,
    summarise_distribution,
    summarise_profitability,
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
            origin_address_id INTEGER,
            destination_address_id INTEGER,
            volume_m3 REAL,
            revenue_total REAL,
            distance_km REAL,
            final_cost REAL,
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
        (1, "2024-01-05", "Acme", 1, 2, 50, 15000, 920.0, 11800),
        (2, "2024-01-18", "Acme", 1, 3, 40, 9000, 1680.0, 7200),
        (3, "2024-02-01", "Beta", 4, 2, 30, 6000, 2400.0, 5100),
        (4, "2024-02-10", "Gamma", 4, 3, 20, 3200, 2700.0, 3600),
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
        assert "revenue_per_km" in df.columns
        assert np.isclose(df["revenue_per_km"].iloc[0], 15000 / 920.0)
        assert "final_cost_per_m3" in df.columns
        assert np.isclose(df["final_cost_per_m3"].iloc[0], 11800 / 50)
        assert np.isclose(df["margin_per_m3"].iloc[0], 300.0 - (11800 / 50))

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
        assert pytest.approx(summary.median, rel=1e-6) == 212.5
        assert summary.below_break_even_count == 3
        assert pytest.approx(summary.mean, rel=1e-6) == 221.25
        assert summary.std_dev == pytest.approx(58.93145736079048, rel=1e-6)
        assert summary.kurtosis == pytest.approx(1.0750900809640118, rel=1e-6)
        assert summary.skewness == pytest.approx(0.8112380153711554, rel=1e-6)

        fig = create_histogram(df, 250.0, bins=10)
        # Expect one trace for the histogram and vertical lines as shapes
        assert fig.data[0].type == "histogram"
        assert any(trace.type == "scatter" for trace in fig.data[1:])
        band_labels = {getattr(ann, "text", None) for ann in fig.layout.annotations}
        assert {"Break-even", "+10%", "-10%"}.issubset(band_labels)
        assert any("kurtosis" in getattr(ann, "text", "") for ann in fig.layout.annotations)
        band_labels.discard(None)
        assert {"Break-even", "+10%", "-10%"}.issubset(band_labels)
        assert any("kurtosis" in (getattr(ann, "text", "") or "") for ann in fig.layout.annotations)
    finally:
        conn.close()


def test_profitability_summary_and_views():
    conn = build_conn()
    try:
        df, _ = load_historical_jobs(conn)
        profitability = summarise_profitability(df)
        assert profitability.revenue_per_km_median == pytest.approx(3.9285714285714284, rel=1e-6)
        assert profitability.revenue_per_km_mean == pytest.approx(6.3366689671037495, rel=1e-6)
        assert profitability.margin_per_m3_median == pytest.approx(37.5, rel=1e-6)
        assert profitability.margin_per_m3_pct_median == pytest.approx(0.21323529411764708, rel=1e-6)
        assert profitability.margin_total_median == pytest.approx(1350.0, rel=1e-6)
        assert profitability.margin_total_pct_median == pytest.approx(0.21323529411764708, rel=1e-6)

        km_fig = create_m3_vs_km_figure(df)
        assert any(trace.type == "scatter" for trace in km_fig.data)
        assert km_fig.layout.title.text.startswith("m³ vs km")

        margin_fig = create_m3_margin_figure(df)
        marker_traces = [trace for trace in margin_fig.data if getattr(trace, "mode", "").startswith("markers")]
        line_traces = [trace for trace in margin_fig.data if getattr(trace, "mode", "") == "lines"]
        assert marker_traces
        assert line_traces
    finally:
        conn.close()


def test_prepare_route_map_data_filters_missing_coordinates():
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "origin_lat": [-27.0, None],
            "origin_lon": [153.0, 150.0],
            "dest_lat": [-33.0, -35.0],
            "dest_lon": [151.0, None],
        }
    )

    result = prepare_route_map_data(df, "id")
    assert len(result) == 1
    assert result.iloc[0]["id"] == 1
    assert result.iloc[0]["map_colour_value"] == "1"


def test_prepare_route_map_data_missing_columns_raise():
    df = pd.DataFrame({"id": [1]})

    with pytest.raises(KeyError):
        prepare_route_map_data(df, "missing")

    with pytest.raises(KeyError):
        prepare_route_map_data(df, "id")
