from __future__ import annotations

import sqlite3

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from analytics.db import ensure_global_parameters_table, set_parameter_value
from analytics.price_distribution import (
    BREAK_EVEN_KEY,
    aggregate_corridor_performance,
    build_heatmap_source,
    create_histogram,
    create_metro_profitability_figure,
    create_m3_margin_figure,
    create_m3_vs_km_figure,
    filter_metro_jobs,
    load_historical_jobs,
    prepare_profitability_route_data,
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


@pytest.fixture()
def metro_profitability_df():
    return pd.DataFrame(
        {
            "price_per_m3": [280.0, 320.0, 260.0, 310.0],
            "revenue_per_km": [8.5, 7.2, 6.8, 9.1],
            "margin_per_m3": [45.0, 55.0, 28.0, 60.0],
            "margin_per_m3_pct": [0.19, 0.21, 0.12, 0.24],
            "final_cost_per_m3": [235.0, 265.0, 232.0, 250.0],
            "distance_km": [45.0, 180.0, 75.0, 95.0],
            "client_display": ["Acme", "Acme", "Beta", "Delta"],
            "corridor_display": [
                "BNE-SYD",
                "SYD-MEL",
                "BNE-IPS",
                "BNE-GC",
            ],
        }
    )


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


def test_filter_metro_jobs_handles_missing_distances(metro_profitability_df):
    df = metro_profitability_df.copy()
    df.loc[len(df)] = {
        "price_per_m3": 300.0,
        "revenue_per_km": 8.0,
        "margin_per_m3": 40.0,
        "margin_per_m3_pct": 0.18,
        "final_cost_per_m3": 260.0,
        "distance_km": np.nan,
        "client_display": "Zeta",
        "corridor_display": "MEL-CBD",
    }

    filtered = filter_metro_jobs(df, max_distance_km=100.0)
    assert set(filtered["distance_km"].tolist()) == {45.0, 75.0, 95.0}
    # Ensure original frame was not modified
    assert len(df) == 5


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


def test_create_metro_profitability_figure_has_multiple_traces(metro_profitability_df):
    fig = create_metro_profitability_figure(metro_profitability_df, max_distance_km=120.0)
    assert len(fig.data) >= 2
    assert any(trace.type == "scatter" for trace in fig.data)
    assert any(trace.type == "histogram" for trace in fig.data)


def test_prepare_profitability_route_data_tags_profitability():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "origin_lat": [-27.4705, -27.4705, -27.4705],
            "origin_lon": [153.0260, 153.0260, 153.0260],
            "dest_lat": [-33.8688, -33.8688, -33.8688],
            "dest_lon": [151.2093, 151.2093, 151.2093],
            "price_per_m3": [240.0, 252.0, 310.0],
            "corridor_display": ["BNE-SYD", "BNE-SYD", "BNE-SYD"],
        }
    )

    result = prepare_profitability_route_data(df, break_even=250.0)
    statuses = result.set_index("id")["profitability_status"].to_dict()

    assert statuses[1] == "Loss-leading"
    assert statuses[2] == "Break-even"
    assert statuses[3] == "Profitable"
    assert all(
        any(keyword in tooltip for keyword in ("Break-even", "Loss-leading", "Profitable"))
        for tooltip in result["tooltip"]
    )


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


def test_build_heatmap_source_counts_and_weights():
    df = pd.DataFrame(
        {
            "origin_lat": [-27.4705, -16.92],
            "origin_lon": [153.026, 145.77],
            "dest_lat": [-33.8688, -37.8136],
            "dest_lon": [151.2093, 144.9631],
            "distance_km": [50.0, 240.0],
            "volume_m3": [12.5, 20.0],
            "margin_total": [2500.0, -500.0],
        }
    )

    result = build_heatmap_source(df)
    assert set(result.columns) == {"lat", "lon", "weight"}
    assert len(result) == 4
    assert result["weight"].eq(1.0).all()

    margin_result = build_heatmap_source(df, weight_column="margin_total")
    assert len(margin_result) == 4
    assert set(margin_result["weight"].round(2)) == {2500.0, -500.0}


def test_build_heatmap_source_metro_and_volume_weighting():
    df = pd.DataFrame(
        {
            "origin_lat": [-27.4705, -16.92, -34.9285],
            "origin_lon": [153.026, 145.77, 138.6007],
            "dest_lat": [-33.8688, -37.8136, -31.9535],
            "dest_lon": [151.2093, 144.9631, 115.857],
            "distance_km": [85.0, 150.0, 95.0],
            "volume_m3": [10.0, 30.0, 5.0],
        }
    )

    volume_result = build_heatmap_source(df, weight_column="volume_m3", metro_only=True)
    assert len(volume_result) == 4
    assert set(volume_result["weight"].round(2)) == {10.0, 5.0}
    assert volume_result["lat"].isin([-27.4705, -33.8688, -34.9285, -31.9535]).all()

    with pytest.raises(KeyError):
        build_heatmap_source(df, weight_column="unknown_metric")

    df_no_distance = df.drop(columns=["distance_km"])
    with pytest.raises(KeyError):
        build_heatmap_source(df_no_distance, metro_only=True)


def test_prepare_route_map_data_missing_columns_raise():
    df = pd.DataFrame({"id": [1]})

    with pytest.raises(KeyError):
        prepare_route_map_data(df, "missing")

    with pytest.raises(KeyError):
        prepare_route_map_data(df, "id")


def test_aggregate_corridor_performance_combines_bidirectional_lanes():
    df = pd.DataFrame(
        {
            "origin": ["Brisbane", "Melbourne", "Brisbane"],
            "destination": ["Melbourne", "Brisbane", "Sydney"],
            "price_per_m3": [300.0, 280.0, 240.0],
            "volume_m3": [20.0, 10.0, 12.0],
            "revenue_total": [6000.0, 2800.0, 2880.0],
            "distance_km": [1700.0, 1700.0, 900.0],
            "margin_per_m3": [50.0, 30.0, 10.0],
            "margin_total": [1000.0, 300.0, 120.0],
            "margin_total_pct": [0.20, 0.12, 0.05],
            "revenue_per_km": [3.53, 1.65, 3.20],
        }
    )

    aggregated = aggregate_corridor_performance(df, break_even=250.0)
    assert set(aggregated["corridor_pair"]) == {
        "Brisbane ↔ Melbourne",
        "Brisbane ↔ Sydney",
    }

    bne_mel = aggregated.loc[
        aggregated["corridor_pair"] == "Brisbane ↔ Melbourne"
    ].iloc[0]
    assert bne_mel["job_count"] == 2
    assert pytest.approx(bne_mel["share_of_jobs"], rel=1e-6) == 2 / 3
    assert pytest.approx(bne_mel["weighted_price_per_m3"], rel=1e-6) == (
        6000.0 + 2800.0
    ) / (20.0 + 10.0)
    assert pytest.approx(bne_mel["below_break_even_ratio"], rel=1e-6) == 0.0
    assert pytest.approx(bne_mel["median_price_per_m3"], rel=1e-6) == 290.0
    assert pytest.approx(bne_mel["share_of_volume"], rel=1e-6) == (20.0 + 10.0) / (
        20.0 + 10.0 + 12.0
    )
    assert pytest.approx(bne_mel["margin_per_m3_median"], rel=1e-6) == 40.0
    assert pytest.approx(bne_mel["margin_total_sum"], rel=1e-6) == 1300.0
    assert pytest.approx(bne_mel["margin_total_pct_median"], rel=1e-6) == 0.16
    assert bne_mel["revenue_per_km_median"] == pytest.approx(
        (3.53 + 1.65) / 2,
        rel=1e-6,
    )

    bne_syd = aggregated.loc[
        aggregated["corridor_pair"] == "Brisbane ↔ Sydney"
    ].iloc[0]
    assert bne_syd["job_count"] == 1
    assert pytest.approx(bne_syd["below_break_even_ratio"], rel=1e-6) == 1.0
    assert pytest.approx(bne_syd["weighted_price_per_m3"], rel=1e-6) == 240.0


def test_aggregate_corridor_performance_handles_missing_columns():
    df = pd.DataFrame(
        {
            "corridor_display": ["BNE → MEL", "MEL → BNE", "BNE-SYD"],
            "price_per_m3": [250.0, 260.0, 240.0],
        }
    )

    aggregated = aggregate_corridor_performance(df, break_even=255.0)
    assert set(aggregated["corridor_pair"]) == {"BNE ↔ MEL", "BNE ↔ SYD"}

    bne_mel = aggregated.loc[aggregated["corridor_pair"] == "BNE ↔ MEL"].iloc[0]
    assert bne_mel["job_count"] == 2
    assert pytest.approx(bne_mel["share_of_jobs"], rel=1e-6) == 2 / 3
    assert pytest.approx(bne_mel["below_break_even_ratio"], rel=1e-6) == 0.5
