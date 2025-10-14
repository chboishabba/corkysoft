from __future__ import annotations

import sqlite3
import warnings

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from analytics.db import ensure_global_parameters_table, set_parameter_value
from analytics.price_distribution import (
    BREAK_EVEN_KEY,
    DRIVER_COST_KEY,
    FUEL_COST_KEY,
    MAINTENANCE_COST_KEY,
    OVERHEAD_COST_KEY,
    aggregate_corridor_performance,
    build_heatmap_source,
    create_histogram,
    create_metro_profitability_figure,
    create_m3_margin_figure,
    create_m3_vs_km_figure,
    build_profitability_export,
    filter_metro_jobs,
    load_historical_jobs,
    load_live_jobs,
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
    conn.execute(
        """
        CREATE TABLE jobs (
            id INTEGER PRIMARY KEY,
            job_date TEXT,
            client TEXT,
            origin TEXT,
            destination TEXT,
            volume_m3 REAL,
            revenue_total REAL,
            distance_km REAL,
            final_cost REAL,
            origin_postcode TEXT,
            destination_postcode TEXT,
            origin_lat REAL,
            origin_lon REAL,
            dest_lat REAL,
            dest_lon REAL,
            updated_at TEXT
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
    live_jobs = [
        (
            101,
            "2024-02-12",
            "Acme",
            "Brisbane",
            "Sydney",
            48.0,
            15600.0,
            920.0,
            12100.0,
            "4000",
            "2000",
            -27.4705,
            153.0260,
            -33.8688,
            151.2093,
            "2024-02-15T03:15:00Z",
        ),
        (
            102,
            "2024-02-20",
            "Beta",
            "Cairns",
            "Sydney",
            30.0,
            8700.0,
            2400.0,
            6000.0,
            "4870",
            "2000",
            -16.9200,
            145.7700,
            -33.8688,
            151.2093,
            "2024-02-21T08:45:00Z",
        ),
        (
            103,
            "2024-03-05",
            "Delta",
            "Brisbane",
            "Melbourne",
            38.0,
            11875.0,
            1680.0,
            9100.0,
            "4000",
            "3000",
            -27.4705,
            153.0260,
            -37.8136,
            144.9631,
            "2024-03-06T11:30:00Z",
        ),
    ]
    conn.executemany(
        """
        INSERT INTO jobs (
            id, job_date, client, origin, destination, volume_m3, revenue_total,
            distance_km, final_cost, origin_postcode, destination_postcode,
            origin_lat, origin_lon, dest_lat, dest_lon, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        live_jobs,
    )
    conn.commit()
    set_parameter_value(conn, FUEL_COST_KEY, 1.0, "Test fuel per km")
    set_parameter_value(conn, DRIVER_COST_KEY, 4.0, "Test driver per km")
    set_parameter_value(conn, MAINTENANCE_COST_KEY, 0.5, "Test maintenance per km")
    set_parameter_value(conn, OVERHEAD_COST_KEY, 2000.0, "Test overhead per job")
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

        assert np.isclose(df["break_even_total"].iloc[0], 7060.0)
        assert np.isclose(df["break_even_per_m3"].iloc[0], 141.2)
        assert np.isclose(df["margin_vs_break_even"].iloc[0], 158.8)
        assert (df["margin_vs_break_even"].iloc[1]) < 0

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


def test_load_live_jobs_supports_filters():
    conn = build_conn()
    try:
        df, mapping = load_live_jobs(
            conn,
            start_date=pd.Timestamp("2024-02-01"),
            end_date=pd.Timestamp("2024-02-28"),
            clients=["Acme"],
            corridor="Brisbane → Sydney",
        )
        assert not df.empty
        assert mapping.volume == "volume_m3"
        assert set(df["client_display"]) == {"Acme"}
        assert all(df["corridor_display"] == "Brisbane → Sydney")
        assert "price_per_m3" in df.columns
        assert "break_even_per_m3" in df.columns

        df_postcode, _ = load_live_jobs(conn, postcode_prefix="48")
        assert set(df_postcode["client_display"].unique()) == {"Beta"}
    finally:
        conn.close()


def test_load_historical_jobs_handles_duplicate_columns():
    conn = build_conn()
    try:
        conn.execute("ALTER TABLE historical_jobs ADD COLUMN origin TEXT")
        conn.execute("ALTER TABLE historical_jobs ADD COLUMN destination TEXT")
        conn.execute("ALTER TABLE historical_jobs ADD COLUMN origin_postcode TEXT")
        conn.execute(
            "ALTER TABLE historical_jobs ADD COLUMN destination_postcode TEXT"
        )
        conn.execute(
            "UPDATE historical_jobs SET origin = 'Legacy', destination = 'Legacy'"
        )
        conn.commit()

        df, mapping = load_historical_jobs(conn)
        assert not df.empty
        assert mapping.origin == "origin"
        assert mapping.destination == "destination"
        assert "corridor_display" in df.columns
        assert df["corridor_display"].notna().all()
        assert "Legacy" not in set(df["origin"].astype(str))
    finally:
        conn.close()


def test_load_historical_jobs_parses_dayfirst_dates_without_warnings():
    conn = build_conn()
    try:
        dayfirst_dates = {
            1: "05/01/2024",
            2: "18/01/2024",
            3: "01/02/2024",
            4: "10/02/2024",
        }
        for job_id, date_str in dayfirst_dates.items():
            conn.execute(
                "UPDATE historical_jobs SET job_date = ? WHERE id = ?",
                (date_str, job_id),
            )
        conn.commit()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            df, mapping = load_historical_jobs(conn)
        assert caught == []

        assert mapping.date == "job_date"
        expected_dates = [
            pd.Timestamp("2024-01-05"),
            pd.Timestamp("2024-01-18"),
            pd.Timestamp("2024-02-01"),
            pd.Timestamp("2024-02-10"),
        ]
        assert df[mapping.date].tolist() == expected_dates
        assert str(df[mapping.date].dtype).startswith("datetime64[ns]")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            filtered_df, filtered_mapping = load_historical_jobs(
                conn, start_date=pd.Timestamp("2024-02-01")
            )
        assert caught == []
        assert filtered_mapping.date == mapping.date
        assert filtered_df[filtered_mapping.date].tolist() == expected_dates[2:]
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


def test_build_profitability_export_shapes_summary_and_corridors():
    conn = build_conn()
    try:
        df, _ = load_historical_jobs(conn)
        export_df = build_profitability_export(df, break_even=250.0, top_n_corridors=2)

        assert list(export_df.columns) == ["section", "metric", "value", "unit", "notes"]

        jobs_row = export_df[export_df["metric"] == "Jobs analysed"].iloc[0]
        assert jobs_row["value"] == 4

        below_break_even = export_df[export_df["metric"] == "Below break-even jobs"].iloc[0]
        assert below_break_even["value"] == 3
        assert "75.0%" in below_break_even["notes"]

        median_price = export_df[export_df["metric"] == "Median price per m³"].iloc[0]
        assert pytest.approx(median_price["value"], rel=1e-6) == 212.5

        top_corridors = export_df[export_df["metric"].str.startswith("Top corridor")]
        assert {"Brisbane → Sydney", "Brisbane → Melbourne"}.issubset(
            set(top_corridors["value"])
        )

        lowest_corridors = export_df[export_df["metric"].str.startswith("Lowest margin corridor")]
        assert "Cairns → Melbourne" in set(lowest_corridors["value"])

        below_corridors = export_df[export_df["metric"] == "Corridors below break-even"].iloc[0]
        assert "Cairns → Melbourne" in below_corridors["value"]
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
