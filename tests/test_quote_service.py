import sqlite3
import sys
from datetime import date
from pathlib import Path
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if "openrouteservice" not in sys.modules:
    class _DummyORSClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass


    sys.modules["openrouteservice"] = types.SimpleNamespace(Client=_DummyORSClient)

from analytics.db import ensure_dashboard_tables
from analytics.price_distribution import load_quotes
from corkysoft.quote_service import (
    PRICING_MODELS,
    QuoteInput,
    QuoteResult,
    build_summary,
    calculate_quote,
    ensure_schema,
    persist_quote,
)
from corkysoft.au_address import GeocodeResult


def _quote_input() -> QuoteInput:
    return QuoteInput(
        origin="Origin",
        destination="Destination",
        cubic_m=30.0,
        quote_date=date(2024, 1, 15),
        modifiers=[],
        target_margin_percent=20.0,
        country="Australia",
    )


def _quote_result() -> QuoteResult:
    model = PRICING_MODELS[0]
    return QuoteResult(
        final_quote=1200.0,
        total_before_margin=1000.0,
        base_subtotal=800.0,
        modifiers_total=200.0,
        seasonal_multiplier=1.0,
        seasonal_label="Base season",
        margin_percent=20.0,
        pricing_model=model,
        base_components={
            "base_callout": 180.0,
            "handling_cost": 200.0,
            "linehaul_cost": 300.0,
        },
        modifier_details=[],
        distance_km=100.0,
        duration_hr=2.0,
        origin_resolved="Origin",
        destination_resolved="Destination",
        origin_lon=1.0,
        origin_lat=2.0,
        dest_lon=3.0,
        dest_lat=4.0,
    )


def test_get_ors_client_requires_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    import corkysoft.quote_service as quote_service

    monkeypatch.setattr(quote_service, "ors", None)
    monkeypatch.setattr(quote_service, "_ORS_CLIENT", None)

    with pytest.raises(RuntimeError, match="openrouteservice client is unavailable"):
        quote_service.get_ors_client()


def test_persist_quote_stores_manual_override() -> None:
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    ensure_dashboard_tables(conn)

    inputs = _quote_input()
    result = _quote_result()
    result.manual_quote = 1100.0
    result.summary_text = build_summary(inputs, result)

    rowid = persist_quote(conn, inputs, result)

    stored = conn.execute(
        "SELECT final_quote, manual_quote FROM quotes"
    ).fetchone()
    assert stored is not None
    final_quote, manual_quote = stored
    assert final_quote == pytest.approx(1200.0)
    assert manual_quote == pytest.approx(1100.0)
    assert rowid == 1


def test_persist_quote_creates_historical_job_entry() -> None:
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    ensure_dashboard_tables(conn)

    inputs = _quote_input()
    result = _quote_result()
    result.summary_text = build_summary(inputs, result)

    persist_quote(conn, inputs, result)

    hist_row = conn.execute(
        """
        SELECT job_date, corridor_display, price_per_m3, distance_km, final_cost,
               origin_address_id, destination_address_id
        FROM historical_jobs
        """
    ).fetchone()
    assert hist_row is not None
    job_date, corridor, price_per_m3, distance_km, final_cost, origin_id, dest_id = hist_row
    assert job_date == inputs.quote_date.isoformat()
    assert corridor == f"{result.origin_resolved} → {result.destination_resolved}"
    assert price_per_m3 == pytest.approx(result.final_quote / inputs.cubic_m)
    assert distance_km == pytest.approx(result.distance_km)
    assert final_cost == pytest.approx(result.total_before_margin)
    assert origin_id is not None
    assert dest_id is not None

    address_count = conn.execute("SELECT COUNT(*) FROM addresses").fetchone()[0]
    assert address_count == 2


def test_build_summary_mentions_manual_amount() -> None:
    inputs = _quote_input()
    result = _quote_result()
    result.manual_quote = 950.0

    summary = build_summary(inputs, result)

    assert "Manual quote override" in summary
    assert "$950.00" in summary


def test_calculate_quote_applies_margin(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    ensure_dashboard_tables(conn)

    inputs = _quote_input()

    origin_geo = GeocodeResult(
        lon=153.01,
        lat=-27.67,
        label="Origin label",
    )
    dest_geo = GeocodeResult(
        lon=153.06,
        lat=-27.50,
        label="Destination label",
    )

    def _fake_route_distance(
        *_args: object, **_kwargs: object
    ) -> tuple[float, float, GeocodeResult, GeocodeResult]:
        return 28.3, 0.5, origin_geo, dest_geo

    monkeypatch.setattr(
        "corkysoft.quote_service.route_distance",
        _fake_route_distance,
    )

    result = calculate_quote(conn, inputs)

    assert result.margin_percent == pytest.approx(inputs.target_margin_percent)
    assert result.final_quote == pytest.approx(
        result.total_before_margin * (1 + inputs.target_margin_percent / 100.0)
    )

    conn.close()


def test_load_quotes_returns_saved_quote() -> None:
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    ensure_dashboard_tables(conn)

    inputs = _quote_input()
    result = _quote_result()
    result.manual_quote = 1100.0
    result.summary_text = build_summary(inputs, result)

    persist_quote(conn, inputs, result)

    df, mapping = load_quotes(conn)

    assert not df.empty
    assert mapping.price == "price_per_m3"
    assert mapping.volume in {"volume_m3", "volume"}

    row = df.iloc[0]
    assert row["client_display"] == "Quote builder"
    assert row["corridor_display"] == "Origin → Destination"
    assert row["price_per_m3"] == pytest.approx(result.manual_quote / inputs.cubic_m)
    assert row["final_cost_total"] == pytest.approx(result.total_before_margin)
    assert row["margin_total"] == pytest.approx(
        result.manual_quote - result.total_before_margin
    )

    conn.close()
