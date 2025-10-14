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

from corkysoft.quote_service import (
    PRICING_MODELS,
    QuoteInput,
    QuoteResult,
    build_summary,
    ensure_schema,
    persist_quote,
)


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


def test_persist_quote_stores_manual_override() -> None:
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)

    inputs = _quote_input()
    result = _quote_result()
    result.manual_quote = 1100.0
    result.summary_text = build_summary(inputs, result)

    persist_quote(conn, inputs, result)

    stored = conn.execute(
        "SELECT final_quote, manual_quote FROM quotes"
    ).fetchone()
    assert stored is not None
    final_quote, manual_quote = stored
    assert final_quote == pytest.approx(1200.0)
    assert manual_quote == pytest.approx(1100.0)


def test_build_summary_mentions_manual_amount() -> None:
    inputs = _quote_input()
    result = _quote_result()
    result.manual_quote = 950.0

    summary = build_summary(inputs, result)

    assert "Manual quote override" in summary
    assert "$950.00" in summary
