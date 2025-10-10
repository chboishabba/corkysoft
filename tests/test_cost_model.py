from datetime import date
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from corkysoft.cost_model import LaneCostCalculator


@pytest.fixture()
def calculator():
    conn = sqlite3.connect(":memory:")
    calc = LaneCostCalculator(connection=conn)
    yield calc
    calc.close()


def test_per_m3_quote_with_modifiers_and_uplift(calculator):
    breakdown = calculator.calculate_quote(
        corridor_code="BNE-SC",
        origin_lat=-27.4705,
        origin_lon=153.0260,
        dest_lat=-26.6500,
        dest_lon=153.0667,
        distance_km=250.0,
        duration_hours=5.5,
        volume_m3=20.0,
        move_date=date(2024, 11, 15),
        modifier_codes=["DIFFICULT_ACCESS", "TV_CRATE"],
        packing_services=["PACKING", "UNPACKING"],
        margin_pct=0.15,
    )

    assert breakdown.pricing_model == "per_m3"
    assert pytest.approx(breakdown.base_amount, rel=1e-6) == 2400.0

    modifier_totals = {item["code"]: item["amount"] for item in breakdown.modifiers}
    assert modifier_totals["DIFFICULT_ACCESS"] == 350.0
    assert modifier_totals["TV_CRATE"] == 90.0
    assert modifier_totals["PACKING"] == 1000.0
    assert modifier_totals["UNPACKING"] == 700.0

    assert breakdown.seasonality["code"] == "PEAK"
    # Subtotal prior to uplift: 2400 + 350 + 90 + 1000 + 700 = 4540
    assert pytest.approx(breakdown.seasonality["amount"], rel=1e-6) == 3632.0

    assert breakdown.margin["percentage"] == 0.15
    # Subtotal after uplift: 4540 + 3632 = 8172 → margin 15% = 1225.8
    assert pytest.approx(breakdown.margin["amount"], rel=1e-6) == pytest.approx(1225.8, rel=1e-6)

    assert pytest.approx(breakdown.total, rel=1e-6) == pytest.approx(9397.8, rel=1e-6)


def test_hourly_model_for_metro_move(calculator):
    breakdown = calculator.calculate_quote(
        corridor_code="BNE-METRO",
        origin_lat=-27.4698,
        origin_lon=153.0251,
        dest_lat=-27.5598,
        dest_lon=152.9715,
        distance_km=120.0,
        duration_hours=3.0,
        volume_m3=18.0,
        move_date=date(2024, 7, 15),
        modifier_codes=["PIANO"],
        margin_pct=0.10,
    )

    assert breakdown.pricing_model == "hourly"
    assert pytest.approx(breakdown.base_amount, rel=1e-6) == 570.0

    modifier_totals = {item["code"]: item["amount"] for item in breakdown.modifiers}
    assert modifier_totals["PIANO"] == 200.0

    assert breakdown.seasonality is None

    assert breakdown.margin["percentage"] == 0.10
    assert pytest.approx(breakdown.margin["amount"], rel=1e-6) == pytest.approx(77.0, rel=1e-6)

    assert pytest.approx(breakdown.total, rel=1e-6) == pytest.approx(847.0, rel=1e-6)

    assert breakdown.metro_rule["origin"].is_metro is True
    assert breakdown.metro_rule["destination"].is_metro is True
