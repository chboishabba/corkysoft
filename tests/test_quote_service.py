import sqlite3
import sys
from datetime import date, datetime, timezone
from pathlib import Path
import types
from typing import List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if "openrouteservice" not in sys.modules:
    class _DummyORSClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass


    class _DummyApiError(Exception):
        pass

    ors_module = types.ModuleType("openrouteservice")
    ors_module.Client = _DummyORSClient

    exceptions_module = types.ModuleType("openrouteservice.exceptions")
    exceptions_module.ApiError = _DummyApiError

    ors_module.exceptions = exceptions_module

    sys.modules["openrouteservice"] = ors_module
    sys.modules["openrouteservice.exceptions"] = exceptions_module

from analytics.db import ensure_dashboard_tables
from analytics.price_distribution import load_quotes
import corkysoft.quote_service as quote_service

from corkysoft.quote_service import (
    PRICING_MODELS,
    ClientDetails,
    QuoteInput,
    QuoteResult,
    build_summary,
    calculate_quote,
    ensure_schema,
    find_client_matches,
    format_client_display,
    persist_quote,
    route_distance,
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

    hist_client = conn.execute(
        "SELECT client, client_id FROM historical_jobs"
    ).fetchone()
    assert hist_client == ("Quote builder", None)

    quote_client = conn.execute(
        "SELECT client_display, client_id FROM quotes"
    ).fetchone()
    assert quote_client == ("Quote builder", None)


def test_persist_quote_records_postcodes() -> None:
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    ensure_dashboard_tables(conn)

    inputs = QuoteInput(
        origin="123 Example Street, Brisbane QLD 4000",
        destination="456 Demo Road, Sydney NSW 2000",
        cubic_m=30.0,
        quote_date=date(2024, 1, 15),
        modifiers=[],
        target_margin_percent=10.0,
        country="Australia",
    )
    result = _quote_result()
    result.origin_resolved = "Brisbane QLD 4000"
    result.destination_resolved = "Sydney NSW 2000"
    result.summary_text = build_summary(inputs, result)

    persist_quote(conn, inputs, result)

    stored_postcodes = conn.execute(
        "SELECT origin_postcode, destination_postcode FROM historical_jobs"
    ).fetchone()
    assert stored_postcodes == ("4000", "2000")

    address_postcodes = [
        row[0]
        for row in conn.execute(
            "SELECT postcode FROM addresses ORDER BY id"
        ).fetchall()
    ]
    assert address_postcodes == ["4000", "2000"]

    address_states = [
        row[0]
        for row in conn.execute(
            "SELECT state FROM addresses ORDER BY id"
        ).fetchall()
    ]
    assert address_states == ["QLD", "NSW"]


def test_persist_quote_creates_client_record() -> None:
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    ensure_dashboard_tables(conn)

    inputs = _quote_input()
    inputs.client_details = ClientDetails(
        first_name="Taylor",
        last_name="Jordan",
        email="taylor@example.com",
        phone="0412000123",
    )
    result = _quote_result()
    result.summary_text = build_summary(inputs, result)

    rowid = persist_quote(conn, inputs, result)

    client_row = conn.execute(
        "SELECT id, first_name, last_name, email, phone FROM clients"
    ).fetchone()
    assert client_row is not None
    client_id, first_name, last_name, email, phone = client_row
    assert first_name == "Taylor"
    assert last_name == "Jordan"
    assert email == "taylor@example.com"
    assert phone == "0412000123"
    assert inputs.client_id == client_id

    quote_client = conn.execute(
        "SELECT client_id, client_display FROM quotes WHERE id = ?",
        (rowid,),
    ).fetchone()
    assert quote_client == (client_id, "Taylor Jordan")

    hist_client = conn.execute(
        "SELECT client, client_id FROM historical_jobs"
    ).fetchone()
    assert hist_client == ("Taylor Jordan", client_id)


@pytest.mark.parametrize(
    "details, expected_display",
    [
        (ClientDetails(first_name="Taylor"), "Taylor"),
        (ClientDetails(last_name="Jordan"), "Jordan"),
        (ClientDetails(email="noname@example.com"), "noname@example.com"),
        (ClientDetails(phone="0412 000 123"), "0412 000 123"),
    ],
)
def test_persist_quote_skips_client_creation_without_identity(
    details: ClientDetails, expected_display: str
) -> None:
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    ensure_dashboard_tables(conn)

    inputs = _quote_input()
    inputs.client_details = details
    result = _quote_result()
    result.summary_text = build_summary(inputs, result)

    rowid = persist_quote(conn, inputs, result)

    client_row = conn.execute("SELECT COUNT(*) FROM clients").fetchone()
    assert client_row == (0,)

    quote_client = conn.execute(
        "SELECT client_id, client_display FROM quotes WHERE id = ?",
        (rowid,),
    ).fetchone()
    assert quote_client == (None, expected_display)


def test_find_client_matches_detects_phone() -> None:
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    timestamp = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO clients (
            first_name, last_name, company_name, email, phone,
            address_line1, address_line2, city, state, postcode,
            country, notes, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "Jordan",
            "Miles",
            None,
            "jordan@example.com",
            "0412000123",
            "1 Sample St",
            None,
            "Brisbane",
            "QLD",
            "4000",
            "Australia",
            None,
            timestamp,
            timestamp,
        ),
    )

    matches = find_client_matches(conn, ClientDetails(phone="(0412) 000 123"))
    assert matches
    assert matches[0].id == 1
    assert "matching phone" in matches[0].reason


def test_format_client_display_prefers_company() -> None:
    display = format_client_display("A", "B", "Acme Pty Ltd")
    assert display == "Acme Pty Ltd"


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


def test_is_routable_point_error_handles_dict_in_second_arg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeApiError(Exception):
        def __init__(self, *args: object, status_code: Optional[int] = None) -> None:
            super().__init__(*args)
            self.status_code = status_code

    fake_exceptions = types.SimpleNamespace(ApiError=FakeApiError)
    monkeypatch.setattr(quote_service, "ors_exceptions", fake_exceptions)

    exc = FakeApiError(
        404,
        {
            "error": {
                "code": 2010,
                "message": "Could not find routable point within radius",
            }
        },
    )

    assert quote_service._is_routable_point_error(exc)


def test_is_routable_point_error_handles_string_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeApiError(Exception):
        def __init__(self, *args: object, status_code: Optional[int] = None) -> None:
            super().__init__(*args)
            self.status_code = status_code

    fake_exceptions = types.SimpleNamespace(ApiError=FakeApiError)
    monkeypatch.setattr(quote_service, "ors_exceptions", fake_exceptions)

    exc = FakeApiError(404, "Could not find routable point within a radius")

    assert quote_service._is_routable_point_error(exc)


def test_route_distance_snaps_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = sqlite3.connect(":memory:")

    origin_geo = GeocodeResult(lon=134.0, lat=-23.9, label="Origin label")
    dest_geo = GeocodeResult(lon=134.5, lat=-24.2, label="Dest label")

    class _ApiError(Exception):
        def __init__(self) -> None:
            super().__init__(
                {
                    "error": {
                        "code": 2010,
                        "message": "Could not find routable point within a radius",
                    }
                }
            )
            self.status_code = 404

    class _Client:
        def __init__(self) -> None:
            self.calls: list[list[list[float]]] = []

        def directions(self, *, coordinates, profile, format):  # noqa: D401
            self.calls.append(coordinates)
            if len(self.calls) == 1:
                raise _ApiError()
            return {"routes": [{"summary": {"distance": 1500.0, "duration": 1800.0}}]}

        def nearest(self, *, coordinates, number):  # noqa: D401
            lon, lat = coordinates[0]
            return {
                "features": [
                    {
                        "geometry": {
                            "coordinates": [lon + 0.01, lat + 0.01],
                        }
                    }
                ]
            }

    def _fake_geocode(
        _conn: sqlite3.Connection,
        place: str,
        _country: str,
        *,
        client: object | None = None,
    ) -> GeocodeResult:
        if place == "Origin":
            return GeocodeResult(
                lon=origin_geo.lon,
                lat=origin_geo.lat,
                label=origin_geo.label,
            )
        return GeocodeResult(
            lon=dest_geo.lon,
            lat=dest_geo.lat,
            label=dest_geo.label,
        )

    client_instance = _Client()
    monkeypatch.setattr(
        "corkysoft.quote_service.get_ors_client",
        lambda client=None: client_instance,
    )
    monkeypatch.setattr("corkysoft.quote_service.geocode_cached", _fake_geocode)

    distance_km, duration_hr, resolved_origin, resolved_dest = route_distance(
        conn,
        "Origin",
        "Destination",
        "Australia",
    )

    assert client_instance.calls and len(client_instance.calls) == 2
    assert pytest.approx(distance_km, rel=1e-6) == 1.5
    assert pytest.approx(duration_hr, rel=1e-6) == 0.5
    assert resolved_origin.lon == pytest.approx(origin_geo.lon + 0.01)
    assert resolved_origin.lat == pytest.approx(origin_geo.lat + 0.01)
    assert "Snapped to nearest routable road" in resolved_origin.suggestions
    assert resolved_dest.lon == pytest.approx(dest_geo.lon + 0.01)
    assert resolved_dest.lat == pytest.approx(dest_geo.lat + 0.01)
    assert "Snapped to nearest routable road" in resolved_dest.suggestions


def test_route_distance_manual_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = sqlite3.connect(":memory:")

    class _Client:
        def __init__(self) -> None:
            self.last_coordinates: Optional[List[List[float]]] = None

        def directions(self, *, coordinates, profile, format):  # type: ignore[override]
            self.last_coordinates = coordinates
            return {"routes": [{"summary": {"distance": 4200.0, "duration": 600.0}}]}

    def _fake_geocode(
        _conn: sqlite3.Connection,
        place: str,
        _country: str,
        *,
        client: object | None = None,
    ) -> GeocodeResult:
        if place == "Origin":
            return GeocodeResult(lon=150.0, lat=-33.0, label="Origin label")
        return GeocodeResult(lon=151.0, lat=-34.0, label="Destination label")

    client_instance = _Client()
    monkeypatch.setattr(
        "corkysoft.quote_service.get_ors_client",
        lambda client=None: client_instance,
    )
    monkeypatch.setattr("corkysoft.quote_service.geocode_cached", _fake_geocode)

    origin_override = (153.02, -27.45)
    destination_override = (153.10, -27.48)

    distance_km, duration_hr, resolved_origin, resolved_dest = quote_service.route_distance(
        conn,
        "Origin",
        "Destination",
        "Australia",
        origin_override=origin_override,
        destination_override=destination_override,
    )

    assert distance_km == pytest.approx(4.2)
    assert duration_hr == pytest.approx(0.1666666, rel=1e-6)
    assert client_instance.last_coordinates == [
        [origin_override[0], origin_override[1]],
        [destination_override[0], destination_override[1]],
    ]
    assert resolved_origin.lon == pytest.approx(origin_override[0])
    assert resolved_origin.lat == pytest.approx(origin_override[1])
    assert resolved_dest.lon == pytest.approx(destination_override[0])
    assert resolved_dest.lat == pytest.approx(destination_override[1])
    assert "Manual pin override used for routing" in resolved_origin.suggestions
    assert "Manual pin override used for routing" in resolved_dest.suggestions


def test_route_distance_falls_back_to_haversine(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = sqlite3.connect(":memory:")

    class _ApiError(Exception):
        def __init__(self) -> None:
            super().__init__(
                {
                    "error": {
                        "code": 2010,
                        "message": "Could not find routable point within a radius",
                    }
                }
            )
            self.status_code = 404

    class _Client:
        def directions(self, *_, **__):
            raise _ApiError()

        def nearest(self, *_, **__):
            return {"features": []}

    origin = GeocodeResult(lon=133.88, lat=-23.7, label="Origin")
    dest = GeocodeResult(lon=134.01, lat=-24.0, label="Dest")

    def _fake_geocode(
        _conn: sqlite3.Connection,
        place: str,
        _country: str,
        *,
        client: object | None = None,
    ) -> GeocodeResult:
        return origin if place == "Origin" else dest

    monkeypatch.setattr("corkysoft.quote_service.get_ors_client", lambda client=None: _Client())
    monkeypatch.setattr("corkysoft.quote_service.geocode_cached", _fake_geocode)

    distance_km, duration_hr, resolved_origin, resolved_dest = route_distance(
        conn,
        "Origin",
        "Destination",
        "Australia",
    )

    expected_distance = quote_service._haversine_km(
        origin.lat,
        origin.lon,
        dest.lat,
        dest.lon,
    )
    assert distance_km == pytest.approx(expected_distance)
    expected_duration = (
        expected_distance / quote_service.FALLBACK_SPEED_KMH
        if expected_distance > 0
        else 0.0
    )
    assert duration_hr == pytest.approx(expected_duration)
    assert "Used straight-line estimate due to missing road network" in resolved_origin.suggestions
    assert "Used straight-line estimate due to missing road network" in resolved_dest.suggestions


def test_load_quotes_returns_saved_quote() -> None:
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    ensure_dashboard_tables(conn)

    inputs = _quote_input()
    inputs.client_details = ClientDetails(company_name="Acme Logistics")
    result = _quote_result()
    result.manual_quote = 1100.0
    result.summary_text = build_summary(inputs, result)

    persist_quote(conn, inputs, result)

    df, mapping = load_quotes(conn)

    assert not df.empty
    assert mapping.price == "price_per_m3"
    assert mapping.volume in {"volume_m3", "volume"}

    row = df.iloc[0]
    assert row["client_display"] == "Acme Logistics"
    assert row["corridor_display"] == "Origin → Destination"
    assert row["price_per_m3"] == pytest.approx(result.manual_quote / inputs.cubic_m)
    assert row["final_cost_total"] == pytest.approx(result.total_before_margin)
    assert row["margin_total"] == pytest.approx(
        result.manual_quote - result.total_before_margin
    )

    conn.close()
