"""Quote calculation and persistence helpers for Corkysoft applications."""
from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

try:  # pragma: no cover - exercised indirectly via integration paths
    import openrouteservice as _ors
except ModuleNotFoundError:  # pragma: no cover - behaviour verified via unit tests
    _ors = None

if TYPE_CHECKING:  # pragma: no cover - hints for type-checkers only
    import openrouteservice as ors
else:
    ors = _ors  # type: ignore[assignment]

from corkysoft.au_address import GeocodeResult, geocode_with_normalization

COUNTRY_DEFAULT = os.environ.get("ORS_COUNTRY", "Australia")
GEOCODE_BACKOFF = 0.2
ROUTE_BACKOFF = 0.2

_ORS_CLIENT: Optional["ors.Client"] = None


def get_ors_client(client: Optional[ors.Client] = None) -> ors.Client:
    """Return an OpenRouteService client.

    Parameters
    ----------
    client:
        Optional client to use. When ``None`` the client will be instantiated
        lazily using the ``ORS_API_KEY`` environment variable.
    """

    if client is not None:
        return client

    if ors is None:
        raise RuntimeError(
            "openrouteservice client is unavailable. Install the 'openrouteservice' package "
            "to enable routing features."
        )

    global _ORS_CLIENT
    if _ORS_CLIENT is None:
        api_key = os.environ.get("ORS_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set ORS_API_KEY env var (export ORS_API_KEY=YOUR_KEY)"
            )
        _ORS_CLIENT = ors.Client(key=api_key)
    return _ORS_CLIENT


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS geocode_cache (
  place TEXT PRIMARY KEY,
  lon REAL NOT NULL,
  lat REAL NOT NULL,
  ts  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS quotes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  quote_date TEXT NOT NULL,
  origin_input TEXT NOT NULL,
  destination_input TEXT NOT NULL,
  origin_resolved TEXT,
  destination_resolved TEXT,
  origin_lon REAL,
  origin_lat REAL,
  dest_lon REAL,
  dest_lat REAL,
  distance_km REAL NOT NULL,
  duration_hr REAL NOT NULL,
  cubic_m REAL NOT NULL,
  pricing_model TEXT NOT NULL,
  base_subtotal REAL NOT NULL,
  base_components TEXT NOT NULL,
  modifiers_applied TEXT NOT NULL,
  modifiers_total REAL NOT NULL,
  seasonal_multiplier REAL NOT NULL,
  seasonal_label TEXT NOT NULL,
  total_before_margin REAL NOT NULL,
  margin_percent REAL,
  manual_quote REAL,
  final_quote REAL NOT NULL,
  summary TEXT NOT NULL
);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Ensure the quote tables exist in *conn*."""

    conn.executescript(SCHEMA_SQL)
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(quotes)")
    }
    if "manual_quote" not in columns:
        conn.execute("ALTER TABLE quotes ADD COLUMN manual_quote REAL")
    conn.commit()


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def _ensure_address_record(
    conn: sqlite3.Connection,
    raw_input: str,
    resolved: str,
    lon: Optional[float],
    lat: Optional[float],
    country: str,
) -> Optional[int]:
    if not _table_exists(conn, "addresses"):
        return None

    normalized = normalize_place(resolved or raw_input)
    if not normalized:
        normalized = normalize_place(raw_input)

    row = conn.execute(
        """
        SELECT id, lon, lat
        FROM addresses
        WHERE normalized = ? AND (country = ? OR (country IS NULL AND ? IS NULL))
        """,
        (normalized, country, country),
    ).fetchone()
    if row is not None:
        address_id = int(row[0])
        conn.execute(
            "UPDATE addresses SET lon = COALESCE(?, lon), lat = COALESCE(?, lat) WHERE id = ?",
            (lon, lat, address_id),
        )
        return address_id

    cursor = conn.execute(
        """
        INSERT INTO addresses (
            raw_input, normalized, city, country, lon, lat
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            raw_input,
            normalized,
            resolved or normalized,
            country,
            lon,
            lat,
        ),
    )
    return int(cursor.lastrowid)


def _insert_historical_job(
    conn: sqlite3.Connection,
    inputs: QuoteInput,
    result: QuoteResult,
    stored_amount: float,
    created_at: str,
    origin_address_id: Optional[int],
    destination_address_id: Optional[int],
) -> None:
    if not _table_exists(conn, "historical_jobs"):
        return

    cubic_m = inputs.cubic_m
    price_per_m3: Optional[float]
    if cubic_m and cubic_m > 0:
        price_per_m3 = stored_amount / cubic_m
    else:
        price_per_m3 = None

    origin_label = result.origin_resolved or normalize_place(inputs.origin)
    destination_label = result.destination_resolved or normalize_place(
        inputs.destination
    )
    corridor_display = f"{origin_label} → {destination_label}"

    conn.execute(
        """
        INSERT INTO historical_jobs (
            job_date, client, corridor_display,
            price_per_m3, revenue_total, revenue,
            volume_m3, volume, distance_km, final_cost,
            origin, destination, origin_postcode, destination_postcode,
            origin_address_id, destination_address_id,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            inputs.quote_date.isoformat(),
            None,
            corridor_display,
            price_per_m3,
            stored_amount,
            stored_amount,
            cubic_m,
            cubic_m,
            result.distance_km,
            result.total_before_margin,
            origin_label,
            destination_label,
            None,
            None,
            origin_address_id,
            destination_address_id,
            created_at,
            created_at,
        ),
    )


def normalize_place(place: str) -> str:
    return " ".join(place.strip().split())


@dataclass
class Modifier:
    id: str
    label: str
    description: str
    calc_type: str  # "flat" or "percent"
    value: float

    def apply(self, base_subtotal: float) -> float:
        if self.calc_type == "flat":
            return self.value
        if self.calc_type == "percent":
            return base_subtotal * self.value
        raise ValueError(f"Unknown modifier type: {self.calc_type}")


DEFAULT_MODIFIERS: Sequence[Modifier] = (
    Modifier(
        id="stairs",
        label="Difficult access / stairs",
        description="Adds time for multi-level access, long carries or elevators.",
        calc_type="flat",
        value=350.0,
    ),
    Modifier(
        id="packing",
        label="Full packing service",
        description="Packing materials + labour for fragile/high-touch loads.",
        calc_type="percent",
        value=0.18,  # 18 % of base subtotal
    ),
    Modifier(
        id="shuttle",
        label="Shuttle / split load",
        description="Applies when truck cannot access property directly.",
        calc_type="flat",
        value=480.0,
    ),
    Modifier(
        id="storage",
        label="Storage handout",
        description="Additional handling when goods are ex-storage facility.",
        calc_type="flat",
        value=220.0,
    ),
    Modifier(
        id="priority",
        label="Priority / expedited window",
        description="Reserve crew + guaranteed delivery window.",
        calc_type="percent",
        value=0.12,
    ),
)


@dataclass
class PricingModel:
    id: str
    label: str
    max_distance_km: Optional[float]
    base_callout: float
    handling_per_m3: float
    linehaul_per_km: float
    reference_m3: float
    minimum_m3: float


PRICING_MODELS: Sequence[PricingModel] = (
    PricingModel(
        id="metro",
        label="Metro / short haul (≤80 km)",
        max_distance_km=80.0,
        base_callout=180.0,
        handling_per_m3=42.0,
        linehaul_per_km=2.80,
        reference_m3=20.0,
        minimum_m3=18.0,
    ),
    PricingModel(
        id="regional",
        label="Regional lane (≤400 km)",
        max_distance_km=400.0,
        base_callout=260.0,
        handling_per_m3=38.0,
        linehaul_per_km=2.35,
        reference_m3=28.0,
        minimum_m3=22.0,
    ),
    PricingModel(
        id="linehaul",
        label="Linehaul / interstate",
        max_distance_km=None,
        base_callout=320.0,
        handling_per_m3=34.0,
        linehaul_per_km=1.95,
        reference_m3=30.0,
        minimum_m3=25.0,
    ),
)


def pelias_geocode(
    place: str, country: str, *, client: Optional[ors.Client] = None
) -> GeocodeResult:
    resolved_client = get_ors_client(client)
    return geocode_with_normalization(resolved_client, place, country)


def geocode_cached(
    conn: sqlite3.Connection,
    place: str,
    country: str,
    *,
    client: Optional[ors.Client] = None,
) -> GeocodeResult:
    norm = normalize_place(place)
    cache_key = f"{norm}, {country}"
    row = conn.execute(
        "SELECT lon, lat FROM geocode_cache WHERE place = ?",
        (cache_key,),
    ).fetchone()
    if row:
        return GeocodeResult(
            lon=float(row[0]),
            lat=float(row[1]),
            label=None,
            normalization=None,
            search_candidates=[norm],
        )

    result = pelias_geocode(norm, country, client=client)
    conn.execute(
        "INSERT OR REPLACE INTO geocode_cache(place, lon, lat, ts) VALUES (?,?,?,?)",
        (
            cache_key,
            result.lon,
            result.lat,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    time.sleep(GEOCODE_BACKOFF)
    return result


def route_distance(
    conn: sqlite3.Connection,
    origin: str,
    destination: str,
    country: str,
    *,
    client: Optional[ors.Client] = None,
) -> Tuple[float, float, GeocodeResult, GeocodeResult]:
    resolved_client = get_ors_client(client)
    origin_geo = geocode_cached(
        conn, origin, country, client=resolved_client
    )
    dest_geo = geocode_cached(
        conn, destination, country, client=resolved_client
    )

    route = resolved_client.directions(
        coordinates=[[origin_geo.lon, origin_geo.lat], [dest_geo.lon, dest_geo.lat]],
        profile="driving-car",
        format="json",
    )
    summary = route["routes"][0]["summary"]
    meters = float(summary["distance"])
    seconds = float(summary["duration"])
    time.sleep(ROUTE_BACKOFF)
    return meters / 1000.0, seconds / 3600.0, origin_geo, dest_geo


def choose_pricing_model(distance_km: float) -> PricingModel:
    for model in PRICING_MODELS:
        if model.max_distance_km is None or distance_km <= model.max_distance_km:
            return model
    return PRICING_MODELS[-1]


def compute_base_subtotal(
    distance_km: float, cubic_m: float, model: PricingModel
) -> Tuple[float, Dict[str, float]]:
    effective_m3 = max(cubic_m, model.minimum_m3)
    load_factor = max(1.0, effective_m3 / model.reference_m3)
    linehaul_cost = distance_km * model.linehaul_per_km * load_factor
    handling_cost = effective_m3 * model.handling_per_m3
    subtotal = model.base_callout + linehaul_cost + handling_cost
    components = {
        "base_callout": model.base_callout,
        "handling_cost": handling_cost,
        "linehaul_cost": linehaul_cost,
        "load_factor": load_factor,
        "effective_m3": effective_m3,
    }
    return subtotal, components


def compute_modifiers(
    base_subtotal: float, selected_ids: Iterable[str]
) -> Tuple[float, List[Dict[str, float]]]:
    details: List[Dict[str, float]] = []
    total = 0.0
    selected = {sid for sid in selected_ids}
    for mod in DEFAULT_MODIFIERS:
        if mod.id not in selected:
            continue
        amount = mod.apply(base_subtotal)
        total += amount
        details.append(
            {
                "id": mod.id,
                "label": mod.label,
                "calc_type": mod.calc_type,
                "value": mod.value,
                "amount": amount,
            }
        )
    return total, details


@dataclass
class SeasonalAdjustment:
    multiplier: float
    label: str


def seasonal_uplift(quote_dt: date) -> SeasonalAdjustment:
    peak_months = {11, 12, 1}
    shoulder_months = {6, 7, 8}
    if quote_dt.month in peak_months:
        return SeasonalAdjustment(multiplier=1.30, label="Peak season 30% uplift")
    if quote_dt.month in shoulder_months:
        return SeasonalAdjustment(multiplier=1.12, label="Winter shoulder 12% uplift")
    return SeasonalAdjustment(multiplier=1.00, label="Base season")


@dataclass
class QuoteInput:
    origin: str
    destination: str
    cubic_m: float
    quote_date: date
    modifiers: List[str]
    target_margin_percent: Optional[float]
    country: str = COUNTRY_DEFAULT


@dataclass
class QuoteResult:
    final_quote: float
    total_before_margin: float
    base_subtotal: float
    modifiers_total: float
    seasonal_multiplier: float
    seasonal_label: str
    margin_percent: Optional[float]
    pricing_model: PricingModel
    base_components: Dict[str, float]
    modifier_details: List[Dict[str, float]]
    distance_km: float
    duration_hr: float
    origin_resolved: str
    destination_resolved: str
    origin_lon: float
    origin_lat: float
    dest_lon: float
    dest_lat: float
    summary_text: str = ""
    origin_candidates: List[str] = field(default_factory=list)
    destination_candidates: List[str] = field(default_factory=list)
    origin_suggestions: List[str] = field(default_factory=list)
    destination_suggestions: List[str] = field(default_factory=list)
    origin_ambiguities: Dict[str, Sequence[str]] = field(default_factory=dict)
    destination_ambiguities: Dict[str, Sequence[str]] = field(default_factory=dict)
    manual_quote: Optional[float] = None


def format_currency(amount: float) -> str:
    return f"${amount:,.2f}"


def build_summary(inputs: QuoteInput, result: QuoteResult) -> str:
    lines = [
        f"Quote date: {inputs.quote_date.isoformat()}",
        f"Route: {result.origin_resolved} → {result.destination_resolved}",
        f"Distance: {result.distance_km:.1f} km ({result.duration_hr:.1f} h)",
        f"Volume: {inputs.cubic_m:.1f} m³",
        "",
        f"Base ({result.pricing_model.label}): {format_currency(result.base_subtotal)}",
    ]
    if result.modifier_details:
        lines.append("Modifiers:")
        for item in result.modifier_details:
            desc = next(
                (m.description for m in DEFAULT_MODIFIERS if m.id == item["id"]),
                "",
            )
            lines.append(
                f"  - {item['label']}: {format_currency(item['amount'])} ({desc})"
            )
    else:
        lines.append("Modifiers: none")
    if result.seasonal_multiplier != 1.0:
        extra = result.total_before_margin - (
            result.base_subtotal + result.modifiers_total
        )
        lines.append(f"{result.seasonal_label}: +{format_currency(extra)}")
    else:
        lines.append("Seasonal uplift: not applied")
    if result.margin_percent is not None:
        margin_amount = result.final_quote - result.total_before_margin
        lines.append(
            f"Margin ({result.margin_percent:.1f}%): +{format_currency(margin_amount)}"
        )
    else:
        lines.append("Margin: not applied")
    lines.append("")
    lines.append(f"Total before margin: {format_currency(result.total_before_margin)}")
    lines.append(f"Final quote: {format_currency(result.final_quote)}")
    if result.manual_quote is not None:
        lines.append(
            f"Manual quote override: {format_currency(result.manual_quote)}"
        )
    return "\n".join(lines)


def calculate_quote(
    conn: sqlite3.Connection,
    inputs: QuoteInput,
    *,
    client: Optional[ors.Client] = None,
) -> QuoteResult:
    distance_km, duration_hr, origin_geo, dest_geo = route_distance(
        conn,
        inputs.origin,
        inputs.destination,
        inputs.country,
        client=client,
    )

    def resolved_label(geo: GeocodeResult, fallback: str) -> str:
        if geo.label:
            return geo.label
        if geo.normalization and geo.normalization.canonical:
            return geo.normalization.canonical
        return fallback

    origin_resolved = resolved_label(origin_geo, normalize_place(inputs.origin))
    destination_resolved = resolved_label(
        dest_geo, normalize_place(inputs.destination)
    )

    o_lon, o_lat = origin_geo.lon, origin_geo.lat
    d_lon, d_lat = dest_geo.lon, dest_geo.lat

    origin_candidates = (
        origin_geo.normalization.candidates
        if origin_geo.normalization
        else origin_geo.search_candidates
    )
    destination_candidates = (
        dest_geo.normalization.candidates
        if dest_geo.normalization
        else dest_geo.search_candidates
    )

    origin_suggestions = (
        origin_geo.normalization.autocorrections
        if origin_geo.normalization
        else origin_geo.suggestions
    )
    destination_suggestions = (
        dest_geo.normalization.autocorrections
        if dest_geo.normalization
        else dest_geo.suggestions
    )

    origin_ambiguities = (
        dict(origin_geo.normalization.ambiguous_tokens)
        if origin_geo.normalization
        else {}
    )
    destination_ambiguities = (
        dict(dest_geo.normalization.ambiguous_tokens)
        if dest_geo.normalization
        else {}
    )

    model = choose_pricing_model(distance_km)
    base_subtotal, base_components = compute_base_subtotal(
        distance_km, inputs.cubic_m, model
    )

    modifiers_total, modifier_details = compute_modifiers(
        base_subtotal, inputs.modifiers
    )

    season = seasonal_uplift(inputs.quote_date)

    pre_margin = (base_subtotal + modifiers_total) * season.multiplier

    margin_percent = inputs.target_margin_percent
    if margin_percent is not None:
        final_quote = pre_margin * (1 + margin_percent / 100.0)
    else:
        final_quote = pre_margin

    result = QuoteResult(
        final_quote=final_quote,
        total_before_margin=pre_margin,
        base_subtotal=base_subtotal,
        modifiers_total=modifiers_total,
        seasonal_multiplier=season.multiplier,
        seasonal_label=season.label,
        margin_percent=margin_percent,
        pricing_model=model,
        base_components=base_components,
        modifier_details=modifier_details,
        distance_km=distance_km,
        duration_hr=duration_hr,
        origin_resolved=origin_resolved,
        destination_resolved=destination_resolved,
        origin_lon=o_lon,
        origin_lat=o_lat,
        dest_lon=d_lon,
        dest_lat=d_lat,
        summary_text="",
        origin_candidates=list(origin_candidates or []),
        destination_candidates=list(destination_candidates or []),
        origin_suggestions=list(origin_suggestions or []),
        destination_suggestions=list(destination_suggestions or []),
        origin_ambiguities=origin_ambiguities,
        destination_ambiguities=destination_ambiguities,
    )
    result.summary_text = build_summary(inputs, result)
    return result


def persist_quote(
    conn: sqlite3.Connection,
    inputs: QuoteInput,
    result: QuoteResult,
    manual_quote: Optional[float] = None,
) -> int:
    manual_value = (
        manual_quote
        if manual_quote is not None
        else result.manual_quote
    )
    created_at = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        """
        INSERT INTO quotes (
            created_at, quote_date,
            origin_input, destination_input,
            origin_resolved, destination_resolved,
            origin_lon, origin_lat, dest_lon, dest_lat,
            distance_km, duration_hr, cubic_m,
            pricing_model, base_subtotal, base_components,
            modifiers_applied, modifiers_total,
            seasonal_multiplier, seasonal_label,
            total_before_margin, margin_percent,
            manual_quote, final_quote, summary
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            created_at,
            inputs.quote_date.isoformat(),
            inputs.origin,
            inputs.destination,
            result.origin_resolved,
            result.destination_resolved,
            result.origin_lon,
            result.origin_lat,
            result.dest_lon,
            result.dest_lat,
            result.distance_km,
            result.duration_hr,
            inputs.cubic_m,
            result.pricing_model.id,
            result.base_subtotal,
            json.dumps(result.base_components),
            json.dumps(result.modifier_details),
            result.modifiers_total,
            result.seasonal_multiplier,
            result.seasonal_label,
            result.total_before_margin,
            result.margin_percent,
            manual_value,
            result.final_quote,
            result.summary_text,
        ),
    )
    quote_rowid = int(cursor.lastrowid)

    stored_amount = (
        float(manual_value)
        if manual_value is not None
        else result.final_quote
    )

    origin_address_id = _ensure_address_record(
        conn,
        inputs.origin,
        result.origin_resolved,
        result.origin_lon,
        result.origin_lat,
        inputs.country,
    )
    destination_address_id = _ensure_address_record(
        conn,
        inputs.destination,
        result.destination_resolved,
        result.dest_lon,
        result.dest_lat,
        inputs.country,
    )
    _insert_historical_job(
        conn,
        inputs,
        result,
        stored_amount,
        created_at,
        origin_address_id,
        destination_address_id,
    )
    conn.commit()
    return quote_rowid


__all__ = [
    "COUNTRY_DEFAULT",
    "DEFAULT_MODIFIERS",
    "GEOCODE_BACKOFF",
    "ROUTE_BACKOFF",
    "Modifier",
    "PricingModel",
    "QuoteInput",
    "QuoteResult",
    "SeasonalAdjustment",
    "calculate_quote",
    "compute_base_subtotal",
    "compute_modifiers",
    "ensure_schema",
    "format_currency",
    "persist_quote",
    "build_summary",
    "get_ors_client",
]
