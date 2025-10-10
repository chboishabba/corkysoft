"""Quote calculation and persistence helpers for Corkysoft applications."""
from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import openrouteservice as ors

COUNTRY_DEFAULT = os.environ.get("ORS_COUNTRY", "Australia")
GEOCODE_BACKOFF = 0.2
ROUTE_BACKOFF = 0.2

_ORS_CLIENT: Optional[ors.Client] = None


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
  final_quote REAL NOT NULL,
  summary TEXT NOT NULL
);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Ensure the quote tables exist in *conn*."""

    conn.executescript(SCHEMA_SQL)
    conn.commit()


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
) -> Tuple[float, float, str]:
    query = f"{place}, {country}"
    resolved_client = get_ors_client(client)
    res = resolved_client.pelias_search(text=query, size=1)
    feats = res.get("features") or []
    if not feats:
        raise ValueError(f"No geocode found for: {query}")
    feat = feats[0]
    lon, lat = feat["geometry"]["coordinates"]
    label = (
        feat["properties"].get("label")
        or feat["properties"].get("name")
        or normalize_place(place)
    )
    return float(lon), float(lat), label


def geocode_cached(
    conn: sqlite3.Connection,
    place: str,
    country: str,
    *,
    client: Optional[ors.Client] = None,
) -> Tuple[float, float, Optional[str]]:
    norm = normalize_place(place)
    cache_key = f"{norm}, {country}"
    row = conn.execute(
        "SELECT lon, lat FROM geocode_cache WHERE place = ?",
        (cache_key,),
    ).fetchone()
    if row:
        return float(row[0]), float(row[1]), None

    lon, lat, label = pelias_geocode(norm, country, client=client)
    conn.execute(
        "INSERT OR REPLACE INTO geocode_cache(place, lon, lat, ts) VALUES (?,?,?,?)",
        (cache_key, lon, lat, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    time.sleep(GEOCODE_BACKOFF)
    return lon, lat, label


def route_distance(
    conn: sqlite3.Connection,
    origin: str,
    destination: str,
    country: str,
    *,
    client: Optional[ors.Client] = None,
) -> Tuple[float, float, str, str, float, float, float, float]:
    resolved_client = get_ors_client(client)
    o_lon, o_lat, o_label = geocode_cached(
        conn, origin, country, client=resolved_client
    )
    d_lon, d_lat, d_label = geocode_cached(
        conn, destination, country, client=resolved_client
    )

    route = resolved_client.directions(
        coordinates=[[o_lon, o_lat], [d_lon, d_lat]],
        profile="driving-car",
        format="json",
    )
    summary = route["routes"][0]["summary"]
    meters = float(summary["distance"])
    seconds = float(summary["duration"])
    time.sleep(ROUTE_BACKOFF)
    return (
        meters / 1000.0,
        seconds / 3600.0,
        o_label,
        d_label,
        o_lon,
        o_lat,
        d_lon,
        d_lat,
    )


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
    summary_text: str


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
    return "\n".join(lines)


def calculate_quote(
    conn: sqlite3.Connection,
    inputs: QuoteInput,
    *,
    client: Optional[ors.Client] = None,
) -> QuoteResult:
    (
        distance_km,
        duration_hr,
        origin_resolved,
        destination_resolved,
        o_lon,
        o_lat,
        d_lon,
        d_lat,
    ) = route_distance(
        conn,
        inputs.origin,
        inputs.destination,
        inputs.country,
        client=client,
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
    )
    result.summary_text = build_summary(inputs, result)
    return result


def persist_quote(
    conn: sqlite3.Connection, inputs: QuoteInput, result: QuoteResult
) -> None:
    conn.execute(
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
            final_quote, summary
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
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
            result.final_quote,
            result.summary_text,
        ),
    )
    conn.commit()


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
