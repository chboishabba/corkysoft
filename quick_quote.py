#!/usr/bin/env python3
"""Quick quote entry CLI for Corkysoft.

Feature 2 implementation: collects quote inputs, queries ORS for distance,
calculates pricing using the Feature 1 model, applies modifiers/seasonal uplift
and optional margin, persists to SQLite and prints a copy-paste summary.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import textwrap
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import openrouteservice as ors

DB_PATH = os.environ.get("ROUTES_DB", "routes.db")
ORS_KEY = os.environ.get("ORS_API_KEY")
COUNTRY_DEFAULT = os.environ.get("ORS_COUNTRY", "Australia")
GEOCODE_BACKOFF = 0.2
ROUTE_BACKOFF = 0.2

if not ORS_KEY:
    raise SystemExit("Set ORS_API_KEY env var (export ORS_API_KEY=YOUR_KEY)")

client = ors.Client(key=ORS_KEY)

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


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def normalize_place(place: str) -> str:
    return " ".join(place.strip().split())


def pelias_geocode(place: str, country: str) -> Tuple[float, float, str]:
    query = f"{place}, {country}"
    res = client.pelias_search(text=query, size=1)
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


def geocode_cached(conn: sqlite3.Connection, place: str, country: str) -> Tuple[float, float, Optional[str]]:
    norm = normalize_place(place)
    cache_key = f"{norm}, {country}"
    row = conn.execute(
        "SELECT lon, lat FROM geocode_cache WHERE place = ?",
        (cache_key,),
    ).fetchone()
    if row:
        return float(row[0]), float(row[1]), None

    lon, lat, label = pelias_geocode(norm, country)
    conn.execute(
        "INSERT OR REPLACE INTO geocode_cache(place, lon, lat, ts) VALUES (?,?,?,?)",
        (cache_key, lon, lat, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    time.sleep(GEOCODE_BACKOFF)
    return lon, lat, label


def route_distance(conn: sqlite3.Connection, origin: str, destination: str, country: str) -> Tuple[float, float, str, str, float, float, float, float]:
    o_lon, o_lat, o_label = geocode_cached(conn, origin, country)
    d_lon, d_lat, d_label = geocode_cached(conn, destination, country)

    route = client.directions(
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


def compute_base_subtotal(distance_km: float, cubic_m: float, model: PricingModel) -> Tuple[float, Dict[str, float]]:
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


def compute_modifiers(base_subtotal: float, selected_ids: Iterable[str]) -> Tuple[float, List[Dict[str, float]]]:
    details: List[Dict[str, float]] = []
    total = 0.0
    selected = {sid for sid in selected_ids}
    for mod in DEFAULT_MODIFIERS:
        if mod.id not in selected:
            continue
        amount = mod.apply(base_subtotal)
        total += amount
        details.append({
            "id": mod.id,
            "label": mod.label,
            "calc_type": mod.calc_type,
            "value": mod.value,
            "amount": amount,
        })
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
            desc = next((m.description for m in DEFAULT_MODIFIERS if m.id == item["id"]), "")
            lines.append(
                f"  - {item['label']}: {format_currency(item['amount'])} ({desc})"
            )
    else:
        lines.append("Modifiers: none")
    if result.seasonal_multiplier != 1.0:
        extra = result.total_before_margin - (result.base_subtotal + result.modifiers_total)
        lines.append(
            f"{result.seasonal_label}: +{format_currency(extra)}"
        )
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


def calculate_quote(conn: sqlite3.Connection, inputs: QuoteInput) -> QuoteResult:
    (
        distance_km,
        duration_hr,
        origin_resolved,
        destination_resolved,
        o_lon,
        o_lat,
        d_lon,
        d_lat,
    ) = route_distance(conn, inputs.origin, inputs.destination, inputs.country)

    model = choose_pricing_model(distance_km)
    base_subtotal, base_components = compute_base_subtotal(distance_km, inputs.cubic_m, model)

    modifiers_total, modifier_details = compute_modifiers(base_subtotal, inputs.modifiers)

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


def persist_quote(conn: sqlite3.Connection, inputs: QuoteInput, result: QuoteResult) -> None:
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


def prompt_input(prompt: str, default: Optional[str] = None) -> str:
    while True:
        suffix = f" [{default}]" if default else ""
        value = input(f"{prompt}{suffix}: ").strip()
        if not value and default is not None:
            return default
        if value:
            return value
        print("This field is required. Please enter a value.")


def prompt_float(prompt: str, default: Optional[float] = None, minimum: Optional[float] = None) -> float:
    while True:
        suffix = ""
        if default is not None:
            suffix = f" [{default}]"
        value = input(f"{prompt}{suffix}: ").strip()
        if not value and default is not None:
            value = str(default)
        try:
            num = float(value)
        except ValueError:
            print("Enter a numeric value.")
            continue
        if minimum is not None and num < minimum:
            print(f"Value must be ≥ {minimum}.")
            continue
        return num


def prompt_date(prompt: str, default: Optional[date] = None) -> date:
    default_str = default.isoformat() if default else None
    while True:
        raw = prompt_input(prompt, default=default_str)
        try:
            return datetime.fromisoformat(raw).date()
        except ValueError:
            print("Enter a date in ISO format (YYYY-MM-DD).")


def prompt_modifiers() -> List[str]:
    print("\nSelect modifiers (tick boxes). Enter comma-separated numbers, blank for none.")
    for idx, mod in enumerate(DEFAULT_MODIFIERS, start=1):
        desc = textwrap.fill(mod.description, width=68)
        desc_lines = desc.splitlines() or [""]
        label_line = f"  [{idx}] {mod.label}"
        if mod.calc_type == "flat":
            value_txt = format_currency(mod.value)
        else:
            value_txt = f"{mod.value*100:.0f}% of base"
        label_line += f" ({value_txt})"
        print(label_line)
        for line in desc_lines:
            print(f"       {line}")
    raw = input("Selection: ").strip()
    if not raw:
        return []
    selected: List[str] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit():
            print(f"Ignoring invalid selection: {part}")
            continue
        idx = int(part)
        if idx < 1 or idx > len(DEFAULT_MODIFIERS):
            print(f"Ignoring invalid selection: {part}")
            continue
        selected.append(DEFAULT_MODIFIERS[idx - 1].id)
    return selected


def prompt_margin() -> Optional[float]:
    raw = input("Target margin % (blank for none): ").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        print("Invalid margin. Skipping margin application.")
        return None


def gather_inputs(args: argparse.Namespace) -> QuoteInput:
    if args.origin and args.destination and args.cubic_m is not None:
        quote_dt = args.date or date.today()
        modifiers = args.modifiers or []
        margin = args.margin
        return QuoteInput(
            origin=args.origin,
            destination=args.destination,
            cubic_m=args.cubic_m,
            quote_date=quote_dt,
            modifiers=modifiers,
            target_margin_percent=margin,
            country=args.country or COUNTRY_DEFAULT,
        )

    print("\n--- Quick Quote Entry ---")
    origin = prompt_input("Origin address / postcode")
    destination = prompt_input("Destination address / postcode")
    cubic_m = prompt_float("Volume (m³)", minimum=1.0)
    quote_dt = prompt_date("Move date", default=date.today())
    modifiers = prompt_modifiers()
    margin = prompt_margin()
    return QuoteInput(
        origin=origin,
        destination=destination,
        cubic_m=cubic_m,
        quote_date=quote_dt,
        modifiers=modifiers,
        target_margin_percent=margin,
        country=args.country or COUNTRY_DEFAULT,
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick quote entry CLI")
    parser.add_argument("--origin")
    parser.add_argument("--destination")
    parser.add_argument("--cubic-m", type=float, dest="cubic_m")
    parser.add_argument("--date", type=lambda s: datetime.fromisoformat(s).date())
    parser.add_argument("--modifier", action="append", dest="modifiers")
    parser.add_argument("--margin", type=float)
    parser.add_argument("--country", default=COUNTRY_DEFAULT)
    parser.add_argument("--no-save", action="store_true", help="Do not persist quote to the database")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    conn = sqlite3.connect(DB_PATH)
    ensure_schema(conn)

    inputs = gather_inputs(args)

    result = calculate_quote(conn, inputs)

    if not args.no_save:
        persist_quote(conn, inputs, result)

    print("\n--- Quote Summary ---")
    print(result.summary_text)

    if not args.no_save:
        rowid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        print(f"\nSaved quote #{rowid} to {DB_PATH}.")
    conn.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
