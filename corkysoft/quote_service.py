"""Quote calculation and persistence helpers for Corkysoft applications."""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from corkysoft.au_address import GeocodeResult
from corkysoft.pricing import (
    DEFAULT_MODIFIERS,
    PRICING_MODELS,
    Modifier,
    PricingModel,
    SeasonalAdjustment,
    choose_pricing_model,
    compute_base_subtotal,
    compute_modifiers,
    seasonal_uplift,
)
from corkysoft.repo import (
    CLIENT_FIELD_NAMES,
    ClientDetails,
    ClientMatch,
    SCHEMA_SQL,
    ensure_schema,
    find_client_matches,
    format_client_display,
    persist_quote,
)
from corkysoft.routing import (
    COUNTRY_DEFAULT,
    FALLBACK_SPEED_KMH,
    GEOCODE_BACKOFF,
    ROUTE_BACKOFF,
    PinSnapResult,
    _is_routable_point_error,
    geocode_cached,
    get_ors_client,
    normalize_place,
    pelias_geocode,
    route_distance,
    snap_coordinates_to_road,
)

if TYPE_CHECKING:  # pragma: no cover - hints for type-checkers only
    import openrouteservice as ors

logger = logging.getLogger(__name__)


@dataclass
class QuoteInput:
    origin: str
    destination: str
    cubic_m: float
    quote_date: date
    modifiers: List[str]
    target_margin_percent: Optional[float]
    country: str = COUNTRY_DEFAULT
    origin_coordinates: Optional[Tuple[float, float]] = None
    destination_coordinates: Optional[Tuple[float, float]] = None
    client_id: Optional[int] = None
    client_details: Optional[ClientDetails] = None


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
    client: Optional["ors.Client"] = None,
) -> QuoteResult:
    distance_km, duration_hr, origin_geo, dest_geo = route_distance(
        conn,
        inputs.origin,
        inputs.destination,
        inputs.country,
        client=client,
        origin_override=inputs.origin_coordinates,
        destination_override=inputs.destination_coordinates,
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
    seasonal = seasonal_uplift(inputs.quote_date)
    total_before_margin = (
        base_subtotal + modifiers_total
    ) * seasonal.multiplier

    if inputs.target_margin_percent:
        final_quote = total_before_margin * (1 + inputs.target_margin_percent / 100.0)
        margin_percent = inputs.target_margin_percent
    else:
        final_quote = total_before_margin
        margin_percent = None

    result = QuoteResult(
        final_quote=final_quote,
        total_before_margin=total_before_margin,
        base_subtotal=base_subtotal,
        modifiers_total=modifiers_total,
        seasonal_multiplier=seasonal.multiplier,
        seasonal_label=seasonal.label,
        margin_percent=margin_percent,
        pricing_model=model,
        base_components=base_components,
        modifier_details=modifier_details,
        distance_km=distance_km,
        duration_hr=duration_hr,
        origin_resolved=origin_resolved,
        destination_resolved=destination_resolved,
        origin_lon=float(origin_geo.lon),
        origin_lat=float(origin_geo.lat),
        dest_lon=float(dest_geo.lon),
        dest_lat=float(dest_geo.lat),
        origin_candidates=list(origin_candidates or []),
        destination_candidates=list(destination_candidates or []),
        origin_suggestions=list(origin_suggestions or []),
        destination_suggestions=list(destination_suggestions or []),
        origin_ambiguities=origin_ambiguities,
        destination_ambiguities=destination_ambiguities,
    )
    result.summary_text = build_summary(inputs, result)
    return result


__all__ = [
    "CLIENT_FIELD_NAMES",
    "COUNTRY_DEFAULT",
    "DEFAULT_MODIFIERS",
    "FALLBACK_SPEED_KMH",
    "GEOCODE_BACKOFF",
    "ROUTE_BACKOFF",
    "Modifier",
    "ClientDetails",
    "ClientMatch",
    "PRICING_MODELS",
    "PricingModel",
    "QuoteInput",
    "QuoteResult",
    "PinSnapResult",
    "SeasonalAdjustment",
    "SCHEMA_SQL",
    "build_summary",
    "calculate_quote",
    "choose_pricing_model",
    "compute_base_subtotal",
    "compute_modifiers",
    "ensure_schema",
    "find_client_matches",
    "format_client_display",
    "format_currency",
    "get_ors_client",
    "normalize_place",
    "pelias_geocode",
    "persist_quote",
    "route_distance",
    "seasonal_uplift",
    "snap_coordinates_to_road",
    "_is_routable_point_error",
    "geocode_cached",
]
