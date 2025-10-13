"""Australian address normalization helpers with suggestion support."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import re
from typing import Iterable, List, Optional, Sequence


STATE_CODES = {"ACT", "NSW", "NT", "QLD", "SA", "TAS", "VIC", "WA"}
DEFAULT_STATE = "QLD"

UNAMBIGUOUS_ABBREVIATIONS = {
    "ave": "Avenue",
    "av": "Avenue",
    "rd": "Road",
    "st": "Street",
    "hwy": "Highway",
    "pde": "Parade",
    "dr": "Drive",
    "wy": "Way",
    "ct": "Court",
}

AMBIGUOUS_ABBREVIATIONS = {
    "cr": ["Crescent", "Court", "Circuit"],
    "crt": ["Court", "Crescent", "Circuit"],
}

POSTCODE_RE = re.compile(r"\b\d{4}\b")


def _collapse_whitespace(value: str) -> str:
    return " ".join(value.strip().split())


def _split_affixes(token: str) -> tuple[str, str, str]:
    prefix = ""
    suffix = ""
    core = token
    while core and not core[0].isalnum():
        prefix += core[0]
        core = core[1:]
    while core and not core[-1].isalnum():
        suffix = core[-1] + suffix
        core = core[:-1]
    return prefix, core, suffix


def _smart_capitalize(word: str) -> str:
    if not word:
        return word
    if word.isdigit():
        return word
    upper = word.upper()
    if upper in STATE_CODES:
        return upper
    if len(word) <= 3 and word.isalpha():
        return word.capitalize()
    return word[0].upper() + word[1:].lower()


def _contains_state(value: str) -> bool:
    lowered = f" {value.lower()} "
    return any(f" {code.lower()} " in lowered for code in STATE_CODES)


def _ensure_default_state(value: str) -> str:
    if not value:
        return value
    if _contains_state(value):
        return value
    if not POSTCODE_RE.search(value):
        return value
    trimmed = value.rstrip()
    if trimmed.endswith(","):
        return f"{trimmed} {DEFAULT_STATE}"
    return f"{trimmed}, {DEFAULT_STATE}"


@dataclass
class AddressNormalization:
    """Represents the normalization outcome for an address string."""

    raw: str
    canonical: str
    alternatives: List[str] = field(default_factory=list)
    ambiguous_tokens: dict[str, Sequence[str]] = field(default_factory=dict)
    autocorrections: List[str] = field(default_factory=list)

    @property
    def candidates(self) -> List[str]:
        """Return unique candidates (canonical + alternatives)."""

        seen: List[str] = []
        for candidate in [self.canonical, *self.alternatives]:
            cleaned = candidate.strip()
            if cleaned and cleaned not in seen:
                seen.append(cleaned)
        return seen

    def add_autocorrections(self, suggestions: Iterable[str]) -> None:
        for suggestion in suggestions:
            cleaned = _collapse_whitespace(str(suggestion))
            if cleaned and cleaned not in self.autocorrections:
                self.autocorrections.append(cleaned)


def normalize_au_address(raw: str) -> AddressNormalization:
    """Normalize an Australian address while preserving ambiguity information."""

    cleaned = _collapse_whitespace(raw)
    if not cleaned:
        return AddressNormalization(raw=raw, canonical="")

    tokens = cleaned.split()
    rendered_tokens: List[str] = []
    ambiguous_positions: List[tuple[int, str, str, Sequence[str], str]] = []
    ambiguous_summary: dict[str, Sequence[str]] = {}

    for idx, token in enumerate(tokens):
        prefix, core, suffix = _split_affixes(token)
        lower_core = core.lower()
        if lower_core in UNAMBIGUOUS_ABBREVIATIONS:
            replacement = UNAMBIGUOUS_ABBREVIATIONS[lower_core]
            rendered_tokens.append(f"{prefix}{replacement}{suffix}")
        elif lower_core in AMBIGUOUS_ABBREVIATIONS:
            canonical_core = _smart_capitalize(core)
            rendered_tokens.append(f"{prefix}{canonical_core}{suffix}")
            expansions = AMBIGUOUS_ABBREVIATIONS[lower_core]
            ambiguous_positions.append((idx, prefix, suffix, expansions, canonical_core))
            ambiguous_summary[canonical_core] = expansions
        else:
            canonical_core = _smart_capitalize(core)
            rendered_tokens.append(f"{prefix}{canonical_core}{suffix}")

    canonical_pre_state = " ".join(rendered_tokens)
    canonical = _ensure_default_state(canonical_pre_state)

    alternatives: List[str] = []
    if ambiguous_positions:
        base_tokens = rendered_tokens.copy()
        for replacements in product(*(pos[3] for pos in ambiguous_positions)):
            candidate_tokens = base_tokens.copy()
            for replacement, (idx, prefix, suffix, _exp, _core) in zip(
                replacements, ambiguous_positions
            ):
                candidate_tokens[idx] = f"{prefix}{replacement}{suffix}"
            candidate = _ensure_default_state(" ".join(candidate_tokens))
            if candidate != canonical and candidate not in alternatives:
                alternatives.append(candidate)

    return AddressNormalization(
        raw=raw,
        canonical=canonical,
        alternatives=alternatives,
        ambiguous_tokens=ambiguous_summary,
    )


@dataclass
class GeocodeResult:
    lon: float
    lat: float
    label: Optional[str] = None
    normalization: Optional[AddressNormalization] = None
    search_candidates: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    query_used: Optional[str] = None

    def __iter__(self):
        """Allow tuple unpacking compatibility (lon, lat, label)."""
        yield self.lon
        yield self.lat
        yield self.label


def _collect_autocorrect_suggestions(features: Sequence[dict]) -> List[str]:
    seen: set[str] = set()
    collected: List[str] = []
    for feat in features:
        props = feat.get("properties") or {}
        for key in ("label", "name", "street", "locality", "region"):
            value = props.get(key)
            if not value:
                continue
            cleaned = _collapse_whitespace(str(value))
            if cleaned and cleaned not in seen:
                collected.append(cleaned)
                seen.add(cleaned)
    return collected


def _coerce_pelias_param(value: Optional[Sequence[str] | str]) -> Optional[List[str]]:
    """Return a list of values accepted by the Pelias client.

    The upstream openrouteservice client expects ``layers`` and ``sources``
    parameters to be sequences.  Historically this code passed comma-separated
    strings which caused a runtime ``TypeError`` with the message
    ``"Expected a list or tuple, but got str"`` when calling the service.
    Accept either a pre-existing sequence or a comma separated string and
    normalise it into a list of values with surrounding whitespace removed.
    """

    if value is None:
        return None
    if isinstance(value, str):
        return [item for item in (part.strip() for part in value.split(",")) if item]
    return list(value)


def geocode_with_normalization(
    client,
    place: str,
    country: str,
    *,
    size: int = 5,
    strict_layers: str | Sequence[str] | None = "address,street,locality",
    strict_sources: str | Sequence[str] | None = "osm,wof",
) -> GeocodeResult:
    """Geocode *place* and return coordinates together with suggestions."""

    cleaned_input = _collapse_whitespace(place)
    normalization: Optional[AddressNormalization] = None

    candidates: List[str]
    if country.lower() in {"australia", "au"}:
        normalization = normalize_au_address(cleaned_input)
        candidates = normalization.candidates or []
        if not candidates and normalization.canonical:
            candidates = [normalization.canonical]
    else:
        candidates = [cleaned_input]

    if not candidates:
        candidates = [cleaned_input or place]

    features: Sequence[dict] = []
    chosen_query: Optional[str] = None
    layers = _coerce_pelias_param(strict_layers)
    sources = _coerce_pelias_param(strict_sources)

    for candidate in candidates:
        query = f"{candidate}, {country}".strip()
        res = client.pelias_search(
            text=query,
            layers=layers,
            sources=sources,
            size=size,
        )
        feats = res.get("features") or []
        if feats:
            features = feats
            chosen_query = query
            break

    if not features:
        fallback_query = f"{place}, {country}".strip()
        res = client.pelias_search(text=fallback_query, size=size)
        features = res.get("features") or []
        chosen_query = fallback_query
        if not features:
            raise ValueError(f"No geocode found for: {place}, {country}")

    feat = features[0]
    lon, lat = feat["geometry"]["coordinates"]
    props = feat.get("properties") or {}
    label = (
        props.get("label")
        or props.get("name")
        or (normalization.canonical if normalization else cleaned_input)
    )

    suggestions = _collect_autocorrect_suggestions(features)
    if normalization is not None:
        normalization.add_autocorrections(suggestions)

    return GeocodeResult(
        lon=float(lon),
        lat=float(lat),
        label=label,
        normalization=normalization,
        search_candidates=list(dict.fromkeys(candidates)),
        suggestions=suggestions,
        query_used=chosen_query,
    )

