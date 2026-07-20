"""Pure coordinate and quote-pin state transitions.

The Streamlit and map-provider layers should render these values, not own their
lifecycle. Keeping transitions here makes manual overrides deterministic,
replayable, and reusable from future frontends.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Any, Mapping

AUSTRALIA_CENTRE_LAT = -25.2744
AUSTRALIA_CENTRE_LON = 133.7751


@dataclass(frozen=True)
class Coordinate:
    lon: float
    lat: float


@dataclass(frozen=True)
class PinState:
    origin: Coordinate
    destination: Coordinate
    default_origin: Coordinate
    default_destination: Coordinate
    manual_override: bool = False


def coordinate_from_mapping(
    value: Mapping[str, Any] | None,
    *,
    fallback: Coordinate | None = None,
) -> Coordinate:
    """Build a coordinate without mutating the supplied mapping."""
    fallback = fallback or Coordinate(AUSTRALIA_CENTRE_LON, AUSTRALIA_CENTRE_LAT)
    if not value or value.get("lon") is None or value.get("lat") is None:
        return fallback
    return Coordinate(lon=float(value["lon"]), lat=float(value["lat"]))


def initialise_pin_state(origin: Coordinate, destination: Coordinate) -> PinState:
    """Create a state whose current pins and provider defaults are aligned."""
    return PinState(
        origin=origin,
        destination=destination,
        default_origin=origin,
        default_destination=destination,
    )


def apply_provider_coordinates(
    state: PinState,
    *,
    origin: Coordinate,
    destination: Coordinate,
) -> PinState:
    """Refresh provider defaults and move pins only when no manual override exists."""
    return PinState(
        origin=state.origin if state.manual_override else origin,
        destination=state.destination if state.manual_override else destination,
        default_origin=origin,
        default_destination=destination,
        manual_override=state.manual_override,
    )


def apply_manual_override(
    state: PinState,
    *,
    origin: Coordinate | None = None,
    destination: Coordinate | None = None,
) -> PinState:
    """Apply explicitly supplied pins while retaining provider defaults."""
    return PinState(
        origin=origin or state.origin,
        destination=destination or state.destination,
        default_origin=state.default_origin,
        default_destination=state.default_destination,
        manual_override=True,
    )


def reset_manual_override(state: PinState) -> PinState:
    """Return pins to the latest provider defaults.

    Calling this repeatedly is idempotent.
    """
    return PinState(
        origin=state.default_origin,
        destination=state.default_destination,
        default_origin=state.default_origin,
        default_destination=state.default_destination,
        manual_override=False,
    )


def coordinates_equal(left: Coordinate, right: Coordinate) -> bool:
    return isclose(left.lon, right.lon) and isclose(left.lat, right.lat)


def pin_state_to_mapping(state: PinState) -> dict[str, Any]:
    """Serialize to the legacy session-state shape during migration."""
    return {
        "enabled": state.manual_override,
        "origin": {"lon": state.origin.lon, "lat": state.origin.lat},
        "destination": {"lon": state.destination.lon, "lat": state.destination.lat},
        "defaults": {
            "origin": {
                "lon": state.default_origin.lon,
                "lat": state.default_origin.lat,
            },
            "destination": {
                "lon": state.default_destination.lon,
                "lat": state.default_destination.lat,
            },
        },
    }


def pin_state_from_mapping(value: Mapping[str, Any]) -> PinState:
    defaults = value.get("defaults") if isinstance(value.get("defaults"), Mapping) else {}
    default_origin = coordinate_from_mapping(defaults.get("origin"))
    default_destination = coordinate_from_mapping(defaults.get("destination"))
    return PinState(
        origin=coordinate_from_mapping(value.get("origin"), fallback=default_origin),
        destination=coordinate_from_mapping(
            value.get("destination"), fallback=default_destination
        ),
        default_origin=default_origin,
        default_destination=default_destination,
        manual_override=bool(value.get("enabled")),
    )


__all__ = [
    "AUSTRALIA_CENTRE_LAT",
    "AUSTRALIA_CENTRE_LON",
    "Coordinate",
    "PinState",
    "apply_manual_override",
    "apply_provider_coordinates",
    "coordinate_from_mapping",
    "coordinates_equal",
    "initialise_pin_state",
    "pin_state_from_mapping",
    "pin_state_to_mapping",
    "reset_manual_override",
]
