from __future__ import annotations

from dashboard.pin_state import (
    Coordinate,
    apply_manual_override,
    apply_provider_coordinates,
    initialise_pin_state,
    pin_state_from_mapping,
    pin_state_to_mapping,
    reset_manual_override,
)


def test_provider_refresh_moves_non_manual_pins() -> None:
    state = initialise_pin_state(Coordinate(1, 2), Coordinate(3, 4))
    updated = apply_provider_coordinates(
        state,
        origin=Coordinate(10, 20),
        destination=Coordinate(30, 40),
    )
    assert updated.origin == Coordinate(10, 20)
    assert updated.destination == Coordinate(30, 40)
    assert updated.manual_override is False


def test_provider_refresh_preserves_manual_pins_but_updates_defaults() -> None:
    state = apply_manual_override(
        initialise_pin_state(Coordinate(1, 2), Coordinate(3, 4)),
        origin=Coordinate(5, 6),
    )
    updated = apply_provider_coordinates(
        state,
        origin=Coordinate(10, 20),
        destination=Coordinate(30, 40),
    )
    assert updated.origin == Coordinate(5, 6)
    assert updated.destination == Coordinate(3, 4)
    assert updated.default_origin == Coordinate(10, 20)
    assert updated.default_destination == Coordinate(30, 40)


def test_reset_manual_override_is_idempotent() -> None:
    state = apply_manual_override(
        initialise_pin_state(Coordinate(1, 2), Coordinate(3, 4)),
        destination=Coordinate(7, 8),
    )
    once = reset_manual_override(state)
    twice = reset_manual_override(once)
    assert once == twice
    assert once.origin == Coordinate(1, 2)
    assert once.destination == Coordinate(3, 4)


def test_legacy_mapping_round_trip_is_stable() -> None:
    state = apply_manual_override(
        initialise_pin_state(Coordinate(1, 2), Coordinate(3, 4)),
        origin=Coordinate(5, 6),
        destination=Coordinate(7, 8),
    )
    encoded = pin_state_to_mapping(state)
    assert pin_state_to_mapping(pin_state_from_mapping(encoded)) == encoded
