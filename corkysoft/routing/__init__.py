"""Routing helpers built around pluggable providers."""
from __future__ import annotations

import logging
import math
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from corkysoft.au_address import GeocodeResult
from corkysoft.routing import providers as _providers
from corkysoft.routing.providers import (
    GoogleMapsRoutingProvider,
    IncompleteRouteError,
    NoRoutablePointError,
    OpenRouteServiceProvider,
    RoutingError,
    RoutingProvider,
    SnapResult,
    get_google_maps_client,
    get_ors_client as _providers_get_ors_client,
    _ORS_CLIENT,
    ors,
    ors_exceptions,
    _is_routable_point_error,
)

logger = logging.getLogger(__name__)

COUNTRY_DEFAULT = os.environ.get("ORS_COUNTRY", "Australia")
GEOCODE_BACKOFF = 0.2
ROUTE_BACKOFF = 0.2
FALLBACK_SPEED_KMH = 65.0
SNAP_SEARCH_RADII = (50, 150, 300, 750, 1500)
_ROUTING_PROVIDER_ENV = "ROUTING_PROVIDER"

# Keep module-level aliases for compatibility with legacy tests expecting to
# patch ``corkysoft.routing.ors`` or ``_ORS_CLIENT`` directly.
ors = _providers.ors
ors_exceptions = _providers.ors_exceptions
_ORS_CLIENT = _providers._ORS_CLIENT


def normalize_place(place: str) -> str:
    """Return a whitespace-normalised version of *place*."""

    return " ".join(place.strip().split())


def _provider_from_name(name: Optional[str], client: Optional[Any] = None) -> RoutingProvider:
    provider_name = (name or os.environ.get(_ROUTING_PROVIDER_ENV, "ors")).strip().lower()
    if provider_name in {"google", "googlemaps", "gmaps"}:
        return GoogleMapsRoutingProvider(client)
    return OpenRouteServiceProvider(client)


def _resolve_provider(
    provider: Optional[RoutingProvider],
    *,
    provider_name: Optional[str] = None,
    client: Optional[Any] = None,
) -> RoutingProvider:
    if provider is not None:
        return provider
    return _provider_from_name(provider_name, client)


def get_ors_client(client: Optional[Any] = None) -> Any:
    """Proxy to :mod:`corkysoft.routing.providers.get_ors_client`."""

    global ors, _ORS_CLIENT
    # Keep provider module state in sync with any monkeypatching applied to this
    # module during tests.
    if ors is not _providers.ors:
        _providers.ors = ors
    if _ORS_CLIENT is not _providers._ORS_CLIENT:
        _providers._ORS_CLIENT = _ORS_CLIENT

    try:
        client_instance = _providers_get_ors_client(client)
    finally:
        ors = _providers.ors
        _ORS_CLIENT = _providers._ORS_CLIENT
    return client_instance


def pelias_geocode(
    place: str,
    country: str,
    *,
    client: Optional[Any] = None,
    provider: Optional[RoutingProvider] = None,
    provider_name: Optional[str] = None,
) -> GeocodeResult:
    """Geocode *place* using the configured provider.

    Historically this function wrapped the OpenRouteService Pelias endpoint.
    To preserve compatibility we default to the ORS-backed provider unless
    callers explicitly provide another provider instance or name.
    """

    resolved_provider = _resolve_provider(
        provider,
        provider_name=provider_name or "ors",
        client=client,
    )
    return resolved_provider.geocode(place, country)


def geocode_cached(
    conn: sqlite3.Connection,
    place: str,
    country: str,
    *,
    provider: Optional[RoutingProvider] = None,
    provider_name: Optional[str] = None,
    client: Optional[Any] = None,
) -> GeocodeResult:
    """Return cached geocoding results, falling back to the routing provider."""

    resolved_provider = _resolve_provider(
        provider,
        provider_name=provider_name,
        client=client,
    )
    norm = normalize_place(place)
    cache_key = f"{norm}, {country}"
    try:
        row = conn.execute(
            """
            SELECT lon, lat, postalcode, region_code, region, locality, county
            FROM geocode_cache
            WHERE place = ?
            """,
            (cache_key,),
        ).fetchone()
    except sqlite3.OperationalError:
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
        row = None

    if row:
        lon, lat, postalcode, region_code, region, locality, county = row
        if any([postalcode, region_code, region, locality, county]):
            return GeocodeResult(
                lon=float(lon),
                lat=float(lat),
                label=None,
                normalization=None,
                search_candidates=[norm],
                postalcode=postalcode,
                region_code=region_code,
                region=region,
                locality=locality,
                county=county,
            )

    result = resolved_provider.geocode(norm, country)
    conn.execute(
        """
        INSERT OR REPLACE INTO geocode_cache(
            place, lon, lat, postalcode, region_code, region, locality, county, ts
        ) VALUES (?,?,?,?,?,?,?,?,?)
        """,
        (
            cache_key,
            result.lon,
            result.lat,
            getattr(result, "postalcode", None),
            getattr(result, "region_code", None),
            getattr(result, "region", None),
            getattr(result, "locality", None),
            getattr(result, "county", None),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    time.sleep(GEOCODE_BACKOFF)
    return result


def _note_geocode(geo: GeocodeResult, note: Optional[str]) -> None:
    if not note:
        return
    if not hasattr(geo, "suggestions") or geo.suggestions is None:
        geo.suggestions = []  # type: ignore[assignment]
    if note not in geo.suggestions:
        geo.suggestions.append(note)


def _haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    earth_radius_km = 6371.0088
    return earth_radius_km * c


@dataclass
class PinSnapResult:
    origin: Tuple[float, float]
    destination: Tuple[float, float]
    notes: Dict[str, str] = field(default_factory=dict)
    changed: bool = False


def snap_coordinates_to_road(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    *,
    provider: Optional[RoutingProvider] = None,
    provider_name: Optional[str] = None,
    client: Optional[Any] = None,
    profile: str = "driving-car",
    radii: Sequence[int] = SNAP_SEARCH_RADII,
) -> PinSnapResult:
    """Return the nearest routable coordinates for *origin* and *destination*."""

    resolved_provider = _resolve_provider(
        provider,
        provider_name=provider_name,
        client=client,
    )

    origin_lon, origin_lat = origin
    dest_lon, dest_lat = destination

    snap_result: Optional[SnapResult]
    try:
        snap_result = resolved_provider.snap_to_road(
            (float(origin_lon), float(origin_lat)),
            (float(dest_lon), float(dest_lat)),
            profile=profile,
            radii=radii,
        )
    except NotImplementedError:  # pragma: no cover - provider opts out explicitly
        snap_result = None

    if snap_result is None:
        return PinSnapResult(
            origin=(float(origin_lon), float(origin_lat)),
            destination=(float(dest_lon), float(dest_lat)),
            notes={},
            changed=False,
        )

    coords = list(snap_result.coordinates)
    if len(coords) >= 2:
        origin_lon, origin_lat = coords[0]
        dest_lon, dest_lat = coords[1]

    return PinSnapResult(
        origin=(float(origin_lon), float(origin_lat)),
        destination=(float(dest_lon), float(dest_lat)),
        notes=dict(snap_result.notes),
        changed=bool(snap_result.notes),
    )


def route_distance(
    conn: sqlite3.Connection,
    origin: str,
    destination: str,
    country: str,
    *,
    provider: Optional[RoutingProvider] = None,
    provider_name: Optional[str] = None,
    client: Optional[Any] = None,
    origin_override: Optional[Tuple[float, float]] = None,
    destination_override: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float, GeocodeResult, GeocodeResult]:
    resolved_provider = _resolve_provider(
        provider,
        provider_name=provider_name,
        client=client,
    )

    origin_geo = geocode_cached(
        conn, origin, country, provider=resolved_provider
    )
    dest_geo = geocode_cached(
        conn, destination, country, provider=resolved_provider
    )

    if origin_override is not None:
        try:
            override_lon, override_lat = origin_override
        except (TypeError, ValueError):
            override_lon, override_lat = origin_override or (None, None)
        else:
            origin_geo.lon = float(override_lon)
            origin_geo.lat = float(override_lat)
            _note_geocode(origin_geo, "Manual pin override used for routing")

    if destination_override is not None:
        try:
            dest_override_lon, dest_override_lat = destination_override
        except (TypeError, ValueError):
            dest_override_lon, dest_override_lat = destination_override or (None, None)
        else:
            dest_geo.lon = float(dest_override_lon)
            dest_geo.lat = float(dest_override_lat)
            _note_geocode(dest_geo, "Manual pin override used for routing")

    coordinates = [
        [origin_geo.lon, origin_geo.lat],
        [dest_geo.lon, dest_geo.lat],
    ]

    profile = "driving-car"

    try:
        route = resolved_provider.directions(coordinates=coordinates, profile=profile)
    except NoRoutablePointError as exc:  # pragma: no cover - exercised via tests
        logger.warning(
            "Routing provider could not find a routable point for %s → %s: %s",
            origin,
            destination,
            exc,
        )
    except IncompleteRouteError as exc:
        logger.warning(
            "Routing provider response missing distance/duration for %s → %s: %s",
            origin,
            destination,
            exc,
        )
    except RoutingError:
        raise
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise RoutingError(str(exc)) from exc
    else:
        time.sleep(ROUTE_BACKOFF)
        return route.distance_km, route.duration_hr, origin_geo, dest_geo

    snap_result: Optional[SnapResult]
    try:
        snap_result = resolved_provider.snap_to_road(
            (origin_geo.lon, origin_geo.lat),
            (dest_geo.lon, dest_geo.lat),
            profile=profile,
            radii=SNAP_SEARCH_RADII,
        )
    except NotImplementedError:
        snap_result = None

    if snap_result is not None:
        coords = list(snap_result.coordinates)
        if len(coords) >= 2:
            origin_geo.lon, origin_geo.lat = map(float, coords[0])
            dest_geo.lon, dest_geo.lat = map(float, coords[1])
        _note_geocode(origin_geo, snap_result.notes.get("origin"))
        _note_geocode(dest_geo, snap_result.notes.get("destination"))
        try:
            route = resolved_provider.directions(
                coordinates=[
                    [origin_geo.lon, origin_geo.lat],
                    [dest_geo.lon, dest_geo.lat],
                ],
                profile=profile,
            )
        except NoRoutablePointError as exc:
            logger.warning(
                "Snapped routing still failed for %s → %s: %s",
                origin,
                destination,
                exc,
            )
        except IncompleteRouteError as exc:
            logger.warning(
                "Snapped routing response missing distance/duration for %s → %s: %s",
                origin,
                destination,
                exc,
            )
        except RoutingError:
            raise
        except Exception as exc:  # pragma: no cover - defensive fallback
            raise RoutingError(str(exc)) from exc
        else:
            time.sleep(ROUTE_BACKOFF)
            return route.distance_km, route.duration_hr, origin_geo, dest_geo

    logger.warning(
        "Falling back to haversine estimate for %s → %s", origin, destination
    )
    distance_km = _haversine_km(
        origin_geo.lat,
        origin_geo.lon,
        dest_geo.lat,
        dest_geo.lon,
    )
    duration_hr = distance_km / FALLBACK_SPEED_KMH if distance_km > 0 else 0.0
    _note_geocode(origin_geo, "Used straight-line estimate due to missing road network")
    _note_geocode(dest_geo, "Used straight-line estimate due to missing road network")
    return distance_km, duration_hr, origin_geo, dest_geo


__all__ = [
    "COUNTRY_DEFAULT",
    "FALLBACK_SPEED_KMH",
    "GEOCODE_BACKOFF",
    "ROUTE_BACKOFF",
    "SNAP_SEARCH_RADII",
    "GoogleMapsRoutingProvider",
    "OpenRouteServiceProvider",
    "PinSnapResult",
    "RoutingProvider",
    "geocode_cached",
    "get_google_maps_client",
    "get_ors_client",
    "_ORS_CLIENT",
    "ors",
    "ors_exceptions",
    "normalize_place",
    "pelias_geocode",
    "route_distance",
    "snap_coordinates_to_road",
]
