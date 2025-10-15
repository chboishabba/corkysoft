"""Routing helpers built around OpenRouteService."""
from __future__ import annotations

import logging
import math
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

try:  # pragma: no cover - exercised indirectly via integration paths
    import openrouteservice as _ors
    from openrouteservice import exceptions as _ors_exceptions
except ModuleNotFoundError:  # pragma: no cover - behaviour verified via unit tests
    _ors = None
    _ors_exceptions = None

if TYPE_CHECKING:  # pragma: no cover - hints for type-checkers only
    import openrouteservice as ors  # noqa: F401
    from openrouteservice import exceptions as ors_exceptions  # noqa: F401
else:
    ors = _ors  # type: ignore[assignment]
    ors_exceptions = _ors_exceptions  # type: ignore[assignment]

from corkysoft.au_address import GeocodeResult, geocode_with_normalization

logger = logging.getLogger(__name__)

COUNTRY_DEFAULT = os.environ.get("ORS_COUNTRY", "Australia")
GEOCODE_BACKOFF = 0.2
ROUTE_BACKOFF = 0.2
FALLBACK_SPEED_KMH = 65.0
SNAP_SEARCH_RADII = (50, 150, 300, 750, 1500)

_ORS_CLIENT: Optional["ors.Client"] = None


def get_ors_client(client: Optional["ors.Client"] = None) -> "ors.Client":
    """Return an OpenRouteService client."""

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


def normalize_place(place: str) -> str:
    """Return a whitespace-normalised version of *place*."""

    return " ".join(place.strip().split())


def pelias_geocode(
    place: str, country: str, *, client: Optional["ors.Client"] = None
) -> GeocodeResult:
    resolved_client = get_ors_client(client)
    return geocode_with_normalization(resolved_client, place, country)


def geocode_cached(
    conn: sqlite3.Connection,
    place: str,
    country: str,
    *,
    client: Optional["ors.Client"] = None,
) -> GeocodeResult:
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

    result = pelias_geocode(norm, country, client=client)
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


def _is_routable_point_error(exc: Exception) -> bool:
    if ors_exceptions is not None and isinstance(exc, ors_exceptions.ApiError):
        args = getattr(exc, "args", ())
        for payload in (arg for arg in args if isinstance(arg, dict)):
            error = payload.get("error") or {}
            message = str(error.get("message") or "").lower()
            code = error.get("code")
            if code == 2010 or "could not find routable point" in message:
                return True

        for text_arg in (arg for arg in args if isinstance(arg, str)):
            if "could not find routable point" in text_arg.lower():
                return True

        if getattr(exc, "status_code", None) == 404:
            text = " ".join(str(arg) for arg in args)
            if "could not find routable point" in text.lower():
                return True
        return False

    args = " ".join(str(arg) for arg in getattr(exc, "args", ()))
    text = (args or str(exc)).lower()
    return (
        "could not find routable point" in text
        or '"code": 2010' in text
        or "'code': 2010" in text
    )


def _extract_snap_coordinates(response: object) -> Optional[Tuple[float, float]]:
    """Return coordinates from an ORS snap/nearest style *response*."""

    if not response:
        return None

    locations: Optional[Iterable[object]] = None
    if isinstance(response, dict):
        locations = response.get("locations") or response.get("features")
    elif isinstance(response, list):
        locations = response

    if not locations:
        return None

    first = next(iter(locations), None)
    if not first:
        return None

    coords: Optional[Iterable[float]] = None
    if isinstance(first, dict):
        coords = (
            first.get("location")
            or first.get("coordinates")
            or (
                first.get("geometry", {}).get("coordinates")
                if isinstance(first.get("geometry"), dict)
                else None
            )
        )
    elif isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
        coords = first

    if not coords:
        return None

    coords_list = list(coords)
    if len(coords_list) < 2:
        return None

    return float(coords_list[0]), float(coords_list[1])


def _snap_to_road(
    client: "ors.Client",
    origin_geo: GeocodeResult,
    dest_geo: GeocodeResult,
    *,
    profile: str = "driving-car",
    radii: Sequence[int] = SNAP_SEARCH_RADII,
) -> Optional[Tuple[List[List[float]], Dict[str, str]]]:
    """Attempt to snap unroutable coordinates to the nearest road."""

    def _snap_single(lon: float, lat: float) -> Optional[Tuple[float, float]]:
        snap_method = getattr(client, "snap", None)
        if callable(snap_method):
            for radius in radii:
                payload = {
                    "locations": [[lon, lat]],
                    "radius": radius,
                    "format": "json",
                }
                try:
                    response = snap_method(profile=profile, **payload)
                except TypeError:
                    try:
                        response = snap_method(
                            payload,
                            profile=profile,
                            format="json",
                        )
                    except Exception:  # pragma: no cover - defensive fallback
                        continue
                except Exception:  # pragma: no cover - upstream failure handled below
                    continue

                coords = _extract_snap_coordinates(response)
                if coords is None:
                    continue

                snapped_lon, snapped_lat = coords
                if snapped_lon == lon and snapped_lat == lat:
                    return None
                return snapped_lon, snapped_lat

        nearest_method = getattr(client, "nearest", None)
        if callable(nearest_method):
            try:
                response = nearest_method(coordinates=[[lon, lat]], number=1)
            except Exception:  # pragma: no cover - upstream failure handled by fallback
                return None

            coords = _extract_snap_coordinates(response)
            if coords is None:
                return None

            snapped_lon, snapped_lat = coords
            if snapped_lon == lon and snapped_lat == lat:
                return None
            return snapped_lon, snapped_lat

        return None

    notes: Dict[str, str] = {}
    snapped_origin = _snap_single(origin_geo.lon, origin_geo.lat)
    snapped_dest = _snap_single(dest_geo.lon, dest_geo.lat)

    changed = False
    if snapped_origin is not None:
        origin_geo.lon, origin_geo.lat = snapped_origin
        notes["origin"] = "Snapped to nearest routable road"
        changed = True
    if snapped_dest is not None:
        dest_geo.lon, dest_geo.lat = snapped_dest
        notes["destination"] = "Snapped to nearest routable road"
        changed = True

    if not changed:
        return None
    return [
        [origin_geo.lon, origin_geo.lat],
        [dest_geo.lon, dest_geo.lat],
    ], notes


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
    client: Optional["ors.Client"] = None,
    profile: str = "driving-car",
    radii: Sequence[int] = SNAP_SEARCH_RADII,
) -> PinSnapResult:
    """Return the nearest routable coordinates for *origin* and *destination*."""

    resolved_client = get_ors_client(client)

    if not hasattr(resolved_client, "snap") and not hasattr(resolved_client, "nearest"):
        raise RuntimeError(
            "Client does not support snapping via 'snap' or 'nearest' endpoints"
        )

    origin_lon, origin_lat = origin
    dest_lon, dest_lat = destination
    origin_geo = GeocodeResult(lon=float(origin_lon), lat=float(origin_lat))
    dest_geo = GeocodeResult(lon=float(dest_lon), lat=float(dest_lat))

    snapped = _snap_to_road(
        resolved_client,
        origin_geo,
        dest_geo,
        profile=profile,
        radii=radii,
    )
    notes: Dict[str, str] = {}
    changed = False
    if snapped is not None:
        _coords, notes = snapped
        changed = bool(notes)

    return PinSnapResult(
        origin=(float(origin_geo.lon), float(origin_geo.lat)),
        destination=(float(dest_geo.lon), float(dest_geo.lat)),
        notes=notes,
        changed=changed,
    )


def route_distance(
    conn: sqlite3.Connection,
    origin: str,
    destination: str,
    country: str,
    *,
    client: Optional["ors.Client"] = None,
    origin_override: Optional[Tuple[float, float]] = None,
    destination_override: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float, GeocodeResult, GeocodeResult]:
    resolved_client = get_ors_client(client)
    origin_geo = geocode_cached(
        conn, origin, country, client=resolved_client
    )
    dest_geo = geocode_cached(
        conn, destination, country, client=resolved_client
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
        route = resolved_client.directions(
            coordinates=coordinates,
            profile=profile,
            format="json",
        )
        summary = route["routes"][0]["summary"]
        meters = float(summary["distance"])
        seconds = float(summary["duration"])
        time.sleep(ROUTE_BACKOFF)
        return meters / 1000.0, seconds / 3600.0, origin_geo, dest_geo
    except Exception as exc:  # pragma: no cover - fallback behaviour tested below
        if not _is_routable_point_error(exc):
            raise
        logger.warning(
            "ORS could not find a routable point for %s → %s: %s", origin, destination, exc
        )

    snapped = _snap_to_road(
        resolved_client,
        origin_geo,
        dest_geo,
        profile=profile,
    )
    if snapped is not None:
        snapped_coords, snap_notes = snapped
        coordinates = snapped_coords
        _note_geocode(origin_geo, snap_notes.get("origin"))
        _note_geocode(dest_geo, snap_notes.get("destination"))
        try:
            route = resolved_client.directions(
                coordinates=coordinates,
                profile=profile,
                format="json",
            )
            summary = route["routes"][0]["summary"]
            meters = float(summary["distance"])
            seconds = float(summary["duration"])
            time.sleep(ROUTE_BACKOFF)
            return meters / 1000.0, seconds / 3600.0, origin_geo, dest_geo
        except Exception as exc:
            if not _is_routable_point_error(exc):
                raise
            logger.warning(
                "Snapped routing still failed for %s → %s: %s", origin, destination, exc
            )

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
    "PinSnapResult",
    "geocode_cached",
    "get_ors_client",
    "normalize_place",
    "pelias_geocode",
    "route_distance",
    "snap_coordinates_to_road",
]
