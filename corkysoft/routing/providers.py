"""Routing provider abstractions for routing helpers."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Protocol, Sequence, Tuple, TYPE_CHECKING

from corkysoft.au_address import GeocodeResult, geocode_with_normalization

try:  # pragma: no cover - optional dependency
    import openrouteservice as _ors
    from openrouteservice import exceptions as _ors_exceptions
except ModuleNotFoundError:  # pragma: no cover - exercised in tests via monkeypatching
    _ors = None
    _ors_exceptions = None

try:  # pragma: no cover - optional dependency
    import googlemaps as _googlemaps
except ModuleNotFoundError:  # pragma: no cover - exercised in tests via monkeypatching
    _googlemaps = None

if TYPE_CHECKING:  # pragma: no cover - hints for type-checkers only
    import openrouteservice as ors
    from openrouteservice import exceptions as ors_exceptions
    import googlemaps
else:
    ors = _ors  # type: ignore[assignment]
    ors_exceptions = _ors_exceptions  # type: ignore[assignment]
    googlemaps = _googlemaps  # type: ignore[assignment]


logger = logging.getLogger(__name__)

Coordinate = Tuple[float, float]


class RoutingError(RuntimeError):
    """Base error for routing provider failures."""


class NoRoutablePointError(RoutingError):
    """Raised when the provider cannot find a routable point."""


class IncompleteRouteError(RoutingError):
    """Raised when a provider response lacks required metrics."""


@dataclass
class RouteResult:
    """Normalised routing response."""

    distance_km: float
    duration_hr: float
    raw: Optional[Mapping[str, Any]] = None


@dataclass
class SnapResult:
    """Coordinates returned by a snapping endpoint."""

    coordinates: Sequence[Sequence[float]]
    notes: Dict[str, str]


@dataclass
class IsochroneResult:
    """Normalised isochrone response."""

    raw: Mapping[str, Any]


class RoutingProvider(Protocol):
    """Protocol describing the routing capabilities needed by helpers."""

    def geocode(self, place: str, country: str) -> GeocodeResult:
        """Geocode ``place`` within ``country``."""

    def directions(
        self,
        coordinates: Sequence[Sequence[float]],
        profile: str = "driving-car",
    ) -> RouteResult:
        """Return routing metrics for ``coordinates``."""

    def snap_to_road(
        self,
        origin: Coordinate,
        destination: Coordinate,
        *,
        profile: str = "driving-car",
        radii: Sequence[int] | None = None,
    ) -> Optional[SnapResult]:
        """Snap coordinates to the nearest routable road if possible."""

    def isochrone(
        self,
        *,
        centre: Coordinate,
        profile: str,
        range_seconds: Sequence[int],
    ) -> Optional[IsochroneResult]:
        """Return isochrone payloads for the supplied ``centre`` if supported."""


_ORS_CLIENT: Optional["ors.Client"] = None


def get_ors_client(client: Optional["ors.Client"] = None) -> "ors.Client":
    """Return an OpenRouteService client, instantiating a shared singleton."""

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


_GOOGLE_CLIENT: Optional[Any] = None


def get_google_maps_client(client: Optional[Any] = None) -> Any:
    """Return a Google Maps client using ``GOOGLE_MAPS_API_KEY`` when necessary."""

    if client is not None:
        return client

    if googlemaps is None:
        raise RuntimeError(
            "googlemaps client is unavailable. Install the 'googlemaps' package to "
            "enable the Google routing provider."
        )

    global _GOOGLE_CLIENT
    if _GOOGLE_CLIENT is None:
        api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set GOOGLE_MAPS_API_KEY env var (export GOOGLE_MAPS_API_KEY=YOUR_KEY)"
            )
        _GOOGLE_CLIENT = googlemaps.Client(key=api_key)
    return _GOOGLE_CLIENT


def _extract_route_metrics(route_payload: Mapping[str, Any]) -> Optional[Tuple[float, float]]:
    """Extract distance (meters) and duration (seconds) from a routing response."""

    routes = route_payload.get("routes")
    if not routes:
        return None

    first_route = routes[0] or {}
    summary = first_route.get("summary") or {}
    distance: Any = summary.get("distance")
    duration: Any = summary.get("duration")

    if distance is None or duration is None:
        for segment in first_route.get("segments", []) or []:
            if distance is None:
                distance = segment.get("distance", distance)
            if duration is None:
                duration = segment.get("duration", duration)
            if distance is not None and duration is not None:
                break

    if distance is None or duration is None:
        return None

    try:
        return float(distance), float(duration)
    except (TypeError, ValueError):
        return None


def _is_routable_point_error(exc: Exception) -> bool:
    """Return whether *exc* describes a non-routable coordinate error."""

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
    """Return coordinates from a snap/nearest style response."""

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


class OpenRouteServiceProvider:
    """Routing provider backed by OpenRouteService."""

    def __init__(self, client: Optional[Any] = None):
        self._client = get_ors_client(client)

    def geocode(self, place: str, country: str) -> GeocodeResult:
        return geocode_with_normalization(self._client, place, country)

    def directions(
        self,
        coordinates: Sequence[Sequence[float]],
        profile: str = "driving-car",
    ) -> RouteResult:
        try:
            route = self._client.directions(
                coordinates=coordinates,
                profile=profile,
                format="json",
            )
        except Exception as exc:  # pragma: no cover - exercised in tests
            if _is_routable_point_error(exc):
                raise NoRoutablePointError(str(exc)) from exc
            raise RoutingError(str(exc)) from exc

        metrics = _extract_route_metrics(route)
        if metrics is None:
            raise IncompleteRouteError(
                "ORS route response missing distance or duration"
            )

        meters, seconds = metrics
        return RouteResult(
            distance_km=meters / 1000.0,
            duration_hr=seconds / 3600.0,
            raw=route,
        )

    def snap_to_road(
        self,
        origin: Coordinate,
        destination: Coordinate,
        *,
        profile: str = "driving-car",
        radii: Sequence[int] | None = None,
    ) -> Optional[SnapResult]:
        radii = tuple(radii or ())

        def _snap_single(lon: float, lat: float) -> Optional[Tuple[float, float]]:
            snap_method = getattr(self._client, "snap", None)
            if callable(snap_method):
                for radius in radii or ():
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

            nearest_method = getattr(self._client, "nearest", None)
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
        origin_lon, origin_lat = origin
        dest_lon, dest_lat = destination
        snapped_origin = _snap_single(origin_lon, origin_lat)
        snapped_dest = _snap_single(dest_lon, dest_lat)

        changed = False
        if snapped_origin is not None:
            origin_lon, origin_lat = snapped_origin
            notes["origin"] = "Snapped to nearest routable road"
            changed = True
        if snapped_dest is not None:
            dest_lon, dest_lat = snapped_dest
            notes["destination"] = "Snapped to nearest routable road"
            changed = True

        if not changed:
            return None
        return SnapResult(
            coordinates=[[origin_lon, origin_lat], [dest_lon, dest_lat]],
            notes=notes,
        )

    def isochrone(
        self,
        *,
        centre: Coordinate,
        profile: str,
        range_seconds: Sequence[int],
    ) -> Optional[IsochroneResult]:
        if not range_seconds:
            return None
        response = self._client.isochrones(
            locations=[[centre[0], centre[1]]],
            profile=profile,
            range=list(range_seconds),
        )
        if not isinstance(response, Mapping):
            raise RoutingError("ORS isochrone response must be a mapping")
        return IsochroneResult(raw=response)


class GoogleMapsRoutingProvider:
    """Routing provider backed by Google Maps Platform."""

    def __init__(self, client: Optional[Any] = None):
        self._client = get_google_maps_client(client)

    def geocode(self, place: str, country: str) -> GeocodeResult:
        response = self._client.geocode(place, components={"country": country})
        if not response:
            raise RoutingError("Google geocode returned no results")
        first = response[0]
        geometry = first.get("geometry") or {}
        location = geometry.get("location") or {}
        try:
            lon = float(location["lng"])
            lat = float(location["lat"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RoutingError("Google geocode missing coordinates") from exc

        formatted = first.get("formatted_address")
        components = {tuple(comp.get("types", [])): comp for comp in first.get("address_components", []) if isinstance(comp, dict)}

        def _component(name: str, short: bool = False) -> Optional[str]:
            for types, comp in components.items():
                if name in types:
                    value = comp.get("short_name" if short else "long_name")
                    return str(value) if value is not None else None
            return None

        postalcode = _component("postal_code")
        region = _component("administrative_area_level_1")
        region_code = _component("administrative_area_level_1", short=True)
        locality = _component("locality") or _component("postal_town")
        county = _component("administrative_area_level_2")

        return GeocodeResult(
            lon=lon,
            lat=lat,
            label=formatted,
            search_candidates=[place],
            postalcode=postalcode,
            region=region,
            region_code=region_code,
            locality=locality,
            county=county,
        )

    def directions(
        self,
        coordinates: Sequence[Sequence[float]],
        profile: str = "driving-car",
    ) -> RouteResult:
        if len(coordinates) < 2:
            raise RoutingError("Google directions requires origin and destination")
        origin = coordinates[0]
        destination = coordinates[-1]
        mode = "driving"
        if profile == "cycling-regular":
            mode = "bicycling"
        elif profile == "foot-walking":
            mode = "walking"

        payload = self._client.directions(
            origin=origin,
            destination=destination,
            mode=mode,
        )
        if not payload:
            raise RoutingError("Google directions returned no routes")
        first_route = payload[0]
        legs = first_route.get("legs") if isinstance(first_route, Mapping) else None
        if not legs:
            raise RoutingError("Google directions missing legs")
        distance_m = 0.0
        duration_s = 0.0
        for leg in legs:
            if not isinstance(leg, Mapping):
                continue
            distance = leg.get("distance") or {}
            duration = leg.get("duration") or {}
            try:
                distance_m += float(distance.get("value", 0.0))
                duration_s += float(duration.get("value", 0.0))
            except (TypeError, ValueError):
                continue
        if not distance_m or not duration_s:
            raise IncompleteRouteError(
                "Google directions missing distance or duration"
            )
        overview = first_route.get("overview_polyline") if isinstance(first_route, Mapping) else None
        encoded = overview.get("points") if isinstance(overview, Mapping) else None
        return RouteResult(
            distance_km=distance_m / 1000.0,
            duration_hr=duration_s / 3600.0,
            raw={"routes": payload, "encoded_polyline": encoded},
        )

    def snap_to_road(
        self,
        origin: Coordinate,
        destination: Coordinate,
        *,
        profile: str = "driving-car",
        radii: Sequence[int] | None = None,
    ) -> Optional[SnapResult]:
        if not hasattr(self._client, "snap_to_roads"):
            return None
        path = [origin, destination]
        response = self._client.snap_to_roads(path=path, interpolate=False)
        if not response:
            return None
        coords: list[list[float]] = []
        for point in response:
            location = point.get("location") if isinstance(point, Mapping) else None
            if not location:
                continue
            lat = location.get("latitude")
            lon = location.get("longitude")
            if lat is None or lon is None:
                continue
            coords.append([float(lon), float(lat)])
        if len(coords) < 2:
            return None
        notes = {
            "origin": "Snapped to nearest routable road",
            "destination": "Snapped to nearest routable road",
        }
        return SnapResult(coordinates=coords[:2], notes=notes)

    def isochrone(
        self,
        *,
        centre: Coordinate,
        profile: str,
        range_seconds: Sequence[int],
    ) -> Optional[IsochroneResult]:
        if not hasattr(self._client, "isochrones"):
            raise NotImplementedError("Google provider does not expose isochrones")
        response = self._client.isochrones(
            centre=centre,
            profile=profile,
            range_seconds=list(range_seconds),
        )
        if not isinstance(response, Mapping):
            raise RoutingError("Google isochrone response must be a mapping")
        return IsochroneResult(raw=response)
