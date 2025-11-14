"""Routing provider abstraction with normalised geometry helpers."""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence

from corkysoft.routing import ROUTE_BACKOFF, get_ors_client

CoordinatePair = tuple[float, float]


class RoutingProvider(Protocol):
    """Protocol describing the routing provider interface."""

    def route_geometry(
        self,
        *,
        origin: CoordinatePair,
        destination: CoordinatePair,
        profile: str = "driving-car",
    ) -> "RouteGeometryResult":
        """Return a normalised route geometry payload."""

    def isochrone(
        self,
        *,
        centre: CoordinatePair,
        profile: str,
        range_seconds: Sequence[int],
    ) -> Optional["IsochroneResult"]:
        """Return network-aware isochrones when supported."""


@dataclass
class RouteGeometryResult:
    """Normalised representation of a routing response."""

    distance_km: float
    duration_hr: float
    feature_collection: Optional[Mapping[str, Any]] = None
    coordinates: Optional[Sequence[CoordinatePair]] = None
    encoded_polyline: Optional[str] = None

    def to_feature_collection(self) -> Mapping[str, Any]:
        """Return a GeoJSON feature collection for the stored geometry."""

        if self.feature_collection:
            # Materialise as a dict to detach from the provider's payload.
            return json.loads(json.dumps(self.feature_collection))

        coordinates = list(self.coordinates or [])
        if not coordinates and self.encoded_polyline:
            coordinates = [
                (lon, lat)
                for lat, lon in decode_polyline(self.encoded_polyline)
            ]

        if not coordinates:
            raise ValueError("Provider response did not include geometry coordinates")

        geojson_coords = [[float(lon), float(lat)] for lon, lat in coordinates]
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "LineString",
                        "coordinates": geojson_coords,
                    },
                }
            ],
        }

    def dumps(self) -> str:
        """Return the feature collection as a compact JSON string."""

        return json.dumps(self.to_feature_collection(), separators=(",", ":"))


@dataclass
class IsochroneResult:
    """Normalised representation of an isochrone response."""

    feature_collection: Optional[Mapping[str, Any]] = None
    coordinates: Optional[Sequence[CoordinatePair]] = None
    encoded_polylines: Optional[Sequence[str]] = None

    def to_lat_lon_lists(self) -> tuple[list[float], list[float]]:
        """Return ``([latitudes], [longitudes])`` for the first polygon."""

        if self.coordinates:
            return _coordinates_to_lists(self.coordinates)

        if self.feature_collection:
            latitudes, longitudes = feature_collection_to_lat_lon(self.feature_collection)
            if latitudes and longitudes:
                return latitudes, longitudes

        for polyline in self.encoded_polylines or []:
            decoded = decode_polyline(polyline)
            if not decoded:
                continue
            latitudes = [float(lat) for lat, _lon in decoded]
            longitudes = [float(lon) for _lat, lon in decoded]
            if latitudes and longitudes and (
                latitudes[0] != latitudes[-1] or longitudes[0] != longitudes[-1]
            ):
                latitudes.append(latitudes[0])
                longitudes.append(longitudes[0])
            return latitudes, longitudes

        return [], []


class OpenRouteServiceProvider:
    """Routing provider backed by OpenRouteService."""

    def __init__(self, client: Optional[Any] = None):
        self._client = get_ors_client(client)

    def route_geometry(
        self,
        *,
        origin: CoordinatePair,
        destination: CoordinatePair,
        profile: str = "driving-car",
    ) -> RouteGeometryResult:
        response = self._client.directions(
            coordinates=[
                [float(origin[0]), float(origin[1])],
                [float(destination[0]), float(destination[1])],
            ],
            profile=profile,
            format="geojson",
        )
        if not isinstance(response, Mapping):
            raise ValueError("ORS response must be a mapping")
        feature_collection = _ensure_feature_collection(response)
        features = feature_collection.get("features")
        if not isinstance(features, Sequence) or not features:
            raise ValueError("ORS response missing features for route geometry")
        first_feature = features[0]
        if not isinstance(first_feature, Mapping):
            raise ValueError("ORS response feature must be a mapping")
        properties = first_feature.get("properties")
        if not isinstance(properties, Mapping):
            raise ValueError("ORS route feature missing properties")
        summary = properties.get("summary")
        if not isinstance(summary, Mapping):
            raise ValueError("ORS route feature missing summary")
        try:
            distance_m = float(summary["distance"])
            duration_s = float(summary["duration"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("ORS summary missing distance or duration") from exc

        time.sleep(ROUTE_BACKOFF)
        return RouteGeometryResult(
            distance_km=distance_m / 1000.0,
            duration_hr=duration_s / 3600.0,
            feature_collection=feature_collection,
        )

    def isochrone(
        self,
        *,
        centre: CoordinatePair,
        profile: str,
        range_seconds: Sequence[int],
    ) -> Optional[IsochroneResult]:
        if not range_seconds:
            return None
        response = self._client.isochrones(
            locations=[[float(centre[0]), float(centre[1])]],
            profile=profile,
            range=list(range_seconds),
        )
        if not isinstance(response, Mapping):
            raise ValueError("ORS isochrone response must be a mapping")
        feature_collection = _ensure_feature_collection(response)
        return IsochroneResult(feature_collection=feature_collection)


class GoogleRoutesProvider:
    """Routing provider backed by Google Maps Platform."""

    def __init__(self, client: Any):
        if client is None:
            raise RuntimeError("Google routing provider requires a client instance")
        self._client = client

    def route_geometry(
        self,
        *,
        origin: CoordinatePair,
        destination: CoordinatePair,
        profile: str = "driving-car",
    ) -> RouteGeometryResult:
        payload = self._invoke_directions(origin, destination, profile)
        distance_m, duration_s, encoded, coordinates = _parse_google_route(payload)
        return RouteGeometryResult(
            distance_km=distance_m / 1000.0,
            duration_hr=duration_s / 3600.0,
            encoded_polyline=encoded,
            coordinates=coordinates,
        )

    def isochrone(
        self,
        *,
        centre: CoordinatePair,
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
        feature_collection: Optional[Mapping[str, Any]] = None
        encoded: list[str] = []

        if isinstance(response, Mapping):
            if response.get("type") == "FeatureCollection":
                feature_collection = _ensure_feature_collection(response)
            else:
                for key in ("encoded_polylines", "polylines", "paths"):
                    value = response.get(key)
                    if isinstance(value, Sequence):
                        encoded.extend(str(item) for item in value if item is not None)
                        break
        elif isinstance(response, Sequence):
            encoded.extend(str(item) for item in response if item is not None)

        if feature_collection is None and not encoded:
            return None
        return IsochroneResult(
            feature_collection=feature_collection,
            encoded_polylines=encoded or None,
        )

    def _invoke_directions(
        self,
        origin: CoordinatePair,
        destination: CoordinatePair,
        profile: str,
    ) -> Any:
        if hasattr(self._client, "directions"):
            return self._client.directions(
                origin=origin,
                destination=destination,
                profile=profile,
            )
        if hasattr(self._client, "compute_routes"):
            return self._client.compute_routes(
                origin=origin,
                destination=destination,
                profile=profile,
            )
        raise RuntimeError("Google routing client does not provide a directions method")


def decode_polyline(polyline: str) -> list[CoordinatePair]:
    """Decode an encoded polyline string into ``(lat, lon)`` pairs."""

    coordinates: list[CoordinatePair] = []
    index = 0
    lat = 0
    lng = 0
    length = len(polyline)

    while index < length:
        for coord in ("lat", "lng"):
            result = 0
            shift = 0
            while True:
                if index >= length:
                    raise ValueError("Invalid polyline encoding")
                b = ord(polyline[index]) - 63
                index += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(result >> 1) if result & 1 else (result >> 1)
            if coord == "lat":
                lat += delta
            else:
                lng += delta
        coordinates.append((lat / 1e5, lng / 1e5))

    return coordinates


def geometry_to_lat_lon(geometry: Any) -> tuple[list[float], list[float]]:
    """Return lat/lon lists from a GeoJSON geometry mapping."""

    if not isinstance(geometry, Mapping):
        return [], []

    gtype = geometry.get("type")
    coords = geometry.get("coordinates")

    if gtype == "Polygon" and isinstance(coords, Sequence):
        for ring in coords:
            latitudes, longitudes = _extract_ring(ring)
            if latitudes and longitudes:
                return latitudes, longitudes
    elif gtype == "MultiPolygon" and isinstance(coords, Sequence):
        for polygon in coords:
            for ring in polygon or []:
                latitudes, longitudes = _extract_ring(ring)
                if latitudes and longitudes:
                    return latitudes, longitudes
    elif gtype == "LineString" and isinstance(coords, Sequence):
        latitudes, longitudes = _extract_ring(coords)
        if latitudes and longitudes:
            return latitudes, longitudes
    elif gtype == "MultiLineString" and isinstance(coords, Sequence):
        for line in coords:
            latitudes, longitudes = _extract_ring(line)
            if latitudes and longitudes:
                return latitudes, longitudes

    return [], []


def feature_collection_to_lat_lon(feature_collection: Any) -> tuple[list[float], list[float]]:
    """Return the first polygon ring from a GeoJSON feature collection."""

    if not isinstance(feature_collection, Mapping):
        return [], []

    features = feature_collection.get("features")
    if not isinstance(features, Sequence):
        return [], []

    sortable: list[tuple[float, Mapping[str, Any]]] = []
    for feature in features:
        if not isinstance(feature, Mapping):
            continue
        properties = feature.get("properties")
        if isinstance(properties, Mapping) and "value" in properties:
            try:
                sortable.append((float(properties["value"]), feature))
            except (TypeError, ValueError):
                sortable.append((float("inf"), feature))
        else:
            sortable.append((float("inf"), feature))

    for _, feature in sorted(sortable, key=lambda item: item[0]):
        latitudes, longitudes = geometry_to_lat_lon(feature.get("geometry"))
        if latitudes and longitudes:
            return latitudes, longitudes

    for feature in features:
        if not isinstance(feature, Mapping):
            continue
        latitudes, longitudes = geometry_to_lat_lon(feature.get("geometry"))
        if latitudes and longitudes:
            return latitudes, longitudes

    return [], []


def get_routing_provider(
    *,
    provider: Optional[RoutingProvider] = None,
    client: Optional[Any] = None,
) -> RoutingProvider:
    """Return a routing provider based on the environment configuration."""

    if provider is not None:
        return provider

    provider_name = os.environ.get("ROUTING_PROVIDER", "ors").strip().lower()
    if provider_name == "google":
        return GoogleRoutesProvider(client)

    return OpenRouteServiceProvider(client)


def _ensure_feature_collection(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return payload as a GeoJSON feature collection."""

    if payload.get("type") == "FeatureCollection":
        return json.loads(json.dumps(payload))
    if payload.get("type") == "Feature":
        return {
            "type": "FeatureCollection",
            "features": [json.loads(json.dumps(payload))],
        }
    raise ValueError("Unsupported GeoJSON payload")


def _extract_ring(ring: Iterable[Any]) -> tuple[list[float], list[float]]:
    latitudes: list[float] = []
    longitudes: list[float] = []
    if not isinstance(ring, Sequence):
        return latitudes, longitudes

    for coord in ring:
        if not isinstance(coord, Sequence) or len(coord) < 2:
            continue
        lon, lat = coord[0], coord[1]
        try:
            longitudes.append(float(lon))
            latitudes.append(float(lat))
        except (TypeError, ValueError):
            continue

    if not latitudes or not longitudes:
        return [], []

    if latitudes[0] != latitudes[-1] or longitudes[0] != longitudes[-1]:
        latitudes.append(latitudes[0])
        longitudes.append(longitudes[0])

    return latitudes, longitudes


def _coordinates_to_lists(coordinates: Sequence[CoordinatePair]) -> tuple[list[float], list[float]]:
    latitudes = [float(lat) for lat, _lon in coordinates]
    longitudes = [float(lon) for _lat, lon in coordinates]
    if latitudes and longitudes and (
        latitudes[0] != latitudes[-1] or longitudes[0] != longitudes[-1]
    ):
        latitudes.append(latitudes[0])
        longitudes.append(longitudes[0])
    return latitudes, longitudes


def _parse_google_route(payload: Any) -> tuple[float, float, Optional[str], Optional[Sequence[CoordinatePair]]]:
    """Extract distance, duration and path information from Google responses."""

    if not isinstance(payload, Mapping):
        raise ValueError("Google directions response must be a mapping")

    route: Mapping[str, Any]
    routes = payload.get("routes")
    if isinstance(routes, Sequence) and routes:
        first = routes[0]
        if not isinstance(first, Mapping):
            raise ValueError("Google routes entry must be a mapping")
        route = first
    else:
        route = payload

    distance_m = _extract_google_distance(route)
    duration_s = _extract_google_duration(route)
    encoded = _extract_google_polyline(route)
    coordinates = _extract_google_coordinates(route)
    return distance_m, duration_s, encoded, coordinates


def _extract_google_distance(route: Mapping[str, Any]) -> float:
    if "distanceMeters" in route:
        return float(route["distanceMeters"])

    legs = route.get("legs")
    if isinstance(legs, Sequence) and legs:
        total = 0.0
        for leg in legs:
            if isinstance(leg, Mapping):
                if "distanceMeters" in leg:
                    total += float(leg["distanceMeters"])
                else:
                    distance = leg.get("distance")
                    if isinstance(distance, Mapping) and "value" in distance:
                        total += float(distance["value"])
        if total > 0:
            return total

    distance = route.get("distance")
    if isinstance(distance, Mapping) and "value" in distance:
        return float(distance["value"])
    if isinstance(distance, str):
        if distance.endswith("m"):
            return float(distance[:-1])
        if distance.endswith("km"):
            return float(distance[:-2]) * 1000.0

    raise ValueError("Google route missing distance information")


def _extract_google_duration(route: Mapping[str, Any]) -> float:
    duration = route.get("duration")
    if isinstance(duration, Mapping) and "value" in duration:
        return float(duration["value"])
    if isinstance(duration, str):
        if duration.endswith("s"):
            return float(duration[:-1])
        if duration.endswith("m"):
            return float(duration[:-1]) * 60.0
        if duration.endswith("h"):
            return float(duration[:-1]) * 3600.0

    legs = route.get("legs")
    if isinstance(legs, Sequence) and legs:
        total = 0.0
        for leg in legs:
            if isinstance(leg, Mapping):
                leg_duration = leg.get("duration")
                if isinstance(leg_duration, Mapping) and "value" in leg_duration:
                    total += float(leg_duration["value"])
        if total > 0:
            return total

    raise ValueError("Google route missing duration information")


def _extract_google_polyline(route: Mapping[str, Any]) -> Optional[str]:
    polyline = route.get("polyline")
    if isinstance(polyline, Mapping):
        if "encodedPolyline" in polyline:
            return str(polyline["encodedPolyline"])
        if "points" in polyline:
            return str(polyline["points"])

    overview = route.get("overview_polyline")
    if isinstance(overview, Mapping) and "points" in overview:
        return str(overview["points"])

    if "polyline" in route and isinstance(route["polyline"], str):
        return str(route["polyline"])

    return None


def _extract_google_coordinates(route: Mapping[str, Any]) -> Optional[Sequence[CoordinatePair]]:
    polyline = route.get("polyline")
    if isinstance(polyline, Mapping) and "geoJsonLinestring" in polyline:
        geojson = polyline["geoJsonLinestring"]
        if isinstance(geojson, Mapping) and geojson.get("type") == "LineString":
            coords = geojson.get("coordinates")
            if isinstance(coords, Sequence):
                extracted: list[CoordinatePair] = []
                for coord in coords:
                    if not isinstance(coord, Sequence) or len(coord) < 2:
                        continue
                    lon, lat = coord[0], coord[1]
                    try:
                        extracted.append((float(lon), float(lat)))
                    except (TypeError, ValueError):
                        continue
                if extracted:
                    return extracted
    return None
