"""Helpers for switching map providers based on the routing configuration."""

from __future__ import annotations

import os
from urllib.parse import urlencode
from typing import Any, Dict, Mapping, Optional


def _resolved_provider() -> str:
    provider = os.environ.get("ROUTING_PROVIDER", "ors")
    return provider.strip().lower()


def using_google_maps() -> bool:
    """Return True when the active routing provider is Google."""

    return _resolved_provider() == "google"


def google_maps_api_key() -> Optional[str]:
    """Return the configured Google Maps API key if available."""

    key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not key:
        return None
    cleaned = key.strip()
    return cleaned or None


def street_view_available() -> bool:
    """Return True when Street View imagery can be requested."""

    return using_google_maps() and google_maps_api_key() is not None


def google_street_view_static_url(
    *,
    lat: float,
    lon: float,
    heading: float | None = None,
    pitch: float = 0.0,
    fov: int = 90,
    size: str = "640x360",
) -> Optional[str]:
    """Return a Google Street View Static API URL when available."""

    api_key = google_maps_api_key()
    if not using_google_maps() or not api_key:
        return None
    params = {
        "size": size,
        "location": f"{float(lat):.6f},{float(lon):.6f}",
        "pitch": f"{float(pitch):.1f}",
        "fov": str(int(fov)),
        "key": api_key,
    }
    if heading is not None:
        params["heading"] = f"{float(heading) % 360:.1f}"
    return "https://maps.googleapis.com/maps/api/streetview?" + urlencode(params)


def google_street_view_360_url(
    *,
    lat: float,
    lon: float,
    heading: float | None = None,
    pitch: float = 0.0,
    fov: int = 90,
) -> Optional[str]:
    """Return a Google Maps Street View URL when the Google provider is active."""

    api_key = google_maps_api_key()
    if not using_google_maps() or not api_key:
        return None
    params = {
        "api": "1",
        "map_action": "pano",
        "viewpoint": f"{float(lat):.6f},{float(lon):.6f}",
        "pitch": f"{float(pitch):.1f}",
        "fov": str(int(fov)),
    }
    if heading is not None:
        params["heading"] = f"{float(heading) % 360:.1f}"
    return "https://www.google.com/maps/@" + "?" + urlencode(params)


def _google_tile_layer(api_key: Optional[str]) -> Dict[str, Any]:
    token = f"&key={api_key}" if api_key else ""
    return {
        "sourcetype": "raster",
        "source": [f"https://mt1.google.com/vt/lyrs=m&x={{x}}&y={{y}}&z={{z}}{token}"],
        "below": "traces",
    }


def plotly_map_layout(
    center: Mapping[str, float],
    zoom: float,
    *,
    engine: str = "mapbox",
    default_style: str = "carto-positron",
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Return layout kwargs for Plotly map-based charts."""

    layout_key = "mapbox" if engine == "mapbox" else "map"
    payload: Dict[str, Any] = {
        "center": {"lat": float(center["lat"]), "lon": float(center["lon"])},
        "zoom": float(zoom),
    }

    if using_google_maps():
        api_key = google_maps_api_key()
        if api_key:
            payload["style"] = "white-bg"
            payload["layers"] = [_google_tile_layer(api_key)]
        else:
            payload["style"] = default_style
    else:
        payload["style"] = default_style

    if extra:
        payload.update(dict(extra))

    return {layout_key: payload}


def pydeck_map_kwargs(default_style: Optional[str]) -> Dict[str, Any]:
    """Return keyword arguments for pydeck Deck initialisation."""

    if using_google_maps():
        api_key = google_maps_api_key()
        if api_key:
            return {
                "map_provider": "google_maps",
                "map_style": None,
                "api_keys": {"google_maps": api_key},
            }

    kwargs: Dict[str, Any] = {}
    if default_style is not None:
        kwargs["map_style"] = default_style
    return kwargs


def folium_map_configuration(
    default_tiles: str = "OpenStreetMap",
    *,
    default_attr: Optional[str] = None,
) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Return base map kwargs and optional tile layer for Folium maps."""

    map_kwargs: Dict[str, Any] = {"tiles": default_tiles}
    if default_attr:
        map_kwargs["attr"] = default_attr

    tile_layer_kwargs: Optional[Dict[str, Any]] = None

    if using_google_maps():
        api_key = google_maps_api_key()
        if api_key:
            map_kwargs.pop("attr", None)
            map_kwargs["tiles"] = None
            tile_layer_kwargs = {
                "tiles": (
                    "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}&key="
                    f"{api_key}"
                ),
                "attr": "Google Maps",
                "name": "Google Maps",
                "overlay": False,
                "control": False,
            }

    return map_kwargs, tile_layer_kwargs
