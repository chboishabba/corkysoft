"""Helpers for switching map providers based on the routing configuration."""

from __future__ import annotations

import os
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


def _google_tile_layer(api_key: Optional[str]) -> Dict[str, Any]:
    token = f"&key={api_key}" if api_key else ""
    return {
        "sourcetype": "raster",
        "source": [f"https://mt1.google.com/vt/lyrs=m&x={{x}}&y={{y}}&z={{z}}{token}"],
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
