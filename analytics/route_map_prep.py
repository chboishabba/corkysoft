"""Helpers for route-map, heatmap, and isochrone visualisation prep."""
from __future__ import annotations

import logging
import math
from typing import Any, Literal, Optional, Sequence, TYPE_CHECKING

import numpy as np
import pandas as pd

from .profitability_map_prep import _coerce_float, _route_display_name
from .routing_provider import (
    OpenRouteServiceProvider,
    RoutingProvider,
    get_routing_provider,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from openrouteservice import Client as ORSClient
else:
    ORSClient = Any  # type: ignore[misc, assignment]


logger = logging.getLogger(__name__)

METRO_DISTANCE_THRESHOLD_KM = 100.0

HEATMAP_WEIGHTING_CANDIDATES: Sequence[tuple[str, Optional[str]]] = (
    ("Job count", None),
    ("Volume (m³)", "volume_m3"),
    ("Margin ($)", "margin_total"),
    ("Margin per m³", "margin_per_m3"),
    ("Margin %", "margin_total_pct"),
    ("Margin per m³ %", "margin_per_m3_pct"),
)


def prepare_route_map_data(
    df: pd.DataFrame,
    colour_column: str,
    *,
    placeholder: str = "Unknown",
) -> pd.DataFrame:
    """Return map-ready rows ensuring coordinates exist and colour labels are set."""

    if colour_column not in df.columns:
        raise KeyError(f"'{colour_column}' column is required to colour the map")

    required_columns = ["origin_lat", "origin_lon", "dest_lat", "dest_lon"]
    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        missing_str = ", ".join(missing_required)
        raise KeyError(f"Dataframe is missing required coordinate columns: {missing_str}")

    filtered = df.dropna(subset=required_columns).copy()
    colour_series = filtered[colour_column].fillna(placeholder)
    filtered["map_colour_value"] = colour_series.astype(str)
    filtered["map_colour_display"] = filtered["map_colour_value"]
    return filtered


def _format_metric_value(value: float, format_spec: str) -> str:
    """Return a human-friendly string for ``value`` based on ``format_spec``."""

    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "n/a"

    if format_spec == "currency":
        return f"${value:,.2f}"
    if format_spec == "currency_per_m3":
        return f"${value:,.2f}/m³"
    if format_spec == "percentage":
        return f"{value * 100:.1f}%"
    if format_spec == "volume":
        return f"{value:,.1f} m³"
    if format_spec == "distance":
        return f"{value:,.1f} km"
    if format_spec == "hours":
        return f"{value:,.1f} hr"

    return f"{value:,.2f}"


def compute_cost_vs_price_percentage(df: pd.DataFrame) -> pd.Series:
    """Return cost as a share of price expressed as a percentage ratio."""

    series_name = "cost_vs_price_pct"
    if df.empty:
        return pd.Series(dtype="float64", name=series_name)

    if "price_per_m3" not in df.columns or "final_cost_per_m3" not in df.columns:
        return pd.Series(
            np.nan,
            index=df.index,
            dtype="float64",
            name=series_name,
        )

    price_series = pd.to_numeric(df["price_per_m3"], errors="coerce")
    cost_series = pd.to_numeric(df["final_cost_per_m3"], errors="coerce")
    safe_denominator = price_series.replace({0: np.nan})
    ratio = cost_series.divide(safe_denominator)
    ratio = ratio.replace([math.inf, -math.inf], np.nan)
    ratio.name = series_name
    return ratio.astype("float64")


def prepare_metric_route_map_data(
    df: pd.DataFrame,
    metric_column: str,
    *,
    format_spec: str = "number",
) -> pd.DataFrame:
    """Return map rows with numeric metrics for continuous colouring."""

    if metric_column not in df.columns:
        raise KeyError(f"'{metric_column}' column is required to colour the map")

    required_columns = ["origin_lat", "origin_lon", "dest_lat", "dest_lon"]
    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        missing_str = ", ".join(missing_required)
        raise KeyError(f"Dataframe is missing required coordinate columns: {missing_str}")

    filtered = df.dropna(subset=required_columns).copy()
    numeric_series = pd.to_numeric(filtered[metric_column], errors="coerce")
    numeric_series = numeric_series.replace([math.inf, -math.inf], pd.NA)
    valid_mask = numeric_series.notna()
    filtered = filtered.loc[valid_mask].copy()
    if filtered.empty:
        return filtered

    numeric_values = numeric_series.loc[valid_mask].astype(float)
    filtered["map_colour_value"] = numeric_values
    filtered["map_colour_display"] = numeric_values.apply(
        lambda value: _format_metric_value(value, format_spec)
    )
    return filtered


def filter_jobs_by_distance(
    df: pd.DataFrame,
    *,
    metro_only: bool = False,
    threshold_km: float = METRO_DISTANCE_THRESHOLD_KM,
    max_distance_km: float | None = None,
) -> pd.DataFrame:
    """Filter jobs by distance when metro-only mode is requested."""

    if max_distance_km is not None:
        threshold_km = float(max_distance_km)

    if not metro_only or df.empty:
        return df.copy()

    candidate_columns = ("distance_km", "distance", "km", "kms")
    distance_column = next((col for col in candidate_columns if col in df.columns), None)
    if distance_column is None:
        return df.copy()

    distances = pd.to_numeric(df[distance_column], errors="coerce")
    mask = distances <= threshold_km
    filtered = df.loc[mask].copy()

    if "distance_km" not in filtered.columns and distance_column != "distance_km":
        filtered["distance_km"] = distances.loc[filtered.index]

    return filtered


def available_heatmap_weightings(df: pd.DataFrame) -> dict[str, Optional[str]]:
    """Return the heatmap weighting options available for the dataframe."""

    options: dict[str, Optional[str]] = {}
    for label, column in HEATMAP_WEIGHTING_CANDIDATES:
        if column is None or column in df.columns:
            options[label] = column
    return options


def build_heatmap_source(
    df: pd.DataFrame,
    weight_column: Optional[str] = None,
    *,
    metro_only: bool = False,
    threshold_km: float = METRO_DISTANCE_THRESHOLD_KM,
) -> pd.DataFrame:
    """Build a point-based dataframe suitable for density heatmaps."""

    if df.empty:
        return pd.DataFrame(columns=["lat", "lon", "weight"])

    scoped = filter_jobs_by_distance(
        df,
        metro_only=metro_only,
        threshold_km=threshold_km,
    )
    if scoped.empty:
        return pd.DataFrame(columns=["lat", "lon", "weight"])

    if weight_column is None:
        weights = pd.Series(1.0, index=scoped.index, dtype=float)
    else:
        if weight_column not in scoped.columns:
            raise KeyError(
                f"'{weight_column}' column is required for heatmap weighting"
            )
        weights = pd.to_numeric(scoped[weight_column], errors="coerce")

    coordinate_pairs = [
        ("origin_lat", "origin_lon"),
        ("dest_lat", "dest_lon"),
    ]

    frames: list[pd.DataFrame] = []
    for lat_column, lon_column in coordinate_pairs:
        if lat_column not in scoped.columns or lon_column not in scoped.columns:
            continue
        coords = scoped[[lat_column, lon_column]].copy()
        coords = coords.rename(columns={lat_column: "lat", lon_column: "lon"})
        coords["weight"] = weights
        coords = coords.dropna(subset=["lat", "lon"])
        coords["weight"] = pd.to_numeric(coords["weight"], errors="coerce")
        coords = coords.dropna(subset=["weight"])
        if not coords.empty:
            frames.append(coords)

    if not frames:
        return pd.DataFrame(columns=["lat", "lon", "weight"])

    result = pd.concat(frames, ignore_index=True)
    result["lat"] = pd.to_numeric(result["lat"], errors="coerce")
    result["lon"] = pd.to_numeric(result["lon"], errors="coerce")
    result = result.dropna(subset=["lat", "lon", "weight"])
    result["weight"] = result["weight"].astype(float)
    return result.reset_index(drop=True)


def _circle_coordinates(
    lat: float,
    lon: float,
    radius_km: float,
    *,
    points: int = 60,
) -> tuple[list[float], list[float]]:
    """Return an approximate circle around ``lat``/``lon`` with radius ``radius_km``."""

    if radius_km <= 0 or not math.isfinite(radius_km):
        return [], []

    lat_rad = math.radians(lat)
    cos_lat = math.cos(lat_rad)
    if abs(cos_lat) < 1e-6:
        cos_lat = 1e-6 if cos_lat >= 0 else -1e-6

    lat_deg_per_km = 1.0 / 110.574
    lon_deg_per_km = 1.0 / (111.320 * cos_lat)

    angles = np.linspace(0.0, 2.0 * math.pi, points, endpoint=False)
    lat_offsets = radius_km * np.sin(angles)
    lon_offsets = radius_km * np.cos(angles)

    latitudes = (lat + lat_offsets * lat_deg_per_km).tolist()
    longitudes = (lon + lon_offsets * lon_deg_per_km).tolist()

    if latitudes and longitudes:
        latitudes.append(latitudes[0])
        longitudes.append(longitudes[0])

    return latitudes, longitudes


def build_isochrone_polygons(
    df: pd.DataFrame,
    *,
    centre: Literal["origin", "destination"] = "origin",
    horizon_hours: float = 4.0,
    default_speed_kmh: float = 70.0,
    max_routes: int = 50,
    points: int = 60,
    routing_provider: Optional[RoutingProvider] = None,
    ors_client: Optional[ORSClient] = None,
    ors_profile: str = "driving-hgv",
    allow_approximate_fallback: bool = False,
) -> pd.DataFrame:
    """Return network-aware isochrone polygons for each route in ``df``."""

    empty = pd.DataFrame(
        columns=[
            "label",
            "centre_lat",
            "centre_lon",
            "radius_km",
            "speed_kmh",
            "latitudes",
            "longitudes",
            "geometry_source",
            "tooltip",
        ]
    )
    if df.empty:
        return empty

    centre_key = centre.lower()
    if centre_key not in {"origin", "destination"}:
        raise ValueError("centre must be 'origin' or 'destination'")

    lat_column = "origin_lat" if centre_key == "origin" else "dest_lat"
    lon_column = "origin_lon" if centre_key == "origin" else "dest_lon"
    if lat_column not in df.columns or lon_column not in df.columns:
        return empty

    distance_column = next(
        (column for column in ("distance_km", "distance", "km", "kms") if column in df.columns),
        None,
    )
    if distance_column is None:
        return empty

    duration_column = next(
        (
            column
            for column in ("duration_hr", "duration_hours", "travel_hours", "duration")
            if column in df.columns
        ),
        None,
    )

    providers: list[RoutingProvider] = []
    if routing_provider is not None:
        providers.append(routing_provider)
    else:
        try:
            providers.append(get_routing_provider(client=None))
        except Exception as exc:  # pragma: no cover
            logger.debug("Unable to initialise primary routing provider for isochrones: %s", exc)
        try:
            ors_provider = OpenRouteServiceProvider(ors_client)
        except Exception as exc:  # pragma: no cover
            logger.debug("Unable to initialise ORS fallback provider for isochrones: %s", exc)
            ors_provider = None
        if ors_provider is not None and not any(
            isinstance(provider, OpenRouteServiceProvider) for provider in providers
        ):
            providers.append(ors_provider)

    range_seconds: list[int] = []
    if horizon_hours > 0 and math.isfinite(horizon_hours):
        seconds = int(round(horizon_hours * 3600.0))
        if seconds > 0:
            range_seconds = [seconds]

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        lat_value = _coerce_float(row.get(lat_column))
        lon_value = _coerce_float(row.get(lon_column))
        if lat_value is None or lon_value is None:
            continue

        distance_value = _coerce_float(row.get(distance_column))
        if distance_value is None or distance_value <= 0:
            continue

        duration_value = _coerce_float(row.get(duration_column)) if duration_column else None
        if duration_value is not None and duration_value > 0:
            speed_kmh = distance_value / duration_value
        else:
            speed_kmh = default_speed_kmh

        if not math.isfinite(speed_kmh) or speed_kmh <= 0:
            speed_kmh = default_speed_kmh

        radius_km = speed_kmh * horizon_hours
        if radius_km <= 0 or not math.isfinite(radius_km):
            continue

        label = _route_display_name(row)

        latitudes: list[float]
        longitudes: list[float]
        geometry_source = "network"
        latitudes, longitudes = [], []
        if range_seconds:
            for provider in providers:
                try:
                    result = provider.isochrone(
                        centre=(float(lon_value), float(lat_value)),
                        profile=ors_profile,
                        range_seconds=range_seconds,
                    )
                except NotImplementedError:
                    result = None
                except Exception as exc:  # pragma: no cover
                    logger.debug(
                        "Routing provider isochrone request failed for %s via %s: %s",
                        label,
                        provider.__class__.__name__,
                        exc,
                    )
                    result = None
                if result:
                    latitudes, longitudes = result.to_lat_lon_lists()
                if latitudes and longitudes:
                    break
        if (not latitudes or not longitudes) and allow_approximate_fallback:
            latitudes, longitudes = _circle_coordinates(
                lat_value,
                lon_value,
                radius_km,
                points=points,
            )
            geometry_source = "approximate_circle"
        if not latitudes or not longitudes:
            continue

        tooltip = (
            f"{label} — {horizon_hours:.1f} hr reach ≈ {radius_km:.0f} km "
            f"(avg {speed_kmh:.0f} km/h)"
        )
        if geometry_source != "network":
            tooltip += " • approximate radius"

        records.append(
            {
                "label": label,
                "centre_lat": lat_value,
                "centre_lon": lon_value,
                "radius_km": radius_km,
                "speed_kmh": speed_kmh,
                "latitudes": latitudes,
                "longitudes": longitudes,
                "geometry_source": geometry_source,
                "tooltip": tooltip,
            }
        )
        if len(records) >= max_routes:
            break

    return pd.DataFrame.from_records(records)
