"""Lane cost calculator implementing Corkysoft business rules."""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .schema import ensure_schema

EARTH_RADIUS_KM = 6371.0


@dataclass
class MetroCheck:
    """Result of evaluating the metro rule for a coordinate."""

    is_metro: bool
    depot_code: Optional[str]
    depot_name: Optional[str]
    distance_km: Optional[float]


@dataclass
class QuoteBreakdown:
    """Explainable line items for a computed quote."""

    pricing_model: str
    base_description: str
    base_amount: float
    modifiers: List[Dict[str, object]] = field(default_factory=list)
    seasonality: Optional[Dict[str, object]] = None
    margin: Optional[Dict[str, object]] = None
    total: float = 0.0
    metro_rule: Dict[str, MetroCheck] = field(default_factory=dict)


class LaneCostCalculator:
    """Calculate lane pricing by combining table-driven rules and modifiers."""

    def __init__(self, db_path: str = "routes.db", connection: Optional[sqlite3.Connection] = None) -> None:
        if connection is None:
            connection = sqlite3.connect(db_path)
            self._owns_connection = True
        else:
            self._owns_connection = False
        self.conn = connection
        self.conn.row_factory = sqlite3.Row
        ensure_schema(self.conn)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._owns_connection:
            self.conn.close()

    def __del__(self) -> None:  # pragma: no cover - defensive close
        try:
            self.close()
        except Exception:
            pass

    def _fetchone(self, query: str, params: Sequence[object]) -> Optional[sqlite3.Row]:
        cur = self.conn.execute(query, params)
        return cur.fetchone()

    def _fetchall(self, query: str, params: Sequence[object] = ()) -> List[sqlite3.Row]:
        cur = self.conn.execute(query, params)
        return cur.fetchall()

    # ------------------------------------------------------------------
    # Metro evaluation
    # ------------------------------------------------------------------
    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
        lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return EARTH_RADIUS_KM * c

    def _nearest_depot(self, lat: float, lon: float) -> MetroCheck:
        depots = self._fetchall(
            "SELECT depot_code, name, latitude, longitude, radius_km FROM depot_metro_zones"
        )
        closest: Optional[Tuple[str, str, float]] = None
        for depot in depots:
            distance_km = self._haversine_km(lat, lon, depot["latitude"], depot["longitude"])
            if closest is None or distance_km < closest[2]:
                closest = (depot["depot_code"], depot["name"], distance_km)
        if closest is None:
            return MetroCheck(False, None, None, None)
        depot_code, depot_name, distance_km = closest
        depot_row = next(d for d in depots if d["depot_code"] == depot_code)
        is_metro = distance_km <= depot_row["radius_km"]
        return MetroCheck(is_metro, depot_code, depot_name, distance_km)

    def evaluate_metro_rule(
        self, origin_lat: float, origin_lon: float, dest_lat: float, dest_lon: float
    ) -> Dict[str, MetroCheck]:
        origin_check = self._nearest_depot(origin_lat, origin_lon)
        dest_check = self._nearest_depot(dest_lat, dest_lon)
        return {"origin": origin_check, "destination": dest_check}

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------
    def _get_lane_rate(self, corridor_code: str) -> sqlite3.Row:
        lane = self._fetchone(
            "SELECT * FROM lane_base_rates WHERE corridor_code = ?", (corridor_code,)
        )
        if lane is None:
            raise ValueError(f"Unknown corridor code: {corridor_code}")
        return lane

    def _get_modifier(self, code: str) -> sqlite3.Row:
        row = self._fetchone(
            "SELECT * FROM modifier_fees WHERE code = ? AND active = 1", (code,)
        )
        if row is None:
            raise ValueError(f"Modifier '{code}' is not defined or inactive")
        return row

    def _get_packing_rate(self, service_code: str, volume_m3: float) -> Optional[sqlite3.Row]:
        row = self._fetchone(
            """
            SELECT * FROM packing_rate_tiers
            WHERE service_code = ?
              AND min_volume <= ?
              AND (max_volume IS NULL OR max_volume > ?)
            ORDER BY min_volume DESC
            LIMIT 1
            """,
            (service_code, volume_m3, volume_m3),
        )
        return row

    def _get_seasonal_uplift(self, move_date: date) -> Optional[sqlite3.Row]:
        all_rows = self._fetchall(
            "SELECT * FROM seasonal_uplifts WHERE active = 1 ORDER BY uplift_pct DESC"
        )
        if not all_rows:
            return None
        month_day = (move_date.month, move_date.day)
        for row in all_rows:
            start = (row["start_month"], row["start_day"])
            end = (row["end_month"], row["end_day"])
            if start <= end:
                if start <= month_day <= end:
                    return row
            else:  # season that wraps year-end
                if month_day >= start or month_day <= end:
                    return row
        return None

    # ------------------------------------------------------------------
    # Quoting
    # ------------------------------------------------------------------
    def calculate_quote(
        self,
        *,
        corridor_code: str,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        distance_km: float,
        duration_hours: float,
        volume_m3: float,
        move_date: date,
        modifier_codes: Optional[Iterable[str]] = None,
        packing_services: Optional[Iterable[str]] = None,
        packing_volume_m3: Optional[float] = None,
        margin_pct: float = 0.0,
    ) -> QuoteBreakdown:
        """Return an explainable cost breakdown for a job."""

        modifier_codes = list(modifier_codes or [])
        packing_services = list(packing_services or [])
        packing_volume = packing_volume_m3 if packing_volume_m3 is not None else volume_m3

        lane = self._get_lane_rate(corridor_code)
        metro_checks = self.evaluate_metro_rule(origin_lat, origin_lon, dest_lat, dest_lon)
        both_metro = all(check.is_metro for check in metro_checks.values())
        pricing_model = (
            "hourly" if both_metro and distance_km < 200 else "per_m3"
        )

        breakdown = QuoteBreakdown(
            pricing_model=pricing_model,
            base_description="",
            base_amount=0.0,
            metro_rule=metro_checks,
        )

        if pricing_model == "hourly":
            base_amount = lane["metro_hourly_rate"] * duration_hours
            base_description = (
                f"Metro hourly @ ${lane['metro_hourly_rate']:.2f}/hr × {duration_hours:.2f} h"
            )
        else:
            base_amount = lane["per_m3_rate"] * volume_m3
            base_description = (
                f"Lane {corridor_code} @ ${lane['per_m3_rate']:.2f}/m³ × {volume_m3:.2f} m³"
            )

        breakdown.base_amount = base_amount
        breakdown.base_description = base_description

        subtotal = base_amount

        for code in modifier_codes:
            modifier = self._get_modifier(code)
            amount = 0.0
            if modifier["charge_type"] == "flat":
                amount = modifier["value"]
            elif modifier["charge_type"] == "per_m3":
                amount = modifier["value"] * volume_m3
            elif modifier["charge_type"] == "percentage":
                amount = base_amount * modifier["value"]
            else:  # pragma: no cover - guard for schema corruption
                raise ValueError(f"Unsupported charge type: {modifier['charge_type']}")
            breakdown.modifiers.append({
                "code": code,
                "description": modifier["description"],
                "amount": amount,
            })
            subtotal += amount

        for service_code in packing_services:
            tier = self._get_packing_rate(service_code, packing_volume)
            if tier is None:
                raise ValueError(
                    f"No packing tier defined for service '{service_code}' at volume {packing_volume} m³"
                )
            amount = tier["rate_per_m3"] * packing_volume
            breakdown.modifiers.append(
                {
                    "code": service_code,
                    "description": tier["description"] or service_code.title(),
                    "amount": amount,
                }
            )
            subtotal += amount

        uplift_row = self._get_seasonal_uplift(move_date)
        uplift_amount = 0.0
        if uplift_row is not None:
            uplift_amount = subtotal * uplift_row["uplift_pct"]
            breakdown.seasonality = {
                "code": uplift_row["code"],
                "description": uplift_row["description"],
                "percentage": uplift_row["uplift_pct"],
                "amount": uplift_amount,
            }
            subtotal += uplift_amount

        margin_amount = 0.0
        if margin_pct:
            margin_amount = subtotal * margin_pct
            breakdown.margin = {
                "percentage": margin_pct,
                "amount": margin_amount,
            }
            subtotal += margin_amount

        breakdown.total = subtotal
        return breakdown


__all__ = ["LaneCostCalculator", "QuoteBreakdown", "MetroCheck"]
