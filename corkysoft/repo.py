"""SQLite schema management and persistence helpers."""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

from corkysoft.au_address import STATE_NAME_TO_CODE
from corkysoft.routing import normalize_place

if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    from corkysoft.quote_service import QuoteInput, QuoteResult

COUNTRY_DEFAULT = "Australia"

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS geocode_cache (
  place TEXT PRIMARY KEY,
  lon REAL NOT NULL,
  lat REAL NOT NULL,
  postalcode TEXT,
  region_code TEXT,
  region TEXT,
  locality TEXT,
  county TEXT,
  ts  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS clients (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  first_name TEXT,
  last_name TEXT,
  company_name TEXT,
  email TEXT,
  phone TEXT,
  address_line1 TEXT,
  address_line2 TEXT,
  city TEXT,
  state TEXT,
  postcode TEXT,
  country TEXT,
  notes TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS quotes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  quote_date TEXT NOT NULL,
  origin_input TEXT NOT NULL,
  destination_input TEXT NOT NULL,
  origin_resolved TEXT,
  destination_resolved TEXT,
  origin_lon REAL,
  origin_lat REAL,
  dest_lon REAL,
  dest_lat REAL,
  distance_km REAL NOT NULL,
  duration_hr REAL NOT NULL,
  cubic_m REAL NOT NULL,
  pricing_model TEXT NOT NULL,
  base_subtotal REAL NOT NULL,
  base_components TEXT NOT NULL,
  modifiers_applied TEXT NOT NULL,
  modifiers_total REAL NOT NULL,
  seasonal_multiplier REAL NOT NULL,
  seasonal_label TEXT NOT NULL,
  total_before_margin REAL NOT NULL,
  margin_percent REAL,
  client_id INTEGER,
  client_display TEXT,
  manual_quote REAL,
  final_quote REAL NOT NULL,
  summary TEXT NOT NULL
);
"""

POSTCODE_RE = re.compile(r"\b(\d{4})\b")
STATE_RE = re.compile(r"\b(ACT|NSW|NT|QLD|SA|TAS|VIC|WA)\b", re.IGNORECASE)

CLIENT_FIELD_NAMES = (
    "first_name",
    "last_name",
    "company_name",
    "email",
    "phone",
    "address_line1",
    "address_line2",
    "city",
    "state",
    "postcode",
    "country",
    "notes",
)


@dataclass
class ClientDetails:
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    company_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address_line1: Optional[str] = None
    address_line2: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postcode: Optional[str] = None
    country: Optional[str] = None
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        for field_name in CLIENT_FIELD_NAMES:
            value = getattr(self, field_name)
            if isinstance(value, str):
                cleaned = value.strip()
                setattr(self, field_name, cleaned or None)

    def has_any_data(self) -> bool:
        return any(getattr(self, field_name) for field_name in CLIENT_FIELD_NAMES)

    def has_identity(self) -> bool:
        if self.company_name:
            return True
        if self.first_name and self.last_name:
            return True
        return False

    def display_name(self) -> Optional[str]:
        if self.company_name:
            return self.company_name
        parts = [part for part in (self.first_name, self.last_name) if part]
        if parts:
            return " ".join(parts)
        return None

    @property
    def normalized_phone(self) -> Optional[str]:
        if not self.phone:
            return None
        digits = re.sub(r"\D+", "", self.phone)
        return digits or None

    @property
    def name_key(self) -> Optional[Tuple[str, str]]:
        if self.first_name and self.last_name:
            return (self.first_name.lower(), self.last_name.lower())
        return None

    @property
    def address_key(self) -> Optional[str]:
        if not self.address_line1 or not self.city:
            return None
        if not (self.postcode or self.state):
            return None
        parts = [
            self.address_line1.lower(),
            (self.address_line2 or "").lower(),
            self.city.lower(),
        ]
        if self.state:
            parts.append(self.state.lower())
        if self.postcode:
            parts.append(self.postcode.lower())
        if self.country:
            parts.append(self.country.lower())
        return "|".join(part for part in parts if part)


@dataclass
class ClientMatch:
    id: int
    display_name: str
    reason: str


def format_client_display(
    first_name: Optional[str],
    last_name: Optional[str],
    company_name: Optional[str],
) -> str:
    details = ClientDetails(
        first_name=first_name,
        last_name=last_name,
        company_name=company_name,
    )
    display = details.display_name()
    return display or "Unnamed client"


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})")
    except sqlite3.OperationalError:
        return False
    return any(row[1] == column for row in rows)


def _ensure_column(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    if _column_exists(conn, table, column):
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Ensure the quote tables exist in *conn*."""

    conn.executescript(SCHEMA_SQL)
    _ensure_column(conn, "geocode_cache", "postalcode", "TEXT")
    _ensure_column(conn, "geocode_cache", "region_code", "TEXT")
    _ensure_column(conn, "geocode_cache", "region", "TEXT")
    _ensure_column(conn, "geocode_cache", "locality", "TEXT")
    _ensure_column(conn, "geocode_cache", "county", "TEXT")
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(quotes)")
    }
    if "manual_quote" not in columns:
        conn.execute("ALTER TABLE quotes ADD COLUMN manual_quote REAL")
    if "client_id" not in columns:
        conn.execute("ALTER TABLE quotes ADD COLUMN client_id INTEGER")
    if "client_display" not in columns:
        conn.execute("ALTER TABLE quotes ADD COLUMN client_display TEXT")
    conn.commit()


def _extract_postcode(*values: Optional[str]) -> Optional[str]:
    for value in values:
        if not value:
            continue
        match = POSTCODE_RE.search(value)
        if match:
            return match.group(1)
    return None


def _extract_state(*values: Optional[str]) -> Optional[str]:
    for value in values:
        if not value:
            continue
        text = str(value)
        match = STATE_RE.search(text)
        if match:
            return match.group(1).upper()
        lowered = f" {text.strip().lower()} "
        for name, code in STATE_NAME_TO_CODE.items():
            target = f" {name} "
            if target in lowered:
                return code
    return None


def _fetch_client_row(
    conn: sqlite3.Connection, client_id: int
) -> Optional[Sequence[Optional[str]]]:
    cursor = conn.execute(
        """
        SELECT id, first_name, last_name, company_name, email, phone,
               address_line1, address_line2, city, state, postcode, country,
               notes
        FROM clients
        WHERE id = ?
        """,
        (client_id,),
    )
    return cursor.fetchone()


def _client_details_from_row(row: Sequence[Optional[str]]) -> ClientDetails:
    return ClientDetails(
        first_name=row[1],
        last_name=row[2],
        company_name=row[3],
        email=row[4],
        phone=row[5],
        address_line1=row[6],
        address_line2=row[7],
        city=row[8],
        state=row[9],
        postcode=row[10],
        country=row[11],
        notes=row[12],
    )


def find_client_matches(
    conn: sqlite3.Connection, details: ClientDetails
) -> List[ClientMatch]:
    if not details.has_any_data():
        return []

    name_key = details.name_key
    phone_key = details.normalized_phone
    address_key = details.address_key
    if not any((name_key, phone_key, address_key)):
        return []

    rows = conn.execute(
        """
        SELECT id, first_name, last_name, company_name, email, phone,
               address_line1, address_line2, city, state, postcode, country,
               notes
        FROM clients
        """
    ).fetchall()

    matches: List[ClientMatch] = []
    for row in rows:
        row_details = _client_details_from_row(row)
        reasons: List[str] = []
        if name_key and row_details.name_key == name_key:
            reasons.append("matching name")
        if phone_key and row_details.normalized_phone == phone_key:
            reasons.append("matching phone")
        if address_key and row_details.address_key == address_key:
            reasons.append("matching address")
        if reasons:
            display = row_details.display_name() or f"Client #{row[0]}"
            matches.append(
                ClientMatch(id=int(row[0]), display_name=display, reason=", ".join(reasons))
            )
    return matches


def _ensure_address_record(
    conn: sqlite3.Connection,
    raw_input: str,
    resolved: str,
    lon: Optional[float],
    lat: Optional[float],
    country: str,
    *,
    postcode: Optional[str] = None,
    state: Optional[str] = None,
) -> Optional[int]:
    if not _table_exists(conn, "addresses"):
        return None

    _ensure_column(conn, "addresses", "state", "TEXT")
    _ensure_column(conn, "addresses", "postcode", "TEXT")

    normalized = normalize_place(resolved or raw_input)
    if not normalized:
        normalized = normalize_place(raw_input)

    has_postcode = _column_exists(conn, "addresses", "postcode")
    has_state = _column_exists(conn, "addresses", "state")

    row = conn.execute(
        """
        SELECT id, lon, lat
        FROM addresses
        WHERE normalized = ? AND (country = ? OR (country IS NULL AND ? IS NULL))
        """,
        (normalized, country, country),
    ).fetchone()
    if row is not None:
        address_id = int(row[0])
        update_fields = ["lon = COALESCE(?, lon)", "lat = COALESCE(?, lat)"]
        params: List[Optional[object]] = [lon, lat]
        if has_state:
            update_fields.append("state = COALESCE(?, state)")
            params.append(state)
        if has_postcode:
            update_fields.append("postcode = COALESCE(?, postcode)")
            params.append(postcode)
        params.append(address_id)
        conn.execute(
            f"UPDATE addresses SET {', '.join(update_fields)} WHERE id = ?",
            tuple(params),
        )
        return address_id

    columns = ["raw_input", "normalized", "city", "country", "lon", "lat"]
    values: List[Optional[object]] = [
        raw_input,
        normalized,
        resolved or normalized,
        country,
        lon,
        lat,
    ]
    if has_state:
        columns.append("state")
        values.append(state)
    if has_postcode:
        columns.append("postcode")
        values.append(postcode)

    placeholders = ", ".join(["?"] * len(columns))
    column_list = ", ".join(columns)
    cursor = conn.execute(
        f"""
        INSERT INTO addresses (
            {column_list}
        ) VALUES ({placeholders})
        """,
        tuple(values),
    )
    return int(cursor.lastrowid)


def _insert_historical_job(
    conn: sqlite3.Connection,
    inputs: "QuoteInput",
    result: "QuoteResult",
    stored_amount: float,
    created_at: str,
    client_id: Optional[int],
    client_display: Optional[str],
    origin_address_id: Optional[int],
    destination_address_id: Optional[int],
    origin_postcode: Optional[str],
    destination_postcode: Optional[str],
    origin_state: Optional[str],
    destination_state: Optional[str],
) -> None:
    if not _table_exists(conn, "historical_jobs"):
        return

    _ensure_column(conn, "historical_jobs", "origin_postcode", "TEXT")
    _ensure_column(conn, "historical_jobs", "destination_postcode", "TEXT")
    _ensure_column(conn, "historical_jobs", "origin_state", "TEXT")
    _ensure_column(conn, "historical_jobs", "destination_state", "TEXT")

    cubic_m = inputs.cubic_m
    price_per_m3: Optional[float]
    if cubic_m and cubic_m > 0:
        price_per_m3 = stored_amount / cubic_m
    else:
        price_per_m3 = None

    origin_label = result.origin_resolved or normalize_place(inputs.origin)
    destination_label = result.destination_resolved or normalize_place(
        inputs.destination
    )
    corridor_display = f"{origin_label} → {destination_label}"

    has_origin_state = _column_exists(conn, "historical_jobs", "origin_state")
    has_destination_state = _column_exists(
        conn, "historical_jobs", "destination_state"
    )
    has_client_id = _column_exists(conn, "historical_jobs", "client_id")

    columns = [
        "job_date",
        "client",
        "corridor_display",
        "price_per_m3",
        "revenue_total",
        "revenue",
        "volume_m3",
        "volume",
        "distance_km",
        "final_cost",
        "origin",
        "destination",
        "origin_postcode",
        "destination_postcode",
        "origin_address_id",
        "destination_address_id",
        "created_at",
        "updated_at",
    ]

    values: List[Optional[object]] = [
        inputs.quote_date.isoformat(),
        client_display,
        corridor_display,
        price_per_m3,
        stored_amount,
        stored_amount,
        cubic_m,
        cubic_m,
        result.distance_km,
        result.total_before_margin,
        origin_label,
        destination_label,
        origin_postcode,
        destination_postcode,
        origin_address_id,
        destination_address_id,
        created_at,
        created_at,
    ]

    if has_client_id:
        columns.insert(2, "client_id")
        values.insert(2, client_id)

    if has_origin_state:
        columns.append("origin_state")
        values.append(origin_state)
    if has_destination_state:
        columns.append("destination_state")
        values.append(destination_state)

    placeholders = ", ".join(["?"] * len(columns))
    conn.execute(
        f"""
        INSERT INTO historical_jobs (
            {', '.join(columns)}
        ) VALUES ({placeholders})
        """,
        tuple(values),
    )


def _ephemeral_client_display(details: ClientDetails) -> Optional[str]:
    display = details.display_name()
    if display:
        return display
    if details.email:
        return details.email
    if details.phone:
        return details.phone
    return None


def _update_client_missing_fields(
    conn: sqlite3.Connection,
    client_id: int,
    details: Optional[ClientDetails],
    timestamp: str,
) -> None:
    if details is None or not details.has_any_data():
        return

    updates: List[str] = []
    params: List[Optional[str]] = []
    for field_name in CLIENT_FIELD_NAMES:
        value = getattr(details, field_name)
        if value:
            updates.append(f"{field_name} = COALESCE({field_name}, ?)")
            params.append(value)
    if not updates:
        return

    updates.append("updated_at = ?")
    params.append(timestamp)
    params.append(client_id)
    conn.execute(
        f"UPDATE clients SET {', '.join(updates)} WHERE id = ?",
        tuple(params),
    )


def _ensure_client_record(
    conn: sqlite3.Connection,
    inputs: "QuoteInput",
    timestamp: str,
) -> Tuple[Optional[int], Optional[str]]:
    client_id = inputs.client_id
    details = inputs.client_details

    if client_id is not None:
        row = _fetch_client_row(conn, client_id)
        if row is None:
            inputs.client_id = None
            client_id = None
        else:
            _update_client_missing_fields(conn, client_id, details, timestamp)
            display = format_client_display(row[1], row[2], row[3])
            return client_id, display

    if details and details.has_any_data():
        normalized_details = ClientDetails(
            first_name=details.first_name,
            last_name=details.last_name,
            company_name=details.company_name,
            email=details.email,
            phone=details.phone,
            address_line1=details.address_line1,
            address_line2=details.address_line2,
            city=details.city,
            state=details.state,
            postcode=details.postcode,
            country=details.country or inputs.country,
            notes=details.notes,
        )

        if not normalized_details.has_identity():
            display = _ephemeral_client_display(normalized_details)
            if display:
                inputs.client_details = normalized_details
                return None, display
            raise ValueError(
                "Client requires a company name or both first and last names to be saved."
            )

        cursor = conn.execute(
            """
            INSERT INTO clients (
                first_name, last_name, company_name, email, phone,
                address_line1, address_line2, city, state, postcode,
                country, notes, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                normalized_details.first_name,
                normalized_details.last_name,
                normalized_details.company_name,
                normalized_details.email,
                normalized_details.phone,
                normalized_details.address_line1,
                normalized_details.address_line2,
                normalized_details.city,
                normalized_details.state,
                normalized_details.postcode,
                normalized_details.country,
                normalized_details.notes,
                timestamp,
                timestamp,
            ),
        )
        client_id = int(cursor.lastrowid)
        display = normalized_details.display_name() or f"Client #{client_id}"
        inputs.client_id = client_id
        inputs.client_details = normalized_details
        return client_id, display

    if details:
        display = _ephemeral_client_display(details)
        if display:
            return None, display
    return None, None


def persist_quote(
    conn: sqlite3.Connection,
    inputs: "QuoteInput",
    result: "QuoteResult",
    manual_quote: Optional[float] = None,
) -> int:
    manual_value = (
        manual_quote
        if manual_quote is not None
        else result.manual_quote
    )
    created_at = datetime.now(timezone.utc).isoformat()
    client_id, client_display = _ensure_client_record(conn, inputs, created_at)
    stored_client_display = client_display or "Quote builder"
    cursor = conn.execute(
        """
        INSERT INTO quotes (
            created_at, quote_date,
            origin_input, destination_input,
            origin_resolved, destination_resolved,
            origin_lon, origin_lat, dest_lon, dest_lat,
            distance_km, duration_hr, cubic_m,
            pricing_model, base_subtotal, base_components,
            modifiers_applied, modifiers_total,
            seasonal_multiplier, seasonal_label,
            total_before_margin, margin_percent,
            client_id, client_display,
            manual_quote, final_quote, summary
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            created_at,
            inputs.quote_date.isoformat(),
            inputs.origin,
            inputs.destination,
            result.origin_resolved,
            result.destination_resolved,
            result.origin_lon,
            result.origin_lat,
            result.dest_lon,
            result.dest_lat,
            result.distance_km,
            result.duration_hr,
            inputs.cubic_m,
            result.pricing_model.id,
            result.base_subtotal,
            json.dumps(result.base_components),
            json.dumps(result.modifier_details),
            result.modifiers_total,
            result.seasonal_multiplier,
            result.seasonal_label,
            result.total_before_margin,
            result.margin_percent,
            client_id,
            stored_client_display,
            manual_value,
            result.final_quote,
            result.summary_text,
        ),
    )
    quote_rowid = int(cursor.lastrowid)

    stored_amount = (
        float(manual_value)
        if manual_value is not None
        else result.final_quote
    )

    origin_postcode = _extract_postcode(
        result.origin_postcode_hint,
        result.origin_resolved,
        inputs.origin,
        *(result.origin_candidates or []),
        *(result.origin_suggestions or []),
    )
    destination_postcode = _extract_postcode(
        result.destination_postcode_hint,
        result.destination_resolved,
        inputs.destination,
        *(result.destination_candidates or []),
        *(result.destination_suggestions or []),
    )

    origin_state = _extract_state(
        result.origin_state_hint,
        result.origin_resolved,
        inputs.origin,
        *(result.origin_candidates or []),
        *(result.origin_suggestions or []),
    )
    destination_state = _extract_state(
        result.destination_state_hint,
        result.destination_resolved,
        inputs.destination,
        *(result.destination_candidates or []),
        *(result.destination_suggestions or []),
    )

    origin_address_id = _ensure_address_record(
        conn,
        inputs.origin,
        result.origin_resolved,
        result.origin_lon,
        result.origin_lat,
        inputs.country,
        postcode=origin_postcode,
        state=origin_state,
    )
    destination_address_id = _ensure_address_record(
        conn,
        inputs.destination,
        result.destination_resolved,
        result.dest_lon,
        result.dest_lat,
        inputs.country,
        postcode=destination_postcode,
        state=destination_state,
    )
    _insert_historical_job(
        conn,
        inputs,
        result,
        stored_amount,
        created_at,
        client_id,
        stored_client_display,
        origin_address_id,
        destination_address_id,
        origin_postcode,
        destination_postcode,
        origin_state,
        destination_state,
    )
    conn.commit()
    return quote_rowid


__all__ = [
    "CLIENT_FIELD_NAMES",
    "ClientDetails",
    "ClientMatch",
    "SCHEMA_SQL",
    "ensure_schema",
    "find_client_matches",
    "format_client_display",
    "persist_quote",
]
