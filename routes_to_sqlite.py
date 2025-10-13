#!/usr/bin/env python3
import os, time, sqlite3, datetime, json, re
import openrouteservice as ors
from datetime import datetime, timezone
import argparse, csv, sys
from typing import Optional

DB_PATH = os.environ.get("ROUTES_DB", "routes.db")
ORS_KEY = os.environ.get("ORS_API_KEY")  # export ORS_API_KEY=xxxx
COUNTRY_DEFAULT = os.environ.get("ORS_COUNTRY", "Australia")
GEOCODE_BACKOFF = 0.2   # seconds between calls (be polite)
ROUTE_BACKOFF = 0.2

if not ORS_KEY:
    raise SystemExit("Set ORS_API_KEY env var (export ORS_API_KEY=YOUR_KEY)")

client = ors.Client(key=ORS_KEY)

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS geocode_cache (
  place TEXT PRIMARY KEY,
  lon REAL NOT NULL,
  lat REAL NOT NULL,
  ts  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  origin TEXT NOT NULL,
  destination TEXT NOT NULL,
  hourly_rate REAL NOT NULL DEFAULT 200.0,
  per_km_rate REAL NOT NULL DEFAULT 0.0,
  country TEXT NOT NULL DEFAULT 'Australia',
  -- outputs
  distance_km REAL,
  duration_hr REAL,
  cost_time REAL,
  cost_distance REAL,
  cost_total REAL,
  updated_at TEXT,
  UNIQUE(origin, destination)
);

CREATE TABLE IF NOT EXISTS historical_jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_date TEXT NOT NULL,
  origin TEXT NOT NULL,
  destination TEXT NOT NULL,
  m3 REAL,
  quoted_price REAL,
  price_per_m3 REAL,
  client TEXT NOT NULL DEFAULT '',
  origin_normalized TEXT,
  destination_normalized TEXT,
  origin_postcode TEXT,
  destination_postcode TEXT,
  origin_lon REAL,
  origin_lat REAL,
  dest_lon REAL,
  dest_lat REAL,
  distance_km REAL,
  duration_hr REAL,
  imported_at TEXT NOT NULL,
  updated_at TEXT,
  UNIQUE(job_date, origin, destination, client, quoted_price)
);
"""

def ensure_schema(conn: sqlite3.Connection):
    conn.executescript(SCHEMA_SQL)
    conn.commit()

def migrate_schema(conn: sqlite3.Connection):
    # Add new columns if they don't exist
    cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}
    def add(col, decl):
        if col not in cols:
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {decl}")
    add("origin_resolved", "TEXT")
    add("destination_resolved", "TEXT")
    add("origin_lon", "REAL")
    add("origin_lat", "REAL")
    add("dest_lon", "REAL")
    add("dest_lat", "REAL")
    add("route_geojson", "TEXT")  # full driving lines
    conn.commit()

# ---------- Address normalization (AU-focused) ----------
ABBREV = {
    " cr ": " Circuit ",
    " crt ": " Court ",
    " ct ": " Court ",
    " ave ": " Avenue ",
    " av ": " Avenue ",
    " rd ": " Road ",
    " st ": " Street ",
    " hwy ": " Highway ",
    " pde ": " Parade ",
    " dr ": " Drive ",
    " wy ": " Way ",
}

def normalize_au_address(s: str) -> str:
    if not s:
        return s
    t = f" {s.strip()} ".lower()
    for k, v in ABBREV.items():
        t = t.replace(k, v.lower())
    t = " ".join(t.split())  # collapse spaces
    # Capitalize words
    t = " ".join(w.capitalize() for w in t.split())
    # If there’s a postcode but no explicit state, default to QLD (common case you showed)
    if " qld " not in (" " + t.lower() + " ") and any(ch.isdigit() for ch in t[-5:]):
        t = t + ", QLD"
    return t

# ---------- Shared utilities ----------
POSTCODE_RE = re.compile(r"\b(\d{4})\b")


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.strip().split())


def normalize_location(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return _normalize_whitespace(value)


def normalize_postcode(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    match = POSTCODE_RE.search(value)
    if not match:
        return None
    return match.group(1)


def parse_date(value: str) -> datetime.date:
    value = value.strip()
    # Try a set of common formats (ISO, AU, US) before falling back to date.fromisoformat
    candidates = (
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%m/%d/%Y",
    )
    for fmt in candidates:
        try:
            return datetime.datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    try:
        return datetime.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Unrecognised date format: {value!r}") from exc


# ---------- Geocoding helpers ----------
STRICT_PELIAS_LAYERS = ["address", "street", "locality"]
STRICT_PELIAS_SOURCES = ["osm", "wof"]


def pelias_geocode(place: str, country: str):
    """
    Try a stricter AU-focused search first (address/street/locality layers),
    then fall back to a looser text search. Returns (lon, lat, label).
    """
    norm = normalize_au_address(place)
    # Strict attempt
    res = client.pelias_search(
        text=f"{norm}, {country}",
        layers=STRICT_PELIAS_LAYERS,
        sources=STRICT_PELIAS_SOURCES,
        size=1
    )
    feats = res.get("features") or []
    if not feats:
        # Fallback: looser search
        res = client.pelias_search(text=f"{place}, {country}", size=1)
        feats = res.get("features") or []
        if not feats:
            raise ValueError(f"No geocode found for: {place}, {country}")

    f = feats[0]
    lon, lat = f["geometry"]["coordinates"]
    label = f["properties"].get("label") or f["properties"].get("name") or norm
    return float(lon), float(lat), label

def geocode_cached(conn: sqlite3.Connection, place: str, country: str):
    """
    Return (lon, lat, label?). Uses cache for coords; label is only returned when we geocode fresh.
    """
    row = conn.execute(
        "SELECT lon, lat FROM geocode_cache WHERE place = ?",
        (f"{place}, {country}",)
    ).fetchone()
    if row:
        return (row[0], row[1], None)

    lon, lat, label = pelias_geocode(place, country)
    conn.execute(
        "INSERT OR REPLACE INTO geocode_cache(place, lon, lat, ts) VALUES (?,?,?,?)",
        (f"{place}, {country}", lon, lat, datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    time.sleep(GEOCODE_BACKOFF)
    return (lon, lat, label)

# ---------- Routing / Costs (with full geometry) ----------
def route_with_geometry(conn: sqlite3.Connection, origin: str, destination: str, country: str):
    o_lon, o_lat, o_label = geocode_cached(conn, origin, country)
    d_lon, d_lat, d_label = geocode_cached(conn, destination, country)

    # Ask ORS for full GeoJSON — includes properties.summary distance/duration + LineString geometry
    route_fc = client.directions(
        coordinates=[[o_lon, o_lat], [d_lon, d_lat]],
        profile="driving-car",
        format="geojson"
    )
    # route_fc is a FeatureCollection with one Feature
    feat = route_fc["features"][0]
    props = feat["properties"]
    summary = props["summary"]
    meters = float(summary["distance"])
    seconds = float(summary["duration"])

    time.sleep(ROUTE_BACKOFF)

    # Return the raw GeoJSON string for storage
    route_geojson = json.dumps(route_fc, separators=(",", ":"))
    return (meters / 1000.0, seconds / 3600.0,
            o_label, d_label, o_lon, o_lat, d_lon, d_lat,
            route_geojson)

def cost_breakdown(distance_km: float, duration_hr: float, hourly_rate: float, per_km_rate: float):
    cost_time = duration_hr * hourly_rate
    cost_dist = distance_km * per_km_rate
    return cost_time, cost_dist, (cost_time + cost_dist)

# ---------- Historical job import ----------
def import_historical_jobs(
    conn: sqlite3.Connection,
    csv_path: str,
    *,
    geocode: bool = False,
    route: bool = False,
    country: str = COUNTRY_DEFAULT,
):
    if route:
        geocode = True

    inserted = 0
    updated = 0
    now = datetime.datetime.now(timezone.utc).isoformat()

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        expected = {"date", "origin", "destination", "m3", "quoted_price", "client"}
        missing = expected - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required headers: {', '.join(sorted(missing))}")

        for idx, row in enumerate(reader, start=2):  # header is line 1
            try:
                date_raw = row.get("date", "").strip()
                origin_raw = row.get("origin", "").strip()
                dest_raw = row.get("destination", "").strip()
                m3_raw = row.get("m3", "").strip()
                price_raw = row.get("quoted_price", "").strip()
                client_raw = row.get("client", "").strip()

                if not (date_raw and origin_raw and dest_raw and price_raw):
                    raise ValueError("Row is missing required values")

                job_date = parse_date(date_raw).isoformat()
                origin = origin_raw
                destination = dest_raw
                client = client_raw or ""

                try:
                    m3 = float(m3_raw) if m3_raw else None
                except ValueError as exc:
                    raise ValueError(f"Invalid m3 value: {m3_raw!r}") from exc

                try:
                    quoted_price = float(price_raw)
                except ValueError as exc:
                    raise ValueError(f"Invalid quoted_price value: {price_raw!r}") from exc

                price_per_m3 = None
                if m3 is not None and m3 > 0:
                    price_per_m3 = quoted_price / m3

                origin_norm = normalize_location(origin)
                dest_norm = normalize_location(destination)
                origin_postcode = normalize_postcode(origin_norm or "")
                dest_postcode = normalize_postcode(dest_norm or "")

                origin_lon = origin_lat = dest_lon = dest_lat = None
                distance_km = duration_hr = None

                if geocode and origin_norm:
                    try:
                        origin_lon, origin_lat, origin_label = geocode_cached(conn, origin_norm, country)
                        origin_norm = origin_label or origin_norm
                    except Exception as exc:
                        print(f"[WARN] Line {idx}: failed to geocode origin {origin!r}: {exc}", file=sys.stderr)

                if geocode and dest_norm:
                    try:
                        dest_lon, dest_lat, dest_label = geocode_cached(conn, dest_norm, country)
                        dest_norm = dest_label or dest_norm
                    except Exception as exc:
                        print(f"[WARN] Line {idx}: failed to geocode destination {destination!r}: {exc}", file=sys.stderr)

                if route and None not in (origin_lon, origin_lat, dest_lon, dest_lat):
                    try:
                        route_res = client.directions(
                            coordinates=[[origin_lon, origin_lat], [dest_lon, dest_lat]],
                            profile="driving-car",
                            format="json",
                        )
                        summary = route_res["routes"][0]["summary"]
                        distance_km = float(summary["distance"]) / 1000.0
                        duration_hr = float(summary["duration"]) / 3600.0
                        time.sleep(ROUTE_BACKOFF)
                    except Exception as exc:
                        print(f"[WARN] Line {idx}: failed to route {origin!r} -> {destination!r}: {exc}", file=sys.stderr)

                params = (
                    job_date,
                    origin,
                    destination,
                    m3,
                    quoted_price,
                    price_per_m3,
                    client,
                    origin_norm,
                    dest_norm,
                    origin_postcode,
                    dest_postcode,
                    origin_lon,
                    origin_lat,
                    dest_lon,
                    dest_lat,
                    distance_km,
                    duration_hr,
                    now,
                    now,
                )

                existing = conn.execute(
                    """
                    SELECT 1 FROM historical_jobs
                    WHERE job_date = ? AND origin = ? AND destination = ? AND quoted_price = ? AND client = ?
                    """,
                    (job_date, origin, destination, quoted_price, client),
                ).fetchone()

                conn.execute(
                    """
                    INSERT INTO historical_jobs (
                        job_date, origin, destination, m3, quoted_price, price_per_m3, client,
                        origin_normalized, destination_normalized, origin_postcode, destination_postcode,
                        origin_lon, origin_lat, dest_lon, dest_lat, distance_km, duration_hr,
                        imported_at, updated_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(job_date, origin, destination, client, quoted_price)
                    DO UPDATE SET
                        m3 = excluded.m3,
                        price_per_m3 = excluded.price_per_m3,
                        origin_normalized = COALESCE(excluded.origin_normalized, historical_jobs.origin_normalized),
                        destination_normalized = COALESCE(excluded.destination_normalized, historical_jobs.destination_normalized),
                        origin_postcode = COALESCE(excluded.origin_postcode, historical_jobs.origin_postcode),
                        destination_postcode = COALESCE(excluded.destination_postcode, historical_jobs.destination_postcode),
                        origin_lon = COALESCE(excluded.origin_lon, historical_jobs.origin_lon),
                        origin_lat = COALESCE(excluded.origin_lat, historical_jobs.origin_lat),
                        dest_lon = COALESCE(excluded.dest_lon, historical_jobs.dest_lon),
                        dest_lat = COALESCE(excluded.dest_lat, historical_jobs.dest_lat),
                        distance_km = COALESCE(excluded.distance_km, historical_jobs.distance_km),
                        duration_hr = COALESCE(excluded.duration_hr, historical_jobs.duration_hr),
                        updated_at = excluded.updated_at
                    """,
                    params,
                )
                if existing:
                    updated += 1
                else:
                    inserted += 1
            except Exception as exc:
                print(f"[ERR] Line {idx}: {exc}", file=sys.stderr)

    conn.commit()
    print(f"Imported historical jobs: {inserted} inserted, {updated} updated.")

# ---------- Processing ----------
def process_pending(conn: sqlite3.Connection, limit: int = 1000):
    rows = conn.execute(
        """
        SELECT id, origin, destination, hourly_rate, per_km_rate, country
        FROM jobs
        WHERE distance_km IS NULL OR duration_hr IS NULL OR route_geojson IS NULL
        LIMIT ?
        """, (limit,)
    ).fetchall()

    if not rows:
        print("No pending jobs.")
        return

    for (jid, origin, dest, hourly_rate, per_km_rate, country) in rows:
        try:
            country = country or COUNTRY_DEFAULT
            (distance_km, duration_hr,
             o_label, d_label, o_lon, o_lat, d_lon, d_lat,
             route_geojson) = route_with_geometry(conn, origin, dest, country)

            cost_time, cost_dist, cost_total = cost_breakdown(distance_km, duration_hr, hourly_rate, per_km_rate)
            conn.execute(
                """
                UPDATE jobs
                SET distance_km=?, duration_hr=?, cost_time=?, cost_distance=?, cost_total=?,
                    origin_resolved=COALESCE(?, origin_resolved),
                    destination_resolved=COALESCE(?, destination_resolved),
                    origin_lon=?, origin_lat=?, dest_lon=?, dest_lat=?,
                    route_geojson=?,
                    updated_at=?
                WHERE id=?
                """,
                (distance_km, duration_hr, cost_time, cost_dist, cost_total,
                 o_label, d_label, o_lon, o_lat, d_lon, d_lat,
                 route_geojson,
                 datetime.now(timezone.utc).isoformat(), jid)
            )
            conn.commit()
            print(f"[OK] #{jid} {origin} → {dest} | {distance_km:.1f} km | {duration_hr:.2f} h | ${cost_total:,.2f}")
        except Exception as e:
            print(f"[ERR] #{jid} {origin} → {dest}: {e}")

# ---------- CLI helpers ----------
def add_job(conn, origin, destination, hourly_rate=200.0, per_km_rate=0.80, country="Australia"):
    conn.execute("""
      INSERT INTO jobs (origin, destination, hourly_rate, per_km_rate, country)
      VALUES (?, ?, ?, ?, ?)
      ON CONFLICT(origin, destination) DO NOTHING
    """, (origin, destination, hourly_rate, per_km_rate, country))
    conn.commit()

def list_jobs(conn):
    rows = conn.execute("""
      SELECT id,
             origin, COALESCE(origin_resolved, origin) AS origin_disp,
             destination, COALESCE(destination_resolved, destination) AS dest_disp,
             ROUND(distance_km,1) AS km,
             ROUND(duration_hr,2) AS hours,
             ROUND(cost_total,2) AS total,
             updated_at
      FROM jobs
      ORDER BY updated_at DESC NULLS LAST, id
    """).fetchall()

    if not rows:
        print("No jobs found.")
        return

    headers = ["ID", "Origin", "→ Origin (resolved)", "Destination", "→ Destination (resolved)",
               "Km", "Hours", "Total $", "Updated (UTC)"]
    widths  = [4, 18, 34, 20, 34, 7, 7, 12, 25]

    def fmt_row(r):
        id_, o, or_, d, dr_, km, hr, tot, upd = r
        items = [
            f"{id_}".ljust(widths[0]),
            (o or "")[:widths[1]-1].ljust(widths[1]),
            (or_ or "")[:widths[2]-1].ljust(widths[2]),
            (d or "")[:widths[3]-1].ljust(widths[3]),
            (dr_ or "")[:widths[4]-1].ljust(widths[4]),
            (f"{km:.1f}" if km is not None else "").rjust(widths[5]),
            (f"{hr:.2f}" if hr is not None else "").rjust(widths[6]),
            (f"{tot:,.2f}" if tot is not None else "").rjust(widths[7]),
            (upd or "").ljust(widths[8]),
        ]
        return "  ".join(items)

    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-" * (sum(widths) + 2*(len(widths)-1)))
    for r in rows:
        print(fmt_row(r))

def add_from_csv(conn, csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            add_job(
                conn,
                row["origin"],
                row["destination"],
                float(row.get("hourly_rate", 200) or 200),
                float(row.get("per_km_rate", 0.80) or 0.80),
                row.get("country") or "Australia"
            )

# ---------- Map generation (Folium) ----------

def map_jobs(conn, out_html="routes_map.html"):
    try:
        import folium
    except ImportError:
        raise SystemExit("Folium not installed. Run: pip install folium")

    rows = conn.execute("""
        SELECT id, origin, destination,
               COALESCE(origin_resolved, origin) AS o_res,
               COALESCE(destination_resolved, destination) AS d_res,
               origin_lon, origin_lat, dest_lon, dest_lat,
               route_geojson
        FROM jobs
        WHERE route_geojson IS NOT NULL
    """).fetchall()

    if not rows:
        print("No routes with geometry to map. Run `run` first.")
        return

    # Compute a rough center
    lats, lons = [], []
    for _, _, _, _, _, olon, olat, dlon, dlat, _ in rows:
        if olat is not None and olon is not None: lats.append(olat); lons.append(olon)
        if dlat is not None and dlon is not None: lats.append(dlat); lons.append(dlon)
    center = [-25.0, 135.0]
    if lats and lons:
        center = [sum(lats)/len(lats), sum(lons)/len(lons)]

    m = folium.Map(location=center, zoom_start=5)

    # Add routes
    for jid, o, d, o_res, d_res, olon, olat, dlon, dlat, gj in rows:
        try:
            fc = json.loads(gj)
            folium.GeoJson(
                fc,
                name=f"#{jid} {o} → {d}",
                tooltip=f"#{jid} {o_res} → {d_res}"
            ).add_to(m)
            if olat is not None and olon is not None:
                folium.Marker([olat, olon], popup=f"#{jid} Origin: {o_res}",
                              icon=folium.Icon(color="blue")).add_to(m)
            if dlat is not None and dlon is not None:
                folium.Marker([dlat, dlon], popup=f"#{jid} Destination: {d_res}",
                              icon=folium.Icon(color="red")).add_to(m)
        except Exception as e:
            print(f"[WARN] Could not plot job #{jid}: {e}")

    folium.LayerControl().add_to(m)

    # Mini legend (just HTML overlay)
    legend_html = """
    <div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999; background: white; padding: 8px 10px; border: 1px solid #bbb; border-radius: 6px; font-size: 12px;">
      <b>Legend</b><br>
      <span style="color:blue;">●</span> Origin<br>
      <span style="color:red;">●</span> Destination<br>
      <span style="color:green;">▬</span> Driving route
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(out_html)
    print(f"Map saved to {out_html}")


# ---------- CLI ----------
def cli():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    a1 = sub.add_parser("add")
    a1.add_argument("origin")
    a1.add_argument("destination")
    a1.add_argument("--hourly", type=float, default=200.0)
    a1.add_argument("--perkm", type=float, default=0.80)
    a1.add_argument("--country", default="Australia")

    a2 = sub.add_parser("add-csv")
    a2.add_argument("csv")

    a_history = sub.add_parser("import-history")
    a_history.add_argument("csv")
    a_history.add_argument("--geocode", action="store_true", help="Geocode origins/destinations while importing")
    a_history.add_argument("--route", action="store_true", help="Calculate route metrics (implies --geocode)")

    sub.add_parser("list")
    sub.add_parser("run")

    a_map = sub.add_parser("map")
    a_map.add_argument("--out", default="routes_map.html", help="Output HTML map path")

    args = p.parse_args()
    conn = sqlite3.connect(DB_PATH)
    ensure_schema(conn)
    migrate_schema(conn)

    if args.cmd == "add":
        add_job(conn, args.origin, args.destination, args.hourly, args.perkm, args.country)
        print("Added.")
    elif args.cmd == "add-csv":
        add_from_csv(conn, args.csv)
        print("Imported.")
    elif args.cmd == "import-history":
        import_historical_jobs(conn, args.csv, geocode=args.geocode, route=args.route)
    elif args.cmd == "list":
        list_jobs(conn)
    elif args.cmd == "run":
        process_pending(conn)
    elif args.cmd == "map":
        map_jobs(conn, out_html=args.out)
    conn.close()

if __name__ == "__main__":
    cli()
