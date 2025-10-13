#!/usr/bin/env python3
import os, time, sqlite3, datetime, json
import openrouteservice as ors
from datetime import datetime, timezone
import argparse, csv, sys

from corkysoft.au_address import GeocodeResult, geocode_with_normalization

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

# ---------- Geocoding helpers ----------
def pelias_geocode(place: str, country: str):
    """
    Try a stricter AU-focused search first (address/street/locality layers),
    then fall back to a looser text search. Returns a :class:`GeocodeResult`.
    """
    return geocode_with_normalization(client, place, country)

def geocode_cached(conn: sqlite3.Connection, place: str, country: str) -> GeocodeResult:
    """Return geocode results with normalization metadata, using the on-disk cache."""
    row = conn.execute(
        "SELECT lon, lat FROM geocode_cache WHERE place = ?",
        (f"{place}, {country}",)
    ).fetchone()
    if row:
        return GeocodeResult(
            lon=float(row[0]),
            lat=float(row[1]),
            label=None,
            normalization=None,
            search_candidates=[place],
        )

    result = pelias_geocode(place, country)
    conn.execute(
        "INSERT OR REPLACE INTO geocode_cache(place, lon, lat, ts) VALUES (?,?,?,?)",
        (
            f"{place}, {country}",
            result.lon,
            result.lat,
            datetime.now(timezone.utc).isoformat(),
        )
    )
    conn.commit()
    time.sleep(GEOCODE_BACKOFF)
    return result

# ---------- Routing / Costs (with full geometry) ----------
def route_with_geometry(
    conn: sqlite3.Connection, origin: str, destination: str, country: str
):
    origin_geo = geocode_cached(conn, origin, country)
    dest_geo = geocode_cached(conn, destination, country)

    def resolved_label(geo: GeocodeResult, fallback: str) -> str:
        if geo.label:
            return geo.label
        if geo.normalization and geo.normalization.canonical:
            return geo.normalization.canonical
        return fallback

    o_label = resolved_label(origin_geo, origin)
    d_label = resolved_label(dest_geo, destination)

    # Ask ORS for full GeoJSON — includes properties.summary distance/duration + LineString geometry
    route_fc = client.directions(
        coordinates=[[origin_geo.lon, origin_geo.lat], [dest_geo.lon, dest_geo.lat]],
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
    return (
        meters / 1000.0,
        seconds / 3600.0,
        origin_geo,
        dest_geo,
        o_label,
        d_label,
        route_geojson,
    )

def cost_breakdown(distance_km: float, duration_hr: float, hourly_rate: float, per_km_rate: float):
    cost_time = duration_hr * hourly_rate
    cost_dist = distance_km * per_km_rate
    return cost_time, cost_dist, (cost_time + cost_dist)

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
            (
                distance_km,
                duration_hr,
                origin_geo,
                dest_geo,
                o_label,
                d_label,
                route_geojson,
            ) = route_with_geometry(conn, origin, dest, country)

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
                (
                    distance_km,
                    duration_hr,
                    cost_time,
                    cost_dist,
                    cost_total,
                    o_label,
                    d_label,
                    origin_geo.lon,
                    origin_geo.lat,
                    dest_geo.lon,
                    dest_geo.lat,
                    route_geojson,
                 datetime.now(timezone.utc).isoformat(), jid)
            )
            conn.commit()
            print(f"[OK] #{jid} {origin} → {dest} | {distance_km:.1f} km | {duration_hr:.2f} h | ${cost_total:,.2f}")
            if origin_geo.normalization:
                if origin_geo.normalization.alternatives:
                    print(
                        f"    Origin candidates: {', '.join(origin_geo.normalization.candidates)}"
                    )
                if origin_geo.normalization.autocorrections:
                    print(
                        f"    Origin suggestions: {', '.join(origin_geo.normalization.autocorrections)}"
                    )
            if dest_geo.normalization:
                if dest_geo.normalization.alternatives:
                    print(
                        f"    Destination candidates: {', '.join(dest_geo.normalization.candidates)}"
                    )
                if dest_geo.normalization.autocorrections:
                    print(
                        f"    Destination suggestions: {', '.join(dest_geo.normalization.autocorrections)}"
                    )
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

    sub.add_parser("list")
    sub.add_parser("run")

    a3 = sub.add_parser("map")
    a3.add_argument("--out", default="routes_map.html", help="Output HTML map path")

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
    elif args.cmd == "list":
        list_jobs(conn)
    elif args.cmd == "run":
        process_pending(conn)
    elif args.cmd == "map":
        map_jobs(conn, out_html=args.out)
    conn.close()

if __name__ == "__main__":
    cli()
