#!/usr/bin/env python3
import os, time, sqlite3, datetime
import openrouteservice as ors
from datetime import datetime, timezone
import argparse, csv, sys

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

def migrate_schema(conn):
    # Add new columns if they don't exist
    cols = [r[1] for r in conn.execute("PRAGMA table_info(jobs)")]
    def add(col, decl):
        if col not in cols:
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {decl}")
    add("origin_resolved", "TEXT")
    add("destination_resolved", "TEXT")
    add("origin_lon", "REAL")
    add("origin_lat", "REAL")
    add("dest_lon", "REAL")
    add("dest_lat", "REAL")
    conn.commit()


def geocode_cached(conn: sqlite3.Connection, place: str, country: str):
    # return (lon, lat) using cache or ORS Pelias
    row = conn.execute("SELECT lon, lat FROM geocode_cache WHERE place = ?", (f"{place}, {country}",)).fetchone()
    if row:
        return (row[0], row[1])

    # Query ORS Pelias
    q = f"{place}, {country}"
    res = client.pelias_search(text=q)
    feats = res.get("features", [])
    if not feats:
        raise ValueError(f"No geocode found for: {q}")
    coords = feats[0]["geometry"]["coordinates"]  # [lon, lat]
    lon, lat = float(coords[0]), float(coords[1])

    conn.execute(
        "INSERT OR REPLACE INTO geocode_cache(place, lon, lat, ts) VALUES (?,?,?,?)",
        (f"{place}, {country}", lon, lat, datetime.now(timezone.utc).isoformat())

    )
    conn.commit()
    time.sleep(GEOCODE_BACKOFF)
    return (lon, lat)

def route_km_hours(conn: sqlite3.Connection, origin: str, destination: str, country: str):
    o_lon, o_lat = geocode_cached(conn, origin, country)
    d_lon, d_lat = geocode_cached(conn, destination, country)

    route = client.directions(
        coordinates=[[o_lon, o_lat], [d_lon, d_lat]],
        profile="driving-car",
        format="json"
    )
    summary = route["routes"][0]["summary"]
    meters = float(summary["distance"])
    seconds = float(summary["duration"])
    time.sleep(ROUTE_BACKOFF)
    return meters / 1000.0, seconds / 3600.0

def cost_breakdown(distance_km: float, duration_hr: float, hourly_rate: float, per_km_rate: float):
    cost_time = duration_hr * hourly_rate
    cost_dist = distance_km * per_km_rate
    return cost_time, cost_dist, (cost_time + cost_dist)

def process_pending(conn: sqlite3.Connection, limit: int = 1000):
    rows = conn.execute(
        """
        SELECT id, origin, destination, hourly_rate, per_km_rate, country
        FROM jobs
        WHERE distance_km IS NULL OR duration_hr IS NULL
        LIMIT ?
        """, (limit,)
    ).fetchall()

    if not rows:
        print("No pending jobs.")
        return

    for (jid, origin, dest, hourly_rate, per_km_rate, country) in rows:
        try:
            country = country or COUNTRY_DEFAULT
            distance_km, duration_hr = route_km_hours(conn, origin, dest, country)
            cost_time, cost_dist, cost_total = cost_breakdown(distance_km, duration_hr, hourly_rate, per_km_rate)
            conn.execute(
                """
                UPDATE jobs
                SET distance_km=?, duration_hr=?, cost_time=?, cost_distance=?, cost_total=?, updated_at=?
                WHERE id=?
                """,
                (distance_km, duration_hr, cost_time, cost_dist, cost_total, datetime.now(timezone.utc).isoformat(), jid)
            )
            conn.commit()
            print(f"[OK] #{jid} {origin} -> {dest}: {distance_km:.1f} km, {duration_hr:.2f} h, total ${cost_total:.2f}")
        except Exception as e:
            print(f"[ERR] #{jid} {origin} -> {dest}: {e}")


def add_job(conn, origin, destination, hourly_rate=200.0, per_km_rate=0.80, country="Australia"):
    conn.execute("""
      INSERT INTO jobs (origin, destination, hourly_rate, per_km_rate, country)
      VALUES (?, ?, ?, ?, ?)
      ON CONFLICT(origin, destination) DO NOTHING
    """, (origin, destination, hourly_rate, per_km_rate, country))
    conn.commit()

def list_jobs(conn):
    rows = conn.execute("""
      SELECT id, origin, destination,
             ROUND(distance_km,1) AS km,
             ROUND(duration_hr,2) AS hours,
             ROUND(cost_time,2) AS cost_time,
             ROUND(cost_distance,2) AS cost_dist,
             ROUND(cost_total,2) AS total,
             updated_at
      FROM jobs
      ORDER BY updated_at DESC NULLS LAST, id
    """).fetchall()
    for r in rows:
        print(r)

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

    args = p.parse_args()
    conn = sqlite3.connect(DB_PATH)
    ensure_schema(conn)

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
    conn.close()

if __name__ == "__main__":
    cli()  # replace previous main() call
