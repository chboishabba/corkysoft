# corkysoft - bill gates' FOMO

A tool for estimating and optimising costs.
<img width="1580" height="1054" alt="image" src="https://github.com/user-attachments/assets/2e39742e-6cc2-472a-9304-11b98e2eb158" />
<img width="1711" height="104" alt="image" src="https://github.com/user-attachments/assets/21426440-2cf8-454d-9aee-a0842c7d80dc" />
<img width="596" height="67" alt="image" src="https://github.com/user-attachments/assets/03b0a748-fa3c-410b-a78c-1274f04efe97" />


# Route Distance & Cost Calculator (OpenRouteService + SQLite)

A Python utility that calculates driving distances, durations, and cost estimates between origins and destinations.  
It uses the [OpenRouteService API](https://openrouteservice.org) (ORS, open-source, OSM-backed) for geocoding and routing, and stores results in a local SQLite database for reuse.

---

## ✨ Features

- Calculate **driving distance** (km) and **duration** (hours) between city names or full street addresses.
- Estimate **billable costs** using an **hourly rate** and/or **per-km rate**.
- Store results in **SQLite** (`routes.db` by default).
- **Geocode caching** (avoids repeated lookups for the same city/address).
- **Address normalization** for Australian street abbreviations (e.g. `cr` → `Circuit`).
- Save **resolved addresses** and coordinates for clarity.
- Pretty, aligned `list` output.

---

## 📦 Installation

1. Clone this repo or copy `routes_to_sqlite.py`.
2. Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install openrouteservice
````

3. Get a free API key from [openrouteservice.org](https://openrouteservice.org/sign-up/).

---

## 🔑 Setup

Export your ORS API key:

```bash
export ORS_API_KEY="your_key_here"
```

Optional environment variables:

* `ROUTES_DB` → SQLite database path (default: `routes.db`)
* `ORS_COUNTRY` → Default country context (default: `Australia`)

---

## 🚀 Usage

Run the script directly:

```bash
python routes_to_sqlite.py <command> [options]
```

### Streamlit price distribution dashboard

The Streamlit app visualises historical jobs by $ per m³ with configurable break-even bands and CSV export.

```bash
streamlit run streamlit_price_distribution.py
```

By default it reads from `routes.db`. Set `CORKYSOFT_DB` or `ROUTES_DB` to point at a different SQLite database.

### Commands

#### Add a job

```bash
python routes_to_sqlite.py add "Melbourne" "Sydney" --hourly 200 --perkm 0.8
```

#### Add jobs from a CSV

```bash
python routes_to_sqlite.py add-csv jobs.csv
```

`jobs.csv` must include headers:

```csv
origin,destination,hourly_rate,per_km_rate,country
Melbourne,Sydney,200,0.8,Australia
Adelaide,Melbourne,180,0.7,Australia
```

#### Process pending jobs

Fetch distances/durations from ORS and update the DB:

```bash
python routes_to_sqlite.py run
```

Example output:

```
[OK] #1 Melbourne → Sydney | 869.4 km | 9.33 h | $2,561.35
[OK] #2 Brisbane → Toowoomba | 125.6 km | 1.81 h | $463.04
```

#### List jobs

```bash
python routes_to_sqlite.py list
```

Example output:

```
ID    Origin             → Origin (resolved)             Destination         → Destination (resolved)          Km      Hours   Total $      Updated (UTC)
---------------------------------------------------------------------------------------------------------------------------------------------------------
5     Melbourne          Melbourne VIC, Australia        Brisbane            Brisbane QLD, Australia          1768.4   18.80   5,175.07     2025-10-02T14:46:00+00:00
1     Melbourne          Melbourne VIC, Australia        Sydney              Sydney NSW, Australia             869.4    9.33   2,561.35     2025-10-02T14:41:10+00:00
```

---

## 🗂 Database Schema

* **addresses**: normalised + geocoded address cache used by jobs and historical imports.
* **jobs**: origin/destination foreign keys into `addresses`, hourly/per-km rates, computed distance, duration, costs, resolved coordinates, timestamps.
* **geocode_cache**: cached lat/lon results keyed by `place,country`.
* **lane_base_rates**: per-m³ and metro-hourly lane pricing keyed by corridor code.
* **modifier_fees**: flat / per-m³ / percentage surcharges such as difficult access or piano handling.
* **packing_rate_tiers**: tiered packing & unpacking rates by cubic metres.
* **seasonal_uplifts**: date-range percentage uplifts (e.g. Oct–Dec peak season).
* **depot_metro_zones**: depot centroids + radius used for the metro vs regional rule.

---

## ⚠️ Notes

* ORS free tier: **2,500 requests/day**.
* No live traffic data — estimates are based on OSM road networks.
* Be polite: the script has built-in backoff between requests.
* If a route fails (e.g. invalid address), the row remains pending.

---

## ✅ Roadmap / Ideas

* Add `--show-coords` to display lat/lon in `list`.
* Add cost breakdown columns (time vs distance).
* Optional export to CSV/Google Sheets.
* Streamlit price distribution dashboard with break-even bands.

---
