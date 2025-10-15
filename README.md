# corkysoft - bill gates' FOMO

A tool for estimating and optimising costs.
<img width="1580" height="1054" alt="image" src="https://github.com/user-attachments/assets/2e39742e-6cc2-472a-9304-11b98e2eb158" />
<img width="1711" height="104" alt="image" src="https://github.com/user-attachments/assets/21426440-2cf8-454d-9aee-a0842c7d80dc" />
<img width="596" height="67" alt="image" src="https://github.com/user-attachments/assets/03b0a748-fa3c-410b-a78c-1274f04efe97" />
<img width="2560" height="1337" alt="Screenshot_20251010_131404" src="https://github.com/user-attachments/assets/fe8af81e-d7e0-436c-8c62-9eb6ccef12c1" />
<img width="2114" height="982" alt="image" src="https://github.com/user-attachments/assets/c686b7a6-baec-4ee9-853f-e1ac11cd76ae" />
<img width="2153" height="637" alt="image" src="https://github.com/user-attachments/assets/0ccfb072-20a5-484e-971a-232b45aad957" />
<img width="955" height="1138" alt="image" src="https://github.com/user-attachments/assets/bb1c7b1d-31c4-460c-a294-5ea43bb0eca8" />
<img width="911" height="1033" alt="image" src="https://github.com/user-attachments/assets/bb7e6f1d-3776-477e-aa11-2bc84c635e71" />


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

Key visuals include:

- A histogram with break-even bands, a fitted bell curve, and kurtosis/skewness call-outs for quick shape diagnostics.
- A dataset selector for comparing imported history, saved quick quotes and live telemetry in a single workflow.
- Optional profitability tabs that compare $/m³ against $/km and contrast quoted vs cost-derived $/m³, including margin outlier tables.
- An interactive Mapbox view showing each route with selectable colouring (job, client, origin city, or destination city) and toggles to focus on lines or points when clusters get dense.
- A travel-time isochrone mode that shades the catchment around each corridor using inferred average speeds for rapid reach comparisons.
- A live network map that blends historical job filters with real-time truck telemetry, colouring corridors by profitability band and highlighting active trucks/routes, with an optional density heatmap to spotlight live clusters.
- A dynamic break-even engine that recalculates per-job cost floors using network-wide fuel, driver, maintenance and overhead settings stored in `global_parameters`.
- Corridor insights summarising job counts, weighted $/m³ and below break-even ratios aggregated into bidirectional lanes for systemic diagnostics.
- A non-technical optimizer tab that recommends corridor price uplifts from the filtered data and offers a CSV export for action lists.

```bash
streamlit run streamlit_price_distribution.py
```

By default it reads from `routes.db`. Set `CORKYSOFT_DB` or `ROUTES_DB` to point at a different SQLite database.

Use the **Import historical jobs from CSV** expander in the sidebar to load data straight into the dashboard. The uploader
accepts the same headers as the CLI importer (`date`, `origin`, `destination`, `m3`, `quoted_price`, `client`) and will
calculate per-m³ rates automatically if only revenue and volume are provided. Switch the dataset selector to **Saved quick
quotes** to analyse submissions from the in-app quote builder alongside historical jobs.

Inside the Quote builder tab you can expand the **Client details** panel to link the quote with an existing customer or create
a new one. The UI highlights potential duplicates whenever the full name, phone number or complete address matches a stored
client so you can reuse or update the right record without losing context. If you just need a quick estimate, leave the client
section blank (or enter partial contact info) and the quote will still be saved without forcing a client record to be created.
client so you can reuse or update the right record without losing context.

### Export profitability summaries

Generate a CSV-ready snapshot of the current profitability filters using the
analytics helper:

```python
from analytics.db import get_connection
from analytics.price_distribution import build_profitability_export, load_historical_jobs

with get_connection() as conn:
    df, _ = load_historical_jobs(conn)
    export_df = build_profitability_export(df, break_even=250.0)
    export_df.to_csv("profitability_summary.csv", index=False)
```

The export includes the key distribution statistics, profitability bands, and
top/bottom corridor opportunities used throughout the dashboard for easy
reporting or spreadsheet analysis.
### Corridor analytics

Use `aggregate_corridor_performance` to collapse the filtered dataset into bidirectional lanes and surface systemic KPIs:

```python
from analytics.price_distribution import aggregate_corridor_performance, load_historical_jobs
from analytics.db import connection_scope

with connection_scope() as conn:
    df, _ = load_historical_jobs(conn)

corridor_summary = aggregate_corridor_performance(df, break_even=250.0)
print(
    corridor_summary[
        [
            "corridor_pair",
            "job_count",
            "weighted_price_per_m3",
            "below_break_even_ratio",
        ]
    ].head()
)
```

#### Mock telemetry ingestion

The Streamlit map expects live data in the `truck_positions` and `active_routes` tables. A mock ingestor keeps these tables fresh:

```bash
python -m analytics.ingest_live_data --interval 5 --iterations 0
```

- `--interval` controls the seconds between updates.
- `--iterations` can limit the run for testing (omit for a continuous loop).
- `--trucks` lets you specify custom truck IDs.

The script reuses historical jobs with geocoded origins/destinations and gracefully falls back to seeded depots so the map always has routes to display.

### Simplex profit optimiser

Use the :mod:`profit_optimizer` module to evaluate the most profitable mix of jobs or lanes when capacity is limited. Decision
variables represent candidate jobs and the coefficients in each constraint model business limits such as available truck hours,
packing teams, or market demand caps.

```python
from profit_optimizer import ProfitOptimizer

optimizer = ProfitOptimizer()
optimizer.add_variable("local_move", profit_per_unit=300.0, upper_bound=40)
optimizer.add_variable("interstate_move", profit_per_unit=500.0)
optimizer.add_constraint(
    "crew_hours",
    coefficients={"local_move": 2.0, "interstate_move": 3.0},
    rhs=120.0,
)
optimizer.add_constraint(
    "truck_days",
    coefficients={"local_move": 1.0, "interstate_move": 2.0},
    rhs=80.0,
)

result = optimizer.solve()
print(result.variable_values)  # -> {'local_move': 0.0, 'interstate_move': 40.0}
print(result.binding_constraints)  # -> ['crew_hours', 'truck_days']
```

Slack values and reduced costs in :class:`profit_optimizer.OptimizationResult` can be used for quick scenario planning, e.g. to
see how much spare capacity remains or whether adding new jobs would increase or decrease total profit.

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

#### Import historical jobs

```bash
python routes_to_sqlite.py import-history historical_jobs.csv --geocode --route
```

`historical_jobs.csv` requires headers `date,origin,destination,m3,quoted_price,client`. The importer normalises whitespace and Australian postcodes, optionally geocodes/resolves addresses, and (with `--route`) enriches rows with travel distance/duration using OpenRouteService.

#### Render a network map

Generate an interactive HTML map of the saved jobs:

```bash
python map_jobs.py --out routes_map.html
```

Add `--show-actual` to overlay the actual OpenRouteService routed geometry as a separate layer (you can toggle between the straight-line and routed views via the map controls):

```bash
python map_jobs.py --show-actual
```

---

## 🗂 Database Schema

* **addresses**: normalised + geocoded address cache used by jobs and historical imports.
* **jobs**: origin/destination foreign keys into `addresses`, hourly/per-km rates, computed distance, duration, costs, resolved coordinates, timestamps.
* **geocode_cache**: cached lat/lon results keyed by `place,country`.
* **historical_jobs**: imported quotes with optional normalised addresses, postcodes, distance/duration enrichments and audit timestamps.
* **truck_positions**: latest lat/lon, status, heading and speed for each active truck.
* **active_routes**: in-flight jobs mapped to trucks with origin/destination coordinates, progress, ETA, profit-band overlays and profitability status tags.
* **lane_base_rates**: per-m³ and metro-hourly lane pricing keyed by corridor code.
* **modifier_fees**: flat / per-m³ / percentage surcharges such as difficult access or piano handling.
* **packing_rate_tiers**: tiered packing & unpacking rates by cubic metres.
* **seasonal_uplifts**: date-range percentage uplifts (e.g. Oct–Dec peak season).
* **depot_metro_zones**: depot centroids + radius used for the metro vs regional rule.
* **global_parameters**: key/value store for network break-even baselines (per-m³ bands) and underlying operating costs (fuel, driver, maintenance, overhead).

---

## ⚠️ Notes

* ORS free tier: **2,500 requests/day**.
* No live traffic data — estimates are based on OSM road networks.
* Be polite: the script has built-in backoff between requests.
* When ORS cannot snap a coordinate to the road network the quote builder now
  retries with the nearest road geometry. If that still fails you can drop
  manual pins inside the UI, edit the latitude/longitude fields directly, or
  use the **Snap pins to nearest road** action before it finally falls back to
  a straight-line distance estimate (all attempts are flagged in the UI
  suggestions and raise a modal warning).
* If a route fails (e.g. invalid address), the row remains pending.

---

## ✅ Roadmap / Ideas

* Add `--show-coords` to display lat/lon in `list`.
* Add cost breakdown columns (time vs distance).
* Optional export to CSV/Google Sheets.
* Streamlit price distribution dashboard with break-even bands.

---
