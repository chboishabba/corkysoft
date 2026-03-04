# Corridor Detection

Corridor detection discovers recurring origin→destination lanes so pricing,
profitability, and benchmarking can be aggregated consistently.

## Definition

Corridors are directional lanes grouped into bidirectional corridor groups.
This keeps directional analytics accurate while still supporting corridor-wide
economics.

Example clusters:

- Melbourne Metro → Sydney Metro
- Brisbane → Cairns
- VIC → NSW (Hume corridor)

Corridors are used to:
- Build $/m3 benchmarks.
- Compare margins by lane.
- Feed the optimizer with corridor-level pricing signals.
- Identify backhaul imbalances.

## Inputs

The existing data model already provides the fields needed:

- `origin`, `destination`
- `origin_lat`, `origin_lon`, `dest_lat`, `dest_lon`
- `distance_km`, `duration_hr`
- `volume_m3`, `price_per_m3`, `margin_per_m3`
- `route_geojson` (when available)

## Corridor Model (Recommended)

Three levels are tracked:

- `clusters`: normalized origin/destination locations.
- `directional_corridors`: origin_cluster → destination_cluster.
- `corridor_groups`: bidirectional grouping (A ↔ B).

Directional corridors retain asymmetric metrics (traffic, tolls, access), while
corridor groups support shared benchmarking and backhaul calculations.

## Baseline Algorithm (Recommended First Pass)

1. Normalize locations to a coarse cluster:
   - Use city, postcode, or a geohash prefix (e.g. precision 5).
2. Build a directional corridor key:
   - `origin_cluster + " → " + destination_cluster`
3. Build a bidirectional group key:
   - `sorted(origin_cluster, destination_cluster)`
4. Aggregate corridor statistics:
   - job_count, median price_per_m3, median margin_per_m3, avg distance_km,
     avg duration_hr
5. Filter to “real” corridors:
   - e.g. `job_count >= 10`

This provides stable, interpretable corridors with low complexity.

## Advanced Option: Geometry-Based Corridors

For regional and interstate moves, cluster by route geometry:

- Use route midpoint clustering (DBSCAN).
- Or compare polylines with a distance metric (Fréchet or similar).

This can surface highway-level corridors (e.g. Hume, Bruce) but should be
introduced after the baseline corridor model is stable.

## Time-Based Corridor Stats

Directional corridors should track time-of-day and weekday performance to
capture asymmetric congestion and loading patterns.

Suggested time buckets:
- `night`
- `morning_peak`
- `day`
- `evening_peak`

Suggested groupings:
- `weekday`
- `weekend`

## $/m3 Distribution and Volume Buckets

Pricing intelligence should be driven by corridor-level $/m3 distributions.
To reduce small-job distortion, compute stats within volume buckets (e.g.
0-10, 10-20, 20-30, 30-40 m3).

Recommended stats per corridor (and per volume bucket):
- p10, p25, p50, p75, p90 for price_per_m3
- break_even_per_m3 overlay
- loss_rate (price_per_m3 below break_even)
- volatility (std dev of price_per_m3)

## Break-Even Overlay (v1)

For the first release, a hard-coded break-even is acceptable:

- `break_even_per_m3 = (distance_km * truck_cost_per_km + duration_hr * crew_cost_per_hr) / volume_m3`

This keeps the histogram interpretable while cost models mature.

## Phantom Corridors (Opportunity Detection)

Detect low-volume corridors with unusually high margins.

Suggested scoring:
- `margin = median_price_per_m3 - break_even_per_m3`
- `phantom_score = margin * log(job_count + 1)`

Default rule of thumb:
- `job_count < 10` and `margin > 40` → phantom corridor candidate.

## Gravity Model (Future)

Use a gravity model to predict demand between clusters:

- `demand ∝ population_a * population_b / distance_km^2`

Compare predicted demand to observed corridor jobs to surface underserved routes.

Gravity formula (transport planning default):

- `T_ij = K * (P_i^alpha * P_j^beta) / c_ij^gamma`
- Start with `alpha=beta=1`, `gamma=1.6`, `c_ij=distance_km`

## Recommended Decisions (Current)

- Cluster source-of-truth: manual regions with geohash fallback.
- Corridor visibility threshold: `job_count >= 6`.
- Denormalize corridor keys into `jobs` for fast dashboard filters.

## Opportunity Scoring (Gravity + $/m3 Distributions)

Combine predicted demand with observed capture and unit economics:

- `gap_ij = pred_ij / (obs_jobs_ij + epsilon)`
- `unit_margin_ij = median_price_per_m3 - break_even_per_m3`
- `opportunity_ij = gap_ij * max(unit_margin_ij, 0)`

Fallbacks for low-sample corridors:
- Use bidirectional corridor group medians when directional data is sparse.
- Otherwise use nearest-neighbor corridors by origin cluster + distance bucket.
- As a last resort, use global medians by distance bucket.

## Geohash Corridors (Optional Automation)

If manual clusters are unavailable, use geohash prefixes to derive corridors:

- `origin_hash = geohash(origin_lat, origin_lon, precision=4)`
- `dest_hash = geohash(dest_lat, dest_lon, precision=4)`
- `corridor_id = origin_hash + " → " + dest_hash`
- `corridor_group_id = sorted(origin_hash, dest_hash)`

Precision guidance:
- `3` ≈ interstate
- `4` ≈ metro
- `5` ≈ suburb

## Backhaul Detection

Once corridors exist, compute directional imbalance:

- `outbound_jobs` vs `inbound_jobs`
- `empty_return_rate = (outbound - inbound) / outbound`

This enables backhaul pricing suggestions.

## Suggested Tables (Future)

- `clusters`:
  - `cluster_id`, `name`, `lat`, `lon`, `cluster_type`
- `directional_corridors`:
  - `corridor_id`, `origin_cluster`, `destination_cluster`, `job_count`
  - `median_price_per_m3`, `median_margin_per_m3`, `avg_distance_km`,
    `avg_duration_hr`
- `corridor_groups`:
  - `group_id`, `cluster_a`, `cluster_b`, `total_jobs`, `imbalance_ratio`
- `corridor_time_stats`:
  - `corridor_id`, `time_bucket`, `avg_duration_hr`, `avg_speed_kph`, `job_count`

## Notes

- Start simple with city/postcode/geohash clustering.
- Directional corridors should roll up into bidirectional groups for backhaul.
- Keep overrides for manual corridor merges/splits.
- Use the same corridor key in analytics + dashboard filters.
