# Corridor Schema Plan

This document proposes the database tables and aggregation flow to support
directional corridors grouped into bidirectional corridor groups.

## Goals

- Capture directional corridor statistics for asymmetric travel conditions.
- Group corridors bidirectionally for shared economics and backhaul analysis.
- Support time-bucketed corridor performance (peak vs off-peak).

## Tables

### `clusters`

Stores normalized origin/destination clusters used for corridor keys.

Columns:
- `cluster_id` (text, pk)
- `name` (text)
- `lat` (real)
- `lon` (real)
- `cluster_type` (text) — e.g. `metro`, `region`, `depot`, `geohash`
- `source` (text) — e.g. `geohash_5`, `postcode`, `manual`
- `created_at` (text)
- `updated_at` (text)

Indexes:
- `clusters(name)`

### `directional_corridors`

Directional corridors from origin_cluster → destination_cluster.

Columns:
- `corridor_id` (text, pk) — `origin_cluster + "->" + destination_cluster`
- `origin_cluster` (text, fk -> clusters.cluster_id)
- `destination_cluster` (text, fk -> clusters.cluster_id)
- `job_count` (integer)
- `median_price_per_m3` (real)
- `p10_price_per_m3` (real)
- `p25_price_per_m3` (real)
- `p75_price_per_m3` (real)
- `p90_price_per_m3` (real)
- `break_even_per_m3` (real)
- `median_margin_per_m3` (real)
- `median_margin_pct` (real)
- `avg_distance_km` (real)
- `avg_duration_hr` (real)
- `avg_speed_kph` (real)
- `visibility_state` (text) — `phantom`, `active`, `mature`
- `last_job_date` (text)
- `updated_at` (text)

Indexes:
- `directional_corridors(origin_cluster, destination_cluster)`

### `corridor_groups`

Bidirectional corridor grouping (A ↔ B).

Columns:
- `group_id` (text, pk) — `sorted(origin_cluster, destination_cluster)`
- `cluster_a` (text, fk -> clusters.cluster_id)
- `cluster_b` (text, fk -> clusters.cluster_id)
- `total_jobs` (integer)
- `imbalance_ratio` (real) — outbound/inbound job ratio
- `median_price_per_m3` (real)
- `p50_price_per_m3` (real)
- `break_even_per_m3` (real)
- `median_margin_per_m3` (real)
- `updated_at` (text)

Indexes:
- `corridor_groups(cluster_a, cluster_b)`

### `corridor_time_stats`

Directional corridor performance by time buckets.

Columns:
- `corridor_id` (text, fk -> directional_corridors.corridor_id)
- `time_bucket` (text) — `night`, `morning_peak`, `day`, `evening_peak`
- `day_group` (text) — `weekday`, `weekend`
- `job_count` (integer)
- `avg_duration_hr` (real)
- `avg_speed_kph` (real)
- `median_margin_per_m3` (real)
- `updated_at` (text)

Primary key:
- `(corridor_id, time_bucket, day_group)`

### `corridor_volume_stats`

Optional table for $/m3 distributions by corridor and volume bucket.

Columns:
- `corridor_id` (text, fk -> directional_corridors.corridor_id)
- `volume_bucket` (text) — e.g. `0-10`, `10-20`, `20-30`, `30-40`, `40+`
- `job_count` (integer)
- `p10_price_per_m3` (real)
- `p25_price_per_m3` (real)
- `p50_price_per_m3` (real)
- `p75_price_per_m3` (real)
- `p90_price_per_m3` (real)
- `break_even_per_m3` (real)
- `loss_rate` (real)
- `volatility_price_per_m3` (real)
- `updated_at` (text)

Primary key:
- `(corridor_id, volume_bucket)`

### `corridor_opportunities`

Optional table for phantom corridor signals.

Columns:
- `corridor_id` (text, fk -> directional_corridors.corridor_id)
- `job_count` (integer)
- `median_price_per_m3` (real)
- `break_even_per_m3` (real)
- `margin_per_m3` (real)
- `phantom_score` (real)
- `predicted_demand` (real)
- `observed_jobs` (integer)
- `gap_ratio` (real)
- `opportunity_score` (real)
- `updated_at` (text)

## Aggregation Flow

1. Build clusters from job origins/destinations (geohash or city normalization).
2. Assign `origin_cluster` and `destination_cluster` per job.
3. Aggregate directional corridor stats.
4. Aggregate corridor groups from directional corridors.
5. Aggregate time buckets for directional corridors.
6. Derive visibility state from job counts (default threshold: 6 jobs).

## Time Buckets (Default)

- `night`: 22:00–05:59
- `morning_peak`: 06:00–09:29
- `day`: 09:30–15:59
- `evening_peak`: 16:00–21:59

## Open Decisions

- Cluster source of truth: geohash vs postcode vs manual overrides.
- Whether to store clusters in `jobs` as denormalized fields for fast filters.
- Threshold for corridor activation (min job count).
- Volume bucket boundaries for $/m3 distributions.
