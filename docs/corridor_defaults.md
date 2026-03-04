# Corridor Defaults

This appendix defines initial defaults for corridor analytics. Adjust as
real data arrives.

## Visibility Thresholds

- `phantom`: 1-5 jobs
- `active`: 6-40 jobs
- `mature`: 41+ jobs

## Volume Buckets

Default bucket edges (m3):
- `0-10`
- `10-20`
- `20-30`
- `30-40`
- `40+`

## Break-Even Constants (v1)

These are placeholders for the $/m3 break-even overlay.

- `truck_cost_per_km`: 2.10
- `crew_cost_per_hr`: 120.00
- `avg_speed_kph`: 70

Derived:
- `break_even_per_m3 = (distance_km * truck_cost_per_km + duration_hr * crew_cost_per_hr) / volume_m3`

## Phantom Corridor Rule (v1)

Flag as phantom when:
- `job_count < 10`
- `median_price_per_m3 - break_even_per_m3 > 40`

Score:
- `phantom_score = (median_price_per_m3 - break_even_per_m3) * log(job_count + 1)`

## Gravity Defaults (v1)

- `alpha = 1.0`
- `beta = 1.0`
- `gamma = 1.6`
- `K = 1.0`
- `cost_term = distance_km`

## Opportunity Score (v1)

- `gap_ij = pred_ij / (obs_jobs_ij + 1)`
- `unit_margin_ij = median_price_per_m3 - break_even_per_m3`
- `opportunity_ij = gap_ij * max(unit_margin_ij, 0)`
