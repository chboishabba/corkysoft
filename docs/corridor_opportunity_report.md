# Corridor Opportunity Report

This document defines the output format for ranking corridor growth and pricing
opportunities using gravity demand + $/m3 distributions.

## Purpose

Surface corridors that are:
- predicted to have demand,
- under-captured by observed jobs,
- profitable per $/m3 distributions.

## Output Fields (CSV / Table)

- `corridor_id`
- `corridor_group_id`
- `origin_cluster`
- `destination_cluster`
- `job_count`
- `predicted_demand`
- `observed_jobs`
- `gap_ratio`
- `median_price_per_m3`
- `break_even_per_m3`
- `unit_margin_per_m3`
- `phantom_score`
- `opportunity_score`
- `last_job_date`

## Sorting

Primary sort:
- `opportunity_score` descending

Secondary sort:
- `unit_margin_per_m3` descending
- `predicted_demand` descending

## Recommended Filters

- Minimum `job_count` for stable distributions (e.g. 6) unless flagged as phantom.
- Optional distance band filters (e.g. 100-1000 km).

## Notes

- For corridors with `observed_jobs = 0`, use bidirectional corridor medians or
  global distance-band medians as price priors.
- `gap_ratio = predicted_demand / (observed_jobs + 1)`.
