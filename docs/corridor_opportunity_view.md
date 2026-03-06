# Corridor Opportunity View (SQL)

This view definition mirrors the corridor opportunity report schema and can be
used as a basis for implementation.

## Assumptions

- `directional_corridors` contains per-corridor stats.
- `corridor_opportunities` contains gravity + opportunity metrics.
- `corridor_groups` provides bidirectional grouping.

## View Definition (SQLite)

```sql
CREATE VIEW IF NOT EXISTS corridor_opportunity_view AS
SELECT
    dc.corridor_id,
    cg.group_id AS corridor_group_id,
    dc.origin_cluster,
    dc.destination_cluster,
    dc.job_count,
    co.predicted_demand,
    co.observed_jobs,
    co.gap_ratio,
    dc.median_price_per_m3,
    dc.break_even_per_m3,
    co.margin_per_m3 AS unit_margin_per_m3,
    co.phantom_score,
    co.opportunity_score,
    dc.last_job_date
FROM directional_corridors dc
LEFT JOIN corridor_groups cg
    ON (
        (cg.cluster_a = dc.origin_cluster AND cg.cluster_b = dc.destination_cluster) OR
        (cg.cluster_a = dc.destination_cluster AND cg.cluster_b = dc.origin_cluster)
    )
LEFT JOIN corridor_opportunities co
    ON co.corridor_id = dc.corridor_id;
```

## Notes

- If corridor groups are not yet materialized, the group join can be removed.
- If opportunity metrics are derived on the fly, replace `corridor_opportunities`
  with a subquery using gravity + $/m3 distribution fields.
