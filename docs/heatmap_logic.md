# Heatmap logic overview

## Current flows

### Historical density heatmap (analytics/price_distribution.py)
- `build_heatmap_source` maps each job into two coordinate rows (origin/destination) with a configurable weight column; null coordinates or weights are dropped before concatenation. Metro filtering is applied via `filter_jobs_by_distance` when requested, using distance fields to keep the points close to each other.
- `available_heatmap_weightings` drives the UI selector by exposing metrics that are present in the current dataframe (e.g., volume or margin variants). Missing columns are excluded rather than throwing, so the dashboard avoids showing unusable options.
- The route heatmap tab builds the source dataframe, chooses colour scales per metric, and applies Plotly density settings with fixed radius/opacity defaults. Hover templates adapt to the chosen weighting to keep units consistent.

### Live network heatmap (analytics/live_data.py)
- `build_live_heatmap_source` merges up to three datasets—historical corridor endpoints, active routes, and live truck positions—into a unified point cloud. Each source gets a static weight (1× historical, 3× active routes, 5× live trucks) and a `source` label to distinguish contributions.
- The live map toggles between the layered overlay and the heatmap mode. When heatmap mode is active, Streamlit sliders set the radius and intensity passed into the Deck.gl `HeatmapLayer`, and the assembled source frame is sent straight to pydeck without additional normalisation.

### Test coverage
- `tests/test_price_distribution.py` asserts weight handling, metro filtering, and input validation for the historical heatmap builder, including fallbacks when a weight column is missing or invalid.
- `tests/test_live_data.py` checks that the live heatmap emphasises active and truck points, preserves source labels, and drops empty/invalid coordinates.

## Observed strengths
- Separation of concerns: data shaping lives in `analytics/*`, while visualisation stays in `dashboard/components`, keeping the Streamlit views thin.
- Defensive handling of missing data: both builders tolerate absent columns or empty frames, returning empty dataframes rather than raising.
- Metric-aware presentation: hover templates and colour scales adjust to the selected weighting so units stay intelligible.

## Improvement opportunities
- **Dynamic weighting controls:** expose the live heatmap weights (historical/active/truck) as UI sliders or config constants so operators can tune how aggressively live signals dominate the density layer.
- **Normalisation & deduplication:** normalise weights by recent observation windows and de-duplicate overlapping coordinates to prevent clusters from overpowering the map when multiple sources share the same endpoint.
- **Viewport-aware radius defaults:** derive Plotly/pydeck radius and intensity from zoom level or point counts to reduce manual tweaking and keep heatmap feel consistent across regions with sparse versus dense data.
- **Validation telemetry:** log how many rows are dropped for missing coordinates or weights and surface the counts in the UI to help data-quality triage.
- **Persisted weighting presets:** capture the chosen weighting/metro options per user session to allow quick recall of commonly used analytic views without reconfiguration.
