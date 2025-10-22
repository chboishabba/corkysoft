# Price history tab

This view helps trace seasonality trends across the filtered dataset, comparing the live period with the equivalent weeks or months from the previous year.

## Controls
- **Aggregation frequency** – switch between daily, weekly, or monthly resampling (internally mapped to Pandas aliases `D`, `W`, and `M`). Changing the cadence adjusts every downstream chart.
- **History range** – choose the start and end dates the resample will cover. The control snaps to the min/max dates available in the filtered dataset when possible.
- **Breakdown metric** – select which metric (Price $/m³, Margin $/m³, or Margin %) feeds the origin/destination charts and distribution summaries. Availability depends on the columns present in the filtered dataframe.

## Charts
- **Overall time series** – combines the selected metrics into a multi-trace chart. Solid lines represent the current period, dashed lines show last year's values shifted forward so timelines align. Plotly connects points from the same metric/series, so a line segment simply shows how that metric moved between successive resampled periods.
- **Origin and destination breakdowns** – two side-by-side charts apply the chosen metric to each lane. Colours follow the origin/destination names and markers highlight each aggregated point. Interpret the dashing the same way as the overall figure.
- **Previous year distribution snapshots** – when historic data exists the tab renders:
  - A histogram for the selected metric (falls back to Price $/m³ if the chosen metric is absent) showing the value spread across the previous year's period.
  - Box plots for origins and destinations highlighting the distribution and outliers for each lane.

## Data requirements and fallbacks
- A valid date column (default `job_date`) must exist. If the column is missing or contains no parsable timestamps the entire section shows guidance messages instead of charts.
- Metrics are coerced to numeric, so non-numeric values are dropped quietly. If all values are missing after coercion the chart gracefully reports that no data is available.
- Previous-year summaries only display if the historical dataset has corresponding rows in the shifted date window.

## Tips
- Keep the history range narrow when layering busy networks—weekly aggregation often strikes the best balance between noise and detail.
- If multiple lines overlap, hover the legend entries to isolate a single lane or use the legend to toggle traces on/off.
- Use the Margin % metric to spot profitable but low-revenue corridors; the box plots quickly surface persistent outliers worth deeper investigation.
