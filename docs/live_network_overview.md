# Live Network Overview — Functional Specification

## Purpose
The Live Network Overview map gives operations and commercial teams a near-real-time, colour-coded view of lane profitability across Australia. It blends the historical cost data already captured in `jobs` and `historical_jobs` with the summarised profitability metrics in `lane_summary` and overlays the results on an interactive geographic base layer. Managers can instantly see which corridors are trading above, near, or below break-even and drill into the supporting telemetry for any lane.

## Data Sources
| Dataset | Tables / Files | Notes |
| --- | --- | --- |
| Lane profitability | `lane_summary` | Aggregated by origin/destination, stores weighted revenue, cost, volume, break-even deltas, and last-updated timestamp. |
| Historical jobs | `jobs`, `historical_jobs` | Supplies the lane catalogue, origin/destination coordinates, and baseline volume weighting. |
| Live telemetry (optional overlay) | `active_routes`, `truck_positions` | Highlights in-flight work and active trucks travelling through each lane. |
| Boundaries | GeoJSON tiles (Mapbox/Leaflet) | Base map of Australia and state borders used for context. |

> **Coordinate handling:** The map prefers pre-computed centroids, start/end coordinates, or encoded polylines stored in the DB. When a lane lacks geometry, it can fall back to the straight-line band between origin and destination using the resolved coordinates from `jobs`/`lane_summary`.

## Visualisation Overview
1. **Lane bands** – Each lane is rendered as a translucent band or polyline, colour-coded on a green→amber→red scale according to the profitability index:
   - `>= +10 %` above break-even → deep green.
   - `-5 %` to `+10 %` → amber/neutral.
   - `< -5 %` → red, with thicker stroke to draw attention.
2. **Directionality** – Bidirectional lanes are aggregated but can optionally show directional arrows derived from `lane_summary.directionality` (if available). For symmetric lanes, a single band is shown with tooltips summarising both directions.
3. **Tooltips & popovers** – Hovering over a lane reveals:
   - Lane name (`Origin → Destination`).
   - Last 30-day volume and revenue/tonnage metrics.
   - Break-even rate vs actual (`$/m³`, `$/km`).
   - Percent of trips below break-even.
   - Status badges (e.g., *Under review*, *Watchlist*).
4. **Legend & filters** – A fixed legend explains the colour scale. Filter controls (sidebar) let users:
   - Toggle between profitability thresholds (e.g., margin %, contribution $/m³).
   - Switch layers (historical vs live telemetry heatmap).
   - Filter by customer segment, lane class (metro/interstate), or minimum trip count.
5. **Live overlays** – Active routes and trucks appear as animated markers or highlighted polylines using the latest data from `active_routes` and `truck_positions`.

## Processing Pipeline
1. **Extract lanes** – Query `lane_summary` to fetch lanes with their aggregated metrics and coordinates. If the table stores JSON geometry or encoded polylines, decode those; otherwise, derive the band endpoints from `jobs`.
2. **Calculate profitability index** – For each lane:
   ```text
   profitability_index = (actual_unit_rate - break_even_unit_rate) / break_even_unit_rate
   ```
   Store both the raw percentage and the absolute dollar delta (`margin_per_m3`, `margin_per_trip`).
3. **Classify colour band** – Map the percentage to a colour bucket. The thresholds are configurable via the `global_parameters` table so finance can adjust break-even tolerances without code changes.
4. **Construct geometries** – Build Folium/Leaflet polygons or buffered polylines using origin/destination coordinates. Apply a small width buffer (5–15 km) to convey corridor breadth. Optionally snap to `active_routes.route_geometry` if precise polylines exist.
5. **Assemble feature collection** – Output a GeoJSON `FeatureCollection` with lane properties. Each feature includes tooltip content, styling metadata (colour, opacity), and unique IDs for streamlit callbacks.
6. **Blend live data** – Join active routes to their corresponding lanes to highlight current runs. Weight them higher in the heatmap layer and attach the truck IDs to the lane tooltip when an active job is present.
7. **Render map** – In Streamlit:
   - Use `st.pydeck_chart` with `PolygonLayer`/`PathLayer` for the lanes.
   - Overlay a `ScatterplotLayer` or `HeatmapLayer` for live telemetry.
   - Provide interactive filters (multiselect, sliders) tied to the data-frame powering the layers.

## User Interaction Flow
1. **Default view** – Shows the last-known state of all lanes with profitability colouring, break-even legend, and the latest refresh timestamp.
2. **Filtering** – Selecting filters re-queries the cached lane dataframe and updates the map plus summary metrics (total profitable lanes, revenue at risk, etc.).
3. **Lane drill-down** – Clicking a lane opens a side panel with:
   - Trend chart (sparkline) of margin over the last 90 days.
   - Top customers and shipment counts.
   - Recent exceptions (from `lane_summary.notes` or incident logs).
4. **Export** – Users can export the filtered dataset as CSV for offline review. The export includes profitability fields and computed KPIs.

## Refresh & Performance Considerations
- **Data refresh cadence** – The dashboard polls SQLite every 60 seconds for live telemetry and every 15 minutes for lane summaries. Batch jobs update `lane_summary` nightly; ad-hoc recomputation can be triggered from the CLI.
- **Caching** – Use Streamlit’s `st.cache_data` for lane summaries (with TTL) and `st.cache_resource` for static geometries to avoid recomputation during interactions.
- **SQLite reads** – Keep queries simple (use indices on `lane_summary.origin`, `lane_summary.destination`). Paginate or limit to top-N lanes when zoomed-out to maintain map performance.
- **Fallbacks** – If no profitability data exists, the map switches to a neutral grey palette and displays a prompt to import jobs or run the profitability aggregation.

## Dependencies & Implementation Notes
- **Python libraries** – `pandas`, `numpy`, `pydeck`/`folium`, `streamlit`, `shapely` (optional for buffering polylines).
- **Configuration** – Break-even thresholds and colour palette stored under `global_parameters` keys (`lane_margin_green`, `lane_margin_amber`).
- **Testing** – Add unit tests for the colour classification helper and GeoJSON assembler. Use fixture SQLite DBs with representative `lane_summary` rows to guarantee deterministic outputs.
- **Extensibility** – Future iterations can ingest weather or incident feeds to change lane status or integrate predictive models that forecast margin shifts based on booking pipeline data.

