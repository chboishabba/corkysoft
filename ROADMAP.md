Unified deliverables map for Corkysoft, aligning current implementation status with the planned roadmap.

---

## Documentation TODOs

- Validate the deliverable status tables against the current code and tests.
- Decide whether `corkysoft/src/dashboard` remains a packaging stub or should be wired to the main Streamlit entry point.
- Add a short contributor note on how to keep README and docs in sync with new features.
- Keep `docs/modules.md` updated when module responsibilities or entry points change.
- Add analytics documentation once the historical import and lane model ship.
- Add a system architecture diagram (truck ↔ server ↔ cloud).
- Keep `docs/positioning.md` aligned with supported integrations and product focus.

---

## 🧱 1. Core Routing & Cost Engine

| Deliverable                           | Status | Description                                        |
| ------------------------------------- | :----: | -------------------------------------------------- |
| **OpenRouteService integration**      |    ✅   | Working API calls for distance/duration.           |
| **CLI tool (`routes_to_sqlite.py`)**  |    ✅   | Add, run, list, and import jobs.                   |
| **SQLite persistence (`routes.db`)**  |    ✅   | Stores all job and geocode data.                   |
| **Geocode caching**                   |    ✅   | Prevents repeated lookups.                         |
| **Cost calculator (hourly + per-km)** |    ✅   | Calculates total cost and components.              |
| **Error handling / back-off**         |    ✅   | Handles rate limits and errors gracefully.         |
| **Address normalisation (AU)**        |    ✅   | Expands street abbreviations (e.g., cr → Circuit). |
| **CSV import/export**                 |    ✅   | Batch job ingestion.                               |
| **Folium route visualisation**        |    ✅   | Produces full-route HTML maps.                     |
| **README.md**                         |    ✅   | Complete and public-ready.                         |
| **Unit tests**                        |   🔜   | To be added for DB + API logic.                    |

---

## 🗺️ 2. Mapping & Visualisation

| Deliverable                           | Status | Description                                 |
| ------------------------------------- | :----: | ------------------------------------------- |
| **Multi-route Folium map**            |    ✅   | Working map output.                         |
| **CustomIcon fix**                    |    ✅   | Bug resolved.                               |
| **Break-even / margin overlays**      |   🔜   | Add markers for ±10–50 % profit.            |
| **Interactive dashboard (Streamlit)** |   🧩   | Core components exist; full dashboard wiring, continuous map colour scaling, and cost vs price overlays still in progress. |
| **Profit & volume heatmaps**          |   🔜   | Highlight high-margin or frequent routes.   |

---

## 📦 3. Data Model & Integration

| Deliverable                  | Status | Description                       |
| ---------------------------- | :----: | --------------------------------- |
| **Jobs + geocode tables**    |    ✅   | Implemented.                      |
| **Schema migration support** |    ✅   | Handles new columns.              |
| **Client registry & dedupe** |    ✅   | Quote builder stores clients, flags duplicates, and allows quotes without forcing a client record. |
| **Historical job import**    |   🔜   | From MoveWare CSV/sheets or operator exports. |
| **$ per m³ calculation**     |   🔜   | Derived metric for profitability. |
| **Corridor / lane table**    |   🔜   | Define recurring routes.          |
| **Modifier tables**          |   🔜   | Access, packing, season, etc.     |
| **Integration staging schema** |   🔜 | Minimal fields for CRM/dispatch/accounting sync. |
| **CSV/API connectors**       |   🔜   | Sync from CRM, dispatch, accounting, and fleet systems. |

---

## 🚛 4. Operational Business Logic

| Deliverable                                              | Status | Description                            |
| -------------------------------------------------------- | :----: | -------------------------------------- |
| **Metro vs regional logic (≤100 km)**                    |    ✅   | Rule defined.                          |
| **Base-rate schedule (Sunshine Coast 120 → Cairns 185)** |   🧩   | Exists verbally, needs table encoding. |
| **Packing / bad-access fees**                            |   🔜   | Add fixed-fee modifiers.               |
| **Seasonal margin uplift (20–80 %)**                     |   🔜   | Add seasonal coefficient.              |
| **Backhaul / container sharing**                         |   🔜   | Handle under-/over-utilisation.        |
| **Truck / driver cost baselines**                        |    ✅   | Base fuel/driver/maintenance + overhead parameters drive break-even engine. |
| **Private cost component ledger**                        |    ✅   | Record crew, truck, fuel and other cost inputs per job inside SQLite.       |

---

## 🧮 5. Analytics & Statistical Modelling

| Deliverable                            | Status | Description                                  |
| -------------------------------------- | :----: | -------------------------------------------- |
| **Airbnb-style $/m³ histogram**        |    ✅   | Jobs sorted left→right by $/m³; bar = count. |
| **Break-even + margin bands**          |    ✅   | Visual overlay on histogram.                 |
| **Loss-leader detection**              |    ✅   | Identify sub-margin jobs.                    |
| **Regression / corridor model**        |   🔜   | Predict margins vs distance/season.          |
| **Terrain & temperature factors**      |   🔜   | Weight costs for harsh routes.               |
| **Driver / truck performance metrics** |   🔜   | Wear, reliability, fuel efficiency.          |

---

## 📸 6. RFID / Camera / Audit System

| Deliverable                            | Status | Description                             |
| -------------------------------------- | :----: | --------------------------------------- |
| **Technical architecture spec**        |    ✅   | Complete multi-layer doc.               |
| **Media ingest doc (PEC/bodycam)**     |    ✅   | Capture→upload pipeline documented.     |
| **Data-model integration (PEC/media)** |   🧩   | Fields drafted; implementation pending. |
| **Pre-Existing-Condition capture**     |   🔜   | Two-photo workflow + customer sign-off. |
| **Event-based bodycam clips**          |   🔜   | Short triggered recordings.             |
| **Video/call processing (Frigate/Whisper + SFM)** |   🔜   | Stub: use Frigate for video detection, Whisper for call transcripts, and SFM to assess access constraints. |
| **Claim-risk scoring**                 |   🔜   | Use dispute data to adjust pricing.     |
| **Hash-verified storage**              |   🔜   | SHA-256 for insurer integrity.          |
| **Privacy & consent controls**         |   🔜   | Face-blur + role-based access.          |

---

## 📊 7. Dashboards & Reporting

| Deliverable                              | Status | Description                                |
| ---------------------------------------- | :----: | ------------------------------------------ |
| **CLI report (`list`)**                  |    ✅   | Clean console output.                      |
| **Streamlit dashboard (MVP)**            |   🔜   | Route map + distribution + summary.        |
| **Insurance / audit bundles (PDF)**      |   🔜   | One-click job evidence packs.              |
| **Automated CSV / Google Sheets export** |   🧩   | Helpers produce CSV-ready profitability summaries. |
| **API endpoints**                        |   🔜   | JSON / REST for ERP + insurer integration. |

---

## 🔐 8. Security & Compliance

| Deliverable                       | Status | Description                           |
| --------------------------------- | :----: | ------------------------------------- |
| **Immutable manifests / hashing** |   🧩   | Partially in design.                  |
| **Privacy safeguards**            |   🔜   | Implement face-blur, RBAC, retention. |
| **PIA / Ethics review**           |   🔜   | Required for video + RFID data.       |

---

## 🪜 9. Documentation & Governance

| Deliverable                        | Status | Description                         |
| ---------------------------------- | :----: | ----------------------------------- |
| **README.md**                      |    ✅   | Current, clear.                     |
| **GM summary**                     |    ✅   | Delivered (non-technical overview). |
| **Architecture diagram**           |   🔜   | Truck ↔ server ↔ cloud schematic.   |
| **Analytics README / docs folder** |   🔜   | Needed once dashboard built.        |

---

## 📈 Overall Progress Snapshot

| Domain                     | Progress |
| -------------------------- | -------- |
| Core Routing & Costing     | ✅ 90 %   |
| Mapping & Visualisation    | 🧩 60 %  |
| Business Logic & Rates     | 🧩 50 %  |
| Analytics & Stats          | 🔜 30 %  |
| RFID / Camera Audit        | 🧩 40 %  |
| Dashboards & Reports       | 🔜 25 %  |
| Documentation & Governance | 🧩 70 %  |

---

### ✅ Fully Met

* Routing, costing, caching, database schema, and core CLI.
* Working maps + README + documentation foundation.

### 🔜 Still Needed

1. Import historical job data (+ $m³ calc).
2. Implement lane / modifier tables.
3. Build Streamlit dashboard with histogram + margin bands.
4. Add audit media integration & privacy layer.

---

## Next Phase Blockers (2026-03-04)

- Historical job ingestion pipeline (analytics cannot mature without real data).
- Corridor / lane data model (required for benchmarking and optimisation).
- Full dashboard implementation (lane heatmaps, margin overlays, operator views).

## High-Leverage Near-Term Features

- Historical job import pipeline (CSV + MoveWare exports).
- Corridor / lane detection and rollups.
- $/m³ market benchmarking overlays (median, percentile, break-even lines).
- Quote recommendation engine (corridor median + modifiers).
- Backhaul detection and discount suggestions.
- Job profitability scoring and risk tags.
- Corridor profitability heatmap layer.
- Automated corridor pricing adjustments.
