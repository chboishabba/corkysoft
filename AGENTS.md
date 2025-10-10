Suggested structure:

corkysoft/
  src/
    dashboard/                   # Streamlit app (now)
      app.py                     # Streamlit entry
      pages/                     # optional multipage
    services/                    # PURE PYTHON (keep UI-free!)
      pricing.py                 # lanes, modifiers, seasons
      routing.py                 # ORS client, cache
      repo.py                    # SQLite read/write
      analytics.py               # histograms, KPIs
    models/                      # pydantic/dataclasses (optional)
    utils/                       # helpers (geo, cache)
  db/
    migrations.sql               # schema DDL
  .env.example
  requirements.txt


Please also check ROADMAP.md and README.md

Update these with any progress you make.

Ensure tests are added along with features to prevent regressions.
