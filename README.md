# VAAYU DARSHIYA

**AI-powered air quality forecasting for Delhi NCR**

Predicts **NO2**, **O3** and **HCHO** (formaldehyde) levels across 7 CPCB monitoring stations in Delhi with 24-hour ahead hourly forecasts and an interactive web dashboard.

Live: [vaayu-darshiya.vercel.app](https://vaayu-darshiya.vercel.app/)

---

## Features

- Per-horizon XGBoost forecasting for NO2, O3 and HCHO (formaldehyde)
- Engineered features — lag values, rolling statistics, cyclic time encoding, wind components, NWP forecasts
- 7 monitoring stations across Delhi NCR
- Interactive dashboard with charts, map overlays, and AI insights
- 3D hero animation with Three.js
- Data export — CSV, JSON, PDF
- Model metrics — R2, MAE, RMSE per horizon, feature importance, seasonal error breakdown
- Zero infrastructure cost — free-tier hosting, single-file project database, no DB server

---

## Monitored Stations

| # | Station |
|---|---|
| 1 | Ashok Vihar |
| 2 | Dwarka |
| 3 | Jawaharlal Nehru Stadium |
| 4 | Northwest Delhi |
| 5 | East Delhi |
| 6 | Northwest Delhi 2 |
| 7 | Northeast Delhi |

---

## Tech Stack

| Layer | Tools |
|---|---|
| Backend | Python, FastAPI, uvicorn |
| ML | XGBoost, scikit-learn, pandas, numpy |
| Frontend | TypeScript, React, TanStack Router, Tailwind CSS |
| UI | Radix UI, shadcn/ui, Recharts, Framer Motion |
| Maps | MapLibre GL JS |
| 3D | Three.js, React Three Fiber |
| Export | jsPDF, jspdf-autotable |
| Deploy | Docker, Render.com, Vercel |

---

## How the Backend Works

The backend is a **FastAPI** server that runs XGBoost models and serves predictions via a REST API.

**Startup process:**
1. Loads the feature scaler and column list from `models/scalers/`
2. Loads 72 trained XGBoost models from `models/exp09v2_models/` — one model per forecast horizon (0–23h) per target (NO2, O3, HCHO). All 72 come from the EXP_12 retrain (`notebooks/exp_12_retrain_db.py`, artefact file names keep the legacy `exp09v2_` prefix); if the HCHO artefacts are absent the HCHO forecast is simply omitted
3. Reads pre-generated prediction CSVs from `models/predictions/`
4. Opens the project database (`backend/database.py`) — on first run it projects the 7 served stations out of `data/vaayu_final_v3.csv` into `data/vaayu_extract_7sites.csv` (~21 MB); later runs load that extract
5. Pre-computes and caches all station data, forecasts, weekly trends, and metrics in memory

**The project database:**

| | |
|---|---|
| File | `data/vaayu_final_v3.csv` |
| Size | 1,461,689 rows × 213 columns, 2.94 GB |
| Coverage | 110 stations across Delhi NCR, 2016-01-29 → 2026-06-01 |
| Pollutants | NO2, O3, PM2.5, computed AQI + category |
| Other fields | ERA5 weather, satellite columns (NO2, O3, HCHO), fire/AOD, land use, population, engineered lags and rollings |

The dashboard serves 7 curated sites. `backend/database.py` maps each to its
nearest sensor in the master file and materialises them into
`data/vaayu_extract_7sites.csv`:

| Site | Station | station_id | Distance |
|---|---|---|---|
| 1 Ashok Vihar | Ashok Vihar, Delhi - DPCC | 8917 | 3 m |
| 2 Dwarka | Dwarka-Sector 8, Delhi - DPCC | 6931 | 107 m |
| 3 Jawaharlal Nehru | Jawaharlal Nehru Stadium, Delhi - DPCC | 5754 | 218 m |
| 4 Northwest Delhi | Narela, Delhi - DPCC | 10485 | 3 m |
| 5 East Delhi | Okhla Phase-2, Delhi - DPCC | 8239 | 3 m |
| 6 Northwest Delhi 2 | Rohini, Delhi - DPCC | 10831 | 2355 m |
| 7 Northeast Delhi | Sonia Vihar, Delhi - DPCC | 8475 | 3 m |

Deleting `data/vaayu_extract_7sites.csv` forces a rebuild on next boot.

**At runtime:**
- Model data (forecasts, metrics) is served from in-memory caches built at boot
- `/api/weekly-trend` and station weather/observations query the database layer, which reads only the ~21 MB extract — never the 2.94 GB source
- CORS is configured to allow the frontend domain

**5 API endpoints:**

| Endpoint | Returns |
|---|---|
| `/api/forecast?site=N` | 24-hour NO2, O3 and HCHO forecast arrays for a station |
| `/api/stations` | NO2, O3, HCHO, temperature, humidity, wind, visibility for all 7 stations, plus database fields: PM2.5, AQI, AQI category, district, season, observation time |
| `/api/metrics` | Model R2, MAE, RMSE per horizon, feature importance, seasonal and site-wise error breakdowns |
| `/api/weekly-trend?site=N` | 14-day daily average NO2, O3, HCHO and PM2.5 for a station |
| `/api/time` | Current server hour, minute, and date |

The health endpoint `/` reports database status (row count, station count, date range).

---

## How the Frontend Works

The frontend is a **TypeScript + React** single-page app built with TanStack Router and served via Vite.

**Data flow:**
1. `data.ts` defines API types and a `fetchWithFallback` function that hits the backend with an 8-second timeout
2. If the live API is unreachable, it falls back to static JSON files or hardcoded data — ensuring the dashboard always loads
3. TanStack React Query manages caching, refetching, and loading states

**Dashboard orchestration:**
- `Dashboard.tsx` is the main component — it holds the selected station, time index, and pollutant layer state
- Child components receive data as props and render independently:
  - **Charts** - 24h area chart, pollutant breakdown, station comparison, 14-day trend, 24h forecast card with an NO2 / O3 / HCHO toggle (Now / Peak / Low / Average readouts per pollutant)
  - **MapPanel** — MapLibre GL map with heatmap overlays, station markers, and an All / NO₂ / O₃ / HCHO layer toggle (All colors each station by its worst pollutant band)
  - **Metrics** — NO2, O3 and HCHO readings with trend indicators
  - **Timeline** — Time slider and quick-hour buttons to scrub through the 24h forecast
  - **AIInsights** — Textual pollution analysis
  - **Alerts** — Real-time pollution alerts by severity
  - **BestTime** — Recommends optimal outdoor time slots
  - **DataExport** — CSV, JSON, PDF downloads with UTF-8 BOM for Excel; site-wise data, 14-day trend, 24-hour forecast and a HCHO-only 24-hour export (the 24-hour chart card also downloads whichever pollutant is toggled)
  - **ModelMetrics** — R2, MAE, RMSE tables, feature importance charts, seasonal error breakdown
  - **MobiusHero** — 3D animated hero section

---

## Project Structure

```
VAAYU_DARSHIYA_MODEL_WORK/
├── backend/
│   ├── api.py              # FastAPI server (5 endpoints)
│   ├── database.py         # Data layer: projects the 7 sites out of vaayu_final_v3.csv
│   └── requirements.txt
├── FRONTEND/
│   ├── src/components/
│   │   ├── Dashboard.tsx   # Main orchestrator
│   │   ├── MobiusHero.tsx  # 3D hero
│   │   └── dashboard/      # Feature components
│   ├── vercel.json
│   └── .env.example
├── models/                 # Trained model artifacts + ACCURACY.md (evaluation report)
├── data/
│   ├── vaayu_final_v3.csv  # Project database (2.94 GB, 110 stations, not tracked)
│   ├── stations.txt        # Station metadata
│   └── vaayu_extract_7sites.csv  # Generated 7-station extract
├── notebooks/              # Experiment scripts
├── scripts/
│   ├── generate-static-api.js  # Snapshots every endpoint into FRONTEND/public/api
│   ├── evaluate_accuracy.py     # Scores the model against its own labels AND the database
│   ├── compare_retrain.py       # Paired old-vs-new retrain table + serving check
│   ├── check_forecast_skill.py  # Persistence/climatology baselines + truth-vs-DB check
│   ├── fill_missing_site_forecasts.py  # Real forecast for stations with no 2025 data (site 3)
│   ├── probe-map.js             # Headless map check: markers, basemap, per-style screenshots
│   └── probe-tabs.js            # Headless UI walk: click tabs/markers and capture the page
├── Dockerfile
└── render.yaml
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/forecast?site=N` | 24h NO2 + O3 + HCHO forecasts |
| GET | `/api/stations` | Current readings for all stations (incl. PM2.5, AQI) |
| GET | `/api/metrics` | Model performance metrics |
| GET | `/api/weekly-trend?site=N` | 14-day daily averages (NO2, O3, HCHO, PM2.5) |
| GET | `/api/time` | Current server time |

---

## AI Insights Live Research (TinyFish)

The AI Insights panel can attach recent Delhi NCR pollution news, official
advisories and credible environmental information retrieved live from the open
web through the [TinyFish](https://docs.tinyfish.ai) **Search** and **Fetch**
APIs. It is strictly additive and clearly separated from the model output:
retrieved items are labelled *External source* with their own links, while the
insight cards keep their "ML model" labels.

| | |
|---|---|
| Endpoint | `GET /api/ai-insights/research?city=Delhi` (isolated router: `backend/ai_insights.py`) |
| Service | `backend/tinyfish_service.py` (stdlib only — no new dependencies) |
| Frontend | `AIInsights.tsx` + `aiInsightsResearch.ts` (AI Insights only) |
| Key | `TINYFISH_API_KEY` — **backend only**, never sent to the browser |
| Failure mode | route answers `{"status": "unavailable", ...}` and the ML insight cards behave exactly as before |

```bash
export TINYFISH_API_KEY="sk-tinyfish-..."   # PowerShell: $env:TINYFISH_API_KEY="..."
cd backend && uvicorn api:app --port 8000
```

---

## Live Dashboard Context (TinyFish)

The **Active Alerts** and **Best Time to Go Outside** sections pull live
official context through the **same** TinyFish client
(`backend/tinyfish_service.py`) — no second client, and only **Search +
Fetch**: paid Web Automation runs and paid Monitors (which also need a public
webhook URL) are deliberately not used.

| | |
|---|---|
| Endpoints | `GET /api/context/alerts` · `POST /api/context/weather-windows` (isolated router: `backend/context_api.py`) |
| Service | `backend/tinyfish_context.py` — one function per feature, TTL cache, background scheduler |
| Frontend | `contextApi.ts` (fetch + types) → `alertMerge.ts` → `Alerts.tsx` · `BestTime.tsx` |
| Key | `TINYFISH_API_KEY` — **backend only**; read from `backend/.env` (gitignored) or the process environment |

```bash
# backend/.env (gitignored — never committed)
TINYFISH_API_KEY=sk-tinyfish-...
```

### TTLs and refresh

| Payload | TTL |
|---|---|
| `/api/context/alerts` | **20 min** (spec range 15–30 min) |
| `/api/context/weather-windows` | **6 h** (IMD updates its city forecast twice a day) |

- **Never fetched per page load.** The first request fills the cache; later
  requests are served from it until the TTL expires. A background thread
  (stdlib, no new dependency) refreshes only payloads that have been served at
  least once **and whose own TTL has elapsed**, so an unused feature costs zero
  quota and a 6 h payload is never refetched on the 20 min alerts cadence.
- `?force=true` bypasses the TTL for a manual refresh (still single-flight, so
  concurrent clicks cost one round-trip).
- **Timeouts + one retry** (6 s search / 8 s fetch). Auth and quota errors
  (401/402/403) and a missing key are never retried.
- **Graceful degradation:** on failure the last good payload is served with
  `"stale": true`, `"fetched_at"` and `"age_seconds"` of the old result (the UI
  shows *"last updated X ago"*). If nothing was ever fetched the response is
  `{"status": "unavailable", ...}` with HTTP 200 and the model cards render
  exactly as before — never a blank card or a stack trace.

### Best Time: ML windows + reasoned adjustment

`POST /api/context/weather-windows` body: `{"slots": [{period, hours, status}]}`.
The ML verdicts are echoed back as `ml_status`; the returned `status` differs
only when the official forecast triggers the rule, and then `reasons` says why:

| Trigger (from the IMD/SAFAR day forecast) | Effect |
|---|---|
| fog stated in the forecast | downgrade **one level** (`Safe → Moderate`, `Moderate → Avoid Outdoor`) + reason |
| wind range at or below `LOW_WIND_KMPH` (8 km/h) | same, reason *"Low wind: pollutants trapped (wind 5 km/h)"* |
| rain | shown as context only — verdict unchanged |

Only values the source actually states are used (`parse_weather` returns `null`
for anything it cannot match — nothing is ever guessed), and the UI renders
the ML numbers, the *"ML forecast"* label and the *"Live advisory"* block as
separate items.

Two quality gates came straight from observing the live source:

- **The city forecast page, not the homepage.** Search results are scored
  (`_weather_candidates`): landing pages (`index_en.php`, `/`) are penalised
  and pages that actually say "forecast" + "Delhi" are preferred, so the
  forecast table — *"30.8 °C … 9.3 km/h … 67 %"* — feeds the windows instead
  of navigation junk and realized observations. At most two pages are fetched
  per refresh.
- **Numbers are read unit-anchored** with a left-edge guard, so `9.3 km/h` is
  never parsed as `3 km/h` (which would have downgraded every window) and
  `29.8 °C` never becomes `8 °C`.
- **Notices must be active.** Anything older than `NOTICE_FRESH_DAYS` (90 days)
  is dropped — an 11-month-old GRAP order is background reading, not a live
  restriction.

### Sources

Only allowlisted official or reputable domains are shown, and every item is
rendered with `source`, `url`, `published_at` and `fetched_at` as a
*"Source: CAQM · 12:30 PM"* label. The allowlists sit at the top of
`backend/tinyfish_context.py`:

```python
OFFICIAL_SOURCES = { "caqm.nic.in": "CAQM", "cpcb.nic.in": "CPCB", ... }  # → "Official" badge
REPUTED_SOURCES  = { "ndtv.com": "NDTV", ... }
```

Searches are pinned to the right sites too: the Best Time weather search uses
`include_domains=mausam.imd.gov.in,imd.gov.in,safar.tropmet.res.in,dpcc.delhigovt.nic.in`
(`WEATHER_SEARCH` in `backend/tinyfish_context.py`), so only official
meteorology can reach the time windows, while crop-burning / traffic notices
come from reputable press plus the official-domain query in `NOTICE_SEARCHES`.

**To add a source:** add its host + display name to the right dict. Matching is
suffix-based, so `www.` and `airquality.` subdomains are covered:

```python
REPUTED_SOURCES = { ..., "scroll.in": "Scroll" }
```

Fetched text is treated as untrusted data: sanitized, length-capped (summaries
≤ 200 chars), and never executed or interpreted as instructions. Severity
(high/medium/low) is keyword-derived, deterministic and separate from any ML
output — live cards are labelled *Official* / *Live advisory*, model cards
*ML forecast*.

### Tests

```bash
cd backend && python -m unittest test_tinyfish_context   # 41 tests, TinyFish fully mocked
cd FRONTEND && npx vitest run src/components/dashboard/Alerts.test.tsx \
                              src/components/dashboard/BestTime.test.tsx
```

---

## Model Performance

- **Split (all targets, EXP_12):** 2018 – Dec 2023 (train, rows `< 2025-01-01`),
  Jan – Mar 2025 (validation, real data — used for model selection),
  Apr – Jun 2025 (test, real data — all reported scores)
- **Recipe:** XGBoost, 199 features (weather + satellite columns + lags/rollings),
  24 hourly horizons per target, winner of screening: extended features + hyperparameter set D
- **Data sources:** CPCB ground stations, Sentinel-5P, ERA5 reanalysis, Open-Meteo API, WAQI API

### Accuracy

| Scored against | NO2 R² | O3 R² | HCHO R² | NO2 MAE | O3 MAE | HCHO MAE |
|---|---|---|---|---|---|---|
| Model's own test labels (t+1h) | 0.685 | 0.811 | 0.541 | 11.16 | 16.47 | 4.51 |
| Model's own test labels (t+24h) | 0.650 | 0.774 | 0.485 | 11.99 | 18.40 | 4.72 |
| Project database (all horizons) | 0.507 | 0.708 | 0.232 | 14.11 | 21.06 | 5.84 |

NO2/O3 are in ppb, HCHO in µmol/m² (the database stores mol/m²; the evaluation
scales it ×1e6 before joining).

The two scoring bases now describe the same series: model `truth` and the
database match exactly at the same site + valid timestamp (correlation **1.0**,
lag 0h → 1.0 for all three targets; 106k–119k joined rows over
2025-04-01 → 2025-06-18).

**Lineage.** Before EXP_12 the production NO₂/O₃ models came from EXP_09v2,
trained on the retired `site_*_train_data.csv` series with NWP inputs
(`NO2_forecast`, `T_forecast`, …) that `vaayu_final_v3.csv` does not contain —
they scored ≈0 (or negative) R² against the database (label mismatch, not a
code bug; preserved in `models/_backup_pre_retrain_20261009/`). EXP_12 retrains
NO₂, O₃ **and** HCHO on the project database itself
(`notebooks/exp_12_retrain_db.py`), so Model Details and the database
validation now agree. Caveat: rows flagged `is_synthetic=1` (the 2023 block)
stay in the training window — validation and test are real-only, so selection
and reported scores are honest.

**Station coverage.** Site 3 (Jawaharlal Nehru Stadium, station 5754) last
reported on 2023-12-31, so it has no rows in the 2025 test window and none of
the retrained models can be scored there.
`scripts/fill_missing_site_forecasts.py` runs the trained EXP_12 models on its
final anchor instead, so the dashboard serves a real (if stale) model forecast
labelled *Model run · 2023-12-31* rather than a zero placeholder that would read
as clean air; those rows carry no truth and are excluded from every score above.

In context: persistence (last-value) baselines on the same test window are
slightly better at t+1h for NO₂ (0.76 vs 0.69) and tie for O₃ (0.81), but
collapse at longer ranges (NO₂ t+24h persistence 0.26 vs model 0.65;
climatology ≈0.26–0.32), where EXP_12 wins decisively. HCHO beats both
baselines everywhere except t+1h (persistence 0.58 vs model 0.54) — and at
t+24h reaches 0.485 where persistence is −0.60.

Regenerate the full breakdowns and report with:

```bash
python scripts/evaluate_accuracy.py   # → models/ACCURACY.md + models/results/*.csv
```

---

## Experiments

The `notebooks/` directory contains multiple experiment iterations. The production model is `exp_12_retrain_db.py` — it retrains NO2, O3 and HCHO on `vaayu_final_v3.csv` itself (chronological split, GPU) and exports every artefact the backend serves. Earlier generations (`exp_09_v2_xgboost.py`, `exp_11_hcho_xgboost.py`) are superseded and preserved in `models/_backup_pre_retrain_20261009/`.

---

## Contributing

Contributions are welcome. To get started:

1. Fork the repository
2. Create a branch for your change (`git checkout -b feature/your-feature`)
3. Make your changes and test them
4. Commit with a clear message (`git commit -m "Add: description of change"`)
5. Push to your fork and open a Pull Request

### Guidelines

- Keep PRs focused — one feature or fix per PR
- Follow existing code style (TypeScript for frontend, PEP 8 for Python)
- Test your changes before submitting
- Describe what your PR does and why
