"""
VAAYU DARSHIYA — FastAPI Backend (Final)
Serves XGBoost predictions in the exact format the frontend expects.
All data is cached at startup so responses are instant.

Data layer:
    Observed readings (weather, weekly trend, observed pollutants) come from
    the project database: data/vaayu_final_v3.csv, projected by backend/database.py
    into data/vaayu_extract_7sites.csv (7 served stations).
    Model predictions still come from models/predictions/*.csv.

Endpoints:
    GET /api/forecast        → 24h NO2 + O3 + HCHO arrays
    GET /api/stations        → 7 sites current NO2/O3/HCHO + weather
    GET /api/metrics         → model R2, MAE, RMSE
    GET /api/weekly-trend    → last 14 days observed NO2/O3/PM2.5/HCHO

Run:
    cd backend/
    uvicorn api:app --port 8000
"""

import os
import json
import math
import joblib
import logging
import numpy as np
import pandas as pd
from datetime import datetime as _dt
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import database

# Multi-model ensemble + live/static source detection (surgical additions).
import ensemble
import data_source

# AI Insights live research (TinyFish) — isolated router, see ai_insights.py.
from ai_insights import router as ai_insights_router

# Live web context (TinyFish) for Alerts / Best Time / Delhi Case — context_api.py
from context_api import router as context_router

# Live layer: SSE broadcast, /api/alerts, /api/weather, /api/status, admin usage.
import live_data
import live_jobs
import live_stream
import tinyfish_service
from live_api import router as live_router

ROOT_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR     = os.path.join(ROOT_DIR, "models", "exp09v2_models")
SCALER_DIR    = os.path.join(ROOT_DIR, "models", "scalers")
PRED_DIR      = os.path.join(ROOT_DIR, "models", "predictions")
RESULT_DIR    = os.path.join(ROOT_DIR, "models", "results")
DATA_DIR      = os.path.join(ROOT_DIR, "data")
# Offline-built HCHO hotspot layer (scripts/build_hcho_hotspots.py).
HCHO_HOTSPOTS_JSON = os.path.join(DATA_DIR, "hcho_hotspots.json")
TARGETS      = ["NO2_target", "O3_target", "HCHO_target"]
FORECAST_LEN = 24
N_SITES      = 7

# HCHO is a satellite column stored in mol/m² in the database and trained,
# scored and served in µmol/m² (×1e6) — see notebooks/exp_11_hcho_xgboost.py.
HCHO_SCALE = 1e6

SITE_META = [
    {"site_id": 1, "name": "Ashok Vihar",       "lat": 28.69536, "lon": 77.18169},
    {"site_id": 2, "name": "Dwarka",             "lat": 28.57180, "lon": 77.07125},
    {"site_id": 3, "name": "Jawaharlal Nehru",   "lat": 28.58278, "lon": 77.23441},
    {"site_id": 4, "name": "Northwest Delhi",    "lat": 28.82286, "lon": 77.10197},
    {"site_id": 5, "name": "East Delhi",         "lat": 28.53077, "lon": 77.27123},
    {"site_id": 6, "name": "Northwest Delhi 2",  "lat": 28.72954, "lon": 77.09601},
    {"site_id": 7, "name": "Northeast Delhi",    "lat": 28.71052, "lon": 77.24951},
]

# ── Pre-computed caches (populated at startup) ──────────────────────────
cached_stations = []     # list[dict] for /api/stations
cached_forecasts = {}    # site_id -> {"no2": [...24], "o3": [...24], "hcho": [...24]}

print("Loading XGBoost models ...")
feat_scaler  = joblib.load(os.path.join(SCALER_DIR, "exp09v2_feat_scaler.pkl"))
feature_cols = joblib.load(os.path.join(SCALER_DIR, "exp09v2_feature_cols.pkl"))

models = {}
for tgt in TARGETS:
    paths = [os.path.join(MODEL_DIR, f"{tgt}_h{h:02d}.pkl") for h in range(FORECAST_LEN)]
    if not all(os.path.exists(p) for p in paths):
        if tgt == "HCHO_target":
            # Older deployment without the EXP_11 artefacts — predictions are
            # what actually get served, so only HCHO is dropped.
            print("  HCHO model files not found — HCHO forecast disabled")
            continue
        raise FileNotFoundError(f"missing model files for {tgt} in {MODEL_DIR}")
    models[tgt] = [joblib.load(p) for p in paths]
    print(f"  Loaded {FORECAST_LEN} models for {tgt}")

print("Loading prediction CSVs ...")
pred_no2 = pd.read_csv(os.path.join(PRED_DIR, "exp09v2_NO2_target_predictions.csv"),
                        parse_dates=["datetime"])
pred_o3  = pd.read_csv(os.path.join(PRED_DIR, "exp09v2_O3_target_predictions.csv"),
                        parse_dates=["datetime"])
# EXP_11 (notebooks/exp_11_hcho_xgboost.py) — absent only in deployments that
# predate the HCHO model, in which case the HCHO forecast is omitted.
_hcho_pred_path = os.path.join(PRED_DIR, "exp09v2_HCHO_target_predictions.csv")
pred_hcho = pd.read_csv(_hcho_pred_path, parse_dates=["datetime"]) \
    if os.path.exists(_hcho_pred_path) else None
if pred_hcho is None:
    print("  HCHO prediction CSV not found — HCHO forecast disabled")

# ── Build caches at startup (one-time cost) ────────────────────────────
cached_metrics = {}
cached_feature_importance = {}

# ── Project the project database down to the 7 served stations ────────
# First run reads data/vaayu_final_v3.csv (2.94 GB) once and writes
# data/vaayu_extract_7sites.csv; later runs load the ~21 MB extract.
print("Loading project database ...")
DB_INFO = database.describe()
print(f"  Database: {DB_INFO}")

print("Building station & forecast caches ...")

# ── Find each site's most recent complete prediction run ──────────────
# The prediction files hold an hourly series of issue-times ("anchors"), each
# with horizons 1..24. Serving full[0] meant the UI showed the very first run
# in the file (2023-01-02) forever; full[-1] is the newest one available.
_today = _dt.now()


def _latest_complete_anchor(pred: pd.DataFrame, site_id: int):
    """Newest issue-time for a site that has all 24 horizons, else its max."""
    site = pred[pred["site_id"] == site_id]
    if site.empty:
        return None
    anchors = site.groupby("datetime")["horizon"].count()
    full = anchors[anchors == FORECAST_LEN].index
    return full[-1] if len(full) else site["datetime"].max()


_site_dates = {}
for meta in SITE_META:
    sid = meta["site_id"]
    _site_dates[sid] = _latest_complete_anchor(pred_no2, sid) or pd.Timestamp(_today)

# Global matched date = site 1's prediction date (used by /api/time)
_matched_anchor = _site_dates.get(1, pd.Timestamp(_today))
print(f"  Site runs: {', '.join(f'{s}: {_site_dates[s]}' for s in range(1, N_SITES+1))}")


def _forecast_by_hour(pred: pd.DataFrame, site_id: int, anchor) -> list:
    """24 predictions indexed by HOUR OF DAY (0-23), i.e. index == `timeIdx`.

    A row at issue-time A with horizon h predicts the reading at A + h hours,
    which falls on hour (A.hour + h) % 24. Rotating by the anchor hour is what
    makes `array[current_hour]` mean "the prediction for this hour" instead of
    "the h-th row in the file".
    """
    site = pred[pred["site_id"] == site_id]
    if site.empty:
        return [0.0] * FORECAST_LEN

    rows = site[site["datetime"] == anchor]
    if len(rows) != FORECAST_LEN:
        # Target file has no matching run — fall back to its own newest one.
        fallback = _latest_complete_anchor(pred, site_id)
        if fallback is None:
            return [0.0] * FORECAST_LEN
        anchor = pd.Timestamp(fallback)
        rows = site[site["datetime"] == anchor]
        if rows.empty:
            return [0.0] * FORECAST_LEN

    out = [0.0] * FORECAST_LEN
    for _, r in rows.iterrows():
        h = int(r["horizon"])
        if 1 <= h <= FORECAST_LEN:
            out[(anchor.hour + h) % FORECAST_LEN] = float(r["prediction"])
    return [round(v, 1) for v in out]


for meta in SITE_META:
    sid = meta["site_id"]
    anchor = _site_dates[sid]

    # --- 24h model forecast, indexed by hour of day ---
    no2_24 = _forecast_by_hour(pred_no2, sid, anchor)
    o3_24  = _forecast_by_hour(pred_o3, sid, anchor)
    # EXP_11 HCHO forecast. Its newest complete run falls back to the HCHO
    # file's own anchor (the NO2 anchor may predate/postdate its test period),
    # exactly as O3 falls back when NO2's anchor is missing from its file.
    hcho_24 = _forecast_by_hour(pred_hcho, sid, anchor) if pred_hcho is not None else None

    # --- observed conditions from the project database (freshest row) ---
    latest = database.latest_timestamp(sid)
    when = latest if latest is not None else pd.Timestamp(anchor)
    weather = database.current_weather(sid, when)
    obs = database.current_observation(sid, when)

    cached_stations.append({
        "name":        meta["name"],
        "site_id":     sid,
        # model output — refreshed per request from cached_forecasts
        "no2":         no2_24[_today.hour],
        "o3":          o3_24[_today.hour],
        "hcho":        (hcho_24[_today.hour] if hcho_24 is not None else None),
        "lat":         meta["lat"],
        "lon":         meta["lon"],
        **weather,
        # extras from the new database — optional for existing consumers
        "obsNo2":      obs["no2"],
        "obsO3":       obs["o3"],
        "obsHcho":     obs.get("hcho"),
        "pm25":        obs["pm25"],
        "aqi":         obs["aqi"],
        "aqiCategory": obs["aqi_category"],
        "district":    obs["district"],
        "season":      obs["season"],
        "observedAt":  str(when),
        "modelRunAt":  str(anchor),
    })

    cached_forecasts[sid] = {
        "no2":    no2_24,
        "o3":     o3_24,
        # omitted (not null) when the HCHO artefacts are absent so consumers
        # can tell "no HCHO model" from "HCHO value unknown".
        **({"hcho": hcho_24} if hcho_24 is not None else {}),
        "label":  meta["name"],
        "weather": weather,
        "matched_date": str(pd.Timestamp(anchor).date()),
        "current_hour": _today.hour,
    }

# ── Per-model forecast caches (multi-model serving) ───────────────────
# Until now only XGBoost had its numbers baked into cached_forecasts, so
# selecting RandomForest/LightGBM/LSTM/Ridge changed the *label* the API
# reported while every chart kept drawing the XGBoost arrays. Each model
# that has its own prediction CSVs now gets its own set of 24h arrays, and
# /api/forecast + /api/stations serve whichever one is active.
#   model_name -> site_id -> {"no2": [...], "o3": [...], "hcho": {...}, "anchor": ts}
MODEL_FORECASTS: dict = {}


def _pred_frames_for(model_name: str) -> dict:
    """Target -> prediction DataFrame for one model.

    XGBoost reuses the frames already read above; the other models are read
    through the ensemble loader, which returns None when their CSVs are absent
    (i.e. still training).
    """
    if model_name == ensemble.XGBOOST_LABEL:
        frames = {"NO2": pred_no2, "O3": pred_o3}
        if pred_hcho is not None:
            frames["HCHO"] = pred_hcho
        return frames
    frames = {}
    for tgt in ("NO2", "O3", "HCHO"):
        df = ensemble._load_preds(model_name, tgt)
        if df is not None and not df.empty:
            frames[tgt] = df
    return frames


def _build_model_forecasts() -> None:
    """Bake 24h arrays for every model that has prediction artefacts."""
    for model_name in ensemble.MODEL_ORDER:
        frames = _pred_frames_for(model_name)
        if "NO2" not in frames:
            continue
        sites = {}
        for meta in SITE_META:
            sid = meta["site_id"]
            # Anchors are per model: a model's newest run need not line up
            # with XGBoost's, so each is resolved from its own file.
            anchor = _latest_complete_anchor(frames["NO2"], sid) \
                or pd.Timestamp(_today)
            entry = {
                "no2": _forecast_by_hour(frames["NO2"], sid, anchor),
                "anchor": anchor,
            }
            if "O3" in frames:
                entry["o3"] = _forecast_by_hour(frames["O3"], sid, anchor)
            if "HCHO" in frames:
                entry["hcho"] = _forecast_by_hour(frames["HCHO"], sid, anchor)
            sites[sid] = entry
        MODEL_FORECASTS[model_name] = sites
        print(f"  Forecast cache: {model_name} "
              f"({', '.join(sorted(frames))})")


_build_model_forecasts()


def _usable(arr) -> bool:
    """An all-zero 24h array means 'this model has no data for this site'.

    Predictions are clipped >= 0 at train time and a flat zero run is not a
    plausible pollutant series, so treat it as missing and let the caller fall
    back per pollutant (e.g. RandomForest's HCHO CSV lacks site 3 while
    XGBoost's has it — without this the dashboard would show 0 everywhere).
    """
    return arr is not None and any(v > 0 for v in arr)


def _resolve_forecast(model_name: Optional[str], site_id: int):
    """(arrays_dict, anchor) for a model+site, falling back to XGBoost.

    The fallback is per POLLUTANT, not per model. Training writes one
    prediction CSV per target at a time, so a model can legitimately have NO2
    on disk while its O3/HCHO files are still being fitted — the registry then
    marks it available with a NO2-only cache. Serving that entry verbatim would
    raise KeyError on `o3` and 500 the endpoint, so any pollutant the chosen
    model lacks (or has only zeros for) is taken from XGBoost, which always has
    all three.
    """
    xgb = MODEL_FORECASTS.get(ensemble.XGBOOST_LABEL, {}).get(site_id) or {}
    cache = MODEL_FORECASTS.get(model_name or "", {}).get(site_id) or {}

    no2, o3, hcho = cache.get("no2"), cache.get("o3"), cache.get("hcho")
    out = {
        "no2": no2 if _usable(no2) else xgb.get("no2", [0.0] * FORECAST_LEN),
        "o3":  o3  if _usable(o3)  else xgb.get("o3",  [0.0] * FORECAST_LEN),
    }
    if not _usable(hcho):
        hcho = xgb.get("hcho")
    if hcho is not None:
        out["hcho"] = hcho

    if cache and _usable(no2):
        return out, cache.get("anchor")
    if xgb:
        return out, xgb.get("anchor")

    fc = cached_forecasts.get(site_id, {})      # registry absent entirely
    out["no2"] = fc.get("no2", [0.0] * FORECAST_LEN)
    out["o3"] = fc.get("o3", [0.0] * FORECAST_LEN)
    if "hcho" in fc:
        out["hcho"] = fc["hcho"]
    return out, None

# ── Cache real metrics from training CSVs ──────────────────────────────
# One results CSV per target, identical schema. NO2/O3 come from EXP_09,
# HCHO from EXP_11 (notebooks/exp_11_hcho_xgboost.py); HCHO rows are absent
# only in deployments that predate the HCHO artefacts.
RESULT_TARGETS = [("NO2", "NO2_target"), ("O3", "O3_target"), ("HCHO", "HCHO_target")]
_res_dfs = {}
for _label, _tgt in RESULT_TARGETS:
    _p = os.path.join(RESULT_DIR, f"exp09v2_xgb_{_tgt}_results.csv")
    if os.path.exists(_p):
        _res_dfs[_label] = pd.read_csv(_p)


def _h0(df: pd.DataFrame):
    return df[df["horizon"] == 0].iloc[0]


if "NO2" in _res_dfs and "O3" in _res_dfs:
    cached_metrics = {
        "t1h": {
            label: {"r2": round(float(h0["r2"]), 4), "mae": round(float(h0["mae"]), 2),
                    "rmse": round(float(h0["rmse"]), 2), "best_iter": int(h0["best_iter"]),
                    "val_r2": round(float(h0["val_r2"]), 4), "val_mae": round(float(h0["val_mae"]), 2),
                    "val_rmse": round(float(h0["val_rmse"]), 2)}
            for label, df in _res_dfs.items() for h0 in [_h0(df)]
        },
        "t6h": {
            label: {k: (round(float(row[k]), 4) if k == "r2" else round(float(row[k]), 2))
                    for k in ("r2", "mae", "rmse")}
            for label, df in _res_dfs.items() for row in [df[df["horizon"] == 5].iloc[0]]
        },
        "t12h": {
            label: {k: (round(float(row[k]), 4) if k == "r2" else round(float(row[k]), 2))
                    for k in ("r2", "mae", "rmse")}
            for label, df in _res_dfs.items() for row in [df[df["horizon"] == 11].iloc[0]]
        },
        "t24h": {
            label: {k: (round(float(row[k]), 4) if k == "r2" else round(float(row[k]), 2))
                    for k in ("r2", "mae", "rmse")}
            for label, df in _res_dfs.items() for row in [df[df["horizon"] == 23].iloc[0]]
        },
        "all_horizons": {
            label: [{"horizon": int(r["horizon"]), "r2": round(float(r["r2"]), 4),
                     "mae": round(float(r["mae"]), 2), "rmse": round(float(r["rmse"]), 2)}
                    for _, r in df.iterrows()]
            for label, df in _res_dfs.items()
        },
        "validation_comparison": {
            label: {
                "test": {"r2": round(float(h0["r2"]), 4), "mae": round(float(h0["mae"]), 2),
                         "rmse": round(float(h0["rmse"]), 2)},
                "val":  {"r2": round(float(h0["val_r2"]), 4), "mae": round(float(h0["val_mae"]), 2),
                         "rmse": round(float(h0["val_rmse"]), 2)},
            }
            for label, df in _res_dfs.items() for h0 in [_h0(df)]
        },
    }

def _num(value, digits: int = 4):
    """Round a value for JSON — NaN/inf become None (strict JSON has no NaN)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return round(f, digits) if math.isfinite(f) else None


# ── Cache seasonal error breakdown ────────────────────────────────────
cached_seasonal = {}
for _label, _tgt in RESULT_TARGETS:
    _p = os.path.join(RESULT_DIR, f"exp09v2_{_tgt}_error_by_season.csv")
    if os.path.exists(_p):
        _sdf = pd.read_csv(_p)
        cached_seasonal[_label] = [
            {"season": r["season"], "horizon": int(r["horizon"]),
             "r2": _num(r["r2"]), "mae": _num(r["mae"], 2), "rmse": _num(r["rmse"], 2)}
            for _, r in _sdf.iterrows()
        ]

# ── Cache site-wise error breakdown ───────────────────────────────────
cached_site_errors = {}
site_names = {s["site_id"]: s["name"] for s in SITE_META}
for _label, _tgt in RESULT_TARGETS:
    _p = os.path.join(RESULT_DIR, f"exp09v2_{_tgt}_error_by_site.csv")
    if os.path.exists(_p):
        _sidf = pd.read_csv(_p)
        cached_site_errors[_label] = [
            {"site_id": int(r["site_id"]),
             "site_name": site_names.get(int(r["site_id"]), f"Site {int(r['site_id'])}"),
             "horizon": int(r["horizon"]),
             "r2": _num(r["r2"]), "mae": _num(r["mae"], 2), "rmse": _num(r["rmse"], 2)}
            for _, r in _sidf.iterrows()
        ]

# ── Cache accuracy of the model against the project database ──────────
# Produced by scripts/evaluate_accuracy.py — scores models/predictions/*.csv
# against data/vaayu_extract_7sites.csv over their overlapping period.
_dbval_path = os.path.join(RESULT_DIR, "exp09v2_database_validation.json")
if os.path.exists(_dbval_path):
    try:
        with open(_dbval_path, encoding="utf-8") as fh:
            cached_metrics = {**cached_metrics, "database_validation": json.load(fh)}
        print("  Database validation: loaded")
    except (OSError, ValueError) as exc:
        print(f"  Database validation: skipped ({exc})")

# ── Cache feature importance (top 10 from h=0 models) ─────────────────
cached_feature_importance = {}
for _label, _tgt in RESULT_TARGETS:
    _p = os.path.join(MODEL_DIR, f"{_tgt}_h00_importance.csv")
    if os.path.exists(_p):
        _idf = pd.read_csv(_p).head(10)
        cached_feature_importance[_label] = [
            {"feature": r["feature"], "importance": round(float(r["importance"]) * 100, 1)}
            for _, r in _idf.iterrows()
        ]

print("Ready — all data cached.")

# ── Multi-model ensemble (surgical addition) ─────────────────────────────
# Loads metrics + rolling-7-day RMSE for all five models, and computes which
# one is active. Defensive: a failure here must never block the server — XGBoost
# (always present) remains the fallback and the app behaves as before.
MODEL_REGISTRY = {}
_active_model = ensemble.XGBOOST_LABEL
_active_reason = "XGBoost default"
_override_model = None          # sticky manual override from POST /api/models/override
_ds_report = data_source.DataSourceReport(source="static", is_live=False,
                                          detail="not evaluated yet")

try:
    MODEL_REGISTRY = ensemble.build_registry()
    _active_model, _active_reason = ensemble.select_active(MODEL_REGISTRY)
    print(f"Ensemble ready — active model: {_active_model} ({_active_reason})")
except Exception as exc:                                   # noqa: BLE001
    print(f"  Ensemble unavailable ({exc}) — serving XGBoost only")
    MODEL_REGISTRY = {}


def current_model_state(feature_row=None):
    """Recompute the active model + data source for one prediction run.

    Returns (model_name, reason, DataSourceReport). Honours a manual override
    and applies the fallback rules (<3 features → Ridge, >40% missing → RF).
    """
    global _active_model, _active_reason, _ds_report
    _ds_report = data_source.detect(feature_row)

    if not MODEL_REGISTRY:
        _active_model, _active_reason = ensemble.XGBOOST_LABEL, "ensemble unavailable"
        return _active_model, _active_reason, _ds_report

    available, missing_frac = ensemble.feature_availability(feature_row)
    _active_model, _active_reason = ensemble.select_active(
        MODEL_REGISTRY, available_features=available,
        missing_fraction=missing_frac, override=_override_model)
    return _active_model, _active_reason, _ds_report


app = FastAPI(title="Vaayu Darshiya API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# AI Insights only — adds GET /api/ai-insights/research and nothing else.
app.include_router(ai_insights_router)

# Context routes — adds /api/context/* and nothing else.
app.include_router(context_router)

# Live routes — /api/stream (SSE), /api/alerts, /api/weather, /api/status,
# /api/admin/tinyfish-usage. Existing endpoints are untouched.
app.include_router(live_router)


@app.on_event("startup")
async def _start_live_layer():
    """Capture the event loop for SSE, then start the scheduler.

    Boot must not block on TinyFish: until this hook returns, uvicorn has not
    finished starting, so an awaited refresh would leave `/health`, `/api/*`
    and the SSE socket all unreachable for as long as a browser run takes.
    The scheduler performs the first refresh itself (job A immediately, the
    rest staggered over the first ~75 s), which gives a first visitor data
    just as quickly without holding the port closed. A failed refresh simply
    leaves the store empty and every card reports its loading/unavailable
    state instead of showing a stale number as live.
    """
    import asyncio
    live_stream.set_event_loop(asyncio.get_event_loop())
    live_store_register_sink()
    live_jobs.start()


@app.on_event("shutdown")
async def _stop_live_layer():
    live_jobs.stop(timeout=3.0)
    if tinyfish_service.is_expired() or not tinyfish_service.is_enabled():
        try:
            tinyfish_service.delete_all_monitors()
        except Exception:                          # noqa: BLE001
            pass


def live_store_register_sink():
    """Persist every TinyFish call so the usage report survives a restart.

    The same rows back the budget baseline: without it, a restarted process
    would see `calls_used() == 0` and happily spend a second 400-call
    allowance inside the same reset day.

    `usage_total_for_day` is passed straight through as the baseline because
    it already takes the day — the wrapper supplies it while holding its usage
    lock, so the baseline must not try to read it back (see set_usage_baseline).
    """
    import live_store
    tinyfish_service.set_usage_sink(live_store.record_usage)
    tinyfish_service.set_usage_baseline(live_store.usage_total_for_day)


@app.get("/")
def health():
    return {
        "status": "ok",
        "project": "Vaayu Darshiya",
        "model": "XGBoost EXP_12 (database retrain)",
        "database": DB_INFO,
    }


@app.get("/health")
def health_probe():
    """Liveness probe — polled by local tooling and Render.

    Extended with the last successful refresh per job (spec §6) so a stale
    source is visible without logging into the box.
    """
    return {
        "status": "ok",
        "live_active": tinyfish_service.is_active(),
        "last_refresh": live_jobs.health_payload(),
        "server_time": live_jobs.now_ist().isoformat(timespec="seconds"),
    }


@app.get("/api/forecast")
def get_forecast(site: int = 1):
    if site < 1 or site > N_SITES:
        site = 1

    # Which model + data source drove this run (surgical addition).
    # Resolved FIRST so the arrays below are that model's own output.
    model_name, model_reason, ds = current_model_state()
    logging.getLogger("vaayu.api").info(
        "forecast site=%s active_model=%s (%s) data_source=%s",
        site, model_name, model_reason, ds.label)

    # Serve the ACTIVE model's arrays, not XGBoost's. Switching the model in
    # the dashboard now changes every chart, not just the badge.
    model_arrays, model_anchor = _resolve_forecast(model_name, site)
    fc = cached_forecasts[site]
    served = {**fc, "no2": model_arrays["no2"], "o3": model_arrays["o3"]}
    if "hcho" in model_arrays:
        served["hcho"] = model_arrays["hcho"]

    # The cached hour is the process start time; report the live one so a
    # long-running server still tells the UI which slot is "now".
    anchor = model_anchor if model_anchor is not None else _site_dates.get(site)
    anchor_str = str(pd.Timestamp(anchor)) if anchor is not None else None

    # Spec §6: never present an archived run as live. `forecast_basis` says
    # exactly where the numbers came from, and `anchor_time` is when the model
    # was issued. The arrays themselves are untouched — anchoring to the latest
    # live reading would be a display offset only, and is reported as such.
    forecast_basis = f"Archived forecast, run {anchor_str[:10]}" if anchor_str else "Archived forecast"

    live_row = next(
        (r for r in live_data.station_payload() if r["site_id"] == site), None)
    live_no2 = (live_row or {}).get("live", {}).get("NO2")

    return {
        **served,
        "current_hour": _dt.now().hour,
        "anchor_time": anchor_str,
        "forecast_basis": forecast_basis,
        "forecast_kind": "ML forecast",
        "active_model": model_name,
        "active_model_reason": model_reason,
        "data_source": ds.label,
        "data_source_detail": ds.detail,
        "display_offset": (
            {"pollutant": "NO2", "live_value": live_no2, "unit": "ppb",
             "note": "display-only offset; model output is unchanged"}
            if live_no2 is not None else None
        ),
    }


@app.get("/api/time")
def get_time():
    now = _dt.now()
    return {
        # The date the served forecast was actually issued for. This used to be
        # hardcoded to `now.replace(year=2024)`, i.e. always 2024-10-08.
        "matched_date": str(pd.Timestamp(_matched_anchor).date()),
        "current_hour": now.hour,
        "current_minute": now.minute,
        "real_now": now.isoformat(),
    }


@app.get("/api/weekly-trend")
def get_weekly_trend(site: int = 1):
    if site < 1 or site > N_SITES:
        site = 1
    # Observed NO2 / O3 / PM2.5 over the last 14 available dates,
    # served from the project database.
    return database.weekly_trend(site, days=14)


@app.get("/api/hcho-hotspots")
def get_hcho_hotspots():
    """HCHO hotspot layer data — the offline-built JSON, served as-is.

    Built by scripts/build_hcho_hotspots.py from data/vaayu_final_v3.csv
    (satellite HCHO vertical column, mol/m² → µmol/m²). The 2.94 GB source is
    never touched at runtime; if the JSON is absent the endpoint reports why
    and the map hides the layer instead of guessing.

    `age_days` is computed per request against today, so a deployment holding
    an old build keeps telling the truth about how stale the pass is.
    """
    if not os.path.exists(HCHO_HOTSPOTS_JSON):
        return {
            "available": False,
            "reason": "HCHO hotspot file not built — run "
                      "python scripts/build_hcho_hotspots.py",
        }
    try:
        with open(HCHO_HOTSPOTS_JSON, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError) as exc:
        # No stack traces to the client — one honest line.
        return {"available": False, "reason": f"HCHO hotspot file unreadable: {exc}"}

    as_of = payload.get("as_of")
    age_days = None
    if as_of:
        try:
            age_days = (_dt.now().date() - _dt.fromisoformat(as_of).date()).days
        except ValueError:
            age_days = None

    return {
        "available": True,
        "as_of": as_of,
        "age_days": age_days,
        "archived": (age_days is not None and age_days > 2),
        "generated_at": payload.get("generated_at"),
        "region": payload.get("region"),
        "unit": payload.get("unit", "µmol/m²"),
        "window_days": payload.get("window_days"),
        "rule": payload.get("rule"),
        "notes": payload.get("notes"),
        "region_stats": payload.get("region_stats"),
        "stations": payload.get("stations", []),
        "no_recent_data": payload.get("no_recent_data", []),
        "hotspots": payload.get("hotspots", []),
    }


@app.get("/api/stations")
def get_stations():
    # no2 / o3 / hcho are model output indexed by hour of day, so re-read them
    # at the request's hour rather than the hour the process happened to boot at.
    hour = _dt.now().hour
    # The map/markers must show the same model as the charts, so resolve the
    # active model once and serve its arrays to every site.
    model_name, model_reason, _ds = current_model_state()
    out = []
    # Live web readings, merged in beside the ML values — never over them
    # (spec §3). Present only when the store has something; otherwise the
    # response shape is unchanged and the UI shows its loading state.
    live = {str(s["site_id"]): s for s in live_data.station_payload()}
    for st in cached_stations:
        model_arrays, model_anchor = _resolve_forecast(model_name, st["site_id"])
        entry = {
            **st,
            "no2": model_arrays["no2"][hour],
            "o3": model_arrays["o3"][hour],
            # Which model produced these numbers — lets the UI label the map.
            "active_model": model_name,
        }
        if "hcho" in model_arrays:
            entry["hcho"] = model_arrays["hcho"][hour]
        if model_anchor is not None:
            entry["modelRunAt"] = str(model_anchor)
        info = live.get(str(st["site_id"]))
        if info:
            # ML fields keep their names; live fields are namespaced so a
            # web-sourced number can never be mistaken for a model output.
            entry["live"] = info["live"]
            entry["liveUnits"] = info["live_units"]
            entry["status"] = info["status"]
            entry["observedAt"] = info["observed_at"]
            entry["source"] = info["source"]
            entry["ageMinutes"] = info["age_minutes"]
            entry["conflict"] = info["conflict"]
            entry["retired"] = info["retired"]
            if info["status"] == "offline":
                entry["statusNote"] = "Offline — no live source"
        out.append(entry)

    # A wrapper so the client can tell "no live data yet" from "live data off".
    return out


@app.get("/api/metrics")
def get_metrics():
    return {
        "model": "XGBoost EXP_12 (database retrain)",
        "metrics": cached_metrics,
        "feature_importance": cached_feature_importance,
        "seasonal": cached_seasonal,
        "site_errors": cached_site_errors,
        # Winner recipe across all three targets in EXP_12: ext features +
        # hyperparameter set D (see notebooks/exp_12_retrain_db.py).
        "hyperparameters": {
            "n_estimators": 2500,
            "learning_rate": 0.03,
            "max_depth": 6,
            "min_child_weight": 20,
            "gamma": 0.1,
            "subsample": 0.6,
            "colsample_bytree": 0.5,
            "reg_alpha": 0.5,
            "reg_lambda": 2.0,
            "early_stopping_rounds": 100,
        },
        # EXP_12 retrains NO2, O3 and HCHO on vaayu_final_v3.csv with one
        # chronological split: train < 2025-01-01, validation Jan-Mar 2025
        # (real data), test Apr-Jun 2025 (real data).
        "training": {
            "train_years": "2018 – Dec 2023 (< 2025-01-01)",
            "val_years": "Jan – Mar 2025 (real)",
            "test_years": "Apr – Jun 2025 (real)",
            "sites": 7,
            "targets": ["NO2_target", "O3_target", "HCHO_target"],
            "horizons": 24,
            "features": 199,
        },
    }


# ── Multi-model ensemble endpoints (surgical addition) ───────────────────
def _serialize_model(name, info):
    """One model's block for GET /api/models."""
    return {
        "name": name,
        "available": info.available,
        "active": name == _active_model,
        "rolling_rmse": info.rolling_rmse,
        # Which targets `rolling_rmse` averages over — the intersection every
        # available model shares, so the numbers are comparable across models.
        "rolling_common_targets": info.rolling_common_targets,
        "rolling_by_target": info.rolling_by_target,
        "reason": info.reason,
        "strongest_horizons": info.strongest_horizons,
        "per_target": {
            tgt: {k: {"r2": s.r2, "mae": s.mae, "rmse": s.rmse, "n": s.n}
                  for k, s in blocks.items()}
            for tgt, blocks in info.per_target.items()
        },
        "horizon_metrics": info.horizon_metrics,
    }


@app.get("/api/models")
def get_models():
    """Per-model metrics, rolling 7-day RMSE, and which model is active.

    Powers the Model Details "Models" section and the status badge. Degrades
    gracefully: with no ensemble artefacts it reports XGBoost as the only
    available model.
    """
    model_name, reason, ds = current_model_state()
    models = [_serialize_model(n, MODEL_REGISTRY[n])
              for n in ensemble.MODEL_ORDER if n in MODEL_REGISTRY]
    return {
        "active_model": model_name,
        "active_reason": reason,
        "override": _override_model,
        "data_source": ds.label,
        "data_source_detail": ds.detail,
        "models": models,
        "available_models": [m["name"] for m in models if m["available"]],
        "rolling_window_days": ensemble.ROLLING_WINDOW_DAYS,
        "fallback_rules": {
            "missing_features_threshold": ensemble.MISSING_FEATURE_FALLBACK_THRESHOLD,
            "min_features_ridge": ensemble.MIN_FEATURES_RIDGE,
        },
    }


class OverrideRequest(BaseModel):
    model: Optional[str] = None    # None / "auto" clears the override


@app.post("/api/models/override")
def set_model_override(req: OverrideRequest):
    """Manually pin the active model, or clear the override (auto-select)."""
    global _override_model, _active_model, _active_reason
    requested = (req.model or "").strip()
    if requested.lower() in ("", "auto", "none"):
        _override_model = None
    else:
        if requested not in ensemble.MODEL_ORDER:
            return {"ok": False, "error": f"unknown model '{requested}'",
                    "allowed": ensemble.MODEL_ORDER}
        if MODEL_REGISTRY and not MODEL_REGISTRY.get(requested, ensemble.ModelInfo(requested)).available:
            return {"ok": False, "error": f"model '{requested}' has no artefacts yet"}
        _override_model = requested
    model_name, reason, ds = current_model_state()
    logging.getLogger("vaayu.api").info(
        "override set to %s -> active_model=%s (%s)", _override_model, model_name, reason)
    return {"ok": True, "override": _override_model,
            "active_model": model_name, "active_reason": reason,
            "data_source": ds.label}
