"""
Multi-model ensemble + dynamic selection (surgical addition).

Five models now exist on the identical feature set and split:

    XGBoost       the original — artefacts in models/exp09v2_models + predictions
    RandomForest  trained by backend/train_fallback_models.py
    LightGBM      "   "      "    "
    Ridge         "   "      "    "
    LSTM          "   "      "    "   (single-layer torch.nn.LSTM)

All five are materialised at train time, so selection here is a lookup + a
cheap score — no training happens at request time.

Selection rules (select_active):
    1. manual override            -> that model (unless its artefacts absent)
    2. < 3 usable features        -> Ridge        (linear baseline, stable on few features)
    3. > 40% missing features     -> RandomForest (robust to ERA5 / cloud-cover gaps)
    4. otherwise                  -> lowest rolling-7-day RMSE across targets

The rolling RMSE in rule 4 is averaged over the targets EVERY available model
shares (see rolling_common_targets), not over whatever each model happens to
have. Averaging per model would compare XGBoost's NO2+O3+HCHO mean against a
still-training model's NO2-only mean, rewarding the latter for having fewer
targets and flipping the winner for the wrong reason.

Artefact loading is defensive: a model with no usable predictions is dropped
rather than crashing the server, so XGBoost (always present) remains the floor.
"""

import glob
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("vaayu.ensemble")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT_DIR, "models", "exp09v2_models")
PRED_DIR = os.path.join(ROOT_DIR, "models", "predictions")
ENSEMBLE_DIR = os.path.join(ROOT_DIR, "models", "ensemble")

FORECAST_LEN = 24
TARGETS = ["NO2", "O3", "HCHO"]
TARGET_KEYS = {t: f"{t}_target" for t in TARGETS}

# Canonical display order + the label XGBoost already uses in the API.
MODEL_ORDER = ["XGBoost", "RandomForest", "LightGBM", "LSTM", "Ridge"]
XGBOOST_LABEL = "XGBoost"

ROLLING_WINDOW_DAYS = 7

# Fallback thresholds (spec).
MISSING_FEATURE_FALLBACK_THRESHOLD = 0.40   # >40% missing -> RandomForest
MIN_FEATURES_RIDGE = 3                      # <3 features   -> Ridge

# A curated, stable core feature set used to judge "how many features are
# present" in the live input (mirrors pipeline WEATHER_CORE + observed).
CORE_FEATURES = [
    "temperature_2m_c", "relative_humidity", "wind_u_10m", "wind_v_10m",
    "rainfall_mm", "pm25", "sat_hcho_column", "boundary_layer_height",
    "ventilation_coeff", "stagnation_index",
]


@dataclass
class ModelScore:
    r2: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    n: int = 0


@dataclass
class ModelInfo:
    name: str
    available: bool = False
    per_target: Dict[str, Dict[str, ModelScore]] = field(default_factory=dict)
    rolling_rmse: Optional[float] = None       # mean over targets, last 7 days
    # Which targets `rolling_rmse` was averaged over — the intersection every
    # available model shares, so the lowest-RMSE selector compares like with
    # like instead of rewarding a model for having fewer targets trained.
    rolling_common_targets: List[str] = field(default_factory=list)
    rolling_by_target: Dict[str, Optional[float]] = field(default_factory=dict)
    horizon_metrics: Dict[str, List[dict]] = field(default_factory=dict)
    strongest_horizons: Dict[str, List[int]] = field(default_factory=dict)
    reason: str = ""


# ── Prediction loading ─────────────────────────────────────────────────────
def _pred_path(model_name: str, target: str) -> Optional[str]:
    """Path to a model's prediction CSV, or None."""
    if model_name == XGBOOST_LABEL:
        p = os.path.join(PRED_DIR, f"exp09v2_{TARGET_KEYS[target]}_predictions.csv")
    else:
        p = os.path.join(ENSEMBLE_DIR, f"{model_name}_{TARGET_KEYS[target]}_predictions.csv")
    return p if os.path.exists(p) else None


def _load_preds(model_name: str, target: str) -> Optional[pd.DataFrame]:
    p = _pred_path(model_name, target)
    if not p:
        return None
    try:
        df = pd.read_csv(p, parse_dates=["datetime"])
    except Exception as exc:
        log.warning("ensemble: could not read %s (%s)", p, exc)
        return None
    if df.empty or not {"datetime", "horizon", "prediction", "truth"}.issubset(df.columns):
        return None
    return df


# ── Metrics ────────────────────────────────────────────────────────────────
def _score(y, p) -> ModelScore:
    y = np.asarray(y, float); p = np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < 2:
        return ModelScore(n=int(len(y)))
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = (1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return ModelScore(
        r2=None if not np.isfinite(r2) else round(float(r2), 4),
        mae=round(float(np.mean(np.abs(y - p))), 2),
        rmse=round(float(np.sqrt(np.mean((y - p) ** 2))), 2),
        n=int(len(y)),
    )


def _rolling_rmse(df: pd.DataFrame, days: int = ROLLING_WINDOW_DAYS) -> Optional[float]:
    """Mean RMSE over the last `days` days present in the predictions."""
    if df is None or df.empty:
        return None
    end = df["datetime"].max()
    start = end - pd.Timedelta(days=days)
    win = df[df["datetime"] >= start]
    if win.empty or not np.isfinite(win["prediction"].to_numpy(float)).any():
        return None
    y = win["truth"].to_numpy(float); p = win["prediction"].to_numpy(float)
    s = _score(y, p)
    return s.rmse


def _strongest_horizons(horizon_rows: List[dict]) -> List[int]:
    """Horizons with the lowest RMSE (best skill), up to 4, in time order."""
    valid = [r for r in horizon_rows if r.get("rmse") is not None]
    if not valid:
        return []
    best = sorted(valid, key=lambda r: r["rmse"])[:4]
    return sorted(int(r["horizon"]) for r in best)


# ── Build the per-model registry (called once at startup) ─────────────────
def build_registry() -> Dict[str, ModelInfo]:
    registry: Dict[str, ModelInfo] = {}
    for name in MODEL_ORDER:
        info = ModelInfo(name=name)
        for tgt in TARGETS:
            df = _load_preds(name, tgt)
            if df is None:
                continue
            info.available = True

            # overall + per-target test metrics (all horizons pooled)
            s = _score(df["truth"].to_numpy(float), df["prediction"].to_numpy(float))
            info.per_target[tgt] = {"all": s}

            # per-horizon metrics (the Model Details table)
            rows = []
            for h, grp in df.groupby("horizon"):
                hs = _score(grp["truth"].to_numpy(float), grp["prediction"].to_numpy(float))
                rows.append({"horizon": int(h), "r2": hs.r2, "mae": hs.mae,
                             "rmse": hs.rmse, "n": hs.n})
            rows.sort(key=lambda r: r["horizon"])
            info.horizon_metrics[tgt] = rows
            info.strongest_horizons[tgt] = _strongest_horizons(rows)

            info.rolling_by_target[tgt] = _rolling_rmse(df)

        if info.available:
            vals = [v for v in info.rolling_by_target.values() if v is not None]
            info.rolling_rmse = round(float(np.mean(vals)), 2) if vals else None
            info.reason = "artefacts present"
        else:
            info.reason = "no prediction artefacts found"

        registry[name] = info

    # Averaging each model's rolling RMSE over WHATEVER TARGETS it happens to
    # have would make the comparison unfair: a model with only NO2 trained
    # averages its one easy target, while XGBoost averages NO2 + O3 + HCHO and
    # is dragged up by O3's larger scale. A model is then rewarded for having
    # fewer targets. Recompute the headline number over the targets every
    # available model has in common, so the lowest-RMSE selector compares like
    # with like. The per-target values above are untouched and still shown.
    common = None
    for name in MODEL_ORDER:
        info = registry[name]
        if not info.available:
            continue
        have = {t for t, v in info.rolling_by_target.items() if v is not None}
        common = have if common is None else (common & have)
    common = common or set()

    if common:
        for name in MODEL_ORDER:
            info = registry[name]
            if not info.available:
                continue
            vals = [info.rolling_by_target[t] for t in sorted(common)
                    if info.rolling_by_target.get(t) is not None]
            info.rolling_rmse = round(float(np.mean(vals)), 2) if vals else None
            info.rolling_common_targets = sorted(common)

    log.info("ensemble: registry built — available=%s common_targets=%s",
             [m for m, i in registry.items() if i.available], sorted(common))
    return registry


# ── Feature-availability probing (for the fallback rules) ─────────────────
def feature_availability(feature_row: Optional[dict] = None,
                         total_features: int = len(CORE_FEATURES)):
    """Return (available_count, missing_fraction) for the live input.

    `feature_row` is a dict of feature->value; a value that is None/NaN counts
    as missing. When no live row is supplied we assume full availability
    (static snapshot has every column), so no fallback is triggered.
    """
    if not feature_row:
        return total_features, 0.0
    missing = 0
    for f in CORE_FEATURES:
        v = feature_row.get(f)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            missing += 1
    return total_features - missing, missing / float(total_features)


def select_active(registry: Dict[str, ModelInfo],
                  available_features: Optional[int] = None,
                  missing_fraction: float = 0.0,
                  override: Optional[str] = None) -> tuple:
    """Pick the serving model. Returns (model_name, reason).

    Precedence: manual override > <3 features (Ridge) > >40% missing (RF) >
    lowest rolling 7-day RMSE.
    """
    avail = [n for n in MODEL_ORDER if registry.get(n) and registry[n].available]
    if not avail:
        return XGBOOST_LABEL, "no registry; defaulting to XGBoost"

    if override and override in registry and registry[override].available:
        return override, "manual override"

    # Rule: fewer than 3 usable features -> Ridge (linear baseline).
    if available_features is not None and available_features < MIN_FEATURES_RIDGE:
        if "Ridge" in avail:
            return "Ridge", f"only {available_features} features (<{MIN_FEATURES_RIDGE})"

    # Rule: >40% missing features -> RandomForest (stable on sparse input).
    if missing_fraction > MISSING_FEATURE_FALLBACK_THRESHOLD and "RandomForest" in avail:
        return "RandomForest", f"{missing_fraction*100:.0f}% features missing (>40%)"

    # Default: lowest rolling 7-day RMSE.
    scored = [(registry[n].rolling_rmse, n) for n in avail
              if registry[n].rolling_rmse is not None]
    if scored:
        rmse, best = min(scored)
        return best, f"lowest rolling {ROLLING_WINDOW_DAYS}-day RMSE ({rmse})"

    # No rolling scores (e.g. still training) -> XGBoost floor.
    if XGBOOST_LABEL in avail:
        return XGBOOST_LABEL, "no rolling scores yet; using XGBoost"
    return avail[0], "first available model"
