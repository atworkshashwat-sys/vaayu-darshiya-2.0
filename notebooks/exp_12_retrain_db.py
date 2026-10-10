"""
EXP_12 - retrain NO2 / O3 / HCHO on the project database
=========================================================

Retrains all three pollutants with the EXP_11 recipe (the only pipeline that
is validated end-to-end against data/vaayu_extract_7sites.csv, the same
database the dashboard displays) so the served forecasts finally agree with
the observations next to them.

Why this exists
---------------
The EXP_09v2 NO2/O3 models were trained on the retired `site_*_train_data.csv`
schema (NWP forecast inputs). Against the project database they score
R2 ~ 0 or below at every horizon - a naive persistence baseline describes the
dashboard's own observations better than the served forecast does. EXP_11
proved the fix: train on vaayu_final_v3.csv directly.

Data facts this script is built around (verified 2026-10-09):
  * 2018 / 2019 / 2025 blocks are real observations; the entire 2023 block
    carries is_synthetic=1 - its targets AND weather features are noised
    (e.g. temperature lag-1 autocorrelation drops 0.95 -> 0.79). It is useful
    training data but must never appear in validation or test metrics.
  * 2024 has no rows at all for the 7 served stations.

Protocol (all three targets, uniform)
-------------------------------------
  train : everything before 2025-01-01  (real 2018/2019 + synthetic 2023)
  val   : 2025-01-01 .. 2025-03-31      (REAL - used for model selection)
  test  : 2025-04-01 .. end             (REAL - every reported metric)

Horizon targets are built by joining on (site, datetime + h) rather than a
positional shift, so missing rows can never move a target in time.

Two feature recipes and three hyperparameter sets are screened per target on
the real validation window (t+1 / t+12 / t+24); the winner trains all 24
horizons. Artefacts use the exact EXP_09v2/EXP_11 names that
backend/api.py, scripts/evaluate_accuracy.py and the Model Details tab read:

  models/exp09v2_models/{target}_h{HH}.pkl (+ _importance.csv)
  models/scalers/exp09v2_feat_scaler.pkl, exp09v2_feature_cols.pkl   (NO2/O3)
  models/scalers/exp11_hcho_feat_scaler.pkl, exp11_hcho_feature_cols.pkl
  models/predictions/exp09v2_{target}_predictions.csv
  models/results/exp09v2_xgb_{target}_results.csv
  models/results/exp09v2_{target}_error_by_{site,season}.csv

Run:
    python notebooks/exp_12_retrain_db.py
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover - non-replaceable stdio
    pass

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_CSV = os.path.join(ROOT_DIR, "data", "vaayu_final_v3.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "models", "exp09v2_models")
SCALER_DIR = os.path.join(ROOT_DIR, "models", "scalers")
PRED_DIR = os.path.join(ROOT_DIR, "models", "predictions")
RESULT_DIR = os.path.join(ROOT_DIR, "models", "results")

# ── GPU DETECTION (same as EXP_09 / EXP_11) ──────────────────────────────────
try:
    _test = xgb.XGBRegressor(tree_method="hist", device="cuda", n_estimators=1)
    _test.fit([[1]], [1])
    TREE_METHOD, DEVICE = "hist", "cuda"
    print("GPU detected - using CUDA")
except Exception:
    TREE_METHOD, DEVICE = "hist", "cpu"
    print("No CUDA - using CPU")

# ── CONFIG ───────────────────────────────────────────────────────────────────
# target column in vaayu_final_v3.csv -> (artefact suffix, scale to serve unit)
TARGETS = {
    "NO2_target": ("no2", 1.0),                    # ppb
    "O3_target": ("o3", 1.0),                      # ppb
    "HCHO_target": ("sat_hcho_column", 1e6),       # mol/m² -> µmol/m²
}
FORECAST_LEN = 24
SCREEN_HORIZONS = [0, 11, 23]      # t+1 / t+12 / t+24 for config screening
EVAL_HORIZONS = [0, 5, 11, 23]
SEASONS = ["Winter", "Summer", "Monsoon", "Post-monsoon"]

# frontend site id -> station_id in vaayu_final_v3.csv (backend/database.py)
STATION_TO_SITE = {
    "8917": 1,   # Ashok Vihar
    "6931": 2,   # Dwarka
    "5754": 3,   # Jawaharlal Nehru
    "10485": 4,  # Northwest Delhi
    "8239": 5,   # East Delhi
    "10831": 6,  # Northwest Delhi 2
    "8475": 7,   # Northeast Delhi
}

# Chronological split - train on history, select and score on REAL 2025 data
# (the 2023 block is synthetic; see header).
TRAIN_END = pd.Timestamp("2025-01-01")   # train: everything before
VAL_END = pd.Timestamp("2025-04-01")     # val:   Jan-Mar 2025 (real)
# test:  2025-04-01 .. database end (real)

FFILL_LIMIT = 8          # feature-only gap bridge; observed gaps run <= 7h
LAG_HOURS = [1, 2, 3, 6, 12, 24, 48, 72]
FWD_HOURS = [1, 2, 3, 6, 12, 24]
ROLL_WINDOWS = [6, 12, 24, 48, 72, 168]

# Weather/driver columns carried as lagged + forward features.
WEATHER_CORE = [
    "temperature_2m_c", "relative_humidity",
    "wind_u_10m", "wind_v_10m", "rainfall_mm",
]
WEATHER_EXTRA = [
    "boundary_layer_height", "ventilation_coeff", "stagnation_index",
    "temp_inversion_proxy", "solar_radiation",
]
# Contemporaneous observations, known at issue time (EXP_09 precedent: its
# NO2_satellite / HCHO_satellite columns were used unlagged).
OBSERVED = [
    "pm25", "sat_no2_column", "sat_o3_column", "aod_550nm",
    "fire_pixel_count", "stubble_burning_flag", "lst_celsius",
    "is_daytime", "solar_zenith_angle", "is_holiday",
]
DRIVER_FLAGS = ["low_ventilation_flag", "blh_critical_flag", "inversion_flag"]
STATIC_SITE = ["nox_t_km2_yr", "night_light_radiance"]

HPARAM_SETS = {
    "A": dict(max_depth=5, min_child_weight=10, gamma=0.2, reg_lambda=2.0),
    "D": dict(max_depth=6, min_child_weight=20, gamma=0.1, reg_lambda=2.0),
}
# screening candidates: name -> (recipe, hparam set)
CANDIDATES = {
    "base_A": ("BASE", "A"),   # EXP_11's exact recipe (reproduces the baseline)
    "ext_A": ("EXT", "A"),     # extended features, EXP_09-style regularisation
    "ext_D": ("EXT", "D"),     # extended features, deeper + stronger min_child
}


def month_to_season(m: int) -> str:
    if m in (12, 1, 2):
        return "Winter"
    if m in (3, 4, 5):
        return "Summer"
    if m in (6, 7, 8, 9):
        return "Monsoon"
    return "Post-monsoon"


def score(y: np.ndarray, p: np.ndarray) -> dict:
    """Metrics on the finite overlap - mirrors scripts/evaluate_accuracy.py."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < 2:
        return {"r2": float("nan"), "mae": float("nan"), "rmse": float("nan"),
                "bias": float("nan"), "n": int(len(y))}
    return {
        "r2": float(r2_score(y, p)),
        "mae": float(np.mean(np.abs(y - p))),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "bias": float(np.mean(p - y)),
        "n": int(len(y)),
    }


def build_targets_temporal(frame: pd.DataFrame, tgt_col: str, n: int) -> np.ndarray:
    """Value of tgt_col exactly h hours after each row, per site (merge on
    time - never positional, so missing rows cannot shift a target)."""
    out = np.full((len(frame), n), np.nan)
    src = frame.drop_duplicates(["site_id", "datetime"])[["site_id", "datetime", tgt_col]]
    lookup = src.set_index(["site_id", "datetime"])[tgt_col]
    sites = frame["site_id"].to_numpy()
    dts = pd.DatetimeIndex(frame["datetime"])
    for h in range(1, n + 1):
        keys = pd.MultiIndex.from_arrays([sites, dts + pd.Timedelta(hours=h)])
        out[:, h - 1] = lookup.reindex(keys).to_numpy(float)
    return out


def base_params(hp_key: str) -> dict:
    hp = HPARAM_SETS[hp_key]
    return dict(
        n_estimators=2500,
        learning_rate=0.03,
        max_depth=hp["max_depth"],
        min_child_weight=hp["min_child_weight"],
        gamma=hp["gamma"],
        subsample=0.6,
        colsample_bytree=0.5,
        reg_alpha=0.5,
        reg_lambda=hp["reg_lambda"],
        random_state=42,
        n_jobs=-1,
        tree_method=TREE_METHOD,
        device=DEVICE,
        eval_metric="rmse",
        early_stopping_rounds=100,
    )


# ── 1. LOAD THE 7 SERVED STATIONS ────────────────────────────────────────────
print("\n[1] Loading data from vaayu_final_v3.csv (7 served stations) ...")
if not os.path.exists(SRC_CSV):
    sys.exit(f"missing database: {SRC_CSV}")

usecols = (["station_id", "datetime"] + [c for c, _ in TARGETS.values()]
           + WEATHER_CORE + WEATHER_EXTRA + OBSERVED + DRIVER_FLAGS + STATIC_SITE)
usecols = list(dict.fromkeys(usecols))          # de-duplicate, keep order
pieces = []
reader = pd.read_csv(
    SRC_CSV, usecols=usecols, chunksize=250_000,
    dtype={"station_id": str, "datetime": str}, low_memory=False,
)
for chunk in reader:
    sub = chunk[chunk["station_id"].isin(STATION_TO_SITE)]
    if not sub.empty:
        pieces.append(sub)
df = pd.concat(pieces, ignore_index=True)
print(f"    Rows: {len(df):,}")

df["site_id"] = df["station_id"].map(STATION_TO_SITE).astype(int)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values(["site_id", "datetime"]).reset_index(drop=True)
df["month"] = df["datetime"].dt.month
df["hour"] = df["datetime"].dt.hour

# ── 2. SEGMENTS (contiguous hourly runs per site) ────────────────────────────
print("\n[2] Splitting into contiguous hourly segments ...")
df["segment"] = (
    df.groupby("site_id")["datetime"]
      .transform(lambda s: (s.diff() != pd.Timedelta(hours=1)).cumsum())
)
n_segs = df["segment"].nunique()
_seg_sizes = df.groupby(["site_id", "segment"], sort=False).size()
print(f"    {n_segs:,} segments ({int((_seg_sizes >= 100).sum())} of >=100h) "
      f"across {df['site_id'].nunique()} sites")

# ── 3. TARGETS (serve unit) + feature-only gap bridge ────────────────────────
print("\n[3] Preparing targets ...")
for tgt, (col, scale) in TARGETS.items():
    df[tgt] = pd.to_numeric(df[col], errors="coerce") * scale
    df[f"{tgt}_feat"] = df.groupby(["site_id", "segment"], sort=False)[tgt] \
                          .transform(lambda s: s.ffill(limit=FFILL_LIMIT))
    raw = df[tgt].notna().mean()
    fill = df[f"{tgt}_feat"].notna().mean()
    print(f"    {tgt:<12} present {raw*100:5.1f}%  after bridge {fill*100:5.1f}%  "
          f"range {df[tgt].min():.2f} .. {df[tgt].max():.2f}")

# ── 4. FEATURE ENGINEERING ───────────────────────────────────────────────────
print("\n[4] Engineering features ...")

# Cyclic time - hour, month, day of week, day of year
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype(np.float32)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype(np.float32)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype(np.float32)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype(np.float32)
df["dayofweek"] = df["datetime"].dt.dayofweek
df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7).astype(np.float32)
df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7).astype(np.float32)
df["day_of_year"] = df["datetime"].dt.dayofyear
df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25).astype(np.float32)
df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25).astype(np.float32)

# Wind speed + direction (from the 10m components)
df["wind_speed"] = np.sqrt(df["wind_u_10m"] ** 2 + df["wind_v_10m"] ** 2)
df["wind_dir"] = np.arctan2(df["wind_v_10m"], df["wind_u_10m"])
df["wind_dir_sin"] = np.sin(df["wind_dir"]).astype(np.float32)
df["wind_dir_cos"] = np.cos(df["wind_dir"]).astype(np.float32)

df["season"] = df["month"].apply(month_to_season)

# Site one-hot (site is known at issue time; lets trees split site baselines)
site_dummies = pd.get_dummies(df["site_id"], prefix="site", dtype=np.float32)
for c in site_dummies.columns:
    df[c] = site_dummies[c]

seg_gb = df.groupby(["site_id", "segment"], sort=False)


def build_weather_lags(cols):
    lag_cols = []
    for col in cols:
        for lag in LAG_HOURS:
            cname = f"{col}_lag{lag}"
            df[cname] = seg_gb[col].shift(lag)
            lag_cols.append(cname)
        for fwd in FWD_HOURS:
            cname = f"{col}_fwd{fwd}"
            df[cname] = seg_gb[col].shift(-fwd)
            lag_cols.append(cname)
    return lag_cols


weather_lag_core = build_weather_lags(WEATHER_CORE)          # both recipes
weather_lag_ext = weather_lag_core + build_weather_lags(WEATHER_EXTRA)

# Target lags (gap-bridged copy) + rolling moments (window excludes present)
target_lag_cols, target_roll_cols, target_std_cols = [], [], []
for tgt in TARGETS:
    for lag in LAG_HOURS:
        cname = f"{tgt}_feat_lag{lag}"
        df[cname] = seg_gb[f"{tgt}_feat"].shift(lag)
        target_lag_cols.append(cname)
    for window in ROLL_WINDOWS:
        cname = f"{tgt}_feat_roll{window}"
        df[cname] = seg_gb[f"{tgt}_feat"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        target_roll_cols.append(cname)
    cname = f"{tgt}_feat_roll_std24"
    df[cname] = seg_gb[f"{tgt}_feat"].transform(
        lambda x: x.shift(1).rolling(24, min_periods=2).std()
    )
    target_std_cols.append(cname)

BASE_TIME = ["hour_sin", "hour_cos", "month_sin", "month_cos",
             "dow_sin", "dow_cos", "doy_sin", "doy_cos",
             "wind_speed", "wind_dir_sin", "wind_dir_cos"]
SITE_DUMMIES = list(site_dummies.columns)

print(f"    weather lags core: {len(weather_lag_core)}  "
      f"ext: {len(weather_lag_ext)}  target lags: {len(target_lag_cols)}  "
      f"rolls: {len(target_roll_cols)}")


def feature_list(recipe: str, tgt: str) -> list:
    """Model feature columns for one target under one recipe."""
    tgt_lags = [c for c in target_lag_cols if c.startswith(tgt)]
    tgt_rolls = [c for c in target_roll_cols if c.startswith(tgt)]
    if recipe == "BASE":
        # EXP_11's exact recipe (100 features)
        return (WEATHER_CORE + BASE_TIME + weather_lag_core
                + tgt_lags + tgt_rolls)
    # EXT: extended drivers + contemporaneous observations + y(t) itself
    tgt_std = [c for c in target_std_cols if c.startswith(tgt)]
    return (WEATHER_CORE + WEATHER_EXTRA + BASE_TIME + OBSERVED
            + DRIVER_FLAGS + STATIC_SITE + SITE_DUMMIES
            + weather_lag_ext
            + [f"{tgt}_feat"]                                # y(t) at issue time
            + tgt_lags + tgt_rolls + tgt_std)


def needed_cols(tgt: str) -> list:
    """Every column that must be non-null for a row to be trainable -
    the EXT superset, so both recipes see identical rows (fair screening)."""
    return sorted(set(WEATHER_CORE + WEATHER_EXTRA + OBSERVED
                      + DRIVER_FLAGS + STATIC_SITE
                      + [f"{tgt}_feat"]
                      + [c for c in target_lag_cols if c.startswith(tgt)]
                      + [c for c in target_roll_cols if c.startswith(tgt)]
                      + [c for c in target_std_cols if c.startswith(tgt)]))


os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)

SUMMARY = []           # rows for the final report table

# ── 5-8. PER TARGET: split, screen, train, export ───────────────────────────
for tgt, (src_col, scale) in TARGETS.items():
    print("\n" + "=" * 76)
    print(f"  {tgt}")
    print("=" * 76)

    # feature-complete rows for this target (real + synthetic history alike)
    mask = df[needed_cols(tgt)].notna().all(axis=1)
    tdf = df[mask].reset_index(drop=True)
    print(f"\n[{tgt}] feature-complete rows: {len(tdf):,}")

    train_df = tdf[tdf["datetime"] < TRAIN_END].copy().reset_index(drop=True)
    val_df = tdf[(tdf["datetime"] >= TRAIN_END)
                 & (tdf["datetime"] < VAL_END)].copy().reset_index(drop=True)
    test_df = tdf[tdf["datetime"] >= VAL_END].copy().reset_index(drop=True)
    for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
        seasons = sorted(part["season"].unique()) if len(part) else []
        print(f"    {name:<6}: {len(part):>7,} rows  "
              f"{str(part['datetime'].min())[:10]} -> {str(part['datetime'].max())[:10]}  "
              f"seasons={seasons}")

    # horizon targets - built by timestamp, never positional
    Y = {name: build_targets_temporal(part, tgt, FORECAST_LEN)
         for name, part in (("train", train_df), ("val", val_df),
                            ("test", test_df))}
    for name in ("train", "val", "test"):
        ok = np.isfinite(Y[name])
        print(f"    {name:<6} truth valid per horizon: t+1h={ok[:,0].mean()*100:.0f}%  "
              f"t+24h={ok[:,23].mean()*100:.0f}%")

    test_meta = test_df[["site_id", "datetime", "season"]].reset_index(drop=True)

    # ── screening on the REAL validation window ─────────────────────────────
    print(f"\n[{tgt}] screening candidates on the REAL validation window ...")
    screen = {}
    for cand, (recipe, hp_key) in CANDIDATES.items():
        cols = feature_list(recipe, tgt)
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(train_df[cols])
        x_va = scaler.transform(val_df[cols])
        rs = {}
        for h in SCREEN_HORIZONS:
            y_tr, y_va = Y["train"][:, h], Y["val"][:, h]
            tr_ok, va_ok = np.isfinite(y_tr), np.isfinite(y_va)
            model = xgb.XGBRegressor(**base_params(hp_key))
            model.fit(x_tr[tr_ok], y_tr[tr_ok],
                      eval_set=[(x_va[va_ok], y_va[va_ok])], verbose=False)
            vp = np.clip(model.predict(x_va[va_ok]), 0, None)
            rs[h] = float(r2_score(y_va[va_ok], vp))
        mean_r2 = float(np.mean(list(rs.values())))
        screen[cand] = {"mean": mean_r2, "recipe": recipe, "hp": hp_key,
                        "n_feat": len(cols)}
        print(f"    {cand:<8} feats={len(cols):>3}  "
              + "  ".join(f"t+{h+1:02d} valR2={rs[h]:.4f}" for h in SCREEN_HORIZONS)
              + f"  | mean={mean_r2:.4f}")
    winner = max(screen, key=lambda c: screen[c]["mean"])
    recipe, hp_key = screen[winner]["recipe"], screen[winner]["hp"]
    print(f"    -> winner: {winner} ({recipe} + hp {hp_key})")

    # ── full training (24 horizons) ────────────────────────────────────────
    cols = feature_list(recipe, tgt)
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(train_df[cols])
    x_va = scaler.transform(val_df[cols])
    x_te = scaler.transform(test_df[cols])

    # scaler artefact - EXP_11's naming convention: the exp09v2_* pair belongs
    # to the NO2/O3 pipeline, exp11_hcho_* to HCHO.
    if tgt == "HCHO_target":
        joblib.dump(scaler, os.path.join(SCALER_DIR, "exp11_hcho_feat_scaler.pkl"))
        joblib.dump(cols, os.path.join(SCALER_DIR, "exp11_hcho_feature_cols.pkl"))
    else:
        joblib.dump(scaler, os.path.join(SCALER_DIR, "exp09v2_feat_scaler.pkl"))
        joblib.dump(cols, os.path.join(SCALER_DIR, "exp09v2_feature_cols.pkl"))

    all_preds = np.zeros((x_te.shape[0], FORECAST_LEN))
    results = {}
    print(f"\n[{tgt}] training {FORECAST_LEN} horizon models "
          f"[{recipe}+hp{hp_key}, {len(cols)} features] ...")
    t0 = time.time()

    for h in range(FORECAST_LEN):
        y_tr, y_va = Y["train"][:, h], Y["val"][:, h]
        tr_ok, va_ok = np.isfinite(y_tr), np.isfinite(y_va)
        model = xgb.XGBRegressor(**base_params(hp_key))
        model.fit(x_tr[tr_ok], y_tr[tr_ok],
                  eval_set=[(x_va[va_ok], y_va[va_ok])], verbose=False)

        all_preds[:, h] = np.clip(model.predict(x_te), 0, None)
        vp = np.clip(model.predict(x_va[va_ok]), 0, None)
        best_iter = model.best_iteration
        best_iter = int(best_iter) if best_iter is not None else -1

        joblib.dump(model, os.path.join(MODEL_DIR, f"{tgt}_h{h:02d}.pkl"))
        importance = pd.DataFrame({
            "feature": cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        importance.to_csv(
            os.path.join(MODEL_DIR, f"{tgt}_h{h:02d}_importance.csv"), index=False
        )

        s_te = score(Y["test"][:, h], all_preds[:, h])
        results[h] = {
            "mae": s_te["mae"], "rmse": s_te["rmse"], "r2": s_te["r2"],
            "val_mae": mean_absolute_error(y_va[va_ok], vp),
            "val_rmse": float(np.sqrt(mean_squared_error(y_va[va_ok], vp))),
            "val_r2": r2_score(y_va[va_ok], vp),
            "best_iter": best_iter,
        }
        if h in EVAL_HORIZONS:
            print(f"    t+{h+1:02d} done (best_iter={best_iter}, "
                  f"val_r2={results[h]['val_r2']:.4f}, test_r2={results[h]['r2']:.4f})",
                  flush=True)

    print(f"    trained in {(time.time()-t0)/60:.1f} min")

    # results CSV (schema identical to EXP_09/EXP_11 - backend/api.py reads it)
    results_df = pd.DataFrame(results).T
    results_df.index.name = "horizon"
    results_path = os.path.join(RESULT_DIR, f"exp09v2_xgb_{tgt}_results.csv")
    results_df.to_csv(results_path)
    print(f"    Saved -> {os.path.relpath(results_path, ROOT_DIR)}")

    # error by site (evaluate_accuracy.py schema)
    site_rows = []
    for site in sorted(test_meta["site_id"].unique()):
        site_mask = (test_meta["site_id"] == site).to_numpy()
        for h in range(FORECAST_LEN):
            m = site_mask & np.isfinite(Y["test"][:, h])
            if m.sum() < 2:
                continue
            s = score(Y["test"][m, h], all_preds[m, h])
            site_rows.append({"site_id": int(site), "horizon": h,
                              "r2": round(s["r2"], 4), "mae": round(s["mae"], 2),
                              "rmse": round(s["rmse"], 2),
                              "bias": round(s["bias"], 2), "n": s["n"]})
    pd.DataFrame(site_rows)[
        ["site_id", "horizon", "r2", "mae", "rmse", "bias", "n"]
    ].to_csv(os.path.join(RESULT_DIR, f"exp09v2_{tgt}_error_by_site.csv"), index=False)

    # error by season (evaluate_accuracy.py schema)
    season_rows = []
    for season in SEASONS:
        season_mask = (test_meta["season"] == season).to_numpy()
        if season_mask.sum() == 0:
            continue
        for h in range(FORECAST_LEN):
            m = season_mask & np.isfinite(Y["test"][:, h])
            if m.sum() < 2:
                continue
            s = score(Y["test"][m, h], all_preds[m, h])
            season_rows.append({"season": season, "horizon": h,
                                "r2": round(s["r2"], 4), "mae": round(s["mae"], 2),
                                "rmse": round(s["rmse"], 2),
                                "bias": round(s["bias"], 2), "n": s["n"]})
    order = {s: i for i, s in enumerate(SEASONS)}
    season_rows.sort(key=lambda r: (order.get(r["season"], 9), r["horizon"]))
    pd.DataFrame(season_rows)[
        ["season", "horizon", "r2", "mae", "rmse", "bias", "n"]
    ].to_csv(os.path.join(RESULT_DIR, f"exp09v2_{tgt}_error_by_season.csv"), index=False)

    # prediction CSV (site_id, datetime, horizon 1..24, prediction, truth)
    rows = []
    for h in range(FORECAST_LEN):
        rows.append(pd.DataFrame({
            "site_id": test_meta["site_id"].to_numpy(),
            "datetime": test_meta["datetime"].to_numpy(),
            "horizon": h + 1,
            "prediction": all_preds[:, h],
            "truth": Y["test"][:, h],
        }))
    pred_df = pd.concat(rows, ignore_index=True)
    pred_path = os.path.join(PRED_DIR, f"exp09v2_{tgt}_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"    Saved -> {os.path.relpath(pred_path, ROOT_DIR)}  "
          f"({len(pred_df):,} rows, {pred_df['datetime'].nunique():,} anchors, "
          f"{str(pred_df['datetime'].min())[:10]} .. {str(pred_df['datetime'].max())[:10]})")

    # ── naive baselines on the SAME real test window (the yardstick) ───────
    print(f"\n[{tgt}] naive baselines on the SAME test window (real 2025 data):")
    persist = test_df[tgt].to_numpy(float)      # y(t) held constant to t+h
    for h in (0, 5, 11, 23):
        s = score(Y["test"][:, h], persist)
        print(f"    persistence t+{h+1:02d}: R2={s['r2']:>8.4f}  MAE={s['mae']:>7.2f}  n={s['n']:,}")
    clim = train_df.groupby(["site_id", "hour"])[tgt].mean()
    clim_pred = test_df.set_index(["site_id", "hour"]).index.map(clim).to_numpy(float)
    s = score(Y["test"][:, 23], clim_pred)
    print(f"    climatology t+24 (site x hour, fit on train): "
          f"R2={s['r2']:>8.4f}  MAE={s['mae']:>7.2f}")

    SUMMARY.append((tgt, recipe, hp_key, len(cols), results))

    # free per-target frames before the next target
    del train_df, val_df, test_df, tdf, x_tr, x_va, x_te, all_preds

# ── 9. FINAL SUMMARY ─────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("  EXP_12 - XGBoost retrain on the project database")
print("  Train < 2025-01-01   Val Jan-Mar 2025 (real)   Test Apr 2025+ (real)")
print(f"  Device: {DEVICE}")
print("=" * 78)
print(f"  {'Target':<12} {'Recipe':<6} {'Hp':<3} {'Feats':>5}  {'Horizon':<8} "
      f"{'MAE':>8} {'RMSE':>8} {'R2':>8} {'ValR2':>8}")
print("  " + "-" * 74)
for tgt, recipe, hp_key, n_feat, results in SUMMARY:
    for h in EVAL_HORIZONS:
        r = results[h]
        print(f"  {tgt:<12} {recipe:<6} {hp_key:<3} {n_feat:>5}  t+{h+1:02d}h     "
              f"{r['mae']:>8.3f} {r['rmse']:>8.3f} {r['r2']:>8.4f} "
              f"{r['val_r2']:>8.4f}")
print("=" * 78)
print("\nDone - EXP_12 complete.")
