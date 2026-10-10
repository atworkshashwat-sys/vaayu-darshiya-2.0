"""
VAAYU DARSHIYA — EXP_11
XGBoost HCHO (formaldehyde) 24-hour forecast — same pipeline as EXP_09 v2
==========================================================================

Mirrors notebooks/EXP_09_FINAL_XGBOOST.py end to end: same hyperparameters
(HPARAM_SETS "A"), same feature recipe (cyclic time + wind + lags + forward
shifts + rolling means), 24 per-horizon XGBoost models with early stopping on
a chronological validation split, and the same artefact set the backend reads
(models, importance CSVs, prediction CSV, results CSV, season/site breakdowns).

Data-driven adaptations (the retired site_*_train_data.csv files no longer
exist, so EXP_09's exact inputs cannot be reproduced):

  [DATA]  Source     : data/vaayu_final_v3.csv, the 7 served stations
                       (station_id -> site_id via backend/database.py).
  [DATA]  NWP fields : the `*_forecast` columns EXP_09 used do not exist in
                       the project database. The five weather fields that
                       database.py translates them onto are used instead:
                         T_forecast -> temperature_2m_c
                         q_forecast -> relative_humidity
                         u_forecast -> wind_u_10m
                         v_forecast -> wind_v_10m
                         w_forecast -> rainfall_mm
                       EXP_09's NO2_forecast / O3_forecast have no counterpart
                       and are omitted (68 of EXP_09's 144 features are
                       target-independent of that; 100 features here).
  [DATA]  Target     : sat_hcho_column — CPCB does not measure formaldehyde,
                       so (exactly like the satellite columns EXP_09 dropped
                       from its inputs) the satellite vertical column is the
                       only HCHO series available. It is in mol/m²; everything
                       below trains, scores and serves it in µmol/m² (×1e6,
                       range ~3–90) so metrics and the API stay readable.
                       Satellites NO2/O3 columns stay out of the inputs,
                       mirroring EXP_09's DROP_BEFORE_LAG.
  [DATA]  Continuity : 2018/2019/2025 exist as many short hourly fragments, so
                       every lag / roll / forward / horizon shift is grouped by
                       (site, segment) where EXP_09 grouped by site only (its
                       data was contiguous). A shift never reaches across a
                       data gap, exactly as shift() would not have on
                       contiguous input.
  [DATA]  Gaps       : sat_hcho_column is ~87% present, in short runs (≤7h).
                       Features use a segment-wise ffill(limit=8) copy of the
                       target so the 8 target lags survive; truth, metrics and
                       the prediction CSV always use the raw series.
  [EVAL]  Split      : EXP_09's 2019-21/2022/2023-24 does not exist for HCHO.
                       Chronological instead:
                         train < 2023-08-01   (2018, 2019, 2023-H1)
                         val    Aug–Sep 2023
                         test   Oct–Dec 2023 + 2025   (all four seasons)
  [EVAL]  NaN filter : EXP_09 kept rows where all 24 horizon targets were
                       present; only ~7% of HCHO rows qualify, so each horizon
                       model trains and is scored on its own valid rows
                       (per-horizon mask over the same X).
  [OUT]   Artefacts  : written straight to models/{exp09v2_models,predictions,
                       results} with the exp09v2_HCHO_target_* names
                       backend/api.py expects. Scaler/feature list use exp11_*
                       names so the NO2/O3 pipeline artefacts are untouched.

Run:
    python notebooks/exp_11_hcho_xgboost.py
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

# ── GPU DETECTION (same as EXP_09) ────────────────────────────────────────────
try:
    _test = xgb.XGBRegressor(tree_method="hist", device="cuda", n_estimators=1)
    _test.fit([[1]], [1])
    TREE_METHOD = "hist"
    DEVICE = "cuda"
    print("GPU detected — using CUDA")
except Exception:
    TREE_METHOD = "hist"
    DEVICE = "cpu"
    print("No CUDA — using CPU")

# ── CONFIG ────────────────────────────────────────────────────────────────────
TARGETS = ["HCHO_target"]
FORECAST_LEN = 24
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

# EXP_09 NWP_FEATURES, translated onto the project database's weather columns
# (see header). EXP_09's NO2_forecast / O3_forecast have no counterpart here.
WEATHER_FEATURES = [
    "temperature_2m_c", "relative_humidity",
    "wind_u_10m", "wind_v_10m", "rainfall_mm",
]
LAG_HOURS = [1, 2, 3, 6, 12, 24, 48, 72]
FWD_HOURS = [1, 2, 3, 6, 12, 24]
ROLL_WINDOWS = [6, 12, 24, 48, 72, 168]

# Chronological split (EXP_09's year split does not exist for HCHO).
TRAIN_END = pd.Timestamp("2023-08-01")   # train: everything before
VAL_END = pd.Timestamp("2023-10-01")    # val:   2023-08-01 .. 2023-09-30
# test:  2023-10-01 .. 2023-12-31  and  2025

TARGET_SCALE = 1e6   # mol/m² -> µmol/m² (train/serve unit)
FFILL_LIMIT = 8      # feature-only gap bridge; observed gap runs never exceed 7h

# Swappable hyperparameter sets — ACTIVE_HPARAM_SET "A" = EXP_09's final set.
ACTIVE_HPARAM_SET = "A"
HPARAM_SETS = {
    "A": dict(max_depth=5, min_child_weight=10, gamma=0.2, reg_lambda=2.0),
    "B": dict(max_depth=4, min_child_weight=12, gamma=0.3, reg_lambda=3.0),
    "C": dict(max_depth=5, min_child_weight=15, gamma=0.2, reg_lambda=2.0),
}


def month_to_season(m: int) -> str:
    """EXP_09's season labels (used for the error-by-season breakdown only)."""
    if m in (12, 1, 2):
        return "Winter"
    if m in (3, 4, 5):
        return "Summer"
    if m in (6, 7, 8, 9):
        return "Monsoon"
    return "Post-monsoon"


def score(y: np.ndarray, p: np.ndarray) -> dict:
    """Metrics on the finite overlap — mirrors scripts/evaluate_accuracy.py."""
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


# ── 1. LOAD THE 7 SERVED STATIONS ─────────────────────────────────────────────
print("\n[1] Loading data from vaayu_final_v3.csv (7 served stations) ...")
if not os.path.exists(SRC_CSV):
    sys.exit(f"missing database: {SRC_CSV}")

usecols = ["station_id", "datetime", "sat_hcho_column"] + WEATHER_FEATURES
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
df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df["hour"] = df["datetime"].dt.hour

# ── 2. SEGMENTS (contiguous hourly runs per site) ─────────────────────────────
# EXP_09 grouped shifts by site only because its input was one contiguous
# series per site. Here 2018/2019/2025 are fragmented; a segment starts
# wherever the hourly cadence breaks, so no shift ever reaches across a gap.
print("\n[2] Splitting into contiguous hourly segments ...")
df["segment"] = (
    df.groupby("site_id")["datetime"]
      .transform(lambda s: (s.diff() != pd.Timedelta(hours=1)).cumsum())
)
n_segs = df["segment"].nunique()
_seg_sizes = df.groupby(["site_id", "segment"], sort=False).size()
print(f"    {n_segs:,} segments ({int((_seg_sizes >= 100).sum())} of >=100h) "
      f"across {df['site_id'].nunique()} sites")

# ── 3. TARGET (µmol/m²) + feature-only gap bridge ─────────────────────────────
print("\n[3] Preparing target ...")
df["HCHO_target"] = pd.to_numeric(df["sat_hcho_column"], errors="coerce") * TARGET_SCALE
# Feature copy: every observed gap (<=7h) bridged so the target lags survive.
# Truth / metrics / prediction CSV below always use the raw HCHO_target.
df["HCHO_target_feat"] = df.groupby(
    ["site_id", "segment"], sort=False
)["HCHO_target"].transform(lambda s: s.ffill(limit=FFILL_LIMIT))
_raw_nn = df["HCHO_target"].notna().mean()
_fill_nn = df["HCHO_target_feat"].notna().mean()
print(f"    raw present: {_raw_nn*100:.1f}%  after feature bridge: {_fill_nn*100:.1f}%")
print(f"    range: {df['HCHO_target'].min():.2f} .. "
      f"{df['HCHO_target'].max():.2f} µmol/m²  "
      f"mean={df['HCHO_target'].mean():.2f}")

# ── 4. FEATURE ENGINEERING (EXP_09's recipe) ──────────────────────────────────
print("\n[4] Engineering features ...")

# Cyclic time — hour, month
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype(np.float32)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype(np.float32)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype(np.float32)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype(np.float32)

# Cyclic time — day of week + day of year
df["dayofweek"] = df["datetime"].dt.dayofweek
df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7).astype(np.float32)
df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7).astype(np.float32)
df["day_of_year"] = df["datetime"].dt.dayofyear
df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25).astype(np.float32)
df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25).astype(np.float32)

# Wind speed + direction
df["wind_speed"] = np.sqrt(df["wind_u_10m"] ** 2 + df["wind_v_10m"] ** 2)
df["wind_dir"] = np.arctan2(df["wind_v_10m"], df["wind_u_10m"])
df["wind_dir_sin"] = np.sin(df["wind_dir"]).astype(np.float32)
df["wind_dir_cos"] = np.cos(df["wind_dir"]).astype(np.float32)

# Season label for the error breakdown (not a model feature, as in EXP_09)
df["season"] = df["month"].apply(month_to_season)

# Lag features — target (gap-bridged copy) + weather, segment-guarded
seg_gb = df.groupby(["site_id", "segment"], sort=False)
lag_cols = []
for col in ["HCHO_target_feat"] + WEATHER_FEATURES:
    for lag in LAG_HOURS:
        cname = f"{col}_lag{lag}"
        df[cname] = seg_gb[col].shift(lag)
        lag_cols.append(cname)

# Forward weather features (known in advance, as EXP_09's NWP fwd shifts)
for col in WEATHER_FEATURES:
    for fwd in FWD_HOURS:
        cname = f"{col}_fwd{fwd}"
        df[cname] = seg_gb[col].shift(-fwd)
        lag_cols.append(cname)

# Rolling features — target, shifted by 1h so the window excludes the present
roll_cols = []
for window in ROLL_WINDOWS:
    cname = f"HCHO_target_feat_roll{window}"
    df[cname] = seg_gb["HCHO_target_feat"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    roll_cols.append(cname)

# Drop rows without a full feature history (same rule as EXP_09: every lag,
# forward and rolling column must be present).
df = df.dropna(subset=lag_cols + roll_cols).reset_index(drop=True)
print(f"    After dropna: {len(df):,} rows")

FEATURE_COLS = (
    WEATHER_FEATURES
    + ["hour_sin", "hour_cos", "month_sin", "month_cos",
       "dow_sin", "dow_cos", "doy_sin", "doy_cos",
       "wind_speed", "wind_dir_sin", "wind_dir_cos"]
    + lag_cols
    + roll_cols
)
print(f"    Total features: {len(FEATURE_COLS)}")

# ── 5. SPLIT (chronological — see header) ─────────────────────────────────────
print("\n[5] Splitting train / val / test ...")
train_df = df[df["datetime"] < TRAIN_END].copy().reset_index(drop=True)
val_df = df[(df["datetime"] >= TRAIN_END) & (df["datetime"] < VAL_END)] \
    .copy().reset_index(drop=True)
test_df = df[df["datetime"] >= VAL_END].copy().reset_index(drop=True)
for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
    seasons = sorted(part["season"].unique()) if len(part) else []
    print(f"    {name:<6}: {len(part):>7,} rows  "
          f"{str(part['datetime'].min())[:10]} -> {str(part['datetime'].max())[:10]}  "
          f"seasons={seasons}")

# ── 6. SCALE (fit on train only, like EXP_09) ─────────────────────────────────
print("\n[6] Scaling ...")
feat_scaler = StandardScaler()
X_train = feat_scaler.fit_transform(train_df[FEATURE_COLS])
X_val = feat_scaler.transform(val_df[FEATURE_COLS])
X_test = feat_scaler.transform(test_df[FEATURE_COLS])
# exp11_* names: the exp09v2_* scaler/feature list belongs to the NO2/O3
# pipeline and must not be overwritten.
joblib.dump(feat_scaler, os.path.join(SCALER_DIR, "exp11_hcho_feat_scaler.pkl"))
joblib.dump(FEATURE_COLS, os.path.join(SCALER_DIR, "exp11_hcho_feature_cols.pkl"))

# ── 7. BUILD HORIZON TARGETS (per split, segment-guarded, raw µmol/m²) ────────
print("\n[7] Building forecast targets ...")


def build_horizon_targets(frame: pd.DataFrame, target_col: str, n: int) -> np.ndarray:
    gb = frame.groupby(["site_id", "segment"], sort=False)
    return np.stack([gb[target_col].shift(-h).values for h in range(1, n + 1)],
                    axis=1)


Y_train = build_horizon_targets(train_df, "HCHO_target", FORECAST_LEN)
Y_val = build_horizon_targets(val_df, "HCHO_target", FORECAST_LEN)
Y_test = build_horizon_targets(test_df, "HCHO_target", FORECAST_LEN)

# EXP_09 filtered rows valid for ALL 24 horizons at once; only ~7% of HCHO
# rows qualify, so validity is kept per horizon and applied at fit/score time.
for name, Y in (("train", Y_train), ("val", Y_val), ("test", Y_test)):
    ok = np.isfinite(Y)
    print(f"    {name:<6}: rows={Y.shape[0]:,}  "
          f"valid per horizon: t+1h={ok[:, 0].mean()*100:.0f}%  "
          f"t+6h={ok[:, 5].mean()*100:.0f}%  "
          f"t+12h={ok[:, 11].mean()*100:.0f}%  "
          f"t+24h={ok[:, 23].mean()*100:.0f}%")

# site_id / datetime / season aligned with the X rows for breakdowns + CSV
train_meta = train_df[["site_id", "datetime", "season"]].reset_index(drop=True)
val_meta = val_df[["site_id", "datetime", "season"]].reset_index(drop=True)
test_meta = test_df[["site_id", "datetime", "season"]].reset_index(drop=True)

# ── 8. TRAIN XGBOOST PER HORIZON ──────────────────────────────────────────────
print("\n[8] Training XGBoost models ...")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)

hp = HPARAM_SETS[ACTIVE_HPARAM_SET]
base_params = dict(
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

results = {}
all_preds_by_target = {}
t_start = time.time()

for tgt in TARGETS:
    print(f"\n    {tgt} — training {FORECAST_LEN} horizon models ...")
    all_preds = np.zeros((X_test.shape[0], FORECAST_LEN))
    results[tgt] = {}

    for h in range(FORECAST_LEN):
        y_tr = Y_train[:, h]
        y_va = Y_val[:, h]
        tr_ok = np.isfinite(y_tr)
        va_ok = np.isfinite(y_va)
        if tr_ok.sum() == 0 or va_ok.sum() == 0:
            raise RuntimeError(
                f"horizon {h}: empty mask (train={tr_ok.sum()}, val={va_ok.sum()})"
            )

        xgb_h = xgb.XGBRegressor(**base_params)
        xgb_h.fit(
            X_train[tr_ok], y_tr[tr_ok],
            eval_set=[(X_val[va_ok], y_va[va_ok])],
            verbose=False,
        )

        # Test predictions for every feature-complete row (an anchor always
        # carries all 24 horizons so backend/api.py sees complete runs);
        # scores below only use rows whose truth exists.
        all_preds[:, h] = np.clip(xgb_h.predict(X_test), 0, None)

        # Validation metrics + best iteration (as EXP_09 [P1])
        val_pred = np.clip(xgb_h.predict(X_val[va_ok]), 0, None)
        best_iter = xgb_h.best_iteration
        best_iter = int(best_iter) if best_iter is not None else -1

        joblib.dump(xgb_h, os.path.join(MODEL_DIR, f"{tgt}_h{h:02d}.pkl"))
        importance = pd.DataFrame({
            "feature": FEATURE_COLS,
            "importance": xgb_h.feature_importances_,
        }).sort_values("importance", ascending=False)
        importance.to_csv(
            os.path.join(MODEL_DIR, f"{tgt}_h{h:02d}_importance.csv"), index=False
        )

        s_te = score(Y_test[:, h], all_preds[:, h])
        results[tgt][h] = {
            "mae": s_te["mae"], "rmse": s_te["rmse"], "r2": s_te["r2"],
            "val_mae": mean_absolute_error(y_va[va_ok], val_pred),
            "val_rmse": float(np.sqrt(mean_squared_error(y_va[va_ok], val_pred))),
            "val_r2": r2_score(y_va[va_ok], val_pred),
            "best_iter": best_iter,
        }

        if h in EVAL_HORIZONS:
            print(f"      t+{h+1:02d} done (best_iter={best_iter}, "
                  f"val_r2={results[tgt][h]['val_r2']:.4f}, "
                  f"test_r2={results[tgt][h]['r2']:.4f})", flush=True)

    all_preds_by_target[tgt] = all_preds

    # Full metrics CSV (the file backend/api.py reads)
    results_df = pd.DataFrame(results[tgt]).T
    results_df.index.name = "horizon"
    results_path = os.path.join(RESULT_DIR, f"exp09v2_xgb_{tgt}_results.csv")
    results_df.to_csv(results_path)

    print(f"\n    {'Horizon':<10}  {'MAE':>8}  {'RMSE':>8}  {'R²':>8}  "
          f"{'ValR²':>8}  {'BestIter':>9}")
    print("    " + "-" * 65)
    for h in EVAL_HORIZONS:
        r = results[tgt][h]
        print(f"    t+{h+1:02d}h       {r['mae']:>8.3f}  {r['rmse']:>8.3f}  "
              f"{r['r2']:>8.4f}  {r['val_r2']:>8.4f}  {r['best_iter']:>9d}")

    print(f"    Saved → {os.path.relpath(results_path, ROOT_DIR)}")

    # ── Error by site (all horizons, evaluate_accuracy.py schema) ────────────
    site_rows = []
    for site in sorted(test_meta["site_id"].unique()):
        site_mask = (test_meta["site_id"] == site).to_numpy()
        for h in range(FORECAST_LEN):
            m = site_mask & np.isfinite(Y_test[:, h])
            if m.sum() < 2:
                continue
            s = score(Y_test[m, h], all_preds[m, h])
            site_rows.append({"site_id": int(site), "horizon": h,
                              "r2": round(s["r2"], 4), "mae": round(s["mae"], 2),
                              "rmse": round(s["rmse"], 2),
                              "bias": round(s["bias"], 2), "n": s["n"]})
    pd.DataFrame(site_rows)[
        ["site_id", "horizon", "r2", "mae", "rmse", "bias", "n"]
    ].to_csv(os.path.join(RESULT_DIR, f"exp09v2_{tgt}_error_by_site.csv"), index=False)
    print(f"    Saved → exp09v2_{tgt}_error_by_site.csv ({len(site_rows)} rows)")

    # ── Error by season (all horizons, evaluate_accuracy.py schema) ──────────
    season_rows = []
    for season in SEASONS:
        season_mask = (test_meta["season"] == season).to_numpy()
        if season_mask.sum() == 0:
            continue
        for h in range(FORECAST_LEN):
            m = season_mask & np.isfinite(Y_test[:, h])
            if m.sum() < 2:
                continue
            s = score(Y_test[m, h], all_preds[m, h])
            season_rows.append({"season": season, "horizon": h,
                                "r2": round(s["r2"], 4), "mae": round(s["mae"], 2),
                                "rmse": round(s["rmse"], 2),
                                "bias": round(s["bias"], 2), "n": s["n"]})
    order = {s: i for i, s in enumerate(SEASONS)}
    season_rows.sort(key=lambda r: (order.get(r["season"], 9), r["horizon"]))
    pd.DataFrame(season_rows)[
        ["season", "horizon", "r2", "mae", "rmse", "bias", "n"]
    ].to_csv(os.path.join(RESULT_DIR, f"exp09v2_{tgt}_error_by_season.csv"), index=False)
    print(f"    Saved → exp09v2_{tgt}_error_by_season.csv ({len(season_rows)} rows)")

elapsed = time.time() - t_start
print(f"\n    Training + export took {elapsed/60:.1f} min")

# ── 9. PREDICTION CSV (site_id, datetime, horizon 1..24, prediction, truth) ───
# Same schema as exp09v2_{NO2,O3}_target_predictions.csv. `truth` is empty
# where the raw target has a gap; predictions exist for every anchor so
# backend/api.py always finds runs with all 24 horizons.
print("\n[9] Saving prediction CSVs ...")
for tgt in TARGETS:
    all_preds = all_preds_by_target[tgt]
    rows = []
    for h in range(FORECAST_LEN):
        rows.append(pd.DataFrame({
            "site_id": test_meta["site_id"].to_numpy(),
            "datetime": test_meta["datetime"].to_numpy(),
            "horizon": h + 1,
            "prediction": all_preds[:, h],
            "truth": Y_test[:, h],
        }))
    pred_df = pd.concat(rows, ignore_index=True)
    pred_path = os.path.join(PRED_DIR, f"exp09v2_{tgt}_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    n_anchor = pred_df["datetime"].nunique()
    print(f"    Saved → {os.path.relpath(pred_path, ROOT_DIR)}  "
          f"({len(pred_df):,} rows, {n_anchor:,} anchors)")

# ── 10. FINAL SUMMARY ─────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  EXP_11 — XGBoost HCHO on project-database weather features")
print(f"  Train < 2023-08-01   Val Aug-Sep 2023   Test Oct-Dec 2023 + 2025")
print(f"  Hyperparameter set: {ACTIVE_HPARAM_SET}  |  Device: {DEVICE}  |  "
      f"Features: {len(FEATURE_COLS)}")
print("=" * 72)
print(f"  {'Target':<13}  {'Horizon':<10}  {'MAE':>8}  {'RMSE':>8}  {'R²':>8}  "
      f"{'ValR²':>8}")
print("  " + "-" * 68)
for tgt in TARGETS:
    for h in EVAL_HORIZONS:
        r = results[tgt][h]
        print(f"  {tgt:<13}  t+{h+1:02d}h       {r['mae']:>8.3f}  {r['rmse']:>8.3f}  "
              f"{r['r2']:>8.4f}  {r['val_r2']:>8.4f}")
print("=" * 72)

# ── 11. TRAIN/VAL/TEST SANITY CHECK (horizon 0) ───────────────────────────────
print("\n[11] Train/Val/Test sanity check (single model, horizon 0) ...")
tr0 = np.isfinite(Y_train[:, 0])
va0 = np.isfinite(Y_val[:, 0])
te0 = np.isfinite(Y_test[:, 0])
xgb_s = xgb.XGBRegressor(
    n_estimators=500, max_depth=5, n_jobs=-1,
    tree_method=TREE_METHOD, device=DEVICE, random_state=42,
)
xgb_s.fit(X_train[tr0], Y_train[tr0, 0])
r2_tr = r2_score(Y_train[tr0, 0], xgb_s.predict(X_train[tr0]))
r2_va = r2_score(Y_val[va0, 0], xgb_s.predict(X_val[va0]))
r2_te = r2_score(Y_test[te0, 0], xgb_s.predict(X_test[te0]))
print(f"    HCHO_target horizon 0 — train R²: {r2_tr:.3f}  "
      f"val R²: {r2_va:.3f}  test R²: {r2_te:.3f}")

print("\nDone — EXP_11 complete.")
