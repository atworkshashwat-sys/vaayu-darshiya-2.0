"""
VAAYU DARSHIYA — EXP_09 v2
XGBoost on NWP Forecast Features (Improved)
- Validation split: 2022
- Both-target NaN filtering
- Wind direction features
- Longer rolling windows
- Better XGBoost params
- Early stopping on VAL not TEST
- Model saving per horizon
- Sites: 1-7
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\Asus\Downloads\VAAYU_DARSHIYA_MODEL_WORK")

# ── GPU DETECTION ─────────────────────────────────────────────────────────────
try:
    _test = xgb.XGBRegressor(tree_method="hist", device="cuda", n_estimators=1)
    _test.fit([[1]], [1])
    TREE_METHOD = "hist"
    DEVICE      = "cuda"
    print("GPU detected — using CUDA")
except Exception:
    TREE_METHOD = "hist"
    DEVICE      = "cpu"
    print("No CUDA — using CPU")

# ── CONFIG ────────────────────────────────────────────────────────────────────
N_SITES       = 7
TRAIN_YEARS   = [2019, 2020, 2021]
VAL_YEARS     = [2022]
TEST_YEARS    = [2023, 2024]
TARGETS       = ["NO2_target", "O3_target"]
FORECAST_LEN  = 24
EVAL_HORIZONS = [0, 5, 11, 23]

NWP_FEATURES    = ["O3_forecast", "NO2_forecast", "T_forecast",
                    "q_forecast", "u_forecast", "v_forecast", "w_forecast"]
DROP_BEFORE_LAG = ["NO2_satellite", "HCHO_satellite", "ratio_satellite"]
LAG_HOURS       = [1, 2, 3, 6, 12, 24, 48, 72]

# ── 1. LOAD ALL SITES ─────────────────────────────────────────────────────────
print("\n[1] Loading data from all sites ...")
all_dfs = []
for i in range(1, N_SITES + 1):
    df_s = pd.read_csv(f"site_{i}_train_data.csv")
    df_s["site_id"] = i
    df_s = df_s.drop(columns=DROP_BEFORE_LAG, errors="ignore")
    all_dfs.append(df_s)
    print(f"    Site {i}: {len(df_s):,} rows")

df = pd.concat(all_dfs, ignore_index=True)
print(f"    Total: {len(df):,} rows")

# ── 2. CHRONOLOGICAL SORT ─────────────────────────────────────────────────────
print("\n[2] Sorting chronologically ...")
df["datetime"] = pd.to_datetime(dict(
    year=df["year"], month=df["month"], day=df["day"], hour=df["hour"]
))
df = df.sort_values(["site_id", "datetime"]).reset_index(drop=True)

# ── 3. FEATURE ENGINEERING ────────────────────────────────────────────────────
print("\n[3] Engineering features ...")

# Cyclic time
df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24).astype(np.float32)
df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24).astype(np.float32)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype(np.float32)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype(np.float32)

# Wind speed + direction
df["wind_speed"]   = np.sqrt(df["u_forecast"]**2 + df["v_forecast"]**2)
df["wind_dir"]     = np.arctan2(df["v_forecast"], df["u_forecast"])
df["wind_dir_sin"] = np.sin(df["wind_dir"]).astype(np.float32)
df["wind_dir_cos"] = np.cos(df["wind_dir"]).astype(np.float32)

# Lag features
lag_cols = []
for col in ["NO2_target", "O3_target"] + NWP_FEATURES:
    for lag in LAG_HOURS:
        cname = f"{col}_lag{lag}"
        df[cname] = df.groupby("site_id")[col].shift(lag)
        lag_cols.append(cname)

# Forward NWP features
for col in NWP_FEATURES:
    for fwd in [1, 2, 3, 6, 12, 24]:
        cname = f"{col}_fwd{fwd}"
        df[cname] = df.groupby("site_id")[col].shift(-fwd)
        lag_cols.append(cname)

# Rolling features
roll_cols = []
for col in ["NO2_target", "O3_target"]:
    for window in [6, 12, 24, 48, 72, 168]:
        cname = f"{col}_roll{window}"
        df[cname] = df.groupby("site_id")[col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        roll_cols.append(cname)

# Drop NaN and reset index
lag_roll_cols = lag_cols + roll_cols
df = df.dropna(subset=lag_roll_cols).reset_index(drop=True)
print(f"    After dropna: {len(df):,} rows")

FEATURE_COLS = (
    NWP_FEATURES
    + ["hour_sin", "hour_cos", "month_sin", "month_cos",
       "wind_speed", "wind_dir_sin", "wind_dir_cos"]
    + lag_cols
    + roll_cols
)
print(f"    Total features: {len(FEATURE_COLS)}")

# ── 4. SPLIT ──────────────────────────────────────────────────────────────────
print("\n[4] Splitting train / val / test ...")
train_df = df[df["year"].isin(TRAIN_YEARS)].copy().reset_index(drop=True)
val_df   = df[df["year"].isin(VAL_YEARS)].copy().reset_index(drop=True)
test_df  = df[df["year"].isin(TEST_YEARS)].copy().reset_index(drop=True)
print(f"    Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

# ── 5. SCALE (fit on train only) ──────────────────────────────────────────────
print("\n[5] Scaling ...")
feat_scaler = StandardScaler()
X_train = feat_scaler.fit_transform(train_df[FEATURE_COLS])
X_val   = feat_scaler.transform(val_df[FEATURE_COLS])
X_test  = feat_scaler.transform(test_df[FEATURE_COLS])
joblib.dump(feat_scaler,  "exp09v2_feat_scaler.pkl")
joblib.dump(FEATURE_COLS, "exp09v2_feature_cols.pkl")

# ── 6. BUILD HORIZON TARGETS ──────────────────────────────────────────────────
print("\n[6] Building forecast targets ...")

def build_horizon_targets(df, target_col, forecast_len):
    targets = []
    for h in range(1, forecast_len + 1):
        targets.append(df.groupby("site_id")[target_col].shift(-h).values)
    return np.stack(targets, axis=1)

Y_train, Y_val, Y_test = {}, {}, {}
for tgt in TARGETS:
    Y_train[tgt] = build_horizon_targets(train_df, tgt, FORECAST_LEN)
    Y_val[tgt]   = build_horizon_targets(val_df,   tgt, FORECAST_LEN)
    Y_test[tgt]  = build_horizon_targets(test_df,  tgt, FORECAST_LEN)

# Filter NaN for BOTH targets simultaneously
train_valid = (
    ~np.isnan(Y_train["NO2_target"]).any(axis=1) &
    ~np.isnan(Y_train["O3_target"]).any(axis=1)
)
val_valid = (
    ~np.isnan(Y_val["NO2_target"]).any(axis=1) &
    ~np.isnan(Y_val["O3_target"]).any(axis=1)
)
test_valid = (
    ~np.isnan(Y_test["NO2_target"]).any(axis=1) &
    ~np.isnan(Y_test["O3_target"]).any(axis=1)
)

X_train = X_train[train_valid]
X_val   = X_val[val_valid]
X_test  = X_test[test_valid]
for tgt in TARGETS:
    Y_train[tgt] = Y_train[tgt][train_valid]
    Y_val[tgt]   = Y_val[tgt][val_valid]
    Y_test[tgt]  = Y_test[tgt][test_valid]

print(f"    Train: {X_train.shape[0]:,}  Val: {X_val.shape[0]:,}  Test: {X_test.shape[0]:,}")

# ── 7. TRAIN XGBOOST PER HORIZON ─────────────────────────────────────────────
print("\n[7] Training XGBoost models ...")
os.makedirs("exp09v2_models", exist_ok=True)
results = {}

for tgt in TARGETS:
    print(f"\n    {tgt} — training {FORECAST_LEN} horizon models ...")
    all_preds = np.zeros((X_test.shape[0], FORECAST_LEN))

    base_params = dict(
        n_estimators=2500,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=10,
        gamma=0.2,
        subsample=0.6,
        colsample_bytree=0.5,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="rmse",
        early_stopping_rounds=100,
    )

    for h in range(FORECAST_LEN):
        y_tr  = Y_train[tgt][:, h]
        y_val = Y_val[tgt][:, h]

        params = base_params.copy()
        xgb_h = xgb.XGBRegressor(**params)
        xgb_h.fit(
            X_train, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        all_preds[:, h] = np.clip(xgb_h.predict(X_test), 0, None)
        joblib.dump(xgb_h, f"exp09v2_models/{tgt}_h{h:02d}.pkl")
        print(f"      t+{h+1:02d} done (best_iter={xgb_h.best_iteration})", flush=True)

    joblib.dump(all_preds, f"exp09v2_xgb_{tgt}_preds.pkl")

    print(f"\n    {'Horizon':<10}  {'MAE':>8}  {'RMSE':>8}  {'R²':>8}")
    print("    " + "-" * 45)
    results[tgt] = {}
    for h in range(FORECAST_LEN):
        p    = all_preds[:, h]
        y    = Y_test[tgt][:, h]
        mae  = mean_absolute_error(y, p)
        rmse = np.sqrt(mean_squared_error(y, p))
        r2   = r2_score(y, p)
        results[tgt][h] = {"mae": mae, "rmse": rmse, "r2": r2}
        if h in EVAL_HORIZONS:
            print(f"    t+{h+1:02d}h       {mae:>8.3f}  {rmse:>8.3f}  {r2:>8.4f}")

    joblib.dump(results[tgt], f"exp09v2_xgb_{tgt}_results.pkl")
    print(f"    Saved → exp09v2_xgb_{tgt}_results.pkl")

# ── 8. FINAL SUMMARY ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  EXP_09 v2 — XGBoost on NWP Features (Improved)")
print("  Train: 2019-2021  Val: 2022  Test: 2023-2024  Sites: 1-7")
print("=" * 60)
print(f"  {'Target':<12}  {'Horizon':<10}  {'MAE':>8}  {'RMSE':>8}  {'R²':>8}")
print("  " + "-" * 58)
for tgt in TARGETS:
    for h in EVAL_HORIZONS:
        r = results[tgt][h]
        print(f"  {tgt:<12}  t+{h+1:02d}h       "
              f"{r['mae']:>8.3f}  {r['rmse']:>8.3f}  {r['r2']:>8.4f}")
print("=" * 60)

print("""
  EXP_09 v1 baseline:
    NO2  t+01h R²=0.7588
    O3   t+01h R²=0.8519
""")

# ── 9. TRAIN SANITY CHECK ─────────────────────────────────────────────────────
print("\n[9] Train sanity check ...")
for tgt in TARGETS:
    xgb_s = xgb.XGBRegressor(n_estimators=200, max_depth=5, n_jobs=-1)
    xgb_s.fit(X_train[:10000], Y_train[tgt][:10000, 0])
    p = xgb_s.predict(X_train[:10000])
    print(f"    {tgt} horizon 0 — train R²: "
          f"{r2_score(Y_train[tgt][:10000, 0], p):.3f}")
print("    (train R² >> test R² = overfitting)")

print("\nDone — EXP_09 v2 complete.")
