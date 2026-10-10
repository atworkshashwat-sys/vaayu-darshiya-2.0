"""
VAAYU DARSHIYA — EXP_11
Hybrid Ensemble: XGBoost + MLP (Late Fusion via Ridge Meta-Learner)

Strategy:
  - XGBoost (exp09v2) produces row-level predictions: shape (N_xgb, 24)
  - MLP (exp09v4/mlp_ablation) produces window-level predictions: shape (N_mlp, 24)
  - Both are aligned on (site_id, forecast_start_datetime, horizon)
  - A Ridge meta-learner is trained on VAL set predictions from both models
  - Final ensemble is evaluated on TEST set

Why Ridge?
  - XGBoost and MLP errors are correlated but complementary
  - Ridge finds optimal weighting without overfitting the small aligned set
  - Fully interpretable: coefficients show how much each model contributes

Outputs:
  - exp11_ensemble_results.csv  — per-horizon R2/MAE/RMSE
  - exp11_ensemble_by_site.csv
  - exp11_ensemble_by_season.csv
  - exp11_meta_weights.pkl      — the fitted Ridge models (one per horizon)
  - exp11_ensemble_predictions.csv — full prediction CSV for frontend
"""

import os, warnings, joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\Asus\Downloads\VAAYU_DARSHIYA_MODEL_WORK")

# ── CONFIG ────────────────────────────────────────────────────────────────────
TARGETS       = ["NO2_target", "O3_target"]
FORECAST_LEN  = 24
EVAL_HORIZONS = [0, 5, 11, 23]
N_SITES       = 7
TRAIN_YEARS   = [2019, 2020, 2021]
VAL_YEARS     = [2022]
TEST_YEARS    = [2023, 2024]
NWP_FEATURES  = ["O3_forecast", "NO2_forecast", "T_forecast",
                  "q_forecast", "u_forecast", "v_forecast", "w_forecast"]
DROP_COLS     = ["NO2_satellite", "HCHO_satellite", "ratio_satellite"]
TARGET_LAG_HOURS    = [1, 2, 3, 6, 12, 24, 48, 72]
TARGET_ROLL_WINDOWS = [6, 12, 24, 48, 72, 168]
INPUT_LEN     = 48
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("  VAAYU DARSHIYA — EXP_11 Hybrid Ensemble")
print("=" * 60)
print(f"  Device: {DEVICE}")

# ── STEP 1: REBUILD FEATURE DATAFRAME (same pipeline as v2/v4) ───────────────
print("\n[1] Loading and preparing data ...")

all_dfs = []
for i in range(1, N_SITES + 1):
    df_s = pd.read_csv(f"site_{i}_train_data.csv")
    df_s["site_id"] = i
    df_s = df_s.drop(columns=DROP_COLS, errors="ignore")
    all_dfs.append(df_s)
df = pd.concat(all_dfs, ignore_index=True)

df["datetime"] = pd.to_datetime(dict(
    year=df["year"], month=df["month"], day=df["day"], hour=df["hour"]))
df = df.sort_values(["site_id", "datetime"]).reset_index(drop=True)
df = df.dropna(subset=TARGETS).reset_index(drop=True)

# Cyclic features
df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24).astype(np.float32)
df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24).astype(np.float32)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype(np.float32)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype(np.float32)
df["dow_sin"]   = np.sin(2 * np.pi * df["datetime"].dt.dayofweek / 7).astype(np.float32)
df["dow_cos"]   = np.cos(2 * np.pi * df["datetime"].dt.dayofweek / 7).astype(np.float32)
df["day_of_year"] = df["datetime"].dt.dayofyear
df["doy_sin"]   = np.sin(2 * np.pi * df["day_of_year"] / 365.25).astype(np.float32)
df["doy_cos"]   = np.cos(2 * np.pi * df["day_of_year"] / 365.25).astype(np.float32)
df["wind_speed"]   = np.sqrt(df["u_forecast"]**2 + df["v_forecast"]**2).astype(np.float32)
df["wind_dir"]     = np.arctan2(df["v_forecast"], df["u_forecast"])
df["wind_dir_sin"] = np.sin(df["wind_dir"]).astype(np.float32)
df["wind_dir_cos"] = np.cos(df["wind_dir"]).astype(np.float32)

target_lag_cols = []
for col in TARGETS:
    for lag in TARGET_LAG_HOURS:
        cname = f"{col}_lag{lag}"
        df[cname] = df.groupby("site_id")[col].shift(lag)
        target_lag_cols.append(cname)

target_roll_cols = []
for col in TARGETS:
    for w in TARGET_ROLL_WINDOWS:
        cname = f"{col}_roll{w}"
        df[cname] = df.groupby("site_id")[col].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        target_roll_cols.append(cname)

NWP_LAG_HOURS = [1, 2, 3, 6, 12, 24]   # matches exp09v2 exactly (NOT 48/72)
nwp_lag_cols = []
for col in NWP_FEATURES:
    for lag in NWP_LAG_HOURS:
        cname = f"{col}_lag{lag}"
        df[cname] = df.groupby("site_id")[col].shift(lag)
        nwp_lag_cols.append(cname)

# XGBoost also has forward NWP features
nwp_lag_cols = []
for col in NWP_FEATURES:
    for lag in [1, 2, 3, 6, 12, 24]:  # ONLY these 6, no 48/72
        cname = f"{col}_lag{lag}"
        df[cname] = df.groupby("site_id")[col].shift(lag)
        nwp_lag_cols.append(cname)

feature_cols = (
    NWP_FEATURES
    + ["hour_sin", "hour_cos", "month_sin", "month_cos",
       "dow_sin", "dow_cos", "doy_sin", "doy_cos",
       "wind_speed", "wind_dir_sin", "wind_dir_cos"]
    + target_lag_cols + target_roll_cols + nwp_lag_cols
)
all_lag_roll = target_lag_cols + target_roll_cols + nwp_lag_cols
df = df.dropna(subset=all_lag_roll).reset_index(drop=True)

def month_to_season(m):
    if m in (12, 1, 2):   return "Winter"
    if m in (3, 4, 5):    return "Summer"
    if m in (6, 7, 8, 9): return "Monsoon"
    return "Post-monsoon"

df["season"] = df["month"].apply(month_to_season)

train_df = df[df["year"].isin(TRAIN_YEARS)].copy().reset_index(drop=True)
val_df   = df[df["year"].isin(VAL_YEARS)].copy().reset_index(drop=True)
test_df  = df[df["year"].isin(TEST_YEARS)].copy().reset_index(drop=True)

print(f"    Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

# ── STEP 2: XGBoost PREDICTIONS on VAL + TEST ─────────────────────────────────
print("\n[2] Generating XGBoost predictions on val and test ...")

# Load XGBoost feature scaler (fitted on train only)
xgb_feat_scaler  = joblib.load("exp09v2_feat_scaler.pkl")
xgb_feature_cols = joblib.load("exp09v2_feature_cols.pkl")
# ── BUILD ANY MISSING XGB FEATURES DYNAMICALLY ───────────────────────────────
print("    Building any missing XGB features ...")
for col_needed in xgb_feature_cols:
    if col_needed in df.columns:
        continue
    if "_lag" in col_needed:
        base = col_needed[:col_needed.rfind("_lag")]
        lag  = int(col_needed.split("_lag")[-1])
        df[col_needed] = df.groupby("site_id")[base].shift(lag)
    elif "_fwd" in col_needed:
        base = col_needed[:col_needed.rfind("_fwd")]
        fwd  = int(col_needed.split("_fwd")[-1])
        df[col_needed] = df.groupby("site_id")[base].shift(-fwd)
    elif "_roll" in col_needed:
        base = col_needed[:col_needed.rfind("_roll")]
        w    = int(col_needed.split("_roll")[-1])
        df[col_needed] = df.groupby("site_id")[base].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean())

# Re-split after adding new columns
train_df = df[df["year"].isin(TRAIN_YEARS)].copy().reset_index(drop=True)
val_df   = df[df["year"].isin(VAL_YEARS)].copy().reset_index(drop=True)
test_df  = df[df["year"].isin(TEST_YEARS)].copy().reset_index(drop=True)
print(f"    XGB features ready.")
# Build XGBoost feature matrices
X_val_xgb  = xgb_feat_scaler.transform(val_df[xgb_feature_cols])
X_test_xgb = xgb_feat_scaler.transform(test_df[xgb_feature_cols])

# Build ground truth horizon targets for val and test (same as exp09v2 step 6)
def build_horizon_targets(df_in, target_col, forecast_len):
    targets = []
    for h in range(1, forecast_len + 1):
        targets.append(df_in.groupby("site_id")[target_col].shift(-h).values)
    return np.stack(targets, axis=1)

Y_val_xgb, Y_test_xgb = {}, {}
for tgt in TARGETS:
    Y_val_xgb[tgt]  = build_horizon_targets(val_df,  tgt, FORECAST_LEN)
    Y_test_xgb[tgt] = build_horizon_targets(test_df, tgt, FORECAST_LEN)

# Filter NaN rows (both targets must be valid)
val_valid = (
    ~np.isnan(Y_val_xgb["NO2_target"]).any(axis=1) &
    ~np.isnan(Y_val_xgb["O3_target"]).any(axis=1)
)
test_valid = (
    ~np.isnan(Y_test_xgb["NO2_target"]).any(axis=1) &
    ~np.isnan(Y_test_xgb["O3_target"]).any(axis=1)
)

X_val_xgb  = X_val_xgb[val_valid]
X_test_xgb = X_test_xgb[test_valid]

val_meta  = val_df.loc[val_valid,  ["site_id", "datetime", "season"]].reset_index(drop=True)
test_meta = test_df.loc[test_valid, ["site_id", "datetime", "season"]].reset_index(drop=True)

for tgt in TARGETS:
    Y_val_xgb[tgt]  = Y_val_xgb[tgt][val_valid]
    Y_test_xgb[tgt] = Y_test_xgb[tgt][test_valid]

# Predict with saved XGBoost models
xgb_val_preds  = {}
xgb_test_preds = {}

for tgt in TARGETS:
    val_p  = np.zeros((X_val_xgb.shape[0],  FORECAST_LEN))
    test_p = np.zeros((X_test_xgb.shape[0], FORECAST_LEN))
    for h in range(FORECAST_LEN):
        model_h = joblib.load(f"exp09v2_models/{tgt}_h{h:02d}.pkl")
        val_p[:, h]  = np.clip(model_h.predict(X_val_xgb),  0, None)
        test_p[:, h] = np.clip(model_h.predict(X_test_xgb), 0, None)
        if (h + 1) % 6 == 0:
            print(f"    {tgt} h{h+1:02d} done", flush=True)
    xgb_val_preds[tgt]  = val_p
    xgb_test_preds[tgt] = test_p

print("    XGBoost predictions ready.")

# ── STEP 3: MLP PREDICTIONS on VAL + TEST ─────────────────────────────────────
print("\n[3] Generating MLP predictions on val and test ...")

# Import MLP model class inline (mirrors exp09v4 exactly)
import torch.nn as nn
from torch.utils.data import Dataset

class LastStepMLPForecaster(nn.Module):
    def __init__(self, n_features, forecast_len, n_stations=7,
                 embed_dim=8, dropout=0.35, trunk_hidden=128, head_hidden=32):
        super().__init__()
        self.station_embed = nn.Embedding(n_stations + 1, embed_dim)
        input_dim = n_features + embed_dim
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, trunk_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(trunk_hidden, trunk_hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(trunk_hidden)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(trunk_hidden, head_hidden), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(head_hidden, 1),
            ) for _ in range(forecast_len)
        ])

    def forward(self, x, sids):
        last_step = x[:, -1, :]
        emb = self.station_embed(sids)
        h = torch.cat([last_step, emb], dim=-1)
        h = self.trunk(h)
        h = self.norm(h)
        return torch.cat([head(h) for head in self.heads], dim=1)

class PollutionDataset(Dataset):
    def __init__(self, feats, targets, station_ids, datetimes, input_len, forecast_len):
        self.X, self.Y = [], []
        self.site_ids = []
        self.forecast_start_datetimes = []
        total_span = input_len + forecast_len
        expected_step = np.timedelta64(1, "h")
        skipped = 0
        for sid in np.unique(station_ids):
            idx = np.where(station_ids == sid)[0]
            f, t, dt = feats[idx], targets[idx], datetimes[idx]
            for i in range(len(idx) - total_span + 1):
                diffs = np.diff(dt[i:i + total_span])
                if not np.all(diffs == expected_step):
                    skipped += 1
                    continue
                self.X.append(f[i:i + input_len])
                self.Y.append(t[i + input_len:i + input_len + forecast_len])
                self.site_ids.append(sid)
                self.forecast_start_datetimes.append(dt[i + input_len])
        self.X = np.array(self.X, dtype=np.float32)
        self.Y = np.array(self.Y, dtype=np.float32)
        self.site_ids = np.array(self.site_ids, dtype=np.int64)
        self.forecast_start_datetimes = np.array(self.forecast_start_datetimes, dtype="datetime64[ns]")
        print(f"    {len(self):,} samples  (skipped {skipped:,} gap windows)")

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i], self.site_ids[i]

# Load MLP scalers
mlp_feat_scaler   = joblib.load("exp09v4_mlp_ablation_feat_scaler.pkl")
mlp_feature_cols  = joblib.load("exp09v4_mlp_ablation_feature_cols.pkl")
mlp_target_scaler = joblib.load("exp09v4_mlp_ablation_target_scaler.pkl")

# Scale val and test features for MLP
val_feats_mlp  = mlp_feat_scaler.transform(val_df[mlp_feature_cols])
test_feats_mlp = mlp_feat_scaler.transform(test_df[mlp_feature_cols])

# Scale targets for MLP (needed to build Dataset Y, then inverse later)
val_targets_scaled  = mlp_target_scaler.transform(val_df[TARGETS].values)
test_targets_scaled = mlp_target_scaler.transform(test_df[TARGETS].values)

print("    Building MLP val dataset ...")
val_ds_mlp = PollutionDataset(
    val_feats_mlp, val_targets_scaled,
    val_df["site_id"].values, val_df["datetime"].values, INPUT_LEN, FORECAST_LEN)

print("    Building MLP test dataset ...")
test_ds_mlp = PollutionDataset(
    test_feats_mlp, test_targets_scaled,
    test_df["site_id"].values, test_df["datetime"].values, INPUT_LEN, FORECAST_LEN)

val_loader_mlp  = DataLoader(val_ds_mlp,  batch_size=512, shuffle=False, num_workers=0)
test_loader_mlp = DataLoader(test_ds_mlp, batch_size=512, shuffle=False, num_workers=0)

n_features = val_ds_mlp.X.shape[2]

mlp_val_preds  = {}
mlp_test_preds = {}
mlp_val_true   = {}
mlp_test_true  = {}

# Also store datetimes and site_ids from MLP windows for alignment
mlp_val_datetimes  = val_ds_mlp.forecast_start_datetimes
mlp_test_datetimes = test_ds_mlp.forecast_start_datetimes
mlp_val_sites      = val_ds_mlp.site_ids
mlp_test_sites     = test_ds_mlp.site_ids

for tgt in TARGETS:
    tgt_idx = TARGETS.index(tgt)
    model = LastStepMLPForecaster(n_features, FORECAST_LEN).to(DEVICE)
    model.load_state_dict(torch.load(
        f"exp09v4_mlp_ablation_{tgt}_best.pt", map_location=DEVICE))
    model.eval()

    def predict_loader(loader):
        all_p, all_y = [], []
        with torch.no_grad():
            for X_b, Y_b, S_b in loader:
                p = model(X_b.to(DEVICE), S_b.to(DEVICE))
                all_p.append(p.cpu().numpy())
                all_y.append(Y_b[:, :, tgt_idx].numpy())
        return np.concatenate(all_p), np.concatenate(all_y)

    vp, vy = predict_loader(val_loader_mlp)
    tp, ty = predict_loader(test_loader_mlp)

    N_val, N_test = vp.shape[0], tp.shape[0]

    # Inverse transform: dummy array trick (same as exp09v4)
    def inv(arr, tgt_idx, N):
        dummy = np.zeros((N * FORECAST_LEN, 2), dtype=np.float32)
        dummy[:, tgt_idx] = arr.flatten()
        return mlp_target_scaler.inverse_transform(dummy)[:, tgt_idx].clip(0, None).reshape(N, FORECAST_LEN)

    mlp_val_preds[tgt]  = inv(vp, tgt_idx, N_val)
    mlp_test_preds[tgt] = inv(tp, tgt_idx, N_test)
    mlp_val_true[tgt]   = inv(vy, tgt_idx, N_val)
    mlp_test_true[tgt]  = inv(ty, tgt_idx, N_test)

    print(f"    {tgt} MLP predictions ready: val={N_val:,}  test={N_test:,}")

# ── STEP 4: ALIGNMENT — join XGB and MLP on (site_id, forecast_start_datetime) ─
print("\n[4] Aligning XGBoost and MLP predictions ...")

# XGBoost: datetime in val_meta is the ANCHOR hour (the current row).
# MLP:     forecast_start_datetimes is the FIRST forecast hour (anchor + 1h).
# To align: XGB anchor datetime + 1h = MLP forecast_start_datetime.

xgb_val_dt  = pd.to_datetime(val_meta["datetime"]) + pd.Timedelta(hours=1)
xgb_test_dt = pd.to_datetime(test_meta["datetime"]) + pd.Timedelta(hours=1)

mlp_val_dt  = pd.to_datetime(mlp_val_datetimes)
mlp_test_dt = pd.to_datetime(mlp_test_datetimes)

def make_align_key(site_ids, datetimes):
    return pd.Series([f"{s}_{d}" for s, d in zip(site_ids, datetimes)])

xgb_val_keys  = make_align_key(val_meta["site_id"].values, xgb_val_dt)
xgb_test_keys = make_align_key(test_meta["site_id"].values, xgb_test_dt)
mlp_val_keys  = make_align_key(mlp_val_sites, mlp_val_dt)
mlp_test_keys = make_align_key(mlp_test_sites, mlp_test_dt)

# Find common keys
val_common  = set(xgb_val_keys) & set(mlp_val_keys)
test_common = set(xgb_test_keys) & set(mlp_test_keys)

print(f"    Val  — XGB: {len(xgb_val_keys):,}  MLP: {len(mlp_val_keys):,}  "
      f"Aligned: {len(val_common):,}")
print(f"    Test — XGB: {len(xgb_test_keys):,}  MLP: {len(mlp_test_keys):,}  "
      f"Aligned: {len(test_common):,}")

if len(val_common) < 1000:
    print("  WARNING: Very few aligned val samples — check datetime offset logic!")

def get_aligned_indices(keys, common):
    key_to_idx = {k: i for i, k in enumerate(keys)}
    sorted_common = sorted(common)
    idx = np.array([key_to_idx[k] for k in sorted_common])
    return idx, sorted_common

xgb_val_idx,  common_val_keys  = get_aligned_indices(xgb_val_keys, val_common)
mlp_val_idx,  _                = get_aligned_indices(mlp_val_keys, val_common)
xgb_test_idx, common_test_keys = get_aligned_indices(xgb_test_keys, test_common)
mlp_test_idx, _                = get_aligned_indices(mlp_test_keys, test_common)

# Aligned metadata for test
aligned_test_meta = test_meta.iloc[xgb_test_idx].reset_index(drop=True)

# ── STEP 5: RIDGE META-LEARNER ────────────────────────────────────────────────
print("\n[5] Training Ridge meta-learner on aligned val predictions ...")

all_results   = {}
ensemble_test_preds = {}
ensemble_test_true  = {}
meta_models   = {}

for tgt in TARGETS:
    tgt_idx = TARGETS.index(tgt)
    print(f"\n  {tgt}:")

    # Aligned val arrays
    xgb_v  = xgb_val_preds[tgt][xgb_val_idx]   # (N_aligned_val, 24)
    mlp_v  = mlp_val_preds[tgt][mlp_val_idx]    # (N_aligned_val, 24)
    true_v = Y_val_xgb[tgt][xgb_val_idx]        # use XGB ground truth (row-level, more complete)

    # Aligned test arrays
    xgb_t  = xgb_test_preds[tgt][xgb_test_idx]  # (N_aligned_test, 24)
    mlp_t  = mlp_test_preds[tgt][mlp_test_idx]   # (N_aligned_test, 24)
    true_t = Y_test_xgb[tgt][xgb_test_idx]       # XGB ground truth

    horizon_models  = []
    test_ensemble_h = np.zeros((len(xgb_test_idx), FORECAST_LEN))
    results_h       = {}

    for h in range(FORECAST_LEN):
        # Stack [xgb_pred, mlp_pred] as features for Ridge
        X_meta_val  = np.stack([xgb_v[:, h],  mlp_v[:, h]],  axis=1)
        X_meta_test = np.stack([xgb_t[:, h],  mlp_t[:, h]],  axis=1)
        y_val_h     = true_v[:, h]
        y_test_h    = true_t[:, h]

        # Drop NaN rows from val (horizon targets at end of series)
        valid = ~np.isnan(y_val_h)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_meta_val[valid], y_val_h[valid])
        horizon_models.append(ridge)

        # Test predictions
        pred_h = np.clip(ridge.predict(X_meta_test), 0, None)
        test_ensemble_h[:, h] = pred_h

        # Test metrics (skip NaN ground truth rows)
        valid_t = ~np.isnan(y_test_h)
        mae  = mean_absolute_error(y_test_h[valid_t], pred_h[valid_t])
        rmse = np.sqrt(mean_squared_error(y_test_h[valid_t], pred_h[valid_t]))
        r2   = r2_score(y_test_h[valid_t], pred_h[valid_t])
        results_h[h] = {"mae": mae, "rmse": rmse, "r2": r2,
                         "coef_xgb": ridge.coef_[0], "coef_mlp": ridge.coef_[1]}

    meta_models[tgt]          = horizon_models
    all_results[tgt]          = results_h
    ensemble_test_preds[tgt]  = test_ensemble_h
    ensemble_test_true[tgt]   = true_t

    # Print horizon summary
    print(f"  {'Horizon':<10}  {'MAE':>8}  {'RMSE':>8}  {'R2':>8}  "
          f"{'w_XGB':>8}  {'w_MLP':>8}")
    print("  " + "-" * 58)
    for h in EVAL_HORIZONS:
        r = results_h[h]
        print(f"  t+{h+1:02d}h      {r['mae']:>8.3f}  {r['rmse']:>8.3f}  {r['r2']:>8.4f}  "
              f"{r['coef_xgb']:>8.4f}  {r['coef_mlp']:>8.4f}")

joblib.dump(meta_models, "exp11_meta_models.pkl")

# ── STEP 6: COMPARISON TABLE ──────────────────────────────────────────────────
XGB_R2 = {
    "NO2_target": {0: 0.7958, 5: 0.6357, 11: 0.6193, 23: 0.5889},
    "O3_target":  {0: 0.8671, 5: 0.7564, 11: 0.7523, 23: 0.7415},
}
MLP_R2 = {
    "NO2_target": {0: 0.6960, 5: 0.5488, 11: 0.5373, 23: 0.4954},
    "O3_target":  {0: 0.7438, 5: 0.6533, 11: 0.6851, 23: 0.6295},
}

print(f"\n{'='*70}")
print("  EXP_11 — Model Comparison (Test R2)")
print(f"{'='*70}")
print(f"  {'Target':<12}  {'Horizon':<10}  {'XGB':>8}  {'MLP':>8}  {'Ensemble':>10}  {'Best':>8}")
print("  " + "-" * 66)
for tgt in TARGETS:
    for h in EVAL_HORIZONS:
        xr  = XGB_R2[tgt][h]
        mr  = MLP_R2[tgt][h]
        er  = all_results[tgt][h]["r2"]
        best = "Ensemble" if er >= max(xr, mr) else ("XGB" if xr >= mr else "MLP")
        print(f"  {tgt:<12}  t+{h+1:02d}h      {xr:>8.4f}  {mr:>8.4f}  {er:>10.4f}  {best:>8}")
print(f"{'='*70}")

# ── STEP 7: PER-SITE BREAKDOWN ────────────────────────────────────────────────
print("\n[7] Per-site breakdown ...")
site_rows = []
for tgt in TARGETS:
    tgt_idx = TARGETS.index(tgt)
    test_sites = aligned_test_meta["site_id"].values
    for sid in range(1, N_SITES + 1):
        mask = test_sites == sid
        for h in EVAL_HORIZONS:
            y = ensemble_test_true[tgt][mask, h]
            p = ensemble_test_preds[tgt][mask, h]
            valid = ~np.isnan(y)
            if valid.sum() < 10: continue
            site_rows.append({
                "target": tgt, "site": sid, "horizon": h + 1,
                "mae":  mean_absolute_error(y[valid], p[valid]),
                "rmse": np.sqrt(mean_squared_error(y[valid], p[valid])),
                "r2":   r2_score(y[valid], p[valid]),
            })
pd.DataFrame(site_rows).to_csv("exp11_ensemble_by_site.csv", index=False)
print("    Saved → exp11_ensemble_by_site.csv")

# ── STEP 8: PER-SEASON BREAKDOWN ─────────────────────────────────────────────
print("\n[8] Per-season breakdown ...")
season_rows = []
for tgt in TARGETS:
    seasons = aligned_test_meta["season"].values
    for season in ["Winter", "Summer", "Monsoon", "Post-monsoon"]:
        mask = seasons == season
        for h in EVAL_HORIZONS:
            y = ensemble_test_true[tgt][mask, h]
            p = ensemble_test_preds[tgt][mask, h]
            valid = ~np.isnan(y)
            if valid.sum() < 10: continue
            season_rows.append({
                "target": tgt, "season": season, "horizon": h + 1,
                "mae":  mean_absolute_error(y[valid], p[valid]),
                "rmse": np.sqrt(mean_squared_error(y[valid], p[valid])),
                "r2":   r2_score(y[valid], p[valid]),
            })
pd.DataFrame(season_rows).to_csv("exp11_ensemble_by_season.csv", index=False)
print("    Saved → exp11_ensemble_by_season.csv")

# ── STEP 9: SAVE RESULTS + PREDICTIONS ───────────────────────────────────────
print("\n[9] Saving results and predictions ...")
results_rows = []
for tgt in TARGETS:
    for h in range(FORECAST_LEN):
        r = all_results[tgt][h]
        results_rows.append({"target": tgt, "horizon": h + 1, **r})
pd.DataFrame(results_rows).to_csv("exp11_ensemble_results.csv", index=False)
print("    Saved → exp11_ensemble_results.csv")

# Full prediction CSV for frontend
pred_rows = []
for tgt in TARGETS:
    for h in range(FORECAST_LEN):
        chunk = pd.DataFrame({
            "site_id":   aligned_test_meta["site_id"].values,
            "datetime":  aligned_test_meta["datetime"].values,
            "horizon":   h + 1,
            "target":    tgt,
            "predicted": ensemble_test_preds[tgt][:, h],
            "actual":    ensemble_test_true[tgt][:, h],
        })
        pred_rows.append(chunk)
pd.concat(pred_rows, ignore_index=True).to_csv(
    "exp11_ensemble_predictions.csv", index=False)
print("    Saved → exp11_ensemble_predictions.csv")

# ── STEP 10: PLOTS ────────────────────────────────────────────────────────────
print("\n[10] Generating plots ...")
for tgt in TARGETS:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{tgt} — Ensemble Actual vs Predicted (Test Set)", fontsize=14)
    for ax_i, h in enumerate(EVAL_HORIZONS):
        ax = axes[ax_i // 2, ax_i % 2]
        y = ensemble_test_true[tgt][:, h]
        p = ensemble_test_preds[tgt][:, h]
        valid = ~np.isnan(y)
        ax.scatter(y[valid], p[valid], alpha=0.1, s=5)
        lims = [0, max(y[valid].max(), p[valid].max()) * 1.05]
        ax.plot(lims, lims, "r--", lw=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"t+{h+1:02d}h  R2={all_results[tgt][h]['r2']:.4f}")
    plt.tight_layout()
    plt.savefig(f"exp11_{tgt}_actual_vs_predicted.png", dpi=150)
    plt.close()
    print(f"    Plot saved: exp11_{tgt}_actual_vs_predicted.png")

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  EXP_11 Complete — Hybrid Ensemble Summary")
print(f"{'='*60}")
for tgt in TARGETS:
    print(f"\n  {tgt}:")
    for h in EVAL_HORIZONS:
        r  = all_results[tgt][h]
        xr = XGB_R2[tgt][h]
        gain = r["r2"] - xr
        print(f"    t+{h+1:02d}h  R2={r['r2']:.4f}  (vs XGB {xr:.4f}, gain={gain:+.4f})")
print(f"\n{'='*60}")
print("Done.")
