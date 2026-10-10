"""
VAAYU DARSHIYA — EXP_09 v3b
BiLSTM + Attention : NO2 & O3 Multi-step Forecasting (Ensemble-Ready)
Input  : 48h window of NWP forecast features + past NO2/O3 (autoregressive)
Output : next 24h of NO2 and O3
Train  : 2019 – 2021
Val    : 2022
Test   : 2023 – 2024

Changes from exp_09v3_bilstm.py:
  - CRITICAL: NO2_target and O3_target added to input features.
    The model now sees its own recent past (autoregressive input) — same
    idea as XGBoost's lag features. Targets in input are at t-48..t-1
    (the 48h window), target output is at t..t+23, so no leakage.
  - Removed time_idx (extrapolated by scaler on val/test, hurt generalization)
  - Smaller model: HIDDEN_SIZE 128→64, NUM_LAYERS 2→1 (less overfitting on 84K samples)
  - Stronger regularization: DROPOUT 0.4→0.5, WEIGHT_DECAY 5e-4→1e-3
  - Milder horizon weights: linspace(0.5, 1.5) → (0.7, 1.3)
  - Early stopping patience 10→6
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import os, math, warnings, joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\Asus\Downloads\VAAYU_DARSHIYA_MODEL_WORK")

# ── CONFIG ────────────────────────────────────────────────────────────────────
N_SITES         = 7
TRAIN_YEARS     = [2019, 2020, 2021]
VAL_YEARS       = [2022]
TEST_YEARS      = [2023, 2024]
TARGETS         = ["NO2_target", "O3_target"]
EVAL_HORIZONS   = [0, 5, 11, 23]      # t+01, t+06, t+12, t+24

INPUT_LEN       = 48       # hours of history fed to the model
FORECAST_LEN    = 24       # hours ahead to predict

HIDDEN_SIZE     = 64       # v3b: 128 → 64 (less overfitting on 84K samples)
NUM_LAYERS      = 1        # v3b: 2 → 1
DROPOUT         = 0.5      # v3b: 0.4 → 0.5
ATTN_HEADS      = 4

BATCH_SIZE      = 256
EPOCHS          = 50
LR              = 1e-4
PATIENCE        = 6        # v3b: 10 → 6 (stop sooner on val plateau)
WEIGHT_DECAY    = 1e-3     # v3b: 5e-4 → 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── 1. LOAD + CONCAT ALL SITES (mirror exp_09_v2_xgboost.py) ──────────────────
print("\n[1] Loading data from all sites ...")
DROP_BEFORE_LAG = ["NO2_satellite", "HCHO_satellite", "ratio_satellite"]
all_dfs = []
for i in range(1, N_SITES + 1):
    df_s = pd.read_csv(f"site_{i}_train_data.csv")
    df_s["site_id"] = i
    df_s = df_s.drop(columns=DROP_BEFORE_LAG, errors="ignore")
    all_dfs.append(df_s)
    print(f"    Site {i}: {len(df_s):,} rows")

df = pd.concat(all_dfs, ignore_index=True)
print(f"    Total: {len(df):,} rows")

# ── 2. CHRONOLOGICAL SORT + DT BUILD ──────────────────────────────────────────
print("\n[2] Sorting chronologically ...")
df["datetime"] = pd.to_datetime(dict(
    year=df["year"], month=df["month"], day=df["day"], hour=df["hour"]
))
df = df.sort_values(["site_id", "datetime"]).reset_index(drop=True)

# Drop rows where either target is NaN (match XGBoost filter)
df = df.dropna(subset=TARGETS).reset_index(drop=True)
print(f"    After dropna: {len(df):,} rows")

# ── 3. FEATURE ENGINEERING ────────────────────────────────────────────────────
print("\n[3] Engineering features ...")
df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24).astype(np.float32)
df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24).astype(np.float32)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype(np.float32)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype(np.float32)

# v3b: NO time_idx — was extrapolated by scaler on val/test (which are
# later years), produced extreme z-scores, hurt generalization.

# v3b: include NO2_target and O3_target as INPUT features (autoregressive).
# Within the 48h input window these are t-48..t-1, the 24h target window is
# t..t+23 — strictly future, no leakage. This gives the BiLSTM the same
# information XGBoost had via its engineered lag features.
RAW_FEATURES = [
    "year", "month", "day", "hour",
    "O3_forecast", "NO2_forecast", "T_forecast",
    "q_forecast", "u_forecast", "v_forecast", "w_forecast",
    "site_id",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "NO2_target", "O3_target",     # v3b: autoregressive inputs
]
feature_cols = list(RAW_FEATURES)
print(f"    Features: {len(feature_cols)}")
print(f"    Targets : {TARGETS}")

# Per-site mean-impute any remaining NaN in features
for col in feature_cols:
    if df[col].isna().any():
        df[col] = df.groupby("site_id")[col].transform(
            lambda s: s.fillna(s.mean())
        )
        # If a site is entirely NaN, fall back to global mean
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())

# ── 4. SPLIT ──────────────────────────────────────────────────────────────────
print("\n[4] Splitting train / val / test ...")
train_mask = df["year"].isin(TRAIN_YEARS)
val_mask   = df["year"].isin(VAL_YEARS)
test_mask  = df["year"].isin(TEST_YEARS)

train_df = df[train_mask].copy().reset_index(drop=True)
val_df   = df[val_mask].copy().reset_index(drop=True)
test_df  = df[test_mask].copy().reset_index(drop=True)

print(f"    Train: {len(train_df):,} rows  ({min(TRAIN_YEARS)}–{max(TRAIN_YEARS)})")
print(f"    Val  : {len(val_df):,} rows  ({VAL_YEARS[0]})")
print(f"    Test : {len(test_df):,} rows  ({min(TEST_YEARS)}–{max(TEST_YEARS)})")

# ── 5. SCALE ──────────────────────────────────────────────────────────────────
print("\n[5] Scaling ...")
feat_scaler   = StandardScaler()
target_scaler = StandardScaler()

train_feats = feat_scaler.fit_transform(train_df[feature_cols])
val_feats   = feat_scaler.transform(val_df[feature_cols])
test_feats  = feat_scaler.transform(test_df[feature_cols])

train_targets = target_scaler.fit_transform(train_df[TARGETS].values)
val_targets   = target_scaler.transform(val_df[TARGETS].values)
test_targets  = target_scaler.transform(test_df[TARGETS].values)

# ── 6. DATASET (per-station windowing, no cross-site leakage) ─────────────────
class PollutionDataset(Dataset):
    """
    Builds (input_window, target_window) pairs per station.
    input_window  : (INPUT_LEN,   n_features)  — past 48h
    target_window : (FORECAST_LEN, n_targets)  — next 24h NO2 & O3
    """
    def __init__(self, feats, targets, station_ids, input_len, forecast_len):
        self.X, self.Y = [], []
        unique_stations = np.unique(station_ids)
        for sid in unique_stations:
            idx   = np.where(station_ids == sid)[0]
            f     = feats[idx]
            t     = targets[idx]
            total = len(idx)
            for i in range(total - input_len - forecast_len + 1):
                self.X.append(f[i : i + input_len])
                self.Y.append(t[i + input_len : i + input_len + forecast_len])
        self.X = np.array(self.X, dtype=np.float32)
        self.Y = np.array(self.Y, dtype=np.float32)
        print(f"    Dataset: {len(self.X):,} samples  "
              f"X={self.X.shape}  Y={self.Y.shape}")

    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]

print("\n[6] Building datasets ...")
train_station_ids = train_df["site_id"].values
val_station_ids   = val_df["site_id"].values
test_station_ids  = test_df["site_id"].values

train_ds = PollutionDataset(train_feats, train_targets,
                             train_station_ids, INPUT_LEN, FORECAST_LEN)
val_ds   = PollutionDataset(val_feats,   val_targets,
                             val_station_ids,   INPUT_LEN, FORECAST_LEN)
test_ds  = PollutionDataset(test_feats,  test_targets,
                             test_station_ids,  INPUT_LEN, FORECAST_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=False)

print(f"    Train batches: {len(train_loader)}")
print(f"    Val   batches: {len(val_loader)}")
print(f"    Test  batches: {len(test_loader)}")

# ── 7. MODEL ──────────────────────────────────────────────────────────────────
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim   = hidden_size * 2,   # BiLSTM doubles hidden size
            num_heads   = num_heads,
            dropout     = 0.1,
            batch_first = True,
        )
        self.norm = nn.LayerNorm(hidden_size * 2)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)         # residual connection


class VaayuBiLSTM(nn.Module):
    def __init__(self, n_features, hidden_size, num_layers, dropout,
                 attn_heads, forecast_len, n_targets):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size  = n_features,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0,
            batch_first = True,
            bidirectional = True,
        )
        self.attention  = AttentionLayer(hidden_size, attn_heads)
        self.dropout    = nn.Dropout(dropout)

        lstm_out_size   = hidden_size * 2
        self.head = nn.Sequential(
            nn.Linear(lstm_out_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, forecast_len * n_targets),
        )
        self.forecast_len = forecast_len
        self.n_targets    = n_targets

    def forward(self, x):
        lstm_out, _  = self.bilstm(x)          # (B, T, H*2)
        attn_out     = self.attention(lstm_out) # (B, T, H*2)
        context      = attn_out[:, -1, :]       # last timestep
        context      = self.dropout(context)
        out          = self.head(context)       # (B, forecast_len * n_targets)
        return out.view(-1, self.forecast_len, self.n_targets)


n_features = train_ds.X.shape[2]
model = VaayuBiLSTM(
    n_features   = n_features,
    hidden_size  = HIDDEN_SIZE,
    num_layers   = NUM_LAYERS,
    dropout      = DROPOUT,
    attn_heads   = ATTN_HEADS,
    forecast_len = FORECAST_LEN,
    n_targets    = len(TARGETS),
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n[7] Model ready — {total_params:,} parameters")

# ── 8. TRAIN ──────────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)
criterion  = nn.HuberLoss(delta=1.0, reduction="mean")   # val
huber_none = nn.HuberLoss(delta=1.0, reduction="none")   # train (per-step weighted)

# Horizon weights: v3b uses milder gradient (0.7 → 1.3) since autoregressive
# inputs make long horizons genuinely learnable — no need to over-emphasize.
horizon_weights = torch.linspace(0.7, 1.3, FORECAST_LEN, device=DEVICE)
print(f"    Horizon weights: {horizon_weights[0].item():.2f} → "
      f"{horizon_weights[-1].item():.2f}")

# AMP only works on CUDA
_amp = getattr(torch, "amp", None)
if _amp is not None and hasattr(_amp, "GradScaler") and DEVICE.type == "cuda":
    scaler_amp   = torch.amp.GradScaler("cuda", enabled=True)
    autocast_ctx = lambda: torch.amp.autocast("cuda", enabled=True)
else:
    scaler_amp   = None
    from contextlib import nullcontext
    autocast_ctx = nullcontext

print(f"\n[8] Training for up to {EPOCHS} epochs …\n")
best_val_loss = float("inf")
patience_counter = 0
history = {"train": [], "val": []}

for epoch in range(1, EPOCHS + 1):
    # — train —
    model.train()
    train_loss = 0.0
    for X_batch, Y_batch in train_loader:
        X_batch = X_batch.to(DEVICE, non_blocking=True)
        Y_batch = Y_batch.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        with autocast_ctx():
            pred       = model(X_batch)
            per_step   = huber_none(pred, Y_batch)   # (B, 24, 2)
            loss       = (per_step * horizon_weights.view(1, -1, 1)).mean()
        if scaler_amp is not None:
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # — validate on 2022 only —
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            X_batch = X_batch.to(DEVICE, non_blocking=True)
            Y_batch = Y_batch.to(DEVICE, non_blocking=True)
            with autocast_ctx():
                pred = model(X_batch)
                loss = criterion(pred, Y_batch)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    scheduler.step(val_loss)
    history["train"].append(train_loss)
    history["val"].append(val_loss)

    improved = val_loss < best_val_loss
    if improved:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "exp09v3b_bilstm_best.pt")
    else:
        patience_counter += 1

    mark = " *" if improved else ""
    print(f"  Epoch {epoch:02d}/{EPOCHS}  "
          f"train={train_loss:.4f}  val={val_loss:.4f}{mark}  "
          f"patience={patience_counter}/{PATIENCE}", flush=True)

    if patience_counter >= PATIENCE:
        print(f"\n  Early stopping at epoch {epoch}.")
        break

# ── 9. EVALUATE ON TEST (2023-2024) ───────────────────────────────────────────
print("\n[9] Evaluating best model on 2023-2024 test set …")
model.load_state_dict(torch.load("exp09v3b_bilstm_best.pt", map_location=DEVICE))
model.eval()

all_preds, all_true = [], []
with torch.no_grad():
    for X_batch, Y_batch in test_loader:
        X_batch = X_batch.to(DEVICE, non_blocking=True)
        with autocast_ctx():
            pred = model(X_batch)
        all_preds.append(pred.cpu().numpy())
        all_true.append(Y_batch.numpy())

all_preds = np.concatenate(all_preds, axis=0)  # (N, 24, 2)
all_true  = np.concatenate(all_true,  axis=0)  # (N, 24, 2)

# Inverse-scale back to µg/m³
N = all_preds.shape[0]
preds_2d = all_preds.reshape(N * FORECAST_LEN, len(TARGETS))
true_2d  = all_true.reshape( N * FORECAST_LEN, len(TARGETS))
preds_2d = target_scaler.inverse_transform(preds_2d)
true_2d  = target_scaler.inverse_transform(true_2d)
preds_2d = np.clip(preds_2d, 0, None)

# Reshape back to (N, 24, 2)
preds_3d = preds_2d.reshape(N, FORECAST_LEN, len(TARGETS))
true_3d  = true_2d.reshape( N, FORECAST_LEN, len(TARGETS))

# Per-horizon metrics (test set)
print("\n" + "=" * 65)
print("  EXP_09 v3b — BiLSTM + Attention Results (autoregressive input)")
print("  Train: 2019-2021  Val: 2022  Test: 2023-2024  Sites: 1-7")
print("=" * 65)
print(f"  {'Target':<12}  {'Horizon':<10}  {'MAE':>8}  {'RMSE':>8}  {'R²':>8}")
print("  " + "-" * 58)

results = {}
for tgt_idx, tgt in enumerate(TARGETS):
    results[tgt] = {}
    for h in range(FORECAST_LEN):
        p = preds_3d[:, h, tgt_idx]
        y = true_3d[:, h, tgt_idx]
        mae  = mean_absolute_error(y, p)
        rmse = np.sqrt(mean_squared_error(y, p))
        r2   = r2_score(y, p)
        results[tgt][h] = {"mae": mae, "rmse": rmse, "r2": r2}
        if h in EVAL_HORIZONS:
            print(f"  {tgt:<12}  t+{h+1:02d}h       "
                  f"{mae:>8.3f}  {rmse:>8.3f}  {r2:>8.4f}")
print("=" * 65)

# Build site_id & datetime arrays aligned with each test sample
# Each test sample corresponds to datetime = t[input_len-1] (the "anchor" hour)
# (the input window ends at row index input_len-1, and the target window
#  starts at input_len, so the forecast is for hours t+1..t+24)
test_anchor_idx = np.array([
    i + (INPUT_LEN - 1)
    for i in range(N)
])
# We need to map sample index → (site_id, datetime)
# Reconstruct: walk through each station in the same order the Dataset used
site_id_per_sample = np.empty(N, dtype=np.int64)
dt_per_sample      = np.empty(N, dtype="datetime64[ns]")
running = 0
for sid in np.unique(test_station_ids):
    idx   = np.where(test_station_ids == sid)[0]
    total = len(idx)
    n_samples = total - INPUT_LEN - FORECAST_LEN + 1
    sample_idx = np.arange(running, running + n_samples)
    anchor_idx = idx[INPUT_LEN - 1 : INPUT_LEN - 1 + n_samples]
    site_id_per_sample[sample_idx] = sid
    dt_per_sample[sample_idx]      = test_df.iloc[anchor_idx]["datetime"].values
    running += n_samples

assert running == N, f"Sample count mismatch: {running} vs {N}"

# ── 10. PER-SITE BREAKDOWN (match XGBoost format) ─────────────────────────────
print("\n[10] Per-site breakdown (test set) ...")
site_results = {}
for tgt_idx, tgt in enumerate(TARGETS):
    site_results[tgt] = []
    print(f"\n    {tgt} — error by site (test set):")
    print(f"    {'Site':<6}  {'Horizon':<10}  {'MAE':>8}  {'RMSE':>8}  {'R²':>8}")
    print("    " + "-" * 50)
    for sid in range(1, N_SITES + 1):
        site_mask = site_id_per_sample == sid
        for h in EVAL_HORIZONS:
            p = preds_3d[site_mask, h, tgt_idx]
            y = true_3d[site_mask, h, tgt_idx]
            if len(y) == 0:
                continue
            mae  = mean_absolute_error(y, p)
            rmse = np.sqrt(mean_squared_error(y, p))
            r2   = r2_score(y, p)
            site_results[tgt].append({
                "site": sid, "horizon": h + 1,
                "mae": mae, "rmse": rmse, "r2": r2,
            })
            print(f"    {sid:<6}  t+{h+1:02d}h       "
                  f"{mae:>8.3f}  {rmse:>8.3f}  {r2:>8.4f}")

# ── 11. PER-SEASON BREAKDOWN (match XGBoost format) ───────────────────────────
def month_to_season(m):
    if m in (12, 1, 2):  return "Winter"
    if m in (3, 4, 5):   return "Summer"
    if m in (6, 7, 8, 9): return "Monsoon"
    return "Post-monsoon"  # 10, 11

seasons_per_sample = np.array([month_to_season(pd.Timestamp(d).month)
                                for d in dt_per_sample])

print("\n[11] Per-season breakdown (test set) ...")
season_results = {}
for tgt_idx, tgt in enumerate(TARGETS):
    season_results[tgt] = []
    print(f"\n    {tgt} — error by season (test set):")
    print(f"    {'Season':<14}  {'Horizon':<10}  {'MAE':>8}  {'RMSE':>8}  {'R²':>8}")
    print("    " + "-" * 56)
    for season in ["Winter", "Summer", "Monsoon", "Post-monsoon"]:
        season_mask = seasons_per_sample == season
        for h in EVAL_HORIZONS:
            p = preds_3d[season_mask, h, tgt_idx]
            y = true_3d[season_mask, h, tgt_idx]
            if len(y) == 0:
                continue
            mae  = mean_absolute_error(y, p)
            rmse = np.sqrt(mean_squared_error(y, p))
            r2   = r2_score(y, p)
            season_results[tgt].append({
                "season": season, "horizon": h + 1,
                "mae": mae, "rmse": rmse, "r2": r2,
            })
            print(f"    {season:<14}  t+{h+1:02d}h       "
                  f"{mae:>8.3f}  {rmse:>8.3f}  {r2:>8.4f}")

# ── 12. SAVE ARTIFACTS (ensemble-ready) ───────────────────────────────────────
print("\n[12] Saving artifacts ...")

# (a) Per-horizon pkl predictions (N, 24) per target — for ensemble blending
for tgt_idx, tgt in enumerate(TARGETS):
    joblib.dump(preds_3d[:, :, tgt_idx],
                f"exp09v3b_bilstm_{tgt}_preds.pkl")
    joblib.dump(results[tgt],
                f"exp09v3b_bilstm_{tgt}_results.pkl")
print("    Saved → exp09v3b_bilstm_{NO2_target,O3_target}_{preds,results}.pkl")

# (b) Per-horizon results CSV
for tgt in TARGETS:
    rows = [{"target": tgt, "horizon": h + 1,
             "mae":  results[tgt][h]["mae"],
             "rmse": results[tgt][h]["rmse"],
             "r2":   results[tgt][h]["r2"]}
            for h in range(FORECAST_LEN)]
    pd.DataFrame(rows).to_csv(
        f"exp09v3b_bilstm_{tgt}_results.csv", index=False
    )
print("    Saved → exp09v3b_bilstm_{NO2_target,O3_target}_results.csv")

# (c) Per-site and per-season CSVs
for tgt in TARGETS:
    pd.DataFrame(site_results[tgt]).to_csv(
        f"exp09v3b_bilstm_{tgt}_error_by_site.csv", index=False
    )
    pd.DataFrame(season_results[tgt]).to_csv(
        f"exp09v3b_bilstm_{tgt}_error_by_season.csv", index=False
    )
print("    Saved → exp09v3b_bilstm_{NO2_target,O3_target}_error_by_{site,season}.csv")

# (d) Long-format predictions CSV (site_id, datetime, horizon, target, actual, predicted)
# — directly comparable to exp09v2_*_predictions.csv for ensemble
for tgt_idx, tgt in enumerate(TARGETS):
    long_rows = []
    for i in range(N):
        for h in range(FORECAST_LEN):
            long_rows.append({
                "site_id":   int(site_id_per_sample[i]),
                "datetime":  pd.Timestamp(dt_per_sample[i]).isoformat(),
                "horizon":   h + 1,
                "target":    tgt,
                "actual":    float(true_3d[i, h, tgt_idx]),
                "predicted": float(preds_3d[i, h, tgt_idx]),
            })
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(f"exp09v3b_{tgt}_predictions.csv", index=False)
    print(f"    Saved → exp09v3b_{tgt}_predictions.csv  "
          f"({len(long_df):,} rows)")

# (e) Scalers (unconditional, once at the end)
joblib.dump(feat_scaler,   "exp09v3b_feat_scaler.pkl")
joblib.dump(target_scaler, "exp09v3b_target_scaler.pkl")
print("    Saved → exp09v3b_{feat,target}_scaler.pkl")

# ── 13. XGBOOST vs BiLSTM SIDE-BY-SIDE ────────────────────────────────────────
print("\n" + "=" * 65)
print("  EXP_09 v2 (XGBoost)  vs  EXP_09 v3b (BiLSTM, autoregressive)")
print("=" * 65)
XGB_RESULTS = {
    "NO2_target": {0:0.7958, 5:0.6357, 11:0.6193, 23:0.5889},
    "O3_target":  {0:0.8671, 5:0.7564, 11:0.7523, 23:0.7415},
}
print(f"  {'Target':<12}  {'Horizon':<10}  {'XGB R²':>8}  {'LSTM R²':>8}  {'Best':>6}")
print("  " + "-" * 56)
for tgt in TARGETS:
    for h in EVAL_HORIZONS:
        xgb_r2 = XGB_RESULTS[tgt][h]
        lstm_r2 = results[tgt][h]["r2"]
        best = "LSTM" if lstm_r2 > xgb_r2 else "XGB"
        print(f"  {tgt:<12}  t+{h+1:02d}h       "
              f"{xgb_r2:>8.4f}  {lstm_r2:>8.4f}  {best:>6}")
print("=" * 65)

print(f"\nDone — EXP_09 v3b complete.")
print(f"  Model checkpoint: exp09v3b_bilstm_best.pt")
print(f"  Predictions CSV : exp09v3b_{{NO2,O3}}_target_predictions.csv")
print(f"  Ensemble hint   : 0.5*XGB + 0.5*LSTM at each horizon, then R².")
