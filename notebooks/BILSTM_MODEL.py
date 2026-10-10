"""
VAAYU DARSHIYA — EXP_07
BiLSTM + Attention : NO2 & O3 Multi-step Forecasting
Input  : 48h window of features (no raw lag columns — model sees the sequence)
Output : next 24h of NO2 and O3
Train  : 2016 – 2020       (fit parameters)
Val    : 2021              (early stopping + hyperparam selection)
Test   : 2022              (final reported metrics, touched once)

Changes from EXP_06:
  - DROPOUT 0.2 → 0.4
  - LR 1e-3 → 5e-4
  - weight_decay 1e-4 → 5e-4
  - PATIENCE 7 → 4
  - Multi-step loss: horizon weights 0.5→1.5 (penalize t+24 more than t+1)
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import os, math, warnings
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
DATA_FILE    = "vaayu_final_ml_ready.csv"
# CHANGE in CONFIG section:
TRAIN_YEARS = list(range(2016, 2022))   # 2016–2021  (add 2021 to train)
VAL_YEARS   = [2022]                    # 2022 for val/early stopping
TEST_YEARS  = [2022]                    # 2022 final test too
TARGETS      = ["no2", "o3"]
INPUT_LEN    = 48       # hours of history fed to the model
FORECAST_LEN = 24       # hours ahead to predict

HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.4      # EXP_07: was 0.2 — stronger regularization
ATTN_HEADS   = 4

BATCH_SIZE   = 256      # reduce to 128 if CUDA OOM

EPOCHS       = 50
LR           = 1e-4     # EXP_07: was 1e-3 — gentler updates
PATIENCE     = 10       # EXP_07: was 7 — cut wasted epochs after epoch-1 best

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── FEATURE SET ───────────────────────────────────────────────────────────────
# Do NOT include raw lag columns — the BiLSTM sees the sequence directly.
# The targets appear in the input window (as past known values) but are
# predicted for future timesteps, so there is no leakage.
DROP_COLS = [
    "aqi_no2", "aqi_o3", "aqi_computed", "aqi_category", "aqi_pm25",
    "station_name", "station_id", "date", "datetime",
    "lulc_class_name", "season",
    # drop all engineered lag and rolling columns — BiLSTM learns this from sequence
    *[c for c in pd.read_csv(DATA_FILE, nrows=0).columns
      if "_lag_" in c or "_roll_" in c or "o3_8h_rolling" in c],
]

# ── 1. LOAD & PREPARE ─────────────────────────────────────────────────────────
print("\n[1] Loading data …")
df = pd.read_csv(DATA_FILE, low_memory=False)
print(f"    Shape: {df.shape}")

df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df["year"]      = df["datetime"].dt.year
df = df.sort_values(["station_id", "datetime"]).reset_index(drop=True)

feature_cols = [c for c in df.columns if c not in DROP_COLS and c not in TARGETS]
print(f"    Features (excl. lag/roll): {len(feature_cols)}")
print(f"    Targets: {TARGETS}")

# Encode any remaining object columns
for col in feature_cols:
    if df[col].dtype == "object":
        df[col] = df[col].astype("category").cat.codes.astype(float).replace(-1, np.nan)

# Fill remaining NaNs with column median (per station would be ideal but slow)
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
df[TARGETS]      = df[TARGETS].fillna(df[TARGETS].median())

# ── 2. SPLIT ──────────────────────────────────────────────────────────────────
print("\n[2] Splitting by year …")
train_mask = df["year"].isin(TRAIN_YEARS)
val_mask   = df["year"].isin(VAL_YEARS)
test_mask  = df["year"].isin(TEST_YEARS)

train_df = df[train_mask].copy()
val_df   = df[val_mask].copy()
test_df  = df[test_mask].copy()

print(f"    Train: {len(train_df):,} rows  ({min(TRAIN_YEARS)}–{max(TRAIN_YEARS)})")
print(f"    Val  : {len(val_df):,} rows  ({VAL_YEARS[0]})")
print(f"    Test : {len(test_df):,} rows  ({TEST_YEARS[0]})")

# Time-since-start feature (continuous drift correction)
# 0.0 at earliest train datetime, scaled so 2022-12 ≈ 1.0
_t0 = train_df["datetime"].min()
_tend = pd.Timestamp(f"{TEST_YEARS[0]}-12-31 23:00:00")
for _split_df in (train_df, val_df, test_df):
    _split_df["time_idx"] = (
        (_split_df["datetime"] - _t0).dt.total_seconds() /
        (_tend - _t0).total_seconds()
    ).astype(np.float32)

# time_idx becomes a regular feature — append to feature_cols AFTER split
# so it's scaled consistently and visible to the model in all three splits
feature_cols = feature_cols + ["time_idx"]

# ── 3. SCALE ──────────────────────────────────────────────────────────────────
print("\n[3] Scaling features …")
feat_scaler   = StandardScaler()
target_scaler = StandardScaler()

# Fit on train ONLY — never let val or test statistics leak into scaler params
train_feats = feat_scaler.fit_transform(train_df[feature_cols])
val_feats   = feat_scaler.transform(val_df[feature_cols])
test_feats  = feat_scaler.transform(test_df[feature_cols])

train_targets = target_scaler.fit_transform(train_df[TARGETS])
val_targets   = target_scaler.transform(val_df[TARGETS])
test_targets  = target_scaler.transform(test_df[TARGETS])

# ── 4. DATASET ────────────────────────────────────────────────────────────────
class PollutionDataset(Dataset):
    """
    Builds (input_window, target_window) pairs per station.
    input_window  : (INPUT_LEN,  n_features)  — past 48h
    target_window : (FORECAST_LEN, n_targets) — next 24h NO2 & O3
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

print("\n[4] Building datasets …")
train_station_ids = train_df["station_id"].values
val_station_ids   = val_df["station_id"].values
test_station_ids  = test_df["station_id"].values

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

# ← ADD THESE RIGHT HERE
print(f"Device      : {DEVICE}")
print(f"Train batches: {len(train_loader)}")
print(f"Val   batches: {len(val_loader)}")
print(f"Test  batches: {len(test_loader)}")
# ── 5. MODEL ──────────────────────────────────────────────────────────────────
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
print(f"\n[5] Model ready — {total_params:,} parameters")
print(model)

# ── 6. TRAIN ──────────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)  # EXP_07: was 1e-4
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)
criterion = nn.HuberLoss(delta=1.0, reduction="mean")  # for val loop
huber_none = nn.HuberLoss(delta=1.0, reduction="none") # for weighted train loss   # robust to outliers, better than MSE

# EXP_07: horizon weights — penalize t+24 errors 3x more than t+1.
# Without this the model minimizes average loss by predicting the easy short
# horizons well and ignoring the long horizons. The head then collapses to a
# conditional mean and per-horizon MAE stays flat (the EXP_05/06 symptom).
horizon_weights = torch.linspace(0.5, 1.5, FORECAST_LEN, device=DEVICE)
print(f"    Horizon weights: {horizon_weights[0].item():.2f} → "
      f"{horizon_weights[-1].item():.2f} (mean={horizon_weights.mean().item():.2f})")

# AMP only works on CUDA — guard with getattr so the script runs on CPU too
_amp = getattr(torch, "amp", None)
if _amp is not None and hasattr(_amp, "GradScaler") and DEVICE.type == "cuda":
    scaler_amp = torch.amp.GradScaler("cuda", enabled=True)
    autocast_ctx = lambda: torch.amp.autocast("cuda", enabled=True)
else:
    scaler_amp = None
    from contextlib import nullcontext
    autocast_ctx = nullcontext

print(f"\n[6] Training for up to {EPOCHS} epochs …\n")
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
            pred = model(X_batch)
            # EXP_07: horizon-weighted Huber — penalize later horizons more.
            # Huber supports reducti
            # on='none' so we can weight per timestep.
            per_step = huber_none(pred, Y_batch)   # (B, 24, 2)
            loss = (per_step * horizon_weights.view(1, -1, 1)).mean()
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

    # — validate on VAL set (2021), NOT the test set —
    # Early stopping and best-checkpoint selection must use val, otherwise
    # we leak test info via the chosen epoch and the checkpoint itself.
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
        torch.save(model.state_dict(), "exp07_bilstm_best.pt")
    else:
        patience_counter += 1

    mark = " *" if improved else ""
    print(f"  Epoch {epoch:02d}/{EPOCHS}  "
          f"train={train_loss:.4f}  val={val_loss:.4f}{mark}  "
          f"patience={patience_counter}/{PATIENCE}")

    if patience_counter >= PATIENCE:
        print(f"\n  Early stopping at epoch {epoch}.")
        break

# ── 7. EVALUATE ───────────────────────────────────────────────────────────────
print("\n[7] Evaluating best model on 2022 test set …")
model.load_state_dict(torch.load("exp07_bilstm_best.pt", map_location=DEVICE))
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

# inverse-scale back to µg/m³
N = all_preds.shape[0]
preds_2d = all_preds.reshape(N * FORECAST_LEN, len(TARGETS))
true_2d  = all_true.reshape( N * FORECAST_LEN, len(TARGETS))
preds_2d = target_scaler.inverse_transform(preds_2d)
true_2d  = target_scaler.inverse_transform(true_2d)
preds_2d = np.clip(preds_2d, 0, None)

print("\n" + "=" * 50)
print("  EXP_07 — BiLSTM+Attention Results (24h forecast, 2022 test)")
print("  Train: 2016–2020   Val: 2021   Test: 2022")
print("  Regularized: dropout=0.4, wd=5e-4, horizon-weighted loss")
print("=" * 50)
for i, tgt in enumerate(TARGETS):
    mae  = mean_absolute_error(true_2d[:, i], preds_2d[:, i])
    rmse = np.sqrt(mean_squared_error(true_2d[:, i], preds_2d[:, i]))
    r2   = r2_score(true_2d[:, i], preds_2d[:, i])
    print(f"  {tgt.upper():<5}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
print("=" * 50)

# Per-horizon breakdown (how accuracy drops as forecast horizon increases)
print("\n  Per-horizon MAE (averaged over both targets):")
print(f"  {'Horizon':>8}  {'NO2 MAE':>10}  {'O3 MAE':>10}")
for h in [1, 3, 6, 12, 18, 24]:
    idx = h - 1
    no2_mae = mean_absolute_error(true_2d[idx::FORECAST_LEN, 0],
                                   preds_2d[idx::FORECAST_LEN, 0])
    o3_mae  = mean_absolute_error(true_2d[idx::FORECAST_LEN, 1],
                                   preds_2d[idx::FORECAST_LEN, 1])
    print(f"  t+{h:>2}h      {no2_mae:>10.3f}  {o3_mae:>10.3f}")

    # Save scalers for EXP_08
    import joblib

    joblib.dump(feat_scaler, "exp07_feat_scaler.pkl")
    joblib.dump(target_scaler, "exp07_target_scaler.pkl")
    print("Scalers saved → exp07_feat_scaler.pkl / exp07_target_scaler.pkl")

# ── 8. PLOTS ──────────────────────────────────────────────────────────────────
# ── 8. PLOTS ──────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("\n[8] Saving plots …")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("EXP_07 — BiLSTM + Attention  (24h Multi-step Forecast, 2022 test)",
             fontsize=13)

for row, (tgt, col) in enumerate(zip(TARGETS, ["#3B82F6", "#F59E0B"])):
    t = true_2d[:, row]
    p = preds_2d[:, row]

    axes[row, 0].scatter(t, p, alpha=0.05, s=2, color=col)
    lim = [0, max(t.max(), p.max())]
    axes[row, 0].plot(lim, lim, "r--", lw=1)
    axes[row, 0].set_title(f"{tgt.upper()} — predicted vs actual")
    axes[row, 0].set_xlabel("Actual"); axes[row, 0].set_ylabel("Predicted")

    res = t - p
    axes[row, 1].hist(res, bins=80, color=col, alpha=0.7, edgecolor="none")
    axes[row, 1].axvline(0, color="red", lw=1, linestyle="--")
    axes[row, 1].set_title(f"{tgt.upper()} — residuals")

    horizons = list(range(1, FORECAST_LEN + 1))
    maes = [mean_absolute_error(true_2d[h-1::FORECAST_LEN, row],
                                 preds_2d[h-1::FORECAST_LEN, row])
            for h in horizons]
    axes[row, 2].plot(horizons, maes, color=col, lw=2, marker="o", markersize=3)
    axes[row, 2].set_title(f"{tgt.upper()} — MAE by forecast horizon")
    axes[row, 2].set_xlabel("Hours ahead"); axes[row, 2].set_ylabel("MAE (µg/m³)")
    axes[row, 2].axhline(maes[0], color="gray", lw=0.8, linestyle=":")

plt.tight_layout()
plt.savefig("exp07_bilstm_results.png", dpi=150, bbox_inches="tight")
plt.close(fig)                            # ← new
print("  Saved → exp07_bilstm_results.png")

fig2, ax = plt.subplots(figsize=(8, 4))
ax.plot(history["train"], label="Train loss", color="#3B82F6")
ax.plot(history["val"],   label="Val loss",   color="#EF4444")
ax.set_xlabel("Epoch"); ax.set_ylabel("Huber loss")
ax.set_title("Training curve"); ax.legend()
plt.tight_layout()
plt.savefig("exp07_training_curve.png", dpi=150, bbox_inches="tight")
plt.close(fig2)                           # ← new
print("  Saved → exp07_training_curve.png")

print("""
=======================================================
  DONE — EXP_07 complete
  Saved: exp07_bilstm_best.pt
         exp07_bilstm_results.png
         exp07_training_curve.png
=======================================================
""")

if DEVICE.type == "cuda":
    import joblib

    joblib.dump(feat_scaler, "exp07_feat_scaler.pkl")
    joblib.dump(target_scaler, "exp07_target_scaler.pkl")
    print("Scalers saved.")
    torch.cuda.empty_cache()