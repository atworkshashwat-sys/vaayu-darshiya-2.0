"""
VAAYU DARSHIYA — EXP_09 v4
BiLSTM + Attention : NO2 & O3 Multi-step Forecasting

Changelog vs v3 (root-cause fixes for R2 ~ 0 / negative):

  [FIX-1] ROOT CAUSE: v3 built every lag/rolling feature from *_forecast
          (the NWP model output) and never from *_target (the ground truth
          you're actually predicting). XGBoost (exp09v2) uses lags of
          NO2_target/O3_target up to 72h back plus rolling means up to 168h —
          that's the dominant signal XGBoost was learning from and the LSTM
          never saw. This version builds the same target-derived lag/rolling
          features, using shift(1) before rolling so no current/future target
          leaks into the window (matches exp09v2 exactly).
  [FIX-2] Added wind_speed, wind_dir_sin/cos, doy_sin/cos to match exp09v2's
          feature set (previously LSTM-only had hour/month cyclic + lags).
  [FIX-3] CRITICAL DATA BUG: v3's PollutionDataset built input windows that
          could span across DIFFERENT SITES / contain internal time gaps.
          This version explicitly checks datetime contiguity per window and
          drops any window with a gap.
  [FIX-5] Model: real additive (Bahdanau-style) temporal attention over all
          48 encoder steps, not just the last hidden state.
  [FIX-6] Regularization de-coupled from "match XGBoost" and re-grounded in
          actual validation behavior.
  [FIX-7] Train/val/test sanity check added (matches exp09v2 step 10).
  [FIX-8] Fixed shape mismatch building the long-format prediction CSV
          (per-sample arrays vs per-sample*horizon arrays).
  [FIX-9] Per-horizon output heads instead of one shared head, mirroring
          XGBoost's 24-independent-models-per-target design.
  [FIX-10] Uniform horizon loss weighting (no more down-weighting h0).
  [FIX-11] Regularization raised based on the observed train>>val>>test gap
          from the real run (dropout 0.2->0.35, weight_decay 1e-5->3e-4).
  [FIX-12] MODEL_TYPE switch added: "bilstm_attn" vs "mlp_ablation", to test
          whether the BiLSTM+attention encoder earns its complexity over a
          plain MLP on the last timestep's features alone. A shared
          build_model() function is now the single source of truth for
          model construction so the training loop and the sanity-check
          reload can never use mismatched architecture args.
  [FIX-13] BUG: section [9]'s sanity check built a FRESH, randomly
          initialized model and evaluated it WITHOUT loading the trained
          checkpoint. Every train/val/test R2 printed in that section was
          therefore meaningless (a random model's R2), not a measure of the
          actual trained model. Fixed by reloading
          exp09v4_{RUN_TAG}_{target_name}_best.pt before evaluating, same
          checkpoint section 8 already evaluated on the test set, so these
          numbers are now directly comparable.
  [FIX-14] RUN_TAG (= MODEL_TYPE) added: every saved file (scalers,
          checkpoints, metrics CSVs/PKLs, prediction CSVs, plots) is now
          namespaced by RUN_TAG so running "mlp_ablation" and then
          "bilstm_attn" back-to-back doesn't overwrite either run's outputs.
"""
import os, sys, warnings, joblib
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
EVAL_HORIZONS   = [0, 5, 11, 23]

NWP_FEATURES    = ["O3_forecast", "NO2_forecast", "T_forecast",
                    "q_forecast", "u_forecast", "v_forecast", "w_forecast"]
DROP_COLS       = ["NO2_satellite", "HCHO_satellite", "ratio_satellite"]
TARGET_LAG_HOURS = [1, 2, 3, 6, 12, 24, 48, 72]   # matches exp09v2
TARGET_ROLL_WINDOWS = [6, 12, 24, 48, 72, 168]    # matches exp09v2

INPUT_LEN       = 48
FORECAST_LEN    = 24

HIDDEN_SIZE     = 64
NUM_LAYERS      = 1
DROPOUT         = 0.35         # [FIX-11] was 0.2 (v4) / 0.5 (v3) — raised
                                # based on the observed train R2=0.85 vs
                                # test R2=0.52-0.61 gap, not to force parity.
EMBED_DIM       = 8
ATTN_DIM        = 32           # [FIX-5] attention projection size
HEAD_HIDDEN     = 32            # [FIX-9] per-horizon head width

# [FIX-12] Flip this to switch which model architecture trains. Run it once
# as "mlp_ablation", then again as "bilstm_attn" -- RUN_TAG (below) keeps
# every output file from the two runs separate so you can compare them.
MODEL_TYPE      = "mlp_ablation"   # "bilstm_attn" or "mlp_ablation"
RUN_TAG         = MODEL_TYPE        # [FIX-14] namespaces all saved filenames

BATCH_SIZE      = 256
EPOCHS          = 100
LR              = 1e-3
PATIENCE        = 25            # [FIX-11] was 20
WEIGHT_DECAY    = 3e-4          # [FIX-11] was 1e-5 (v4) / 5e-4 (v3)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"Model type: {MODEL_TYPE}  (RUN_TAG={RUN_TAG})")

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
print("\n[1] Loading ...")
all_dfs = []
for i in range(1, N_SITES + 1):
    df_s = pd.read_csv(f"site_{i}_train_data.csv")
    df_s["site_id"] = i
    df_s = df_s.drop(columns=DROP_COLS, errors="ignore")
    all_dfs.append(df_s)
    print(f"    Site {i}: {len(df_s):,} rows")
df = pd.concat(all_dfs, ignore_index=True)

# ── 2. SORT ───────────────────────────────────────────────────────────────────
print("\n[2] Sorting ...")
df["datetime"] = pd.to_datetime(dict(
    year=df["year"], month=df["month"], day=df["day"], hour=df["hour"]))
df = df.sort_values(["site_id", "datetime"]).reset_index(drop=True)
df = df.dropna(subset=TARGETS).reset_index(drop=True)

# ── 3. FEATURES ───────────────────────────────────────────────────────────────
print("\n[3] Features (now matching exp09v2 feature set) ...")

# Cyclic time
df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24).astype(np.float32)
df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24).astype(np.float32)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype(np.float32)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype(np.float32)
df["dow_sin"]   = np.sin(2 * np.pi * df["datetime"].dt.dayofweek / 7).astype(np.float32)
df["dow_cos"]   = np.cos(2 * np.pi * df["datetime"].dt.dayofweek / 7).astype(np.float32)
df["day_of_year"] = df["datetime"].dt.dayofyear
df["doy_sin"]   = np.sin(2 * np.pi * df["day_of_year"] / 365.25).astype(np.float32)  # [FIX-2]
df["doy_cos"]   = np.cos(2 * np.pi * df["day_of_year"] / 365.25).astype(np.float32)  # [FIX-2]

# Wind speed + direction  [FIX-2]
df["wind_speed"]   = np.sqrt(df["u_forecast"]**2 + df["v_forecast"]**2).astype(np.float32)
df["wind_dir"]     = np.arctan2(df["v_forecast"], df["u_forecast"])
df["wind_dir_sin"] = np.sin(df["wind_dir"]).astype(np.float32)
df["wind_dir_cos"] = np.cos(df["wind_dir"]).astype(np.float32)

# [FIX-1] TARGET-DERIVED lag features — this is the signal v3 was missing.
# shift() only looks backward, so no leakage of current/future target.
target_lag_cols = []
for col in TARGETS:
    for lag in TARGET_LAG_HOURS:
        cname = f"{col}_lag{lag}"
        df[cname] = df.groupby("site_id")[col].shift(lag)
        target_lag_cols.append(cname)

# [FIX-1] TARGET-DERIVED rolling features, shift(1) before rolling so the
# window never includes the current timestep (matches exp09v2 exactly).
target_roll_cols = []
for col in TARGETS:
    for w in TARGET_ROLL_WINDOWS:
        cname = f"{col}_roll{w}"
        df[cname] = df.groupby("site_id")[col].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        target_roll_cols.append(cname)

# NWP forecast lags (kept from v3 — still useful as auxiliary signal)
nwp_lag_cols = []
for col in NWP_FEATURES:
    for lag in [1, 2, 3, 6, 12, 24]:
        cname = f"{col}_lag{lag}"
        df[cname] = df.groupby("site_id")[col].shift(lag)
        nwp_lag_cols.append(cname)

feature_cols = (
    NWP_FEATURES
    + ["hour_sin", "hour_cos", "month_sin", "month_cos",
       "dow_sin", "dow_cos", "doy_sin", "doy_cos",
       "wind_speed", "wind_dir_sin", "wind_dir_cos"]
    + target_lag_cols
    + target_roll_cols
    + nwp_lag_cols
)
print(f"    {len(feature_cols)} features ({len(target_lag_cols)} target-lag, "
      f"{len(target_roll_cols)} target-roll, {len(nwp_lag_cols)} nwp-lag)")

# [FIX-1] Drop rows with any NaN in the new lag/rolling columns instead of
# mean-filling them. Matches exp09v2's dropna() behavior on lag_roll_cols.
before = len(df)
df = df.dropna(subset=feature_cols).reset_index(drop=True)
print(f"    Dropped {before - len(df):,} rows with incomplete lag/rolling history "
      f"({before:,} -> {len(df):,})")

# ── 4. SPLIT + SCALE ─────────────────────────────────────────────────────────
print("\n[4] Split + Scale ...")
train_df = df[df["year"].isin(TRAIN_YEARS)].copy().reset_index(drop=True)
val_df   = df[df["year"].isin(VAL_YEARS)].copy().reset_index(drop=True)
test_df  = df[df["year"].isin(TEST_YEARS)].copy().reset_index(drop=True)
print(f"    Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

feat_scaler   = StandardScaler()
target_scaler = StandardScaler()

train_feats = feat_scaler.fit_transform(train_df[feature_cols])
val_feats   = feat_scaler.transform(val_df[feature_cols])
test_feats  = feat_scaler.transform(test_df[feature_cols])

joblib.dump(feat_scaler,   f"exp09v4_{RUN_TAG}_feat_scaler.pkl")     # [FIX-14]
joblib.dump(feature_cols,  f"exp09v4_{RUN_TAG}_feature_cols.pkl")    # [FIX-14]

train_targets_scaled = target_scaler.fit_transform(train_df[TARGETS].values)
val_targets_scaled   = target_scaler.transform(val_df[TARGETS].values)
test_targets_scaled  = target_scaler.transform(test_df[TARGETS].values)
joblib.dump(target_scaler, f"exp09v4_{RUN_TAG}_target_scaler.pkl")    # [FIX-14]

# ── 5. DATASET ────────────────────────────────────────────────────────────────
class PollutionDataset(Dataset):
    """
    Builds (input_len -> forecast_len) windows per site.

    [FIX-3] Each candidate window is checked for datetime contiguity across
    the full input+forecast span (1-hour spacing, no gaps). Windows that
    straddle a gap are silently skipped instead of silently trained on as
    if they were continuous.
    """
    def __init__(self, feats, targets, station_ids, datetimes, input_len, forecast_len):
        self.X, self.Y = [], []
        self.site_ids = []
        self.anchor_datetimes = []
        self.forecast_start_datetimes = []
        total_span = input_len + forecast_len
        expected_step = np.timedelta64(1, "h")
        skipped_gaps = 0

        for sid in np.unique(station_ids):
            idx = np.where(station_ids == sid)[0]
            f, t, dt = feats[idx], targets[idx], datetimes[idx]
            for i in range(len(idx) - total_span + 1):
                window_dt = dt[i:i + total_span]
                diffs = np.diff(window_dt)
                if not np.all(diffs == expected_step):
                    skipped_gaps += 1
                    continue
                self.X.append(f[i:i + input_len])
                self.Y.append(t[i + input_len:i + input_len + forecast_len])
                self.site_ids.append(sid)
                self.anchor_datetimes.append(dt[i + input_len - 1])
                self.forecast_start_datetimes.append(dt[i + input_len])

        self.X = np.array(self.X, dtype=np.float32)
        self.Y = np.array(self.Y, dtype=np.float32)
        self.site_ids = np.array(self.site_ids, dtype=np.int64)
        self.anchor_datetimes = np.array(self.anchor_datetimes, dtype="datetime64[ns]")
        self.forecast_start_datetimes = np.array(self.forecast_start_datetimes, dtype="datetime64[ns]")
        print(f"    {len(self):,} samples  X={self.X.shape}  Y={self.Y.shape}  "
              f"(skipped {skipped_gaps:,} windows with time gaps)")

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i], self.site_ids[i]

print("\n[5] Datasets ...")
train_ds = PollutionDataset(train_feats, train_targets_scaled,
                            train_df["site_id"].values, train_df["datetime"].values, INPUT_LEN, FORECAST_LEN)
val_ds = PollutionDataset(val_feats, val_targets_scaled,
                          val_df["site_id"].values, val_df["datetime"].values, INPUT_LEN, FORECAST_LEN)
test_ds = PollutionDataset(test_feats, test_targets_scaled,
                           test_df["site_id"].values, test_df["datetime"].values, INPUT_LEN, FORECAST_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── 6. MODEL ──────────────────────────────────────────────────────────────────
class BiLSTMAttnForecaster(nn.Module):
    """
    [FIX-5] Real additive (Bahdanau-style) attention over all encoder steps.
    [FIX-9] Per-horizon output heads instead of one shared Linear(64, 24),
    mirroring XGBoost's 24-independent-models-per-target design.
    """
    def __init__(self, n_features, hidden, forecast_len,
                 n_stations=7, embed_dim=8, attn_dim=32, dropout=0.2,
                 head_hidden=48):
        super().__init__()
        self.forecast_len = forecast_len
        self.station_embed = nn.Embedding(n_stations + 1, embed_dim)
        input_dim = n_features + embed_dim

        self.bilstm = nn.LSTM(input_dim, hidden, num_layers=1,
                              batch_first=True, bidirectional=True)

        h2 = hidden * 2
        self.attn_query = nn.Parameter(torch.randn(attn_dim) * 0.01)
        self.attn_proj   = nn.Linear(h2, attn_dim)
        self.attn_score  = nn.Linear(attn_dim, 1, bias=False)

        self.norm = nn.LayerNorm(h2)
        self.dropout = nn.Dropout(dropout)

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h2, head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, 1),
            )
            for _ in range(forecast_len)
        ])

    def forward(self, x, sids):
        B, T, _ = x.shape
        emb = self.station_embed(sids).unsqueeze(1).expand(-1, T, -1)
        x = torch.cat([x, emb], dim=-1)
        out, _ = self.bilstm(x)                      # (B, T, h2)

        proj = torch.tanh(self.attn_proj(out))        # (B, T, attn_dim)
        scores = self.attn_score(proj).squeeze(-1)    # (B, T)
        weights = torch.softmax(scores, dim=1)        # (B, T)
        ctx = torch.bmm(weights.unsqueeze(1), out).squeeze(1)  # (B, h2)

        ctx = self.norm(ctx)
        ctx = self.dropout(ctx)

        per_horizon = [head(ctx) for head in self.heads]   # list of (B, 1)
        return torch.cat(per_horizon, dim=1)                # (B, forecast_len)


class LastStepMLPForecaster(nn.Module):
    """
    Ablation: ignores the 48-step window entirely, uses only the LAST
    timestep's features (the anchor hour) through a plain MLP with the
    same per-horizon heads as the BiLSTM model. If this gets comparable
    R2 to the full BiLSTM+attention model, the sequence modeling isn't
    earning its complexity -- the lag/rolling features already contain
    the temporal signal, and the real lever is MLP capacity, not
    recurrence/attention.
    """
    def __init__(self, n_features, forecast_len, n_stations=7,
                 embed_dim=8, dropout=0.35, trunk_hidden=128, head_hidden=32):
        super().__init__()
        self.station_embed = nn.Embedding(n_stations + 1, embed_dim)
        input_dim = n_features + embed_dim

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, trunk_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(trunk_hidden, trunk_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(trunk_hidden)

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(trunk_hidden, head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, 1),
            )
            for _ in range(forecast_len)
        ])

    def forward(self, x, sids):
        last_step = x[:, -1, :]                      # (B, n_features) -- only the anchor hour
        emb = self.station_embed(sids)                # (B, embed_dim)
        h = torch.cat([last_step, emb], dim=-1)
        h = self.trunk(h)
        h = self.norm(h)
        per_horizon = [head(h) for head in self.heads]
        return torch.cat(per_horizon, dim=1)


def build_model():
    """[FIX-12] Single source of truth for model construction, so the
    training loop and the sanity-check reload can never use mismatched
    architecture args (which would either crash load_state_dict on a shape
    mismatch, or -- worse -- silently succeed with the wrong weights)."""
    if MODEL_TYPE == "bilstm_attn":
        return BiLSTMAttnForecaster(n_features, HIDDEN_SIZE, FORECAST_LEN,
                                    N_SITES, EMBED_DIM, ATTN_DIM, DROPOUT,
                                    HEAD_HIDDEN).to(DEVICE)
    else:
        return LastStepMLPForecaster(n_features, FORECAST_LEN, N_SITES,
                                     EMBED_DIM, DROPOUT, head_hidden=HEAD_HIDDEN).to(DEVICE)


n_features = train_ds.X.shape[2]
print(f"\n[6] Model ...")

# ── 7. TRAIN (separate per target) ───────────────────────────────────────────
# [FIX-10] Uniform horizon weighting -- each horizon now has its own
# independent head (FIX-9), so there's no shared-capacity tradeoff left to
# manage via loss weighting. Down-weighting h0 only hurt it for no benefit.
horizon_weights = torch.ones(FORECAST_LEN, device=DEVICE)
huber = nn.HuberLoss(delta=1.0, reduction="none")

results = {}
preds_by_target = {}
true_by_target = {}

for target_name in TARGETS:
    tgt_idx = TARGETS.index(target_name)
    print(f"\n{'='*60}")
    print(f"  Training: {target_name}")
    print(f"{'='*60}")

    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=EPOCHS,
        steps_per_epoch=len(train_loader), pct_start=0.3)

    best_val = float("inf")
    wait = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_loss = 0.0
        for X_b, Y_b, S_b in train_loader:
            X_b = X_b.to(DEVICE)
            Y_b = Y_b[:, :, tgt_idx].to(DEVICE)
            S_b = S_b.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_b, S_b)
            loss = (huber(pred, Y_b) * horizon_weights.unsqueeze(0)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            t_loss += loss.item()
        t_loss /= len(train_loader)

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for X_b, Y_b, S_b in val_loader:
                X_b = X_b.to(DEVICE)
                Y_b = Y_b[:, :, tgt_idx].to(DEVICE)
                S_b = S_b.to(DEVICE)
                pred = model(X_b, S_b)
                v_loss += (huber(pred, Y_b) * horizon_weights.unsqueeze(0)).mean().item()
        v_loss /= len(val_loader)

        improved = v_loss < best_val
        if improved:
            best_val = v_loss
            wait = 0
            torch.save(model.state_dict(), f"exp09v4_{RUN_TAG}_{target_name}_best.pt")  # [FIX-14]
        else:
            wait += 1

        if epoch % 10 == 0 or improved:
            lr_now = optimizer.param_groups[0]["lr"]
            mark = " *" if improved else ""
            print(f"  Ep {epoch:03d}/{EPOCHS}  train={t_loss:.4f}  val={v_loss:.4f}{mark}  lr={lr_now:.2e}")

        if wait >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    # ── 8. EVALUATE ───────────────────────────────────────────────────────────
    print(f"\n  Evaluating {target_name} ...")
    model.load_state_dict(torch.load(f"exp09v4_{RUN_TAG}_{target_name}_best.pt", map_location=DEVICE))  # [FIX-14]
    model.eval()

    all_p, all_y = [], []
    with torch.no_grad():
        for X_b, Y_b, S_b in test_loader:
            pred = model(X_b.to(DEVICE), S_b.to(DEVICE))
            all_p.append(pred.cpu().numpy())
            all_y.append(Y_b[:, :, tgt_idx].numpy())

    all_p = np.concatenate(all_p)
    all_y = np.concatenate(all_y)

    N = all_p.shape[0]
    dummy_p = np.zeros((N * FORECAST_LEN, 2), dtype=np.float32)
    dummy_y = np.zeros((N * FORECAST_LEN, 2), dtype=np.float32)
    dummy_p[:, tgt_idx] = all_p.flatten()
    dummy_y[:, tgt_idx] = all_y.flatten()
    preds_inv = target_scaler.inverse_transform(dummy_p)[:, tgt_idx].clip(0, None)
    true_inv  = target_scaler.inverse_transform(dummy_y)[:, tgt_idx]

    preds_3d = preds_inv.reshape(N, FORECAST_LEN)
    true_3d  = true_inv.reshape(N, FORECAST_LEN)

    print(f"\n  {target_name} Results:")
    print(f"  {'Horizon':<10}  {'MAE':>8}  {'RMSE':>8}  {'R2':>8}")
    print("  " + "-" * 40)

    results[target_name] = {}
    for h in range(FORECAST_LEN):
        mae  = mean_absolute_error(true_3d[:, h], preds_3d[:, h])
        rmse = np.sqrt(mean_squared_error(true_3d[:, h], preds_3d[:, h]))
        r2   = r2_score(true_3d[:, h], preds_3d[:, h])
        results[target_name][h] = {"mae": mae, "rmse": rmse, "r2": r2}
        if h in EVAL_HORIZONS:
            print(f"  t+{h+1:02d}h      {mae:>8.3f}  {rmse:>8.3f}  {r2:>8.4f}")

    preds_by_target[target_name] = preds_3d
    true_by_target[target_name]  = true_3d

    # Per-site
    print(f"\n  {target_name} by site:")
    print(f"  {'Site':<6}  {'Horizon':<10}  {'MAE':>8}  {'RMSE':>8}  {'R2':>8}")
    print("  " + "-" * 50)
    site_res = []
    for sid in range(1, N_SITES + 1):
        mask = test_ds.site_ids == sid
        for h in EVAL_HORIZONS:
            p, y = preds_3d[mask, h], true_3d[mask, h]
            if len(y) == 0: continue
            site_res.append({"site": sid, "horizon": h+1,
                             "mae": mean_absolute_error(y, p),
                             "rmse": np.sqrt(mean_squared_error(y, p)),
                             "r2": r2_score(y, p)})
            print(f"  {sid:<6}  t+{h+1:02d}h      {site_res[-1]['mae']:>8.3f}  {site_res[-1]['rmse']:>8.3f}  {site_res[-1]['r2']:>8.4f}")
    pd.DataFrame(site_res).to_csv(f"exp09v4_{RUN_TAG}_{target_name}_by_site.csv", index=False)  # [FIX-14]

    # Per-season
    def month_to_season(m):
        if m in (12, 1, 2):   return "Winter"
        if m in (3, 4, 5):    return "Summer"
        if m in (6, 7, 8, 9): return "Monsoon"
        return "Post-monsoon"

    seasons = np.array([month_to_season(pd.Timestamp(d).month) for d in test_ds.forecast_start_datetimes])
    print(f"\n  {target_name} by season:")
    season_res = []
    for season in ["Winter", "Summer", "Monsoon", "Post-monsoon"]:
        sm = seasons == season
        for h in EVAL_HORIZONS:
            p, y = preds_3d[sm, h], true_3d[sm, h]
            if len(y) == 0: continue
            season_res.append({"season": season, "horizon": h+1,
                               "mae": mean_absolute_error(y, p),
                               "rmse": np.sqrt(mean_squared_error(y, p)),
                               "r2": r2_score(y, p)})
            print(f"  {season:<14}  t+{h+1:02d}h      {season_res[-1]['mae']:>8.3f}  {season_res[-1]['rmse']:>8.3f}  {season_res[-1]['r2']:>8.4f}")
    pd.DataFrame(season_res).to_csv(f"exp09v4_{RUN_TAG}_{target_name}_by_season.csv", index=False)  # [FIX-14]

    # Save artifacts
    joblib.dump(preds_3d, f"exp09v4_{RUN_TAG}_{target_name}_preds.pkl")              # [FIX-14]
    joblib.dump(results[target_name], f"exp09v4_{RUN_TAG}_{target_name}_results.pkl")  # [FIX-14]
    pd.DataFrame([{"target": target_name, "horizon": h+1, **results[target_name][h]}
                  for h in range(FORECAST_LEN)]).to_csv(
        f"exp09v4_{RUN_TAG}_{target_name}_results.csv", index=False)  # [FIX-14]

    # [FIX-8] site_id / datetime are one-per-SAMPLE (length N); actual/predicted
    # are one-per-(SAMPLE, HORIZON) after flatten() (length N*FORECAST_LEN).
    # np.repeat each per-sample array FORECAST_LEN times so every row lines up.
    long_df = pd.DataFrame({
        "site_id":   np.repeat(test_ds.site_ids, FORECAST_LEN),
        "datetime":  np.repeat(
            [str(pd.Timestamp(d)) for d in test_ds.forecast_start_datetimes],
            FORECAST_LEN),
        "horizon":   np.tile(np.arange(1, FORECAST_LEN + 1), N),
        "target":    target_name,
        "actual":    true_3d.flatten(),
        "predicted": preds_3d.flatten(),
    })
    long_df.to_csv(f"exp09v4_{RUN_TAG}_{target_name}_predictions.csv", index=False)  # [FIX-14]

    # ── PLOTS ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{target_name} - Actual vs Predicted (Test Set) [{RUN_TAG}]", fontsize=14)
    for ax_i, h in enumerate(EVAL_HORIZONS):
        ax = axes[ax_i // 2, ax_i % 2]
        ax.scatter(true_3d[:, h], preds_3d[:, h], alpha=0.1, s=5)
        lims = [0, max(true_3d[:, h].max(), preds_3d[:, h].max()) * 1.05]
        ax.plot(lims, lims, "r--", lw=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"t+{h+1:02d}h  R2={results[target_name][h]['r2']:.4f}")
    plt.tight_layout()
    plt.savefig(f"exp09v4_{RUN_TAG}_{target_name}_actual_vs_predicted.png", dpi=150)  # [FIX-14]
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    for h in EVAL_HORIZONS:
        errors = preds_3d[:, h] - true_3d[:, h]
        ax.hist(errors, bins=80, alpha=0.5, label=f"t+{h+1:02d}h", density=True)
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("Prediction Error")
    ax.set_ylabel("Density")
    ax.set_title(f"{target_name} - Error Distribution [{RUN_TAG}]")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"exp09v4_{RUN_TAG}_{target_name}_error_dist.png", dpi=150)  # [FIX-14]
    plt.close()
    print(f"    Plots saved")

# ── 9. TRAIN/VAL/TEST SANITY CHECK ───────────────────────────────────────────
# [FIX-7] Matches exp09v2 step 10: quick over/underfitting read at horizon 0.
# [FIX-13] BUG FIXED: this section previously built a FRESH, randomly
# initialized model and evaluated it WITHOUT loading the trained checkpoint
# -- every number it printed was for a random model, not the trained one.
# Fixed by reloading exp09v4_{RUN_TAG}_{target_name}_best.pt (the same
# checkpoint section 8 evaluated on the test set) before computing R2 here,
# so these numbers are now directly comparable to the test R2 printed above.
print(f"\n{'='*60}")
print("  [9] Train/Val/Test sanity check (horizon 0, reloaded best model)")
print(f"{'='*60}")
for target_name in TARGETS:
    tgt_idx = TARGETS.index(target_name)
    model = build_model()
    model.load_state_dict(torch.load(f"exp09v4_{RUN_TAG}_{target_name}_best.pt", map_location=DEVICE))  # [FIX-13][FIX-14]
    model.eval()

    def r2_at_h0(loader):
        ps, ys = [], []
        with torch.no_grad():
            for X_b, Y_b, S_b in loader:
                pred = model(X_b.to(DEVICE), S_b.to(DEVICE))[:, 0]
                ps.append(pred.cpu().numpy())
                ys.append(Y_b[:, 0, tgt_idx].numpy())
        return r2_score(np.concatenate(ys), np.concatenate(ps))

    r2_tr  = r2_at_h0(train_loader)
    r2_va  = r2_at_h0(val_loader)
    r2_te  = r2_at_h0(test_loader)
    print(f"  {target_name} horizon 0 (scaled space) — train R2: {r2_tr:.3f}  "
          f"val R2: {r2_va:.3f}  test R2: {r2_te:.3f}")
print("  (train R2 >> val/test R2 = overfitting; all near 0 = underfitting/no signal)")

# ── 10. XGBOOST vs LSTM/MLP ───────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  XGBoost vs {RUN_TAG}")
print(f"{'='*60}")

xgb_r2_file = "exp09v2_xgboost_results.pkl"
if os.path.exists(xgb_r2_file):
    XGB_RESULTS = joblib.load(xgb_r2_file)
else:
    XGB_RESULTS = {
        "NO2_target": {0: 0.7958, 5: 0.6357, 11: 0.6193, 23: 0.5889},
        "O3_target":  {0: 0.8671, 5: 0.7564, 11: 0.7523, 23: 0.7415},
    }

print(f"  {'Target':<12}  {'Horizon':<10}  {'XGB R2':>8}  {RUN_TAG[:8]:>8}  {'Best':>6}")
print("  " + "-" * 56)
for tgt in TARGETS:
    for h in EVAL_HORIZONS:
        xgb_r2 = XGB_RESULTS[tgt][h]
        lstm_r2 = results[tgt][h]["r2"]
        best = RUN_TAG if lstm_r2 > xgb_r2 else "XGB"
        print(f"  {tgt:<12}  t+{h+1:02d}h      {xgb_r2:>8.4f}  {lstm_r2:>8.4f}  {best:>6}")
print(f"{'='*60}")
print("\nDone.")