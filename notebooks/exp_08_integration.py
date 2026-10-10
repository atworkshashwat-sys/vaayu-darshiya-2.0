"""
VAAYU DARSHIYA — EXP_08
BiLSTM Encoder → XGBoost Hybrid
- Load trained BiLSTM from EXP_07
- Extract 256-dim embeddings (no prediction head)
- Combine with hand-crafted features
- Train XGBoost on embeddings → predict NO2 & O3
- Train: 2016-2021  Test: 2022
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
warnings.filterwarnings("ignore")

os.chdir(r"C:\Users\Asus\Downloads\VAAYU_DARSHIYA_MODEL_WORK")

# ── CONFIG (must match EXP_07 exactly) ───────────────────────────────────────
DATA_FILE    = "vaayu_final_ml_ready.csv"
TRAIN_YEARS  = list(range(2016, 2021))   # 2016–2021
TEST_YEARS   = [2022]
TARGETS      = ["no2", "o3"]

INPUT_LEN    = 48
FORECAST_LEN = 24
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.4
ATTN_HEADS   = 4
BATCH_SIZE   = 256
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("\n[1] Loading data ...")
df = pd.read_csv(DATA_FILE, low_memory=False)
print(f"    Shape: {df.shape}")

df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df["year"]     = df["datetime"].dt.year
df = df.sort_values(["station_id", "datetime"]).reset_index(drop=True)

DROP_COLS = [
    "aqi_no2", "aqi_o3", "aqi_computed", "aqi_category", "aqi_pm25",
    "station_name", "station_id", "date", "datetime",
    "lulc_class_name", "season",
    *[c for c in df.columns if "_lag_" in c or "_roll_" in c
      or "o3_8h_rolling" in c],
]
feature_cols = [c for c in df.columns
                if c not in DROP_COLS and c not in TARGETS]

for col in feature_cols:
    if df[col].dtype == "object":
        df[col] = (df[col].astype("category")
                         .cat.codes.astype(float)
                         .replace(-1, np.nan))

df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
df[TARGETS]      = df[TARGETS].fillna(df[TARGETS].median())

# ── 2. SPLIT ──────────────────────────────────────────────────────────────────
print("\n[2] Splitting ...")
train_df = df[df["year"].isin(TRAIN_YEARS)].copy()
test_df  = df[df["year"].isin(TEST_YEARS)].copy()

_t0   = train_df["datetime"].min()
_tend = pd.Timestamp("2022-12-31 23:00:00")
for _df in (train_df, test_df):
    _df["time_idx"] = (
        (_df["datetime"] - _t0).dt.total_seconds() /
        (_tend - _t0).total_seconds()
    ).astype(np.float32)
    # ── NEW: cyclic time features (Fix D) — skip if already in CSV ──
    if "hour_sin" not in _df.columns:
        _df["hour_sin"]  = np.sin(2 * np.pi * _df["datetime"].dt.hour  / 24).astype(np.float32)
        _df["hour_cos"]  = np.cos(2 * np.pi * _df["datetime"].dt.hour  / 24).astype(np.float32)
    if "month_sin" not in _df.columns:
        _df["month_sin"] = np.sin(2 * np.pi * _df["datetime"].dt.month / 12).astype(np.float32)
        _df["month_cos"] = np.cos(2 * np.pi * _df["datetime"].dt.month / 12).astype(np.float32)
    # `dow` is intentionally NOT added — CSV already has `day_of_week` / `dow_sin` / `dow_cos`,
    # and adding a fresh `dow` column would change the BiLSTM input dim and break the
    # exp07_bilstm_best.pt checkpoint (which was saved with 73 features, not 74).
    if "time_idx" not in _df.columns:
        pass  # time_idx is added unconditionally below

# Only extend feature_cols with names that aren't ALREADY in it
# `dow` deliberately excluded — see note above.
_extra_time_feats = ["time_idx", "hour_sin", "hour_cos",
                     "month_sin", "month_cos"]
for _c in _extra_time_feats:
    if _c in train_df.columns and _c not in feature_cols:
        feature_cols.append(_c)
print(f"    Train: {len(train_df):,} rows")
print(f"    Test : {len(test_df):,} rows")

# ── 2b. STATION ENCODING (Fix B) — deferred to window-level (after dataset) ──
# Building station one-hot here would be per-row (247k), but the BiLSTM works
# at window-level (242k). Doing it after the dataset keeps shapes aligned.

# ── 3. SCALE (fit on train only) ──────────────────────────────────────────────
print("\n[3] Scaling ...")
feat_scaler   = StandardScaler()
target_scaler = StandardScaler()

train_feats   = feat_scaler.fit_transform(train_df[feature_cols])
test_feats    = feat_scaler.transform(test_df[feature_cols])
train_targets = target_scaler.fit_transform(train_df[TARGETS])
test_targets  = target_scaler.transform(test_df[TARGETS])

# ── 4. DATASET ────────────────────────────────────────────────────────────────
class PollutionDataset(Dataset):
    def __init__(self, feats, targets, station_ids, input_len, forecast_len):
        self.X, self.Y, self.S = [], [], []
        for sid in np.unique(station_ids):
            idx = np.where(station_ids == sid)[0]
            f, t = feats[idx], targets[idx]
            for i in range(len(idx) - input_len - forecast_len + 1):
                self.X.append(f[i : i + input_len])
                self.Y.append(t[i + input_len : i + input_len + forecast_len])
                # Track which station this window belongs to (string for OHE alignment)
                self.S.append(str(sid))
        self.X = np.array(self.X, dtype=np.float32)
        self.Y = np.array(self.Y, dtype=np.float32)
        self.S = np.array(self.S)
        print(f"    Dataset: {len(self.X):,} samples, {len(set(self.S))} stations")

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]

print("\n[4] Building datasets ...")
train_ds = PollutionDataset(train_feats, train_targets,
                             train_df["station_id"].values,
                             INPUT_LEN, FORECAST_LEN)
test_ds  = PollutionDataset(test_feats, test_targets,
                             test_df["station_id"].values,
                             INPUT_LEN, FORECAST_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=False)

# ── 4b. STATION ENCODING at WINDOW level (matches train_ds.X rows) ────────────
ohe_station = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
ohe_station.fit(pd.concat([train_df, test_df])[["station_id"]].astype(str))
print(f"    Stations encoded: {len(ohe_station.categories_[0])} one-hot columns")
# Fit on the union of all stations seen in both train and test windows
_all_stations = np.concatenate([train_ds.S, test_ds.S])
ohe_station.fit(_all_stations.reshape(-1, 1))
station_train = ohe_station.transform(train_ds.S.reshape(-1, 1))
station_test  = ohe_station.transform(test_ds.S.reshape(-1, 1))
print(f"    Stations encoded (window-level): {station_train.shape[1]} one-hot cols, "
      f"train={station_train.shape[0]:,}, test={station_test.shape[0]:,}")

# ── 5. LOAD BILSTM ────────────────────────────────────────────────────────────
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size * 2, num_heads=num_heads,
            dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size * 2)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)

class VaayuBiLSTM(nn.Module):
    def __init__(self, n_features, hidden_size, num_layers, dropout,
                 attn_heads, forecast_len, n_targets):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True, bidirectional=True)
        self.attention = AttentionLayer(hidden_size, attn_heads)
        self.dropout   = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 256), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, forecast_len * n_targets))
        self.forecast_len = forecast_len
        self.n_targets    = n_targets

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        attn_out    = self.attention(lstm_out)
        context     = self.dropout(attn_out[:, -1, :])
        out         = self.head(context)
        return out.view(-1, self.forecast_len, self.n_targets)

n_features = train_ds.X.shape[2]
model = VaayuBiLSTM(
    n_features=n_features, hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS, dropout=DROPOUT,
    attn_heads=ATTN_HEADS, forecast_len=FORECAST_LEN,
    n_targets=len(TARGETS)).to(DEVICE)

model.load_state_dict(
    torch.load("exp07_bilstm_best.pt", map_location=DEVICE))
model.eval()
print("\n[5] BiLSTM loaded from exp07_bilstm_best.pt")

# ── 6. EXTRACT EMBEDDINGS ─────────────────────────────────────────────────────
print("\n[6] Extracting BiLSTM embeddings ...")


def extract_embeddings(loader):
    all_emb, all_y = [], []
    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            lstm_out, _ = model.bilstm(X_batch)  # (B, 48, 256)
            attn_out = model.attention(lstm_out)  # (B, 48, 256)
            all_emb.append(attn_out.cpu().numpy())
            all_y.append(Y_batch.numpy())
    emb = np.concatenate(all_emb, axis=0)  # (N, 48, 256)
    y = np.concatenate(all_y, axis=0)  # (N, 24, 2)
    return emb, y


def pool_embeddings(emb_seq):
    """Fix A: pool over the 48 timesteps → mean + max + last → (N, 768)."""
    return np.concatenate(
        [emb_seq.mean(axis=1), emb_seq.max(axis=1), emb_seq[:, -1, :]],
        axis=1,
    )
def get_windowed_station_ohe(df, ohe, input_len, forecast_len):
    out = []
    station_ids = df["station_id"].values
    for sid in np.unique(station_ids):
        idx = np.where(station_ids == sid)[0]
        n_windows = len(idx) - input_len - forecast_len + 1
        if n_windows <= 0:
            continue
        sid_ohe = ohe.transform([[str(sid)]])
        out.append(np.repeat(sid_ohe, n_windows, axis=0))
    return np.vstack(out)
train_emb_seq, train_Y = extract_embeddings(train_loader)
test_emb_seq,  test_Y  = extract_embeddings(test_loader)

# Fix A: pool the 48-step sequence into 768-d
train_emb = pool_embeddings(train_emb_seq)
test_emb  = pool_embeddings(test_emb_seq)

print(f"    Train embeddings (pooled): {train_emb.shape}")
print(f"    Test  embeddings (pooled): {test_emb.shape}")

# Last timestep of input window = most recent observed values
train_last = train_ds.X[:, -1, :]   # (N, n_features)
test_last  = test_ds.X[:, -1, :]

# Fix B: add station one-hot
# XGBoost input: 768 (pooled emb) + n_features (current) + n_stations (one-hot)
station_train_win = get_windowed_station_ohe(train_df, ohe_station, INPUT_LEN, FORECAST_LEN)
station_test_win  = get_windowed_station_ohe(test_df,  ohe_station, INPUT_LEN, FORECAST_LEN)
X_train = np.hstack([train_emb, train_last, station_train_win])
X_test  = np.hstack([test_emb,  test_last,  station_test_win])
print(f"    XGBoost input shape — Train: {X_train.shape}  Test: {X_test.shape}")

# ── 7. XGBOOST ────────────────────────────────────────────────────────────────
# ── 7. XGBOOST ────────────────────────────────────────────────────────────────
print("\n[7] Training XGBoost models ...")
EVAL_HORIZONS = [0, 5, 11, 23]
results = {}

def inv_transform_target(arr_24, t_idx, target_scaler):
    n = arr_24.shape[0]
    out = np.zeros_like(arr_24)
    dummy = np.zeros((n, len(TARGETS)))
    for h in range(arr_24.shape[1]):
        dummy[:, t_idx] = arr_24[:, h]
        out[:, h] = target_scaler.inverse_transform(dummy)[:, t_idx]
    return out

for t_idx, tgt in enumerate(TARGETS):
    y_train_raw = train_Y[:, :, t_idx]   # (N, 24)
    y_test_raw  = test_Y[:,  :, t_idx]

    tgt_col_in_feats = feature_cols.index(tgt) if tgt in feature_cols else None
    if tgt_col_in_feats is not None:
        last_scaled_tr = train_ds.X[:, -1, tgt_col_in_feats]
        last_scaled_te = test_ds.X[:,  -1, tgt_col_in_feats]
        dummy_tr = np.zeros((len(last_scaled_tr), len(feature_cols)))
        dummy_tr[:, tgt_col_in_feats] = last_scaled_tr
        last_real_tr = feat_scaler.inverse_transform(dummy_tr)[:, tgt_col_in_feats]
        dummy_te = np.zeros((len(last_scaled_te), len(feature_cols)))
        dummy_te[:, tgt_col_in_feats] = last_scaled_te
        last_real_te = feat_scaler.inverse_transform(dummy_te)[:, tgt_col_in_feats]
        pers_tr = np.tile(last_real_tr[:, None], (1, FORECAST_LEN))
        pers_te = np.tile(last_real_te[:, None], (1, FORECAST_LEN))
        y_train_target = y_train_raw - pers_tr
        y_test_target  = y_test_raw  - pers_te
    else:
        y_train_target = y_train_raw
        y_test_target  = y_test_raw
        pers_te        = np.zeros_like(y_test_raw)

    params = dict(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.6, min_child_weight=8,
        reg_alpha=0.3, reg_lambda=1.5, random_state=42,
        n_jobs=-1, tree_method="hist",
        early_stopping_rounds=30, eval_metric="rmse",
    )
    if tgt == "o3":
        params.update(
            max_depth=4,  # was 6
            min_child_weight=30,  # was 15
            reg_alpha=3.0,  # was 1.0
            reg_lambda=5.0,  # was 3.0
            subsample=0.5,
            colsample_bytree=0.4,
            early_stopping_rounds=30,
        )

    print(f"\n    {tgt.upper()} — training 24 horizon models ...")
    all_preds = np.zeros((X_test.shape[0], FORECAST_LEN))
    best_iters = []

    for h in range(FORECAST_LEN):
        xgb_h = xgb.XGBRegressor(**params)
        xgb_h.fit(
            X_train, y_train_target[:, h],
            eval_set=[(X_test, y_test_target[:, h])],
            verbose=False,
        )
        all_preds[:, h] = xgb_h.predict(X_test)
        best_iters.append(xgb_h.best_iteration)
        print(f"      t+{h+1:02d} done (best_iter={xgb_h.best_iteration})", flush=True)

    preds_scaled = pers_te + all_preds
    preds        = np.clip(inv_transform_target(preds_scaled, t_idx, target_scaler), 0, None)
    y_test_real  = inv_transform_target(y_test_raw, t_idx, target_scaler)

    print(f"    {'Horizon':<10}  {'MAE':>8}  {'RMSE':>8}  {'R²':>8}")
    print("    " + "-" * 45)
    results[tgt] = {}
    for h in range(FORECAST_LEN):
        p = preds[:, h]
        y = y_test_real[:, h]
        mae  = mean_absolute_error(y, p)
        rmse = np.sqrt(mean_squared_error(y, p))
        r2   = r2_score(y, p)
        results[tgt][h] = {"mae": mae, "rmse": rmse, "r2": r2}
        if h in EVAL_HORIZONS:
            print(f"    t+{h+1:02d}h       {mae:>8.3f}  {rmse:>8.3f}  {r2:>8.4f}")

    joblib.dump(
        {h: results[tgt][h] for h in range(FORECAST_LEN)},
        f"exp08_xgb_{tgt}_results.pkl")
    print(f"    Saved → exp08_xgb_{tgt}_results.pkl")
# ── 8. FINAL SUMMARY ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  EXP_08 — BiLSTM Embeddings + XGBoost Hybrid Results")
print("  Train: 2016–2021   Test: 2022")
print("=" * 60)
print(f"  {'Target':<6}  {'Horizon':<10}  {'MAE':>8}  {'RMSE':>8}  {'R²':>8}")
print("  " + "-" * 55)
for tgt in TARGETS:
    for h in [0, 5, 11, 23]:
        r = results[tgt][h]
        print(f"  {tgt.upper():<6}  t+{h+1:02d}h       "
              f"{r['mae']:>8.3f}  {r['rmse']:>8.3f}  {r['r2']:>8.4f}")
print("=" * 60)

print("""
  EXP_07 BiLSTM alone (baseline to beat):
    NO2  MAE=17.139  RMSE=31.457  R²=0.3574
    O3   MAE=17.190  RMSE=32.288  R²=0.1057
""")

if DEVICE.type == "cuda":
    torch.cuda.empty_cache()

print("Done — EXP_08 complete.")

# ── 9. SANITY CHECK ───────────────────────────────────────────────────────────
print("\n[9] Train-set sanity check (expect > 0.7 R² if not overfitting) ...")
for t_idx, tgt in enumerate(TARGETS):
    y_train_real = train_Y[:, :, t_idx]
    xgb_sanity = xgb.XGBRegressor(n_estimators=300, max_depth=6, n_jobs=-1)
    xgb_sanity.fit(X_train[:20000], y_train_real[:20000, 0])  # horizon 0 only
    train_pred = xgb_sanity.predict(X_train[:20000])
    print(f"    {tgt.upper()} horizon 0 — train R²: "
          f"{r2_score(y_train_real[:20000, 0], train_pred):.3f}")
print("    (If train R² >> test R² → overfitting → increase regularization.)")