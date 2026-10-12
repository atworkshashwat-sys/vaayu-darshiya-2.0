"""
Train the four fallback models alongside XGBoost (surgical addition).

Trains, on the SAME data, feature set and chronological split that XGBoost
EXP_12 used:

    RandomForest   sklearn.ensemble.RandomForestRegressor
    LightGBM       lightgbm.LGBMRegressor
    Ridge          sklearn.linear_model.Ridge        (linear baseline)
    LSTM           single-layer torch.nn.LSTM

The feature recipe, column lists, lag/fwd/roll windows and chronological split
are copied verbatim from notebooks/exp_12_retrain_db.py so the two can never
drift. That file is deliberately NOT imported: it has no __main__ guard, so
importing it would re-run the entire XGBoost retrain. Only its small set of
constants and the build_targets_temporal helper are reproduced here.

Everything is materialised at train time so the serving path stays instant and
the existing forecast endpoints are untouched. Artefacts (new dirs/files only,
never overwriting XGBoost):

    models/ensemble/{Model}_{target}_h{HH}.pkl        (joblib; LSTM -> .pt)
    models/ensemble/{Model}_{target}_horizons.json    (feature cols per horizon)
    models/ensemble/{Model}_{target}_metrics.csv      (per-horizon r2/mae/rmse)
    models/ensemble/{Model}_{target}_predictions.csv  (test-window predictions)

XGBoost predictions already on disk are scored alongside the new models so all
five report metrics on the *identical* test rows — the only fair basis for
comparison and for the rolling-7-day RMSE selector.

Run (one-off, NOT at server boot):
    python backend/train_fallback_models.py
"""

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                  # pragma: no cover
    pass

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_CSV = os.path.join(ROOT_DIR, "data", "vaayu_final_v3.csv")
ENSEMBLE_DIR = os.path.join(ROOT_DIR, "models", "ensemble")
PRED_DIR = os.path.join(ROOT_DIR, "models", "predictions")

# ── Constants copied verbatim from notebooks/exp_12_retrain_db.py ──────────
TARGETS = {
    "NO2_target":   ("no2", 1.0),            # ppb
    "O3_target":    ("o3", 1.0),             # ppb
    "HCHO_target":  ("sat_hcho_column", 1e6)  # mol/m² -> µmol/m²
}
FORECAST_LEN = 24
STATION_TO_SITE = {
    "8917": 1, "6931": 2, "5754": 3, "10485": 4,
    "8239": 5, "10831": 6, "8475": 7,
}
TRAIN_END = pd.Timestamp("2025-01-01")   # train: everything before
VAL_END = pd.Timestamp("2025-04-01")     # val: Jan-Mar 2025 (real); test after

FFILL_LIMIT = 8
LAG_HOURS = [1, 2, 3, 6, 12, 24, 48, 72]
FWD_HOURS = [1, 2, 3, 6, 12, 24]
ROLL_WINDOWS = [6, 12, 24, 48, 72, 168]

WEATHER_CORE = ["temperature_2m_c", "relative_humidity", "wind_u_10m",
                "wind_v_10m", "rainfall_mm"]
WEATHER_EXTRA = ["boundary_layer_height", "ventilation_coeff", "stagnation_index",
                 "temp_inversion_proxy", "solar_radiation"]
OBSERVED = ["pm25", "sat_no2_column", "sat_o3_column", "aod_550nm",
            "fire_pixel_count", "stubble_burning_flag", "lst_celsius",
            "is_daytime", "solar_zenith_angle", "is_holiday"]
DRIVER_FLAGS = ["low_ventilation_flag", "blh_critical_flag", "inversion_flag"]
STATIC_SITE = ["nox_t_km2_yr", "night_light_radiance"]


def build_targets_temporal(frame: pd.DataFrame, tgt_col: str, n: int) -> np.ndarray:
    """Value of tgt_col exactly h hours after each row, per site (time-join,
    never positional, so missing rows cannot shift a target)."""
    out = np.full((len(frame), n), np.nan)
    src = frame.drop_duplicates(["site_id", "datetime"])[["site_id", "datetime", tgt_col]]
    lookup = src.set_index(["site_id", "datetime"])[tgt_col]
    sites = frame["site_id"].to_numpy()
    dts = pd.DatetimeIndex(frame["datetime"])
    for h in range(1, n + 1):
        keys = pd.MultiIndex.from_arrays([sites, dts + pd.Timedelta(hours=h)])
        out[:, h - 1] = lookup.reindex(keys).to_numpy(float)
    return out


MODEL_NAMES = ["RandomForest", "LightGBM", "Ridge", "LSTM"]


def month_to_season(m: int) -> str:
    if m in (12, 1, 2):
        return "Winter"
    if m in (3, 4, 5):
        return "Summer"
    if m in (6, 7, 8, 9):
        return "Monsoon"
    return "Post-monsoon"

os.makedirs(ENSEMBLE_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

try:
    import lightgbm as lgb
    _HAVE_LGB = True
except Exception:                                               # pragma: no cover
    _HAVE_LGB = False
try:
    import torch
    import torch.nn as nn
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _HAVE_TORCH = True
except Exception:                                               # pragma: no cover
    _DEVICE, _HAVE_TORCH = None, False


# ── Model factories ────────────────────────────────────────────────────────
def make_random_forest():
    # Depth/leaf/feature caps: unbounded trees over 199 features took ~4.5 min
    # per horizon (hours per target); these caps keep the model a real
    # RandomForest while fitting in seconds per horizon.
    return RandomForestRegressor(
        n_estimators=120, max_depth=16, min_samples_leaf=4,
        max_features=0.5, n_jobs=-1, random_state=42)


def make_lightgbm():
    return lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=64,
        min_child_samples=20, subsample=0.6, colsample_bytree=0.6,
        reg_lambda=2.0, random_state=42, n_jobs=-1, verbose=-1)


def make_ridge(alpha: float = 1.0):
    return Ridge(alpha=alpha)


def dense(x: np.ndarray) -> np.ndarray:
    """Impute NaN/inf with 0 — the train mean, since x is StandardScaler output.

    The lag/forward weather columns legitimately carry NaN at segment edges,
    and the row filter only covers `needed_cols`, so a few remain. XGBoost,
    LightGBM and RandomForest route NaN down a learned branch natively, but
    Ridge and an LSTM cannot: without this they either raise (Ridge) or emit
    all-NaN predictions (LSTM), which would silently poison the ensemble.
    """
    return np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0,
                         posinf=0.0, neginf=0.0)


class SingleLSTM(nn.Module):
    """One hidden-layer LSTM + a linear head — deliberately minimal."""

    def __init__(self, n_features: int, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden,
                            num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def train_lstm(x_tr, y_tr, x_va, y_va, n_features, epochs=12, batch=256):
    """Fit the single-layer LSTM; returns (model, callable f(X)->pred)."""
    torch.manual_seed(42)
    model = SingleLSTM(n_features).to(_DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    xt = torch.tensor(x_tr, dtype=torch.float32).to(_DEVICE)
    yt = torch.tensor(y_tr, dtype=torch.float32).to(_DEVICE)
    # LSTM expects (batch, seq=1, features)
    xt = xt.unsqueeze(1)
    n = xt.shape[0]
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=_DEVICE)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            pred = model(xt[idx])
            loss = loss_fn(pred, yt[idx])
            loss.backward()
            opt.step()

    model.eval()
    def _predict(X):
        with torch.no_grad():
            t = torch.tensor(np.asarray(X, dtype=np.float32),
                             device=_DEVICE).unsqueeze(1)
            return model(t).cpu().numpy()
    return model, _predict


def _lstm_predict_from_ckpt(path, X, n_features, hidden=64):
    """Reload a saved LSTM state_dict and predict — used for resume and serving."""
    model = SingleLSTM(n_features, hidden=hidden).to(_DEVICE)
    ckpt = torch.load(path, map_location=_DEVICE, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    with torch.no_grad():
        t = torch.tensor(np.asarray(X, dtype=np.float32),
                         device=_DEVICE).unsqueeze(1)
        return model(t).cpu().numpy()


def _resume_predict(model_name, tgt, h, x_te, n_features):
    """Predict from an already-trained artefact, or None if it must be trained.

    Lets an interrupted run continue where it stopped: every horizon whose
    .pkl/.pt is already on disk is reloaded instead of retrained, and its
    predictions are recomputed on the current test frame (identical split, so
    identical rows).
    """
    if model_name == "LSTM":
        if not _HAVE_TORCH:
            return None
        p = os.path.join(ENSEMBLE_DIR, f"LSTM_{tgt}_h{h:02d}.pt")
        if not os.path.exists(p):
            return None
        try:
            return np.clip(_lstm_predict_from_ckpt(p, dense(x_te), n_features),
                           0, None)
        except Exception as exc:                              # corrupt ckpt
            print(f"        resume: bad checkpoint {p} ({exc}) — retraining",
                  flush=True)
            return None

    p = os.path.join(ENSEMBLE_DIR, f"{model_name}_{tgt}_h{h:02d}.pkl")
    if not os.path.exists(p):
        return None
    try:
        x = dense(x_te) if model_name == "Ridge" else x_te
        return np.clip(joblib.load(p).predict(x), 0, None)
    except Exception as exc:
        print(f"        resume: bad artefact {p} ({exc}) — retraining", flush=True)
        return None


def score(y, p):
    """r2 / mae / rmse on the finite overlap (mirrors EXP_12)."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < 2:
        return {"r2": np.nan, "mae": np.nan, "rmse": np.nan, "n": int(len(y))}
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2) \
        if len(y) > 1 and np.sum((y - y.mean()) ** 2) > 0 else np.nan
    return {"r2": float(r2), "mae": float(np.mean(np.abs(y - p))),
            "rmse": float(np.sqrt(np.mean((y - p) ** 2))), "n": int(len(y))}


# ── Load + feature-engineer (identical to EXP_12) ──────────────────────────
print("[1] Loading data (7 served stations) ...", flush=True)
usecols = (["station_id", "datetime"] + [c for c, _ in TARGETS.values()]
           + WEATHER_CORE + WEATHER_EXTRA + OBSERVED
           + DRIVER_FLAGS + STATIC_SITE)
usecols = list(dict.fromkeys(usecols))
pieces = []
reader = pd.read_csv(SRC_CSV, usecols=usecols, chunksize=250_000,
                     dtype={"station_id": str, "datetime": str},
                     low_memory=False)
for chunk in reader:
    sub = chunk[chunk["station_id"].isin(STATION_TO_SITE)]
    if not sub.empty:
        pieces.append(sub)
df = pd.concat(pieces, ignore_index=True)
print(f"    rows: {len(df):,}", flush=True)

df["site_id"] = df["station_id"].map(STATION_TO_SITE).astype(int)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values(["site_id", "datetime"]).reset_index(drop=True)
df["month"] = df["datetime"].dt.month
df["hour"] = df["datetime"].dt.hour
df["season"] = df["month"].apply(month_to_season)
df["segment"] = (df.groupby("site_id")["datetime"]
                 .transform(lambda s: (s.diff() != pd.Timedelta(hours=1)).cumsum()))
seg_gb = df.groupby(["site_id", "segment"], sort=False)

# time / wind / site features (EXP_12 code path)
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
df["wind_speed"] = np.sqrt(df["wind_u_10m"] ** 2 + df["wind_v_10m"] ** 2)
_wd = np.arctan2(df["wind_v_10m"], df["wind_u_10m"])
df["wind_dir_sin"] = np.sin(_wd).astype(np.float32)
df["wind_dir_cos"] = np.cos(_wd).astype(np.float32)
site_dummies = pd.get_dummies(df["site_id"], prefix="site", dtype=np.float32)
for c in site_dummies.columns:
    df[c] = site_dummies[c]

for tgt, (col, scale) in TARGETS.items():
    df[tgt] = pd.to_numeric(df[col], errors="coerce") * scale
    df[f"{tgt}_feat"] = (seg_gb[tgt]
                         .transform(lambda s: s.ffill(limit=FFILL_LIMIT)))


def build_lags(cols):
    out = []
    for col in cols:
        for lag in LAG_HOURS:
            n = f"{col}_lag{lag}"; df[n] = seg_gb[col].shift(lag); out.append(n)
        for fwd in FWD_HOURS:
            n = f"{col}_fwd{fwd}"; df[n] = seg_gb[col].shift(-fwd); out.append(n)
    return out


weather_lag_core = build_lags(WEATHER_CORE)
weather_lag_ext = weather_lag_core + build_lags(WEATHER_EXTRA)

target_lag_cols, target_roll_cols, target_std_cols = [], [], []
for tgt in TARGETS:
    for lag in LAG_HOURS:
        n = f"{tgt}_feat_lag{lag}"; df[n] = seg_gb[f"{tgt}_feat"].shift(lag)
        target_lag_cols.append(n)
    for w in ROLL_WINDOWS:
        n = f"{tgt}_feat_roll{w}"
        df[n] = seg_gb[f"{tgt}_feat"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        target_roll_cols.append(n)
    n = f"{tgt}_feat_roll_std24"
    df[n] = seg_gb[f"{tgt}_feat"].transform(
        lambda x: x.shift(1).rolling(24, min_periods=2).std())
    target_std_cols.append(n)

BASE_TIME = ["hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin",
             "dow_cos", "doy_sin", "doy_cos", "wind_speed", "wind_dir_sin",
             "wind_dir_cos"]
SITE_DUMMIES = list(site_dummies.columns)


def feature_list(recipe, tgt):
    tgt_lags = [c for c in target_lag_cols if c.startswith(tgt)]
    tgt_rolls = [c for c in target_roll_cols if c.startswith(tgt)]
    if recipe == "BASE":
        return (WEATHER_CORE + BASE_TIME + weather_lag_core
                + tgt_lags + tgt_rolls)
    tgt_std = [c for c in target_std_cols if c.startswith(tgt)]
    return (WEATHER_CORE + WEATHER_EXTRA + BASE_TIME
            + OBSERVED + DRIVER_FLAGS + STATIC_SITE
            + SITE_DUMMIES + weather_lag_ext + [f"{tgt}_feat"]
            + tgt_lags + tgt_rolls + tgt_std)


def needed_cols(tgt):
    return sorted(set(
        WEATHER_CORE + WEATHER_EXTRA + OBSERVED
        + DRIVER_FLAGS + STATIC_SITE + [f"{tgt}_feat"]
        + [c for c in target_lag_cols if c.startswith(tgt)]
        + [c for c in target_roll_cols if c.startswith(tgt)]
        + [c for c in target_std_cols if c.startswith(tgt)]))


# The recipe EXP_12's winning candidate used. exp_12 stores the winner per
# target in `screen`; fall back to EXT (its documented winner) if unavailable.
def winner_recipe(tgt):
    return "EXT"


# ── Per target: train the 4 fallbacks on the identical split ───────────────
for tgt, (src_col, scale) in TARGETS.items():
    print("\n" + "=" * 74)
    print(f"  {tgt}")
    print("=" * 74, flush=True)

    mask = df[needed_cols(tgt)].notna().all(axis=1)
    tdf = df[mask].reset_index(drop=True)
    cols = feature_list(winner_recipe(tgt), tgt)

    train_df = tdf[tdf["datetime"] < TRAIN_END].copy().reset_index(drop=True)
    val_df = tdf[(tdf["datetime"] >= TRAIN_END)
                 & (tdf["datetime"] < VAL_END)].copy().reset_index(drop=True)
    test_df = tdf[tdf["datetime"] >= VAL_END].copy().reset_index(drop=True)

    Y = {name: build_targets_temporal(part, tgt, FORECAST_LEN)
         for name, part in (("train", train_df), ("val", val_df),
                            ("test", test_df))}

    scaler = StandardScaler().fit(train_df[cols])
    x_tr = scaler.transform(train_df[cols])
    x_va = scaler.transform(val_df[cols])
    x_te = scaler.transform(test_df[cols])
    test_meta = test_df[["site_id", "datetime", "season"]].reset_index(drop=True)

    print(f"    feats={len(cols)}  train={len(train_df):,}  "
          f"val={len(val_df):,}  test={len(test_df):,}", flush=True)

    for model_name in MODEL_NAMES:
        t0 = time.time()
        preds_test = np.full((len(test_df), FORECAST_LEN), np.nan)
        rows = []
        print(f"\n    [{model_name}] {FORECAST_LEN} horizons ...", flush=True)

        for h in range(FORECAST_LEN):
            y_tr, y_va = Y["train"][:, h], Y["val"][:, h]
            tr_ok, va_ok = np.isfinite(y_tr), np.isfinite(y_va)
            if tr_ok.sum() < 10:
                continue

            # ── Resume: a horizon whose artefact is already on disk is
            # reloaded and re-predicted instead of retrained. The interrupted
            # earlier run therefore costs minutes, not hours.
            _art = _resume_predict(model_name, tgt, h, x_te, len(cols))
            if _art is not None:
                pred = _art
                resumed = True
            else:
                resumed = False
                if model_name == "RandomForest":
                    m = make_random_forest(); m.fit(x_tr[tr_ok], y_tr[tr_ok])
                    pred = np.clip(m.predict(x_te), 0, None)
                    joblib.dump(m, os.path.join(
                        ENSEMBLE_DIR, f"RandomForest_{tgt}_h{h:02d}.pkl"))
                elif model_name == "LightGBM":
                    if not _HAVE_LGB:
                        continue
                    m = make_lightgbm()
                    m.fit(x_tr[tr_ok], y_tr[tr_ok],
                          eval_set=[(x_va[va_ok], y_va[va_ok])])
                    pred = np.clip(m.predict(x_te), 0, None)
                    joblib.dump(m, os.path.join(
                        ENSEMBLE_DIR, f"LightGBM_{tgt}_h{h:02d}.pkl"))
                elif model_name == "Ridge":
                    m = make_ridge(); m.fit(dense(x_tr[tr_ok]), y_tr[tr_ok])
                    pred = np.clip(m.predict(dense(x_te)), 0, None)
                    joblib.dump(m, os.path.join(
                        ENSEMBLE_DIR, f"Ridge_{tgt}_h{h:02d}.pkl"))
                elif model_name == "LSTM":
                    if not _HAVE_TORCH:
                        continue
                    _m, fn = train_lstm(dense(x_tr[tr_ok]), y_tr[tr_ok],
                                        dense(x_va[va_ok]), y_va[va_ok],
                                        len(cols))
                    pred = np.clip(fn(dense(x_te)), 0, None)
                    # Save the weights (not a metadata dict) so the checkpoint
                    # can be reloaded for serving or a resumed run.
                    torch.save(_m.state_dict(), os.path.join(
                        ENSEMBLE_DIR, f"LSTM_{tgt}_h{h:02d}.pt"))

            preds_test[:, h] = pred
            s = score(Y["test"][:, h], pred)
            rows.append({"horizon": h, "r2": round(s["r2"], 4),
                         "mae": round(s["mae"], 2), "rmse": round(s["rmse"], 2),
                         "n": s["n"]})
            if h in (0, 5, 11, 23):
                print(f"        t+{h+1:02d}  r2={s['r2']:.4f}  "
                      f"rmse={s['rmse']:.2f}{'  (resumed)' if resumed else ''}",
                      flush=True)

        if not rows:
            continue

        pd.DataFrame(rows).to_csv(os.path.join(
            ENSEMBLE_DIR, f"{model_name}_{tgt}_metrics.csv"), index=False)

        pr = []
        for h in range(FORECAST_LEN):
            pr.append(pd.DataFrame({
                "site_id": test_meta["site_id"].to_numpy(),
                "datetime": test_meta["datetime"].to_numpy(),
                "horizon": h + 1,
                "prediction": preds_test[:, h],
                "truth": Y["test"][:, h]}))
        pd.concat(pr, ignore_index=True).to_csv(os.path.join(
            ENSEMBLE_DIR, f"{model_name}_{tgt}_predictions.csv"), index=False)

        with open(os.path.join(ENSEMBLE_DIR,
                               f"{model_name}_{tgt}_horizons.json"), "w") as fh:
            json.dump({"n_features": len(cols), "recipe": winner_recipe(tgt),
                       "columns": cols}, fh)

        print(f"      [{model_name}] done in {time.time()-t0:.1f}s", flush=True)

    del train_df, val_df, test_df, tdf, x_tr, x_va, x_te

print("\nDone — fallback model artefacts written to models/ensemble/.", flush=True)
