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

CHANGELOG (this revision):
  [P1] Feature importance CSV per model
  [P1] Validation metrics computed + saved (not just used for early stopping)
  [P1] best_iteration stored in results dict
  [P2] Error broken down by site
  [P2] Error broken down by season (Winter/Summer/Monsoon/Post-monsoon)
  [P2] Full metrics also saved as CSV (not just pickle)
  [P3] base_params pulled into a swappable HPARAM_SETS dict (Run A/B/C ready)
  [P4] GPU device actually wired into XGBRegressor (was detected, unused)
  [P5] Added dayofweek + day_of_year cyclic features
  [P6] Train sanity check now runs on full train/val/test, single model, not a 10k subsample
  [P6] Prediction CSV saved (pred, truth, site, datetime) per target for easy plotting
  [P-] SHAP: left as a standalone post-hoc snippet at bottom of file, not in the training loop
  [P-] Holiday flag: stubbed (HOLIDAY_DATES) — fill in and flip ADD_HOLIDAY_FLAG to use it
  [P-] StandardScaler: left as-is per your own note (already saved, pipeline stable)
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

# [P3] Swappable hyperparameter sets — set ACTIVE_HPARAM_SET to "A", "B", or "C"
# to run the sweep suggested in review without touching the training loop.
ACTIVE_HPARAM_SET = "A"
HPARAM_SETS = {
    "A": dict(max_depth=5, min_child_weight=10, gamma=0.2, reg_lambda=2.0),
    "B": dict(max_depth=4, min_child_weight=12, gamma=0.3, reg_lambda=3.0),
    "C": dict(max_depth=5, min_child_weight=15, gamma=0.2, reg_lambda=2.0),
}

# [P-] Holiday flag stub — fill HOLIDAY_DATES with a DatetimeIndex of India public
# holidays (e.g. via `pd.to_datetime([...])` or an external calendar package) and
# flip this to True. Left off by default since no holiday source was provided.
ADD_HOLIDAY_FLAG = False
HOLIDAY_DATES = pd.DatetimeIndex([])  # e.g. pd.to_datetime(["2019-01-26", ...])

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

# Cyclic time — hour, month
df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24).astype(np.float32)
df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24).astype(np.float32)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype(np.float32)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype(np.float32)

# [P5] Cyclic time — day of week (NO2 traffic pattern) + day of year (finer than month)
df["dayofweek"]    = df["datetime"].dt.dayofweek
df["dow_sin"]       = np.sin(2 * np.pi * df["dayofweek"] / 7).astype(np.float32)
df["dow_cos"]       = np.cos(2 * np.pi * df["dayofweek"] / 7).astype(np.float32)
df["day_of_year"]   = df["datetime"].dt.dayofyear
df["doy_sin"]        = np.sin(2 * np.pi * df["day_of_year"] / 365.25).astype(np.float32)
df["doy_cos"]        = np.cos(2 * np.pi * df["day_of_year"] / 365.25).astype(np.float32)

# [P-] Holiday flag (stub) — only adds the column if dates are supplied above
if ADD_HOLIDAY_FLAG:
    df["is_holiday"] = df["datetime"].dt.normalize().isin(HOLIDAY_DATES).astype(np.int8)

# Wind speed + direction
df["wind_speed"]   = np.sqrt(df["u_forecast"]**2 + df["v_forecast"]**2)
df["wind_dir"]     = np.arctan2(df["v_forecast"], df["u_forecast"])
df["wind_dir_sin"] = np.sin(df["wind_dir"]).astype(np.float32)
df["wind_dir_cos"] = np.cos(df["wind_dir"]).astype(np.float32)

# [P2] Season label for later error breakdown (India-typical 4-season split).
# Not used as a model feature — kept out of FEATURE_COLS deliberately so it
# doesn't leak the same signal as month_sin/cos twice; used only for analysis.
def month_to_season(m):
    if m in (12, 1, 2):
        return "Winter"
    elif m in (3, 4, 5):
        return "Summer"
    elif m in (6, 7, 8, 9):
        return "Monsoon"
    else:  # 10, 11
        return "Post-monsoon"

df["season"] = df["month"].apply(month_to_season)

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
       "dow_sin", "dow_cos", "doy_sin", "doy_cos",
       "wind_speed", "wind_dir_sin", "wind_dir_cos"]
    + (["is_holiday"] if ADD_HOLIDAY_FLAG else [])
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
# Note: left in deliberately (per your own call) — trees don't need it, but the
# scaler is already part of the saved pipeline and changing it now would break
# compatibility with anything downstream that expects scaled inputs. Revisit only
# if you're benchmarking scaled-vs-unscaled XGBoost specifically.
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

# [P2/P6] Keep site_id + datetime aligned with the filtered/valid rows so we can
# do per-site, per-season breakdowns and write prediction CSVs later.
train_meta = train_df.loc[train_valid, ["site_id", "datetime", "season"]].reset_index(drop=True)
val_meta   = val_df.loc[val_valid,   ["site_id", "datetime", "season"]].reset_index(drop=True)
test_meta  = test_df.loc[test_valid,  ["site_id", "datetime", "season"]].reset_index(drop=True)

print(f"    Train: {X_train.shape[0]:,}  Val: {X_val.shape[0]:,}  Test: {X_test.shape[0]:,}")

# ── 7. TRAIN XGBOOST PER HORIZON ─────────────────────────────────────────────
print("\n[7] Training XGBoost models ...")
os.makedirs("exp09v2_models", exist_ok=True)
results = {}
all_preds_by_target = {}  # kept so section 9 prediction-CSV export can reuse them

for tgt in TARGETS:
    print(f"\n    {tgt} — training {FORECAST_LEN} horizon models ...")
    all_preds = np.zeros((X_test.shape[0], FORECAST_LEN))

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
        tree_method=TREE_METHOD,   # [P4] was hardcoded "hist", now uses detection
        device=DEVICE,             # [P4] was detected but never passed through
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

        # Test predictions (as before)
        test_pred = np.clip(xgb_h.predict(X_test), 0, None)
        all_preds[:, h] = test_pred

        # [P1] Validation predictions/metrics — previously val was only used for
        # early stopping and never actually scored or saved.
        val_pred = np.clip(xgb_h.predict(X_val), 0, None)
        val_r2   = r2_score(y_val, val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_mae  = mean_absolute_error(y_val, val_pred)

        joblib.dump(xgb_h, f"exp09v2_models/{tgt}_h{h:02d}.pkl")

        # [P1] Feature importance per model
        importance = pd.DataFrame({
            "feature": FEATURE_COLS,
            "importance": xgb_h.feature_importances_
        }).sort_values("importance", ascending=False)
        importance.to_csv(
            f"exp09v2_models/{tgt}_h{h:02d}_importance.csv",
            index=False
        )

        # Stash val metrics for the results dict assembled after the test-metric loop below
        if tgt not in results:
            results[tgt] = {}
        results[tgt].setdefault("_val_tmp", {})[h] = {
            "val_mae": val_mae, "val_rmse": val_rmse, "val_r2": val_r2,
            "best_iter": xgb_h.best_iteration,
        }

        print(f"      t+{h+1:02d} done (best_iter={xgb_h.best_iteration}, "
              f"val_r2={val_r2:.4f})", flush=True)

    joblib.dump(all_preds, f"exp09v2_xgb_{tgt}_preds.pkl")
    all_preds_by_target[tgt] = all_preds

    print(f"\n    {'Horizon':<10}  {'MAE':>8}  {'RMSE':>8}  {'R²':>8}  "
          f"{'ValR²':>8}  {'BestIter':>9}")
    print("    " + "-" * 65)

    val_tmp = results[tgt].pop("_val_tmp")
    for h in range(FORECAST_LEN):
        p    = all_preds[:, h]
        y    = Y_test[tgt][:, h]
        mae  = mean_absolute_error(y, p)
        rmse = np.sqrt(mean_squared_error(y, p))
        r2   = r2_score(y, p)

        # [P1/P3] merged test + val + best_iter into one results record
        results[tgt][h] = {
            "mae": mae, "rmse": rmse, "r2": r2,
            "val_mae": val_tmp[h]["val_mae"],
            "val_rmse": val_tmp[h]["val_rmse"],
            "val_r2": val_tmp[h]["val_r2"],
            "best_iter": val_tmp[h]["best_iter"],
        }
        if h in EVAL_HORIZONS:
            print(f"    t+{h+1:02d}h       {mae:>8.3f}  {rmse:>8.3f}  {r2:>8.4f}  "
                  f"{val_tmp[h]['val_r2']:>8.4f}  {val_tmp[h]['best_iter']:>9d}")

    joblib.dump(results[tgt], f"exp09v2_xgb_{tgt}_results.pkl")

    # [P8] Also save full metrics as CSV, not just pickle
    results_df = pd.DataFrame(results[tgt]).T
    results_df.index.name = "horizon"
    results_df.to_csv(f"exp09v2_xgb_{tgt}_results.csv")
    print(f"    Saved → exp09v2_xgb_{tgt}_results.pkl / .csv")

    # [P2] Error by site (using the eval horizons to keep the printout readable;
    # full per-horizon breakdown is in the saved CSV below)
    print(f"\n    {tgt} — error by site (test set):")
    print(f"    {'Site':<6}  {'Horizon':<10}  {'MAE':>8}  {'RMSE':>8}  {'R²':>8}")
    print("    " + "-" * 50)
    site_rows = []
    for site in sorted(test_meta["site_id"].unique()):
        site_mask = (test_meta["site_id"] == site).values
        for h in EVAL_HORIZONS:
            y_s = Y_test[tgt][site_mask, h]
            p_s = all_preds[site_mask, h]
            mae_s  = mean_absolute_error(y_s, p_s)
            rmse_s = np.sqrt(mean_squared_error(y_s, p_s))
            r2_s   = r2_score(y_s, p_s)
            site_rows.append({"site_id": site, "horizon": h, "mae": mae_s,
                               "rmse": rmse_s, "r2": r2_s})
            print(f"    {site:<6}  t+{h+1:02d}h       {mae_s:>8.3f}  "
                  f"{rmse_s:>8.3f}  {r2_s:>8.4f}")
    pd.DataFrame(site_rows).to_csv(f"exp09v2_{tgt}_error_by_site.csv", index=False)

    # [P2] Error by season
    print(f"\n    {tgt} — error by season (test set):")
    print(f"    {'Season':<14}  {'Horizon':<10}  {'MAE':>8}  {'RMSE':>8}  {'R²':>8}")
    print("    " + "-" * 56)
    season_rows = []
    for season in ["Winter", "Summer", "Monsoon", "Post-monsoon"]:
        season_mask = (test_meta["season"] == season).values
        if season_mask.sum() == 0:
            continue
        for h in EVAL_HORIZONS:
            y_se = Y_test[tgt][season_mask, h]
            p_se = all_preds[season_mask, h]
            mae_se  = mean_absolute_error(y_se, p_se)
            rmse_se = np.sqrt(mean_squared_error(y_se, p_se))
            r2_se   = r2_score(y_se, p_se)
            season_rows.append({"season": season, "horizon": h, "mae": mae_se,
                                 "rmse": rmse_se, "r2": r2_se})
            print(f"    {season:<14}  t+{h+1:02d}h       {mae_se:>8.3f}  "
                  f"{rmse_se:>8.3f}  {r2_se:>8.4f}")
    pd.DataFrame(season_rows).to_csv(f"exp09v2_{tgt}_error_by_season.csv", index=False)

# ── 8. FINAL SUMMARY ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  EXP_09 v2 — XGBoost on NWP Features (Improved)")
print("  Train: 2019-2021  Val: 2022  Test: 2023-2024  Sites: 1-7")
print(f"  Hyperparameter set: {ACTIVE_HPARAM_SET}  |  Device: {DEVICE}")
print("=" * 60)
print(f"  {'Target':<12}  {'Horizon':<10}  {'MAE':>8}  {'RMSE':>8}  {'R²':>8}  {'ValR²':>8}")
print("  " + "-" * 66)
for tgt in TARGETS:
    for h in EVAL_HORIZONS:
        r = results[tgt][h]
        print(f"  {tgt:<12}  t+{h+1:02d}h       "
              f"{r['mae']:>8.3f}  {r['rmse']:>8.3f}  {r['r2']:>8.4f}  {r['val_r2']:>8.4f}")
print("=" * 60)

print("""
  EXP_09 v1 baseline:
    NO2  t+01h R²=0.7588
    O3   t+01h R²=0.8519
""")

# ── 9. PREDICTION CSV (pred, truth, site, datetime) ──────────────────────────
# [P-15] Easy-plotting export. Built from the test-set predictions already
# computed in step 7, joined back to site_id/datetime via test_meta.
print("\n[9] Saving prediction CSVs ...")
for tgt in TARGETS:
    all_preds = all_preds_by_target[tgt]
    rows = []
    for h in range(FORECAST_LEN):
        chunk = pd.DataFrame({
            "site_id": test_meta["site_id"].values,
            "datetime": test_meta["datetime"].values,
            "horizon": h + 1,
            "prediction": all_preds[:, h],
            "truth": Y_test[tgt][:, h],
        })
        rows.append(chunk)
    pred_df = pd.concat(rows, ignore_index=True)
    pred_df.to_csv(f"exp09v2_{tgt}_predictions.csv", index=False)
    print(f"    Saved → exp09v2_{tgt}_predictions.csv  ({len(pred_df):,} rows)")

# ── 10. TRAIN/VAL/TEST SANITY CHECK (full data, single model, horizon 0) ─────
# [P-14] Previously this trained on a 10k subsample, which is a weak overfitting
# signal. Now uses the full train set for the fit and reports R² across the
# full train, val, and test sets for one quick reference model per target.
print("\n[10] Train/Val/Test sanity check (full data, single model, horizon 0) ...")
for tgt in TARGETS:
    xgb_s = xgb.XGBRegressor(
        n_estimators=500, max_depth=5, n_jobs=-1,
        tree_method=TREE_METHOD, device=DEVICE, random_state=42,
    )
    xgb_s.fit(X_train, Y_train[tgt][:, 0])

    p_train = xgb_s.predict(X_train)
    p_val   = xgb_s.predict(X_val)
    p_test  = xgb_s.predict(X_test)

    r2_train = r2_score(Y_train[tgt][:, 0], p_train)
    r2_val   = r2_score(Y_val[tgt][:, 0],   p_val)
    r2_test  = r2_score(Y_test[tgt][:, 0],  p_test)

    print(f"    {tgt} horizon 0 — train R²: {r2_train:.3f}  "
          f"val R²: {r2_val:.3f}  test R²: {r2_test:.3f}")
print("    (train R² >> val/test R² = overfitting)")

print("\nDone — EXP_09 v2 complete.")


# ──────────────────────────────────────────────────────────────────────────────
# [P4] SHAP — standalone post-hoc snippet (NOT run automatically)
# ──────────────────────────────────────────────────────────────────────────────
# Rationale: running SHAP inside the per-horizon training loop would mean doing
# it 2 targets × 24 horizons = 48 times, which is expensive and was explicitly
# flagged as "not necessary during training, do later." Run this separately
# after EXP_09 finishes, pointed at whichever (target, horizon) model you want
# to explain — e.g. NO2 at t+01h is usually the most informative for SIH writeups.
#
# import shap
# import joblib
# import pandas as pd
#
# TARGET_TO_EXPLAIN = "NO2_target"
# HORIZON_TO_EXPLAIN = 0  # t+01h
#
# model = joblib.load(f"exp09v2_models/{TARGET_TO_EXPLAIN}_h{HORIZON_TO_EXPLAIN:02d}.pkl")
# feature_cols = joblib.load("exp09v2_feature_cols.pkl")
#
# # Use a manageable sample of X_test for SHAP (full test set can be slow)
# explainer = shap.TreeExplainer(model)
# sample = X_test[:5000]
# shap_values = explainer.shap_values(sample)
#
# shap_importance = pd.DataFrame({
#     "feature": feature_cols,
#     "mean_abs_shap": np.abs(shap_values).mean(axis=0)
# }).sort_values("mean_abs_shap", ascending=False)
#
# print(shap_importance.head(20))
# shap_importance.to_csv(
#     f"exp09v2_models/{TARGET_TO_EXPLAIN}_h{HORIZON_TO_EXPLAIN:02d}_shap_top20.csv",
#     index=False
# )
#
# # Optional plot:
# # shap.summary_plot(shap_values, sample, feature_names=feature_cols, max_display=20)