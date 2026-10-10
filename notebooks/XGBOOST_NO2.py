"""
VAAYU DARSHIYA — EXP_02 (corrected)
XGBoost Baseline : NO2 Prediction
Dataset : vaayu_final_ml_ready.csv
Train   : 2016–2022
Test    : 2025–2026  (2023 missing, 2024 NO2 is 100% null — both skipped)
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

os.chdir(r"C:\Users\Asus\Downloads\VAAYU_DARSHIYA_MODEL_WORK")

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_FILE   = "vaayu_final_ml_ready.csv"
TARGET      = "no2"
TRAIN_YEARS = list(range(2016, 2023))       # 2016 – 2022
TEST_YEARS  = [2025, 2026]                  # 2025 – 2026

# Must never be used as features — cause leakage or are identifiers
DROP_ALWAYS = [
    "aqi_no2", "aqi_o3", "aqi_computed", "aqi_category", "aqi_pm25",
    "station_name", "station_id", "date", "datetime",
    "lulc_class_name", "season",
]

# Other targets — not predicting now
OTHER_TARGETS = ["o3", "pm25"]

MIN_TEST_ROWS = 100     # stations with fewer test rows excluded from per-station stats

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
print("[1] Loading data …")
df = pd.read_csv(DATA_FILE, low_memory=False)
print(f"    Full shape : {df.shape}")
print(f"\n    Year distribution:")
print(df["year"].value_counts().sort_index().to_string())

# ── 2. SPLIT ──────────────────────────────────────────────────────────────────
print("\n[2] Splitting …")
train_df = df[df["year"].isin(TRAIN_YEARS)].copy()
test_df  = df[df["year"].isin(TEST_YEARS)].copy()

print(f"    Train ({min(TRAIN_YEARS)}–{max(TRAIN_YEARS)}) : {len(train_df):,} rows")
print(f"    Test  ({TEST_YEARS})     : {len(test_df):,} rows")

# Drop rows where target is missing
train_df = train_df.dropna(subset=[TARGET])
test_df  = test_df.dropna(subset=[TARGET])

print(f"    Train after dropna : {len(train_df):,}")
print(f"    Test  after dropna : {len(test_df):,}")

if len(test_df) == 0:
    raise ValueError("Test set has no valid NO2 rows. Check your data.")

# ── 3. FEATURES ───────────────────────────────────────────────────────────────
print("\n[3] Preparing features …")
cols_to_drop = set(DROP_ALWAYS + OTHER_TARGETS + [TARGET])
feature_cols = [c for c in df.columns if c not in cols_to_drop]

# Convert any remaining object/string columns to numeric category codes
for col in feature_cols:
    if train_df[col].dtype == "object":
        combined   = pd.concat([train_df[col], test_df[col]], axis=0).astype("category")
        categories = combined.cat.categories
        train_df[col] = pd.Categorical(train_df[col], categories=categories).codes.astype(float)
        test_df[col]  = pd.Categorical(test_df[col],  categories=categories).codes.astype(float)
        train_df[col].replace(-1, np.nan, inplace=True)
        test_df[col].replace(-1, np.nan, inplace=True)

X_train = train_df[feature_cols]
y_train = train_df[TARGET]
X_test  = test_df[feature_cols]
y_test  = test_df[TARGET]

print(f"    Feature count : {len(feature_cols)}")
print(f"    X_train shape : {X_train.shape}")
print(f"    X_test  shape : {X_test.shape}")

# ── 4. TRAIN ──────────────────────────────────────────────────────────────────
print("\n[4] Training XGBoost …")
model = xgb.XGBRegressor(
    n_estimators          = 1000,
    learning_rate         = 0.05,
    max_depth             = 7,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    min_child_weight      = 5,
    reg_alpha             = 0.1,
    reg_lambda            = 1.0,
    random_state          = 42,
    n_jobs                = -1,
    tree_method           = "hist",
    device                = "cuda",     # uses your RTX 3050 — change to "cpu" if error
    early_stopping_rounds = 50,
    eval_metric           = "rmse",
    verbosity             = 1,
)

model.fit(
    X_train, y_train,
    eval_set  = [(X_test, y_test)],
    verbose   = 100,
)

print(f"\n    Best iteration : {model.best_iteration}")

# ── 5. EVALUATE ───────────────────────────────────────────────────────────────
print("\n[5] Evaluating …")
y_pred = np.clip(model.predict(X_test), 0, None)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
mean_no2 = y_test.mean()

print(f"""
=============================================
  EXP_02 Results — XGBoost NO2 Baseline
=============================================
  MAE        : {mae:.4f}  µg/m³
  RMSE       : {rmse:.4f} µg/m³
  R²         : {r2:.4f}
  Mean NO2   : {mean_no2:.2f} µg/m³
  MAE/Mean   : {mae/mean_no2*100:.1f}%
  (target: below 15% = good, below 10% = strong)
=============================================
""")

# ── 6. FEATURE IMPORTANCE ─────────────────────────────────────────────────────
print("[6] Feature importance …")
fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(f"\n  Top 25 features:\n{fi.head(25).to_string()}")
fi.reset_index().rename(columns={"index": "feature", 0: "importance"}).to_csv(
    "exp02_no2_feature_importance.csv", index=False
)
print("  Saved → exp02_no2_feature_importance.csv")

# ── 7. PER-STATION PERFORMANCE ────────────────────────────────────────────────
print("\n[7] Per-station performance …")
test_df = test_df.copy()
test_df["pred"] = y_pred

station_stats = (
    test_df.groupby("station_name")
    .apply(lambda g: pd.Series({
        "mae" : mean_absolute_error(g[TARGET], g["pred"]),
        "rmse": np.sqrt(mean_squared_error(g[TARGET], g["pred"])),
        "r2"  : r2_score(g[TARGET], g["pred"]) if len(g) > 1 else np.nan,
        "n"   : len(g),
    }))
    .query(f"n >= {MIN_TEST_ROWS}")
    .round(3)
)

print(f"\n  Best 5 stations (lowest MAE):\n{station_stats.nsmallest(5, 'mae').to_string()}")
print(f"\n  Worst 5 stations (highest MAE):\n{station_stats.nlargest(5, 'mae').to_string()}")
station_stats.to_csv("exp02_no2_station_metrics.csv")
print("  Saved → exp02_no2_station_metrics.csv")

# ── 8. PLOTS ──────────────────────────────────────────────────────────────────
print("\n[8] Generating plots …")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"EXP_02 — XGBoost NO₂ Baseline  (R²={r2:.3f}  MAE={mae:.2f}  RMSE={rmse:.2f})", fontsize=12)

# Scatter
ax = axes[0]
ax.scatter(y_test, y_pred, alpha=0.1, s=3, color="#3B82F6")
lims = [0, max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, "r--", lw=1.2)
ax.set_xlabel("Actual NO₂ (µg/m³)")
ax.set_ylabel("Predicted NO₂ (µg/m³)")
ax.set_title("Predicted vs Actual")

# Residuals
residuals = y_test.values - y_pred
ax = axes[1]
ax.hist(residuals, bins=100, color="#8B5CF6", alpha=0.7, edgecolor="none")
ax.axvline(0, color="red", lw=1.2, linestyle="--")
ax.set_xlabel("Residual (Actual − Predicted)")
ax.set_ylabel("Count")
ax.set_title(f"Residuals  (mean={residuals.mean():.2f})")

# Feature importance
ax = axes[2]
fi.head(20).sort_values().plot(kind="barh", ax=ax, color="#10B981")
ax.set_xlabel("Importance")
ax.set_title("Top 20 features")
ax.tick_params(axis="y", labelsize=8)

plt.tight_layout()
plt.savefig("exp02_no2_results.png", dpi=150, bbox_inches="tight")
print("  Saved → exp02_no2_results.png")

# ── 9. SAVE MODEL ─────────────────────────────────────────────────────────────
model.save_model("exp02_no2_xgb_model.json")
print("  Model saved → exp02_no2_xgb_model.json")

print("""
=======================================================
  DONE
  Train : 2016–2022
  Test  : 2025–2026
  File  : vaayu_final_ml_ready.csv (564K rows)

  Next → share MAE/RMSE and move to EXP_03 (O3)
=======================================================
""")