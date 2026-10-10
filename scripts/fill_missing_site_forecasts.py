"""Generate a genuine EXP_12 forecast for sites missing from the prediction CSVs.

Station 5754 (site 3, Jawaharlal Nehru Stadium) has no rows after 2023-12-31 in
``data/vaayu_final_v3.csv``, so EXP_12's real 2025 test window contained nothing
for it and ``models/predictions/exp09v2_*_predictions.csv`` have no site-3 rows.
The backend then serves a zero placeholder anchored on *now* — which the
dashboard would display as "clean air measured today" for a station that has
reported nothing since 2023.

This script runs the *already trained* EXP_12 models on each missing site's last
common feature-complete row (its final anchor) and appends those 24 horizons to
all three prediction CSVs. Truth stays NaN (valid times fall after the station's
last observation); ``scripts/evaluate_accuracy.py`` scores on finite pairs only,
so the accuracy reports are unaffected — the result is an honest model run
labelled with the station's real "Model run" date (2023-12-31) in the UI.

Mirrors notebooks/exp_12_retrain_db.py sections 1-4 (keep in sync). Idempotent:
sites that already have prediction rows are skipped.
"""

import os
import shutil
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover - non-replaceable stdio
    pass

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_CSV = os.path.join(ROOT_DIR, "data", "vaayu_final_v3.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "models", "exp09v2_models")
SCALER_DIR = os.path.join(ROOT_DIR, "models", "scalers")
PRED_DIR = os.path.join(ROOT_DIR, "models", "predictions")

# ── CONFIG (identical to notebooks/exp_12_retrain_db.py) ─────────────────────
TARGETS = {
    "NO2_target": ("no2", 1.0),                    # ppb
    "O3_target": ("o3", 1.0),                      # ppb
    "HCHO_target": ("sat_hcho_column", 1e6),       # mol/m² -> µmol/m²
}
FORECAST_LEN = 24

STATION_TO_SITE = {
    "8917": 1,   # Ashok Vihar
    "6931": 2,   # Dwarka
    "5754": 3,   # Jawaharlal Nehru
    "10485": 4,  # Northwest Delhi
    "8239": 5,   # East Delhi
    "10831": 6,  # Northwest Delhi 2
    "8475": 7,   # Northeast Delhi
}
SITE_TO_STATION = {v: k for k, v in STATION_TO_SITE.items()}

TRAIN_END = pd.Timestamp("2025-01-01")
FFILL_LIMIT = 8
LAG_HOURS = [1, 2, 3, 6, 12, 24, 48, 72]
FWD_HOURS = [1, 2, 3, 6, 12, 24]
ROLL_WINDOWS = [6, 12, 24, 48, 72, 168]

WEATHER_CORE = [
    "temperature_2m_c", "relative_humidity",
    "wind_u_10m", "wind_v_10m", "rainfall_mm",
]
WEATHER_EXTRA = [
    "boundary_layer_height", "ventilation_coeff", "stagnation_index",
    "temp_inversion_proxy", "solar_radiation",
]
OBSERVED = [
    "pm25", "sat_no2_column", "sat_o3_column", "aod_550nm",
    "fire_pixel_count", "stubble_burning_flag", "lst_celsius",
    "is_daytime", "solar_zenith_angle", "is_holiday",
]
DRIVER_FLAGS = ["low_ventilation_flag", "blh_critical_flag", "inversion_flag"]
STATIC_SITE = ["nox_t_km2_yr", "night_light_radiance"]

RECIPE = "EXT"       # EXP_12 winner for all three targets


def month_to_season(m: int) -> str:
    if m in (12, 1, 2):
        return "Winter"
    if m in (3, 4, 5):
        return "Summer"
    if m in (6, 7, 8, 9):
        return "Monsoon"
    return "Post-monsoon"


# ── 1. LOAD THE 7 SERVED STATIONS (same projection as EXP_12) ────────────────
if not os.path.exists(SRC_CSV):
    sys.exit(f"missing database: {SRC_CSV}")

usecols = (["station_id", "datetime"] + [c for c, _ in TARGETS.values()]
           + WEATHER_CORE + WEATHER_EXTRA + OBSERVED + DRIVER_FLAGS + STATIC_SITE)
usecols = list(dict.fromkeys(usecols))
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
print(f"[1] Loaded {len(df):,} rows")

df["site_id"] = df["station_id"].map(STATION_TO_SITE).astype(int)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values(["site_id", "datetime"]).reset_index(drop=True)
df["month"] = df["datetime"].dt.month
df["hour"] = df["datetime"].dt.hour

# ── 2. SEGMENTS + 3. TARGETS + bridge ────────────────────────────────────────
df["segment"] = (
    df.groupby("site_id")["datetime"]
      .transform(lambda s: (s.diff() != pd.Timedelta(hours=1)).cumsum())
)
for tgt, (col, scale) in TARGETS.items():
    df[tgt] = pd.to_numeric(df[col], errors="coerce") * scale
    df[f"{tgt}_feat"] = df.groupby(["site_id", "segment"], sort=False)[tgt] \
                          .transform(lambda s: s.ffill(limit=FFILL_LIMIT))

# ── 4. FEATURES (identical construction order to EXP_12) ─────────────────────
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
df["wind_dir"] = np.arctan2(df["wind_v_10m"], df["wind_u_10m"])
df["wind_dir_sin"] = np.sin(df["wind_dir"]).astype(np.float32)
df["wind_dir_cos"] = np.cos(df["wind_dir"]).astype(np.float32)

df["season"] = df["month"].apply(month_to_season)

site_dummies = pd.get_dummies(df["site_id"], prefix="site", dtype=np.float32)
for c in site_dummies.columns:
    df[c] = site_dummies[c]

seg_gb = df.groupby(["site_id", "segment"], sort=False)


def build_weather_lags(cols):
    lag_cols = []
    for col in cols:
        for lag in LAG_HOURS:
            cname = f"{col}_lag{lag}"
            df[cname] = seg_gb[col].shift(lag)
            lag_cols.append(cname)
        for fwd in FWD_HOURS:
            cname = f"{col}_fwd{fwd}"
            df[cname] = seg_gb[col].shift(-fwd)
            lag_cols.append(cname)
    return lag_cols


weather_lag_core = build_weather_lags(WEATHER_CORE)
weather_lag_ext = weather_lag_core + build_weather_lags(WEATHER_EXTRA)

target_lag_cols, target_roll_cols, target_std_cols = [], [], []
for tgt in TARGETS:
    for lag in LAG_HOURS:
        cname = f"{tgt}_feat_lag{lag}"
        df[cname] = seg_gb[f"{tgt}_feat"].shift(lag)
        target_lag_cols.append(cname)
    for window in ROLL_WINDOWS:
        cname = f"{tgt}_feat_roll{window}"
        df[cname] = seg_gb[f"{tgt}_feat"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        target_roll_cols.append(cname)
    cname = f"{tgt}_feat_roll_std24"
    df[cname] = seg_gb[f"{tgt}_feat"].transform(
        lambda x: x.shift(1).rolling(24, min_periods=2).std()
    )
    target_std_cols.append(cname)

BASE_TIME = ["hour_sin", "hour_cos", "month_sin", "month_cos",
             "dow_sin", "dow_cos", "doy_sin", "doy_cos",
             "wind_speed", "wind_dir_sin", "wind_dir_cos"]
SITE_DUMMIES = list(site_dummies.columns)


def feature_list(recipe: str, tgt: str) -> list:
    tgt_lags = [c for c in target_lag_cols if c.startswith(tgt)]
    tgt_rolls = [c for c in target_roll_cols if c.startswith(tgt)]
    if recipe == "BASE":
        return (WEATHER_CORE + BASE_TIME + weather_lag_core
                + tgt_lags + tgt_rolls)
    tgt_std = [c for c in target_std_cols if c.startswith(tgt)]
    return (WEATHER_CORE + WEATHER_EXTRA + BASE_TIME + OBSERVED
            + DRIVER_FLAGS + STATIC_SITE + SITE_DUMMIES
            + weather_lag_ext
            + [f"{tgt}_feat"]
            + tgt_lags + tgt_rolls + tgt_std)


def needed_cols(tgt: str) -> list:
    return sorted(set(WEATHER_CORE + WEATHER_EXTRA + OBSERVED
                      + DRIVER_FLAGS + STATIC_SITE
                      + [f"{tgt}_feat"]
                      + [c for c in target_lag_cols if c.startswith(tgt)]
                      + [c for c in target_roll_cols if c.startswith(tgt)]
                      + [c for c in target_std_cols if c.startswith(tgt)]))


# ── 5. WHICH SITES ARE MISSING FROM THE PREDICTION CSVs? ─────────────────────
missing = {}          # tgt -> sorted list of site_ids absent from its CSV
for tgt in TARGETS:
    path = os.path.join(PRED_DIR, f"exp09v2_{tgt}_predictions.csv")
    pred = pd.read_csv(path, usecols=["site_id"])
    have = set(pred["site_id"].unique())
    missing[tgt] = sorted(set(range(1, 8)) - have)

all_missing = sorted({s for v in missing.values() for s in v})
print(f"[5] sites missing per target: {missing}")
if not all_missing:
    print("    nothing to do - every site already has prediction rows")
    sys.exit(0)

# One common anchor per site: the last row that is feature-complete for ALL
# three targets, so the backend serves a single consistent "Model run" date.
common_mask = df[list(dict.fromkeys(
    sum((needed_cols(t) for t in TARGETS), [])))].notna().all(axis=1)
anchors = (df[common_mask]
           .groupby("site_id")["datetime"].max().to_dict())
print(f"[5] last feature-complete anchor per site: "
      + ", ".join(f"{s}: {str(anchors[s])[:16]}" for s in sorted(anchors)))

# ── 6. PER TARGET: re-fit scaler (verification) + predict at the anchor ──────
added_rows = []
for tgt, (src_col, scale) in TARGETS.items():
    if not missing[tgt]:
        print(f"\n[{tgt}] all sites present - skipped")
        continue

    cols = feature_list(RECIPE, tgt)
    print(f"\n[{tgt}] {len(cols)} features ({RECIPE} recipe)")

    # feature-complete rows + train window, exactly as in EXP_12
    mask = df[needed_cols(tgt)].notna().all(axis=1)
    tdf = df[mask].reset_index(drop=True)
    train_df = tdf[tdf["datetime"] < TRAIN_END]

    # EXP_12 dumped NO2 and O3 scalers to the same file, so the on-disk
    # exp09v2_* pair belongs to O3 (written last) and NO2's was overwritten.
    # StandardScaler is deterministic, so re-fitting on the identical train
    # rows reproduces it exactly - verified against the artefacts below.
    scaler = StandardScaler().fit(train_df[cols])

    if tgt == "HCHO_target":
        disk_pkl = os.path.join(SCALER_DIR, "exp11_hcho_feat_scaler.pkl")
        disk_cols = os.path.join(SCALER_DIR, "exp11_hcho_feature_cols.pkl")
    else:
        disk_pkl = os.path.join(SCALER_DIR, "exp09v2_feat_scaler.pkl")
        disk_cols = os.path.join(SCALER_DIR, "exp09v2_feature_cols.pkl")
    try:
        disk = joblib.load(disk_pkl)
        diff = float(np.nanmax(np.abs(disk.mean_ - scaler.mean_)))
        print(f"    scaler re-fit vs artefact {os.path.basename(disk_pkl)}: "
              f"max|dmean|={diff:.3e}  (identical if the artefact is this "
              f"target's - NO2's was overwritten by O3)")
        on_cols = joblib.load(disk_cols)
        print(f"    feature cols vs artefact: "
              f"{'identical' if on_cols == cols else 'DIFFERENT (expected for NO2)'}")
    except FileNotFoundError:
        print("    (no scaler artefact on disk - nothing to verify against)")

    for site in missing[tgt]:
        anchor = pd.Timestamp(anchors[site])
        row = tdf[(tdf["site_id"] == site) & (tdf["datetime"] == anchor)]
        if row.empty:
            print(f"    site {site}: no feature-complete row at {anchor} - SKIPPED")
            continue
        x = scaler.transform(row[cols])

        preds = []
        for h in range(FORECAST_LEN):
            model = joblib.load(os.path.join(MODEL_DIR, f"{tgt}_h{h:02d}.pkl"))
            p = float(np.clip(model.predict(x)[0], 0, None))
            preds.append(p)

        for h in range(FORECAST_LEN):
            added_rows.append({
                "tgt": tgt,
                "site_id": site,
                "datetime": anchor,
                "horizon": h + 1,
                "prediction": preds[h],
                "truth": np.nan,
            })
        print(f"    site {site}: anchor {str(anchor)[:16]} "
              f"({SITE_TO_STATION.get(site)}) - predictions "
              f"{min(preds):.1f} .. {max(preds):.1f} "
              f"(mean {sum(preds)/len(preds):.1f})")

if not added_rows:
    print("\nnothing generated")
    sys.exit(0)

# ── 7. APPEND TO THE PREDICTION CSVs (with a temp backup) ────────────────────
new = pd.DataFrame(added_rows)
backup_dir = tempfile.mkdtemp(prefix="pred_backup_")
print(f"\n[7] appending {len(new)} rows; pre-append copies in {backup_dir}")

for tgt in TARGETS:
    sub = new[new["tgt"] == tgt]
    if sub.empty:
        continue
    path = os.path.join(PRED_DIR, f"exp09v2_{tgt}_predictions.csv")
    shutil.copy2(path, os.path.join(backup_dir, os.path.basename(path)))
    pred = pd.read_csv(path)
    add = sub[["site_id", "datetime", "horizon", "prediction", "truth"]].copy()
    add["datetime"] = pd.Timestamp(anchors[int(add["site_id"].iloc[0])])
    out = pd.concat([pred, add], ignore_index=True)
    out.to_csv(path, index=False)
    print(f"    {os.path.basename(path)}: {len(pred):,} -> {len(out):,} rows "
          f"(site {sorted(sub['site_id'].unique())} added, "
          f"anchor {str(pd.Timestamp(add['datetime'].iloc[0]))[:16]})")

print("\ndone - restart the backend to pick up the new prediction CSVs")
