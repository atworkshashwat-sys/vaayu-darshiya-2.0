"""
VAAYU DARSHIYA - old vs retrained model comparison
==================================================

Scores the backed-up EXP_09v2/EXP_11 models against the EXP_12 retrain on the
same real-data window so the before/after is apples-to-apples:

  1. HCHO: old model vs new model, paired on identical (site, anchor, horizon)
     rows inside the new test window (2025-04-01+), scored against the
     project database at valid time (truth == database for HCHO by
     construction, so prediction-vs-truth is prediction-vs-database).
  2. NO2 / O3: the old models hold no 2025 predictions at all (their test set
     ends 2024-06-27), so the honest comparison is new-model skill against the
     naive baselines on the new test window - printed here for the record.
  3. Serving check: latest complete anchor per target after the retrain.

Usage:
    python scripts/compare_retrain.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED = os.path.join(ROOT, "models", "predictions")
BACKUP = os.path.join(ROOT, "models", "_backup_pre_retrain_20261009", "predictions")
EXTRACT = os.path.join(ROOT, "data", "vaayu_extract_7sites.csv")

SITE_TO_STATION = {1: 8917, 2: 6931, 3: 5754, 4: 10485, 5: 8239, 6: 10831, 7: 8475}
TARGETS = {"NO2_target": ("no2", 1.0), "O3_target": ("o3", 1.0),
           "HCHO_target": ("sat_hcho_column", 1e6)}
HORIZONS = (1, 6, 12, 24)


def score(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < 2:
        return {"r2": np.nan, "mae": np.nan, "n": len(y)}
    ss = ((y - y.mean()) ** 2).sum()
    return {"r2": 1.0 - ((y - p) ** 2).sum() / ss if ss else np.nan,
            "mae": float(np.abs(y - p).mean()), "n": int(len(y))}


def load(path):
    d = pd.read_csv(path, parse_dates=["datetime"])
    d["valid"] = d["datetime"] + pd.to_timedelta(d["horizon"], unit="h")
    return d


def main() -> None:
    db = pd.read_csv(EXTRACT)
    db["datetime"] = pd.to_datetime(db["datetime"])

    print("=" * 72)
    print("1. Serving check after the retrain (latest complete anchor)")
    print("=" * 72)
    for tgt in TARGETS:
        f = os.path.join(PRED, f"exp09v2_{tgt}_predictions.csv")
        if not os.path.exists(f):
            print(f"  {tgt}: MISSING")
            continue
        d = pd.read_csv(f, parse_dates=["datetime"])
        anchors = d.groupby(["site_id", "datetime"]).size()
        full = anchors[anchors == 24]
        print(f"  {tgt:<12} anchors={d['datetime'].nunique():,}  "
              f"window={str(d['datetime'].min())[:10]}..{str(d['datetime'].max())[:10]}  "
              f"latest full anchor={full.index.get_level_values('datetime').max()}")

    print()
    print("=" * 72)
    print("2. HCHO old vs new - paired on identical rows (real 2025 test window)")
    print("=" * 72)
    old_f = os.path.join(BACKUP, "exp09v2_HCHO_target_predictions.csv")
    new_f = os.path.join(PRED, "exp09v2_HCHO_target_predictions.csv")
    if os.path.exists(old_f) and os.path.exists(new_f):
        old = load(old_f)
        new = load(new_f)
        win = new["datetime"].min()
        old = old[old["datetime"] >= win]
        keys = ["site_id", "datetime", "horizon"]
        paired = old.merge(new, on=keys, suffixes=("_old", "_new"))
        print(f"  window: {str(win)[:10]}+  paired rows: {len(paired):,}")
        print(f"  {'h':>3} | {'old R2':>8} {'old MAE':>8} | {'new R2':>8} {'new MAE':>8} | {'n':>8}")
        for h in HORIZONS:
            p = paired[paired["horizon"] == h]
            so = score(p["truth_old"], p["prediction_old"])
            sn = score(p["truth_new"], p["prediction_new"])
            print(f"  {h:>3} | {so['r2']:>8.4f} {so['mae']:>8.3f} | "
                  f"{sn['r2']:>8.4f} {sn['mae']:>8.3f} | {so['n']:>8,}")
        # all horizons pooled
        so = score(paired["truth_old"], paired["prediction_old"])
        sn = score(paired["truth_new"], paired["prediction_new"])
        print(f"  all | {so['r2']:>8.4f} {so['mae']:>8.3f} | "
              f"{sn['r2']:>8.4f} {sn['mae']:>8.3f} | {so['n']:>8,}")
    else:
        print("  old or new HCHO predictions missing - skipped")

    print()
    print("=" * 72)
    print("3. NO2 / O3 new model vs naive baselines (real test window)")
    print("=" * 72)
    print("  (the old NO2/O3 models hold no 2025 predictions - their test set")
    print("   ends 2024-06-27 - so their only scorecard is the old window: DB")
    print("   R2 = -0.063 (NO2) / +0.097 (O3) from models/ACCURACY.md)")
    for tgt, (col, scale) in TARGETS.items():
        f = os.path.join(PRED, f"exp09v2_{tgt}_predictions.csv")
        if not os.path.exists(f):
            continue
        if tgt == "HCHO_target":
            continue
        d = load(f)
        # persistence = db value at the anchor time; target = db value at valid
        dbt = db.set_index(["station_id", "datetime"])[col] * scale
        keys = d[["site_id", "datetime"]].copy()
        keys["station_id"] = keys["site_id"].map(SITE_TO_STATION)
        d["persist"] = [dbt.get((s, t), np.nan)
                        for s, t in zip(keys["station_id"], keys["datetime"])]
        d["db_valid"] = [dbt.get((s, t), np.nan)
                         for s, t in zip(keys["station_id"], d["valid"])]
        print(f"\n  {tgt}:")
        for h in HORIZONS:
            p = d[d["horizon"] == h]
            sm = score(p["db_valid"], p["prediction"])
            sp = score(p["db_valid"], p["persist"])
            print(f"    h+{h:<2} model R2={sm['r2']:>8.4f} MAE={sm['mae']:>7.2f} | "
                  f"persistence R2={sp['r2']:>8.4f} MAE={sp['mae']:>7.2f} | n={sm['n']:,}")


if __name__ == "__main__":
    main()
