"""
VAAYU DARSHIYA - forecast skill check
=====================================

Answers "is the data the dashboard serves accurate?" with numbers the model
report cannot give, by scoring the forecasts against the project database the
dashboard itself displays (data/vaayu_extract_7sites.csv):

  1. Serving sanity - the API replays models/predictions/*.csv, rotated by the
     anchor hour. Verify the served values really are those rows (units,
     rotation, no off-by-one).
  2. Skill vs naive baselines on the database - persistence (last observation)
     and site x hour-of-day climatology, scored over the same overlap window.
  3. Agreement between the model's own `truth` and the database, to re-confirm
     the NO2/O3 label mismatch the report documents.

Usage:
    python scripts/check_forecast_skill.py
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
EXTRACT = os.path.join(ROOT, "data", "vaayu_extract_7sites.csv")
PRED_DIR = os.path.join(ROOT, "models", "predictions")

SITE_NAMES = {
    1: "Ashok Vihar", 2: "Dwarka", 3: "Jawaharlal Nehru", 4: "Northwest Delhi",
    5: "East Delhi", 6: "Northwest Delhi 2", 7: "Northeast Delhi",
}
# site_id (models/predictions/*.csv) -> CPCB station_id (data/vaayu_extract_7sites.csv)
SITE_TO_STATION = {1: 8917, 2: 6931, 3: 5754, 4: 10485, 5: 8239, 6: 10831, 7: 8475}
# database column -> (display name, scale to serve-unit)
TARGETS = {
    "no2": ("NO2", 1.0),            # ppb
    "o3": ("O3", 1.0),              # ppb
    "sat_hcho_column": ("HCHO", 1e6),  # mol/m2 -> umol/m2
}
HORIZONS = (1, 6, 12, 24)


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < 2:
        return float("nan")
    ss = float(((y - y.mean()) ** 2).sum())
    return float("nan") if ss == 0 else 1.0 - float(((y - p) ** 2).sum()) / ss


def mae(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    return float(np.mean(np.abs(y[ok] - p[ok]))) if ok.any() else float("nan")


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def main() -> None:
    df = pd.read_csv(EXTRACT)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=False)
    df = df.sort_values(["station_id", "datetime"]).reset_index(drop=True)
    print(f"database: {len(df):,} rows, {df['datetime'].min()} .. {df['datetime'].max()}")
    print(f"sites: {sorted(df['site_id'].unique().tolist()) if 'site_id' in df.columns else sorted(df['station_id'].unique().tolist())}")

    site_col = "site_id" if "site_id" in df.columns else "station_id"
    df["hour"] = df["datetime"].dt.hour
    df["date"] = df["datetime"].dt.date

    print("\n=== 1. Served forecast vs the prediction files (rotation + units) ===")
    for tgt, (label, _scale) in TARGETS.items():
        f = os.path.join(PRED_DIR, f"exp09v2_{label}_target_predictions.csv")
        if not os.path.exists(f):
            print(f"  {label}: prediction file missing -> skipped")
            continue
        pred = pd.read_csv(f, parse_dates=["datetime"])
        sid = 1
        site = pred[pred["site_id"] == sid]
        anchors = site.groupby("datetime")["horizon"].count()
        full = anchors[anchors == 24].index
        anchor = full.max()
        rows = site[site["datetime"] == anchor].set_index("horizon")["prediction"]
        by_hour = {}
        for h in range(1, 25):
            by_hour[(anchor.hour + h) % 24] = float(rows.loc[h]) if h in rows.index else np.nan
        served = [round(by_hour.get(i, float("nan")), 2) for i in range(24)]
        direct = [round(float(rows.loc[h]), 2) if h in rows.index else None for h in range(1, 25)]
        print(f"  {label} site1 anchor={anchor} hour={anchor.hour}")
        print(f"    hour-indexed (what /api/forecast serves): {served[:6]} ... {served[-3:]}")
        print(f"    raw h1..h24 from the CSV               : {direct[:6]} ... {direct[-3:]}")

    print("\n=== 2. Database describe (are the units what the dashboard shows?) ===")
    for col, (label, scale) in TARGETS.items():
        v = df[col] * scale
        print(f"  {label:<5} n={v.notna().sum():>7,}  min={v.min():8.2f}  mean={v.mean():8.2f}  max={v.max():9.2f}")

    print("\n=== 3. Naive baseline skill on the database ===")
    print("    (scored 2024-01-01 onwards; persistence = value observed h hours earlier,")
    print("     climatology = site x hour-of-day mean fitted on data up to 2023-12-31)")
    fit = df[df["datetime"] < "2024-01-01"]
    for col, (label, scale) in TARGETS.items():
        sub = df[[site_col, "datetime", "hour", col]].copy()
        sub["v"] = sub[col] * scale
        g = sub.groupby(site_col, sort=False)["v"]
        for h in HORIZONS:
            sub[f"persist_{h}"] = g.shift(h)      # observation h hours before t
            sub[f"target_{h}"] = g.shift(-h)      # observation h hours after t
        clim = (fit.groupby([site_col, "hour"])[col].mean() * scale).rename("clim")
        sub = sub.merge(clim, left_on=[site_col, "hour"], right_index=True, how="left")
        mask = sub["datetime"] >= "2024-01-01"
        print(f"\n  {label} (serve unit, scale x{scale:g})")
        print(f"    {'h':>3} | {'persistence R2':>14} {'MAE':>8} | {'climatology R2':>14} {'MAE':>8} | {'n':>8}")
        for h in HORIZONS:
            y = sub.loc[mask, f"target_{h}"].to_numpy(float)
            ps = sub.loc[mask, f"persist_{h}"].to_numpy(float)
            pc = sub.loc[mask, "clim"].to_numpy(float)
            n = int(np.isfinite(y).sum())
            print(f"    {h:>3} | {r2(y, ps):>14.4f} {mae(y, ps):>8.2f} | {r2(y, pc):>14.4f} {mae(y, pc):>8.2f} | {n:>8,}")

    print("\n=== 4. Model truth vs database (re-check of the label mismatch) ===")
    # predictions use site_id 1..7, the extract uses the CPCB station_id.
    # `truth` in the prediction files is already in serve units (ppb / umol/m2);
    # the database column is scaled up to serve units instead.
    for col, (label, scale) in TARGETS.items():
        f = os.path.join(PRED_DIR, f"exp09v2_{label}_target_predictions.csv")
        if not os.path.exists(f):
            continue
        all_h = pd.read_csv(f, parse_dates=["datetime"])
        all_h["station_id"] = all_h["site_id"].map(SITE_TO_STATION)

        def _join(h: int) -> pd.DataFrame:
            part = all_h[all_h["horizon"] == h][["station_id", "datetime", "truth"]].copy()
            # `datetime` is the anchor; truth/prediction describe anchor + h
            part["valid_time"] = part["datetime"] + pd.Timedelta(hours=h)
            j = part.merge(
                df[[site_col, "datetime", col]].rename(columns={col: "db", "datetime": "db_datetime"}),
                left_on=["station_id", "valid_time"], right_on=[site_col, "db_datetime"], how="inner",
            )
            j["db"] = j["db"] * scale
            return j

        full = _join(1)
        if full.empty:
            print(f"  {label}: no overlap between predictions and database")
            continue
        print(
            f"  {label:<5} h+1 overlap={len(full):>8,}  corr={corr(full['truth'], full['db']):>7.4f}  "
            f"R2={r2(full['db'], full['truth']):>8.4f}  MAE={mae(full['db'], full['truth']):>7.2f}"
        )
        print("    per horizon, scored against the database:")
        for h in (1, 6, 12, 24):
            j = _join(h)
            print(
                f"      h+{h:<2} n={len(j):>8,}  R2={r2(j['db'], j['truth']):>8.4f}  "
                f"MAE={mae(j['db'], j['truth']):>7.2f}  corr={corr(j['db'], j['truth']):>7.4f}"
            )


if __name__ == "__main__":
    main()
