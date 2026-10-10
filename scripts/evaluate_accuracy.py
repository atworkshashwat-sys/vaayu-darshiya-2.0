"""
VAAYU DARSHIYA — model accuracy evaluation
==========================================

Runs the production XGBoost predictions (EXP_12 retrain — artefact file names
keep the legacy `exp09v2_` prefix) through two evaluations and writes the
artefacts the backend and the Model Details tab read:

  1. Model's own test set
       prediction vs the `truth` column inside models/predictions/*.csv
       (this is what results/exp09v2_xgb_*_results.csv already reports).

  2. Project database
       prediction vs the observed `no2` / `o3` in data/vaayu_extract_7sites.csv
       (the 7 stations projected out of data/vaayu_final_v3.csv) over the
       period where the two sources overlap.

It also generates the two breakdown CSVs that were never committed but are
expected by backend/api.py:

       models/results/exp09v2_{NO2,O3,HCHO}_target_error_by_season.csv
       models/results/exp09v2_{NO2,O3,HCHO}_target_error_by_site.csv

Outputs:
       models/results/exp09v2_database_validation.json   (→ /api/metrics)
       models/ACCURACY.md                                (human readable report)

Usage:
    python scripts/evaluate_accuracy.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# Windows consoles default to cp1252, which cannot encode the arrows/markdown
# used in the report and in the log lines below.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover - non-replacable stdio
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_DIR = os.path.join(ROOT, "models", "predictions")
RESULT_DIR = os.path.join(ROOT, "models", "results")
EXTRACT_CSV = os.path.join(ROOT, "data", "vaayu_extract_7sites.csv")
REPORT_MD = os.path.join(ROOT, "models", "ACCURACY.md")
VALIDATION_JSON = os.path.join(RESULT_DIR, "exp09v2_database_validation.json")

TARGETS = {"NO2": "no2", "O3": "o3", "HCHO": "sat_hcho_column"}
# The database stores HCHO as a satellite column in mol/m² while the
# predictions are served in µmol/m² (×1e6; see notebooks/exp_11_hcho_xgboost.py
# — EXP_12 keeps the same serving unit).
# Scale the database side so the join scores like-for-like.
TARGET_SCALE = {"NO2": 1.0, "O3": 1.0, "HCHO": 1e6}
SITE_TO_STATION = {
    1: "8917", 2: "6931", 3: "5754", 4: "10485",
    5: "8239", 6: "10831", 7: "8475",
}
SITE_NAMES = {
    1: "Ashok Vihar", 2: "Dwarka", 3: "Jawaharlal Nehru", 4: "Northwest Delhi",
    5: "East Delhi", 6: "Northwest Delhi 2", 7: "Northeast Delhi",
}


# ── metrics helpers ────────────────────────────────────────────────────
def _r2(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) < 2:
        return float("nan")
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - np.sum((y - p) ** 2) / ss_tot)


def score(y, p) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) == 0:
        return {"n": 0, "r2": None, "mae": None, "rmse": None, "bias": None}
    return {
        "n": int(len(y)),
        # NaN (e.g. zero-variance series) must become null: the artefacts are
        # parsed with JSON.parse in the browser, which rejects NaN literals.
        "r2": _finite(_r2(y, p)),
        "mae": _finite(np.mean(np.abs(y - p)), 2),
        "rmse": _finite(np.sqrt(np.mean((y - p) ** 2)), 2),
        "bias": _finite(np.mean(p - y), 2),
    }


def _finite(value, digits: int = 4):
    """round() a metric, mapping NaN/±inf to None for strict JSON output."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return round(v, digits) if np.isfinite(v) else None


def _corr(a, b, digits: int = 4):
    """Pearson correlation over finite pairs only; None when undefined."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        return _finite(np.corrcoef(a, b)[0, 1], digits)


def json_safe(obj):
    """Recursively map NaN/±inf to null — JSON.parse rejects NaN literals."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)) and not np.isfinite(obj):
        return None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def month_to_season(m: int) -> str:
    if m in (12, 1, 2):
        return "Winter"
    if m in (3, 4, 5):
        return "Summer"
    if m in (6, 7, 8, 9):
        return "Monsoon"
    return "Post-monsoon"


# ── 1. load predictions ────────────────────────────────────────────────
def load_predictions() -> dict:
    out = {}
    for tgt in TARGETS:
        path = os.path.join(PRED_DIR, f"exp09v2_{tgt}_target_predictions.csv")
        if not os.path.exists(path):
            # e.g. HCHO before EXP_11 has run on this checkout
            print(f"  {tgt}: skipped — {os.path.basename(path)} not found")
            continue
        df = pd.read_csv(path, parse_dates=["datetime"])
        df["season"] = df["datetime"].dt.month.apply(month_to_season)
        # predictions store horizon as 1..24; results.csv stores 0..23
        df["horizon0"] = df["horizon"] - 1
        out[tgt] = df
        print(f"  {tgt}: {len(df):,} rows "
              f"{df['datetime'].min().date()} → {df['datetime'].max().date()}")
    return out


# ── 2. breakdown CSVs (season + site) ──────────────────────────────────
def write_breakdowns(preds: dict) -> None:
    os.makedirs(RESULT_DIR, exist_ok=True)
    for tgt, df in preds.items():
        season_rows, site_rows = [], []

        for (season, h0), g in df.groupby(["season", "horizon0"]):
            s = score(g["truth"], g["prediction"])
            if s["n"] < 2:      # e.g. site 3's stale anchor has no truth
                continue
            season_rows.append({"season": season, "horizon": int(h0), **s})

        for (sid, h0), g in df.groupby(["site_id", "horizon0"]):
            s = score(g["truth"], g["prediction"])
            if s["n"] < 2:
                continue
            site_rows.append({"site_id": int(sid), "horizon": int(h0), **s})

        # Seasonal rows in calendar order, site rows by site then horizon.
        order = {"Winter": 0, "Summer": 1, "Monsoon": 2, "Post-monsoon": 3}
        season_rows.sort(key=lambda r: (order.get(r["season"], 9), r["horizon"]))
        site_rows.sort(key=lambda r: (r["site_id"], r["horizon"]))

        s_path = os.path.join(RESULT_DIR, f"exp09v2_{tgt}_target_error_by_season.csv")
        t_path = os.path.join(RESULT_DIR, f"exp09v2_{tgt}_target_error_by_site.csv")
        pd.DataFrame(season_rows)[["season", "horizon", "r2", "mae", "rmse", "bias", "n"]] \
            .to_csv(s_path, index=False)
        pd.DataFrame(site_rows)[["site_id", "horizon", "r2", "mae", "rmse", "bias", "n"]] \
            .to_csv(t_path, index=False)
        print(f"  wrote {os.path.basename(s_path)} ({len(season_rows)} rows)")
        print(f"  wrote {os.path.basename(t_path)} ({len(site_rows)} rows)")


# ── 3. accuracy against the project database ───────────────────────────
def database_validation(preds: dict) -> dict:
    if not os.path.exists(EXTRACT_CSV):
        raise FileNotFoundError(
            f"{EXTRACT_CSV} not found — run the backend once, or "
            "`python -c \"import sys; sys.path.insert(0,'backend'); "
            "import database; database.build_extract()\"` to build it."
        )

    db = pd.read_csv(EXTRACT_CSV, dtype={"station_id": str})
    db["datetime"] = pd.to_datetime(db["datetime"])
    db["site_id"] = (
        db["station_id"].map({v: k for k, v in SITE_TO_STATION.items()}).astype("float")
    )
    db = db.dropna(subset=["site_id"])
    db["site_id"] = db["site_id"].astype(int)

    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "database": {
            "extract": os.path.relpath(EXTRACT_CSV, ROOT).replace("\\", "/"),
            "rows": int(len(db)),
            "min_date": str(db["datetime"].min().date()),
            "max_date": str(db["datetime"].max().date()),
        },
        "targets": {},
    }

    for tgt, col in TARGETS.items():
        if tgt not in preds:
            continue
        pred = preds[tgt]
        # like-for-like units: the database side is scaled (HCHO: mol/m² → µmol/m²)
        scale = TARGET_SCALE.get(tgt, 1.0)
        db_side = db[["site_id", "datetime", col]].copy()
        if scale != 1.0:
            db_side[col] = db_side[col] * scale
        db_side = db_side.rename(columns={"datetime": "db_datetime"})
        # `datetime` in the prediction files is the ANCHOR (issue time); both
        # `truth` and `prediction` describe the valid time datetime + horizon,
        # so the database side must be joined at that offset. Joining on the
        # anchor shifts every row by h hours and understates agreement.
        pred = pred.copy()
        pred["valid_time"] = pred["datetime"] + pd.to_timedelta(pred["horizon"], unit="h")
        joined = pred.merge(
            db_side,
            left_on=["site_id", "valid_time"],
            right_on=["site_id", "db_datetime"],
            how="inner",
        ).dropna(subset=[col, "prediction"])

        # (a) how well the model does on the labels it was scored against
        test_scores = {
            "t1h": score(pred.loc[pred.horizon == 1, "truth"],
                         pred.loc[pred.horizon == 1, "prediction"]),
            "t24h": score(pred.loc[pred.horizon == 24, "truth"],
                          pred.loc[pred.horizon == 24, "prediction"]),
            "all": score(pred["truth"], pred["prediction"]),
        }

        # (b) how well the same predictions do against the database
        db_scores = {
            "t1h": score(joined.loc[joined.horizon == 1, col],
                         joined.loc[joined.horizon == 1, "prediction"]),
            "t24h": score(joined.loc[joined.horizon == 24, col],
                          joined.loc[joined.horizon == 24, "prediction"]),
            "all": score(joined[col], joined["prediction"]),
        }

        # (c) do the two sources even describe the same series?
        #     compare each prediction row's `truth` with the database value
        #     at the identical site + timestamp. HCHO truth carries gaps, so
        #     the correlation must drop non-finite pairs (np.corrcoef would
        #     propagate a NaN through the whole matrix).
        agree = score(joined["truth"], joined[col])
        corr = _corr(joined["truth"], joined[col])

        # (d) lag scan — rule out a simple time-shift between the sources
        h1 = (
            joined[joined.horizon == 1]
            .dropna(subset=["truth", col])
            .sort_values("datetime")
        )
        lag_corr = {}
        for lag in (0, 1, -1):
            a = h1["truth"].astype(float).to_numpy()
            b = h1[col].astype(float).to_numpy()
            if lag > 0:
                lag_corr[f"{lag:+d}h"] = _corr(a[:-lag], b[lag:])
            elif lag < 0:
                lag_corr[f"{lag:+d}h"] = _corr(a[-lag:], b[:lag])
            else:
                lag_corr["0h"] = _corr(a, b)

        per_site = []
        for sid, g in joined[joined.horizon == 1].groupby("site_id"):
            s = score(g[col], g["prediction"])
            per_site.append({
                "site_id": int(sid),
                "site_name": SITE_NAMES.get(int(sid), f"Site {int(sid)}"),
                **s,
            })

        verdict = (
            "model tracks the database" if (db_scores["all"]["r2"] or 0) > 0.3
            else "model does not track the database"
        )

        result["targets"][tgt] = {
            "database_column": col,
            "model_test": test_scores,
            "database": db_scores,
            "source_agreement": {**agree, "correlation": corr, "lag_correlation": lag_corr},
            "per_site_t1h": per_site,
            "overlap": {
                "start": str(joined["datetime"].min().date()) if len(joined) else None,
                "end": str(joined["datetime"].max().date()) if len(joined) else None,
                "rows": int(len(joined)),
            },
            "verdict": verdict,
        }

        print(f"  {tgt}: test R2={test_scores['all']['r2']}  "
              f"database R2={db_scores['all']['r2']}  "
              f"source corr={corr}  n={len(joined):,}")

    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(VALIDATION_JSON, "w", encoding="utf-8") as fh:
        json.dump(json_safe(result), fh, indent=2, allow_nan=False)
    print(f"  wrote {os.path.relpath(VALIDATION_JSON, ROOT)}")
    return result


# ── 4. markdown report ─────────────────────────────────────────────────
def write_report(preds: dict, validation: dict) -> None:
    lines = [
        "# Model accuracy — XGBoost (EXP_12 retrain on the project database)",
        "",
        f"Generated {validation['generated_at']} by `scripts/evaluate_accuracy.py`.",
        "",
        "Production models are retrained on the project database "
        "(`data/vaayu_final_v3.csv`, EXP_12). This report answers two "
        "separate questions:",
        "",
        "1. **Is the model accurate on the labels it was trained and scored "
        "against?** (`models/predictions/*.csv` → `truth`)",
        "2. **Does that accuracy carry over to the new database?** "
        "(predictions joined to `data/vaayu_extract_7sites.csv` on "
        "site + valid time, i.e. anchor + horizon)",
        "",
        "---",
        "",
        "## 1. Accuracy on the model's own test set",
        "",
        "7 sites, 24 hourly horizons.",
        "",
    ]

    for tgt in preds:
        df = preds[tgt]
        t1 = df[df.horizon == 1]
        t24 = df[df.horizon == 24]
        # score() masks non-finite pairs — HCHO truth has gaps by design
        s1 = score(t1["truth"], t1["prediction"])
        s24 = score(t24["truth"], t24["prediction"])
        sall = score(df["truth"], df["prediction"])
        unit = "µmol/m²" if tgt == "HCHO" else "ppb"
        # scored anchors only — e.g. site 3's stale anchor (the last date its
        # station reported) carries no truth and is masked out of every score
        scored = df[df["truth"].notna()]
        span = scored if len(scored) else df
        lines += [
            f"### {tgt}",
            "",
            f"Test period {span['datetime'].min().date()} → {span['datetime'].max().date()}, "
            f"values in {unit}.",
            "",
            "| Horizon | MAE | RMSE | R² | n |",
            "|---|---|---|---|---|",
            f"| t+1h  | {s1['mae']} | {s1['rmse']} | {s1['r2']} | {s1['n']:,} |",
            f"| t+24h | {s24['mae']} | {s24['rmse']} | {s24['r2']} | {s24['n']:,} |",
            f"| all   | {sall['mae']} | {sall['rmse']} | {sall['r2']} | {sall['n']:,} |",
            "",
        ]

    lines += [
        "---",
        "",
        "## 2. Accuracy against the new database",
        "",
        f"Database: `{validation['database']['extract']}` — "
        f"{validation['database']['rows']:,} rows, "
        f"{validation['database']['min_date']} → {validation['database']['max_date']}.",
        "",
    ]

    for tgt, block in validation["targets"].items():
        ovl = block["overlap"]
        lines += [
            f"### {tgt} → `database_column = {block['database_column']}`",
            "",
            f"Overlap joined on (site_id, anchor + horizon): **{ovl['rows']:,} rows**, "
            f"{ovl['start']} → {ovl['end']}.",
            "",
            "| Evaluation | MAE | RMSE | R² | Bias |",
            "|---|---|---|---|---|",
            f"| Model's own `truth` (t+1h) | {block['model_test']['t1h']['mae']} | "
            f"{block['model_test']['t1h']['rmse']} | {block['model_test']['t1h']['r2']} | "
            f"{block['model_test']['t1h']['bias']} |",
            f"| Model's own `truth` (all) | {block['model_test']['all']['mae']} | "
            f"{block['model_test']['all']['rmse']} | {block['model_test']['all']['r2']} | "
            f"{block['model_test']['all']['bias']} |",
            f"| **New database** (t+1h) | {block['database']['t1h']['mae']} | "
            f"{block['database']['t1h']['rmse']} | **{block['database']['t1h']['r2']}** | "
            f"{block['database']['t1h']['bias']} |",
            f"| **New database** (all) | {block['database']['all']['mae']} | "
            f"{block['database']['all']['rmse']} | **{block['database']['all']['r2']}** | "
            f"{block['database']['all']['bias']} |",
            "",
            f"- Correlation between the model's `truth` and the database at the "
            f"same site + valid timestamp: **{block['source_agreement']['correlation']}**",
            f"- Lag scan (rules out a time shift): "
            + ", ".join(f"{k} → {v}" for k, v in block["source_agreement"]["lag_correlation"].items()),
            "",
        ]

    lines += [
        "---",
        "",
        "## 3. Model lineage — why this used to disagree",
        "",
        "**Before EXP_12** the production NO₂/O₃ models came from EXP_09v2, "
        "trained on the retired `site_*_train_data.csv` schema whose inputs "
        "include NWP fields (`NO2_forecast`, `O3_forecast`, `T_forecast`, "
        "`q_forecast`, …) that **do not exist in `vaayu_final_v3.csv`**. The "
        "two sources also record different series: scanning all 110 stations "
        "of the master file for the best match to each old site caps out at a "
        "correlation of ≈0.49. So the legacy NO₂/O₃ predictions scored ≈0 (or "
        "negative) R² against the database — a label mismatch, not a code "
        "bug. Those artefacts are preserved in "
        "`models/_backup_pre_retrain_20261009/`.",
        "",
        "**EXP_12 retrains all three targets on the project database itself** "
        "(`notebooks/exp_12_retrain_db.py`): inputs are columns that exist in "
        "`vaayu_final_v3.csv` (ERA5-style weather + satellite columns + lags "
        "and rollings), split chronologically — train < 2025-01-01, "
        "validation Jan–Mar 2025 (real data, used for model selection), test "
        "Apr 2025 → Jun 2025 (real data, all scores above). With the same "
        "series on both sides, sections 1 and 2 now agree.",
        "",
        "### Caveats",
        "",
        "- Rows flagged `is_synthetic=1` (the 2023 block: noised targets "
        "**and** weather) stay in the training window; validation and test "
        "are real-only, so selection and reported scores are honest, but the "
        "model has seen some synthetic training data.",
        "- HCHO was already database-trained (EXP_11); EXP_12 re-splits it "
        "under the same real-only protocol, so its numbers are not directly "
        "comparable with EXP_11's half-synthetic test window.",
        "- `scripts/compare_retrain.py` prints the paired old-vs-new table "
        "against the backup artefacts.",
        "",
        "### What this means",
        "",
        "- **Model Details → Per-Horizon Performance** and **Accuracy vs the "
        "project database** now come from models trained on the database and "
        "agree closely.",
        "- The legacy EXP_09v2 numbers remain available in the backup "
        "(`models/_backup_pre_retrain_20261009/ACCURACY.md`).",
        "",
    ]

    with open(REPORT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  wrote {os.path.relpath(REPORT_MD, ROOT)}")


def main() -> None:
    print("[1/4] Loading predictions ...")
    preds = load_predictions()
    print("[2/4] Writing season/site breakdowns ...")
    write_breakdowns(preds)
    print("[3/4] Scoring against the project database ...")
    validation = database_validation(preds)
    print("[4/4] Writing report ...")
    write_report(preds, validation)
    print("Done.")


if __name__ == "__main__":
    main()
