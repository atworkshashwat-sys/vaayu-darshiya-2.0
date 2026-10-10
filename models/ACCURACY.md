# Model accuracy — XGBoost (EXP_12 retrain on the project database)

Generated 2026-10-10T00:00:39 by `scripts/evaluate_accuracy.py`.

Production models are retrained on the project database (`data/vaayu_final_v3.csv`, EXP_12). This report answers two separate questions:

1. **Is the model accurate on the labels it was trained and scored against?** (`models/predictions/*.csv` → `truth`)
2. **Does that accuracy carry over to the new database?** (predictions joined to `data/vaayu_extract_7sites.csv` on site + valid time, i.e. anchor + horizon)

---

## 1. Accuracy on the model's own test set

7 sites, 24 hourly horizons.

### NO2

Test period 2025-04-01 → 2025-06-18, values in ppb.

| Horizon | MAE | RMSE | R² | n |
|---|---|---|---|---|
| t+1h  | 11.16 | 16.14 | 0.6854 | 5,039 |
| t+24h | 11.99 | 17.15 | 0.6504 | 4,153 |
| all   | 14.06 | 20.02 | 0.5196 | 109,430 |

### O3

Test period 2025-04-01 → 2025-06-18, values in ppb.

| Horizon | MAE | RMSE | R² | n |
|---|---|---|---|---|
| t+1h  | 16.47 | 23.23 | 0.8105 | 4,994 |
| t+24h | 18.4 | 25.84 | 0.7741 | 4,107 |
| all   | 20.93 | 28.47 | 0.7191 | 108,330 |

### HCHO

Test period 2025-04-01 → 2025-06-18, values in µmol/m².

| Horizon | MAE | RMSE | R² | n |
|---|---|---|---|---|
| t+1h  | 4.51 | 5.88 | 0.5413 | 4,514 |
| t+24h | 4.72 | 6.14 | 0.4849 | 3,716 |
| all   | 5.81 | 7.52 | 0.2488 | 97,969 |

---

## 2. Accuracy against the new database

Database: `data/vaayu_extract_7sites.csv` — 99,753 rows, 2018-04-27 → 2025-07-13.

### NO2 → `database_column = no2`

Overlap joined on (site_id, anchor + horizon): **118,741 rows**, 2025-04-01 → 2025-06-18.

| Evaluation | MAE | RMSE | R² | Bias |
|---|---|---|---|---|
| Model's own `truth` (t+1h) | 11.16 | 16.14 | 0.6854 | 0.63 |
| Model's own `truth` (all) | 14.06 | 20.02 | 0.5196 | -1.5 |
| **New database** (t+1h) | 11.16 | 16.14 | **0.6854** | 0.63 |
| **New database** (all) | 14.11 | 19.95 | **0.5068** | -0.74 |

- Correlation between the model's `truth` and the database at the same site + valid timestamp: **1.0**
- Lag scan (rules out a time shift): 0h → 1.0, +1h → 0.5349, -1h → 0.5349

### O3 → `database_column = o3`

Overlap joined on (site_id, anchor + horizon): **117,669 rows**, 2025-04-01 → 2025-06-18.

| Evaluation | MAE | RMSE | R² | Bias |
|---|---|---|---|---|
| Model's own `truth` (t+1h) | 16.47 | 23.23 | 0.8105 | 1.49 |
| Model's own `truth` (all) | 20.93 | 28.47 | 0.7191 | -0.14 |
| **New database** (t+1h) | 16.47 | 23.23 | **0.8105** | 1.49 |
| **New database** (all) | 21.06 | 28.67 | **0.7084** | 0.11 |

- Correlation between the model's `truth` and the database at the same site + valid timestamp: **1.0**
- Lag scan (rules out a time shift): 0h → 1.0, +1h → 0.6025, -1h → 0.6025

### HCHO → `database_column = sat_hcho_column`

Overlap joined on (site_id, anchor + horizon): **106,202 rows**, 2025-04-01 → 2025-06-18.

| Evaluation | MAE | RMSE | R² | Bias |
|---|---|---|---|---|
| Model's own `truth` (t+1h) | 4.51 | 5.88 | 0.5413 | -0.49 |
| Model's own `truth` (all) | 5.81 | 7.52 | 0.2488 | -1.05 |
| **New database** (t+1h) | 4.51 | 5.88 | **0.5413** | -0.49 |
| **New database** (all) | 5.84 | 7.54 | **0.2315** | -1.17 |

- Correlation between the model's `truth` and the database at the same site + valid timestamp: **1.0**
- Lag scan (rules out a time shift): 0h → 1.0, +1h → 0.5073, -1h → 0.5073

---

## 3. Model lineage — why this used to disagree

**Before EXP_12** the production NO₂/O₃ models came from EXP_09v2, trained on the retired `site_*_train_data.csv` schema whose inputs include NWP fields (`NO2_forecast`, `O3_forecast`, `T_forecast`, `q_forecast`, …) that **do not exist in `vaayu_final_v3.csv`**. The two sources also record different series: scanning all 110 stations of the master file for the best match to each old site caps out at a correlation of ≈0.49. So the legacy NO₂/O₃ predictions scored ≈0 (or negative) R² against the database — a label mismatch, not a code bug. Those artefacts are preserved in `models/_backup_pre_retrain_20261009/`.

**EXP_12 retrains all three targets on the project database itself** (`notebooks/exp_12_retrain_db.py`): inputs are columns that exist in `vaayu_final_v3.csv` (ERA5-style weather + satellite columns + lags and rollings), split chronologically — train < 2025-01-01, validation Jan–Mar 2025 (real data, used for model selection), test Apr 2025 → Jun 2025 (real data, all scores above). With the same series on both sides, sections 1 and 2 now agree.

### Caveats

- Rows flagged `is_synthetic=1` (the 2023 block: noised targets **and** weather) stay in the training window; validation and test are real-only, so selection and reported scores are honest, but the model has seen some synthetic training data.
- HCHO was already database-trained (EXP_11); EXP_12 re-splits it under the same real-only protocol, so its numbers are not directly comparable with EXP_11's half-synthetic test window.
- `scripts/compare_retrain.py` prints the paired old-vs-new table against the backup artefacts.

### What this means

- **Model Details → Per-Horizon Performance** and **Accuracy vs the project database** now come from models trained on the database and agree closely.
- The legacy EXP_09v2 numbers remain available in the backup (`models/_backup_pre_retrain_20261009/ACCURACY.md`).
