# data/

| File | Size | Tracked? |
|---|---|---|
| `vaayu_extract_7sites.csv` | 23 MB | **yes** |
| `stations.txt` | < 1 KB | **yes** |
| `hcho_hotspots.json` | ~18 KB | **yes** |
| `vaayu_final_v3.csv` | 2.94 GB | **no — see below** |

## Why `vaayu_final_v3.csv` is not in this repository

GitHub rejects any single file over **100 MB**, and this one is 2.94 GB
(1,461,689 rows × 213 columns, 110 stations across Delhi NCR,
2016-01-29 → 2026-06-01). It therefore cannot be committed here — this is a
platform limit, not a choice about the data.

The file is listed in `.gitignore` for that reason. Keep it in `data/` locally.

## What you get without it

`data/vaayu_extract_7sites.csv` — the 7 curated sites the dashboard actually
serves — **is** committed. `backend/database.py` loads it directly on every run
after the first, so a fresh clone starts and serves data with no extra step.

## Regenerating the extract

Only needed if you obtain the master file and want to rebuild the projection:

```bash
# 1. Place vaayu_final_v3.csv in this directory
# 2. Delete the existing extract so it is rebuilt
rm data/vaayu_extract_7sites.csv
# 3. Start the backend — the first run re-projects the 7 sites
python backend/api.py
```

`backend/database.py` maps each of the 7 frontend site ids to its nearest
sensor in the master file and writes `data/vaayu_extract_7sites.csv` (~21 MB).
Later runs load that extract instead of re-reading 2.94 GB.

## `hcho_hotspots.json`

The offline-built HCHO satellite hotspot layer, served at runtime by
`GET /api/hcho-hotspots`. Built by `scripts/build_hcho_hotspots.py` (chunked,
so the 2.94 GB master is never loaded whole): per-station latest daily value,
7-day mean and days-available, plus region median/P75/P90 and the ranked
hotspot clusters. The map's HCHO toggle reads it to draw the smooth IDW field
and the ranked hotspot outlines. Regenerate it whenever the master CSV updates:

```bash
python scripts/build_hcho_hotspots.py --csv data/vaayu_final_v3.csv
```

## Obtaining the master file

The dataset is not redistributable through this repository. Get it from whoever
maintains the project's data pipeline, then drop it in this folder unchanged —
the filename must be exactly `vaayu_final_v3.csv`.

> Do not commit it, even in chunks or via a force-add. Use the extract, or an
> external store such as Git LFS on a paid plan, Google Drive, or S3.
