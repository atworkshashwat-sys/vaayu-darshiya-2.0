"""
VAAYU DARSHIYA — OpenAQ v3 gap-filler
Pulls hourly NO2 + O3 measurements for Delhi-NCR stations in the missing
2023-01-01 → 2024-05-21 window and merges them into vaayu_final_ml_ready.csv.

API:   GET https://api.openaq.org/v3/sensors/{id}/hours
Auth:  X-API-Key header
Page:  1000 rows per page, walk with `page` until meta.page > meta.pages

Run:
    set OPENAQ_API_KEY=xxxxxxxx   (Windows)
    python download_openaq_gap.py
"""

import os, sys, time
import requests
import pandas as pd
import numpy as np
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
API_KEY  = os.environ.get("OPENAQ_API_KEY")
if not API_KEY:
    sys.exit("ERROR: set OPENAQ_API_KEY environment variable first")

BASE      = "https://api.openaq.org/v3"
DATA_FILE = "vaayu_final_ml_ready.csv"
OUT_FILE  = "vaayu_openaq_gap_2023_2024.csv"
MERGED    = "vaayu_final_ml_ready_v2.csv"

DATE_FROM = "2023-01-01T00:00:00Z"
DATE_TO   = "2024-05-22T00:00:00Z"     # exclusive end
PARAMS    = {"no2": 7, "o3": 10}       # OpenAQ v3 parameter IDs (verified 2024)
PARAM_MAP = {7: "no2", 10: "o3"}      # invert for the response side

PAGE_LIMIT = 1000
SLEEP_S    = 0.4                       # stay under OpenAQ rate limits
MAX_RETRIES = 5
TIMEOUT_S  = 60

# ── HELPERS ───────────────────────────────────────────────────────────────────
def http_get(url, params):
    headers = {"X-API-Key": API_KEY}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=TIMEOUT_S)
            # Retry transient errors; fail fast on auth/not-found so we don't burn ~12s on a bad key
            if r.status_code in (401, 403, 404, 429) or r.status_code >= 500:
                if r.status_code in (401, 403, 404):
                    r.raise_for_status()  # raise immediately, no retry
                wait = SLEEP_S * (2 ** attempt)
                print(f"    HTTP {r.status_code} — retrying in {wait:.1f}s "
                      f"(attempt {attempt}/{MAX_RETRIES})")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == MAX_RETRIES:
                raise
            wait = SLEEP_S * (2 ** attempt)
            print(f"    {e} — retrying in {wait:.1f}s")
            time.sleep(wait)
    return None


def find_sensors_for_param(parameter_id):
    """
    Search OpenAQ v3 for active sensors measuring `parameter_id` in Delhi-NCR.
    Bounding box ≈ 28.2–28.9 N, 76.8–77.6 E.
    Returns list of dicts: {sensor_id, sensor_name, lat, lon, owner, parameter_id}.

    v3 note: `/sensors` has no spatial filter — search by `bbox` on `/locations`,
    then harvest each location's `sensors` array.
    """
    url = f"{BASE}/locations"
    bbox = (76.80, 28.20, 77.60, 28.90)  # W, S, E, N — Delhi-NCR (minLon,minLat,maxLon,maxLat)
    params = {
        "bbox":          f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "parameters_id": parameter_id,
        "is_active":     "true",
        "limit":         1000,
    }
    rows = []
    page = 1
    while True:
        params["page"] = page
        data = http_get(url, params)
        if not data:
            break
        for loc in data.get("results", []):
            coords = loc.get("coordinates") or {}
            owner  = (loc.get("owner") or {}).get("name")
            for s in loc.get("sensors", []) or []:
                # v3 sensor object: {id, name, parameter: {id, name, units}}
                if (s.get("parameter") or {}).get("id") != parameter_id:
                    continue
                rows.append({
                    "sensor_id":    s.get("id"),
                    "sensor_name":  s.get("name"),
                    "lat":          coords.get("latitude"),
                    "lon":          coords.get("longitude"),
                    "owner":        owner,
                    "parameter_id": parameter_id,
                })
        meta = data.get("meta", {})
        if page >= meta.get("pages", 1):
            break
        page += 1
        time.sleep(SLEEP_S)
    return rows


def fetch_sensor_hours(sensor_id, parameter_id):
    """
    Paginate /v3/sensors/{id}/hours for the gap window. Returns DataFrame with
    columns: sensor_id, datetime (UTC), value, parameter_id.
    """
    url  = f"{BASE}/sensors/{sensor_id}/hours"
    q    = {
        "date_from": DATE_FROM,
        "date_to":   DATE_TO,
        "limit":     PAGE_LIMIT,
        "page":      1,
    }
    out = []
    while True:
        data = http_get(url, q)
        if not data:
            break
        for row in data.get("results", []):
            # v3 hours response: row['period']['datetimeFrom']['utc'] is the hour start
            period = row.get("period", {})
            dt_from = (period.get("datetimeFrom") or {}).get("utc")
            if not dt_from:
                continue
            value = row.get("value")
            if value is None:
                continue
            out.append({
                "sensor_id":    sensor_id,
                "datetime":     pd.to_datetime(dt_from, utc=True),
                "value":        float(value),
                "parameter_id": parameter_id,
            })
        meta = data.get("meta", {})
        if q["page"] >= meta.get("pages", 1):
            break
        q["page"] += 1
        time.sleep(SLEEP_S)
    return out


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Gap window: {DATE_FROM}  →  {DATE_TO}")
    print()

    # 1) load existing stations so we can match by lat/lon
    print("[1] Loading existing stations …")
    base = pd.read_csv(DATA_FILE, low_memory=False,
                       usecols=["station_id", "station_name", "latitude", "longitude"])
    base_stations = (base.drop_duplicates("station_id")
                        [["station_id", "station_name", "latitude", "longitude"]]
                        .reset_index(drop=True))
    print(f"    {len(base_stations)} unique stations in master CSV")

    # 2) find active NO2 and O3 sensors in Delhi-NCR
    print("\n[2] Querying OpenAQ for active sensors …")
    sensors_no2 = find_sensors_for_param(PARAMS["no2"])
    print(f"    NO2 sensors found: {len(sensors_no2)}")
    sensors_o3  = find_sensors_for_param(PARAMS["o3"])
    print(f"    O3  sensors found: {len(sensors_o3)}")
    all_sensors = sensors_no2 + sensors_o3

    # 3) match each OpenAQ sensor to the nearest master station (within ~3 km)
    print("\n[3] Matching OpenAQ sensors to master stations …")
    from math import radians, sin, cos, asin, sqrt
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
        return 2*R*asin(sqrt(a))

    matched = []
    for s in all_sensors:
        if s["lat"] is None or s["lon"] is None:
            continue
        d = base_stations.apply(
            lambda r: haversine(s["lat"], s["lon"], r["latitude"], r["longitude"]),
            axis=1
        )
        idx = d.idxmin()
        if d[idx] <= 3.0:  # 3 km cutoff
            matched.append({
                "sensor_id":    s["sensor_id"],
                "sensor_name":  s["sensor_name"],
                "parameter_id": s["parameter_id"],
                "lat":          s["lat"],
                "lon":          s["lon"],
                "station_id":   int(base_stations.loc[idx, "station_id"]),
                "station_name": base_stations.loc[idx, "station_name"],
                "dist_km":      float(d[idx]),
            })
    matched_df = pd.DataFrame(matched)
    if len(matched_df):
        matched_df["parameter"] = matched_df["parameter_id"].map(PARAM_MAP)
    print(f"    {len(matched_df)} sensors matched within 3 km of a master station")
    if len(matched_df):
        print(matched_df.groupby("parameter_id")["station_id"].nunique()
                       .rename({7: "no2", 10: "o3"}))

    # 4) pull hourly data for each matched sensor
    print(f"\n[4] Downloading hourly values for {len(matched_df)} sensors …")
    all_rows = []
    for i, row in matched_df.iterrows():
        print(f"    [{i+1}/{len(matched_df)}] sensor={row.sensor_id} "
              f"param={row.parameter_id} station={row.station_id} "
              f"({row.station_name}) dist={row.dist_km:.2f} km …", end=" ", flush=True)
        rows = fetch_sensor_hours(int(row.sensor_id), int(row.parameter_id))
        print(f"{len(rows):,} rows")
        all_rows.extend(rows)

    if not all_rows:
        print("No rows downloaded. Exiting.")
        return

    raw = pd.DataFrame(all_rows)
    raw["parameter"] = raw["parameter_id"].map(PARAM_MAP)
    print(f"\n[5] Total raw rows: {len(raw):,}")
    print(raw["parameter"].value_counts().to_string())

    # 5) pivot to wide: one row per (station, datetime), with no2 and o3 columns
    print("\n[6] Pivoting to station-hour grid …")
    raw = raw.merge(matched_df[["sensor_id", "station_id", "parameter"]],
                    on=["sensor_id", "parameter"], how="left")
    wide = (raw.pivot_table(index=["station_id", "datetime"],
                           columns="parameter",
                           values="value",
                           aggfunc="mean")      # if multiple sensors per station+param
                  .reset_index())
    # round + clip to plausible µg/m³ ranges
    if "no2" in wide.columns:
        wide["no2"] = wide["no2"].clip(0, 500).round(2)
    if "o3" in wide.columns:
        wide["o3"]  = wide["o3"].clip(0, 600).round(2)
    print(f"    {len(wide):,} station-hour rows in gap window")
    print(f"    Stations covered: {wide['station_id'].nunique()}")
    print(f"    Datetime range : {wide['datetime'].min()}  →  {wide['datetime'].max()}")

    # 6) write gap-only file for inspection
    wide["datetime"] = wide["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    wide.to_csv(OUT_FILE, index=False)
    print(f"\n[7] Wrote gap-only file: {OUT_FILE}")

    # 7) merge back into the master CSV
    print(f"\n[8] Merging into {DATA_FILE} → {MERGED} …")
    master = pd.read_csv(DATA_FILE, low_memory=False)
    master["datetime"] = pd.to_datetime(master["datetime"], utc=True, errors="coerce")

    # update no2 / o3 from gap rows where the gap value is non-null
    # (don't overwrite existing good values)
    for col in ["no2", "o3"]:
        if col not in wide.columns:
            continue
        update = wide[["station_id", "datetime", col]].dropna(subset=[col])
        update["datetime"] = pd.to_datetime(update["datetime"], utc=True)
        # use merge: set master[col] = update[col] on the matching key
        master = master.merge(
            update.rename(columns={col: f"{col}_gap"}),
            on=["station_id", "datetime"],
            how="left",
        )
        mask = master[f"{col}_gap"].notna()
        master.loc[mask, col] = master.loc[mask, f"{col}_gap"]
        master = master.drop(columns=[f"{col}_gap"])
        print(f"    Filled {mask.sum():,} {col} values from OpenAQ")

    master["datetime"] = master["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    master.to_csv(MERGED, index=False)
    print(f"\nDone. New master file: {MERGED}")
    print(f"Gap-only file kept at: {OUT_FILE}  (for QA before swapping)")


if __name__ == "__main__":
    main()
