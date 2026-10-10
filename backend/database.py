"""
VAAYU DARSHIYA — Data Layer
===========================

The project's database is a single flat file:

    data/vaayu_final_v3.csv    1,461,689 rows x 213 columns, 110 stations,
                               2016-01-29 -> 2026-06-01

The API serves 7 curated sites. This module projects those 7 stations out of
the 110-station master file into a compact extract and answers the queries the
endpoints need. The extract is written once (a materialised view, ~12 MB) so
that boot and per-request work never touch the 2.94 GB source.

Site -> station mapping (nearest sensor, haversine distance in parentheses):
    site 1  Ashok Vihar       -> 8917   Ashok Vihar, Delhi - DPCC        (3 m)
    site 2  Dwarka            -> 6931   Dwarka-Sector 8, Delhi - DPCC  (107 m)
    site 3  Jawaharlal Nehru  -> 5754   Jawaharlal Nehru Stadium       (218 m)
    site 4  Northwest Delhi   -> 10485  Narela, Delhi - DPCC             (3 m)
    site 5  East Delhi        -> 8239   Okhla Phase-2, Delhi - DPCC      (3 m)
    site 6  Northwest Delhi 2 -> 10831  Rohini, Delhi - DPCC          (2355 m)
    site 7  Northeast Delhi   -> 8475   Sonia Vihar, Delhi - DPCC        (3 m)

Column translation from the retired site_*_train_data.csv files:
    NO2_target -> no2          u_forecast -> wind_u_10m
    O3_target  -> o3           v_forecast -> wind_v_10m
    (day)      -> date         T_forecast -> temperature_2m_c
                               q_forecast -> relative_humidity
                               w_forecast -> rainfall_mm

HCHO (EXP_11): sat_hcho_column is the satellite formaldehyde vertical column,
stored in mol/m². Everything this module serves converts it to µmol/m² — the
unit the HCHO model trains, scores and predicts in.
"""

import csv
import os
import shutil
import time
import uuid

import numpy as np
import pandas as pd

ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT_DIR, "data")
SOURCE_CSV = os.path.join(DATA_DIR, "vaayu_final_v3.csv")
EXTRACT_CSV = os.path.join(DATA_DIR, "vaayu_extract_7sites.csv")

N_SITES = 7

# Frontend site id (1-7) -> station_id in vaayu_final_v3.csv
SITE_TO_STATION = {
    1: "8917",
    2: "6931",
    3: "5754",
    4: "10485",
    5: "8239",
    6: "10831",
    7: "8475",
}

# Columns pulled out of the master file, in master-file order.
USECOLS = [
    "station_id", "station_name", "latitude", "longitude",
    "datetime", "date", "hour", "month", "year",
    "no2", "o3", "pm25", "sat_hcho_column",
    "aqi_computed", "aqi_category",
    "temperature_2m_c", "relative_humidity",
    "wind_u_10m", "wind_v_10m", "total_precipitation", "rainfall_mm",
    "district_name", "season", "is_synthetic",
]

NUMERIC = [
    "latitude", "longitude", "hour", "month", "year",
    "no2", "o3", "pm25", "sat_hcho_column", "aqi_computed",
    "temperature_2m_c", "relative_humidity",
    "wind_u_10m", "wind_v_10m", "total_precipitation", "rainfall_mm",
    "is_synthetic",
]

# The database stores HCHO as a satellite vertical column in mol/m²; every
# value the API serves is converted to µmol/m² (×1e6, range ~3–90) so it is
# readable alongside the ppb pollutants — same unit the HCHO model was trained
# and scored in (notebooks/exp_11_hcho_xgboost.py).
HCHO_SCALE = 1e6

DEFAULT_WEATHER = {
    "temp": 30.0, "humidity": 50.0, "wind": 8.0, "windDir": 180, "visibility": 5.0,
}

_frame = None          # lazily loaded extract
_frame_error = None    # sticky failure so we never re-read 2.9 GB in a loop


def _sweep_stale_tmp() -> None:
    """Remove leftover *.tmp files from an interrupted build."""
    try:
        for name in os.listdir(DATA_DIR):
            if name.startswith(os.path.basename(EXTRACT_CSV)) and name.endswith(".tmp"):
                try:
                    os.remove(os.path.join(DATA_DIR, name))
                except OSError:
                    pass  # held by another live worker — leave it alone
    except OSError:
        pass


def _commit(tmp: str, dest: str) -> None:
    """Move the finished extract into place.

    os.replace can fail transiently on Windows when antivirus or an indexer
    holds the destination, and when another worker process is doing the same
    build. Retry briefly, then fall back to a copy.
    """
    for attempt in range(6):
        try:
            os.replace(tmp, dest)
            return
        except OSError:
            if os.path.exists(dest):
                # Another process won the race — use its extract.
                os.remove(tmp)
                return
            time.sleep(0.5 * (attempt + 1))
    # Last resort: copy over the top rather than failing the boot.
    shutil.copyfile(tmp, dest)
    os.remove(tmp)


def _extract_is_current() -> bool:
    """True when the extract exists, is newer than the source, and carries
    every column USECOLS asks for.

    The column check matters: when USECOLS grows (sat_hcho_column was added
    for the HCHO model) an existing extract still looks up-to-date by mtime
    but is missing the new column, so it has to be rebuilt.
    """
    if not os.path.exists(EXTRACT_CSV):
        return False
    if os.path.exists(SOURCE_CSV) and \
            os.path.getmtime(EXTRACT_CSV) < os.path.getmtime(SOURCE_CSV):
        return False
    try:
        with open(EXTRACT_CSV, newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh), []) or []
    except (OSError, StopIteration):
        return False
    return all(col in header for col in USECOLS)


def build_extract(force: bool = False, progress: bool = True) -> str:
    """Project the 7 served stations out of the master CSV into EXTRACT_CSV.

    If the master CSV is absent but a previously built extract exists, the
    extract is used as-is — that is the normal deployed state, since the
    2.94 GB source is not committed to git.
    """
    if not os.path.exists(SOURCE_CSV):
        if os.path.exists(EXTRACT_CSV):
            return EXTRACT_CSV
        raise FileNotFoundError(
            f"database source not found: {SOURCE_CSV}\n"
            "Place vaayu_final_v3.csv in data/ to enable the database layer."
        )

    if not force and _extract_is_current():
        return EXTRACT_CSV

    wanted = set(SITE_TO_STATION.values())
    # Unique temp name per process so two workers never clobber each other.
    _sweep_stale_tmp()
    tmp = f"{EXTRACT_CSV}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    rows = 0
    if progress:
        print("Projecting 7 stations out of vaayu_final_v3.csv ...", flush=True)

    try:
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            wrote_header = False
            reader = pd.read_csv(
                SOURCE_CSV,
                usecols=USECOLS,
                chunksize=200_000,
                dtype=str,
                low_memory=False,
            )
            for chunk in reader:
                sub = chunk[chunk["station_id"].isin(wanted)]
                if sub.empty:
                    continue
                if not wrote_header:
                    writer.writerow(USECOLS)
                    wrote_header = True
                writer.writerows(sub.reindex(columns=USECOLS).itertuples(index=False, name=None))
                rows += len(sub)
                if progress:
                    print(f"  {rows:,} rows ...", flush=True)
    except BaseException:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise

    if rows == 0:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise ValueError(
            "no rows matched the 7 served stations — check SITE_TO_STATION mapping"
        )

    _commit(tmp, EXTRACT_CSV)
    if progress:
        size = os.path.getsize(EXTRACT_CSV) / (1024 * 1024)
        print(f"  extract ready: {rows:,} rows, {size:.1f} MB", flush=True)
    return EXTRACT_CSV


def ensure_extract(progress: bool = True) -> str:
    return build_extract(force=False, progress=progress)


def frame() -> pd.DataFrame:
    """Load (and cache) the extract.

    A failure is sticky for the life of the process: retrying would mean
    re-reading the 2.94 GB source on every request.
    """
    global _frame, _frame_error
    if _frame is not None:
        return _frame
    if _frame_error is not None:
        raise _frame_error

    try:
        ensure_extract()
        df = pd.read_csv(EXTRACT_CSV, dtype={"station_id": str, "date": str})
        for col in NUMERIC:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    except Exception as exc:
        _frame_error = RuntimeError(f"database unavailable: {exc}")
        raise _frame_error

    _frame = df
    return _frame


def available() -> bool:
    """True when the database layer can serve queries."""
    try:
        frame()
        return True
    except Exception:
        return False


def site_frame(site: int) -> pd.DataFrame:
    """All rows for a frontend site id (1-7). Empty frame when unknown/unavailable."""
    try:
        df = frame()
    except Exception:
        return pd.DataFrame(columns=USECOLS)
    station = SITE_TO_STATION.get(site)
    if station is None:
        return pd.DataFrame(columns=USECOLS)
    out = df[df["station_id"] == station]
    return out if not out.empty else pd.DataFrame(columns=USECOLS)


def weekly_trend(site: int, days: int = 14) -> dict:
    """Last `days` available dates with mean observed NO2 / O3 / PM2.5 / HCHO.

    Dates are the data's own calendar dates. They used to be remapped onto the
    last `days` days from today, which quietly relabelled (say) June 2025
    readings as last week once the extract fell behind the wall clock — the
    chart now shows when the reading was actually taken.

    HCHO is a satellite column stored in mol/m²; it is served in µmol/m².
    """
    empty = {"days": [], "no2": [], "o3": [], "pm25": [], "hcho": []}
    sdf = site_frame(site)
    if sdf.empty:
        return empty

    all_dates = sdf["date"].dropna().drop_duplicates().sort_values().tolist()
    if not all_dates:
        return empty
    last_dates = all_dates[-days:]

    grouped = {d: g for d, g in sdf.groupby("date")}
    has_hcho = "sat_hcho_column" in sdf.columns

    out = {"days": [], "no2": [], "o3": [], "pm25": [], "hcho": []}
    for actual in last_dates:
        rows = grouped.get(actual)
        if rows is None:
            continue

        def avg(col: str) -> float:
            vals = rows[col].dropna()
            return round(float(vals.mean()), 1) if not vals.empty else 0.0

        out["days"].append(str(pd.Timestamp(actual).date()))
        out["no2"].append(avg("no2"))
        out["o3"].append(avg("o3"))
        out["pm25"].append(avg("pm25"))
        if has_hcho:
            vals = rows["sat_hcho_column"].dropna()
            out["hcho"].append(
                round(float(vals.mean()) * HCHO_SCALE, 1) if not vals.empty else None
            )
        else:
            out["hcho"].append(None)

    return out


def _pick_hour(rows: pd.DataFrame, hour: int) -> pd.Series:
    """Row at `hour`, else the last row of that date."""
    at_hour = rows[rows["hour"] == hour]
    if not at_hour.empty:
        return at_hour.iloc[0]
    return rows.iloc[-1]


def latest_timestamp(site: int):
    """Newest datetime available for a site in the database, or None."""
    sdf = site_frame(site)
    if sdf.empty:
        return None
    ts = sdf["datetime"].max()
    return ts if pd.notna(ts) else None


def current_weather(site: int, when) -> dict:
    """Weather block for a site at a given datetime.

    Falls back to DEFAULT_WEATHER when the site has no rows for that date,
    matching the previous per-site CSV behaviour.
    """
    weather = dict(DEFAULT_WEATHER)
    sdf = site_frame(site)
    if sdf.empty:
        return weather

    day = sdf[
        (sdf["year"] == when.year)
        & (sdf["month"] == when.month)
        & (sdf["date"] == pd.Timestamp(when).normalize())
    ]
    if day.empty:
        latest = sdf.iloc[-1]
    else:
        latest = _pick_hour(day, when.hour)

    u = float(latest.get("wind_u_10m", 0) or 0)
    v = float(latest.get("wind_v_10m", 0) or 0)
    temp = latest.get("temperature_2m_c")
    humidity = latest.get("relative_humidity")
    rainfall = latest.get("rainfall_mm")

    if pd.notna(temp):
        weather["temp"] = round(float(temp), 1)
    if pd.notna(humidity):
        weather["humidity"] = round(float(humidity), 1)
    if pd.notna(rainfall):
        # rain suppresses visibility: 10 km clear -> 1 km in heavy rain
        weather["visibility"] = round(max(1.0, min(10.0, 10 - float(rainfall) * 2)), 1)

    weather["wind"] = round(math_hypot(u, v) * 3.6, 1)
    weather["windDir"] = int(math_wind_dir(u, v))
    return weather


def current_observation(site: int, when) -> dict:
    """Observed pollutants for a site at a given datetime (from the database)."""
    obs = {"no2": None, "o3": None, "pm25": None, "aqi": None,
           "aqi_category": None, "district": None, "season": None, "hcho": None}
    sdf = site_frame(site)
    if sdf.empty:
        return obs

    day = sdf[
        (sdf["year"] == when.year)
        & (sdf["month"] == when.month)
        & (sdf["date"] == pd.Timestamp(when).normalize())
    ]
    row = _pick_hour(day, when.hour) if not day.empty else sdf.iloc[-1]

    for key, col in (("no2", "no2"), ("o3", "o3"), ("pm25", "pm25"), ("aqi", "aqi_computed")):
        val = row.get(col)
        obs[key] = round(float(val), 1) if pd.notna(val) else None

    # Satellite HCHO column: mol/m² in the database, served in µmol/m².
    val = row.get("sat_hcho_column")
    if val is not None and pd.notna(val):
        obs["hcho"] = round(float(val) * HCHO_SCALE, 1)

    for key, col in (("aqi_category", "aqi_category"),
                     ("district", "district_name"),
                     ("season", "season")):
        val = row.get(col)
        obs[key] = str(val) if pd.notna(val) else None

    return obs


def math_hypot(u: float, v: float) -> float:
    return float(np.sqrt(u * u + v * v))


def math_wind_dir(u: float, v: float) -> float:
    """Meteorological wind direction (direction the wind blows FROM), 0-360."""
    return (np.degrees(np.arctan2(-u, -v)) + 360) % 360


def describe() -> dict:
    """Diagnostics for /api — never raises."""
    try:
        df = frame()
        return {
            "source": SOURCE_CSV,
            "extract": EXTRACT_CSV,
            "rows": int(len(df)),
            "stations": int(df["station_id"].nunique()),
            "min_date": str(df["date"].min().date()),
            "max_date": str(df["date"].max().date()),
        }
    except Exception as exc:  # pragma: no cover - diagnostics only
        return {"error": str(exc)}
