"""
VAAYU DARSHIYA — HCHO hotspot build (offline)
==============================================

Reads the 2.94 GB master database (data/vaayu_final_v3.csv) in chunks — never
at runtime — and writes a small file the backend serves as-is:

    data/hcho_hotspots.json   (~25 KB)

What HCHO is here
-----------------
`sat_hcho_column` is the Sentinel-5P satellite formaldehyde VERTICAL COLUMN in
mol/m². Everything below converts to µmol/m² (×1e6, same HCHO_SCALE the API and
the models use — backend/database.py, notebooks/exp_11_hcho_xgboost.py). It is
NOT ground-level air and it has no WHO/CPCB ambient limit, so a hotspot can only
be defined RELATIVE to the region, never against a statutory band.

In this database the stored series is hourly with gaps (~85 % present), so the
"latest pass" value for a station is the MEAN of that station's hourly column
values on the latest date it has any.

Hotspot rule (simple and explainable)
-------------------------------------
    hotspot  <=>  value >= region P90 of the latest pass
              AND  value >= 31 µmol/m²  (the shared "Elevated" band floor,
                                         HCHO_BAND_EDGES[1] + 1 in data.ts)

Adjacent hotspot stations (<= 8 km) are grouped into one cluster; clusters are
ranked by peak value. If nothing qualifies, `hotspots` is empty and the map
prints "No clear hotspot on <date>" — nothing is forced.

Usage:
    python scripts/build_hcho_hotspots.py [--csv PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover - non-replaceable stdio
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(ROOT, "data", "vaayu_final_v3.csv")
DEFAULT_OUT = os.path.join(ROOT, "data", "hcho_hotspots.json")

# The database stores HCHO in mol/m²; the app serves µmol/m². Same factor as
# backend/database.py:HCHO_SCALE and the EXP_11/EXP_12 training pipelines.
HCHO_SCALE = 1e6

WINDOW_DAYS = 7          # persistence / recency window
CLUSTER_KM = 8.0         # stations closer than this join one cluster
HULL_BUFFER_KM = 2.0     # outline drawn this far outside the member points
ELEVATED_FLOOR = 31.0    # µmol/m² — "Elevated" band floor from data.ts
REGION = "Delhi-NCR"

# --- tiny geo helpers (no new dependencies) --------------------------------

_EARTH_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * _EARTH_KM * math.asin(min(1.0, math.sqrt(a)))


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile (numpy's default, type 7).

    q in 0..100. Raises ValueError on an empty list — callers decide what an
    empty region means (there is no median of nothing).
    """
    if not values:
        raise ValueError("percentile of an empty sequence")
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * (q / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


# --- hotspot rule -----------------------------------------------------------

def hotspot_threshold(p90: float, floor: float = ELEVATED_FLOOR) -> float:
    """Region P90, but never below the shared "Elevated" band floor."""
    return max(float(p90), float(floor))


def is_hotspot(value: float, p90: float, floor: float = ELEVATED_FLOOR) -> bool:
    """A station is a hotspot when it clears BOTH the relative P90 and the
    absolute Elevated floor. HCHO has no statutory limit, so the floor only
    keeps noise on a very flat pass from being called a hotspot."""
    return float(value) >= hotspot_threshold(p90, floor)


# --- clustering -------------------------------------------------------------

def _union_find(groups: list[int], i: int) -> int:
    while groups[i] != i:
        groups[i] = groups[groups[i]]
        i = groups[i]
    return i


def cluster_points(points: list[dict], max_km: float = CLUSTER_KM) -> list[list[int]]:
    """Group points closer than `max_km` into one cluster (union-find).

    `points` = [{"lat": .., "lon": ..}, ...]; returns index groups, each sorted
    by that point's own value descending when a "value" key is present.
    """
    n = len(points)
    groups = list(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if haversine_km(points[i]["lat"], points[i]["lon"],
                             points[j]["lat"], points[j]["lon"]) <= max_km:
                a, b = _union_find(groups, i), _union_find(groups, j)
                if a != b:
                    groups[b] = a

    by_root: dict[int, list[int]] = {}
    for idx in range(n):
        by_root.setdefault(_union_find(groups, idx), []).append(idx)

    out = []
    for members in by_root.values():
        if "value" in points[0]:
            members.sort(key=lambda k: -points[k]["value"])
        else:
            members.sort()
        out.append(members)
    # Rank-independent ordering here; the caller sorts clusters by peak value.
    return out


def convex_hull(pairs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Andrew's monotone chain. `pairs` are (lon, lat); returns CCW hull."""
    pts = sorted(set(pairs))
    if len(pts) <= 1:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def buffered_outline(lat: float, lon: float, radius_km: float = HULL_BUFFER_KM,
                     segments: int = 24) -> list[list[float]]:
    """Closed ring of `segments` points around one station ([lon, lat])."""
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * math.cos(math.radians(lat)))
    ring = []
    for k in range(segments):
        ang = 2 * math.pi * k / segments
        ring.append([round(lon + dlon * math.cos(ang), 5),
                     round(lat + dlat * math.sin(ang), 5)])
    ring.append(ring[0])
    return ring


def cluster_outline(lats: list[float], lons: list[float],
                    radius_km: float = HULL_BUFFER_KM) -> list[list[float]]:
    """Outline for a cluster: convex hull of the member points, each buffered
    by `radius_km` (sampling around every member and hulling the result). One
    member becomes a small circle; two become a stadium; three or more a
    smoothed hull. Always a closed [lon, lat] ring."""
    samples: list[tuple[float, float]] = []
    seg = 12
    for lat, lon in zip(lats, lons):
        dlat = radius_km / 111.32
        dlon = radius_km / (111.32 * math.cos(math.radians(lat)))
        for k in range(seg):
            ang = 2 * math.pi * k / seg
            samples.append((lon + dlon * math.cos(ang), lat + dlat * math.sin(ang)))
    hull = convex_hull(samples)
    if len(hull) < 3:  # degenerate (all points within one pixel) — fall back
        return buffered_outline(lats[0], lons[0], radius_km)
    ring = [[round(x, 5), round(y, 5)] for x, y in hull]
    ring.append(ring[0])
    return ring


# --- daily aggregation ------------------------------------------------------

def daily_means(rows) -> dict[str, dict[str, float]]:
    """station_id -> {date -> mean µmol/m²} over the scanned window rows.

    `rows` is an iterable of (station_id, date, value_umol). The stored series
    is hourly with gaps, so a station's value "on a day" is that day's mean.
    """
    acc: dict[tuple[str, str], list[float]] = {}
    for sid, d, val in rows:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        acc.setdefault((sid, d), []).append(float(val))
    out: dict[str, dict[str, float]] = {}
    for (sid, d), vals in acc.items():
        out.setdefault(sid, {})[d] = round(sum(vals) / len(vals), 1)
    return out


def build_payload(by_station: dict[str, dict[str, float]],
                  meta: dict[str, dict],
                  as_of: str) -> dict:
    """Pure step: daily means -> the exact JSON the endpoint serves.

    by_station: station_id -> {date -> daily mean µmol/m²}
    meta:       station_id -> {"name", "lat", "lon"}
    as_of:      "YYYY-MM-DD" of the latest pass with any HCHO

    Stations with no value inside the WINDOW_DAYS ending `as_of` are excluded
    from `stations` and listed in `no_recent_data` — they take no part in the
    region stats, the threshold or the interpolation.

    A station with window data but nothing ON the latest pass (a partial week)
    stays in `stations` with its own `date`, flagged `latest_pass: False`. The
    region stats, the P90 threshold and the hotspot rule are computed on the
    latest-pass cross-section only, so one date is never mixed with another.
    """
    as_of_d = date.fromisoformat(as_of)
    window = [(as_of_d - timedelta(days=i)).isoformat() for i in range(WINDOW_DAYS)]
    window_set = set(window)

    stations = []
    excluded = []
    for sid, days in sorted(by_station.items()):
        in_window = {d: v for d, v in days.items() if d in window_set}
        info = meta.get(sid, {})
        entry_base = {"id": sid,
                      "name": info.get("name", f"Station {sid}"),
                      "lat": round(float(info.get("lat", 0.0)), 5),
                      "lon": round(float(info.get("lon", 0.0)), 5)}
        if not in_window:
            excluded.append({**entry_base, "reason": "no HCHO in the last 7 days"})
            continue
        latest = max(in_window)
        stations.append({
            **entry_base,
            "value": in_window[latest],
            "date": latest,
            "mean_7d": round(sum(in_window.values()) / len(in_window), 1),
            "days": len(in_window),
            "latest_pass": latest == as_of,
        })

    pass_stations = [s for s in stations if s["latest_pass"]]
    stats = {"as_of": as_of, "n_stations": len(pass_stations),
             "n_stations_window": len(stations),
             "median": None, "p75": None, "p90": None,
             "threshold": None, "elevated_floor": ELEVATED_FLOOR}
    hotspots: list[dict] = []
    if pass_stations:
        values = [s["value"] for s in pass_stations]
        median = percentile(values, 50)
        p75 = percentile(values, 75)
        p90 = percentile(values, 90)
        threshold = hotspot_threshold(p90)
        stats.update({"median": round(median, 1), "p75": round(p75, 1),
                      "p90": round(p90, 1), "threshold": round(threshold, 1)})

        # Per-day thresholds inside the window, for the persistence count:
        # each day uses its own cross-section of the latest-pass stations.
        pass_ids = {s["id"] for s in pass_stations}
        day_thresholds: dict[str, float] = {}
        for d in window:
            dv = [by_station[sid][d] for sid in pass_ids if d in by_station[sid]]
            if dv:
                day_thresholds[d] = hotspot_threshold(percentile(dv, 90))

        by_id = {s["id"]: s for s in pass_stations}
        flagged = [s for s in pass_stations if s["value"] >= threshold]
        flagged_ids = [s["id"] for s in flagged]
        groups = cluster_points(flagged, CLUSTER_KM)
        built = []
        for members in groups:
            pts = [by_id[flagged_ids[m]] for m in members]
            peak = max(p["value"] for p in pts)
            persistence = sum(
                1 for d in window
                if any(by_station[p["id"]].get(d, -1) >= day_thresholds.get(d, math.inf)
                       for p in pts)
            )
            built.append({
                "peak": peak,
                "ratio_to_median": round(peak / median, 1) if median > 0 else None,
                "persistence": persistence,
                "days": WINDOW_DAYS,
                "stations": [{"id": p["id"], "name": p["name"], "value": p["value"]}
                             for p in pts],
                "polygon": cluster_outline([p["lat"] for p in pts],
                                           [p["lon"] for p in pts]),
                "center": [round(sum(p["lon"] for p in pts) / len(pts), 5),
                           round(sum(p["lat"] for p in pts) / len(pts), 5)],
            })
        built.sort(key=lambda c: -c["peak"])
        for rank, cluster in enumerate(built, start=1):
            hotspots.append({"rank": rank, **cluster})

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "as_of": as_of,
        "region": REGION,
        "unit": "µmol/m²",
        "window_days": WINDOW_DAYS,
        "cluster_km": CLUSTER_KM,
        "hotspots": hotspots,
        "stations": stations,
        "no_recent_data": excluded,
        "region_stats": stats,
        "rule": (f"hotspot = value ≥ region P90 ({stats['p90']}) and ≥ the "
                 f"Elevated floor ({ELEVATED_FLOOR:.0f} µmol/m²); adjacent "
                 f"hotspots within {CLUSTER_KM:.0f} km form one cluster"),
        "notes": {
            "measurement": "Sentinel-5P satellite HCHO vertical column, not ground-level air",
            "scale": "database stores mol/m²; values here are ×1e6 = µmol/m²",
            "pass": "one station-day = mean of that day's hourly column values",
            "resolution": "TROPOMI pixel ≈ 5 × 3.5 km — stations inside one pixel are not independent",
            "relative": "HCHO has no WHO/CPCB limit; hotspots are relative to the region on the pass date",
        },
    }


# --- CSV scan ---------------------------------------------------------------

def scan(csv_path: str, chunksize: int = 200_000) -> tuple[dict, dict, str]:
    """One chunked pass over the master CSV.

    Returns (by_station, meta, as_of): daily means in µmol/m², station
    metadata for every station in the file, and the latest date carrying any
    HCHO. The 2.94 GB source is only ever touched by this offline script —
    the runtime endpoint serves the JSON.
    """
    import pandas as pd

    usecols = ["station_id", "station_name", "latitude", "longitude",
               "date", "sat_hcho_column"]
    sums: dict[tuple[str, str], float] = {}
    counts: dict[tuple[str, str], int] = {}
    meta: dict[str, dict] = {}
    as_of = ""

    reader = pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize,
                         dtype={"station_id": str, "date": str})
    for chunk in reader:
        for sid, name, lat, lon in chunk[
                ["station_id", "station_name", "latitude", "longitude"]
        ].drop_duplicates().itertuples(index=False):
            if sid not in meta:
                meta[sid] = {"name": str(name), "lat": float(lat), "lon": float(lon)}
        h = chunk[chunk["sat_hcho_column"].notna()]
        if h.empty:
            continue
        vals = pd.to_numeric(h["sat_hcho_column"], errors="coerce") * HCHO_SCALE
        ok = vals.notna()
        h, vals = h[ok], vals[ok]
        for sid, d, val in zip(h["station_id"], h["date"], vals):
            key = (sid, d)
            sums[key] = sums.get(key, 0.0) + float(val)
            counts[key] = counts.get(key, 0) + 1
            if d > as_of:
                as_of = d

    by_station: dict[str, dict[str, float]] = {}
    for (sid, d), total in sums.items():
        by_station.setdefault(sid, {})[d] = round(total / counts[(sid, d)], 1)
    return by_station, meta, as_of


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build data/hcho_hotspots.json")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="master database CSV")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output JSON path")
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args(argv)

    if not os.path.exists(args.csv):
        print(f"  master CSV not found: {args.csv}", file=sys.stderr)
        return 1

    t0 = time.time()
    print(f"Scanning {args.csv} in chunks ...", flush=True)
    by_station, meta, as_of = scan(args.csv, args.chunksize)
    print(f"  {len(meta)} stations, latest HCHO date {as_of} "
          f"({time.time() - t0:.0f}s)", flush=True)

    payload = build_payload(by_station, meta, as_of)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)

    stats = payload["region_stats"]
    print(f"  wrote {args.out}")
    print(f"  stations with recent HCHO: {len(payload['stations'])} "
          f"(excluded: {len(payload['no_recent_data'])})")
    print(f"  region {payload['region']}: median={stats['median']} "
          f"P75={stats['p75']} P90={stats['p90']} "
          f"threshold={stats['threshold']} µmol/m²")
    print(f"  hotspots: {len(payload['hotspots'])}"
          if payload["hotspots"] else
          f"  hotspots: none — 'No clear hotspot on {as_of}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
