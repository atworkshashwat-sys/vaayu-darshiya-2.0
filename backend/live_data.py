"""
Live data jobs (spec §2) — the only consumers of the TinyFish wrapper for
readings, alerts, weather and the case-study figures.

    job A  refresh_readings()    7 stations, NO2 / O3 / PM2.5 / AQI   every 60 min
    job B  refresh_alerts()      GRAP / CAQM / IMD / Delhi Govt       every 30-60 min
    job C  refresh_weather()     temp, humidity, wind, visibility     every 2-3 h
    job D  refresh_case_study()  annual PM2.5, winter AQI, news strip once

Every job follows the same contract:
  1. ask TinyFish (via tinyfish_service — never directly)
  2. extract ONLY values the source actually stated
  3. validate with live_validate (range / unit / timestamp) — rejects are
     counted, never guessed around
  4. persist through live_store
  5. return a summary dict; on failure the previous values stay put and are
     served with their age

Station naming
--------------
The frontend calls the sites "Northwest Delhi", "East Delhi" … because those
were the names in the archived dataset. The live CPCB/DPCC bulletins use the
real station names, so each site carries those aliases for matching. The API
keeps serving the frontend names (spec §6: keep existing response shapes) and
adds the live source name beside them.

Site 3 (Jawaharlal Nehru Stadium) stopped reporting on 2023-12-31. There is no
live source for it, so it is reported as "offline" rather than being silently
mapped onto a different location.
"""

import re
import time
from datetime import datetime, timedelta, timezone

import live_store
import live_validate
import tinyfish_context
import tinyfish_service
from live_validate import Rejected, freshness, resolve_conflict, validate

IST = timezone(timedelta(hours=5, minutes=30))

# ── Station registry ────────────────────────────────────────────────────────
# `name` is what the API keeps serving (existing response shape). `aliases`
# are the names CPCB/DPCC use in their real-time bulletins, tried in order.
STATIONS = [
    {"site_id": 1, "name": "Ashok Vihar",      "lat": 28.69536, "lon": 77.18169,
     "aliases": ["Ashok Vihar"]},
    {"site_id": 2, "name": "Dwarka",           "lat": 28.57180, "lon": 77.07125,
     "aliases": ["Dwarka", "Dwarka Sector 8", "Dwarka-Sector 8"]},
    # Stopped reporting 2023-12-31 — no live source exists, so it is served as
    # "offline" and never coloured on the map.
    {"site_id": 3, "name": "Jawaharlal Nehru", "lat": 28.58278, "lon": 77.23441,
     "aliases": ["Jawaharlal Nehru Stadium"], "retired": "2023-12-31"},
    {"site_id": 4, "name": "Northwest Delhi",  "lat": 28.82286, "lon": 77.10197,
     "aliases": ["Narela"]},
    {"site_id": 5, "name": "East Delhi",       "lat": 28.53077, "lon": 77.27123,
     "aliases": ["Okhla", "Okhla Phase 2", "Okhla Phase-2"]},
    {"site_id": 6, "name": "Northwest Delhi 2", "lat": 28.72954, "lon": 77.09601,
     "aliases": ["Rohini"]},
    {"site_id": 7, "name": "Northeast Delhi",  "lat": 28.71052, "lon": 77.24951,
     "aliases": ["Sonia Vihar"]},
]

STATION_BY_ID = {s["site_id"]: s for s in STATIONS}

# Official / reputable hosts for live readings (spec §2, ground rule 4).
# Verified during setup: dpccairdata.com is DPCC's own live portal (the only
# source carrying station-wise values), aqicn.org republishes DPCC + CPCB
# data, and airquality.cpcb.gov.in is CPCB's national index portal.
LIVE_SOURCE_DOMAINS = (
    "cpcb.nic.in,cpcb.gov.in,airquality.cpcb.in,airquality.cpcb.gov.in,"
    "dpcc.delhigovt.nic.in,dpccairdata.com,"
    "caqm.nic.in,imd.gov.in,safar.tropmet.res.in,delhi.gov.in,openaq.org,"
    # WAQI is on the spec's §2 allowlist and publishes per-station live AQI.
    "aqicn.org,waqi.info"
)

POLLUTANTS = ("NO2", "O3", "PM25", "AQI")

# Column labels a bulletin may use for each pollutant.
_POLLUTANT_TOKENS = {
    "NO2":   ("no2", "no₂", "nitrogen dioxide", "nox as no2"),
    "O3":    ("o3", "o₃", "ozone"),
    "PM25":  ("pm2.5", "pm25", "pm 2.5", "fine particulate"),
    "AQI":   ("aqi", "air quality index", "overall aqi"),
}

MAX_READING_AGE_MINUTES = live_validate.STALE_AFTER_MINUTES


def now_ist():
    return datetime.now(IST)


def _iso(dt=None):
    return live_store.iso(dt or now_ist())


# ── Extraction: strict JSON out of fetched page text (spec §2) ─────────────
def _normalise(text):
    """Lowercase and strip markup while PRESERVING line breaks.

    Deliberately not `tinyfish_context.sanitize`: that helper collapses all
    whitespace into single spaces to build a one-line summary, which would
    fuse every row of a station bulletin into a single line and destroy the
    structure both extraction strategies depend on.
    """
    text = "" if text is None else str(text)
    text = re.sub(r"<[^>]*>", " ", text)                    # HTML tags
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)    # markdown links
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f\u00a0"
                  r"\u200b-\u200f\u2028\u2029\ufeff]", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(re.sub(r"[ \t]+", " ", ln).strip()
                     for ln in text.lower().split("\n"))


def _station_pattern(station):
    """Match a station name inside a bulletin line, tolerating punctuation."""
    parts = []
    for alias in station["aliases"] + [station["name"]]:
        escaped = re.escape(alias.lower())
        escaped = escaped.replace(r"\ ", r"[\s\-]*").replace(r"\-", r"[\s\-]*")
        parts.append(escaped)
    return re.compile(r"(?<![a-z])(" + "|".join(sorted(set(parts), key=len, reverse=True)) + r")(?![a-z])")


def _pollutant_pattern(pollutant):
    tokens = _POLLUTANT_TOKENS[pollutant]
    return re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(t) for t in tokens) + r")(?![a-z0-9])")


def extract_station_readings(text):
    """Pull {site_id: {pollutant: {"value","unit","observed_at","source"}}}
    out of a fetched bulletin.

    Two strategies, in order:
      1. a labelled row — the station name on a line that also names the
         pollutant and its value ("Ashok Vihar NO2: 42 µg/m³")
      2. a table row — station name followed by numbers in the order the
         header declared them

    Anything it cannot tie to an explicit pollutant label is ignored. This
    function NEVER returns a value it could not place.
    """
    out = {}
    plain = _normalise(text)
    if not plain:
        return out

    lines = [ln.strip() for ln in plain.splitlines() if ln.strip()]

    # ── Strategy 1: "station … pollutant … value unit" on one line ─────────
    for station in STATIONS:
        if station.get("retired"):
            continue                                   # offline, never fetched
        pattern = _station_pattern(station)
        for line in lines:
            if not pattern.search(line):
                continue
            for pollutant in POLLUTANTS:
                found = _labelled_value(line, pollutant)
                if found is None:
                    continue
                out.setdefault(station["site_id"], {})[pollutant] = found

    # ── Strategy 2: header-ordered table ───────────────────────────────────
    if not out:
        out.update(_extract_table(lines))

    return out


def _labelled_value(line, pollutant):
    """`no2: 42 µg/m3` / `o3 42.5` / `pm2.5 — 96 (ug/m3)` on a station line.

    Every occurrence of the label is tried, not just the first: a page heading
    such as "Delhi real time air quality index" also contains the token
    "air quality index", and taking that first match would silently discard a
    value that is stated correctly further along the line. Only a label that is
    immediately followed by a number counts — a bare label yields nothing.
    """
    pattern = _pollutant_pattern(pollutant)
    for match in pattern.finditer(line):
        tail = line[match.end():]
        # The value must be the first number after the label, with an optional
        # separator; a label at end-of-line yields nothing (never a guess).
        value_match = re.match(
            r"[\s:：=\-–—|,]*"
            r"([<>]?\s*\d[\d,]*(?:\.\d+)?)"
            r"[\s\(\[]*([a-zµμμ^/\s³]*)", tail)
        if not value_match:
            continue
        raw_value = value_match.group(1)
        raw_unit = (value_match.group(2) or "").strip()

        value = live_validate.parse_number(raw_value)
        if value is None:
            continue
        unit = _resolve_unit(pollutant, raw_unit, line)
        if unit is None:
            continue
        return {"value": value, "unit": unit, "observed_at": _timestamp_from(line)}
    return None


def _resolve_unit(pollutant, declared, line):
    """Unit for a value: declared first, else the line's context.

    An AQI value is unitless (index). A gas value on a CPCB/DPCC bulletin is
    µg/m³; PM2.5 is likewise µg/m³. When the line states a unit, that wins.
    """
    declared = (declared or "").strip()
    if pollutant == "AQI":
        return "index"
    canonical = live_validate.canonical_unit(declared)
    if canonical in ("ug/m3", "ppb"):
        return canonical
    # No explicit unit: CPCB/DPCC bulletins state gases and PM in µg/m³.
    if re.search(r"ug\s*/?\s*m|µg|μg|micro", line):
        return "ug/m3"
    if re.search(r"\bppb\b", line):
        return "ppb"
    return "ug/m3"


def _timestamp_from(line):
    """An explicit timestamp in the line, else None (caller falls back)."""
    match = re.search(
        r"(\d{1,2}[:.]\d{2}\s*(?:am|pm)?|"
        r"\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}|"
        r"\d{4}-\d{2}-\d{2}(?:[t ]\d{1,2}:\d{2}(?::\d{2})?)?)", line)
    if not match:
        return None
    parsed = live_validate._parse_timestamp(match.group(1).replace(".", ":"))
    return parsed


def _extract_table(lines):
    """Header-ordered table: `station | no2 | o3 | pm2.5 | aqi`."""
    header_index, columns = None, None
    for index, line in enumerate(lines):
        found = [(p, _pollutant_pattern(p).search(line)) for p in POLLUTANTS]
        hits = [p for p, m in found if m]
        if len(hits) >= 2 and index + 1 < len(lines):
            header_index, columns = index, found
            break
    if header_index is None:
        return {}

    out = {}
    for line in lines[header_index + 1:]:
        if not line or "|" not in line and "\t" not in line:
            if not re.search(r"\d", line):
                continue
        cells = [c.strip(" :-–—|") for c in re.split(r"[|\t]", line)]
        cells = [c for c in cells if c]
        if len(cells) < 3:
            continue
        station = _match_station(cells[0])
        if station is None or station.get("retired"):
            continue
        numbers = [live_validate.parse_number(c) for c in cells[1:]]
        for (pollutant, _match), number in zip(columns, numbers):
            if number is None or pollutant is None:
                continue
            unit = "index" if pollutant == "AQI" else "ug/m3"
            out.setdefault(station["site_id"], {})[pollutant] = {
                "value": number, "unit": unit, "observed_at": None,
            }
    return out


def _match_station(text):
    """Which registry station a table cell names, or None."""
    cell = (text or "").strip().lower()
    if not cell:
        return None
    for station in STATIONS:
        for alias in station["aliases"] + [station["name"]]:
            alias_low = alias.lower()
            if cell == alias_low or cell.startswith(alias_low) or alias_low in cell:
                return station
    return None


def match_station_by_name(name):
    """Which of our 7 sites a live source names, or None.

    Deliberately strict: a name must match a site or one of its aliases after
    punctuation is normalised. A DPCC station that is simply *near* one of our
    sites (Anand Vihar is close to East Delhi) returns None — substituting it
    would put a real number next to the wrong label, which is the one thing
    the spec's "reject, never guess" rule exists to prevent.
    """
    cleaned = re.sub(r"[^a-z0-9]+", " ", str(name or "").lower()).strip()
    if not cleaned:
        return None
    for station in STATIONS:
        candidates = [station["name"]] + list(station.get("aliases", []))
        for alias in candidates:
            normalised = re.sub(r"[^a-z0-9]+", " ", alias.lower()).strip()
            if cleaned == normalised:
                return station
    return None


def city_payload():
    """The Delhi-wide bulletin row, or None when no bulletin is stored.

    Kept separate from the 7 sites: it is a city average, not a station, and
    presenting it as one would mislabel the geography.
    """
    row = next(
        (r for r in live_store.latest_readings()
         if r["station_id"] == CITY_STATION_ID and r["pollutant"] == "AQI"),
        None)
    if row is None:
        return None
    now = now_ist()
    status = freshness(row["observed_at"], now)
    return {
        "city": "Delhi",
        "aqi": row["value"],
        "unit": row["unit"],
        "status": status,
        "observed_at": row["observed_at"],
        "source": row["source"],
        "url": row["url"],
        "age_minutes": live_validate.age_minutes(row["observed_at"], now),
    }


# ── Job A: station readings ─────────────────────────────────────────────────
READINGS_TTL = 60 * 60              # 60 min (spec §2A)

# DPCC's live portal renders its station map client-side: the fetched HTML is a
# 286-char shell with no station names in it, so Search/Fetch cannot read it.
# That is exactly the "page needs JavaScript" case the spec allows web
# automation for — one run, strict JSON out, still through the single wrapper
# so it is gated by the cap and logged like any other call.
DPCC_AUTOMATION_URL = "https://dpccairdata.com/"
DPCC_AUTOMATION_TIMEOUT = 180.0
DPCC_AUTOMATION_SECONDS = 120

_DPCC_SCHEMA = {
    "type": "object",
    "properties": {
        "observed_at": {"type": "string"},
        "source": {"type": "string"},
        "stations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "no2": {"type": "number"},
                    "o3": {"type": "number"},
                    "pm25": {"type": "number"},
                    "aqi": {"type": "number"},
                    "unit": {"type": "string"},
                },
                "required": ["name"],
            },
        },
    },
    "required": ["stations"],
}

_DPCC_GOAL = (
    "This page shows real-time ambient air quality data for Delhi monitoring "
    "stations as a clickable map. Open every station location you can reach "
    "and record its readings. Return one entry per station with: name, no2, "
    "o3, pm25 and aqi as plain numbers, the unit the page states for the gas "
    "values (for example ug/m3 or ppb), the page's own observation/last-updated "
    "timestamp as observed_at, and the issuing body as source. Only include a "
    "number the page actually displays; if a station does not show a field, "
    "omit that property. Do not estimate or interpolate any value."
)

# Second source: a Delhi-wide bulletin that IS readable with plain Fetch. It
# carries an overall AQI, per-pollutant AQI sub-indices and weather, and gives
# job A a second opinion for the §3 conflict check. Note the sub-indices are
# AQI points, NOT concentrations — they are stored as `index`, never as ppb.
CITY_BULLETIN_URLS = [
    "https://aqicn.org/city/delhi/",
    "https://cpcb.nic.in/aqi_report_new.php",
]

CITY_STATION_ID = "CITY"           # pseudo-station for the Delhi-wide bulletin

# ── Per-station readings (spec §2A) ─────────────────────────────────────────
# WAQI publishes a live page per DPCC monitor and is on the spec's §2 source
# allowlist; its Delhi data is attributed to DPCC, so the instrument behind
# each page is the same one this project's archive was built from. Each slug
# was resolved against WAQI's own search and then verified — WAQI answers an
# unknown slug by silently serving the Delhi *city* page, so `name_hint` below
# is checked on every read and a mismatch is rejected rather than stored.
#
# Site 3 (Jawaharlal Nehru Stadium) stopped reporting 2023-12-31 and is
# deliberately absent: it is served as "offline", never given a neighbour's
# numbers.
WAQI_STATIONS = (
    {"site_id": 1, "slug": "satyawati-college",
     "name_hint": "Satyawati College"},
    {"site_id": 2, "slug": "national-institute-of-malaria-research--sector-8--dwarka",
     "name_hint": "Dwarka"},
    {"site_id": 4, "slug": "narela", "name_hint": "Narela"},
    {"site_id": 5, "slug": "dite-okhla", "name_hint": "Okhla"},
    {"site_id": 6,
     "slug": "shaheed-sukhdev-college-of-business-studies--rohini",
     "name_hint": "Rohini"},
    {"site_id": 7, "slug": "sonia-vihar-water-treatment-plant-djb",
     "name_hint": "Sonia Vihar"},
)

WAQI_BASE = "https://aqicn.org/city/delhi/"


def waqi_url(slug):
    return f"{WAQI_BASE}{slug}/"


_SHARE_NAME = _STATION_HEADING = re.compile(
    # WAQI heads a station page with "Narela, Delhi AQI:" and the city page
    # with a bare "Delhi AQI:". The comma is therefore what separates a real
    # station from the fallback page WAQI serves for an unknown slug. The
    # Share widget that would name the station sits past the fetched excerpt,
    # so the heading is the only reliable marker inside the text we get back.
    #
    # WAQI uses two heading shapes - "Narela, Delhi AQI: Narela, Delhi
    # Real-time Air Quality Index (AQI)." and, on the longer station names,
    # "Sonia Vihar Water Treatment Plant DJB, Delhi Real-time Air Quality
    # Index (AQI)." with no "AQI:" at all. Both contain ", Delhi Real-time
    # Air Quality Index", so that common fragment is what is matched; the
    # city fallback reads "Delhi AQI: Delhi Real-time ..." with no comma
    # before "Delhi", so it can never satisfy this pattern.
    # `[*\s]*` tolerates the markdown emphasis Fetch wraps the heading in -
    # the same page arrives as "Narela, Delhi AQI:" from one call and
    # "**Narela, Delhi AQI**:" from another.
    r"([A-Za-z][^|\n:*]{0,60}?),\s*Delhi\s+Real[-\s]?time\s+Air\s+Quality",
    re.I)

# Search terms used only as a fallback when both fetch sources fail.
READINGS_PURPOSE = ("Find the current station-wise NO2, O3, PM2.5 and AQI "
                    "readings for the Delhi monitoring stations that the "
                    "Vaayu Darshiya dashboard displays as live readings")

DP_FIELD_MAP = {"no2": "NO2", "o3": "O3", "pm25": "PM25", "aqi": "AQI"}


def _normalise_unit(raw):
    """The unit string an automation run reported, normalised.

    DPCC labels its columns "µg/m3"; TinyFish may hand back any of the usual
    spellings, and an unrecognised one is rejected downstream rather than
    assumed to be ppb (spec §3).
    """
    text = str(raw or "").strip().lower()
    if "ppb" in text:
        return "ppb"
    if "ug" in text or "µg" in text or "μg" in text or "micro" in text:
        return "ug/m3"
    if "index" in text or text == "aqi":
        return "index"
    return text


def fetch_dpcc_readings(fetched_at):
    """Web-automation pass over DPCC's map → validated reading candidates.

    Returns (rows, source_meta, errors). Never raises: a failed run leaves
    `rows` empty and the caller falls back to the fetch sources.
    """
    try:
        result = tinyfish_service.automate(
            DPCC_AUTOMATION_URL, _DPCC_GOAL, output_schema=_DPCC_SCHEMA,
            timeout=DPCC_AUTOMATION_TIMEOUT,
            max_duration_seconds=DPCC_AUTOMATION_SECONDS,
            purpose="read the live station-wise air quality table for the "
                    "Vaayu Darshiya dashboard")
    except tinyfish_service.TinyFishError as exc:
        return [], [], [str(exc)], []

    source_name = str(result.get("source") or "DPCC").strip() or "DPCC"
    observed_raw = str(result.get("observed_at") or "").strip()
    observed = live_validate._parse_timestamp(observed_raw)
    # A date without a time is not enough to judge staleness — keep the date
    # but let the caller fall back to the fetch time for the hour.
    if observed is not None and (observed.hour, observed.minute, observed.second) == (0, 0, 0):
        observed = None

    unit = _normalise_unit(
        next((s.get("unit") for s in (result.get("stations") or [])
              if isinstance(s, dict) and s.get("unit")), "ug/m3"))

    rows, unmatched = [], []
    for entry in result.get("stations") or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        station = match_station_by_name(name)
        if station is None:
            # A real station that is not one of our 7. Never mapped onto a
            # nearby site — that would be inventing a reading (spec §3).
            unmatched.append(name)
            continue
        if station.get("retired"):
            # One of ours, but it stopped reporting on 2023-12-31. Its card is
            # already served as "offline", so a stray live value for it is
            # dropped here rather than resurrecting a dead station.
            continue
        for dp_field, pollutant in DP_FIELD_MAP.items():
            value = entry.get(dp_field)
            if value is None:
                continue
            entry_unit = "index" if pollutant == "AQI" else unit
            rows.append({
                "station_id": str(station["site_id"]),
                "station_name": station["name"],
                "pollutant": pollutant,
                "value": value,
                "unit": entry_unit,
                "observed_at": observed or fetched_at,
                "source": source_name,
                "url": DPCC_AUTOMATION_URL,
            })

    meta = {
        "name": source_name, "url": DPCC_AUTOMATION_URL,
        "fetched_at": fetched_at,
        "observed_at": live_store.iso(observed) if observed else None,
        "kind": "station",
    }
    return rows, ([meta] if rows else []), [], unmatched


def fetch_city_bulletin(fetched_at):
    """Delhi-wide bulletin via plain Fetch → (row dict | None, meta, error).

    Readable without automation, so it doubles as the fallback that keeps a
    city-wide live number on screen when the station pass fails.
    """
    for url in CITY_BULLETIN_URLS:
        try:
            page = tinyfish_service.fetch_text(url, limit=9000, ttl=0)
        except tinyfish_service.TinyFishError as exc:
            continue
        text = page.get("text") or ""
        parsed = parse_city_bulletin(text)
        if not parsed:
            continue
        source_name, _ = tinyfish_context.source_for(url)
        observed = parsed.get("observed") or fetched_at
        return {
            "station_id": CITY_STATION_ID,
            "station_name": "Delhi (city-wide)",
            "pollutant": "AQI",
            "value": parsed["aqi"],
            "unit": "index",
            "observed_at": observed,
            "source": source_name,
            "url": url,
        }, {
            "name": source_name, "url": url, "fetched_at": fetched_at,
            "observed_at": live_store.iso(observed)
            if isinstance(observed, datetime) else None,
            "kind": "city",
        }, None
    return None, None, "no city bulletin could be read"


def fetch_waqi_readings(fetched_at):
    """Per-station AQI via WAQI → (rows, meta, errors).

    ONE `fetch_many` call covers all six live sites, which is what keeps job A
    inside TINYFISH_DAILY_CALL_CAP for a full 24 h window — seven separate
    `fetch_text` calls would not.

    WAQI publishes per-pollutant AQI *sub-indices* alongside the station AQI.
    Those are indices, not concentrations, so they are deliberately NOT stored
    as NO2/O3/PM2.5: writing "NO2 = 11" when the page means "NO2 sub-index 11"
    would be inventing a reading (spec §3). Only the station AQI, which really
    is an index, is kept.

    Returns empty rows with an error string when the batch fails, so the
    caller keeps its previous readings and serves them with their age.
    """
    specs = [s for s in WAQI_STATIONS
             if not STATION_BY_ID.get(s["site_id"], {}).get("retired")]
    urls = [waqi_url(s["slug"]) for s in specs]
    if not urls:
        return [], [], ["no live WAQI stations configured"]

    try:
        batch = tinyfish_service.fetch_many(
            urls, ttl=0,
            purpose="read the live station-wise AQI for the Vaayu Darshiya "
                    "dashboard's Delhi monitoring sites")
    except tinyfish_service.TinyFishError as exc:
        return [], [], [str(exc)]

    rows, errors, unmatched = [], [], []
    results = batch.get("results") or []

    # Keyed by URL, NOT by position: `results[]` holds one entry per
    # *successfully* fetched URL, so a single failure would shift every later
    # entry one slot left and quietly label station N's reading as station N+1.
    # The result echoes the URL we asked for, which is the only safe join.
    by_url = {}
    for page in results:
        if not isinstance(page, dict):
            continue
        original = str(page.get("url") or "").strip()
        final = str(page.get("final_url") or "").strip()
        if original:
            by_url[original] = page
        if final:
            by_url.setdefault(final, page)
    for failure in batch.get("errors") or []:
        if isinstance(failure, dict) and failure.get("url"):
            errors.append(f"{failure['url']}: {failure.get('error', 'failed')}")

    for spec in specs:
        url = waqi_url(spec["slug"])
        page = by_url.get(url)
        if page is None:
            errors.append(f"{spec['slug']}: no result")
            continue
        text = page.get("text") or ""
        parsed = parse_city_bulletin(text)
        if not parsed:
            errors.append(f"{spec['slug']}: no AQI on the page")
            continue

        # Fallback guard: an unknown slug serves the Delhi *city* page, whose
        # Share line names the city rather than the station. Accepting it
        # would put one city-wide number on six station cards.
        seen_match = _SHARE_NAME.search(text)
        seen = seen_match.group(1) if seen_match else None
        if not seen or seen.strip().lower() == "delhi":
            unmatched.append(spec["slug"])
            continue
        if spec["name_hint"].lower() not in seen.lower():
            # A real station, but not the one this slug was verified against.
            unmatched.append(f"{spec['slug']} -> {seen}")
            continue

        station = STATION_BY_ID[spec["site_id"]]
        observed = parsed.get("observed") or fetched_at
        rows.append({
            "station_id": str(station["site_id"]),
            "station_name": station["name"],
            "pollutant": "AQI",
            "value": parsed["aqi"],
            "unit": "index",
            "observed_at": observed,
            "source": "WAQI (DPCC)",
            "url": waqi_url(spec["slug"]),
        })

    meta = {
        "name": "WAQI (DPCC)", "url": WAQI_BASE, "fetched_at": fetched_at,
        "observed_at": None, "kind": "station",
    }
    if unmatched:
        # Surfaced in /api/status so an operator can see why a card is dark
        # rather than assuming the scheduler stalled.
        errors.append(f"unmatched WAQI pages: {', '.join(unmatched)}")
    return rows, ([meta] if rows else []), errors


_AQICN_RE = {
    "overall": re.compile(r"\|\s*(\d{1,3})\s*\|\s*(?:good|moderate|unhealthy|"
                          r"hazardous|severe|very poor|poor|satisfactory)", re.I),
    "updated": re.compile(r"updated on ([^|]{4,40}?)(?:\s*\||\s+temperature)", re.I),
    "temp": re.compile(r"temperature:\s*(\d{1,3})\s*°?c", re.I),
    "humidity": re.compile(r"humidity\s*\|\s*(\d{1,3})", re.I),
    "wind": re.compile(r"\|\s*wind\s*\|\s*(\d{1,3})", re.I),
    "sub": re.compile(r"(pm2\.5|pm10|o3|no2|so2|co)\s*aqi\s*\|\s*(\d{1,3})", re.I),
}


def parse_city_bulletin(text):
    """Extract the Delhi-wide AQI and weather from a bulletin page.

    Returns {} when the page does not actually state an AQI — a page that is
    mostly navigation chrome must not be mistaken for a reading.
    """
    if not text:
        return {}
    plain = _normalise(text)
    out = {}

    sub = {m.group(1).lower().replace(".", ""): int(m.group(2))
           for m in _AQICN_RE["sub"].finditer(plain)}
    if sub:
        out["sub_indices"] = sub

    overall = None
    match = _AQICN_RE["overall"].search(plain)
    if match:
        overall = int(match.group(1))
    elif sub:
        # The worst sub-index determines the overall AQI (CPCB's own rule).
        overall = max(sub.values())
    if overall is None or not (0 <= overall <= 500):
        return {}
    out["aqi"] = overall

    match = _AQICN_RE["updated"].search(plain)
    if match:
        observed = _bulletin_timestamp(match.group(1))
        if observed is not None:
            out["observed"] = observed

    for key, pattern in (("temp_c", _AQICN_RE["temp"]),
                         ("humidity_pct", _AQICN_RE["humidity"]),
                         ("wind_kmph", _AQICN_RE["wind"])):
        match = pattern.search(plain)
        if match:
            out[key] = live_validate.parse_number(match.group(1))
    return out


def _bulletin_timestamp(raw):
    """'saturday 18:00' / 'oct 10 2026 18:00' → an IST datetime, else None."""
    text = str(raw or "").strip().lower().rstrip(",")
    if not text:
        return None
    clock = re.search(r"(\d{1,2})[:.](\d{2})\s*(am|pm)?", text)
    hour = minute = None
    if clock:
        hour, minute = int(clock.group(1)), int(clock.group(2))
        if clock.group(3) == "pm" and hour < 12:
            hour += 12
        elif clock.group(3) == "am" and hour == 12:
            hour = 0
    elif re.search(r"\d{1,2}\s*(am|pm)", text):
        clock = re.search(r"(\d{1,2})\s*(am|pm)", text)
        hour, minute = int(clock.group(1)), 0
        if clock.group(2) == "pm" and hour < 12:
            hour += 12
        elif clock.group(2) == "am" and hour == 12:
            hour = 0

    day = re.search(r"(\d{1,2})\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
                    r"(?:\s+(\d{4}))?", text)
    if day:
        month = "janfebmaraprmayjunjulaugsepoctnovdec".index(day.group(2)) // 3 + 1
        year = int(day.group(3)) if day.group(3) else now_ist().year
        try:
            return datetime(year, month, int(day.group(1)),
                            hour or 0, minute or 0, tzinfo=IST)
        except ValueError:
            return None
    if hour is not None:
        today = now_ist()
        candidate = today.replace(hour=hour, minute=minute, second=0, microsecond=0)
        # A clock time with no date: yesterday when it is ahead of now.
        return candidate if candidate <= today else candidate - timedelta(days=1)
    return None


def refresh_readings(force=False):
    """Job A — the latest live reading for each station, plus the Delhi bulletin.

    Two independent passes:
      1. web automation over DPCC's map — the only source that carries
         station-wise values, and the only reason job A spends an automation
         call instead of a fetch
      2. a plain-Fetch Delhi-wide bulletin, which also supplies the second
         opinion for the §3 conflict check and keeps a city-level number on
         screen when no station reports

    A station is matched to a site ONLY by name. DPCC currently runs Anand
    Vihar, Mandir Marg, Punjabi Bagh, R.K. Puram, Major Dhyan Chand National
    Stadium, Nehru Nagar and Rohini — of our seven sites only Rohini is in that
    network. The other six are reported as offline rather than being silently
    replaced by a nearby station that is not the same location.

    Raises TinyFishError when both passes fail, so the caller keeps the
    previous readings and serves them with their age.
    """
    fetched_at_dt = now_ist()
    fetched_at = _iso(fetched_at_dt)

    # Pass 1 — per-station AQI from WAQI. One batched Fetch, no JavaScript
    # needed, so it is tried first (spec §2: prefer Search/Fetch).
    waqi_rows, waqi_sources, waqi_errors = fetch_waqi_readings(fetched_at_dt)

    # Pass 2 — DPCC's map via automation, run only when WAQI gave nothing.
    # It is a real browser session, so spending one every hour on a source
    # that already answers in Fetch would burn the cap for no extra coverage.
    if waqi_rows:
        dpcc_rows, sources, dpcc_errors, unmatched = [], [], [], []
    else:
        dpcc_rows, sources, dpcc_errors, unmatched = fetch_dpcc_readings(fetched_at_dt)

    city_row, city_meta, city_error = fetch_city_bulletin(fetched_at_dt)
    sources = waqi_sources + sources
    if city_meta:
        sources.append(city_meta)

    # Fail only when nothing could be READ. If a source answered but none of
    # the stations it named is one of our seven, that is a real (if empty)
    # result: the job succeeds and every site is reported offline, which is
    # the truth. Reporting it as an error would hide the actual reason.
    if not waqi_rows and not dpcc_rows and city_row is None and (
            dpcc_errors or waqi_errors or city_error):
        detail = {"dpcc_errors": dpcc_errors, "city_error": city_error,
                  "waqi_errors": waqi_errors}
        live_store.record_job("readings", "error",
                              error="; ".join(dpcc_errors or [city_error or ""]),
                              detail=detail)
        raise tinyfish_service.TinyFishError(
            "no live readings could be read from WAQI, DPCC or the city bulletin")

    # ── Validate, resolve conflicts, persist ───────────────────────────────
    accepted, rejected = [], 0
    existing = {(row["station_id"], row["pollutant"]): row
                for row in live_store.latest_readings()}
    now = fetched_at_dt

    all_rows = list(waqi_rows) + list(dpcc_rows)
    if city_row:
        all_rows.append(city_row)

    for payload in all_rows:
        try:
            row = validate(
                payload["pollutant"], payload["value"], payload["unit"],
                observed_at=payload.get("observed_at") or fetched_at_dt,
                source=payload.get("source", "unknown"), now=now)
        except Rejected:
            rejected += 1                             # keep the previous value
            continue
        row.update({
            "station_id": payload["station_id"],
            "station_name": payload["station_name"],
            "url": payload.get("url"),
            "status": freshness(row["observed_at"], now),
        })
        prior = existing.get((payload["station_id"], payload["pollutant"]))
        row, conflicted = resolve_conflict(
            row,
            {**prior, "value": prior["value"]} if prior else None)
        if conflicted:
            row["status"] = "stale"                   # do not present as clean live
        row.setdefault("conflict", 0)
        accepted.append(row)

    live_store.save_readings(accepted)
    live_store.prune()

    covered = {int(row["station_id"]) for row in accepted
               if row["station_id"].isdigit()}
    offline = [s["name"] for s in STATIONS
               if s.get("retired") or s["site_id"] not in covered]

    detail = {
        "accepted": len(accepted), "rejected": rejected,
        "stations_covered": sorted(covered), "offline": offline,
        "sources": len(sources),
        # Visible in /api/status so an operator can see DPCC's current network
        # differs from the 7 archived sites, rather than guessing why stations
        # are dark.
        "unmatched_live_stations": unmatched,
        "dpcc_errors": dpcc_errors,
        "waqi_errors": waqi_errors,
        "city_error": city_error,
    }
    live_store.record_job("readings", "ok", detail=detail)
    return {
        "readings": accepted,
        "sources": sources,
        "rejected": rejected,
        "offline": offline,
        "fetched_at": fetched_at,
        "city": city_payload(),
        **detail,
    }



# ── Job B: official alerts (reuses the existing context extractor) ─────────
ALERTS_TTL = 45 * 60                # 30-60 min (spec §2B)


def refresh_alerts(force=False):
    """Job B — official GRAP/CAQM/IMD advisories, then reputable press."""
    payload = tinyfish_context.get_alerts(force=force)
    items = payload.get("items") or []
    stored = [{
        "id": item["id"], "title": item["title"],
        "summary": item.get("summary", "")[:500],
        "severity": item.get("severity", "low"),
        "source": item.get("source", "unknown"),
        "url": item["url"],
        "official": 1 if item.get("official") else 0,
        "kind": _alert_kind(item),
        "published_at": item.get("published_at"),
        "fetched_at": item.get("fetched_at") or _iso(),
    } for item in items if isinstance(item, dict) and item.get("url")]

    live_store.save_alerts(stored)
    live_store.prune()
    live_store.record_job("alerts", "ok", detail={"items": len(stored)})
    return {**payload, "items": stored, "fetched_at": _iso()}


def _alert_kind(item):
    blob = f"{item.get('title', '')} {item.get('summary', '')}".lower()
    if any(t in blob for t in ("grap", "stage 1", "stage 2", "stage 3", "stage 4",
                               "caqm", "curbs", "restrictions")):
        return "grap"
    if any(t in blob for t in ("fog", "rain", "weather", "imd", "cold wave", "heat")):
        return "weather"
    if any(t in blob for t in ("stubble", "crop", "traffic", "vehicle")):
        return "notice"
    return "news"


# ── Job C: weather ──────────────────────────────────────────────────────────
WEATHER_TTL = 2 * 3600              # 2-3 h (spec §2C)


def refresh_weather(force=False):
    """Job C — Delhi temperature, humidity, wind, visibility, rain/fog flags.

    Reuses the existing IMD parser in tinyfish_context (it already reads only
    the values a source states) and persists the result.
    """
    payload = tinyfish_context.get_weather_windows([], force=force)
    weather = payload.get("weather") or {}
    parsed = tinyfish_context.parse_weather(weather.get("text", "")) \
        if weather.get("text") else {}

    wind_max = parsed.get("wind_max_kmph") or weather.get("wind_max_kmph")
    row = {
        "city": "Delhi",
        "temp_c": parsed.get("temp_c") if parsed.get("temp_c") is not None
                  else weather.get("temp_c"),
        "humidity_pct": parsed.get("humidity_pct") if parsed.get("humidity_pct") is not None
                        else weather.get("humidity_pct"),
        "wind_min_kmph": parsed.get("wind_min_kmph"),
        "wind_max_kmph": wind_max,
        # The forecast rarely states visibility; derive it from rain when it
        # does not, and leave it null rather than inventing a number.
        "visibility_km": _visibility(weather.get("text"), parsed.get("rain")),
        "fog": 1 if (parsed.get("fog") or weather.get("fog")) else 0,
        "rain": 1 if (parsed.get("rain") or weather.get("rain")) else 0,
        "text": tinyfish_context.sanitize(weather.get("text", ""), limit=400),
        "source": weather.get("source", "unknown"),
        "url": weather.get("url"),
        "observed_at": weather.get("fetched_at"),
        "fetched_at": _iso(),
        "status": "stale" if payload.get("stale") else "live",
    }
    live_store.save_weather(row)
    live_store.prune()
    live_store.record_job("weather", "ok",
                          detail={"source": row["source"], "fog": row["fog"]})
    return {**payload, "weather": row, "fetched_at": _iso()}


def _visibility(text, rain):
    """Visibility in km: stated first, else rain-derived, else None."""
    match = re.search(
        r"(?<![\d.])(\d{1,3}(?:\.\d+)?)\s*(?:km|kilometre|kilometer)\b",
        tinyfish_context.sanitize(text or "", limit=2000).lower())
    if match and 0 < float(match.group(1)) <= 30:
        value = float(match.group(1))
        if "visibility" in (text or "").lower():
            return round(value, 1)
    if rain:
        return None                                    # derived elsewhere, not guessed here
    return None


# ── Job D: Delhi case-study stats (refresh once) ───────────────────────────
CASE_STATS = [
    {"key": "annual_pm25", "label": "Annual mean PM2.5",
     "query": "Delhi annual average PM2.5 CPCB annual report micrograms",
     "pattern": r"annual\s+(?:mean\s+)?pm\s*2\.?5[^0-9]{0,80}?(\d{1,3}(?:\.\d+)?)",
     "unit": "µg/m³"},
    {"key": "winter_aqi", "label": "Winter peak AQI",
     "query": "Delhi winter peak AQI CPCB worst air quality index value",
     "pattern": r"(?:peak|worst|maximum|max)\s+aqi[^0-9]{0,60}?(\d{3,4})",
     "unit": "index"},
    {"key": "news_strip", "label": "Latest Delhi air quality report",
     "query": "Delhi air quality latest report CPCB CAQM",
     "pattern": None, "unit": None},
]


def refresh_case_study(force=False):
    """Job D — the figures behind "Delhi's Drastic Case", each with a source
    and an "as of" date. Values the sources do not state are simply absent."""
    fetched_at = _iso()
    stored, sources = [], []
    for stat in CASE_STATS:
        try:
            results = tinyfish_service.search(
                stat["query"], domain_type="web", page_size=4,
                include_domains=LIVE_SOURCE_DOMAINS, purpose=CASE_STATS[0]["label"])
        except tinyfish_service.TinyFishError:
            continue
        for result in results:
            url = result.get("url", "")
            if not url or not tinyfish_context.is_allowed(url):
                continue
            text, title = result.get("snippet", ""), result.get("title", "")
            try:
                page = tinyfish_service.fetch_text(url, limit=4000, ttl=0)
                text = page.get("text") or text
                url = page.get("url") or url
            except tinyfish_service.TinyFishError:
                pass
            if stat["pattern"]:
                match = re.search(stat["pattern"], _normalise(text))
                if not match:
                    continue
                value = match.group(1)
            else:
                summary = tinyfish_context.summarize(text, limit=280)
                if not summary:
                    continue
                value = summary
            source_name, _ = tinyfish_context.source_for(url)
            stored.append({
                "stat_key": stat["key"], "label": stat["label"],
                "value": str(value)[:500], "unit": stat["unit"],
                "as_of": tinyfish_context.parse_published(
                    None, result.get("date")) or fetched_at[:10],
                "source": source_name, "url": url, "fetched_at": fetched_at,
            })
            sources.append({"name": source_name, "url": url, "fetched_at": fetched_at})
            break                                       # first usable source wins

    live_store.save_case_stats(stored)
    live_store.record_job("case_study", "ok", detail={"stats": len(stored)})
    return {"stats": stored, "sources": sources, "fetched_at": fetched_at}


# ── Read side: what the API serves ──────────────────────────────────────────
def station_payload():
    """Merge live readings onto the 7 sites, keeping the existing shape.

    Each station gains: status, observed_at, source, age_minutes and a
    `live` block holding the web-sourced values. ML forecast fields (no2, o3,
    hcho) are NEVER written here — they stay in their own fields (spec §3).
    """
    latest = {(row["station_id"], row["pollutant"]): row
              for row in live_store.latest_readings()}
    now = now_ist()
    out = []
    for station in STATIONS:
        readings = {p: latest.get((str(station["site_id"]), p)) for p in POLLUTANTS}
        # Status is recomputed from the age at READ time, never taken from the
        # row's stored status: a reading stored as `live` at 09:00 must be
        # reported `stale` at 14:00 if the source went quiet.
        for row in readings.values():
            if row:
                row["status"] = freshness(row["observed_at"], now)
        fresh = [r for r in readings.values() if r and r["status"] == "live"]
        if station.get("retired"):
            status, observed, source, age = "offline", None, None, None
        elif fresh:
            newest = max(fresh, key=lambda r: r["observed_at"])
            status, observed = newest["status"], newest["observed_at"]
            source, age = newest["source"], live_validate.age_minutes(observed, now)
        elif any(readings.values()):
            any_row = next(r for r in readings.values() if r)
            status, observed = "stale", any_row["observed_at"]
            source, age = any_row["source"], live_validate.age_minutes(observed, now)
        else:
            status, observed, source, age = "offline", None, None, None

        out.append({
            "site_id": station["site_id"],
            "name": station["name"],
            "lat": station["lat"], "lon": station["lon"],
            "status": status,
            "observed_at": observed,
            "source": source,
            "age_minutes": age,
            "retired": station.get("retired"),
            "live": {p: (r["value"] if r else None) for p, r in readings.items()},
            "live_units": {p: (r["unit"] if r else None) for p, r in readings.items()},
            "conflict": any(r and r.get("conflict") for r in readings.values()),
        })
    return out


def station_payload_envelope():
    """`station_payload()` plus the freshness summary the header needs (§8).

    Same array under `stations`, plus overall `status` / `updated_at` / a
    human label so the UI can say "Live - updated 14:32 IST", "Stale - last
    update 2 h ago" or "Live context ended" without recomputing anything.
    """
    stations = station_payload()
    now = now_ist()

    observed = [s["observed_at"] for s in stations if s.get("observed_at")]
    latest = max(observed) if observed else None
    age = live_validate.age_minutes(latest, now) if latest else None

    live_count = sum(1 for s in stations if s["status"] == "live")
    stale_count = sum(1 for s in stations if s["status"] == "stale")
    offline_count = sum(1 for s in stations if s["status"] == "offline")

    import tinyfish_service
    if tinyfish_service.is_expired():
        status = "ended"
    elif not observed:
        status = "unavailable"
    elif live_count:
        status = "live"
    elif stale_count:
        status = "stale"
    else:
        status = "unavailable"

    return {
        "stations": stations,
        "status": status,
        "updated_at": latest,
        "age_minutes": age,
        "counts": {"live": live_count, "stale": stale_count,
                   "offline": offline_count},
        "label": _header_label(status, latest, age),
        "generated_at": now.isoformat(timespec="seconds"),
    }


def _header_label(status, latest, age):
    """Header text (spec §8): Live / Stale / Live context ended."""
    import tinyfish_service
    if status == "ended":
        import live_stream
        return live_stream._ended_label() or "Live context ended"
    if status == "unavailable" or latest is None:
        return "No live data yet"
    stamp = live_store.parse_iso(latest)
    hhmm = stamp.strftime("%H:%M") if stamp else ""
    if status == "live":
        return f"Live - updated {hhmm} IST"
    return f"Stale - last update {age} min ago"


def weather_payload():
    row = live_store.latest_weather()
    if not row:
        return {"status": "unavailable", "message": "No live weather yet."}
    age = live_validate.age_minutes(row.get("observed_at") or row.get("fetched_at"))
    return {
        "city": row.get("city", "Delhi"),
        "temp_c": row.get("temp_c"),
        "humidity_pct": row.get("humidity_pct"),
        "wind_min_kmph": row.get("wind_min_kmph"),
        "wind_max_kmph": row.get("wind_max_kmph"),
        "visibility_km": row.get("visibility_km"),
        "fog": bool(row.get("fog")),
        "rain": bool(row.get("rain")),
        "text": row.get("text"),
        "source": row.get("source"),
        "url": row.get("url"),
        "observed_at": row.get("observed_at") or row.get("fetched_at"),
        "fetched_at": row.get("fetched_at"),
        "age_minutes": age,
        "status": row.get("status", "live") if (age or 0) <= MAX_READING_AGE_MINUTES
                  else "stale",
    }


def alerts_payload(limit=8):
    items = live_store.latest_alerts(limit=limit)
    return {
        "items": [{
            "id": i["id"], "title": i["title"], "summary": i["summary"],
            "severity": i["severity"], "source": i["source"], "url": i["url"],
            "official": bool(i["official"]), "kind": i.get("kind"),
            "published_at": i.get("published_at"), "fetched_at": i.get("fetched_at"),
        } for i in items],
        "fetched_at": items[0]["fetched_at"] if items else None,
    }


def case_payload():
    return {"stats": live_store.case_stats()}
