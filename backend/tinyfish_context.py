"""
Live web context for the dashboard sections outside AI Insights.

One module, one TinyFish client (`tinyfish_service`), Search + Fetch only — no
paid Web Automation runs and no paid Monitors (both declined by the operator;
Monitors also need a public webhook URL, which a localhost backend has not).

Feature functions, one per section:
    get_alerts()              Active Alerts              TTL 20 min  [this file]
    get_weather_windows()     Best Time to Go Outside    TTL 6 h     [Feature 2]
    get_delhi_case()          Delhi's Drastic Case       TTL 12 h    [Feature 3]
    get_station_crosscheck()  Model vs reported          TTL 6 h     [Feature 4]

Shared plumbing:
    _serve()          TTL cache + single-flight refresh + last-good fallback
                      (a failed refresh serves the previous payload flagged
                      `stale`, never an empty card)
    start_scheduler() background daemon thread that refreshes served entries
                      once their own TTL has elapsed (checked every
                      REFRESH_SECONDS — a 6 h payload is not refetched on the
                      20 min alerts cadence)
    sanitize()        untrusted web text → plain, length-capped string

Ground rules honoured here:
  * key stays backend-side (tinyfish_service reads TINYFISH_API_KEY / .env)
  * every payload carries `fetched_at`, `stale` and `sources`
  * fetched text is data only: sanitized, truncated, never parsed as
    instructions and never executed
"""

import hashlib
import re
import threading
import time
from datetime import date as _date
from datetime import datetime as _dt

import tinyfish_service

# ── Cadence and TTLs (spec: alerts 15–30 min, case 6–24 h, advisory 24 h) ──
REFRESH_SECONDS = 20 * 60      # background scheduler interval
ALERTS_TTL = 20 * 60           # Active Alerts
MAX_ALERTS = 4                 # official/press items served per refresh
ALERT_FETCHES = 5              # pages extracted per refresh (free tier: 1k/day)
FRESH_DAYS = 540               # 18 months: older docs are reference, not alerts
SUMMARY_CHARS = 200            # spec: summary max 200 chars

# ── Source allowlists (ground rule 8: official or reputable only) ──────────
# host -> display name. Matching is suffix-based, so "www.cpcb.nic.in" and
# "airquality.cpcb.in" both count as CPCB.
OFFICIAL_SOURCES = {
    "caqm.nic.in": "CAQM",
    "cpcb.nic.in": "CPCB",
    "imd.gov.in": "IMD",
    "safar.tropmet.res.in": "SAFAR (IITM)",
    "dpcc.delhigovt.nic.in": "DPCC",
    "delhi.gov.in": "Govt of Delhi",
    "pib.gov.in": "PIB",
    "moef.gov.in": "MoEFCC",
    "who.int": "WHO",
    "openaq.org": "OpenAQ",
    "cpcb.gov.in": "CPCB",
}
REPUTED_SOURCES = {
    # WAQI / World Air Quality Index — the spec's §2 allowlist names WAQI
    # explicitly as a permitted source for station readings.
    "aqicn.org": "WAQI",
    "ndtv.com": "NDTV",
    "hindustantimes.com": "Hindustan Times",
    "timesofindia.indiatimes.com": "Times of India",
    "indianexpress.com": "The Indian Express",
    "thehindu.com": "The Hindu",
    "business-standard.com": "Business Standard",
    "livemint.com": "Mint",
}

# Queries run per refresh. The last one is restricted to official domains so a
# busy news cycle can never crowd CAQM/CPCB/IMD out of the list.
ALERT_SEARCHES = [
    {"domain_type": "news", "query": "Delhi NCR GRAP stage CAQM air quality advisory restrictions"},
    {"domain_type": "news", "query": "Delhi air quality IMD dense fog warning schools construction restrictions"},
    {"domain_type": "web", "query": "Delhi NCR GRAP stage air quality advisory",
     "include_domains": "caqm.nic.in,cpcb.nic.in,imd.gov.in,dpcc.delhigovt.nic.in,delhi.gov.in"},
]
ALERT_PURPOSE = ("Find current official pollution advisories for Delhi NCR (GRAP stage, "
                 "CAQM directions, school/construction restrictions, IMD fog or weather "
                 "warnings) for the Vaayu Darshiya Active Alerts panel")

# ── Severity vocabulary (deterministic, keyword based — never modelled) ────
_HIGH_MARKERS = (
    "stage 4", "grap 4", "grap level 4", "severe+", "severe air", "hazardous",
    "health emergency", "emergency measures", "red alert", "schools closed",
    "shut schools", "complete ban", "total ban", "ban on all", "halted",
)
_MEDIUM_MARKERS = (
    "stage 3", "stage 2", "grap", "restrict", "restriction", "curb", "ban",
    "advisory", "warn", "alert", "fog", "smog", "dust", "construction",
    "truck", "schools", "exam", "burst", "cracker", "cold day", "cold wave",
    "aqi", "pollution", "air quality", "no2", "pm2.5", "pm25",
)
_BOILERPLATE = (
    "subscribe", "newsletter", "sign in", "click here", "read more",
    "advertisement", "download app", "follow us", "updated on", "prefer ",
    "live updates", "app store", "telegram",
)

# Page chrome, not content: if a summary is mostly this, the source stated
# nothing citable and the item is dropped rather than shown as an alert.
_JUNK_MARKERS = (
    "accessibility options", "text to speech", "bigger text", "small text",
    "line height", "skip to content", "reset all settings", "cookie policy",
    "enable javascript", "browser does not support", "loading comments",
)
# "CAQM: Home", "Home", "Welcome — CPCB" style landing pages.
_HOME_TITLE = re.compile(r"^(?:[\w.:'’-]+\s+)*home(?:page)?$", re.IGNORECASE)

# ── Sanitisation of untrusted web text (ground rule 7) ─────────────────────
_CONTROL = re.compile("[\x00-\x08\x0b-\x1f\x7f\u00a0\u200b-\u200f\u2028\u2029\ufeff\ufffd]")
_HTML_TAG = re.compile(r"<[^>]*>")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_WS = re.compile(r"\s+")

# ── Cache + single-flight plumbing ─────────────────────────────────────────
_cache = {}                       # feature -> {"data": dict, "stored_at": epoch}
_cache_lock = threading.Lock()
_feature_locks = {}
_feature_locks_lock = threading.Lock()
_scheduler_state = threading.Lock()
_scheduler_started = False
_refresher_running = set()        # features currently being refreshed
_REGISTRY = {}                    # feature -> {"ttl": seconds, "refresh": callable}


def _now_iso(epoch=None):
    return _dt.fromtimestamp(epoch or time.time()).isoformat(timespec="seconds")


def _feature_lock(feature):
    with _feature_locks_lock:
        return _feature_locks.setdefault(feature, threading.Lock())


def _get_entry(feature):
    with _cache_lock:
        entry = _cache.get(feature)
    return dict(entry) if entry else None


def _put_entry(feature, data):
    with _cache_lock:
        _cache[feature] = {"data": data, "stored_at": time.time()}


def warm_features():
    """Features already served at least once (the scheduler only refreshes these)."""
    with _cache_lock:
        return sorted(_cache)


def sanitize(text, limit=500):
    """Untrusted web text → single-line plain string, capped at `limit`.

    Strips HTML/markdown links and control characters. The result is only ever
    stored and displayed: nothing in this module parses fetched text as
    instructions, code or markup.
    """
    text = "" if text is None else str(text)
    text = _MD_LINK.sub(r"\1", text)
    text = _HTML_TAG.sub(" ", text)
    text = _CONTROL.sub("", text)
    text = _WS.sub(" ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(",;:—-")
    return cut + " …"


def _host(url):
    from urllib.parse import urlparse
    return (urlparse(str(url)).netloc or "").lower().removeprefix("www.")


def _matches(host, domains):
    return any(host == d or host.endswith("." + d) for d in domains)


def source_for(url, publisher="", site_name=""):
    """Display name for an allowlisted URL, or None when the source is not allowed."""
    host = _host(url)
    for name, display in OFFICIAL_SOURCES.items():
        if host == name or host.endswith("." + name):
            return display, True
    for name, display in REPUTED_SOURCES.items():
        if host == name or host.endswith("." + name):
            return display, False
    return (publisher or site_name or host), False


def is_allowed(url):
    host = _host(url)
    return _matches(host, OFFICIAL_SOURCES) or _matches(host, REPUTED_SOURCES)


def is_official(url):
    return _matches(_host(url), OFFICIAL_SOURCES)


def _retry(fn, attempts=1):
    """One retry on transient errors; auth/quota problems are never retried."""
    try:
        return fn()
    except tinyfish_service.TinyFishError as exc:
        message = str(exc)
        non_retryable = ("HTTP 401", "HTTP 402", "HTTP 403", "TINYFISH_API_KEY is not set")
        if attempts <= 0 or any(marker in message for marker in non_retryable):
            raise
        time.sleep(0.5)
        return _retry(fn, attempts - 1)


def register_feature(feature, ttl, refresher):
    """Declare a feature's TTL and refresher — the scheduler reads this too,
    so a 6 h feature is never refetched on the 20 min alerts cadence."""
    _REGISTRY[feature] = {"ttl": ttl, "refresh": refresher}


def _serve(feature, force=False):
    """TTL cache → single-flight refresh → last good (stale) → unavailable."""
    spec = _REGISTRY[feature]
    ttl, refresher = spec["ttl"], spec["refresh"]
    entry = _get_entry(feature)
    if entry and not force and time.time() - entry["stored_at"] < ttl:
        return _envelope(feature, entry, ttl, stale=False)

    with _feature_lock(feature):
        entry = _get_entry(feature)
        if entry and not force and time.time() - entry["stored_at"] < ttl:
            return _envelope(feature, entry, ttl, stale=False)
        _refresher_running.add(feature)
        try:
            data = _retry(refresher)
            _put_entry(feature, data)
        except tinyfish_service.TinyFishError as exc:
            last = _get_entry(feature)
            if last:
                payload = _envelope(feature, last, ttl, stale=True)
                payload["message"] = f"Refresh failed ({exc}); showing the last result."
                return payload
            return _unavailable(feature, str(exc), ttl)
        finally:
            _refresher_running.discard(feature)

    return _envelope(feature, _get_entry(feature), ttl, stale=False)


def _envelope(feature, entry, ttl, stale):
    payload = dict(entry["data"])
    age = max(0, int(time.time() - entry["stored_at"]))
    payload.update({
        "status": "stale" if stale else "ok",
        "stale": stale,
        "fetched_at": _now_iso(entry["stored_at"]),
        "age_seconds": age,
        "ttl_seconds": ttl,
    })
    return payload


def _unavailable(feature, error, ttl):
    return {
        "status": "unavailable",
        "stale": False,
        "fetched_at": None,
        "age_seconds": None,
        "ttl_seconds": ttl,
        "sources": [],
        "items": [],
        "message": f"Live context unavailable ({error}).",
        "feature": feature,
    }


# ── Background scheduler (stdlib thread — no new dependency) ───────────────
def refresh_warm_features():
    """Re-run every feature that has been served at least once AND whose TTL has
    elapsed — a 6 h payload is never refetched on the 20 min alerts cadence."""
    for feature in warm_features():
        spec = _REGISTRY.get(feature)
        if spec is None or feature in _refresher_running:
            continue
        entry = _get_entry(feature)
        if entry is None or time.time() - entry["stored_at"] < spec["ttl"]:
            continue                                   # not due yet
        with _feature_lock(feature):
            entry = _get_entry(feature)
            if entry is None or time.time() - entry["stored_at"] < spec["ttl"]:
                continue
            try:
                data = _retry(spec["refresh"])
                _put_entry(feature, data)
            except tinyfish_service.TinyFishError:
                pass  # keep the last good payload; it will be served as stale


def _scheduler_loop(interval):
    while True:
        time.sleep(interval)
        refresh_warm_features()


def start_scheduler(interval=REFRESH_SECONDS):
    """Start the background refresher once (idempotent, daemon thread)."""
    global _scheduler_started
    with _scheduler_state:
        if _scheduler_started:
            return False
        _scheduler_started = True
        thread = threading.Thread(
            target=_scheduler_loop, args=(interval,),
            name="tinyfish-context-scheduler", daemon=True,
        )
        thread.start()
    return True


# ── Severity / summary helpers ─────────────────────────────────────────────
def severity_of(title, summary):
    blob = f"{title} {summary}".lower()
    if any(marker in blob for marker in _HIGH_MARKERS):
        return "high"
    if any(marker in blob for marker in _MEDIUM_MARKERS):
        return "medium"
    return "low"


def summarize(text, limit=SUMMARY_CHARS):
    """First useful sentence(s) of a fetched page, ≤ `limit` chars, verbatim."""
    plain = sanitize(text, limit=1200)
    if not plain:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", plain)
    picked = ""
    for sentence in sentences:
        words = sentence.split()
        low = sentence.lower()
        if len(words) < 4:
            continue                                    # too short to inform
        if len(sentence) < 100 and any(m in low for m in _BOILERPLATE):
            continue                                    # pure boilerplate line
        candidate = f"{picked} {sentence}".strip()
        if len(candidate) > limit:
            break
        picked = candidate
        if len(picked) >= 120:
            break
    if not picked:
        picked = sentences[0] if sentences else plain
    return sanitize(picked, limit=limit)


def is_useful(title, summary):
    """Gate: a source must state something citable.

    Drops landing pages ("CAQM: Home") and page-chrome summaries (accessibility
    widgets, cookie banners) instead of showing them as an alert.
    """
    if _HOME_TITLE.match((title or "").strip()):
        return False
    if len(summary or "") < 25:
        return False
    low = summary.lower()
    return not any(marker in low for marker in _JUNK_MARKERS)


def parse_published(raw, fallback=None):
    """Best-effort publication timestamp (ISO) from TinyFish date strings.

    Handles absolute dates and relative "3 weeks ago" strings; returns None
    rather than guessing when neither is available.
    """
    for value in (raw, fallback):
        if not value:
            continue
        text = str(value).strip()
        match = re.search(r"\d{4}-\d{2}-\d{2}", text)
        if match:
            return match.group(0)
        match = re.search(r"\d{1,2}\s+[A-Z][a-z]{2,8}\s+\d{4}", text)   # 19 Dec 2025
        if match:
            try:
                return _dt.strptime(match.group(0), "%d %b %Y").date().isoformat()
            except ValueError:
                pass
        match = re.search(r"[A-Z][a-z]{2,8}\s+\d{1,2},?\s+\d{4}", text)  # Dec 19, 2025
        if match:
            stamp = match.group(0).replace(",", "")
            for fmt in ("%b %d %Y", "%B %d %Y"):
                try:
                    return _dt.strptime(stamp, fmt).date().isoformat()
                except ValueError:
                    continue
        match = re.match(r"(\d+)\s+(minute|hour|day|week|month|year)s?\s+ago", text, re.I)
        if match:
            units = {"minute": 60, "hour": 3600, "day": 86400,
                     "week": 604800, "month": 2592000, "year": 31536000}
            seconds = int(match.group(1)) * units[match.group(2).lower()]
            return _dt.fromtimestamp(time.time() - seconds).date().isoformat()
        if text.lower() == "today":
            return _dt.now().date().isoformat()
        if text.lower() == "yesterday":
            return _dt.fromtimestamp(time.time() - 86400).date().isoformat()
    return None


def _tokens(text):
    return set(re.findall(r"[a-z0-9]{3,}", (text or "").lower()))


def _near_duplicate(a, b):
    if a["url"].lower().rstrip("/") == b["url"].lower().rstrip("/"):
        return True
    ta, tb = _tokens(a.get("title")), _tokens(b.get("title"))
    if not ta or not tb:
        return False
    return len(ta & tb) / min(len(ta), len(tb)) >= 0.7


def dedupe(items):
    """Drop repeat coverage of the same story (overlap coefficient ≥ 0.7)."""
    kept = []
    for item in items:
        if any(_near_duplicate(item, existing) for existing in kept):
            continue
        kept.append(item)
    return kept


def _age_days(published):
    """Days since publication; unknown/undated counts as 0 (never punished)."""
    if not published:
        return 0
    try:
        return max(0, (_dt.now().date() - _date.fromisoformat(str(published)[:10])).days)
    except ValueError:
        return 0


def prefer_recent(items, max_age=FRESH_DAYS):
    """Prefer current advisories; keep older ones only when that is all there is.

    A three-year-old document is reference material, not an active alert — but
    dropping every result would leave the panel empty for no good reason, so the
    fallback keeps them (their `published_at` is always shown in the UI).
    """
    fresh = [item for item in items if _age_days(item.get("published_at")) <= max_age]
    return fresh or list(items)


def _alert_id(url):
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


# ── Feature 1: Active Alerts ───────────────────────────────────────────────
def refresh_alerts():
    """Search (official + reputable) → allowlist → dedupe → Fetch → JSON.

    Returns {"items": [...], "sources": [...]} or raises TinyFishError so the
    caller can fall back to the last good payload.
    """
    if not tinyfish_service.api_key():
        raise tinyfish_service.TinyFishError("TINYFISH_API_KEY is not set")

    candidates = []
    seen = set()
    searched_any = False
    errors = []
    for spec in ALERT_SEARCHES:
        try:
            results = _retry(lambda spec=spec: tinyfish_service.search(
                spec["query"],
                domain_type=spec.get("domain_type", "news"),
                page_size=8,
                include_domains=spec.get("include_domains"),
                purpose=ALERT_PURPOSE,
            ))
        except tinyfish_service.TinyFishError as exc:
            errors.append(str(exc))
            continue
        searched_any = True
        for result in results if isinstance(results, (list, tuple)) else []:
            if not isinstance(result, dict):
                continue                      # malformed entry — never trust it
            url = result.get("url", "")
            if not url or url in seen or not is_allowed(url):
                continue
            seen.add(url)
            candidates.append(result)

    if not searched_any:
        raise tinyfish_service.TinyFishError(errors[0] if errors else "search failed")

    # Official sources first, then reputable press, preserving search rank.
    candidates.sort(key=lambda item: 0 if is_official(item["url"]) else 1)
    if not candidates:
        raise tinyfish_service.TinyFishError("no results from allowlisted sources")

    fetched_at = _now_iso()
    picked = candidates[:ALERT_FETCHES]

    # ONE batched Fetch for every candidate page instead of one call each.
    # Job B runs every 45 min (spec §5); five separate fetches plus a retry
    # each was ~10 calls a cycle, which does not fit inside the daily cap.
    #
    # The response lists only the URLs that *succeeded*, so pages are joined by
    # URL, never by array position — position would attribute the second page
    # to the first URL the moment any single fetch failed.
    pages = {}
    try:
        batch = tinyfish_service.fetch_many(
            [r["url"] for r in picked], ttl=0,
            purpose=ALERT_PURPOSE)
        for page in batch.get("results") or []:
            if not isinstance(page, dict):
                continue
            original = str(page.get("url") or "").strip()
            final = str(page.get("final_url") or "").strip()
            if original:
                pages[original] = page
            if final:
                pages.setdefault(final, page)
    except tinyfish_service.TinyFishError:
        pages = {}      # snippets below still yield a citable summary

    items = []
    for result in picked:
        url = result["url"]
        source_name, official = source_for(
            url, result.get("publisher", ""), result.get("site_name", ""))
        page = pages.get(url)

        # Prefer the final URL, but never link to a host that is not itself on
        # the allowlist — a redirect must not smuggle another source past the UI.
        final_url = (page or {}).get("final_url") or (page or {}).get("url") or ""
        canonical = final_url if final_url and is_allowed(final_url) else url
        raw_text = (page or {}).get("text") or result.get("snippet") or ""
        summary = summarize(raw_text)
        title = sanitize(result.get("title", ""), limit=160)
        if not title or not is_useful(title, summary):
            continue                      # landing page / page chrome — skip
        published = parse_published(
            (page or {}).get("published_date"), result.get("date"))
        items.append({
            "id": _alert_id(canonical),
            "title": title,
            "severity": severity_of(title, summary),
            "summary": summary,                       # ≤ 200 chars
            "source": source_name,
            "url": canonical,
            "official": official,
            "published_at": published,
            "fetched_at": fetched_at,
        })

    if not items:
        raise tinyfish_service.TinyFishError("pages could not be extracted")

    items = prefer_recent(dedupe(items))[:MAX_ALERTS]
    # Highest severity first inside each badge group; official stays on top.
    rank = {"high": 0, "medium": 1, "low": 2}
    items.sort(key=lambda item: (0 if item["official"] else 1, rank[item["severity"]]))
    # Only sources behind the items that actually survived are advertised.
    sources = [
        {"name": item["source"], "url": item["url"], "fetched_at": item["fetched_at"]}
        for item in items
    ]
    return {"items": items, "sources": sources}


def get_alerts(force=False):
    """Active Alerts payload: official/press advisories with sources + TTL."""
    payload = _serve("alerts", force=force)
    payload["feature"] = "alerts"
    return payload


# ── Feature 2: Best Time to Go Outside ──────────────────────────────────────
WEATHER_TTL = 6 * 3600            # IMD refreshes its city forecast twice a day
MAX_NOTICES = 3                   # crop-burning / traffic notices per payload
NOTICE_FRESH_DAYS = 90            # an "active" notice older than this is not active
LOW_WIND_KMPH = 8                 # whole range ≤ this → poor dispersion
LADDER = ("Safe", "Moderate", "Avoid Outdoor")   # one-step downgrade ladder

WEATHER_SEARCH = {
    "domain_type": "web",
    "query": "Delhi weather forecast today wind speed humidity temperature fog rain",
    "include_domains": "mausam.imd.gov.in,imd.gov.in,safar.tropmet.res.in,dpcc.delhigovt.nic.in",
}
WEATHER_PURPOSE = ("Find today's official Delhi weather forecast (wind speed, humidity, "
                   "temperature, fog and rain) for the Vaayu Darshiya Best Time windows")

# Crop-burning and traffic/ban notices: reputable press plus official bans.
NOTICE_SEARCHES = [
    {"domain_type": "news",
     "query": "Delhi stubble burning crop residue traffic vehicle restrictions notice today"},
    {"domain_type": "web",
     "query": "Delhi stubble burning traffic restrictions ban notice",
     "include_domains": "caqm.nic.in,cpcb.nic.in,dpcc.delhigovt.nic.in,delhi.gov.in,pib.gov.in"},
]
NOTICE_PURPOSE = ("Find active Delhi crop-burning and traffic/vehicle restriction notices "
                  "for the Vaayu Darshiya Best Time windows")


def parse_weather(text):
    """Pull only the values the source actually states; unmatched stays None.

    Fetched text is untrusted: sanitized first, then read with fixed patterns —
    never executed and never treated as instructions. No value is ever guessed.

    Every numeric pattern guards its left edge with ``(?<![\\d.])`` so a decimal
    is never read as its tail: IMD's "9.3 km/h" must not become "3 km/h" and
    "29.8 °C" must not become 8 °C.
    """
    blob = sanitize(text, limit=4000)
    low = blob.lower()
    out = {
        "wind_min_kmph": None, "wind_max_kmph": None,
        "humidity_pct": None, "temp_c": None,
        "fog": False, "rain": False,
    }

    # Wind: a stated range first, then a single speed ("Calm 0 km/h" → 0).
    match = (
        re.search(
            r"(?<![\d.])(\d{1,3}(?:\.\d{1,2})?)\s*(?:-|–|—|to)\s*"
            r"(?<![\d.])(\d{1,3}(?:\.\d{1,2})?)\s*"
            r"(?:kmph|km/h|kilometres per hour|km per hour)", low)
        or re.search(
            r"(?<![\d.])(\d{1,3}(?:\.\d{1,2})?)\s*"
            r"(?:kmph|km/h|kilometres per hour|km per hour)", low))
    if match:
        numbers = [float(group) for group in match.groups() if group]
        out["wind_min_kmph"] = round(min(numbers), 1)
        out["wind_max_kmph"] = round(max(numbers), 1)

    # Temperature: keyword-anchored prose, then a bare "30.8 °C" / "30.4oC".
    match = (
        re.search(
            r"(?:temperature|temp\.?|maximum|max\.?|minimum|min\.?)\s*"
            r"(?:of|is|:)?\s*(-?\d{1,3}(?:\.\d+)?)\s*"
            r"(?:°\s*c|degrees?\s*c\b|c\b)", low)
        or re.search(r"(?<![\d.])(-?\d{1,3}(?:\.\d+)?)\s*℃", low)
        or re.search(r"(?<![\d.])(-?\d{1,3}(?:\.\d+)?)\s*(?:°|o|degrees?\s*)c\b", low))
    if match and -20 <= float(match.group(1)) <= 55:
        out["temp_c"] = round(float(match.group(1)), 1)

    # Humidity: keyword-anchored, else the first percentage on a page that
    # already yielded temperature *and* wind (i.e. a meteorological table such
    # as IMD's "30.8 °C … 9.3 km/h … 67 %"). Never on an unrelated page.
    match = re.search(r"(?:humidity|relative humidity|rh)[^0-9%]{0,30}"
                      r"(\d{1,3}(?:\.\d+)?)\s*%", low)
    if match is None and out["temp_c"] is not None and out["wind_max_kmph"] is not None:
        match = re.search(r"(?<![\d.])(\d{1,3}(?:\.\d+)?)\s*%", low)
    if match and 0 <= float(match.group(1)) <= 100:
        out["humidity_pct"] = int(round(float(match.group(1))))

    out["fog"] = bool(re.search(r"\bfog\b", low)) and not bool(
        re.search(r"\b(?:no|nil|free of|without)\s+fog\b", low))
    out["rain"] = bool(re.search(
        r"\b(?:rain|rainfall|raining|thundershowers?|thunderstorms?|showers?|drizzle)\b",
        low)) and not bool(re.search(r"\b(?:no|without|dry)\s+(?:rain|shower)\b", low))
    return out


def _num(value):
    """Readable number for reason strings: 9.3 → "9.3", 5.0 → "5"."""
    value = round(float(value), 1)
    return str(int(value)) if value == int(value) else str(value)


def wind_phrase(weather):
    """Human-readable wind line, or None when the source stated none."""
    if not weather:
        return None
    lo, hi = weather.get("wind_min_kmph"), weather.get("wind_max_kmph")
    if lo is None and hi is None:
        return None
    if lo is not None and hi is not None and lo != hi:
        return f"wind {_num(lo)}–{_num(hi)} km/h"
    return f"wind {_num(hi if hi is not None else lo)} km/h"


def notice_kind(title, summary):
    blob = f"{title} {summary}".lower()
    if any(word in blob for word in ("stubble", "crop residue", "crop burning", "paddy", "burning")):
        return "crop-burning"
    if any(word in blob for word in ("traffic", "vehicle", "truck", "congestion", "route", "ban on")):
        return "traffic"
    return "notice"


def _is_home_url(url):
    """Site landing page ("mausam.imd.gov.in/index_en.php") — not a forecast."""
    from urllib.parse import urlparse
    path = (urlparse(url).path or "/").lower().rstrip("/") or "/"
    name = path.rsplit("/", 1)[-1]
    return name.startswith("index") or path in ("/", "/home", "/home.html")


def weather_candidate_score(result):
    """Prefer a real Delhi forecast page over the site's homepage.

    The homepage mixes navigation junk and *realized* observations with the
    forecast; the city forecast page is the one that states today's values.
    """
    blob = f"{result.get('title', '')} {result.get('snippet', '')}".lower()
    score = 0
    if not _is_home_url(result.get("url", "")):
        score += 2
    if "forecast" in blob or "पूर्वानुमान" in blob:
        score += 1
    if "delhi" in blob or "ncr" in blob:
        score += 1
    return score


def _weather_candidates(results):
    """Allowed results, best first; ties keep the search engine's order."""
    scored = []
    for rank, result in enumerate(results if isinstance(results, (list, tuple)) else []):
        if not isinstance(result, dict):
            continue
        url = result.get("url", "")
        if not url or not is_allowed(url):
            continue
        scored.append((-weather_candidate_score(result), rank, result))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [result for _, _, result in scored]


def refresh_weather_windows():
    """Official day forecast + active crop-burning/traffic notices.

    Returns {"weather": {...}, "notices": [...], "sources": [...]} or raises
    TinyFishError so the caller can fall back to the last good payload.
    Weather is the payload's core: if no official forecast can be read, the
    whole refresh fails (and the previous one is served as stale). Notices are
    best-effort — their failure never discards a good forecast.
    """
    if not tinyfish_service.api_key():
        raise tinyfish_service.TinyFishError("TINYFISH_API_KEY is not set")

    fetched_at = _now_iso()
    results = _retry(lambda: tinyfish_service.search(
        WEATHER_SEARCH["query"],
        domain_type=WEATHER_SEARCH["domain_type"],
        page_size=5,
        include_domains=WEATHER_SEARCH["include_domains"],
        purpose=WEATHER_PURPOSE,
    ))
    candidates = _weather_candidates(results)
    if not candidates:
        raise tinyfish_service.TinyFishError("no official forecast result")

    # Try the best two pages max: one homepage-shaped miss costs one fetch.
    weather, sources, last_error = None, [], "forecast page could not be read"
    for best in candidates[:2]:
        page = None
        try:
            page = _retry(lambda url=best["url"]: tinyfish_service.fetch_text(url, limit=1500))
        except tinyfish_service.TinyFishError as exc:
            last_error = str(exc)
        raw = (page or {}).get("text") or best.get("snippet") or ""
        text = sanitize(raw, limit=320)
        # Days the source has not published yet come back as "NA °C NA %"—
        # they state nothing, so the display text stops before the first one.
        cut = text.find("NA °C")
        if 40 < cut:
            text = text[:cut].rstrip(" ,;—-")
        if not text or not is_useful(sanitize(best.get("title", ""), limit=160), text):
            continue
        final_url = (page or {}).get("url") or ""
        weather_url = final_url if final_url and is_allowed(final_url) else best["url"]
        source_name, _ = source_for(
            best["url"], best.get("publisher", ""), best.get("site_name", ""))
        weather = {
            "source": source_name,
            "url": weather_url,
            "fetched_at": fetched_at,
            "text": text,                    # the source's own words, verbatim
            **parse_weather(raw),            # only values the source stated
        }
        sources = [{"name": source_name, "url": weather_url, "fetched_at": fetched_at}]
        break
    if weather is None:
        raise tinyfish_service.TinyFishError(last_error)

    # ── Notices (best effort) ────────────────────────────────────────────
    notices, seen = [], set()
    for spec in NOTICE_SEARCHES:
        try:
            found = _retry(lambda spec=spec: tinyfish_service.search(
                spec["query"], domain_type=spec["domain_type"], page_size=6,
                include_domains=spec.get("include_domains"), purpose=NOTICE_PURPOSE,
            ))
        except tinyfish_service.TinyFishError:
            continue
        for result in (found if isinstance(found, (list, tuple)) else []):
            if not isinstance(result, dict):
                continue
            url = result.get("url", "")
            if not url or url in seen or not is_allowed(url):
                continue
            seen.add(url)
            notices.append(result)

    for result in notices[:MAX_NOTICES]:
        page = None
        try:
            page = _retry(lambda url=result["url"]: tinyfish_service.fetch_text(url, limit=900))
        except tinyfish_service.TinyFishError:
            page = None
        final_url = (page or {}).get("url") or ""
        notice_url = final_url if final_url and is_allowed(final_url) else result["url"]
        title = sanitize(result.get("title", ""), limit=160)
        summary = summarize((page or {}).get("text") or result.get("snippet") or "")
        if not is_useful(title, summary):
            continue
        source_name, _ = source_for(
            result["url"], result.get("publisher", ""), result.get("site_name", ""))
        result["_built"] = {
            "id": _alert_id(notice_url),
            "kind": notice_kind(title, summary),
            "title": title,
            "summary": summary,                    # ≤ 200 chars
            "source": source_name,
            "url": notice_url,
            "published_at": parse_published(
                (page or {}).get("published_date"), result.get("date")),
            "fetched_at": fetched_at,
        }
    built = [r.pop("_built") for r in notices[:MAX_NOTICES] if r.get("_built")]
    # "Active" notices must actually be recent: anything older than
    # NOTICE_FRESH_DAYS is background reading, not a live restriction.
    # Undated items are kept (the UI shows no published date rather than one).
    built = [item for item in built
             if not item["published_at"] or _age_days(item["published_at"]) <= NOTICE_FRESH_DAYS]
    notices = dedupe(built)[:MAX_NOTICES]
    sources.extend(
        {"name": n["source"], "url": n["url"], "fetched_at": n["fetched_at"]}
        for n in notices
    )
    return {"weather": weather, "notices": notices, "sources": sources}


def adjust_slots(slots, weather):
    """Apply the fog / low-wind rule to the ML windows.

    The model verdict always comes back beside the adjusted one: web context may
    only move a window one step down the ladder, only when the official forecast
    states fog or low wind, and always with the reason that was applied.
    """
    adjusted = []
    for slot in (slots if isinstance(slots, list) else []):
        if not isinstance(slot, dict):
            continue
        ml_status = str(slot.get("status") or "").strip()
        if ml_status not in LADDER:
            ml_status = "Safe"
        status, reasons = ml_status, []

        if weather and isinstance(weather, dict):
            triggers = []
            if weather.get("fog"):
                triggers.append("Fog forecast: pollutants linger in still air")
            wind_max = weather.get("wind_max_kmph")
            if isinstance(wind_max, (int, float)) and wind_max <= LOW_WIND_KMPH:
                triggers.append(f"Low wind: pollutants trapped ({wind_phrase(weather)})")
            if triggers:
                status = LADDER[min(len(LADDER) - 1, LADDER.index(ml_status) + 1)]
                reasons = triggers if status != ml_status else []

        adjusted.append({
            "period": str(slot.get("period") or ""),
            "hours": str(slot.get("hours") or ""),
            "ml_status": ml_status,
            "status": status,
            "adjusted": status != ml_status,
            "reasons": reasons,
        })
    return adjusted


def get_weather_windows(slots, force=False):
    """Best Time payload: cached forecast + notices + reasoned adjustments."""
    payload = _serve("weather_windows", force=force)
    payload["feature"] = "weather_windows"
    payload.setdefault("weather", None)
    payload.setdefault("notices", [])
    payload["slots"] = adjust_slots(slots, payload.get("weather"))
    return payload


register_feature("alerts", ALERTS_TTL, refresh_alerts)
register_feature("weather_windows", WEATHER_TTL, refresh_weather_windows)
