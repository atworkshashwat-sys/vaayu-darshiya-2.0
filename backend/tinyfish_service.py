"""
TinyFish client — the ONLY code path that talks to TinyFish.

Every TinyFish HTTP request in this project (Search, Fetch, Automation,
Monitor) goes through `search()`, `fetch_text()`, `automate()` or the
`monitors_*()` functions in this module. Nothing else opens a socket to a
TinyFish host, so the budget gate, the key failover and the call log can never
be bypassed.

Environment (backend only — never logged, never sent to the browser):
    TINYFISH_API_KEY_1 / _2 / _3   up to three keys; key_1 is used first
    TINYFISH_ENABLED               "true"/"false" — false blocks every call
    TINYFISH_EXPIRES_AT            ISO timestamp (IST) — the fixed window end
    TINYFISH_DAILY_CALL_CAP        hard cap on TinyFish calls per UTC day
    TINYFISH_API_KEY               legacy single-key name, honoured as key_1
                                   when the _1/_2/_3 names are all unset

Docs: https://docs.tinyfish.ai (search-api, fetch-api, monitor, error-codes)

Ground rules enforced here (see the spec):
  * a call only happens when enabled AND not expired AND under the cap —
    otherwise the caller's cache is served and no request is made
  * an auth / insufficient-balance error rotates to the next key and retries
    that call once; timeouts and 5xx NEVER rotate keys
  * exactly one retry per attempt, no retry loops
  * every call is logged with its type, key LABEL (key_1 …) and outcome; the
    key material itself is never written anywhere
"""

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from datetime import datetime, timedelta, timezone

SEARCH_ENDPOINT = "https://api.search.tinyfish.ai"
FETCH_ENDPOINT = "https://api.fetch.tinyfish.ai"
AGENT_ENDPOINT = "https://agent.tinyfish.ai"          # Monitors live here

# IST is fixed (no DST) — +05:30 all year.
IST = timezone(timedelta(hours=5, minutes=30))

# Tight timeouts: this runs inside request paths and must never hold a UI. A
# single retry happens inside the wrapper, so the worst case is 2x these.
SEARCH_TIMEOUT = 30.0
# TinyFish's own guidance: "Set your client-side timeout to at least 150
# seconds" for batch requests (110 s per-URL backend budget, 120 s CDN ceiling).
# The old 12 s cutoff was abandoning pages that were still loading fine — it is
# what produced the run of `timed out` errors, and each one wasted both a call
# and 12 s of a job.
FETCH_TIMEOUT = 90.0
FETCH_BATCH_TIMEOUT = 150.0        # batch: server may take up to 120 s CDN-wide
MONITOR_TIMEOUT = 12.0

# Excerpt of the top article, cleaned to plain-ish text for the panel.
EXCERPT_CHARS = 480

# ── Keys, window and cap ────────────────────────────────────────────────────
KEY_ENV_NAMES = ("TINYFISH_API_KEY_1", "TINYFISH_API_KEY_2", "TINYFISH_API_KEY_3")
LEGACY_KEY_ENV = "TINYFISH_API_KEY"
ENABLED_ENV = "TINYFISH_ENABLED"
EXPIRES_ENV = "TINYFISH_EXPIRES_AT"
CAP_ENV = "TINYFISH_DAILY_CALL_CAP"
DEFAULT_DAILY_CAP = 400            # shared across all keys
DEFAULT_WINDOW_HOURS = 24          # fixed window when TINYFISH_EXPIRES_AT is unset

# ── Error taxonomy (docs: /error-codes) ─────────────────────────────────────
# These HTTP statuses mean "this credential cannot pay for the call" → rotate.
KEY_ERROR_STATUSES = (401, 402, 403)
# 429 / 5xx / transport are transient → one retry on the SAME key, never a
# rotation (rotating on a rate limit would just burn the other keys too).
TRANSIENT_STATUSES = (429, 500, 502, 503, 504)

log = logging.getLogger("tinyfish")
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    log.addHandler(_handler)
    log.setLevel(logging.INFO)
log.propagate = False

_ENV_LOADED = False
_ENV_FILE = {}          # name -> value, parsed once from backend/.env


class TinyFishError(Exception):
    """Transport / HTTP / payload problem talking to TinyFish.

    `status` is the HTTP status (None for transport errors), `code` the
    TinyFish error code from the response body when one was present. The
    message keeps the historical `HTTP <status>` wording because callers
    (tinyfish_context._retry) classify on it.
    """

    def __init__(self, message, status=None, code=None, retry_after=None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.retry_after = retry_after

    @property
    def is_key_error(self):
        """True when another key might succeed: auth or insufficient balance."""
        if self.status is not None:
            return self.status in KEY_ERROR_STATUSES
        return any(f"HTTP {s}" in str(self) for s in KEY_ERROR_STATUSES)

    @property
    def is_transient(self):
        """True for timeouts / rate limits / 5xx — retried on the same key."""
        if self.status is not None:
            return self.status in TRANSIENT_STATUSES
        return not self.is_key_error


def _load_env_file():
    """Parse backend/.env (gitignored) once into `_ENV_FILE`.

    Values are kept in a module-level dict rather than injected into
    `os.environ`. Mutating os.environ was fragile: a caller running inside
    `mock.patch.dict(os.environ, …, clear=True)` would latch the loader while
    the environment was transiently empty, and the restored environment would
    then permanently lose the keys. Reading the file into our own dict is
    stable, and real environment variables still take precedence.

    Stdlib-only on purpose: the project carries no python-dotenv dependency.
    Keys are only ever read, never logged.
    """
    global _ENV_LOADED, _ENV_FILE
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, _, value = line.partition("=")
                name = name.strip()
                if name:
                    _ENV_FILE[name] = value.strip().strip('"').strip("'")
    except OSError:
        pass  # unreadable .env must never break startup


def env(name, default=""):
    """Real environment variable first, then backend/.env, then `default`.

    An explicitly-set-but-empty variable is treated as unset, so a test can
    blank a key by exporting "" without having to know about the .env file.
    """
    _load_env_file()
    value = os.environ.get(name)
    if value is not None and str(value).strip():
        return str(value).strip()
    if value is None:
        file_value = _ENV_FILE.get(name)
        if file_value and str(file_value).strip():
            return str(file_value).strip()
    return default


def _truthy(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


# ── Window (enabled / expires_at) ───────────────────────────────────────────
_window_start = time.time()      # when this process booted
_window_default_seconds = DEFAULT_WINDOW_HOURS * 3600


def expires_at():
    """End of the fixed live window as an aware datetime (IST).

    Falls back to `boot + 24 h` so the window is always bounded even when the
    operator forgets to set TINYFISH_EXPIRES_AT.
    """
    _load_env_file()
    raw = env(EXPIRES_ENV)
    if raw:
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            log.warning("TINYFISH_EXPIRES_AT is not ISO — falling back to a 24 h window")
            parsed = None
        if parsed is not None:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=IST)   # documented as IST
            return parsed
    return datetime.fromtimestamp(_window_start + _window_default_seconds, tz=IST)


def is_enabled():
    _load_env_file()
    return _truthy(env(ENABLED_ENV), default=True)


def is_expired(now=None):
    return (now or datetime.now(timezone.utc)) >= expires_at()


def is_active(now=None):
    """The master switch: enabled AND inside the window."""
    return is_enabled() and not is_expired(now)


def remaining_seconds(now=None):
    return max(0, int((expires_at() - (now or datetime.now(timezone.utc))).total_seconds()))


def daily_cap():
    _load_env_file()
    try:
        cap = int(env(CAP_ENV, str(DEFAULT_DAILY_CAP)))
    except (TypeError, ValueError):
        return DEFAULT_DAILY_CAP
    return cap if cap > 0 else DEFAULT_DAILY_CAP


# ── Keys ────────────────────────────────────────────────────────────────────
_preferred_label = None         # label of the key that last succeeded
_preferred_lock = threading.Lock()


def configured_keys():
    """[(label, key), …] for every key that is actually set, in failover order.

    The legacy TINYFISH_API_KEY name is used as key_1 only when none of the
    _1/_2/_3 names are set, so rotating deployments are never shadowed by a
    stale single key.
    """
    _load_env_file()
    found = []
    for index, name in enumerate(KEY_ENV_NAMES):
        value = env(name)
        if value:
            found.append((f"key_{index + 1}", value))
    if found:
        return found
    legacy = env(LEGACY_KEY_ENV)
    return [("key_1", legacy)] if legacy else []


def api_key():
    """The key that would be used first, or None when none is configured.

    Kept for the existing callers (tinyfish_context, ai_insights) that gate on
    "is a key set at all". Never logged.
    """
    keys = configured_keys()
    if not keys:
        return None
    with _preferred_lock:
        preferred = _preferred_label
    for label, value in keys:
        if label == preferred:
            return value
    return keys[0][1]


def _mark_preferred(label):
    """Remember which key worked so the next call starts there."""
    global _preferred_label
    with _preferred_lock:
        _preferred_label = label


def keys_available():
    return bool(configured_keys())


# ── Usage accounting ────────────────────────────────────────────────────────
USAGE_LOG_LIMIT = 500
RESET_UTC_DAY = True          # the TinyFish free allowance resets at 00:00 UTC

_usage_lock = threading.Lock()
_usage_day = None
_usage = {}                   # key label -> counters
_call_log = deque(maxlen=USAGE_LOG_LIMIT)
_monitor_ids = []             # Monitor ids this process created (deleted at expiry)
_monitor_ids_lock = threading.Lock()
_usage_sink = None            # optional persistence hook (set by live_store)
_usage_baseline = None        # optional durable total hook (set by live_store)


def set_usage_sink(fn):
    """Register `fn(event_dict)` to persist each call (live_store uses this)."""
    global _usage_sink
    _usage_sink = fn


def set_usage_baseline(fn):
    """Register `fn(day) -> int`: that reset day's calls from durable storage.

    `_usage` is in-memory and starts empty with every process, so a restart
    would otherwise reset the budget and hand out a second 400-call allowance
    for the same day. `calls_used()` takes the larger of the two numbers: the
    persisted total already includes this process's calls (each one is written
    through the sink), so the baseline is a floor, never an addition.

    `day` is passed in rather than read back through `usage_day()`. Both
    callers already hold `_usage_lock` — a plain, non-reentrant lock — so
    asking for the day again would deadlock: `/api/status` hung on exactly
    that, and because the lock is shared with every TinyFish call, the whole
    engine froze behind it.
    """
    global _usage_baseline
    _usage_baseline = fn


def _baseline_total(day):
    if _usage_baseline is None:
        return 0
    try:
        return max(0, int(_usage_baseline(day) or 0))
    except Exception:                        # pragma: no cover - storage is best effort
        return 0


def usage_day():
    """The TinyFish allowance window in force (YYYY-MM-DD, resets 00:00 UTC).

    For callers that do not already hold `_usage_lock`; everything inside the
    locked region must read `_usage_day` directly.
    """
    with _usage_lock:
        _roll_day_locked()
        return _usage_day


def _roll_day_locked():
    """Start a fresh window when the UTC day changes (allowance reset)."""
    global _usage_day
    day = datetime.now(timezone.utc).date().isoformat() if RESET_UTC_DAY \
        else datetime.now(IST).date().isoformat()
    if _usage_day != day:
        _usage_day = day
        _usage.clear()


def _slot(label):
    _roll_day_locked()
    return _usage.setdefault(label, {
        "search": 0, "fetch": 0, "monitor": 0, "automation": 0, "total": 0,
        "errors": 0, "failovers": 0,
        "last_success_at": None, "last_error_at": None, "last_error": None,
    })


def calls_used():
    with _usage_lock:
        _roll_day_locked()
        total = sum(slot["total"] for slot in _usage.values())
        # Never let a restart understate the day's spend (see set_usage_baseline).
        return max(total, _baseline_total(_usage_day))


def budget_remaining():
    return max(0, daily_cap() - calls_used())


def budget_reached():
    return budget_remaining() <= 0


def _record(call_type, label, ok, error=None, rotated=False):
    now_iso = datetime.now(IST).isoformat(timespec="seconds")
    with _usage_lock:
        slot = _slot(label)
        slot[call_type] = slot.get(call_type, 0) + 1
        slot["total"] += 1
        if ok:
            slot["last_success_at"] = now_iso
        else:
            slot["errors"] += 1
            slot["last_error_at"] = now_iso
            slot["last_error"] = str(error) if error else None
        if rotated:
            slot["failovers"] += 1
        entry = {
            "at": now_iso, "type": call_type, "key": label,
            "success": ok, "error": str(error) if error else None,
            "rotated": rotated,
            # The TinyFish reset day (00:00 UTC), so the persisted total can be
            # counted against the right allowance window and not the IST date.
            "day": _usage_day,
        }
        _call_log.append(entry)
        snapshot = dict(slot)
    log.info("tinyfish call type=%s key=%s ok=%s%s%s",
             call_type, label, ok,
             "" if not rotated else " rotated",
             "" if ok else f" error={error}")
    if _usage_sink is not None:
        try:
            _usage_sink(entry)
        except Exception:                        # pragma: no cover
            log.debug("usage sink failed", exc_info=True)
    return entry


def usage_snapshot():
    """Totals per key + the cap — served by /api/admin/tinyfish-usage.

    Keys appear only as key_1 / key_2 / key_3; no key material is included.
    """
    with _usage_lock:
        _roll_day_locked()
        per_key = []
        configured = {label for label, _ in configured_keys()}
        for label in sorted(set(configured) | set(_usage)):
            slot = dict(_usage.get(label, {}))
            per_key.append({
                "key": label,
                "configured": label in configured,
                "active": label == active_key_label(),
                "search": slot.get("search", 0),
                "fetch": slot.get("fetch", 0),
                "monitor": slot.get("monitor", 0),
                "automation": slot.get("automation", 0),
                "total": slot.get("total", 0),
                "errors": slot.get("errors", 0),
                "failovers": slot.get("failovers", 0),
                "last_success_at": slot.get("last_success_at"),
                "last_error_at": slot.get("last_error_at"),
                "last_error": slot.get("last_error"),
            })
        # The same floor as calls_used(): without it, /api/status would show a
        # freshly restarted process as having spent nothing today.
        used = max(sum(s["total"] for s in _usage.values()),
                   _baseline_total(_usage_day))
        cap = daily_cap()
        log_tail = list(_call_log)[-50:]
        day = _usage_day
    return {
        "enabled": is_enabled(),
        "expired": is_expired(),
        "expires_at": expires_at().isoformat(timespec="seconds"),
        "remaining_seconds": remaining_seconds(),
        "cap": cap,
        "used": used,
        "remaining": max(0, cap - used),
        "day": day,
        "window": "utc_day" if RESET_UTC_DAY else "ist_day",
        "keys": per_key,
        "active_key": active_key_label(),
        "calls": log_tail,
        "monitors": list(_monitor_ids),
        "monitors_deleted_at": _monitors_deleted_at,
    }


def active_key_label():
    keys = configured_keys()
    if not keys:
        return None
    with _preferred_lock:
        preferred = _preferred_label
    for label, _value in keys:
        if label == preferred:
            return label
    return keys[0][0]


def call_log(limit=50):
    with _usage_lock:
        return list(_call_log)[-limit:]


def reset_usage():
    """Clear counters and the log (tests / manual reset)."""
    global _usage_day, _monitors_deleted_at, _preferred_label
    with _usage_lock:
        _usage.clear()
        _call_log.clear()
        _usage_day = None
    with _monitor_ids_lock:
        _monitor_ids.clear()
    with _preferred_lock:
        _preferred_label = None
    _monitors_deleted_at = None


def reset_window_default():
    """Re-anchor the default 24 h window at 'now' (tests)."""
    global _window_start
    _window_start = time.time()


# ── Transport ───────────────────────────────────────────────────────────────
def _site_name(url):
    netloc = urllib.parse.urlparse(str(url)).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


def _error_code_from(body):
    """TinyFish error code from a response body, or None."""
    if not body:
        return None
    try:
        parsed = json.loads(body)
    except ValueError:
        return None
    error = parsed.get("error") if isinstance(parsed, dict) else None
    if isinstance(error, dict):
        return error.get("code")
    return None


def _call(url, data=None, timeout=SEARCH_TIMEOUT, key=None, call_type="search"):
    """One HTTP call to TinyFish with an explicit key. Never rotates or retries."""
    headers = {"X-API-Key": key or "", "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url, data=data, headers=headers, method="POST" if data is not None else "GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", "replace")
        except Exception:
            body = ""
        retry_after = exc.headers.get("Retry-After") if exc.headers else None
        try:
            retry_after = int(retry_after) if retry_after else None
        except (TypeError, ValueError):
            retry_after = None
        raise TinyFishError(f"HTTP {exc.code}", status=exc.code,
                            code=_error_code_from(body),
                            retry_after=retry_after) from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, TimeoutError) or "timed out" in str(reason).lower():
            raise TinyFishError("timed out") from exc
        raise TinyFishError(str(reason) or exc.__class__.__name__) from exc
    except (TimeoutError, OSError) as exc:
        raise TinyFishError("timed out" if isinstance(exc, TimeoutError)
                            else (str(exc) or exc.__class__.__name__)) from exc

    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise TinyFishError("TinyFish returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise TinyFishError("Unexpected TinyFish payload")
    return payload


def _request(build, call_type, timeout=SEARCH_TIMEOUT):
    """Gate → attempt → (rotate on key error | retry once on transient).

    `build(key)` performs one attempt and returns the parsed payload. This is
    the single choke point: it applies the enabled/expiry/cap gate, walks the
    key chain, logs every call and enforces "one retry, no retry loops".

    Returns (payload, label_of_the_key_that_succeeded).
    """
    if not is_enabled():
        raise TinyFishError("TinyFish is disabled (TINYFISH_ENABLED=false)")
    if is_expired():
        raise TinyFishError(f"Live context window ended at "
                            f"{expires_at().strftime('%H:%M IST')}")

    keys = configured_keys()
    if not keys:
        raise TinyFishError("TINYFISH_API_KEY is not set")

    if budget_reached():
        raise TinyFishError(
            f"TinyFish daily call cap reached ({daily_cap()}); serving cache")

    # Walk the chain starting at the key that last worked, so a rotation does
    # not permanently demote a recovered key_1 on the next call.
    with _preferred_lock:
        preferred = _preferred_label
    start = 0
    for offset, (label, _value) in enumerate(keys):
        if label == preferred:
            start = offset
            break
    chain = keys[start:] + keys[:start]

    last_error = None
    for offset, (label, key) in enumerate(chain):
        try:
            payload = build(key)
        except TinyFishError as exc:
            last_error = exc
            if exc.is_transient:
                # One retry, SAME key. Never a loop, never a rotation — a
                # rate limit or a 5xx says nothing about the credential.
                _record(call_type, label, ok=False, error=exc)
                time.sleep(1.0 if exc.status == 429 else 0.4)
                try:
                    payload = build(key)          # the single retry
                except TinyFishError as retry_exc:
                    _record(call_type, label, ok=False, error=retry_exc)
                    raise
                # The retry worked — this key is fine, use the result.
                _mark_preferred(label)
                _record(call_type, label, ok=True)
                return payload, label
            if exc.is_key_error:
                # Auth / insufficient balance → the next key may be able to pay.
                rotated = offset + 1 < len(chain)
                _record(call_type, label, ok=False, error=exc, rotated=rotated)
                if rotated:
                    log.warning("tinyfish key failover %s → %s (%s)",
                                label, chain[offset + 1][0], exc)
                continue
            _record(call_type, label, ok=False, error=exc)
            raise

        _mark_preferred(label)
        _record(call_type, label, ok=True)
        return payload, label

    # Every key in the chain was rejected for auth/balance reasons.
    raise last_error if last_error else TinyFishError("all TinyFish keys failed")


# ── Search ──────────────────────────────────────────────────────────────────
def search(query, domain_type="news", page_size=5, timeout=SEARCH_TIMEOUT,
           include_domains=None, purpose=None, location="IN", language="en",
           recency_minutes=None):
    """Ranked TinyFish Search results for `query`.

    Returns a list of {title, url, snippet, site_name, publisher, date}.
    Raises TinyFishError when no key works, the cap is hit or the window ended.
    """
    params = {
        "query": query,
        "domain_type": domain_type,
        "location": location,
        "language": language,
        "purpose": purpose or (
            "Delhi NCR pollution news and official air quality advisories "
            "for the Vaayu Darshiya live dashboard"
        ),
    }
    if include_domains:
        params["include_domains"] = include_domains
    if recency_minutes:
        params["recency_minutes"] = int(recency_minutes)
    url = f"{SEARCH_ENDPOINT}?{urllib.parse.urlencode(params)}"

    payload, _label = _request(lambda key: _call(url, timeout=timeout, key=key),
                               "search", timeout=timeout)

    items = []
    for raw in payload.get("results") or []:
        if not isinstance(raw, dict):
            continue
        url_, title = raw.get("url"), raw.get("title")
        if not url_ or not title:
            continue  # a result without a link cannot be cited
        items.append({
            "title": str(title),
            "url": str(url_),
            "snippet": str(raw.get("snippet") or "").strip(),
            "site_name": str(raw.get("site_name") or _site_name(url_)),
            "publisher": str(raw.get("publisher") or ""),
            "date": str(raw.get("date") or ""),
        })
        if len(items) >= page_size:
            break
    return items


# ── Fetch ───────────────────────────────────────────────────────────────────
def _clean_excerpt(text, limit=EXCERPT_CHARS):
    """Markdown page text → a short plain excerpt (never invents wording)."""
    import re
    if not text:
        return ""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)   # links → label
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]  # drop headings
    joined = re.sub(r"\s+", " ", " ".join(lines)).strip()
    joined = re.sub(r"\*\*([^*]+)\*\*", r"\1", joined)
    joined = re.sub(r"\*([^*]+)\*", r"\1", joined)
    if len(joined) <= limit:
        return joined
    return joined[:limit].rsplit(" ", 1)[0] + " …"


def excerpt(text, limit=EXCERPT_CHARS):
    """`_clean_excerpt` for callers that fetch a *batch* instead of one page.

    `fetch_text()` shapes every page this way before returning it, so a
    caller holding raw `fetch_many()` markdown has to apply the same shaping
    itself — otherwise the parsers see different text than a single fetch
    would have handed them.
    """
    return _clean_excerpt(str(text or ""), limit)


def fetch_text(url, timeout=FETCH_TIMEOUT, limit=EXCERPT_CHARS, ttl=None,
               purpose=None, raw_limit=None):
    """Clean extracted text for one URL via TinyFish Fetch.

    ttl=0 forces a live fetch (what the live-reading jobs need so a cached
    page is never mistaken for a current reading).

    Returns {url, title, site_name, published_date, text} or raises
    TinyFishError. Per-URL failures arrive in `errors[]` on a 200 response and
    are converted into a TinyFishError so callers can keep their cache.
    """
    body = {
        "urls": [str(url)],
        "format": "markdown",
        "purpose": purpose or (
            "Extract the article text so the dashboard can quote a short, "
            "faithful excerpt beside the headline"
        ),
    }
    if ttl is not None:
        body["ttl"] = int(ttl)

    payload, _label = _request(
        lambda key: _call(FETCH_ENDPOINT, data=json.dumps(body).encode("utf-8"),
                          timeout=timeout, key=key, call_type="fetch"),
        "fetch", timeout=timeout)

    results = payload.get("results") or []
    if not results or not isinstance(results[0], dict):
        errors = [e for e in (payload.get("errors") or []) if isinstance(e, dict)]
        detail = "; ".join(str(e.get("error")) for e in errors if e.get("error"))
        raise TinyFishError(f"Fetch returned no content{': ' + detail if detail else ''}")

    first = results[0]
    final_url = str(first.get("final_url") or first.get("url") or url)
    text = _clean_excerpt(str(first.get("text") or ""), raw_limit or limit)
    return {
        "url": final_url,
        "title": str(first.get("title") or ""),
        "site_name": _site_name(final_url),
        # Raw publication date as the source states it ("" when unstated).
        "published_date": str(first.get("published_date") or ""),
        "text": text,
    }


def fetch_many(urls, timeout=FETCH_BATCH_TIMEOUT, ttl=None, purpose=None, format_="markdown"):
    """Fetch up to 10 URLs in ONE call (the Fetch API's batch limit).

    Returns {"results": [...], "errors": [...]}. One HTTP request = one call
    against the daily cap, which is why the live-reading job uses this instead
    of 7 separate fetch_text() calls.
    """
    body = {"urls": [str(u) for u in urls][:10], "format": format_}
    if ttl is not None:
        body["ttl"] = int(ttl)
    if purpose:
        body["purpose"] = str(purpose)[:2000]
    payload, _label = _request(
        lambda key: _call(FETCH_ENDPOINT, data=json.dumps(body).encode("utf-8"),
                          timeout=timeout, key=key, call_type="fetch"),
        "fetch", timeout=timeout)
    return {
        "results": [r for r in (payload.get("results") or []) if isinstance(r, dict)],
        "errors": [e for e in (payload.get("errors") or []) if isinstance(e, dict)],
    }


# ── Web Automation (Agent API) ──────────────────────────────────────────────
# Reserved for pages that genuinely need JavaScript. CPCB's real-time station
# dashboard is a client-rendered shell — its HTML carries no station names — so
# Search/Fetch cannot read it and the spec's fallback applies: "use web
# automation only if the page needs JavaScript". Every run is one call against
# the same cap, gate and log as Search and Fetch.
AUTOMATE_ENDPOINT = f"{AGENT_ENDPOINT}/v1/automation/run"
# A browser run is slower than a fetch, and the HTTP timeout must outlast
# `max_duration_seconds` or the request is cut off while the run is still live.
AUTOMATE_TIMEOUT = 180.0


def automate(url, goal, output_schema=None, timeout=AUTOMATE_TIMEOUT,
             max_duration_seconds=120, browser_profile="lite", purpose=None):
    """Run one goal-directed browser automation and return its `result`.

    `output_schema` is a JSON Schema (object root) so the agent returns strict
    JSON instead of prose — see the Agent API reference for the supported
    keyword allowlist. The returned dict is untrusted like everything else
    from the web: the caller validates every field before storing it.
    """
    body = {
        "url": str(url),
        "goal": str(goal)[:4000],
        "browser_profile": browser_profile,
        "agent_config": {"max_duration_seconds": int(max_duration_seconds)},
    }
    if output_schema is not None:
        body["output_schema"] = output_schema
    if purpose:
        body["purpose"] = str(purpose)[:2000]

    payload, _label = _request(
        lambda key: _call(AUTOMATE_ENDPOINT,
                          data=json.dumps(body).encode("utf-8"),
                          timeout=timeout, key=key, call_type="automation"),
        "automation", timeout=timeout)

    status = str(payload.get("status") or "").upper()
    if status and status != "COMPLETED":
        error = payload.get("error")
        detail = error.get("message") if isinstance(error, dict) else str(error or "")
        raise TinyFishError(f"automation {status}{': ' + detail if detail else ''}")

    result = payload.get("result")
    if not isinstance(result, dict):
        raise TinyFishError("automation returned no structured result")
    return result


# ── Monitors (optional early trigger for the alerts job) ────────────────────
_monitors_deleted_at = None


def _agent_call(method, path, body=None, call_type="monitor"):
    """One call to the agent host (Monitors), through the same gate."""
    url = f"{AGENT_ENDPOINT}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None

    def attempt(key):
        return _call(url, data=data, timeout=MONITOR_TIMEOUT, key=key,
                     call_type=call_type)

    payload, _label = _request(attempt, call_type, timeout=MONITOR_TIMEOUT)
    return payload


def create_monitor(name, config, schedule_cron, purpose=None, monitor_type="fetch"):
    """Create a TinyFish Monitor and remember its id so expiry can delete it."""
    payload = _agent_call("POST", "/v1/monitors", {
        "type": monitor_type,
        "name": str(name)[:100],
        "config": config,
        "schedule_cron": schedule_cron,
        **({"purpose": str(purpose)[:2000]} if purpose else {}),
    })
    monitor = payload.get("monitor") if isinstance(payload, dict) else None
    if isinstance(monitor, dict) and monitor.get("id"):
        with _monitor_ids_lock:
            _monitor_ids.append(str(monitor["id"]))
        return monitor
    raise TinyFishError("TinyFish returned no monitor id")


def list_monitors():
    payload = _agent_call("GET", "/v1/monitors")
    monitors = payload.get("monitors") if isinstance(payload, dict) else []
    return monitors if isinstance(monitors, list) else []


def delete_monitor(monitor_id):
    """DELETE one Monitor. Idempotent: a 404 is treated as already deleted."""
    try:
        _agent_call("DELETE", f"/v1/monitors/{urllib.parse.quote(str(monitor_id))}")
        return True
    except TinyFishError as exc:
        if exc.status == 404:
            return True
        log.warning("could not delete monitor %s: %s", monitor_id, exc)
        return False


def delete_all_monitors():
    """Delete every Monitor this process created (spec: clean up at expiry).

    Also sweeps any Monitor still listed by the account so a crash between
    create and expiry cannot leave an orphan polling forever.
    """
    global _monitors_deleted_at
    ids = []
    with _monitor_ids_lock:
        ids.extend(_monitor_ids)
        _monitor_ids.clear()
    try:
        ids.extend(str(m.get("id")) for m in list_monitors() if m.get("id"))
    except TinyFishError:
        pass
    deleted = 0
    for monitor_id in dict.fromkeys(ids):        # de-dupe, keep order
        if delete_monitor(monitor_id):
            deleted += 1
    _monitors_deleted_at = datetime.now(IST).isoformat(timespec="seconds")
    log.info("tinyfish monitors deleted: %d", deleted)
    return deleted
