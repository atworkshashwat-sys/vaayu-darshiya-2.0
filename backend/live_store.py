"""
Live-context store — a small SQLite file for everything TinyFish retrieves.

Design goals (spec §4):
  * small and local: one file, no DB server, no ORM
  * every row carries source / url / observed_at / fetched_at / status so a
    number on screen can always be traced back to where it came from
  * at least 7 days of history so the trend charts keep working and the ML
    forecast can be validated against the live readings later
  * the TinyFish call log is persisted here too, so the usage report survives a
    restart

Tables
    readings     station pollutant values (job A)
    alerts       official + press advisories (job B)
    weather      Delhi conditions (job C)
    case_stats   Delhi's Drastic Case figures (job D)
    job_state    last success / last error per job (drives /api/status)
    usage_log    every TinyFish call: type, key LABEL, outcome

Nothing in here talks to TinyFish or to the network — it is pure storage, so
it is trivially testable and cannot blow the daily call cap.
"""

import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")

# Overridable so tests never touch the real file.
DB_PATH = os.environ.get("VAAYU_LIVE_DB") or os.path.join(DATA_DIR, "live_context.sqlite")

HISTORY_DAYS = 7                      # spec: keep ≥ 7 days of history

_conn = None
_conn_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id   TEXT    NOT NULL,      -- "1".."7" (frontend site id)
    station_name TEXT    NOT NULL,
    pollutant    TEXT    NOT NULL,      -- NO2 | O3 | PM25 | AQI
    value        REAL    NOT NULL,
    unit         TEXT    NOT NULL,      -- ppb | ug/m3 | index
    observed_at  TEXT    NOT NULL,      -- ISO 8601, always IST
    source       TEXT    NOT NULL,      -- "CPCB", "DPCC", ...
    url          TEXT,
    fetched_at   TEXT    NOT NULL,
    status       TEXT    NOT NULL,      -- live | stale | offline
    conflict     INTEGER NOT NULL DEFAULT 0,   -- 1 when sources disagreed >30%
    UNIQUE(station_id, pollutant, observed_at, source)
);
CREATE INDEX IF NOT EXISTS idx_readings_station ON readings(station_id, pollutant);
CREATE INDEX IF NOT EXISTS idx_readings_observed ON readings(observed_at);

CREATE TABLE IF NOT EXISTS alerts (
    id           TEXT    PRIMARY KEY,   -- sha1 of the canonical url
    title        TEXT    NOT NULL,
    summary      TEXT    NOT NULL,
    severity     TEXT    NOT NULL,      -- high | medium | low
    source       TEXT    NOT NULL,
    url          TEXT    NOT NULL,
    official     INTEGER NOT NULL,
    kind         TEXT,                  -- grap | weather | notice | news
    published_at TEXT,
    fetched_at   TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alerts_fetched ON alerts(fetched_at);

CREATE TABLE IF NOT EXISTS weather (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    city           TEXT    NOT NULL DEFAULT 'Delhi',
    temp_c         REAL,
    humidity_pct   INTEGER,
    wind_min_kmph  REAL,
    wind_max_kmph  REAL,
    visibility_km  REAL,
    fog            INTEGER NOT NULL DEFAULT 0,
    rain           INTEGER NOT NULL DEFAULT 0,
    text           TEXT,
    source         TEXT    NOT NULL,
    url            TEXT,
    observed_at    TEXT,
    fetched_at     TEXT    NOT NULL,
    status         TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_weather_fetched ON weather(fetched_at);

CREATE TABLE IF NOT EXISTS case_stats (
    stat_key   TEXT    PRIMARY KEY,     -- annual_pm25 | winter_aqi | news_strip
    label      TEXT    NOT NULL,
    value      TEXT    NOT NULL,        -- already formatted for display
    unit       TEXT,
    as_of      TEXT,                    -- the figure's own "as of" date
    source     TEXT    NOT NULL,
    url        TEXT,
    fetched_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS job_state (
    job              TEXT PRIMARY KEY,
    last_success_at  TEXT,
    last_attempt_at  TEXT,
    last_error       TEXT,
    status           TEXT,              -- ok | error | never_run
    detail           TEXT               -- JSON: counts, exclusions, ...
);

CREATE TABLE IF NOT EXISTS usage_log (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    at       TEXT    NOT NULL,
    type     TEXT    NOT NULL,          -- search | fetch | monitor | automation
    key      TEXT    NOT NULL,          -- key_1 | key_2 | key_3
    success  INTEGER NOT NULL,
    error    TEXT,
    rotated  INTEGER NOT NULL DEFAULT 0,
    day      TEXT                       -- TinyFish reset day (00:00 UTC)
);
CREATE INDEX IF NOT EXISTS idx_usage_at ON usage_log(at);
"""

# `day` was added after the first tables were created in the field, so an
# existing file is upgraded in place rather than being thrown away (losing the
# usage history would lose the only record of calls already paid for).
MIGRATIONS = (
    "ALTER TABLE usage_log ADD COLUMN day TEXT",
)


def now_ist():
    return datetime.now(IST)


def iso(dt):
    """ISO 8601 with the IST offset, e.g. 2026-10-10T18:04:00+05:30."""
    if dt is None:
        return None
    if isinstance(dt, str):
        return dt
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)
    return dt.astimezone(IST).isoformat(timespec="seconds")


def parse_iso(value):
    """ISO string → aware datetime, or None when unparseable (never guessed)."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=IST)
    return parsed.astimezone(IST)


def connect():
    """One shared connection; SQLite is used behind a lock (stdlib default)."""
    global _conn
    with _conn_lock:
        if _conn is not None:
            return _conn
        os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
        # `isolation_level=None` puts SQLite in autocommit, so no implicit
        # transaction can be left open by an abandoned cursor. This matters
        # because the connection is shared across threads: a write left open
        # by the scheduler would otherwise block every reader - including
        # /api/status - for as long as it stayed open, with no error to
        # explain the hang.
        #
        # `busy_timeout` turns any residual contention into a fast, reportable
        # "database is locked" instead of an indefinite stall. A monitoring
        # endpoint that hangs is worse than one that says it is busy.
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False,
                                isolation_level=None, timeout=5.0)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA synchronous=NORMAL")
        _conn.execute("PRAGMA busy_timeout=5000")
        _conn.executescript(_SCHEMA)
        for statement in MIGRATIONS:
            try:
                _conn.execute(statement)
            except sqlite3.OperationalError as exc:
                # Already applied ("duplicate column name") — anything else is
                # a real failure and should surface.
                if "duplicate column" not in str(exc).lower():
                    raise
        _conn.commit()
        return _conn


def reset(path=None):
    """Point the store at another file and reopen (tests)."""
    global _conn, DB_PATH
    with _conn_lock:
        if _conn is not None:
            try:
                _conn.close()
            except sqlite3.Error:
                pass
            _conn = None
        if path is not None:
            DB_PATH = path
    return connect()


def close():
    global _conn
    with _conn_lock:
        if _conn is not None:
            try:
                _conn.close()
            except sqlite3.Error:
                pass
            _conn = None


# ── Housekeeping ────────────────────────────────────────────────────────────
def prune(keep_days=HISTORY_DAYS):
    """Drop rows older than `keep_days` so the file stays small."""
    cutoff = iso(now_ist() - timedelta(days=keep_days))
    conn = connect()
    with _conn_lock:
        for table, column in (("readings", "fetched_at"), ("alerts", "fetched_at"),
                              ("weather", "fetched_at"), ("usage_log", "at")):
            conn.execute(f"DELETE FROM {table} WHERE {column} < ?", (cutoff,))
        conn.commit()


# ── Readings (job A) ────────────────────────────────────────────────────────
def save_readings(rows):
    """Insert validated readings. `rows` are dicts in the extraction schema:
    {station_id, station_name, pollutant, value, unit, observed_at, source,
     url, fetched_at, status, conflict}.
    """
    if not rows:
        return 0
    conn = connect()
    with _conn_lock:
        conn.executemany(
            """INSERT OR REPLACE INTO readings
               (station_id, station_name, pollutant, value, unit, observed_at,
                source, url, fetched_at, status, conflict)
               VALUES (:station_id, :station_name, :pollutant, :value, :unit,
                       :observed_at, :source, :url, :fetched_at, :status,
                       :conflict)""",
            rows)
        conn.commit()
    return len(rows)


def latest_readings():
    """Newest row per (station_id, pollutant), regardless of source.

    The source of the winning row travels with the value, so the UI can say
    "CPCB" next to a number instead of an anonymous figure.
    """
    conn = connect()
    with _conn_lock:
        rows = conn.execute(
            """SELECT r.* FROM readings r
               JOIN (SELECT station_id, pollutant, MAX(observed_at) AS mx
                     FROM readings GROUP BY station_id, pollutant) m
                 ON  r.station_id = m.station_id
                 AND r.pollutant  = m.pollutant
                 AND r.observed_at = m.mx"""
        ).fetchall()
    return [dict(row) for row in rows]


def readings_history(station_id, pollutant, days=HISTORY_DAYS):
    """Time-ordered readings for one station+pollutant (trend charts)."""
    since = iso(now_ist() - timedelta(days=days))
    conn = connect()
    with _conn_lock:
        rows = conn.execute(
            """SELECT value, unit, observed_at, source, status
               FROM readings
               WHERE station_id = ? AND pollutant = ? AND observed_at >= ?
               ORDER BY observed_at""",
            (str(station_id), pollutant, since)).fetchall()
    return [dict(row) for row in rows]


def conflict_rows():
    """Readings flagged as a >30% disagreement between two sources."""
    conn = connect()
    with _conn_lock:
        rows = conn.execute(
            "SELECT * FROM readings WHERE conflict = 1 ORDER BY observed_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


# ── Alerts (job B) ──────────────────────────────────────────────────────────
def save_alerts(items):
    if not items:
        return 0
    conn = connect()
    with _conn_lock:
        conn.executemany(
            """INSERT OR REPLACE INTO alerts
               (id, title, summary, severity, source, url, official, kind,
                published_at, fetched_at)
               VALUES (:id, :title, :summary, :severity, :source, :url,
                       :official, :kind, :published_at, :fetched_at)""",
            items)
        conn.commit()
    return len(items)


def latest_alerts(limit=8):
    """Newest alerts, official ones first, then by severity."""
    conn = connect()
    with _conn_lock:
        rows = conn.execute(
            """SELECT * FROM alerts ORDER BY fetched_at DESC, official DESC LIMIT ?""",
            (limit,)).fetchall()
    items = [dict(row) for row in rows]
    rank = {"high": 0, "medium": 1, "low": 2}
    items.sort(key=lambda a: (0 if a["official"] else 1, rank.get(a["severity"], 3)))
    return items


def seen_alert_ids():
    conn = connect()
    with _conn_lock:
        rows = conn.execute("SELECT id FROM alerts").fetchall()
    return {row["id"] for row in rows}


# ── Weather (job C) ─────────────────────────────────────────────────────────
def save_weather(row):
    conn = connect()
    with _conn_lock:
        conn.execute(
            """INSERT INTO weather
               (city, temp_c, humidity_pct, wind_min_kmph, wind_max_kmph,
                visibility_km, fog, rain, text, source, url, observed_at,
                fetched_at, status)
               VALUES (:city, :temp_c, :humidity_pct, :wind_min_kmph,
                       :wind_max_kmph, :visibility_km, :fog, :rain, :text,
                       :source, :url, :observed_at, :fetched_at, :status)""",
            row)
        conn.commit()
    return row


def latest_weather():
    conn = connect()
    with _conn_lock:
        row = conn.execute(
            "SELECT * FROM weather ORDER BY fetched_at DESC LIMIT 1").fetchone()
    return dict(row) if row else None


# ── Case-study stats (job D) ────────────────────────────────────────────────
def save_case_stats(items, judged=None):
    """Write this refresh's figures, dropping the ones it disproved.

    `judged` is the set of stat keys the refresh actually searched for. A key
    that was judged but produced no row is a figure the sources no longer
    state, so its stored row is deleted — leaving it would serve last week's
    number indefinitely. A key whose search failed was never judged and keeps
    its previous row: a transient failure must not erase a figure that was
    correct yesterday.
    """
    conn = connect()
    with _conn_lock:
        if items:
            conn.executemany(
                """INSERT OR REPLACE INTO case_stats
                   (stat_key, label, value, unit, as_of, source, url, fetched_at)
                   VALUES (:stat_key, :label, :value, :unit, :as_of, :source,
                           :url, :fetched_at)""",
                items)
        if judged:
            kept = {item["stat_key"] for item in items}
            stale = [key for key in judged if key not in kept]
            if stale:
                conn.executemany("DELETE FROM case_stats WHERE stat_key = ?",
                                 [(key,) for key in stale])
        conn.commit()
    return len(items)


def case_stats():
    conn = connect()
    with _conn_lock:
        rows = conn.execute("SELECT * FROM case_stats ORDER BY stat_key").fetchall()
    return [dict(row) for row in rows]


# ── Job state (drives /api/status and /health) ─────────────────────────────
def record_job(job, status, error=None, detail=None):
    import json as _json
    conn = connect()
    at = iso(now_ist())
    with _conn_lock:
        conn.execute(
            """INSERT INTO job_state
               (job, last_success_at, last_attempt_at, last_error, status, detail)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(job) DO UPDATE SET
                 last_success_at = CASE WHEN ? = 'ok' THEN excluded.last_success_at
                                        ELSE job_state.last_success_at END,
                 last_attempt_at = excluded.last_attempt_at,
                 last_error      = excluded.last_error,
                 status          = excluded.status,
                 detail          = excluded.detail""",
            (job, at if status == "ok" else None, at, error, status,
             _json.dumps(detail) if detail is not None else None, status))
        conn.commit()


def job_states():
    conn = connect()
    with _conn_lock:
        rows = conn.execute("SELECT * FROM job_state").fetchall()
    return {row["job"]: dict(row) for row in rows}


# ── TinyFish usage log (persisted) ──────────────────────────────────────────
def record_usage(event):
    """Sink registered with tinyfish_service.set_usage_sink()."""
    conn = connect()
    with _conn_lock:
        conn.execute(
            "INSERT INTO usage_log (at, type, key, success, error, rotated, day) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (event.get("at"), event.get("type"), event.get("key"),
             1 if event.get("success") else 0, event.get("error"),
             1 if event.get("rotated") else 0, event.get("day")))
        conn.commit()


def usage_total_for_day(day):
    """Calls recorded on TinyFish's `day` (YYYY-MM-DD at 00:00 UTC).

    Backs tinyfish_service.set_usage_baseline() so the budget gate survives a
    restart. Rows written before the `day` column existed are excluded — they
    cannot be attributed to a reset window, and under-counting one legacy row
    is safer than letting a restart silently refill the allowance.
    """
    if not day:
        return 0
    conn = connect()
    with _conn_lock:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM usage_log WHERE day = ?", (day,)
        ).fetchone()
    return int(row["n"]) if row else 0


def usage_totals():
    """Per-key totals from the persisted log (survives a restart)."""
    conn = connect()
    with _conn_lock:
        rows = conn.execute(
            """SELECT key,
                      COUNT(*)                     AS total,
                      SUM(success)                 AS ok,
                      SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS errors,
                      SUM(rotated)                 AS failovers,
                      MAX(at)                      AS last_at
               FROM usage_log GROUP BY key ORDER BY key""").fetchall()
    return [dict(row) for row in rows]


def usage_log(limit=100):
    conn = connect()
    with _conn_lock:
        rows = conn.execute(
            "SELECT * FROM usage_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]
