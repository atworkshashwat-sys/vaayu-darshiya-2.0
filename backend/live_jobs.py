"""
Background scheduler for the live data jobs (spec §5).

    job A  readings    every 60 min
    job B  alerts      every 45 min      (inside the 30-60 min band)
    job C  weather     every 150 min     (inside the 2-3 h band)
    job D  case study  once at startup

Implemented with the project's existing pattern — one daemon thread that wakes
on an interval — rather than APScheduler, because that is what tinyfish_context
already does and adding a dependency needs operator sign-off (it was not asked
for). Nothing here imports a package that is not already in requirements.txt.

Behaviour:
  * jobs are staggered on first run so four TinyFish calls never land together
  * one full refresh happens at startup, so the first visitor sees data
  * each job publishes exactly ONE event per completed, validated refresh
  * the whole scheduler stops at TINYFISH_EXPIRES_AT, at which point every
    Monitor this process created is deleted and the site reports
    "Live context ended - last updated HH:MM IST"
  * a job that raises keeps the previous values (the store already holds them)
    and records the failure in job_state instead of killing the thread
"""

import logging
import threading
import time
from datetime import datetime, timedelta, timezone

import live_data
import live_store
import live_stream
import tinyfish_service

log = logging.getLogger("tinyfish.jobs")

IST = timezone(timedelta(hours=5, minutes=30))

# name -> (interval seconds, callable, SSE event type or None)
JOBS = {
    "readings":   (60 * 60,           live_data.refresh_readings,  "readings"),
    "alerts":     (45 * 60,           live_data.refresh_alerts,    "alerts"),
    "weather":    (150 * 60,          live_data.refresh_weather,   "weather"),
    "case_study": (24 * 60 * 60,      live_data.refresh_case_study, None),
}

# Stagger the first run of each job so startup does not burst four calls.
FIRST_RUN_DELAY = {
    "readings":   0,
    "alerts":     20,
    "weather":    45,
    "case_study": 75,
}

_thread = None
_stop = threading.Event()
_started = False
_started_lock = threading.Lock()
_last_expiry_cleanup = None


def now_ist():
    return datetime.now(IST)


def _event_payload(job, result):
    """The SSE body for a completed job — the same JSON the REST route serves."""
    if job == "readings":
        return {"stations": live_data.station_payload(),
                "fetched_at": result.get("fetched_at"),
                "offline": result.get("offline", [])}
    if job == "alerts":
        return live_data.alerts_payload()
    if job == "weather":
        return live_data.weather_payload()
    return {"stats": result.get("stats", [])}


def run_job(job, force=False):
    """Run one job once, publish its event, never raise into the caller.

    Returns (ok, error). A failure is recorded in job_state so /api/status can
    show it, and the previously stored values are simply left alone — the site
    then serves them with their age instead of an empty card.
    """
    spec = JOBS.get(job)
    if spec is None:
        return False, f"unknown job {job!r}"
    _interval, fn, event_type = spec
    try:
        result = fn(force=force)
    except Exception as exc:                        # noqa: BLE001 - jobs must not die
        live_store.record_job(job, "error", error=str(exc)[:500])
        log.warning("job %s failed: %s", job, exc)
        return False, str(exc)

    if event_type:
        try:
            live_stream.publish(event_type, _event_payload(job, result))
        except Exception:                           # noqa: BLE001 - broadcast is best effort
            log.debug("publish failed for %s", job, exc_info=True)
    return True, None


def _handle_expiry():
    """At TINYFISH_EXPIRES_AT: stop refreshing, delete Monitors, say so once."""
    global _last_expiry_cleanup
    if tinyfish_service.is_active():
        return False
    if _last_expiry_cleanup:
        return True
    log.info("live context window ended at %s — stopping the scheduler",
             tinyfish_service.expires_at().strftime("%H:%M IST"))
    try:
        tinyfish_service.delete_all_monitors()
    except Exception as exc:                        # noqa: BLE001 - cleanup is best effort
        log.warning("monitor cleanup failed: %s", exc)
    _last_expiry_cleanup = datetime.now(IST).isoformat(timespec="seconds")
    _stop.set()
    try:
        live_stream.publish("status", status_payload())
    except Exception:                               # noqa: BLE001
        pass
    return True


def status_payload():
    """Per-source health for /api/status and the `status` SSE event."""
    states = live_store.job_states()
    jobs = {}
    for job, (_interval, _fn, event_type) in JOBS.items():
        state = states.get(job, {})
        jobs[job] = {
            "interval_seconds": _interval,
            "event": event_type,
            "status": state.get("status", "never_run"),
            "last_success_at": state.get("last_success_at"),
            "last_attempt_at": state.get("last_attempt_at"),
            "last_error": state.get("last_error"),
            "detail": state.get("detail"),
        }
    active = tinyfish_service.is_active()
    ended_at = tinyfish_service.expires_at()
    return {
        "enabled": tinyfish_service.is_enabled(),
        "active": active,
        "expired": tinyfish_service.is_expired(),
        "expires_at": ended_at.isoformat(timespec="seconds"),
        "remaining_seconds": tinyfish_service.remaining_seconds(),
        "ended_at": _last_expiry_cleanup,
        "message": (
            None if active else
            f"Live context ended - last updated "
            f"{_last_expiry_cleanup or ended_at.strftime('%H:%M IST')}"
        ),
        "scheduler_running": is_running(),
        "jobs": jobs,
        "tinyfish": tinyfish_service.usage_snapshot(),
        "stream": live_stream.stats(),
        "server_time": now_ist().isoformat(timespec="seconds"),
    }


def health_payload():
    """The `last successful refresh per job` block for /health (spec §6)."""
    states = live_store.job_states()
    return {
        job: states.get(job, {}).get("last_success_at")
        for job in JOBS
    }


def _loop():
    next_due = {}
    first = True
    while not _stop.is_set():
        if not tinyfish_service.is_active():
            _handle_expiry()
            if _stop.is_set():
                break
            _stop.wait(30)
            continue

        for job, (interval, _fn, _event) in JOBS.items():
            if _stop.is_set():
                break
            now = time.monotonic()
            if job not in next_due:
                next_due[job] = now + FIRST_RUN_DELAY.get(job, 0)
                continue
            if now < next_due[job]:
                continue
            # Schedule the next run from the completion time so a slow fetch
            # cannot pile up overlapping runs of the same job.
            next_due[job] = now + interval
            run_job(job)

        first = False
        _stop.wait(5)


def start():
    """Start the scheduler once (idempotent; safe to call from app import)."""
    global _thread, _started
    with _started_lock:
        if _started and _thread is not None and _thread.is_alive():
            return _thread
        _stop.clear()
        _thread = threading.Thread(target=_loop, name="live-jobs", daemon=True)
        _thread.start()
        _started = True
        log.info("live scheduler started: %s",
                 ", ".join(f"{j} {i // 60}m" for j, (i, _f, _e) in JOBS.items()))
    return _thread


def stop(timeout=5.0):
    """Stop the scheduler and wait for it (tests / shutdown)."""
    global _started
    _stop.set()
    thread = _thread
    if thread is not None and thread.is_alive():
        thread.join(timeout=timeout)
    _started = False


def is_running():
    return _thread is not None and _thread.is_alive() and not _stop.is_set()


def refresh_all(force=True):
    """Run A→B→C→D once, in order, on the caller's thread (startup / manual).

    Used for the one full refresh at boot (spec §5) and by tests. Returns
    {job: (ok, error)}; no job's failure prevents the others from running.
    """
    return {job: run_job(job, force=force) for job in
            ("readings", "alerts", "weather", "case_study")}
