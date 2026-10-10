"""
Live vs static data-source detection (surgical addition).

The forecast path reads model artefacts, but the *inputs* that would drive a
live prediction come from the live layer (backend/live_data.py, TinyFish). This
module answers one question per prediction run:

    "Are we serving from live inputs, or from the most recent static snapshot?"

Contract
--------
detect(feature_frame) -> DataSourceReport with:
    source       "live" | "static"
    is_live      bool
    fallback     True when live fetch failed/timed out (>10 s) and we fell back
    detail       human-readable note (which source / why)
    fetched_at   when the live payload was captured (ISO) or None

A live fetch that exceeds LIVE_TIMEOUT_SECONDS, raises, or returns no usable
values is treated as a miss: we fall back to the most recent static snapshot
(the database extract) and flag it, so the caller can label the response
"static (fallback)". Every outcome is logged.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("vaayu.datasource")

# Live fetches that exceed this are considered dead and fall back to static.
LIVE_TIMEOUT_SECONDS = 10.0


@dataclass
class DataSourceReport:
    source: str = "static"                 # "live" | "static"
    is_live: bool = False
    fallback: bool = False                 # live attempted but failed
    detail: str = ""
    fetched_at: Optional[str] = None
    error: Optional[str] = None
    extras: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Response-ready label: 'live', 'static', or 'static (fallback)'."""
        if self.is_live:
            return "live"
        return "static (fallback)" if self.fallback else "static"


def _live_payload():
    """Pull the freshest live inputs, or None. Never raises.

    Uses live_data's stored readings when present — those are what job A
    refreshes. A missing/empty store means no live source is available.
    """
    try:
        import live_data
    except Exception as exc:                                  # pragma: no cover
        log.warning("data_source: live_data import failed: %s", exc)
        return None

    try:
        payload = live_data.station_payload()
    except Exception as exc:
        log.warning("data_source: live_data.station_payload() failed: %s", exc)
        return None

    if not payload:
        return None

    # Count stations that actually carry a fresh live reading.
    fresh = [s for s in payload if s.get("status") == "live"]
    if not fresh:
        return None
    return {"stations": payload, "fresh_count": len(fresh)}


def detect(feature_frame=None, prefer_live: bool = True) -> DataSourceReport:
    """Determine whether the current prediction run uses live or static data.

    `feature_frame` is accepted for future use (a caller may already know how
    many features are present) but the live/static decision is driven purely by
    whether the live layer has fresh values within the timeout.
    """
    if not prefer_live:
        return DataSourceReport(source="static", is_live=False, fallback=False,
                                detail="live layer disabled for this run")

    t0 = time.time()
    payload = _live_payload()
    elapsed = time.time() - t0

    if elapsed > LIVE_TIMEOUT_SECONDS:
        log.warning("data_source: live fetch timed out (%.1fs > %.0fs) -> static fallback",
                    elapsed, LIVE_TIMEOUT_SECONDS)
        return DataSourceReport(
            source="static", is_live=False, fallback=True,
            detail=f"live fetch exceeded {LIVE_TIMEOUT_SECONDS:.0f}s; "
                   f"using most recent static snapshot",
            error=f"timeout after {elapsed:.1f}s")

    if payload is None:
        log.info("data_source: no live values available -> static")
        return DataSourceReport(source="static", is_live=False, fallback=False,
                                detail="no live readings available; using static snapshot")

    log.info("data_source: live (%d fresh stations, %.1fs)",
             payload["fresh_count"], elapsed)
    return DataSourceReport(
        source="live", is_live=True, fallback=False,
        detail=f"{payload['fresh_count']} fresh live station(s)",
        fetched_at=_now_iso(),
        extras={"fresh_stations": payload["fresh_count"]})


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
