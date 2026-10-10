"""
HTTP routes for live web context (TinyFish) — isolated to the dashboard
sections that consume it.

    GET  /api/context/alerts            Active Alerts       (Feature 1)
    POST /api/context/weather-windows   Best Time           (Feature 2)
    GET  /api/context/delhi-case        Delhi's Drastic Case (Feature 3)

Mounted by api.py with `app.include_router(context_router)`. Existing
endpoints are untouched; every response is JSON with `fetched_at`, `stale`,
`ttl_seconds` and `sources` so the UI can render loading / fresh / stale /
unavailable states without guessing.

`?force=true` bypasses the TTL for a manual refresh (still single-flight, so
concurrent clicks cost one TinyFish round-trip).
"""

from fastapi import APIRouter, Body

import tinyfish_context

router = APIRouter(prefix="/api/context", tags=["context"])

# Background refresher: wakes every REFRESH_SECONDS and refreshes only features
# that have actually been served (nothing is fetched if unused), and only once
# that feature's own TTL has elapsed (alerts 20 min, weather windows 6 h).
tinyfish_context.start_scheduler()


@router.get("/alerts")
def context_alerts(force: bool = False):
    """Official + reputable pollution advisories for Delhi NCR."""
    return tinyfish_context.get_alerts(force=force)


@router.post("/weather-windows")
def context_weather_windows(payload: dict = Body(default={}), force: bool = False):
    """Weather/notices context + reasoned adjustments for the ML time windows.

    Body: {"slots": [{"period", "hours", "status"}]}. Every ML verdict is
    echoed back as `ml_status`; the returned `status` differs only when the
    official forecast triggers a one-step downgrade — and then `reasons` says
    why. Malformed input degrades to an empty slot list, never a 500.
    """
    slots = payload.get("slots") if isinstance(payload, dict) else []
    if not isinstance(slots, list):
        slots = []
    return tinyfish_context.get_weather_windows(slots, force=force)
