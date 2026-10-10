"""
HTTP routes for live web context (TinyFish) — isolated to the dashboard
sections that consume it.

    GET /api/context/alerts            Active Alerts      (Feature 1)
    GET /api/context/weather-windows   Best Time          (Feature 2)
    GET /api/context/delhi-case        Delhi's Drastic Case (Feature 3)

Mounted by api.py with `app.include_router(context_router)`. Existing
endpoints are untouched; every response is JSON with `fetched_at`, `stale`,
`ttl_seconds` and `sources` so the UI can render loading / fresh / stale /
unavailable states without guessing.

`?force=true` bypasses the TTL for a manual refresh (still single-flight, so
concurrent clicks cost one TinyFish round-trip).
"""

from fastapi import APIRouter

import tinyfish_context

router = APIRouter(prefix="/api/context", tags=["context"])

# Background refresher: wakes every REFRESH_SECONDS and refreshes only the
# features that have actually been served (nothing is fetched if unused).
tinyfish_context.start_scheduler()


@router.get("/alerts")
def context_alerts(force: bool = False):
    """Official + reputable pollution advisories for Delhi NCR."""
    return tinyfish_context.get_alerts(force=force)
