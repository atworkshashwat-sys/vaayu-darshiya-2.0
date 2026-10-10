"""
AI Insights live research route (TinyFish) — isolated from every other endpoint.

Mounted by api.py with `app.include_router(ai_insights_router)`; it adds exactly
one path and touches nothing else:

    GET /api/ai-insights/research?city=Delhi

Contract:
    Always HTTP 200 and self-describing, so the AI Insights panel can show a
    retrieval status without ever breaking:
        status "ok"          → TinyFish Search returned results (items may still be empty)
        status "unavailable" → no API key, timeout, rate limit or upstream error
    The ML insight cards are generated in the browser from forecast data and
    never depend on this route, so TinyFish being down cannot affect them.
"""

from datetime import datetime as _dt

from fastapi import APIRouter

import tinyfish_service

router = APIRouter(prefix="/api/ai-insights", tags=["ai-insights"])

# Search is free up to 12k requests/day but rate limited to 30/min — cache the
# answer so tab refocus/re-renders do not burn quota.
CACHE_TTL_SECONDS = 600
FAILURE_CACHE_TTL_SECONDS = 60
MAX_ITEMS = 5

_cache = {}


def live_context():
    """Current live readings / weather / alerts, read straight from the store.

    No network calls: jobs A-C already refresh on their own schedule, so this
    is a pure local read and cannot spend the daily budget or stall the route.

    `confidence` is lowered when the panel would otherwise be reasoning over a
    conflicting or stale number, so the UI can say so instead of implying the
    insight is fully backed (spec §8).
    """
    try:
        import live_data
        import live_store
    except Exception:                                # noqa: BLE001
        return None

    readings = [
        {"name": s["name"], "aqi": s["live"].get("AQI"), "status": s["status"],
         "observed_at": s["observed_at"], "source": s["source"]}
        for s in live_data.station_payload()
    ]
    weather = live_data.weather_payload()
    alerts = live_data.alerts_payload(limit=8).get("items") or []

    reasons = []
    if any(r["status"] == "stale" for r in readings):
        reasons.append("some station readings are stale")
    if any(r.get("conflict") for r in live_data.station_payload()):
        reasons.append("sources disagree on at least one reading")
    if weather.get("status") == "stale":
        reasons.append("weather is stale")

    return {
        "readings": readings,
        "weather": weather,
        "alerts": alerts,
        "official_alerts": sum(1 for a in alerts if a.get("official")),
        "confidence": "reduced" if reasons else "high",
        "confidence_reasons": reasons,
        "collected_at": _dt.now().isoformat(timespec="seconds"),
    }


def _cache_get(key):
    entry = _cache.get(key)
    if not entry:
        return None
    expires, payload = entry
    if expires < _dt.now().timestamp():
        _cache.pop(key, None)
        return None
    return payload


def _cache_put(key, payload, ttl_seconds):
    _cache[key] = (_dt.now().timestamp() + ttl_seconds, payload)


def _build_query(city):
    city = (city or "Delhi").strip() or "Delhi"
    return f"{city} air quality pollution news and official advisory"


def _base(city, query):
    return {
        "status": "unavailable",
        "provider": "TinyFish Search",
        "city": city,
        "query": query,
        "retrieved_at": _dt.now().isoformat(timespec="seconds"),
        "cached": False,
        "items": [],
        "excerpt": None,
        "message": None,
        # Attached even when Search fails: a live reading from the store is
        # still a true statement, and it is what lets the panel show a
        # confidence badge rather than going blank.
        "live": live_context(),
    }


@router.get("/research")
def get_research(city: str = "Delhi"):
    """Latest Delhi NCR pollution news / advisories for the AI Insights section."""
    city = (city or "").strip()[:60] or "Delhi"
    cache_key = city.lower()

    cached = _cache_get(cache_key)
    if cached is not None:
        return {**cached, "cached": True}

    query = _build_query(city)
    try:
        items = tinyfish_service.search(query)
        if not items:
            # News-only search can come back empty — one wider retry on the web.
            items = tinyfish_service.search(query, domain_type="web")
    except tinyfish_service.TinyFishError as exc:
        payload = _base(city, query)
        payload["message"] = f"Live research unavailable ({exc})"
        _cache_put(cache_key, payload, FAILURE_CACHE_TTL_SECONDS)
        return payload

    # Fetch adds a short verbatim excerpt from the top result. Optional: if it
    # fails, the headlines/snippets above still stand on their own.
    excerpt = None
    if items:
        try:
            excerpt = tinyfish_service.fetch_text(items[0]["url"])
        except tinyfish_service.TinyFishError:
            excerpt = None

    payload = _base(city, query)
    payload.update({
        "status": "ok",
        "items": items[:MAX_ITEMS],
        "excerpt": excerpt,
    })
    _cache_put(cache_key, payload, CACHE_TTL_SECONDS)
    return payload
