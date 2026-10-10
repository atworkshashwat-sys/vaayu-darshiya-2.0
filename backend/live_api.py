"""
HTTP routes for the live layer: SSE broadcast, alerts, weather, status (spec §6, §7).

    GET  /api/stream                     SSE — readings | alerts | weather | status
    GET  /api/alerts                     official first, then model, de-duplicated
    GET  /api/weather                    current weather + fog/wind flags
    GET  /api/case-study                 stored case figures (reads only, no calls)
    GET  /api/status                     per-source health, calls vs cap, key_1
    GET  /api/admin/tinyfish-usage       protected: totals per key and the cap
    POST /api/admin/tinyfish/refresh     protected: force one job now (ops/demo)

Mounted by api.py with `app.include_router(live_router)`. Existing endpoints
are untouched; /api/stations is extended in place rather than replaced.

Auth for the admin routes: a bearer token from LIVE_ADMIN_TOKEN. When that
variable is unset the routes answer 404 — an operator who has not configured a
token gets no exposed admin surface rather than an open one.
"""

import os

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

import live_data
import live_jobs
import live_store
import live_stream
import tinyfish_service

router = APIRouter(prefix="/api", tags=["live"])

# SSE must never be cached or buffered — a proxy that buffers would defeat the
# whole point of a push channel (spec §7.5).
SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",     # nginx: disable proxy buffering
}


def _admin_token():
    return tinyfish_service.env("LIVE_ADMIN_TOKEN")


def _require_admin(authorization: str | None):
    token = _admin_token()
    if not token:
        # Not configured → the endpoint does not exist, rather than existing
        # with a default password.
        raise HTTPException(status_code=404, detail="admin routes not configured")
    provided = ""
    if authorization:
        scheme, _, value = authorization.partition(" ")
        provided = value if scheme.lower() == "bearer" else authorization
    if provided != token:
        raise HTTPException(status_code=401, detail="invalid admin token")


@router.get("/stream")
async def stream(request: Request):
    """Server-Sent Events channel (spec §7.1).

    Each connection gets its own bounded queue; a job's single validated result
    is fanned out to every client. Heartbeats go out every 25 s so idle proxies
    do not reap the socket, and the connection is dropped cleanly when the
    browser goes away.
    """
    try:
        client = live_stream.subscribe()
    except ConnectionRefusedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    async def generate():
        try:
            async for chunk in live_stream.stream(client):
                # A client that vanished mid-send raises here; the finally
                # block then removes it so it is never counted as connected.
                if await request.is_disconnected():
                    break
                yield chunk
        finally:
            live_stream.unsubscribe(client)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.get("/alerts")
def alerts(limit: int = 8):
    """Official advisories first (flagged `official`), then model-generated."""
    if limit < 1 or limit > 50:
        limit = 8
    payload = live_data.alerts_payload(limit=limit)
    payload["server_time"] = live_jobs.now_ist().isoformat(timespec="seconds")
    payload["active"] = tinyfish_service.is_active()
    return payload


@router.get("/weather")
def weather():
    payload = live_data.weather_payload()
    payload["server_time"] = live_jobs.now_ist().isoformat(timespec="seconds")
    payload["active"] = tinyfish_service.is_active()
    return payload


@router.get("/case-study")
def case_study():
    """The stored "Delhi's Drastic Case" figures (job D), never fetched here.

    Job D runs once a day in the background; this route only reads the store,
    so opening the page cannot spend a single call against the daily cap.
    Each figure keeps its own source and as-of date — one the sources never
    stated simply does not appear, rather than being filled in.
    """
    payload = live_data.case_payload()
    payload["server_time"] = live_jobs.now_ist().isoformat(timespec="seconds")
    payload["active"] = tinyfish_service.is_active()
    return payload


@router.get("/status")
def status():
    """Per-source health, last success, calls used vs cap, key in use as key_N."""
    return live_jobs.status_payload()


@router.get("/admin/tinyfish-usage")
def tinyfish_usage(authorization: str | None = Header(default=None)):
    """Totals per key plus the cap. Keys appear only as key_1 / key_2 / key_3."""
    _require_admin(authorization)
    snapshot = tinyfish_service.usage_snapshot()
    return {
        **snapshot,
        # The persisted log survives a restart, so the totals here are the
        # ones that matter for "did we stay under the cap".
        "persisted": live_store.usage_totals(),
        "recent": live_store.usage_log(limit=100),
    }


@router.post("/admin/tinyfish/refresh")
def admin_refresh(job: str = "readings",
                  authorization: str | None = Header(default=None)):
    """Force one job now (used for demos and for verifying a fresh fetch)."""
    _require_admin(authorization)
    if job not in live_jobs.JOBS:
        raise HTTPException(status_code=400, detail=f"unknown job {job!r}")
    ok, error = live_jobs.run_job(job, force=True)
    return {"job": job, "ok": ok, "error": error,
            "calls_used": tinyfish_service.calls_used(),
            "cap": tinyfish_service.daily_cap()}
