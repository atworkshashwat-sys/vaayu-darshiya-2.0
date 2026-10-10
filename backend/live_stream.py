"""
Server-Sent Events broadcast for the live dashboard (spec §7).

SSE rather than WebSockets because it rides on plain HTTP: no new dependency,
it survives Render's free-tier proxy, and the browser reconnects on its own.

    GET /api/stream            event: readings | alerts | weather | status

One in-process broker holds every connected client. After a scheduler job
finishes and validates, it calls `publish(event, payload)` exactly once and the
broker fans that single dict out to all connections — the scheduler never knows
how many browsers are attached.

Safety (spec §7.2, §5 hard rule 5):
  * `MAX_CLIENTS` caps concurrency; a client beyond the cap gets an immediate
    `error` event and a clean close, never a hung socket
  * each connection owns its own queue with a bounded size, so one stalled
    browser cannot grow memory without limit — the oldest event is dropped
  * a disconnect is detected on the next heartbeat and the client is removed
"""

import asyncio
import json
import threading
import time
from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))

MAX_CLIENTS = 200              # spec: cap concurrent connections
QUEUE_SIZE = 50                # events buffered per client before dropping old
HEARTBEAT_SECONDS = 25         # spec: a comment every 25 s keeps proxies awake
CLIENT_IDLE_TIMEOUT = 90       # no heartbeat ack in this long → drop the client

EVENT_TYPES = ("readings", "alerts", "weather", "status")


class _Client:
    """One connected browser: a bounded asyncio queue of event dicts."""

    __slots__ = ("queue", "connected_at", "last_seen", "events_sent")

    def __init__(self):
        self.queue = asyncio.Queue(maxsize=QUEUE_SIZE)
        self.connected_at = time.time()
        self.last_seen = time.time()
        self.events_sent = 0

    def offer(self, event):
        """Push without blocking the scheduler. Drop the oldest when full."""
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                self.queue.get_nowait()          # make room, drop stale event
            except asyncio.Empty:
                pass
            try:
                self.queue.put_nowait(event)
            except asyncio.QueueFull:
                pass                             # give up on this one event


_clients = set()
_clients_lock = threading.Lock()
_loop = None                  # the running event loop, captured on first use
_stats = {"published": 0, "delivered": 0, "rejected": 0, "peak": 0}


def _now_iso():
    return datetime.now(IST).isoformat(timespec="seconds")


def set_event_loop(loop):
    """Capture the server's event loop so publish() can be called from a
    scheduler thread (a different thread cannot touch an asyncio.Queue)."""
    global _loop
    _loop = loop


def client_count():
    with _clients_lock:
        return len(_clients)


def stats():
    with _clients_lock:
        peak = _stats["peak"]
    return {**_stats, "connected": client_count(), "peak": peak}


def subscribe():
    """Register a new client. Raises ConnectionRefusedError when the cap is
    reached, so the route can refuse cleanly instead of accepting a socket it
    will never serve."""
    client = _Client()
    with _clients_lock:
        if len(_clients) >= MAX_CLIENTS:
            _stats["rejected"] += 1
            raise ConnectionRefusedError(
                f"live stream is at its {MAX_CLIENTS} connection cap")
        _clients.add(client)
        _stats["peak"] = max(_stats["peak"], len(_clients))
    return client


def unsubscribe(client):
    with _clients_lock:
        _clients.discard(client)


def publish(event_type, payload):
    """Fan one event out to every connected client (spec §7.2).

    Safe from any thread: it schedules the queueing onto the captured event
    loop. If the loop is gone (shutdown) this is a no-op — a broadcast must
    never take the scheduler down with it.
    """
    if event_type not in EVENT_TYPES:
        raise ValueError(f"unknown event type {event_type!r}")

    event = {"type": event_type, "server_time": _now_iso(), "data": payload}
    with _clients_lock:
        clients = list(_clients)
        _stats["published"] += 1
    if not clients:
        return event

    loop = _loop

    def _deliver():
        for client in clients:
            client.offer(event)
            client.events_sent += 1
        with _clients_lock:
            _stats["delivered"] += len(clients)

    if loop is None or loop.is_closed():
        # Tests and shutdown: deliver inline so behaviour stays observable.
        _deliver()
        return event
    try:
        loop.call_soon_threadsafe(_deliver)
    except RuntimeError:                          # loop already stopping
        pass
    return event


def reset():
    """Drop every client and zero the counters (tests)."""
    global _loop
    with _clients_lock:
        _clients.clear()
        _stats.update({"published": 0, "delivered": 0, "rejected": 0, "peak": 0})
    _loop = None


# ── Wire format ─────────────────────────────────────────────────────────────
def format_event(event):
    """One SSE frame: `event:` + `data:` + a trailing blank line.

    The payload is JSON on a single `data:` line, which is what EventSource
    reassembles. `server_time` travels inside the JSON so the client can show
    "updated HH:MM IST" without trusting its own clock.
    """
    body = json.dumps(event["data"], default=str, ensure_ascii=False)
    return (f"event: {event['type']}\n"
            f"data: {json.dumps({'type': event['type'], 'server_time': event['server_time'], 'data': event['data']}, default=str, ensure_ascii=False)}\n\n")


def heartbeat():
    """SSE comment line — ignored by EventSource, keeps idle proxies open."""
    return f": heartbeat {_now_iso()}\n\n"


def initial_snapshot_event(snapshot):
    """The `status` frame sent on connect so a fresh tab has data at once."""
    return {"type": "status", "server_time": _now_iso(), "data": snapshot}


async def stream(client):
    """Async generator yielding SSE bytes for one connection (spec §7.1).

    Yields the heartbeat on its own schedule and any published events as they
    arrive. Returns when the client disappears or the server shuts down; the
    caller is responsible for `unsubscribe()`.
    """
    yield format_event(initial_snapshot_event({"kind": "connected",
                                               "max_clients": MAX_CLIENTS}))
    last_beat = time.time()
    while True:
        try:
            event = await asyncio.wait_for(
                client.queue.get(), timeout=HEARTBEAT_SECONDS)
        except asyncio.TimeoutError:
            now = time.time()
            if now - client.last_seen > CLIENT_IDLE_TIMEOUT:
                return                            # the browser is gone
            if now - last_beat >= HEARTBEAT_SECONDS:
                last_beat = now
                yield heartbeat()
            continue
        client.last_seen = time.time()
        yield format_event(event)
        last_beat = time.time()
