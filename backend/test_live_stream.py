"""
Tests for the SSE broker and the live routes (spec §7, §10).

Run:
    cd backend && python -m unittest -v test_live_stream

TinyFish and the store are mocked — no quota is spent, no network is touched.
The headline case from the spec: two clients connected, one scheduler event,
both receive it; a disconnected client is cleaned up.
"""

import asyncio
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import live_stream
from fastapi import FastAPI
from fastapi.testclient import TestClient


def drain(client, want=1, timeout=2.0):
    """Pull `want` events off a client's queue within `timeout` seconds."""

    async def _pull():
        out = []
        for _ in range(want):
            try:
                out.append(await asyncio.wait_for(client.queue.get(),
                                                  timeout=timeout))
            except asyncio.TimeoutError:
                break
        return out

    return asyncio.run(_pull())


class FanoutTests(unittest.TestCase):
    """spec §10: two clients, one event, both receive it."""

    def setUp(self):
        live_stream.reset()

    def tearDown(self):
        live_stream.reset()

    def test_two_clients_both_receive_the_same_event(self):
        a, b = live_stream.subscribe(), live_stream.subscribe()
        self.assertEqual(live_stream.client_count(), 2)

        live_stream.publish("readings", {"stations": [{"site_id": 1}],
                                         "fetched_at": "2026-10-10T18:00:00+05:30"})

        got_a, got_b = drain(a, 1), drain(b, 1)
        self.assertEqual(len(got_a), 1)
        self.assertEqual(len(got_b), 1)
        # Same event object content, both typed `readings`, both stamped.
        self.assertEqual(got_a[0]["type"], "readings")
        self.assertEqual(got_b[0]["type"], "readings")
        self.assertEqual(got_a[0]["server_time"], got_b[0]["server_time"])
        self.assertEqual(got_a[0]["data"], got_b[0]["data"])

    def test_publish_before_anyone_connects_is_a_noop_not_a_crash(self):
        event = live_stream.publish("weather", {"fog": True})
        self.assertEqual(event["type"], "weather")
        self.assertEqual(live_stream.client_count(), 0)

    def test_unknown_event_type_is_refused(self):
        with self.assertRaises(ValueError):
            live_stream.publish("nonsense", {})

    def test_each_event_type_keeps_its_name(self):
        client = live_stream.subscribe()
        for kind in live_stream.EVENT_TYPES:
            live_stream.publish(kind, {"job": kind})
        events = drain(client, 4)
        self.assertEqual([e["type"] for e in events], list(live_stream.EVENT_TYPES))


class LifecycleTests(unittest.TestCase):
    """A disconnected client must be cleaned up (spec §10)."""

    def setUp(self):
        live_stream.reset()

    def tearDown(self):
        live_stream.reset()

    def test_unsubscribe_removes_the_client(self):
        client = live_stream.subscribe()
        self.assertEqual(live_stream.client_count(), 1)
        live_stream.unsubscribe(client)
        self.assertEqual(live_stream.client_count(), 0)

    def test_unsubscribing_twice_is_safe(self):
        client = live_stream.subscribe()
        live_stream.unsubscribe(client)
        live_stream.unsubscribe(client)          # the `finally` may run twice
        self.assertEqual(live_stream.client_count(), 0)

    def test_connection_cap_is_enforced(self):
        live_stream.MAX_CLIENTS = 3
        try:
            for _ in range(3):
                live_stream.subscribe()
            with self.assertRaises(ConnectionRefusedError):
                live_stream.subscribe()
            self.assertEqual(live_stream.client_count(), 3)
            self.assertEqual(live_stream.stats()["rejected"], 1)
        finally:
            live_stream.MAX_CLIENTS = 200

    def test_a_full_queue_drops_the_oldest_event_instead_of_blocking(self):
        client = live_stream.subscribe()
        for i in range(live_stream.QUEUE_SIZE + 25):
            live_stream.publish("status", {"i": i})
        # Never grew past its cap, and the newest event is still there.
        self.assertEqual(client.queue.qsize(), live_stream.QUEUE_SIZE)
        events = drain(client, live_stream.QUEUE_SIZE)
        self.assertEqual(events[-1]["data"]["i"], live_stream.QUEUE_SIZE + 24)

    def test_stats_track_published_and_delivered(self):
        a, b = live_stream.subscribe(), live_stream.subscribe()
        live_stream.publish("alerts", {"items": []})
        drain(a, 1), drain(b, 1)
        stats = live_stream.stats()
        self.assertEqual(stats["published"], 1)
        self.assertEqual(stats["delivered"], 2)
        self.assertEqual(stats["peak"], 2)


class WireFormatTests(unittest.TestCase):
    def test_event_frame_has_type_and_single_line_json(self):
        frame = live_stream.format_event(
            {"type": "weather", "server_time": "2026-10-10T18:00:00+05:30",
             "data": {"fog": True, "temp_c": 24.5}})
        self.assertTrue(frame.startswith("event: weather\n"))
        self.assertIn("data: {", frame)
        self.assertTrue(frame.endswith("\n\n"))       # SSE frame terminator
        # Exactly one data: line — a bare newline inside the JSON would split
        # the frame and EventSource would deliver a truncated payload.
        self.assertEqual(frame.count("\ndata: "), 1)

    def test_frame_payload_round_trips(self):
        frame = live_stream.format_event(
            {"type": "readings", "server_time": "2026-10-10T18:00:00+05:30",
             "data": {"stations": [{"name": "Ashok Vihar", "status": "live"}]}})
        payload = json.loads(frame.split("data: ", 1)[1].strip())
        self.assertEqual(payload["type"], "readings")
        self.assertEqual(payload["data"]["stations"][0]["name"], "Ashok Vihar")
        self.assertIn("server_time", payload)

    def test_heartbeat_is_an_sse_comment(self):
        beat = live_stream.heartbeat()
        self.assertTrue(beat.startswith(": heartbeat "))
        self.assertTrue(beat.endswith("\n\n"))


class StreamGeneratorTests(unittest.TestCase):
    """The async generator that backs GET /api/stream."""

    def setUp(self):
        live_stream.reset()

    def tearDown(self):
        live_stream.reset()

    def test_connects_then_delivers_published_events(self):
        async def scenario():
            client = live_stream.subscribe()
            gen = live_stream.stream(client)
            first = await asyncio.wait_for(anext(gen), timeout=2.0)   # snapshot
            live_stream.publish("alerts", {"items": []})
            second = await asyncio.wait_for(anext(gen), timeout=2.0)
            await gen.aclose()
            return first, second

        first, second = asyncio.run(scenario())
        self.assertIn("event: status", first)          # the connect snapshot
        self.assertIn("event: alerts", second)

    def test_times_out_to_a_heartbeat_so_proxies_stay_awake(self):
        live_stream.HEARTBEAT_SECONDS = 0.1

        async def scenario():
            client = live_stream.subscribe()
            gen = live_stream.stream(client)
            await anext(gen)                            # connect snapshot
            chunk = await asyncio.wait_for(anext(gen), timeout=2.0)
            await gen.aclose()
            return chunk

        try:
            chunk = asyncio.run(scenario())
        finally:
            live_stream.HEARTBEAT_SECONDS = 25
        self.assertTrue(chunk.startswith(": heartbeat"))


class LiveRouteTests(unittest.TestCase):
    """The JSON routes (stream is covered by the generator tests above)."""

    @classmethod
    def setUpClass(cls):
        app = FastAPI()
        from live_api import router
        app.include_router(router)
        cls.client = TestClient(app)

    def setUp(self):
        live_stream.reset()

    def tearDown(self):
        live_stream.reset()

    def test_weather_reports_unavailable_rather_than_500(self):
        # Empty store: the card says unavailable, the endpoint still answers.
        with mock.patch("live_data.weather_payload",
                                 return_value={"status": "unavailable",
                                               "message": "No live weather yet."}):
            res = self.client.get("/api/weather")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "unavailable")

    def test_status_reports_the_cap_and_never_a_key(self):
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertIn("tinyfish", body)
        self.assertEqual(body["tinyfish"]["cap"], 400)
        self.assertIn("jobs", body)
        self.assertIn("readings", body["jobs"])
        # Key material must never appear in an admin-style response.
        self.assertNotIn("sk-tinyfish", json.dumps(body))

    def test_status_announces_the_expired_state(self):
        with mock.patch("live_jobs.tinyfish_service.is_active",
                                 return_value=False), \
                mock.patch("live_jobs.tinyfish_service.is_expired",
                                    return_value=True), \
                mock.patch("live_jobs.is_running", return_value=False):
            body = self.client.get("/api/status").json()
        self.assertTrue(body["expired"])
        self.assertIn("Live context ended", body["message"])

    def test_admin_usage_is_locked_behind_a_token(self):
        # Unconfigured token → the route does not exist (404, not an open API).
        res = self.client.get("/api/admin/tinyfish-usage")
        self.assertIn(res.status_code, (401, 404))

    def test_admin_usage_accepts_the_configured_bearer_token(self):
        with mock.patch("live_api._admin_token", return_value="s3cret"):
            denied = self.client.get("/api/admin/tinyfish-usage")
            allowed = self.client.get("/api/admin/tinyfish-usage",
                                      headers={"Authorization": "Bearer s3cret"})
        self.assertEqual(denied.status_code, 401)
        self.assertEqual(allowed.status_code, 200)
        self.assertNotIn("sk-tinyfish", allowed.text)

    def test_stream_refuses_new_clients_at_the_cap(self):
        live_stream.MAX_CLIENTS = 0
        try:
            res = self.client.get("/api/stream")
        finally:
            live_stream.MAX_CLIENTS = 200
        self.assertEqual(res.status_code, 503)

    def test_alerts_always_answer_200_with_an_official_flag(self):
        with mock.patch(
                "live_data.alerts_payload",
                return_value={"items": [{"id": "a", "title": "CAQM",
                                         "official": True, "severity": "high",
                                         "source": "CAQM", "url": "https://caqm.nic.in/x",
                                         "summary": "", "kind": "grap",
                                         "published_at": None,
                                         "fetched_at": "2026-10-10T18:00:00+05:30"}],
                              "fetched_at": "2026-10-10T18:00:00+05:30"}):
            body = self.client.get("/api/alerts").json()
        self.assertTrue(body["items"][0]["official"])
        self.assertIn("server_time", body)


if __name__ == "__main__":
    unittest.main()
