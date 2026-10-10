"""
Targeted tests for the Feature 1 (Active Alerts) live-context module.

Run:
    cd backend && python -m unittest -v test_tinyfish_context

stdlib `unittest`, TinyFish fully mocked — no API quota is spent and no
network call is made.
"""

import os
import re
import sys
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tinyfish_context as ctx
import tinyfish_service
from context_api import router
from fastapi import FastAPI
from fastapi.testclient import TestClient

OFFICIAL_URL = "https://caqm.nic.in/grap-stage-3-delhi"
PRESS_URL = "https://www.ndtv.com/delhi-news/grap-stage-3-2026"
JUNK_URL = "https://totally-random-blog.example.com/pollution"

SEARCH_RESULTS = [
    {"title": "CAQM imposes GRAP Stage 3 restrictions in Delhi NCR",
     "url": OFFICIAL_URL, "snippet": "Construction activities banned under GRAP Stage 3.",
     "site_name": "caqm.nic.in", "publisher": "CAQM", "date": "1 day ago"},
    {"title": "Delhi reels under severe air quality as GRAP Stage 3 kicks in",
     "url": PRESS_URL, "snippet": "Schools asked to avoid outdoor activities.",
     "site_name": "ndtv.com", "publisher": "NDTV", "date": "1 day ago"},
    {"title": "Random blog on pollution",
     "url": JUNK_URL, "snippet": "not an official source", "site_name": "example.com"},
]

FETCH_RESULT = {
    "url": OFFICIAL_URL,
    "title": "CAQM imposes GRAP Stage 3 restrictions in Delhi NCR",
    "site_name": "caqm.nic.in",
    "published_date": "2026-10-08",
    "text": "# GRAP Stage 3\n\nUpdated on: Oct 8, 2026 By CAQM Desk\n"
            "The Commission has directed that construction activities remain "
            "banned across Delhi NCR until further notice. Schools have been "
            "asked to shift to online classes. Subscribe to our newsletter "
            "for more updates.",
}


def fresh():
    """Empty cache so every test starts cold."""
    ctx._cache.clear()


class HelperTests(unittest.TestCase):
    def test_sanitize_strips_html_and_control_chars(self):
        out = ctx.sanitize("<script>alert(1)</script>Bad value\x00 here", limit=100)
        self.assertNotIn("<script>", out)
        self.assertNotIn("\x00", out)
        self.assertIn("Bad value here", out)

    def test_sanitize_caps_length_at_word_boundary(self):
        out = ctx.sanitize("one two three four five six", limit=12)
        self.assertLessEqual(len(out), 16)
        self.assertTrue(out.endswith("…"))

    def test_summarize_is_short_and_skips_boilerplate(self):
        summary = ctx.summarize(FETCH_RESULT["text"])
        self.assertLessEqual(len(summary), ctx.SUMMARY_CHARS)
        self.assertIn("construction activities remain banned", summary.lower())
        self.assertNotIn("subscribe", summary.lower())

    def test_is_useful_drops_homepages_and_page_chrome(self):
        chrome = "Accessibility Options Text To Speech Bigger Text Small Text " \
                 "Line Height Highlight Links Reset All Settings"
        self.assertFalse(ctx.is_useful("CAQM: Home", chrome))
        self.assertFalse(ctx.is_useful("Home", "A landing page that states nothing about advisories."))
        self.assertFalse(ctx.is_useful("CAQM imposes GRAP Stage 3", "too short"))
        self.assertFalse(ctx.is_useful(
            "CAQM issues advisory",
            "Accessibility Options and then a real sentence about the advisory itself."))
        self.assertTrue(ctx.is_useful(
            "CAQM imposes GRAP Stage 3 restrictions",
            "Construction activities remain banned across Delhi NCR until further notice."))

    def test_prefer_recent_prefers_current_and_keeps_old_as_last_resort(self):
        old = {"id": "old", "published_at": "2023-12-07"}
        new = {"id": "new", "published_at": "2026-10-01"}
        undated = {"id": "undated", "published_at": None}
        self.assertEqual(
            [i["id"] for i in ctx.prefer_recent([old, new, undated])],
            ["new", "undated"],
        )
        # everything old → show it anyway (the date is rendered in the UI)
        self.assertEqual([i["id"] for i in ctx.prefer_recent([old])], ["old"])

    def test_severity_mapping(self):
        self.assertEqual(ctx.severity_of("GRAP Stage 4 imposed", ""), "high")
        self.assertEqual(ctx.severity_of("CAQM advisory on air quality", ""), "medium")
        self.assertEqual(ctx.severity_of("Rain likely this week", ""), "low")

    def test_dedupe_drops_near_duplicate_headlines(self):
        items = [
            {"title": "CAQM imposes GRAP Stage 3 restrictions in Delhi",
             "url": "https://caqm.nic.in/a"},
            {"title": "CAQM imposes GRAP Stage 3 restrictions in Delhi NCR",
             "url": "https://ndtv.com/b"},
            {"title": "IMD forecasts dense fog over the plains",
             "url": "https://imd.gov.in/c"},
        ]
        kept = ctx.dedupe(items)
        self.assertEqual(len(kept), 2)

    def test_allowlist_blocks_unofficial_sources(self):
        self.assertTrue(ctx.is_allowed(OFFICIAL_URL))
        self.assertTrue(ctx.is_allowed(PRESS_URL))
        self.assertFalse(ctx.is_allowed(JUNK_URL))
        self.assertTrue(ctx.is_official(OFFICIAL_URL))
        self.assertFalse(ctx.is_official(PRESS_URL))

    def test_parse_published_handles_absolute_and_relative(self):
        self.assertEqual(ctx.parse_published("2026-10-08"), "2026-10-08")
        self.assertEqual(ctx.parse_published("Dec 19, 2025"), "2025-12-19")
        self.assertEqual(ctx.parse_published("2 days ago")[:4], "2026")
        self.assertIsNone(ctx.parse_published("soon"))


class RefreshAlertsTests(unittest.TestCase):
    def setUp(self):
        fresh()

    def tearDown(self):
        fresh()

    def test_success_builds_structured_items(self):
        def fake_fetch(url, **kwargs):
            return {**FETCH_RESULT, "url": url, "site_name": ctx._host(url)}

        with mock.patch.object(tinyfish_service, "search", return_value=SEARCH_RESULTS), \
                mock.patch.object(tinyfish_service, "fetch_text", side_effect=fake_fetch):
            data = ctx.refresh_alerts()

        items = data["items"]
        self.assertEqual(len(items), 2)
        # official source sorts above press coverage
        self.assertTrue(items[0]["official"])
        self.assertEqual(items[0]["source"], "CAQM")
        self.assertFalse(items[1]["official"])
        self.assertEqual(items[1]["source"], "NDTV")
        first = items[0]
        for field in ("id", "title", "severity", "summary", "source", "url",
                      "published_at", "fetched_at"):
            self.assertIn(field, first)
        self.assertLessEqual(len(first["summary"]), 200)
        self.assertIn(first["severity"], ("high", "medium", "low"))
        self.assertEqual(first["published_at"], "2026-10-08")
        # junk domain never survives
        self.assertTrue(all(i["url"] != JUNK_URL for i in items))
        self.assertTrue(data["sources"])

    def test_fetch_failure_falls_back_to_search_snippet(self):
        with mock.patch.object(tinyfish_service, "search", return_value=SEARCH_RESULTS), \
                mock.patch.object(tinyfish_service, "fetch_text",
                                  side_effect=tinyfish_service.TinyFishError("timed out")):
            data = ctx.refresh_alerts()
        self.assertTrue(data["items"])
        self.assertIn("Construction activities banned", data["items"][0]["summary"])
        # published_at is either a date derived from the stated "1 day ago",
        # or None when the source gives no date at all — never invented.
        published = data["items"][0]["published_at"]
        self.assertTrue(published is None or re.fullmatch(r"\d{4}-\d{2}-\d{2}", published))

    def test_redirect_to_unlisted_host_keeps_original_link(self):
        def fake_fetch(url, **kwargs):
            # TinyFish followed a redirect off the allowlisted domain.
            return {**FETCH_RESULT, "url": "https://evil.example.net/landing"}

        with mock.patch.object(tinyfish_service, "search", return_value=SEARCH_RESULTS), \
                mock.patch.object(tinyfish_service, "fetch_text", side_effect=fake_fetch):
            data = ctx.refresh_alerts()

        for item in data["items"]:
            self.assertTrue(
                item["url"].startswith(("https://caqm.nic.in", "https://www.ndtv.com")),
                f"unexpected link target: {item['url']}",
            )
            self.assertNotIn("evil.example.net", item["url"])

    def test_timeout_without_cache_reports_unavailable(self):
        with mock.patch.object(tinyfish_service, "search",
                               side_effect=tinyfish_service.TinyFishError("timed out")):
            payload = ctx.get_alerts()
        self.assertEqual(payload["status"], "unavailable")
        self.assertEqual(payload["items"], [])
        self.assertIsNone(payload["fetched_at"])
        self.assertIn("timed out", payload["message"])

    def test_malformed_results_do_not_raise(self):
        malformed = [{"title": "no url at all"}, {"url": ""}, None, "string-entry"]
        with mock.patch.object(tinyfish_service, "search", return_value=malformed), \
                mock.patch.object(tinyfish_service, "fetch_text", return_value={}):
            payload = ctx.get_alerts()
        self.assertEqual(payload["status"], "unavailable")
        self.assertEqual(payload["items"], [])

    def test_stale_payload_is_served_when_refresh_fails(self):
        good = {"items": [{"id": "x", "title": "Old alert", "severity": "high",
                           "summary": "s", "source": "CAQM", "url": OFFICIAL_URL,
                           "official": True, "published_at": None,
                           "fetched_at": "2026-10-10T09:00:00"}],
                "sources": []}
        ctx._cache["alerts"] = {"data": good, "stored_at": time.time()}
        with mock.patch.object(tinyfish_service, "search",
                               side_effect=tinyfish_service.TinyFishError("HTTP 429")):
            payload = ctx.get_alerts(force=True)

        self.assertEqual(payload["status"], "stale")
        self.assertTrue(payload["stale"])
        self.assertEqual(payload["items"][0]["title"], "Old alert")
        self.assertGreaterEqual(payload["age_seconds"], 0)
        self.assertIn("showing the last result", payload["message"])

    def test_cache_prevents_repeated_tinyfish_calls(self):
        with mock.patch.object(tinyfish_service, "search", return_value=SEARCH_RESULTS) as search, \
                mock.patch.object(tinyfish_service, "fetch_text", return_value=FETCH_RESULT) as fetch:
            ctx.get_alerts()
            calls_after_first = search.call_count
            ctx.get_alerts()
            self.assertEqual(search.call_count, calls_after_first)  # TTL hit
            ctx.get_alerts(force=True)
            self.assertGreater(search.call_count, calls_after_first)  # manual refresh

    def test_transient_error_is_retried_once(self):
        # 3 queries; the first query's attempt times out and is retried once,
        # so side_effect needs err + 3 successful results.
        with mock.patch.object(tinyfish_service, "search",
                               side_effect=[tinyfish_service.TinyFishError("timed out"),
                                            SEARCH_RESULTS, SEARCH_RESULTS,
                                            SEARCH_RESULTS]) as search, \
                mock.patch.object(tinyfish_service, "fetch_text", return_value=FETCH_RESULT):
            data = ctx.refresh_alerts()
        self.assertTrue(data["items"])
        self.assertGreaterEqual(search.call_count, 2)  # the retry actually happened

    def test_auth_error_is_not_retried(self):
        with mock.patch.object(tinyfish_service, "search",
                               side_effect=tinyfish_service.TinyFishError("HTTP 401")) as search:
            with self.assertRaises(tinyfish_service.TinyFishError):
                ctx.refresh_alerts()
        # exactly one attempt per query — no doubling, so 401 was never retried
        self.assertEqual(search.call_count, len(ctx.ALERT_SEARCHES))

    def test_missing_key_fails_closed(self):
        with mock.patch.dict(os.environ, {"TINYFISH_API_KEY": ""}):
            with self.assertRaises(tinyfish_service.TinyFishError):
                ctx.refresh_alerts()


class ContextEndpointTests(unittest.TestCase):
    def setUp(self):
        fresh()
        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)

    def tearDown(self):
        fresh()

    def test_alerts_endpoint_returns_envelope(self):
        with mock.patch.object(tinyfish_service, "search", return_value=SEARCH_RESULTS), \
                mock.patch.object(tinyfish_service, "fetch_text", return_value=FETCH_RESULT):
            res = self.client.get("/api/context/alerts")

        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["status"], "ok")
        self.assertIn("fetched_at", body)
        self.assertIn("ttl_seconds", body)
        self.assertIn("sources", body)
        self.assertTrue(body["items"])
        self.assertTrue(body["sources"][0]["url"].startswith("https://"))

    def test_alerts_endpoint_unavailable_is_still_http_200(self):
        with mock.patch.object(tinyfish_service, "search",
                               side_effect=tinyfish_service.TinyFishError("timed out")):
            res = self.client.get("/api/context/alerts")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "unavailable")


if __name__ == "__main__":
    unittest.main(verbosity=2)
