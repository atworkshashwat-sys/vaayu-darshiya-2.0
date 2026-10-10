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


def fake_batch(page_for_url):
    """Build a `fetch_many()` return value keyed by the URL it came from.

    The real API echoes each requested URL and omits any that failed, so the
    fake has to do the same: a bare list of page bodies would not exercise the
    URL join that stops one site's page being attributed to another's link.
    `page_for_url` returns `None` to simulate a URL the API dropped.
    """
    def fetch_many(urls, **kwargs):
        results = []
        for url in urls:
            page = page_for_url(url)
            if page is not None:
                results.append({**page, "url": url})
        return {"results": results, "errors": []}
    return fetch_many


def failing_batch(*args, **kwargs):
    raise tinyfish_service.TinyFishError("timed out")


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

    def test_summarize_never_quotes_a_sites_navigation(self):
        """A news home page has no full stop for hundreds of characters.

        That makes its whole menu a single "sentence", so the short-line
        boilerplate test never fires and the panel would have rendered
        "Edition IN * IN * US Weather Sign In ## TOI Today's ePaper * News"
        as the alert's summary.
        """
        nav_page = (
            "Edition IN * IN * US Weather Sign In ## TOI Today's ePaper * News"
            " * City News * delhi News * Cold wave grips Delhi: AQI in 'very "
            "poor' category. Flights were delayed at the airport due to dense fog."
        )
        summary = ctx.summarize(nav_page)
        self.assertNotIn("ePaper", summary)
        self.assertIn("Flights were delayed", summary)

    def test_a_page_that_is_only_navigation_summarises_to_nothing(self):
        # Nothing citable was stated, so the item is dropped by is_useful
        # rather than shown as a card built from a table of contents.
        self.assertEqual(ctx.summarize("Edition IN * IN * US Weather Sign In ## TOI"), "")

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
        def page_for(url):
            return {**FETCH_RESULT, "url": url, "site_name": ctx._host(url)}

        with mock.patch.object(tinyfish_service, "search", return_value=SEARCH_RESULTS), \
                mock.patch.object(tinyfish_service, "fetch_many",
                                  side_effect=fake_batch(page_for)):
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
                mock.patch.object(tinyfish_service, "fetch_many",
                                  side_effect=failing_batch):
            data = ctx.refresh_alerts()
        self.assertTrue(data["items"])
        self.assertIn("Construction activities banned", data["items"][0]["summary"])
        # published_at is either a date derived from the stated "1 day ago",
        # or None when the source gives no date at all — never invented.
        published = data["items"][0]["published_at"]
        self.assertTrue(published is None or re.fullmatch(r"\d{4}-\d{2}-\d{2}", published))

    def test_redirect_to_unlisted_host_keeps_original_link(self):
        def page_for(url):
            # TinyFish followed a redirect off the allowlisted domain: the
            # request URL is echoed unchanged and the landing page is separate.
            return {**FETCH_RESULT, "url": url,
                    "final_url": "https://evil.example.net/landing"}

        with mock.patch.object(tinyfish_service, "search", return_value=SEARCH_RESULTS), \
                mock.patch.object(tinyfish_service, "fetch_many",
                                  side_effect=fake_batch(page_for)):
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
                mock.patch.object(tinyfish_service, "fetch_many",
                                  return_value={"results": [], "errors": []}):
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
                mock.patch.object(tinyfish_service, "fetch_many",
                                  side_effect=fake_batch(lambda _url: FETCH_RESULT)) as batch:
            ctx.get_alerts()
            calls_after_first = search.call_count
            self.assertEqual(batch.call_count, 1)   # one batch, not one per URL
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
                mock.patch.object(tinyfish_service, "fetch_many",
                                  side_effect=fake_batch(lambda _url: FETCH_RESULT)):
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
        with mock.patch.dict(os.environ, {
            "TINYFISH_API_KEY_1": "", "TINYFISH_API_KEY_2": "",
            "TINYFISH_API_KEY_3": ""}, clear=True):
            with self.assertRaises(tinyfish_service.TinyFishError):
                ctx.refresh_alerts()

    def test_every_candidate_page_costs_exactly_one_batched_call(self):
        """Budget guard (spec §5, §8): job B must fit inside the daily cap.

        Fetching page by page made each refresh 3 searches + up to 5 fetches
        *with a retry apiece* — ~13 calls every 45 minutes, which is roughly
        400 on its own. One batch covers every candidate.
        """
        with mock.patch.object(tinyfish_service, "search", return_value=SEARCH_RESULTS), \
                mock.patch.object(tinyfish_service, "fetch_many",
                                  side_effect=fake_batch(lambda _u: FETCH_RESULT)) as batch:
            data = ctx.refresh_alerts()

        self.assertTrue(data["items"])
        self.assertEqual(batch.call_count, 1)
        # Every candidate URL is handed to that single call.
        requested = batch.call_args[0][0]
        self.assertEqual(len(requested), len(set(requested)))
        self.assertTrue(all(u.startswith("https://") for u in requested))


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
                mock.patch.object(tinyfish_service, "fetch_many",
                                  side_effect=fake_batch(lambda _u: FETCH_RESULT)):
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


# ── Feature 2 fixtures ──────────────────────────────────────────────────────
WEATHER_URL = "https://mausam.imd.gov.in/delhi/city_weather.php"
WEATHER_PAGE_TEXT = (
    "Delhi Forecast: Mainly clear sky. Wind speed 8-12 kmph. Humidity 68%. "
    "Maximum temperature 34°C. Shallow fog in the morning. No rain."
)
WEATHER_RESULT = {
    "title": "Delhi Weather Forecast - IMD",
    "url": WEATHER_URL,
    "snippet": "Wind speed 8-12 kmph. Humidity 68%. Fog in the morning.",
    "site_name": "mausam.imd.gov.in",
    "publisher": "IMD",
    "date": "today",
}
# Live-search shape: the site homepage ranks *above* the real forecast page,
# and its body mixes navigation junk with realized observations.
IMD_HOME_RESULT = {
    "title": "IMD Mausam - India Meteorological Department",
    "url": "https://mausam.imd.gov.in/index_en.php",
    "snippet": "Get accurate weather forecasts, cyclone alerts, radar images, "
               "and current conditions for all Indian cities.",
    "site_name": "mausam.imd.gov.in",
    "publisher": "IMD",
}
IMD_TABLE_RESULT = {
    "title": "Delhi/NCR | IMD Weather Forecast",
    "url": "https://mausam.imd.gov.in/delhiums/",
    "snippet": "Current Weather Across Delhi - NCR Delhi Gurugram Faridabad",
    "site_name": "mausam.imd.gov.in",
    "publisher": "IMD",
}
IMD_TABLE_TEXT = (
    "Oct 10, 2026 Oct 11, 2026 Oct 12, 2026  30.8 °C  Northeasterly 9.3 km/h  "
    "67 % Forecast  31.0 °C  13 km/h  55 % Forecast  29.8 °C  Calm 0 km/h  65 % Forecast  "
    "NA °C  NA % Forecast  NA °C  NA % Forecast"
)
IMD_HOME_TEXT = (
    "1?? ?\"? Districtwise Stationwise Weather Realized Dust suspended in the air "
    "30.4oC Feel Like 34.5oC 64% East-northeasterly 9.3 Km/h Observation time"
)
NOTICE_RESULTS = [
    {"title": "Delhi air: stubble burning pushes GRAP curbs",
     "url": "https://www.ndtv.com/delhi-news/stubble-burning-delhi",
     "snippet": "Farm residue burning in Punjab rose sharply this week, officials said.",
     "site_name": "ndtv.com", "publisher": "NDTV", "date": "1 day ago"},
    {"title": "Truck entry restricted in Delhi under GRAP",
     "url": "https://caqm.nic.in/notices/truck-restriction",
     "snippet": "Heavy goods vehicles are barred from entering the city limits.",
     "site_name": "caqm.nic.in", "publisher": "CAQM", "date": "2 days ago"},
    {"title": "Unofficial blog",
     "url": "https://totally-random-blog.example.com/pollution",
     "snippet": "not an allowlisted source", "site_name": "example.com"},
]
NOTICE_PAGE = {
    "url": "https://www.ndtv.com/delhi-news/stubble-burning-delhi",
    "title": "Delhi air: stubble burning pushes GRAP curbs",
    "site_name": "ndtv.com",
    "published_date": "2026-10-09",
    "text": "Farm residue burning in Punjab rose sharply this week, officials said. "
            "Authorities urged farmers to stop stubble burning. Subscribe to our newsletter.",
}
NOTICE_PAGE_CAQM = {
    "url": "https://caqm.nic.in/notices/truck-restriction",
    "title": "Truck entry restricted in Delhi under GRAP",
    "site_name": "caqm.nic.in",
    "published_date": "2026-10-08",
    "text": "Heavy goods vehicles are barred from entering the city limits. "
            "Traffic police issued an advisory for commuters on the border routes.",
}


def fake_weather_search(query, **kwargs):
    if "weather" in query.lower():
        return [WEATHER_RESULT]
    return NOTICE_RESULTS


def fake_weather_fetch(url, **kwargs):
    if "mausam.imd.gov.in" in url:
        return {**WEATHER_RESULT, "url": url, "text": WEATHER_PAGE_TEXT,
                "published_date": ""}
    if "caqm.nic.in" in url:
        return {**NOTICE_PAGE_CAQM, "url": url}
    return {**NOTICE_PAGE, "url": url}


class WeatherWindowsTests(unittest.TestCase):
    def setUp(self):
        fresh()

    def tearDown(self):
        fresh()

    # ── parsing ──────────────────────────────────────────────────────────
    def test_parse_weather_reads_only_stated_values(self):
        parsed = ctx.parse_weather(WEATHER_PAGE_TEXT)
        self.assertEqual(parsed["wind_min_kmph"], 8)
        self.assertEqual(parsed["wind_max_kmph"], 12)
        self.assertEqual(parsed["humidity_pct"], 68)
        self.assertEqual(parsed["temp_c"], 34)
        self.assertTrue(parsed["fog"])
        self.assertFalse(parsed["rain"])          # the source says "No rain."

    def test_parse_weather_leaves_unknowns_as_none(self):
        parsed = ctx.parse_weather("A page that mentions nothing meteorological today.")
        self.assertIsNone(parsed["wind_min_kmph"])
        self.assertIsNone(parsed["humidity_pct"])
        self.assertIsNone(parsed["temp_c"])
        self.assertFalse(parsed["fog"])
        self.assertFalse(parsed["rain"])

    def test_parse_weather_low_wind_and_negated_fog(self):
        parsed = ctx.parse_weather("Wind speed 5 kmph. Humidity 55%.")
        self.assertEqual(parsed["wind_max_kmph"], 5)
        self.assertFalse(parsed["fog"])
        # explicit negation wording must suppress the flag
        self.assertFalse(ctx.parse_weather("Clear weather, no fog expected.")["fog"])

    # ── adjustment rules ─────────────────────────────────────────────────
    def test_adjust_downgrades_one_step_with_visible_reason(self):
        slots = [
            {"period": "Early Morning", "hours": "5:00 AM – 7:00 AM", "status": "Safe"},
            {"period": "Afternoon Peak", "hours": "12:00 PM – 5:00 PM", "status": "Moderate"},
            {"period": "Evening", "hours": "6:00 PM – 9:00 PM", "status": "Avoid Outdoor"},
        ]
        weather = {"fog": True, "wind_min_kmph": 8, "wind_max_kmph": 12}
        out = ctx.adjust_slots(slots, weather)

        self.assertEqual(out[0]["ml_status"], "Safe")
        self.assertEqual(out[0]["status"], "Moderate")
        self.assertTrue(out[0]["adjusted"])
        self.assertTrue(out[0]["reasons"])
        self.assertEqual(out[1]["status"], "Avoid Outdoor")   # one step only
        self.assertEqual(out[2]["status"], "Avoid Outdoor")   # already worst
        self.assertFalse(out[2]["adjusted"])
        self.assertEqual(out[2]["reasons"], [])

    def test_adjust_low_wind_reason_cites_the_value(self):
        out = ctx.adjust_slots(
            [{"period": "x", "status": "Safe"}],
            {"fog": False, "wind_min_kmph": 5, "wind_max_kmph": 5},
        )
        self.assertEqual(out[0]["status"], "Moderate")
        self.assertIn("Low wind: pollutants trapped", out[0]["reasons"][0])
        self.assertIn("5 km/h", out[0]["reasons"][0])

    def test_adjust_leaves_verdicts_alone_without_triggers(self):
        slots = [{"period": "a", "status": "Safe"}, {"period": "b", "status": "Moderate"}]
        out = ctx.adjust_slots(slots, {"fog": False, "wind_max_kmph": 18})
        self.assertEqual([o["status"] for o in out], ["Safe", "Moderate"])
        self.assertTrue(all(not o["adjusted"] and o["reasons"] == [] for o in out))

    def test_adjust_without_weather_is_identity(self):
        slots = [{"period": "a", "status": "Safe"}]
        out = ctx.adjust_slots(slots, None)
        self.assertEqual(out[0]["status"], "Safe")
        self.assertEqual(out[0]["ml_status"], "Safe")
        self.assertFalse(out[0]["adjusted"])

    def test_adjust_tolerates_malformed_slots(self):
        out = ctx.adjust_slots(["junk", None, {"period": "p", "status": "nonsense"}], None)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["ml_status"], "Safe")   # unknown verdict → Safe

    # ── refresh / cache / failures ───────────────────────────────────────
    def test_refresh_returns_weather_and_notices_with_sources(self):
        with mock.patch.object(tinyfish_service, "search", side_effect=fake_weather_search), \
                mock.patch.object(tinyfish_service, "fetch_many", side_effect=fake_batch(fake_weather_fetch)):
            data = ctx.refresh_weather_windows()

        self.assertEqual(data["weather"]["source"], "IMD")
        self.assertEqual(data["weather"]["humidity_pct"], 68)
        self.assertEqual(data["weather"]["wind_max_kmph"], 12)
        self.assertTrue(data["weather"]["url"].startswith("https://"))
        self.assertEqual(len(data["notices"]), 2)
        kinds = {n["kind"] for n in data["notices"]}
        self.assertEqual(kinds, {"crop-burning", "traffic"})
        for notice in data["notices"]:
            for field in ("source", "url", "fetched_at", "published_at", "summary"):
                self.assertIn(field, notice)
            self.assertLessEqual(len(notice["summary"]), 200)
        self.assertGreaterEqual(len(data["sources"]), 3)

    def test_every_weather_page_costs_one_batched_call(self):
        """Budget guard (spec §5): job C must fit inside the daily cap.

        A fetch per forecast candidate plus a fetch per notice was up to five
        calls — six whenever one of them retried — every 150 minutes, which is
        roughly 130 a day for a single observation. The candidates and the
        notice pages now each travel together, so the whole job is 3 searches
        + 2 fetches however many URLs are read.
        """
        with mock.patch.object(tinyfish_service, "search",
                               side_effect=fake_weather_search) as search, \
                mock.patch.object(tinyfish_service, "fetch_many",
                                  side_effect=fake_batch(fake_weather_fetch)) as batch:
            data = ctx.refresh_weather_windows()

        self.assertTrue(data["weather"])
        self.assertEqual(search.call_count, 3)   # forecast + 2 notice queries
        self.assertEqual(batch.call_count, 2)    # candidates, then notices
        for call in batch.call_args_list:
            urls = list(call.args[0])
            self.assertEqual(len(urls), len(set(urls)))
            self.assertTrue(all(u.startswith("https://") for u in urls))

    def test_timeout_without_cache_keeps_the_ml_windows(self):
        with mock.patch.object(tinyfish_service, "search",
                               side_effect=tinyfish_service.TinyFishError("timed out")):
            payload = ctx.get_weather_windows([{"period": "Early Morning", "status": "Safe"}])

        self.assertEqual(payload["status"], "unavailable")
        self.assertIsNone(payload["weather"])
        self.assertEqual(payload["notices"], [])
        # the ML verdicts still come back untouched — never a blank card
        self.assertEqual(payload["slots"][0]["ml_status"], "Safe")
        self.assertEqual(payload["slots"][0]["status"], "Safe")
        self.assertEqual(payload["slots"][0]["reasons"], [])

    def test_malformed_search_results_do_not_raise(self):
        with mock.patch.object(tinyfish_service, "search",
                               return_value=[None, "junk", {"url": ""}]), \
                mock.patch.object(tinyfish_service, "fetch_many", return_value={"results": [], "errors": []}):
            payload = ctx.get_weather_windows([{"period": "Evening", "status": "Moderate"}])
        self.assertEqual(payload["status"], "unavailable")
        self.assertEqual(payload["slots"][0]["status"], "Moderate")

    def test_stale_forecast_is_served_when_refresh_fails(self):
        good = {"weather": {"source": "IMD", "url": WEATHER_URL,
                            "fetched_at": "2026-10-10T06:00:00", "text": "Foggy.",
                            "wind_min_kmph": 3, "wind_max_kmph": 5,
                            "humidity_pct": 90, "temp_c": 22, "fog": True, "rain": False},
                "notices": [], "sources": []}
        ctx._cache["weather_windows"] = {"data": good, "stored_at": time.time()}
        with mock.patch.object(tinyfish_service, "search",
                               side_effect=tinyfish_service.TinyFishError("HTTP 429")):
            payload = ctx.get_weather_windows([{"period": "a", "status": "Safe"}], force=True)

        self.assertEqual(payload["status"], "stale")
        self.assertTrue(payload["stale"])
        self.assertEqual(payload["weather"]["humidity_pct"], 90)
        # the stale forecast still drives a visible, reasoned downgrade
        self.assertEqual(payload["slots"][0]["status"], "Moderate")
        self.assertTrue(payload["slots"][0]["reasons"])

    # ── live-search quality (regression: values seen against mausam.imd.gov.in)
    def test_parse_weather_reads_the_imd_table_without_decimal_drift(self):
        parsed = ctx.parse_weather(IMD_TABLE_TEXT)
        # "9.3 km/h" must not become 3 km/h, "30.8 °C" must not become 8 °C
        self.assertEqual(parsed["temp_c"], 30.8)
        self.assertEqual(parsed["wind_min_kmph"], 9.3)
        self.assertEqual(parsed["wind_max_kmph"], 9.3)
        self.assertEqual(parsed["humidity_pct"], 67)
        self.assertFalse(parsed["fog"])
        self.assertFalse(parsed["rain"])
        # 9.3 km/h is not low wind → a good forecast changes no verdict
        out = ctx.adjust_slots([{"period": "a", "status": "Safe"}], parsed)
        self.assertFalse(out[0]["adjusted"])
        self.assertEqual(out[0]["status"], "Safe")
        self.assertEqual(out[0]["reasons"], [])

    def test_homepage_result_ranks_below_the_forecast_page(self):
        ranked = ctx._weather_candidates([
            IMD_HOME_RESULT, IMD_TABLE_RESULT,
            {"url": "https://totally-random-blog.example.com/x"},   # not allowlisted
        ])
        self.assertEqual(
            [result["url"] for result in ranked],
            [IMD_TABLE_RESULT["url"], IMD_HOME_RESULT["url"]],
        )

    def test_refresh_reads_the_forecast_page_not_the_homepage(self):
        def search(query, **kwargs):
            if "weather" in query.lower():
                return [IMD_HOME_RESULT, IMD_TABLE_RESULT]   # homepage first, as live
            return NOTICE_RESULTS

        def fetch(url, **kwargs):
            if "index_en.php" in url:
                return {**IMD_HOME_RESULT, "url": url, "text": IMD_HOME_TEXT,
                        "published_date": ""}
            return {**IMD_TABLE_RESULT, "url": url, "text": IMD_TABLE_TEXT,
                    "published_date": ""}

        with mock.patch.object(tinyfish_service, "search", side_effect=search), \
                mock.patch.object(tinyfish_service, "fetch_many", side_effect=fake_batch(fetch)):
            data = ctx.refresh_weather_windows()

        self.assertEqual(data["weather"]["url"], IMD_TABLE_RESULT["url"])
        self.assertEqual(data["weather"]["temp_c"], 30.8)
        self.assertEqual(data["weather"]["humidity_pct"], 67)
        self.assertEqual(data["weather"]["wind_max_kmph"], 9.3)
        # the display text stops before the unpublished "NA" days
        self.assertNotIn("NA °C", data["weather"]["text"])
        self.assertTrue(data["weather"]["text"].startswith("Oct 10, 2026"))

    def test_notices_older_than_the_active_window_are_dropped(self):
        news = [
            {"title": "Delhi stubble burning rises this week",
             "url": "https://www.ndtv.com/delhi-news/stubble-2026",
             "snippet": "Farm residue burning increased sharply in Punjab this week, officials said.",
             "site_name": "ndtv.com", "publisher": "NDTV", "date": "2 days ago"},
            {"title": "GRAP-IV restrictions continue till December",
             "url": "https://www.thehindu.com/news/grap-iv-2024",
             "snippet": "The restrictions on construction and trucks continue in the region.",
             "site_name": "thehindu.com", "publisher": "The Hindu", "date": "2024-11-28"},
        ]

        def search(query, **kwargs):
            return [WEATHER_RESULT] if "weather" in query.lower() else news

        def fetch(url, **kwargs):
            if "mausam" in url:
                return {**WEATHER_RESULT, "url": url, "text": WEATHER_PAGE_TEXT,
                        "published_date": ""}
            if "thehindu" in url:
                return {"url": url, "title": "GRAP-IV restrictions continue till December",
                        "site_name": "thehindu.com", "published_date": "2024-11-28",
                        "text": "The restrictions on construction and trucks continue "
                                "in the region."}
            return {**NOTICE_PAGE, "url": url,
                    "title": "Delhi stubble burning rises this week",
                    "published_date": "2026-10-09"}

        with mock.patch.object(tinyfish_service, "search", side_effect=search), \
                mock.patch.object(tinyfish_service, "fetch_many", side_effect=fake_batch(fetch)):
            data = ctx.refresh_weather_windows()

        urls = [notice["url"] for notice in data["notices"]]
        self.assertIn("https://www.ndtv.com/delhi-news/stubble-2026", urls)
        self.assertNotIn("https://www.thehindu.com/news/grap-iv-2024", urls)

    def test_weather_ttl_is_registered_and_scheduler_respects_it(self):
        self.assertEqual(ctx._REGISTRY["weather_windows"]["ttl"], 6 * 3600)
        self.assertEqual(ctx._REGISTRY["alerts"]["ttl"], 20 * 60)

        refresher = mock.Mock(return_value={"x": 1})
        with mock.patch.dict(ctx._REGISTRY, {"demo": {"ttl": 60, "refresh": refresher}}):
            ctx._cache["demo"] = {"data": {"x": 1}, "stored_at": time.time()}
            ctx.refresh_warm_features()               # fresh → not due
            refresher.assert_not_called()
            ctx._cache["demo"]["stored_at"] = time.time() - 120
            ctx.refresh_warm_features()               # past TTL → refreshed
            refresher.assert_called_once()


class WeatherEndpointTests(unittest.TestCase):
    def setUp(self):
        fresh()
        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)

    def tearDown(self):
        fresh()

    def test_post_weather_windows_returns_envelope_and_adjusted_slots(self):
        body = {"slots": [
            {"period": "Early Morning", "hours": "5:00 AM – 7:00 AM", "status": "Safe"},
            {"period": "Evening", "hours": "6:00 PM – 9:00 PM", "status": "Avoid Outdoor"},
        ]}
        with mock.patch.object(tinyfish_service, "search", side_effect=fake_weather_search), \
                mock.patch.object(tinyfish_service, "fetch_many", side_effect=fake_batch(fake_weather_fetch)):
            res = self.client.post("/api/context/weather-windows", json=body)

        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("fetched_at", payload)
        self.assertIn("ttl_seconds", payload)
        self.assertTrue(payload["sources"])
        self.assertEqual(payload["weather"]["source"], "IMD")
        self.assertEqual(payload["slots"][0]["ml_status"], "Safe")
        self.assertEqual(payload["slots"][0]["status"], "Moderate")  # fog in forecast
        self.assertTrue(payload["slots"][0]["reasons"])
        self.assertEqual(payload["slots"][1]["status"], "Avoid Outdoor")

    def test_post_with_malformed_body_still_returns_200(self):
        with mock.patch.object(tinyfish_service, "search",
                               side_effect=tinyfish_service.TinyFishError("timed out")):
            res = self.client.post("/api/context/weather-windows",
                                   json={"slots": "not-a-list"})
        self.assertEqual(res.status_code, 200)
        self.assertIn("slots", res.json())

    def test_empty_body_is_accepted(self):
        with mock.patch.object(tinyfish_service, "search",
                               side_effect=tinyfish_service.TinyFishError("timed out")):
            res = self.client.post("/api/context/weather-windows")
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertEqual(payload["slots"], [])
        self.assertEqual(payload["status"], "unavailable")


if __name__ == "__main__":
    unittest.main(verbosity=2)
