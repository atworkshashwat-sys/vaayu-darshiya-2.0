"""
Targeted tests for the AI Insights TinyFish integration only.

Run:
    cd backend && python -m unittest -v test_ai_insights

stdlib `unittest` — pytest is not a project dependency. Nothing here touches
the ML pipeline; the service calls are mocked so tests never spend API quota.
"""

import os
import sys
import unittest
import urllib.error
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tinyfish_service
from ai_insights import router
from fastapi import FastAPI
from fastapi.testclient import TestClient

# All three key slots blank: the wrapper must fail closed (no call, no crash).
NO_KEYS = {
    "TINYFISH_API_KEY_1": "",
    "TINYFISH_API_KEY_2": "",
    "TINYFISH_API_KEY_3": "",
}

SEARCH_PAYLOAD = {
    "query": "Delhi air quality pollution news and official advisory",
    "results": [
        {
            "position": 1,
            "site_name": "example.com",
            "publisher": "Example Times",
            "date": "2 days ago",
            "title": "CAQM issues advisory for Delhi NCR",
            "snippet": "Schools asked to avoid outdoor sports.",
            "url": "https://example.com/advisory",
        },
        {"position": 2, "title": "Dropped — no url", "url": ""},
    ],
    "total_results": 2,
    "page": 0,
}

FETCH_PAYLOAD = {
    "results": [{
        "url": "https://example.com/advisory",
        "final_url": "https://example.com/advisory",
        "title": "CAQM issues advisory for Delhi NCR",
        "text": "# CAQM issues advisory\n\nThe commission directed schools to "
                "avoid outdoor sports. [Read more](https://example.com/more)",
    }],
    "errors": [],
}


class TinyFishServiceTests(unittest.TestCase):
    def test_search_without_key_fails_closed(self):
        with mock.patch.dict(os.environ, NO_KEYS, clear=True):
            with self.assertRaises(tinyfish_service.TinyFishError):
                tinyfish_service.search("Delhi air quality")

    def test_api_key_read_from_environment_only(self):
        with mock.patch.dict(os.environ, {"TINYFISH_API_KEY_1": " sk-test-123 "},
                             clear=True):
            self.assertEqual(tinyfish_service.api_key(), "sk-test-123")
        with mock.patch.dict(os.environ, NO_KEYS, clear=True):
            self.assertIsNone(tinyfish_service.api_key())

    def test_search_maps_results_and_skips_unlinkable_ones(self):
        with mock.patch.dict(os.environ, {"TINYFISH_API_KEY_1": "sk-test"}, clear=True), \
                mock.patch.object(tinyfish_service, "_call", return_value=SEARCH_PAYLOAD):
            items = tinyfish_service.search("Delhi air quality")
        self.assertEqual(len(items), 1)  # the url-less result is dropped
        self.assertEqual(items[0]["title"], "CAQM issues advisory for Delhi NCR")
        self.assertEqual(items[0]["url"], "https://example.com/advisory")
        self.assertEqual(items[0]["publisher"], "Example Times")

    def test_http_error_becomes_tinyfish_error(self):
        def raise_http(*args, **kwargs):
            raise urllib.error.HTTPError("https://api.search.tinyfish.ai", 429, "Too Many Requests", None, None)

        with mock.patch.dict(os.environ, {"TINYFISH_API_KEY": "sk-test"}), \
                mock.patch.object(tinyfish_service.urllib.request, "urlopen", side_effect=raise_http):
            with self.assertRaises(tinyfish_service.TinyFishError) as ctx:
                tinyfish_service.search("Delhi air quality")
        self.assertIn("429", str(ctx.exception))

    def test_fetch_text_returns_clean_excerpt(self):
        with mock.patch.dict(os.environ, {"TINYFISH_API_KEY": "sk-test"}), \
                mock.patch.object(tinyfish_service, "_call", return_value=FETCH_PAYLOAD):
            page = tinyfish_service.fetch_text("https://example.com/advisory")
        self.assertEqual(page["site_name"], "example.com")
        self.assertIn("The commission directed schools", page["text"])
        self.assertNotIn("#", page["text"])       # heading marker stripped
        self.assertNotIn("[Read more]", page["text"])  # link label kept, markdown gone

    def test_fetch_text_without_key_fails_closed(self):
        with mock.patch.dict(os.environ, {"TINYFISH_API_KEY": ""}):
            with self.assertRaises(tinyfish_service.TinyFishError):
                tinyfish_service.fetch_text("https://example.com/advisory")


class ResearchRouteTests(unittest.TestCase):
    """HTTP-level tests of GET /api/ai-insights/research on its own router."""

    def setUp(self):
        import ai_insights
        ai_insights._cache.clear()
        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)

    def tearDown(self):
        import ai_insights
        ai_insights._cache.clear()

    def test_ok_response_carries_items_and_excerpt(self):
        with mock.patch.object(tinyfish_service, "search", return_value=[SEARCH_PAYLOAD["results"][0]]), \
                mock.patch.object(tinyfish_service, "fetch_text", return_value={
                    "url": "https://example.com/advisory",
                    "title": "CAQM issues advisory for Delhi NCR",
                    "site_name": "example.com",
                    "text": "The commission directed schools to avoid outdoor sports.",
                }):
            res = self.client.get("/api/ai-insights/research", params={"city": "Delhi"})

        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["cached"], False)
        self.assertEqual(body["city"], "Delhi")
        self.assertEqual(len(body["items"]), 1)
        self.assertEqual(body["items"][0]["url"], "https://example.com/advisory")
        self.assertIn("commission directed", body["excerpt"]["text"])

    def test_second_call_is_served_from_cache(self):
        with mock.patch.object(tinyfish_service, "search", return_value=[SEARCH_PAYLOAD["results"][0]]) as search, \
                mock.patch.object(tinyfish_service, "fetch_text", return_value=None):
            first = self.client.get("/api/ai-insights/research").json()
            second = self.client.get("/api/ai-insights/research").json()

        self.assertEqual(first["cached"], False)
        self.assertEqual(second["cached"], True)
        self.assertEqual(search.call_count, 1)

    def test_tinyfish_failure_returns_200_unavailable(self):
        with mock.patch.object(
            tinyfish_service, "search",
            side_effect=tinyfish_service.TinyFishError("HTTP 429"),
        ):
            res = self.client.get("/api/ai-insights/research")

        self.assertEqual(res.status_code, 200)  # never breaks the panel
        body = res.json()
        self.assertEqual(body["status"], "unavailable")
        self.assertEqual(body["items"], [])
        self.assertIsNone(body["excerpt"])
        self.assertIn("429", body["message"])

    def test_excerpt_failure_still_returns_search_results(self):
        with mock.patch.object(tinyfish_service, "search", return_value=[SEARCH_PAYLOAD["results"][0]]), \
                mock.patch.object(
                    tinyfish_service, "fetch_text",
                    side_effect=tinyfish_service.TinyFishError("timeout"),
                ):
            body = self.client.get("/api/ai-insights/research").json()

        self.assertEqual(body["status"], "ok")
        self.assertEqual(len(body["items"]), 1)
        self.assertIsNone(body["excerpt"])

    def test_empty_news_search_retries_on_the_web(self):
        with mock.patch.object(tinyfish_service, "search", side_effect=[[], [SEARCH_PAYLOAD["results"][0]]]) as search, \
                mock.patch.object(tinyfish_service, "fetch_text", return_value=None):
            body = self.client.get("/api/ai-insights/research").json()

        self.assertEqual(body["status"], "ok")
        self.assertEqual(search.call_count, 2)
        self.assertEqual(search.call_args_list[1].kwargs.get("domain_type"), "web")


class MountingTests(unittest.TestCase):
    def test_api_mounts_the_router_only_for_ai_insights(self):
        with open(os.path.join(os.path.dirname(__file__), "api.py"), encoding="utf-8") as fh:
            source = fh.read()
        self.assertIn("app.include_router(ai_insights_router)", source)
        self.assertIn("from ai_insights import router as ai_insights_router", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
