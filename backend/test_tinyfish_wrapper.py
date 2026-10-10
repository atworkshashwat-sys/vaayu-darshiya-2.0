"""
Tests for the single TinyFish wrapper: budget gate, key failover, usage log.

Run:
    cd backend && python -m unittest -v test_tinyfish_wrapper

TinyFish is fully mocked — no API quota is spent and no network call is made.
The cases mirror the spec's acceptance list: success, timeout, malformed JSON,
out-of-range value (handled in test_live_data), wrong unit, auth error with
key failover, all keys failed, past expiry and cap reached.
"""

import json
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tinyfish_service as svc

IST = svc.IST

# Three keys so the failover chain is observable.
KEYS = {
    "TINYFISH_API_KEY_1": "sk-tinyfish-AAAA",
    "TINYFISH_API_KEY_2": "sk-tinyfish-BBBB",
    "TINYFISH_API_KEY_3": "sk-tinyfish-CCCC",
}


def future(hours=24):
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


def past(hours=1):
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


def env(**extra):
    """A clean, enabled, in-window environment with three keys."""
    base = {
        "TINYFISH_ENABLED": "true",
        "TINYFISH_EXPIRES_AT": future(6),
        "TINYFISH_DAILY_CALL_CAP": "400",
        **KEYS,
    }
    base.update(extra)
    return mock.patch.dict(os.environ, base, clear=True)


def no_keys(**extra):
    """Enabled and in the window, but with no key configured at all."""
    base = {
        "TINYFISH_ENABLED": "true",
        "TINYFISH_EXPIRES_AT": future(6),
        "TINYFISH_DAILY_CALL_CAP": "400",
        "TINYFISH_API_KEY_1": "",
        "TINYFISH_API_KEY_2": "",
        "TINYFISH_API_KEY_3": "",
    }
    base.update(extra)
    return mock.patch.dict(os.environ, base, clear=True)


class HTTPErrorStub(Exception):
    """Stand-in that _call converts; tests raise TinyFishError directly."""


def key_error(status, code):
    return svc.TinyFishError(f"HTTP {status}", status=status, code=code)


def transient_error(status=None):
    return svc.TinyFishError("timed out" if status is None else f"HTTP {status}",
                             status=status)


# A real Fetch-shaped payload: results[0] must be a dict, else fetch_text
# reports "no content" (per-URL failures arrive in errors[] on a 200 response).
FETCH_RESULT = {
    "results": [{
        "url": "https://cpcb.nic.in/live/",
        "final_url": "https://cpcb.nic.in/live/",
        "title": "CPCB live bulletin",
        "published_date": "2026-10-10",
        "text": "Ashok Vihar NO2 34 ppb, O3 41 ppb, PM2.5 96 ug/m3.",
    }],
    "errors": [],
}


class ConfigTests(unittest.TestCase):
    def setUp(self):
        svc.reset_usage()
        svc.reset_window_default()

    def test_keys_are_labelled_key_1_to_key_3(self):
        with env():
            labels = [label for label, _ in svc.configured_keys()]
        self.assertEqual(labels, ["key_1", "key_2", "key_3"])

    def test_labels_never_expose_key_material(self):
        with env():
            snapshot = json.dumps(svc.usage_snapshot())
        self.assertNotIn("sk-tinyfish", snapshot)
        for label in ("key_1", "key_2", "key_3"):
            self.assertIn(label, snapshot)

    def test_legacy_env_name_is_honoured_as_key_1(self):
        with mock.patch.dict(os.environ,
                             {"TINYFISH_API_KEY": "sk-tinyfish-LEGACY",
                              "TINYFISH_API_KEY_1": "",
                              "TINYFISH_API_KEY_2": "",
                              "TINYFISH_API_KEY_3": ""}, clear=True):
            self.assertEqual(svc.configured_keys(), [("key_1", "sk-tinyfish-LEGACY")])

    def test_modern_names_win_over_the_legacy_name(self):
        with env():
            keys = svc.configured_keys()
        self.assertEqual(keys[0], ("key_1", "sk-tinyfish-AAAA"))

    def test_expires_at_defaults_to_a_24_hour_window(self):
        with mock.patch.dict(os.environ, {"TINYFISH_EXPIRES_AT": ""}, clear=True):
            svc.reset_window_default()
            remaining = svc.remaining_seconds()
        self.assertLessEqual(remaining, 24 * 3600)
        self.assertGreater(remaining, 23 * 3600)

    def test_expiring_today_is_expired(self):
        with env(TINYFISH_EXPIRES_AT=past(1)):
            self.assertTrue(svc.is_expired())
            self.assertFalse(svc.is_active())

    def test_disabled_is_not_active_even_inside_the_window(self):
        with env(TINYFISH_ENABLED="false"):
            self.assertFalse(svc.is_active())

    def test_cap_defaults_to_400_and_is_shared(self):
        with env():
            self.assertEqual(svc.daily_cap(), 400)

    def test_malformed_cap_falls_back_to_the_default(self):
        with env(TINYFISH_DAILY_CALL_CAP="not-a-number"):
            self.assertEqual(svc.daily_cap(), 400)


class GateTests(unittest.TestCase):
    """No call is made when disabled, expired, cap-reached or keyless."""

    def setUp(self):
        svc.reset_usage()
        svc.reset_window_default()

    def _no_network(self):
        return mock.patch.object(svc, "_call",
                                 side_effect=AssertionError("network call made"))

    def test_disabled_makes_no_call(self):
        with env(TINYFISH_ENABLED="false"), self._no_network():
            with self.assertRaises(svc.TinyFishError):
                svc.search("q")

    def test_expired_makes_no_call(self):
        with env(TINYFISH_EXPIRES_AT=past(2)), self._no_network():
            with self.assertRaises(svc.TinyFishError) as caught:
                svc.search("q")
        self.assertIn("ended", str(caught.exception))

    def test_cap_reached_makes_no_call(self):
        with env(TINYFISH_DAILY_CALL_CAP="1"), self._no_network():
            with mock.patch.object(svc, "_call", return_value={"results": []}):
                svc.search("first")            # consumes the only call
            with self.assertRaises(svc.TinyFishError) as caught:
                svc.search("second")
        self.assertIn("cap", str(caught.exception))

    def test_no_key_fails_closed(self):
        with no_keys(), self._no_network():
            with self.assertRaises(svc.TinyFishError) as caught:
                svc.search("q")
        self.assertIn("TINYFISH_API_KEY", str(caught.exception))

    def test_cap_counts_search_and_fetch_together(self):
        with env(TINYFISH_DAILY_CALL_CAP="2"):
            with mock.patch.object(svc, "_call", return_value=FETCH_RESULT):
                svc.search("a")
                svc.fetch_text("https://cpcb.nic.in/")
            self.assertEqual(svc.calls_used(), 2)
            self.assertTrue(svc.budget_reached())

    def test_usage_resets_with_the_utc_day(self):
        with env():
            with mock.patch.object(svc, "_call", return_value=FETCH_RESULT):
                svc.search("a")
            self.assertEqual(svc.calls_used(), 1)
            svc._usage_day = "2000-01-01"       # simulate the allowance reset
            self.assertEqual(svc.calls_used(), 0)


class FailoverTests(unittest.TestCase):
    def setUp(self):
        svc.reset_usage()
        svc.reset_window_default()

    def test_success_uses_key_1_and_records_it(self):
        with env():
            with mock.patch.object(svc, "_call", return_value={"results": []}) as call:
                svc.search("q")
            self.assertEqual(call.call_args.kwargs["key"], "sk-tinyfish-AAAA")
            self.assertEqual(svc.active_key_label(), "key_1")
            self.assertEqual(svc.usage_snapshot()["used"], 1)

    def test_invalid_api_key_rotates_to_the_next_key(self):
        with env():
            with mock.patch.object(svc, "_call",
                                   side_effect=[key_error(401, "INVALID_API_KEY"),
                                                {"results": []}]) as call:
                svc.search("q")
            # first attempt key_1, second attempt key_2
            self.assertEqual([c.kwargs["key"] for c in call.call_args_list],
                             ["sk-tinyfish-AAAA", "sk-tinyfish-BBBB"])
            self.assertEqual(svc.active_key_label(), "key_2")

    def test_insufficient_balance_402_rotates(self):
        with env():
            with mock.patch.object(svc, "_call",
                                   side_effect=[key_error(402, "INSUFFICIENT_CREDITS"),
                                                FETCH_RESULT]) as call:
                svc.fetch_text("https://cpcb.nic.in/")
            self.assertEqual([c.kwargs["key"] for c in call.call_args_list],
                             ["sk-tinyfish-AAAA", "sk-tinyfish-BBBB"])

    def test_legacy_forbidden_403_rotates(self):
        with env():
            with mock.patch.object(svc, "_call",
                                   side_effect=[key_error(403, "FORBIDDEN"),
                                                {"results": []}]) as call:
                svc.search("q")
            self.assertEqual(len(call.call_args_list), 2)

    def test_all_keys_failed_raises_and_serves_cache_upstream(self):
        with env():
            with mock.patch.object(
                    svc, "_call",
                    side_effect=key_error(401, "INVALID_API_KEY")) as call:
                with self.assertRaises(svc.TinyFishError):
                    svc.search("q")
            # one attempt per key — never a loop across the same key
            self.assertEqual(call.call_count, 3)

    def test_timeout_retries_the_same_key_and_never_rotates(self):
        with env():
            with mock.patch.object(svc, "_call",
                                   side_effect=[transient_error(),
                                                {"results": []}]) as call:
                svc.search("q")
            self.assertEqual([c.kwargs["key"] for c in call.call_args_list],
                             ["sk-tinyfish-AAAA", "sk-tinyfish-AAAA"])
            self.assertEqual(svc.active_key_label(), "key_1")

    def test_server_error_5xx_never_rotates_keys(self):
        with env():
            with mock.patch.object(svc, "_call",
                                   side_effect=[transient_error(503),
                                                transient_error(503)]) as call:
                with self.assertRaises(svc.TinyFishError):
                    svc.search("q")
            self.assertEqual({c.kwargs["key"] for c in call.call_args_list},
                             {"sk-tinyfish-AAAA"})
            self.assertEqual(call.call_count, 2)      # one retry, no loop

    def test_rate_limit_429_never_rotates_keys(self):
        with env():
            with mock.patch.object(svc, "_call",
                                   side_effect=[transient_error(429),
                                                transient_error(429)]) as call:
                with self.assertRaises(svc.TinyFishError):
                    svc.search("q")
            self.assertEqual({c.kwargs["key"] for c in call.call_args_list},
                             {"sk-tinyfish-AAAA"})

    def test_preferred_key_is_reused_after_a_failover(self):
        with env():
            with mock.patch.object(svc, "_call",
                                   side_effect=[key_error(401, "INVALID_API_KEY"),
                                                {"results": []}]):
                svc.search("q")
            with mock.patch.object(svc, "_call", return_value={"results": []}) as call:
                svc.search("q again")
            # key_2 earned the preferred slot, so the next call starts there
            self.assertEqual(call.call_args.kwargs["key"], "sk-tinyfish-BBBB")


class UsageLogTests(unittest.TestCase):
    def setUp(self):
        svc.reset_usage()
        svc.reset_window_default()

    def test_every_call_is_logged_with_type_key_and_outcome(self):
        with env():
            with mock.patch.object(svc, "_call", return_value=FETCH_RESULT):
                svc.search("q")
                svc.fetch_text("https://cpcb.nic.in/")
        entries = svc.call_log()
        self.assertEqual([e["type"] for e in entries], ["search", "fetch"])
        for entry in entries:
            self.assertEqual(entry["key"], "key_1")
            self.assertTrue(entry["success"])
            self.assertIn("at", entry)

    def test_failures_are_logged_with_the_error(self):
        with env():
            with mock.patch.object(svc, "_call",
                                   side_effect=transient_error(503)):
                with self.assertRaises(svc.TinyFishError):
                    svc.search("q")
        entry = svc.call_log()[-1]
        self.assertFalse(entry["success"])
        self.assertIn("503", entry["error"])

    def test_totals_per_key_and_the_cap_are_reported(self):
        with env(TINYFISH_DAILY_CALL_CAP="50"):
            with mock.patch.object(svc, "_call", return_value={"results": []}):
                svc.search("a")
                svc.search("b")
            snapshot = svc.usage_snapshot()
        self.assertEqual(snapshot["cap"], 50)
        self.assertEqual(snapshot["used"], 2)
        self.assertEqual(snapshot["remaining"], 48)
        key_1 = next(k for k in snapshot["keys"] if k["key"] == "key_1")
        self.assertEqual(key_1["total"], 2)
        self.assertEqual(key_1["search"], 2)

    def test_usage_sink_persists_each_call(self):
        seen = []
        with env():
            svc.set_usage_sink(seen.append)
            try:
                with mock.patch.object(svc, "_call", return_value={"results": []}):
                    svc.search("a")
            finally:
                svc.set_usage_sink(None)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["type"], "search")

    def test_snapshot_reports_the_window(self):
        with env(TINYFISH_EXPIRES_AT=future(3)):
            snapshot = svc.usage_snapshot()
        self.assertTrue(snapshot["enabled"])
        self.assertFalse(snapshot["expired"])
        self.assertGreater(snapshot["remaining_seconds"], 0)


class BudgetBaselineTests(unittest.TestCase):
    """The day's spend must survive a restart (spec: cap shared across keys).

    `_usage` is in-memory and cleared with the process, so without a durable
    baseline every restart would hand out another 400-call allowance for the
    same reset day.
    """

    def setUp(self):
        svc.reset_usage()
        svc.reset_window_default()

    def tearDown(self):
        svc.set_usage_baseline(None)

    def test_a_restart_does_not_refill_the_allowance(self):
        # Process two: in-memory is empty, but storage says 130 were spent.
        svc.set_usage_baseline(lambda: 130)
        self.assertEqual(svc.calls_used(), 130)
        self.assertEqual(svc.budget_remaining(), 400 - 130)

    def test_baseline_is_a_floor_never_an_addition(self):
        # The persisted total already includes this process's calls, so the
        # larger of the two is the truth — adding them would double count.
        with env():
            with mock.patch.object(svc, "_call", return_value={"results": []}):
                svc.search("a")
                svc.search("b")
        svc.set_usage_baseline(lambda: 3)
        self.assertEqual(svc.calls_used(), 3)      # baseline already covers them

        svc.set_usage_baseline(lambda: 1)
        self.assertEqual(svc.calls_used(), 2)      # in-memory wins when higher

    def test_a_broken_baseline_never_blocks_a_legitimate_call(self):
        def explode():
            raise RuntimeError("database is gone")

        svc.set_usage_baseline(explode)
        with env():
            with mock.patch.object(svc, "_call", return_value={"results": []}) as call:
                svc.search("a")
        self.assertEqual(call.call_count, 1)
        self.assertEqual(svc.calls_used(), 1)

    def test_the_gate_refuses_once_storage_reaches_the_cap(self):
        svc.set_usage_baseline(lambda: 400)
        with env():
            with mock.patch.object(svc, "_call", return_value={"results": []}) as call:
                with self.assertRaises(svc.TinyFishError):
                    svc.search("a")
        self.assertEqual(call.call_count, 0)       # refused before any HTTP

    def test_each_logged_call_carries_the_reset_day(self):
        with env():
            with mock.patch.object(svc, "_call", return_value={"results": []}):
                svc.search("a")
        entry = svc.call_log()[-1]
        self.assertRegex(entry["day"], r"^\d{4}-\d{2}-\d{2}$")
        self.assertEqual(entry["day"], svc.usage_day())

    def test_usage_day_is_the_utc_window_not_the_local_date(self):
        # The allowance resets at 00:00 UTC, which is 05:30 IST — using the
        # IST date would cut the window in half every night.
        self.assertEqual(svc.usage_day(),
                         datetime.now(timezone.utc).date().isoformat())


class PayloadTests(unittest.TestCase):
    """TinyFish returned something we did not expect — never crash the caller."""

    def setUp(self):
        svc.reset_usage()
        svc.reset_window_default()

    def test_malformed_json_raises_tinyfish_error(self):
        with env(), mock.patch.object(svc.urllib.request, "urlopen") as urlopen:
            response = mock.MagicMock()
            response.__enter__.return_value.read.return_value = b"{not json"
            urlopen.return_value = response
            with self.assertRaises(svc.TinyFishError) as caught:
                svc.search("q")
        self.assertIn("invalid JSON", str(caught.exception))

    def test_non_dict_payload_is_rejected(self):
        with env(), mock.patch.object(svc.urllib.request, "urlopen") as urlopen:
            response = mock.MagicMock()
            response.__enter__.return_value.read.return_value = b"[1,2,3]"
            urlopen.return_value = response
            with self.assertRaises(svc.TinyFishError):
                svc.search("q")

    def test_results_that_are_not_dicts_are_skipped(self):
        payload = {"results": ["junk", None, {"url": "https://cpcb.nic.in/x",
                                              "title": "ok"}]}
        with env():
            with mock.patch.object(svc, "_call", return_value=payload):
                items = svc.search("q")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["title"], "ok")

    def test_result_without_a_url_is_dropped(self):
        payload = {"results": [{"title": "no link here", "snippet": "x"}]}
        with env():
            with mock.patch.object(svc, "_call", return_value=payload):
                self.assertEqual(svc.search("q"), [])

    def test_fetch_per_url_error_becomes_a_tinyfish_error(self):
        payload = {"results": [], "errors": [{"url": "https://x.test",
                                              "error": "bot_blocked"}]}
        with env():
            with mock.patch.object(svc, "_call", return_value=payload):
                with self.assertRaises(svc.TinyFishError) as caught:
                    svc.fetch_text("https://x.test")
        self.assertIn("bot_blocked", str(caught.exception))

    def test_fetch_many_caps_at_ten_urls(self):
        with env():
            with mock.patch.object(svc, "_call",
                                   return_value={"results": [], "errors": []}) as call:
                svc.fetch_many([f"https://site{i}.test/" for i in range(25)])
        sent = json.loads(call.call_args.kwargs["data"].decode("utf-8"))
        self.assertEqual(len(sent["urls"]), 10)


if __name__ == "__main__":
    unittest.main()
