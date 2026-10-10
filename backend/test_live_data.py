"""
Tests for the live-data layer: validation, extraction and the SQLite store.

Run:
    cd backend && python -m unittest -v test_live_data

TinyFish is fully mocked. These cover the spec's §3 acceptance list: unit and
range checks, IST timestamps, the 3-hour staleness rule, the 30% conflict rule
and "never overwrite an ML forecast value".
"""

import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import live_data
import live_store
import live_validate
from live_validate import Rejected

IST = timezone(timedelta(hours=5, minutes=30))
NOW = datetime(2026, 10, 10, 18, 0, tzinfo=IST)


class UnitConversionTests(unittest.TestCase):
    """Documented molar factor: ppb = µg/m³ × 24.45 / M."""

    def test_no2_micrograms_to_ppb(self):
        # 46.01 µg/m³ of NO2 = 24.45 ppb exactly (one molar volume).
        value, unit = live_validate.to_ppb("NO2", 46.01, "ug/m3")
        self.assertEqual(unit, "ppb")
        self.assertAlmostEqual(value, 24.45, places=1)

    def test_o3_micrograms_to_ppb(self):
        value, unit = live_validate.to_ppb("O3", 48.0, "ug/m3")
        self.assertEqual(unit, "ppb")
        self.assertAlmostEqual(value, 24.45, places=1)

    def test_ppb_passes_through_unchanged(self):
        value, unit = live_validate.to_ppb("NO2", 42, "ppb")
        self.assertEqual((value, unit), (42.0, "ppb"))

    def test_unicode_microgram_spellings_are_accepted(self):
        for spelling in ("µg/m³", "μg/m3", "ug/m³", "µg/m3"):
            _value, unit = live_validate.to_ppb("NO2", 24.45, spelling)
            self.assertEqual(unit, "ppb")

    def test_unknown_unit_is_rejected_not_guessed(self):
        with self.assertRaises(Rejected):
            live_validate.to_ppb("NO2", 42, "furlongs")

    def test_conversion_factor_is_documented_per_pollutant(self):
        self.assertAlmostEqual(live_validate.PPB_PER_UG["NO2"], 1.882, places=3)
        self.assertAlmostEqual(live_validate.PPB_PER_UG["O3"], 1.963, places=3)


class RangeTests(unittest.TestCase):
    def test_no2_within_range_is_accepted(self):
        row = live_validate.validate("NO2", 120, "ug/m3", NOW, "CPCB", now=NOW)
        self.assertEqual(row["unit"], "ppb")
        self.assertAlmostEqual(row["value"], 63.76, places=1)

    def test_no2_above_500_ppb_is_rejected(self):
        # 1000 µg/m³ → 531 ppb, above the 500 ppb ceiling.
        with self.assertRaises(Rejected) as caught:
            live_validate.validate("NO2", 1000, "ug/m3", NOW, "CPCB", now=NOW)
        self.assertIn("outside", str(caught.exception))

    def test_o3_above_300_ppb_is_rejected(self):
        with self.assertRaises(Rejected):
            live_validate.validate("O3", 700, "ug/m3", NOW, "CPCB", now=NOW)

    def test_pm25_bounds(self):
        live_validate.validate("PM25", 999, "ug/m3", NOW, "CPCB", now=NOW)
        with self.assertRaises(Rejected):
            live_validate.validate("PM25", 1001, "ug/m3", NOW, "CPCB", now=NOW)

    def test_aqi_bounds(self):
        live_validate.validate("AQI", 500, "index", NOW, "CPCB", now=NOW)
        with self.assertRaises(Rejected):
            live_validate.validate("AQI", 501, "index", NOW, "CPCB", now=NOW)

    def test_negative_is_rejected(self):
        with self.assertRaises(Rejected):
            live_validate.validate("NO2", -1, "ppb", NOW, "CPCB", now=NOW)

    def test_missing_value_is_rejected(self):
        with self.assertRaises(Rejected):
            live_validate.validate("NO2", None, "ppb", NOW, "CPCB", now=NOW)

    def test_nan_and_infinity_are_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(Rejected):
                live_validate.validate("NO2", bad, "ppb", NOW, "CPCB", now=NOW)


class TimestampTests(unittest.TestCase):
    def test_timestamp_is_stored_in_ist(self):
        utc_now = datetime(2026, 10, 10, 12, 30, tzinfo=timezone.utc)
        row = live_validate.validate("NO2", 24.45, "ppb", utc_now, "CPCB", now=NOW)
        self.assertTrue(row["observed_at"].endswith("+05:30"))
        self.assertIn("T18:00", row["observed_at"])

    def test_reading_older_than_three_hours_is_stale(self):
        observed = NOW - timedelta(hours=3, minutes=1)
        self.assertEqual(live_validate.freshness(observed, NOW), "stale")

    def test_reading_within_three_hours_is_live(self):
        self.assertEqual(live_validate.freshness(NOW - timedelta(minutes=90), NOW),
                         "live")

    def test_unparseable_timestamp_is_offline_not_guessed(self):
        self.assertEqual(live_validate.freshness("yesterday", NOW), "offline")

    def test_age_minutes(self):
        self.assertEqual(live_validate.age_minutes(NOW - timedelta(minutes=42), NOW), 42)


class ParseNumberTests(unittest.TestCase):
    def test_plain_and_decimal(self):
        self.assertEqual(live_validate.parse_number("34"), 34.0)
        self.assertEqual(live_validate.parse_number("34.8"), 34.8)

    def test_censored_values_take_the_bound(self):
        self.assertEqual(live_validate.parse_number("<1"), 1.0)
        self.assertEqual(live_validate.parse_number(">900"), 900.0)

    def test_thousands_separator_and_unicode_minus(self):
        self.assertEqual(live_validate.parse_number("1,204"), 1204.0)
        self.assertEqual(live_validate.parse_number("\u22125"), -5.0)

    def test_non_numeric_returns_none(self):
        for text in ("", "  ", "N/A", "NA", None, "very high"):
            self.assertIsNone(live_validate.parse_number(text))


class ConflictTests(unittest.TestCase):
    def _reading(self, value, source, observed="2026-10-10T18:00:00+05:30"):
        return {"value": value, "source": source, "observed_at": observed,
                "pollutant": "NO2", "unit": "ppb", "conflict": 0}

    def test_sources_within_30_percent_agree(self):
        row, conflicted = live_validate.resolve_conflict(
            self._reading(100, "DPCC"), self._reading(120, "CPCB"))
        self.assertFalse(conflicted)

    def test_sources_more_than_30_percent_apart_conflict(self):
        row, conflicted = live_validate.resolve_conflict(
            self._reading(100, "DPCC"), self._reading(200, "CPCB"))
        self.assertTrue(conflicted)

    def test_the_official_cpcb_value_wins_a_conflict(self):
        row, _ = live_validate.resolve_conflict(
            self._reading(100, "OpenAQ"), self._reading(200, "CPCB"))
        self.assertEqual(row["source"], "CPCB")
        self.assertEqual(row["conflict"], 1)

    def test_single_source_never_conflicts(self):
        row, conflicted = live_validate.resolve_conflict(
            self._reading(400, "CPCB"), None)
        self.assertFalse(conflicted)
        self.assertEqual(row["conflict"], 0)


BULLETIN = """
Delhi Real Time Air Quality Index
Ashok Vihar  NO2: 42 µg/m3   O3: 61 ug/m3   PM2.5: 96   AQI: 168
Dwarka  NO2: 38 µg/m3  O3: 55 ug/m3  PM2.5: 104  AQI: 175
Narela  NO2: 51 µg/m3  O3: 48 ug/m3  PM2.5: 121  AQI: 199
Okhla Phase 2  NO2: 47 µg/m3  O3: 66 ug/m3  PM2.5: 88  AQI: 152
Rohini  NO2: 33 µg/m3  O3: 52 ug/m3  PM2.5: 133  AQI: 214
Sonia Vihar  NO2: 29 µg/m3  O3: 44 ug/m3  PM2.5: 79  AQI: 141
"""


class ExtractionTests(unittest.TestCase):
    def test_extracts_every_named_station(self):
        found = live_data.extract_station_readings(BULLETIN)
        self.assertEqual(sorted(found), [1, 2, 4, 5, 6, 7])

    def test_offline_station_is_never_extracted(self):
        # Site 3 retired in 2023 — no amount of text invents a reading for it.
        found = live_data.extract_station_readings(
            "Jawaharlal Nehru Stadium  NO2: 12 µg/m3  AQI: 90")
        self.assertEqual(found, {})

    def test_values_come_back_with_a_unit(self):
        found = live_data.extract_station_readings(BULLETIN)
        self.assertEqual(found[1]["NO2"]["value"], 42.0)
        self.assertEqual(found[1]["NO2"]["unit"], "ug/m3")
        self.assertEqual(found[1]["AQI"]["unit"], "index")

    def test_a_value_without_a_label_is_ignored(self):
        found = live_data.extract_station_readings(
            "Ashok Vihar  42 61 96 168")
        self.assertEqual(found, {})

    def test_unlabelled_garbage_extracts_nothing(self):
        self.assertEqual(live_data.extract_station_readings("hello world"), {})


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        live_store.reset(os.path.join(self.tmp.name, "live.sqlite"))

    def tearDown(self):
        live_store.close()
        self.tmp.cleanup()

    def _row(self, station="1", pollutant="NO2", value=42.0, observed=None, **extra):
        row = {"station_id": station, "station_name": "Ashok Vihar",
               "pollutant": pollutant, "value": value, "unit": "ppb",
               "observed_at": observed or "2026-10-10T18:00:00+05:30",
               "source": "CPCB", "url": "https://cpcb.nic.in/",
               "fetched_at": "2026-10-10T18:00:00+05:30",
               "status": "live", "conflict": 0}
        row.update(extra)
        return row

    def test_readings_round_trip(self):
        live_store.save_readings([self._row()])
        latest = live_store.latest_readings()
        self.assertEqual(len(latest), 1)
        self.assertEqual(latest[0]["source"], "CPCB")

    def _usage(self, day, **extra):
        event = {"at": "2026-10-10T18:00:00+05:30", "type": "fetch",
                 "key": "key_1", "success": 1, "error": None,
                 "rotated": 0, "day": day}
        event.update(extra)
        live_store.record_usage(event)

    def test_usage_is_counted_per_reset_day(self):
        self._usage("2026-10-10")
        self._usage("2026-10-10")
        self._usage("2026-10-09")
        self.assertEqual(live_store.usage_total_for_day("2026-10-10"), 2)
        self.assertEqual(live_store.usage_total_for_day("2026-10-09"), 1)
        self.assertEqual(live_store.usage_total_for_day("2026-10-11"), 0)
        self.assertEqual(live_store.usage_total_for_day(None), 0)

    def test_a_legacy_database_is_migrated_without_losing_history(self):
        # A file created before `day` existed must keep its rows: they are the
        # only record of calls already paid for.
        path = os.path.join(self.tmp.name, "legacy.sqlite")
        legacy = sqlite3.connect(path)
        legacy.execute(
            "CREATE TABLE usage_log ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " at TEXT NOT NULL, type TEXT NOT NULL, key TEXT NOT NULL,"
            " success INTEGER NOT NULL, error TEXT,"
            " rotated INTEGER NOT NULL DEFAULT 0)")
        legacy.execute(
            "INSERT INTO usage_log (at, type, key, success, error, rotated) "
            "VALUES ('2026-10-10T10:00:00+05:30', 'fetch', 'key_1', 1, NULL, 0)")
        legacy.commit()
        legacy.close()

        live_store.reset(path)
        self.assertEqual(len(live_store.usage_log()), 1)   # history survived
        self._usage("2026-10-10")                          # new schema works
        self.assertEqual(live_store.usage_total_for_day("2026-10-10"), 1)

    def test_latest_returns_the_newest_observation(self):
        live_store.save_readings([
            self._row(observed="2026-10-10T17:00:00+05:30", value=10),
            self._row(observed="2026-10-10T18:00:00+05:30", value=20),
        ])
        self.assertEqual(live_store.latest_readings()[0]["value"], 20.0)

    def test_history_is_kept_for_at_least_seven_days(self):
        for day in range(9):
            observed = f"2026-10-{10 - day:02d}T18:00:00+05:30"
            live_store.save_readings([self._row(observed=observed)])
        live_store.prune()
        kept = live_store.readings_history("1", "NO2", days=7)
        self.assertGreaterEqual(len(kept), 7)

    def test_alerts_prefer_official_when_served(self):
        live_store.save_alerts([
            {"id": "press1", "title": "Press", "summary": "s", "severity": "high",
             "source": "NDTV", "url": "https://ndtv.com/a", "official": 0,
             "kind": "news", "published_at": None,
             "fetched_at": "2026-10-10T18:00:00+05:30"},
            {"id": "gov1", "title": "CAQM", "summary": "s", "severity": "low",
             "source": "CAQM", "url": "https://caqm.nic.in/b", "official": 1,
             "kind": "grap", "published_at": None,
             "fetched_at": "2026-10-10T17:00:00+05:30"},
        ])
        alerts = live_store.latest_alerts()
        self.assertEqual(alerts[0]["id"], "gov1")

    def test_job_state_records_last_success(self):
        live_store.record_job("readings", "ok", detail={"accepted": 5})
        states = live_store.job_states()
        self.assertEqual(states["readings"]["status"], "ok")
        self.assertIsNotNone(states["readings"]["last_success_at"])

    def test_usage_log_persists_per_key_totals(self):
        live_store.record_usage({"at": "2026-10-10T18:00:00+05:30", "type": "search",
                                 "key": "key_1", "success": True, "rotated": False})
        totals = live_store.usage_totals()
        self.assertEqual(totals[0]["key"], "key_1")
        self.assertEqual(totals[0]["total"], 1)


class StationPayloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        live_store.reset(os.path.join(self.tmp.name, "live.sqlite"))

    def tearDown(self):
        live_store.close()
        self.tmp.cleanup()

    def test_retired_station_is_reported_offline(self):
        payload = {p["site_id"]: p for p in live_data.station_payload()}
        self.assertEqual(payload[3]["status"], "offline")
        self.assertEqual(payload[3]["retired"], "2023-12-31")

    def test_live_status_carries_age_and_source(self):
        observed = live_store.iso(datetime.now(IST) - timedelta(minutes=20))
        live_store.save_readings([{
            "station_id": "1", "station_name": "Ashok Vihar", "pollutant": "NO2",
            "value": 42.0, "unit": "ppb", "observed_at": observed,
            "source": "CPCB", "url": "https://cpcb.nic.in/",
            "fetched_at": observed, "status": "live", "conflict": 0}])
        payload = {p["name"]: p for p in live_data.station_payload()}["Ashok Vihar"]
        self.assertEqual(payload["status"], "live")
        self.assertEqual(payload["source"], "CPCB")
        self.assertEqual(payload["age_minutes"], 20)

    def test_old_reading_is_reported_stale_with_its_age(self):
        observed = live_store.iso(datetime.now(IST) - timedelta(hours=5))
        live_store.save_readings([{
            "station_id": "1", "station_name": "Ashok Vihar", "pollutant": "NO2",
            "value": 42.0, "unit": "ppb", "observed_at": observed,
            "source": "CPCB", "url": "https://cpcb.nic.in/",
            "fetched_at": observed, "status": "live", "conflict": 0}])
        payload = {p["name"]: p for p in live_data.station_payload()}["Ashok Vihar"]
        self.assertEqual(payload["status"], "stale")
        self.assertEqual(payload["age_minutes"], 300)

    def test_live_values_never_touch_ml_fields(self):
        # The payload only ever exposes web values under `live` (spec §3).
        payload = live_data.station_payload()
        for station in payload:
            self.assertIn("live", station)
            self.assertNotIn("no2_ml", station)


class RefreshReadingsTests(unittest.TestCase):
    """Job A end-to-end with TinyFish mocked.

    Job A has two independent passes — web automation over DPCC's map (the only
    source with station-wise values) and a plain-Fetch Delhi bulletin — so the
    tests exercise both, plus the case where only one of them works.
    """

    # The real shape DPCC's map returns (captured during setup, 10 Oct 2026).
    # Note the network: of our 7 sites only Rohini appears; the rest are DPCC
    # stations elsewhere in Delhi.
    DPCC = {
        "observed_at": "Saturday, October 10, 2026 18:25:00 IST",
        "source": "Delhi Pollution Control Committee",
        "stations": [
            {"name": "Anand Vihar", "no2": 16.8, "o3": 30.8, "pm25": 59,
             "aqi": 59, "unit": "µg/m3"},
            {"name": "Punjabi Bagh", "no2": 27.9, "o3": 26.0, "pm25": 72,
             "aqi": 72, "unit": "µg/m3"},
            {"name": "R.K. Puram", "no2": 29.3, "o3": 39.5, "pm25": 64,
             "aqi": 64, "unit": "µg/m3"},
            {"name": "Nehru Nagar", "no2": 41.0, "o3": 0.0, "pm25": 34,
             "aqi": 34, "unit": "µg/m3"},
            {"name": "Rohini", "no2": 23.5, "o3": 29.9, "pm25": 58,
             "aqi": 58, "unit": "µg/m3"},
        ],
    }

    AQICN = (
        "Delhi AQI: Delhi Real-time Air Quality Index (AQI). | | 159 | Unhealthy "
        "Updated on Saturday 18:00 temperature: 33°C | PM2.5 AQI | 159 | | 50 | 206 "
        "| PM10 AQI | 107 | | O3 AQI | 13 | | NO2 AQI | 14 | | Temp. | 33 | | "
        "Humidity | 57 | | Wind | 1 | | 0 | 3 |"
    )

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        live_store.reset(os.path.join(self.tmp.name, "live.sqlite"))

    def tearDown(self):
        live_store.close()
        self.tmp.cleanup()

    def _run(self, dpcc=None, bulletin=None, waqi=None):
        """Run job A with all three passes mocked.

        `None` means that pass failed. `waqi` is the batched Fetch payload;
        it defaults to an empty batch so the DPCC automation pass runs.
        """
        svc = tinyfish_service_stub()
        patches = []

        # Pass 1 — WAQI batched Fetch.
        if waqi is None:
            patches.append(mock.patch.object(
                svc, "fetch_many", side_effect=svc.TinyFishError("timed out")))
        else:
            patches.append(mock.patch.object(svc, "fetch_many",
                                             return_value=waqi))

        # Pass 2 — DPCC web automation.
        if dpcc is None:
            patches.append(mock.patch.object(
                svc, "automate", side_effect=svc.TinyFishError("timed out")))
        else:
            patches.append(mock.patch.object(svc, "automate",
                                             return_value=dpcc))

        # Pass 3 — the plain-Fetch city bulletin.
        if bulletin is None:
            patches.append(mock.patch.object(
                svc, "fetch_text", side_effect=svc.TinyFishError("page_not_found")))
        else:
            patches.append(mock.patch.object(
                svc, "fetch_text",
                return_value={"url": live_data.CITY_BULLETIN_URLS[0],
                              "title": "Delhi", "site_name": "aqicn.org",
                              "published_date": "", "text": bulletin}))

        with patches[0], patches[1], patches[2]:
            return live_data.refresh_readings()

    def test_station_pass_maps_only_real_name_matches(self):
        result = self._run(dpcc=self.DPCC)
        # Rohini is in DPCC's network and maps to site 6, with all four
        # pollutants it reports.
        rohini = {r["pollutant"] for r in result["readings"]
                  if r["station_name"] == "Northwest Delhi 2"}
        self.assertEqual(rohini, {"NO2", "O3", "PM25", "AQI"})
        # Anand Vihar / Nehru Nagar are NOT our sites and are never mapped on.
        names = {r["station_name"] for r in result["readings"]}
        self.assertNotIn("Anand Vihar", names)
        self.assertNotIn("East Delhi", names)

    def test_unmatched_live_stations_are_reported_for_diagnosis(self):
        result = self._run(dpcc=self.DPCC)
        self.assertIn("Anand Vihar", result["unmatched_live_stations"])
        self.assertIn("Nehru Nagar", result["unmatched_live_stations"])
        self.assertNotIn("Rohini", result["unmatched_live_stations"])

    def test_gases_are_converted_from_micrograms_to_ppb(self):
        result = self._run(dpcc=self.DPCC)
        no2 = next(r for r in result["readings"]
                   if r["station_name"] == "Northwest Delhi 2"
                   and r["pollutant"] == "NO2")
        self.assertEqual(no2["unit"], "ppb")
        # 23.5 µg/m³ ÷ 1.882 ≈ 12.5 ppb
        self.assertAlmostEqual(no2["value"], 12.49, places=1)

    def test_aqi_is_stored_as_an_index_not_a_concentration(self):
        result = self._run(dpcc=self.DPCC)
        aqi = next(r for r in result["readings"]
                   if r["station_name"] == "Northwest Delhi 2"
                   and r["pollutant"] == "AQI")
        self.assertEqual(aqi["unit"], "index")
        self.assertEqual(aqi["value"], 58.0)

    def test_readings_carry_source_and_observed_at(self):
        result = self._run(dpcc=self.DPCC)
        row = next(r for r in result["readings"]
                   if r["station_name"] == "Northwest Delhi 2")
        self.assertEqual(row["source"], "Delhi Pollution Control Committee")
        self.assertTrue(row["url"].startswith("https://"))
        self.assertIn("T18:25", row["observed_at"])   # the page's own timestamp

    def test_city_bulletin_is_stored_separately_from_the_sites(self):
        result = self._run(dpcc=self.DPCC, bulletin=self.AQICN)
        self.assertIsNotNone(result["city"])
        self.assertEqual(result["city"]["aqi"], 159.0)
        self.assertEqual(result["city"]["city"], "Delhi")
        # It is a city average, so it must not appear as one of the 7 sites.
        names = {p["site_id"] for p in live_data.station_payload()}
        self.assertNotIn("CITY", names)

    def test_city_bulletin_survives_when_the_station_pass_fails(self):
        result = self._run(dpcc=None, bulletin=self.AQICN)
        self.assertEqual(result["city"]["aqi"], 159.0)
        self.assertTrue(any("timed out" in e for e in result["dpcc_errors"]))

    def test_a_bulletin_without_a_number_is_not_a_reading(self):
        # Mostly navigation chrome — must not be mistaken for a value.
        with self.assertRaises(tinyfish_service_stub().TinyFishError):
            self._run(dpcc=None,
                      bulletin="Home Sitemap RTI Skip to main content About us")

    def test_both_passes_failing_keeps_the_previous_values(self):
        live_store.save_readings([{
            "station_id": "6", "station_name": "Northwest Delhi 2",
            "pollutant": "NO2", "value": 42.0, "unit": "ppb",
            "observed_at": "2026-01-01T10:00:00+05:30", "source": "CPCB",
            "url": "https://cpcb.nic.in/", "fetched_at": "2026-01-01T10:00:00+05:30",
            "status": "stale", "conflict": 0}])
        with self.assertRaises(tinyfish_service_stub().TinyFishError):
            self._run(dpcc=None, bulletin=None)
        self.assertEqual(live_store.latest_readings()[0]["value"], 42.0)

    def test_out_of_range_value_is_rejected_and_previous_kept(self):
        # Recent, so the 7-day prune does not silently remove the row the
        # assertion depends on.
        observed = live_store.iso(datetime.now(IST) - timedelta(hours=1))
        live_store.save_readings([{
            "station_id": "6", "station_name": "Northwest Delhi 2",
            "pollutant": "AQI", "value": 90.0, "unit": "index",
            "observed_at": observed, "source": "CPCB",
            "url": "https://cpcb.nic.in/", "fetched_at": observed,
            "status": "stale", "conflict": 0}])
        bad = {"observed_at": "Saturday, October 10, 2026 18:25:00 IST",
               "source": "Delhi Pollution Control Committee",
               "stations": [{"name": "Rohini", "aqi": 9000, "unit": "index"}]}
        result = self._run(dpcc=bad)
        self.assertGreaterEqual(result["rejected"], 1)
        self.assertEqual(
            live_store.latest_readings()[0]["value"], 90.0)   # untouched

    def test_retired_station_is_never_given_a_live_reading(self):
        dpcc = {"observed_at": "Saturday, October 10, 2026 18:25:00 IST",
                "source": "DPCC",
                "stations": [
                    {"name": "Jawaharlal Nehru Stadium", "no2": 20.0, "o3": 30.0,
                     "pm25": 50.0, "aqi": 50.0, "unit": "µg/m3"},
                    # A live site alongside it, so the job has something to
                    # store and the assertion is not vacuous.
                    {"name": "Rohini", "no2": 23.5, "o3": 29.9, "pm25": 58,
                     "aqi": 58.0, "unit": "µg/m3"},
                ]}
        result = self._run(dpcc=dpcc, bulletin=self.AQICN)
        self.assertEqual(
            [r for r in result["readings"] if r["station_id"] == "3"], [])
        self.assertIn("Jawaharlal Nehru", result["offline"])
        self.assertNotIn("Jawaharlal Nehru",
                         result["unmatched_live_stations"])

    def test_offline_stations_are_reported(self):
        result = self._run(dpcc=self.DPCC)
        self.assertIn("Jawaharlal Nehru", result["offline"])
        # Only Rohini reports, so the other five sites are offline too.
        self.assertIn("Ashok Vihar", result["offline"])

    def test_a_station_with_no_value_for_a_pollutant_is_omitted(self):
        dpcc = {"observed_at": "Saturday, October 10, 2026 18:25:00 IST",
                "source": "DPCC",
                "stations": [{"name": "Rohini", "aqi": 58.0, "unit": "index"}]}
        result = self._run(dpcc=dpcc)
        pollutants = {r["pollutant"] for r in result["readings"]}
        self.assertEqual(pollutants, {"AQI"})        # NO2/O3/PM2.5 simply absent

    def test_a_label_with_no_number_is_ignored(self):
        self.assertEqual(
            live_data.extract_station_readings("Ashok Vihar NO2:"), {})


class StationMatchingTests(unittest.TestCase):
    """Name matching must be strict — a nearby station is not our station."""

    def test_exact_and_alias_names_match(self):
        self.assertEqual(live_data.match_station_by_name("Rohini")["site_id"], 6)
        self.assertEqual(live_data.match_station_by_name("Ashok Vihar")["site_id"], 1)
        self.assertEqual(live_data.match_station_by_name("Dwarka-Sector 8")["site_id"], 2)

    def test_punctuation_and_case_are_tolerated(self):
        self.assertEqual(live_data.match_station_by_name("  rohini  ")["site_id"], 6)
        self.assertEqual(live_data.match_station_by_name("SONIA VIHAR")["site_id"], 7)

    def test_a_different_station_near_a_site_does_not_match(self):
        # Anand Vihar sits inside East Delhi's district but is not our site.
        self.assertIsNone(live_data.match_station_by_name("Anand Vihar"))
        self.assertIsNone(live_data.match_station_by_name("Nehru Nagar"))
        self.assertIsNone(live_data.match_station_by_name("Punjabi Bagh"))

    def test_a_partial_name_does_not_match(self):
        self.assertIsNone(live_data.match_station_by_name("Rohini Sector 3"))

    def test_empty_and_junk_never_match(self):
        self.assertIsNone(live_data.match_station_by_name(""))
        self.assertIsNone(live_data.match_station_by_name(None))
        self.assertIsNone(live_data.match_station_by_name("###"))


def tinyfish_service_stub():
    import tinyfish_service
    return tinyfish_service


class WaqiStationPassTests(unittest.TestCase):
    """Job A's per-station pass — spec §2A, §3.

    WAQI answers an unknown slug by silently serving the Delhi *city* page,
    so the single most important behaviour here is that a fallback page is
    never accepted as a station. Getting that wrong would put one city-wide
    number on six station cards.
    """

    # Real WAQI page text, trimmed to the parts the parser reads.
    STATION = (
        "| | Narela, Delhi AQI: Narela, Delhi Real-time Air Quality Index (AQI). "
        "| | 152 | Unhealthy Updated on Saturday 19:00 temperature: 33°C | "
        "| current | | past 2 days | min | max | PM2.5 AQI | 161 | | 99 | 212 | "
        "| PM10 AQI | 104 | | 65 | 303 | | O3 AQI | 4 | | 3 | 38 | "
        "| NO2 AQI | 11 | | 8 | 32 |"
    )
    # The page WAQI serves when the slug does not exist — a city average.
    CITY_FALLBACK = (
        "Delhi AQI: Delhi Real-time Air Quality Index (AQI). | | 159 | Unhealthy "
        "Updated on Saturday 19:00 temperature: 33°C | PM2.5 AQI | 159 | | 50 | 206 |"
    )
    # Correct heading but no AQI figure anywhere.
    NO_AQI = "| | Narela, Delhi AQI: Narela, Delhi Real-time Air Quality Index (AQI). |"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        live_store.reset(os.path.join(self.tmp.name, "live.sqlite"))

    def tearDown(self):
        live_store.close()
        self.tmp.cleanup()

    @staticmethod
    def _batch(results):
        return {"results": list(results), "errors": []}

    def _call(self, batch):
        svc = tinyfish_service_stub()
        with mock.patch.object(svc, "fetch_many", return_value=batch):
            return live_data.fetch_waqi_readings(live_data._iso())

    def _live(self):
        return [s for s in live_data.WAQI_STATIONS
                if not live_data.STATION_BY_ID[s["site_id"]].get("retired")]

    def _pages(self, builder):
        """One realistic page per live site, each keyed by its request URL.

        The real response always echoes `url`, and job A joins on it: the API
        returns only the URLs that *succeeded*, so position is not identity.
        The fixture has to carry the field, or it would not exercise the join
        that stops one station's reading landing on another's card.

        Serving the same page for every station is exactly the mistake the
        `name_hint` check exists to catch, so the fixture must not do it.
        """
        return self._batch([
            {"url": live_data.waqi_url(spec["slug"]), "text": builder(spec)}
            for spec in self._live()
        ])

    def _station_page(self, spec, aqi=152):
        return (
            f"| | {spec['name_hint']}, Delhi AQI: {spec['name_hint']}, Delhi "
            f"Real-time Air Quality Index (AQI). | | {aqi} | Unhealthy "
            "Updated on Saturday 19:00 temperature: 33°C | "
            "| current | | past 2 days | min | max | PM2.5 AQI | 161 | | 99 | 212 | "
            "| PM10 AQI | 104 | | 65 | 303 | | O3 AQI | 4 | | 3 | 38 | "
            "| NO2 AQI | 11 | | 8 | 32 |"
        )

    def test_a_real_station_page_is_accepted_as_an_index(self):
        rows, meta, errors = self._call(self._pages(self._station_page))
        self.assertEqual(len(rows), 6)             # one per live site
        for row in rows:
            self.assertEqual(row["pollutant"], "AQI")
            self.assertEqual(row["unit"], "index")
            self.assertEqual(row["value"], 152.0)
            self.assertTrue(row["url"].startswith("https://aqicn.org/"))
            self.assertTrue(row["source"])
        self.assertEqual(meta[0]["kind"], "station")

    def test_aqi_sub_indices_are_never_stored_as_concentrations(self):
        rows, _, _ = self._call(self._pages(self._station_page))
        # The page states PM2.5 AQI 161 / NO2 AQI 11. Neither is a
        # concentration, so nothing but the AQI itself may be kept.
        self.assertEqual({r["pollutant"] for r in rows}, {"AQI"})

    def test_the_city_fallback_page_is_rejected_not_mislabelled(self):
        rows, _, errors = self._call(self._pages(lambda spec: self.CITY_FALLBACK))
        self.assertEqual(rows, [])
        self.assertTrue(any("unmatched" in e for e in errors))

    def test_a_page_without_an_aqi_is_rejected(self):
        rows, _, errors = self._call(self._pages(lambda spec: self.NO_AQI))
        self.assertEqual(rows, [])
        self.assertTrue(any("no AQI" in e for e in errors))

    def test_the_retired_site_is_never_requested(self):
        live = self._live()
        self.assertEqual(len(live), 6)
        self.assertNotIn(3, [s["site_id"] for s in live])

    def test_a_timeout_returns_no_rows_and_an_error(self):
        svc = tinyfish_service_stub()
        with mock.patch.object(svc, "fetch_many",
                               side_effect=svc.TinyFishError("timed out")):
            rows, meta, errors = live_data.fetch_waqi_readings(live_data._iso())
        self.assertEqual(rows, [])
        self.assertEqual(meta, [])
        self.assertTrue(errors and "timed out" in errors[0])

    def test_malformed_pages_do_not_raise(self):
        rows, _, errors = self._call(self._pages(lambda spec: "<html>???</html>"))
        self.assertEqual(rows, [])
        self.assertTrue(errors)

    def test_a_failed_url_never_shifts_a_reading_onto_the_wrong_station(self):
        """`results[]` lists only successes, so array position is not identity.

        The API omits a URL that failed. Joined positionally, dropping the
        third result would attach the fourth station's page to the third
        station's card — a real number under the wrong label.
        """
        live = self._live()
        pages = [
            {"url": live_data.waqi_url(spec["slug"]),
             "text": self._station_page(spec)}
            for i, spec in enumerate(live) if i != 2      # the API omits one
        ]
        rows, _, _errors = self._call(self._batch(pages))

        # Every surviving reading is attributed to the station its own URL
        # names, and the one missing result costs exactly one station.
        self.assertEqual(len(rows), len(live) - 1)
        for row in rows:
            spec = next(s for s in live
                        if live_data.waqi_url(s["slug"]) == row["url"])
            self.assertEqual(int(row["station_id"]), spec["site_id"],
                             f"{row['url']} reported as site {row['station_id']}")

    def test_a_successful_waqi_pass_skips_the_automation_call(self):
        """Budget guard: Fetch already answered, so no browser run is spent."""
        svc = tinyfish_service_stub()
        batch = self._pages(self._station_page)
        with mock.patch.object(svc, "fetch_many", return_value=batch), \
                mock.patch.object(svc, "automate",
                                  side_effect=svc.TinyFishError("should not run")) as auto, \
                mock.patch.object(svc, "fetch_text",
                                  side_effect=svc.TinyFishError("no bulletin")):
            result = live_data.refresh_readings(force=True)
        auto.assert_not_called()
        covered = set(result["stations_covered"])
        self.assertEqual(covered, {1, 2, 4, 5, 6, 7})
        self.assertEqual(result["offline"], ["Jawaharlal Nehru"])


class CaseStudyTests(unittest.TestCase):
    """Job D — the reference figures behind "Delhi's Drastic Case".

    Two properties are worth pinning down: every candidate page is read in ONE
    batched call (a fetch per candidate was up to 16 calls and — with several
    candidates timing out at 90 s each — over six minutes for three numbers),
    and a figure the sources no longer state is dropped rather than served
    indefinitely.
    """

    ANNUAL = {
        "url": "https://cpcb.nic.in/reports/delhi-annual-pm25",
        "title": "Delhi annual average PM2.5 CPCB annual report",
        "published_date": "2026-08-01",
        "text": "The annual mean pm2.5 for Delhi was 118 micrograms per cubic "
                "metre across the continuous monitoring network.",
    }
    WINTER = {
        "url": "https://cpcb.nic.in/reports/delhi-winter-peak",
        "title": "Delhi winter peak AQI CPCB worst air quality index value",
        "published_date": "2026-01-15",
        "text": "The peak aqi reached 494 during january, the worst reading of "
                "the season.",
    }
    # An entire visible body that is a widget: states nothing citable.
    CHROME = {
        "url": "https://cpcb.nic.in/reports/latest-air-quality",
        "title": "Latest Delhi air quality report CPCB CAQM",
        "published_date": "2026-10-09",
        "text": "Accessibility Options Text To Speech Bigger Text Small Text "
                "Line Height Highlight Links Reset All Settings.",
    }
    HOME = {
        "url": "https://caqm.nic.in/",
        "title": "Commission for Air Quality Management",
        "published_date": "2026-10-09",
        "text": "The Commission was constituted to address air quality in NCR.",
    }

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        live_store.reset(os.path.join(self.tmp.name, "live.sqlite"))

    def tearDown(self):
        live_store.close()
        self.tmp.cleanup()

    @staticmethod
    def _page(result):
        return {**result, "url": result["url"], "final_url": result["url"]}

    def _refresh(self, searches, pages):
        """Run job D with a scripted result per figure and one batched fetch."""
        svc = tinyfish_service_stub()
        by_url = {page["url"]: page for page in pages}

        def fetch_many(urls, **kwargs):
            return {"results": [self._page(by_url[u]) for u in urls
                                if u in by_url],
                    "errors": []}

        with mock.patch.object(svc, "search", side_effect=searches) as search, \
                mock.patch.object(svc, "fetch_many",
                                  side_effect=fetch_many) as batch:
            data = live_data.refresh_case_study(force=True)
        return data, search, batch

    # ── budget ───────────────────────────────────────────────────────────
    def test_every_candidate_is_read_in_one_batched_call(self):
        data, search, batch = self._refresh(
            searches=[[self.ANNUAL], [self.WINTER], [self.CHROME]],
            pages=[self.ANNUAL, self.WINTER, self.CHROME])

        self.assertEqual(search.call_count, 3)          # one query per figure
        self.assertEqual(batch.call_count, 1)           # every page at once
        self.assertEqual(list(batch.call_args.args[0]),
                         [self.ANNUAL["url"], self.WINTER["url"],
                          self.CHROME["url"]])

        values = {s["stat_key"]: s["value"] for s in data["stats"]}
        self.assertEqual(values["annual_pm25"], "118")
        self.assertEqual(values["winter_aqi"], "494")

    def test_nothing_to_read_means_no_call_at_all(self):
        """A zero-URL batch would still be billed against the daily cap."""
        _data, _search, batch = self._refresh(
            searches=[[self.HOME], [self.HOME], [self.HOME]], pages=[self.HOME])
        batch.assert_not_called()

    def test_a_call_never_carries_more_than_one_candidate_per_figure(self):
        """Nine live-fetched CPCB pages in one call came back HTTP 504.

        The batch ceiling is a time budget as much as a URL count, so round 1
        asks for each figure's best candidate and only an unanswered figure
        asks for a second round. One slow figure cannot fail the other two.
        """
        extra = {**self.ANNUAL,
                 "url": "https://cpcb.nic.in/reports/delhi-pm25-detail",
                 "text": "The annual mean pm2.5 for Delhi was 118 micrograms."}
        data, _search, batch = self._refresh(
            searches=[[self.ANNUAL, extra], [self.WINTER], [self.CHROME]],
            pages=[self.ANNUAL, extra, self.WINTER, self.CHROME])

        self.assertEqual(batch.call_count, 2)              # round 1, then 2
        for call in batch.call_args_list:
            self.assertLessEqual(len(call.args[0]),
                                 live_data.CASE_FETCH_BATCH)
        self.assertEqual(list(batch.call_args_list[0].args[0]),
                         [self.ANNUAL["url"], self.WINTER["url"],
                          self.CHROME["url"]])
        self.assertEqual(list(batch.call_args_list[1].args[0]), [extra["url"]])

        # The first candidate already answered the figure, so its second was
        # read but never needed — and a figure that round 1 settled is settled.
        self.assertEqual([s["stat_key"] for s in data["stats"]],
                         ["annual_pm25", "winter_aqi"])
        self.assertEqual(data["stats"][0]["value"], "118")

    # ── quality of what gets stored ──────────────────────────────────────
    def test_a_site_homepage_is_never_used_as_a_figure(self):
        data, _search, _batch = self._refresh(
            searches=[[self.HOME], [self.HOME], [self.HOME]], pages=[self.HOME])
        self.assertEqual(data["stats"], [])

    def test_a_summary_made_of_page_chrome_is_rejected_and_purged(self):
        # An earlier refresh stored a "report" that is really a widget bar.
        live_store.save_case_stats([{
            "stat_key": "news_strip", "label": "Latest Delhi air quality report",
            "value": "Accessibility Options Text To Speech", "unit": None,
            "as_of": "2026-09-01", "source": "CAQM",
            "url": "https://caqm.nic.in/",
            "fetched_at": "2026-09-01T00:00:00+05:30",
        }])

        data, _search, _batch = self._refresh(
            searches=[[self.ANNUAL], [self.WINTER], [self.CHROME]],
            pages=[self.ANNUAL, self.WINTER, self.CHROME])

        self.assertNotIn("news_strip",
                         [s["stat_key"] for s in data["stats"]])
        # Judged and disproved, so its row is gone rather than served forever.
        self.assertEqual([r["stat_key"] for r in live_store.case_stats()],
                         ["annual_pm25", "winter_aqi"])

    def test_a_redirect_off_an_allowlisted_source_is_not_cited(self):
        """Fetch followed a CPCB page through to its bare IP address.

        Naming "49.50.115.127" as a figure's source would cite no authority at
        all, so the citation keeps the page that was actually asked for.
        """
        svc = tinyfish_service_stub()
        stray = {**self.ANNUAL,
                 "final_url": "https://49.50.115.127/reports/delhi-annual-pm25"}
        with mock.patch.object(svc, "search",
                               side_effect=[[self.ANNUAL], [self.WINTER],
                                            [self.CHROME]]), \
                mock.patch.object(svc, "fetch_many",
                                  return_value={"results": [stray],
                                                "errors": []}):
            data = live_data.refresh_case_study(force=True)

        annual = next(s for s in data["stats"] if s["stat_key"] == "annual_pm25")
        self.assertEqual(annual["url"], self.ANNUAL["url"])
        self.assertEqual(annual["source"], "CPCB")

    def test_a_failed_search_keeps_the_previous_figure(self):
        """A transient failure must not erase a figure that was correct."""
        live_store.save_case_stats([{
            "stat_key": "annual_pm25", "label": "Annual mean PM2.5",
            "value": "99.7", "unit": "µg/m³", "as_of": "2026-08-01",
            "source": "CPCB", "url": "https://cpcb.nic.in/reports/old",
            "fetched_at": "2026-08-01T00:00:00+05:30",
        }])

        svc = tinyfish_service_stub()
        with mock.patch.object(svc, "search",
                               side_effect=[svc.TinyFishError("timed out"),
                                            [self.WINTER], [self.CHROME]]), \
                mock.patch.object(svc, "fetch_many",
                                  return_value={"results": [self._page(self.WINTER),
                                                            self._page(self.CHROME)],
                                                "errors": []}):
            live_data.refresh_case_study(force=True)

        rows = {row["stat_key"]: row for row in live_store.case_stats()}
        self.assertEqual(rows["annual_pm25"]["value"], "99.7")
        # The other two were judged; news_strip had nothing citable to store.
        self.assertNotIn("news_strip", rows)


if __name__ == "__main__":
    unittest.main()
