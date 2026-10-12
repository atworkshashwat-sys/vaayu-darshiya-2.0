"""
Tests for the HCHO hotspot build logic (scripts/build_hcho_hotspots.py).

Run:
    cd backend && python -m unittest -v test_hcho_hotspots

Covers the acceptance list: the P90 rule, clustering, the persistence count,
the empty-hotspot case, stations with missing data, and the mol/m² → µmol/m²
scaling. Everything here is pure logic — no CSV scan, no network.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import build_hcho_hotspots as hs

AS_OF = "2026-06-01"
WINDOW = ["2026-05-26", "2026-05-27", "2026-05-28", "2026-05-29",
          "2026-05-30", "2026-05-31", "2026-06-01"]


class PercentileAndThresholdTests(unittest.TestCase):
    """The relative rule: region P90, floored at the Elevated band."""

    def test_percentile_linear_matches_numpy_default(self):
        self.assertAlmostEqual(hs.percentile(list(range(1, 11)), 90), 9.1, places=6)
        self.assertAlmostEqual(hs.percentile([1.0, 2.0], 50), 1.5, places=6)

    def test_percentile_single_value(self):
        self.assertEqual(hs.percentile([42.0], 90), 42.0)

    def test_percentile_empty_raises(self):
        with self.assertRaises(ValueError):
            hs.percentile([], 90)

    def test_threshold_never_below_elevated_floor(self):
        # P90 below the floor -> floor wins (a flat pass is not a hotspot).
        self.assertEqual(hs.hotspot_threshold(12.4), hs.ELEVATED_FLOOR)
        self.assertEqual(hs.hotspot_threshold(12.4), 31.0)
        # P90 above the floor -> the relative rule wins.
        self.assertEqual(hs.hotspot_threshold(45.0), 45.0)

    def test_is_hotspot_requires_both_conditions(self):
        # Below P90 and below the floor.
        self.assertFalse(hs.is_hotspot(30.0, p90=31.0))
        # Above the floor but below the (higher) P90 -> still not a hotspot.
        self.assertFalse(hs.is_hotspot(40.0, p90=45.0))
        # Clears both.
        self.assertTrue(hs.is_hotspot(46.0, p90=45.0))
        # Flat region: P90 is tiny, the floor is what stops noise firing.
        self.assertFalse(hs.is_hotspot(20.0, p90=12.0))


class ClusteringTests(unittest.TestCase):
    """Adjacent hotspot points (<= 8 km) join one cluster."""

    def test_nearby_points_merge(self):
        # ~5 km apart on the Delhi grid.
        pts = [{"lat": 28.50, "lon": 77.10, "value": 50.0},
               {"lat": 28.545, "lon": 77.10, "value": 60.0}]
        groups = hs.cluster_points(pts, max_km=8.0)
        self.assertEqual(len(groups), 1)
        # Sorted by value descending inside the cluster.
        self.assertEqual(groups[0], [1, 0])

    def test_distant_points_stay_separate(self):
        # ~33 km apart.
        pts = [{"lat": 28.40, "lon": 77.08, "value": 50.0},
               {"lat": 28.58, "lon": 77.43, "value": 90.0}]
        groups = hs.cluster_points(pts, max_km=8.0)
        self.assertEqual(len(groups), 2)

    def test_chain_merges_transitively(self):
        # a-b and b-c are within 8 km, a-c are not -> one cluster.
        pts = [{"lat": 28.50, "lon": 77.10, "value": 40.0},
               {"lat": 28.55, "lon": 77.10, "value": 45.0},
               {"lat": 28.60, "lon": 77.10, "value": 50.0}]
        groups = hs.cluster_points(pts, max_km=8.0)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0], [2, 1, 0])

    def test_outline_is_a_closed_ring(self):
        ring = hs.cluster_outline([28.50, 28.52], [77.10, 77.13])
        self.assertGreaterEqual(len(ring), 4)
        self.assertEqual(ring[0], ring[-1])
        # Every vertex carries [lon, lat].
        for x, y in ring:
            self.assertTrue(76.5 < x < 78.0 and 28.0 < y < 29.0)


class BuildPayloadTests(unittest.TestCase):
    """build_payload: the exact JSON the endpoint serves."""

    META = {
        "A": {"name": "Alpha", "lat": 28.50, "lon": 77.10},
        "B": {"name": "Beta", "lat": 28.55, "lon": 77.15},
        "C": {"name": "Gamma", "lat": 28.80, "lon": 77.40},
    }

    def _payload(self, by_station):
        return hs.build_payload(by_station, self.META, AS_OF)

    def test_hotspot_cluster_with_persistence_and_ratio(self):
        # A is above the Elevated floor on 5 of the 7 days, B is background:
        # 05-27..05-30 and 06-01 at 40, with 05-26 and 05-31 below the floor.
        a = {WINDOW[1]: 40.0, WINDOW[2]: 40.0, WINDOW[3]: 40.0,
             WINDOW[4]: 40.0, WINDOW[5]: 20.0, WINDOW[6]: 40.0,
             WINDOW[0]: 20.0}
        b = {d: 10.0 for d in WINDOW}
        payload = self._payload({"A": a, "B": b})

        self.assertEqual(payload["as_of"], AS_OF)
        self.assertEqual(len(payload["hotspots"]), 1)
        hs_ = payload["hotspots"][0]
        self.assertEqual(hs_["rank"], 1)
        self.assertEqual(hs_["peak"], 40.0)
        self.assertEqual(hs_["persistence"], 5)
        self.assertEqual(hs_["days"], 7)
        # median of [40, 10] = 25 -> 40/25 = 1.6x
        self.assertAlmostEqual(hs_["ratio_to_median"], 1.6, places=1)
        self.assertEqual([s["id"] for s in hs_["stations"]], ["A"])
        # Outline is closed and centred near Alpha.
        self.assertEqual(hs_["polygon"][0], hs_["polygon"][-1])
        self.assertAlmostEqual(hs_["center"][1], 28.50, places=2)

        stats = payload["region_stats"]
        self.assertEqual(stats["as_of"], AS_OF)
        self.assertEqual(stats["n_stations"], 2)
        self.assertEqual(stats["median"], 25.0)
        self.assertEqual(stats["elevated_floor"], 31.0)

    def test_no_hotspot_is_not_forced(self):
        # The real-data shape: P90 of the pass is far below the 31 floor.
        by = {
            "A": {d: 13.0 for d in WINDOW},
            "B": {d: 10.0 for d in WINDOW},
            "C": {d: 9.0 for d in WINDOW},
        }
        payload = self._payload(by)
        self.assertEqual(payload["hotspots"], [])
        self.assertEqual(payload["region_stats"]["threshold"], 31.0)
        self.assertEqual(len(payload["stations"]), 3)

    def test_stations_missing_recent_data_are_excluded(self):
        by = {
            "A": {d: 40.0 for d in WINDOW},
            # C only has data before the window -> excluded entirely.
            "C": {"2020-01-01": 80.0},
        }
        payload = self._payload(by)
        ids = [s["id"] for s in payload["stations"]]
        self.assertEqual(ids, ["A"])
        excluded = [s["id"] for s in payload["no_recent_data"]]
        self.assertEqual(excluded, ["C"])
        self.assertEqual(payload["no_recent_data"][0]["reason"],
                         "no HCHO in the last 7 days")

    def test_partial_week_station_kept_but_out_of_latest_pass_stats(self):
        by = {
            "A": {d: 40.0 for d in WINDOW},          # full week, on the pass
            "B": {WINDOW[0]: 60.0},                  # only the 26th, not the pass
        }
        payload = self._payload(by)
        b = next(s for s in payload["stations"] if s["id"] == "B")
        self.assertEqual(b["date"], WINDOW[0])
        self.assertEqual(b["days"], 1)
        self.assertFalse(b["latest_pass"])
        # Stats are the latest-pass cross-section only: A alone.
        self.assertEqual(payload["region_stats"]["n_stations"], 1)
        self.assertEqual(payload["region_stats"]["n_stations_window"], 2)

    def test_hotspots_ranked_by_peak_across_clusters(self):
        # Two clustered hotspots near Delhi, one 40 km away with a higher peak,
        # plus background stations so the region P90 sits below the hotspots.
        meta = dict(self.META)
        by = {
            "A": {d: 58.0 for d in WINDOW},
            "B": {d: 55.0 for d in WINDOW},          # ~7 km from A -> same cluster
            "C": {d: 80.0 for d in WINDOW},          # separate cluster, higher peak
        }
        for i in range(18):                          # regional background
            gid = f"G{i}"
            meta[gid] = {"name": f"BG {i}", "lat": 28.30 + (i % 9) * 0.02,
                         "lon": 76.80 + (i // 9) * 0.06}
            by[gid] = {d: 10.0 for d in WINDOW}

        payload = hs.build_payload(by, meta, AS_OF)
        # P90 of the 21-station pass lands exactly on B (55.0) -> all three
        # qualify; below that B would (correctly) be excluded.
        self.assertEqual(payload["region_stats"]["p90"], 55.0)
        ranks = [(h["rank"], h["peak"], len(h["stations"])) for h in payload["hotspots"]]
        self.assertEqual(sorted(ranks), [(1, 80.0, 1), (2, 58.0, 2)])
        # C is rank 1; the A+B cluster is rank 2 with both members.
        cluster = next(h for h in payload["hotspots"] if h["rank"] == 2)
        self.assertEqual(sorted(s["id"] for s in cluster["stations"]), ["A", "B"])


class ScalingAndAggregationTests(unittest.TestCase):
    """mol/m² -> µmol/m² (×1e6) and hourly -> daily mean, end to end on a CSV."""

    CSV = (
        "station_id,station_name,latitude,longitude,date,sat_hcho_column\n"
        "1,Test A,28.50,77.10,2026-06-01,1.5e-05\n"
        "1,Test A,28.50,77.10,2026-06-01,2.5e-05\n"
        "1,Test A,28.50,77.10,2026-05-28,\n"
        "2,Test B,28.60,77.20,2026-05-26,5.0e-06\n"
    )

    def test_scan_scales_and_averages(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tiny.csv")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.CSV)
            by_station, meta, as_of = hs.scan(path, chunksize=10)

        # 1.5e-5 and 2.5e-5 mol/m² -> mean 20.0 µmol/m², not 2.0e-5.
        self.assertAlmostEqual(by_station["1"]["2026-06-01"], 20.0, places=6)
        self.assertAlmostEqual(by_station["2"]["2026-05-26"], 5.0, places=6)
        self.assertEqual(as_of, "2026-06-01")
        self.assertEqual(set(meta), {"1", "2"})

        payload = hs.build_payload(by_station, meta, as_of)
        a = next(s for s in payload["stations"] if s["id"] == "1")
        self.assertEqual(a["value"], 20.0)
        self.assertEqual(a["days"], 1)          # the NaN row on 05-28 is ignored

    def test_daily_means_skips_gaps(self):
        rows = [("1", "2026-06-01", 10.0), ("1", "2026-06-01", None),
                ("1", "2026-06-01", 30.0), ("1", "2026-05-31", float("nan"))]
        out = hs.daily_means(rows)
        self.assertEqual(out["1"], {"2026-06-01": 20.0})


class OutputContractTests(unittest.TestCase):
    """The payload must serialise to strict JSON and carry the served fields."""

    def test_payload_is_strict_json_with_required_keys(self):
        by = {"A": {d: 40.0 for d in WINDOW}}
        meta = {"A": {"name": "Alpha", "lat": 28.5, "lon": 77.1}}
        payload = hs.build_payload(by, meta, AS_OF)
        text = json.dumps(payload, allow_nan=False)   # strict: no NaN/Infinity
        roundtrip = json.loads(text)
        for key in ("as_of", "region_stats", "stations", "no_recent_data",
                    "hotspots", "rule", "notes"):
            self.assertIn(key, roundtrip)
        self.assertEqual(roundtrip["unit"], "µmol/m²")
        self.assertIn("Sentinel-5P", roundtrip["notes"]["measurement"])
        self.assertIn("not ground-level", roundtrip["notes"]["measurement"])


class GeometryTests(unittest.TestCase):
    def test_haversine_zero_for_same_point(self):
        self.assertAlmostEqual(hs.haversine_km(28.5, 77.1, 28.5, 77.1), 0.0)

    def test_haversine_delhi_scale(self):
        # 1 degree of latitude ~ 111 km.
        self.assertAlmostEqual(hs.haversine_km(28.0, 77.0, 29.0, 77.0),
                               111.3, delta=0.5)

    def test_convex_hull_of_square(self):
        pts = [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)]
        hull = hs.convex_hull(pts)
        self.assertEqual(len(hull), 4)
        self.assertNotIn((0.5, 0.5), hull)


if __name__ == "__main__":
    unittest.main()
