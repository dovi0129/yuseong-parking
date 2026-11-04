"""Known-answer checks for split hygiene, units, normalization, and coverage."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from pyproj import Transformer
from shapely.geometry import box

from .common import BLOCKS, DAY_TYPES, day_type, hour_block
from .geocode import KakaoGeocoder, apply_geocode_overrides
from .prepare import normalize_location, spot_id
from .priority import capture_summary, daily_capture, fit_scores
from .processing import processing_metrics


def reports(rows):
    frame = pd.DataFrame(rows, columns=["spot_id", "ts"])
    frame["ts"] = pd.to_datetime(frame.ts)
    frame["date"] = frame.ts.dt.normalize()
    frame["day_type"] = frame.ts.map(day_type)
    frame["hour_block"] = frame.ts.dt.hour.map(hour_block)
    return frame


class NormalizeTests(unittest.TestCase):
    def test_parcel_normalization(self):
        self.assertEqual(normalize_location("대전광역시 유성구 봉명동 0656 - ８번지 부근"), ("봉명동 656-8", "봉명동", "address"))

    def test_mountain_is_not_dropped(self):
        self.assertEqual(normalize_location("대전 유성구 세동 산 12-1 주변")[0], "세동 산 12-1")
        self.assertNotEqual(spot_id("세동 산 12-1"), spot_id("세동 12-1"))

    def test_keyword_and_road_preserved(self):
        self.assertEqual(normalize_location("대전광역시 유성구 관평동 동화중학교 주변"), ("관평동 동화중학교", "관평동", "keyword"))
        self.assertEqual(normalize_location("대학로 211 부근")[0], "대학로 211")

    def test_same_parcel_fixed_rule_and_distinct_subparcel(self):
        a = normalize_location("봉명동 656-8번지 부근(리슈빌 앞)")[0]
        b = normalize_location("봉명동 656-8 주변")[0]
        self.assertEqual(spot_id(a), spot_id(b))
        self.assertNotEqual(spot_id(a), spot_id("봉명동 656-9"))


class CalendarTests(unittest.TestCase):
    def test_block_boundaries(self):
        expected = ["00-07"] * 7 + ["07-10"] * 3 + ["10-14"] * 4 + ["14-17"] * 3 + ["17-21"] * 4 + ["21-24"] * 3
        self.assertEqual([hour_block(h) for h in range(24)], expected)
        for h in (-1, 24):
            with self.assertRaises(ValueError):
                hour_block(h)

    def test_temporary_and_substitute_holidays(self):
        for ts in ("2024-10-01", "2024-05-06", "2024-02-12", "2024-04-10"):
            self.assertEqual(day_type(pd.Timestamp(ts)), "주말")
        self.assertEqual(day_type(pd.Timestamp("2024-07-17")), "평일")


class PriorityTests(unittest.TestCase):
    def test_zero_days_are_exposures(self):
        data = reports([("A", "2024-01-02 12:00"), ("A", "2024-01-02 12:30")])
        score = fit_scores(data, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-04"))
        value = score[(score.method == "freq_baseline") & (score.day_type == "평일") & (score.hour_block == "10-14")].score.iloc[0]
        self.assertAlmostEqual(value, 1.)  # two reports / two weekdays, not two / one observed day

    def test_test_period_append_cannot_change_fit(self):
        original = reports([("A", "2024-01-02 12:00"), ("B", "2024-06-05 18:00")])
        changed = pd.concat([original, reports([("TEST_ONLY", "2024-10-02 12:00")] * 100)], ignore_index=True)
        a = fit_scores(original, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-10-01"))
        b = fit_scores(changed, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-10-01"))
        assert_frame_equal(a, b)
        self.assertNotIn("TEST_ONLY", set(b.spot_id))

    def test_final_inner_boundary_excludes_august(self):
        data = reports([("A", "2024-07-31 23:59"), ("FUTURE", "2024-08-01 00:00")])
        score = fit_scores(data, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-08-01"))
        self.assertEqual(set(score.spot_id), {"A"})

    def test_shrinkage_preserves_spot_daily_mass(self):
        data = reports([("A", "2024-01-02 12:00"), ("A", "2024-01-03 18:00"), ("B", "2024-01-06 12:00")])
        score = fit_scores(data, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01"))
        a = score.groupby(["spot_id", "day_type", "method"]).score.sum().unstack("method")
        np.testing.assert_allclose(a.freq_baseline, a.shrinkage, atol=1e-12)
        self.assertTrue(np.isfinite(score.score).all())
        self.assertTrue((score.score >= 0).all())

    def test_unseen_spots_stay_in_denominator(self):
        score = pd.DataFrame([{"spot_id": "A", "day_type": "평일", "hour_block": "10-14", "score": 1., "method": "freq_baseline"}])
        data = reports([("A", "2024-01-02 12:00"), ("UNSEEN", "2024-01-02 12:00"), ("A", "2024-01-02 18:00")])
        result = daily_capture(score, data, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-04"))
        result = result[result.K == 20]
        self.assertEqual(len(result), 3)
        self.assertEqual(int(result.total.sum()), 3)
        self.assertEqual(int(result.captured.sum()), 1)

    def test_null_arm_equal_methods_have_exact_zero_paired_delta(self):
        rows = []
        for method in ("freq_baseline", "shrinkage"):
            for k in (20, 50, 100):
                rows += [{"date": pd.Timestamp("2024-10-01"), "method": method, "K": k, "captured": 50, "total": 100},
                         {"date": pd.Timestamp("2024-10-02"), "method": method, "K": k, "captured": 0, "total": 1}]
        summary = capture_summary(pd.DataFrame(rows), bootstrap=100)
        np.testing.assert_allclose(summary.capture_rate, 50 / 101)
        np.testing.assert_allclose(summary.delta_vs_baseline, 0, atol=1e-12)
        np.testing.assert_allclose(summary.delta_ci_low, 0, atol=1e-12)
        np.testing.assert_allclose(summary.delta_ci_high, 0, atol=1e-12)


class GeospatialTests(unittest.TestCase):
    def test_override_handles_string_coordinate_cache_without_mutating_raw(self):
        raw = pd.DataFrame([{"spot_id": "A", "lat": "", "lon": "", "quality": "no_local_match", "geocode_method": "failed"}])
        corrections = pd.DataFrame([{"spot_id": "A", "lat": 36.35, "lon": 127.34, "quality": "ok", "geocode_method": "manual"}]).set_index("spot_id")
        actual = apply_geocode_overrides(raw, corrections)
        self.assertAlmostEqual(actual.lat.iloc[0], 36.35)
        self.assertEqual(actual.geocode_method.iloc[0], "manual")
        self.assertEqual(raw.lat.iloc[0], "")

    def test_accident_column_order_roundtrip(self):
        forward = Transformer.from_crs(5179, 4326, always_xy=True)
        backward = Transformer.from_crs(4326, 5179, always_xy=True)
        lon, lat = forward.transform(984157, 1819127)
        self.assertTrue(127 < lon < 128 and 36 < lat < 37)
        np.testing.assert_allclose(backward.transform(lon, lat), [984157, 1819127], atol=1e-5)

    @patch("src.pipeline.geocode.time.sleep")
    def test_address_exactness_and_district_boundary(self, _sleep):
        client = KakaoGeocoder("dummy-value-not-a-real-key", box(127.2, 36.2, 127.5, 36.6), interval=0)
        response = unittest.mock.Mock(status_code=200)
        response.json.return_value = {"documents": [{"x": "127.3486", "y": "36.3567", "address_name": "대전 유성구 봉명동 656-8"}]}
        client.session.get = unittest.mock.Mock(return_value=response)
        result = client.search("봉명동 656-8", "address")
        self.assertEqual(result["quality"], "ok")
        self.assertAlmostEqual(result["lon"], 127.3486)
        self.assertEqual(result["geocode_method"], "kakao_address")
        response.json.return_value["documents"][0]["address_name"] = "대전 유성구 봉명동 656-9"
        self.assertEqual(client.search("봉명동 656-8", "address")["geocode_method"], "failed")

    def test_auth_error_does_not_leak_credentials(self):
        client = KakaoGeocoder("dummy-value-not-a-real-key", box(127.2, 36.2, 127.5, 36.6))
        response = unittest.mock.Mock(status_code=401)
        client.session.get = unittest.mock.Mock(return_value=response)
        with self.assertRaises(RuntimeError) as error:
            client.search("봉명동 656-8", "address")
        self.assertNotIn("dummy-value", str(error.exception))


class ProcessingTests(unittest.TestCase):
    def test_threshold_is_training_only_and_grouping_units_are_distinct(self):
        train = reports([("A", "2024-01-02 12:00")] * 10)
        train["split"] = "train"
        test = reports([("A", "2024-10-02 12:00"), ("A", "2024-10-02 12:30"),
                        ("A", "2024-10-02 18:00"), ("B", "2024-10-02 12:00"),
                        ("B", "2024-10-03 12:00")])
        test["split"] = "test"
        metrics = processing_metrics(pd.concat([train, test], ignore_index=True))
        threshold = metrics[(metrics.metric == "train_threshold_coverage") & (metrics.threshold == 10)].iloc[0]
        self.assertEqual(threshold.n_spots_train, 1)
        self.assertAlmostEqual(threshold.coverage_rate, 3 / 5)
        spot_day = metrics[metrics.metric == "spot_date_groups"].iloc[0]
        spot_day_block = metrics[metrics.metric == "spot_date_hour_block_groups"].iloc[0]
        self.assertEqual(spot_day.n_units_after, 3)
        self.assertEqual(spot_day_block.n_units_after, 4)
        self.assertAlmostEqual(spot_day.reduction_rate, .4)
        self.assertAlmostEqual(spot_day_block.reduction_rate, .2)


if __name__ == "__main__":
    unittest.main()
