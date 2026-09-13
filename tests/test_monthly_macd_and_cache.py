import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import a_share_risk_engine as eng


def _synthetic_monthly_series(current_change: float) -> tuple[pd.Series, pd.Timestamp]:
    now = pd.Timestamp.now().normalize()
    prior_months = pd.date_range(
        end=now - pd.offsets.MonthEnd(1), periods=60, freq="ME"
    )
    values = np.linspace(2000.0, 4000.0, 60)
    series = pd.Series(values, index=prior_months)
    series.loc[now] = 4000.0 + current_change
    return series, now


class MonthlyMACDTests(unittest.TestCase):
    def test_live_month_death_cross_is_detected_before_month_end(self):
        daily, now = _synthetic_monthly_series(0.0)
        alert = eng.evaluate_monthly_macd("SSE", daily, "synthetic", now=now)

        self.assertTrue(alert.is_live_month)
        self.assertEqual("DEATH_CROSS_LIVE", alert.level)
        self.assertEqual("RISK_DOWN", alert.action)
        self.assertLessEqual(alert.gap, 0)
        self.assertGreater(alert.previous_completed_gap, 0)

    def test_positive_but_narrow_gap_triggers_advance_warning(self):
        daily, now = _synthetic_monthly_series(20.0)
        alert = eng.evaluate_monthly_macd("SSE", daily, "synthetic", now=now)

        self.assertEqual("PRE_DEATH_CROSS_CRITICAL", alert.level)
        self.assertGreater(alert.gap, 0)
        self.assertIsNotNone(alert.cross_price)
        self.assertLessEqual(alert.distance_to_cross_pct, 2.5)

    def test_sse_death_cross_becomes_factor_not_override(self):
        daily, now = _synthetic_monthly_series(0.0)
        alert = eng.evaluate_monthly_macd("SSE", daily, "synthetic", now=now)
        factors = eng.build_monthly_macd_factors([alert])
        sse_factor = next(x for x in factors if x.name == "上证指数月线MACD")

        self.assertGreater(sse_factor.signal, 0)
        self.assertEqual(0.8, sse_factor.signal)

    def test_sse_death_cross_does_not_force_sell(self):
        daily, now = _synthetic_monthly_series(0.0)
        alert = eng.evaluate_monthly_macd("SSE", daily, "synthetic", now=now)
        macd_factors = eng.build_monthly_macd_factors([alert])
        factors = [
            eng.FactorResult("bull", "test", 100.0, -0.7, 1.0, "", "test")
        ]
        factors.extend(macd_factors)

        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state"
            with patch.object(eng, "STATE_DIR", state), patch.object(eng, "CACHE_DIR", state / "series_cache"):
                hub = eng.DataHub()
                result = eng.score_engine(factors, {}, hub, [alert])

        self.assertNotEqual("SELL / 上证月线MACD死叉", result.action)

    def test_golden_cross_live_is_detected(self):
        now = pd.Timestamp.now().normalize()
        daily = pd.Series([1.0] * 90, index=pd.date_range(end=now, periods=90, freq="D"))
        idx = pd.date_range(end=now, periods=40, freq="ME")
        gap = np.concatenate([np.linspace(-1.2, -0.4, 38), [-0.2, 0.1]])
        frame = pd.DataFrame({"close": 3000.0, "dif": 0.0, "dea": 0.0, "gap": gap}, index=idx)
        with patch.object(eng, "_monthly_macd_frame", return_value=frame):
            alert = eng.evaluate_monthly_macd("SSE", daily, "synthetic", now=now)
        self.assertEqual("GOLDEN_CROSS_LIVE", alert.level)
        self.assertEqual("RISK_UP", alert.action)

    def test_bullish_state_is_detected(self):
        now = pd.Timestamp.now().normalize()
        daily = pd.Series([1.0] * 90, index=pd.date_range(end=now, periods=90, freq="D"))
        idx = pd.date_range(end=now, periods=40, freq="ME")
        gap = np.concatenate([np.linspace(-1.5, -0.5, 38), [-0.8, -0.3]])
        frame = pd.DataFrame({"close": 3000.0, "dif": 0.0, "dea": 0.0, "gap": gap}, index=idx)
        with patch.object(eng, "_monthly_macd_frame", return_value=frame):
            alert = eng.evaluate_monthly_macd("SSE", daily, "synthetic", now=now)
        self.assertEqual("BULLISH", alert.level)
        self.assertEqual("RISK_UP", alert.action)

    def test_multi_index_monthly_macd_resonance(self):
        alerts = [
            eng.MonthlyMACDAlert("SSE", "上证指数", "2026-01-01", "x", "BEARISH", "RISK_DOWN", True, 1, 1, 1, -1, 1, None, None, None, None, ""),
            eng.MonthlyMACDAlert("CSI300", "沪深300", "2026-01-01", "x", "DEATH_CROSS_LIVE", "RISK_DOWN", True, 1, 1, 1, -1, 1, None, None, None, None, ""),
            eng.MonthlyMACDAlert("CSI1000", "中证1000", "2026-01-01", "x", "BEARISH", "RISK_DOWN", True, 1, 1, 1, -1, 1, None, None, None, None, ""),
        ]
        adj, _ = eng.compute_resonance({}, set(), alerts)
        self.assertEqual(3.0, adj)


class CacheAndMergeTests(unittest.TestCase):
    def test_new_source_overwrites_overlap_and_keeps_old_dates(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state"
            with patch.object(eng, "STATE_DIR", state), patch.object(eng, "CACHE_DIR", state / "series_cache"):
                hub = eng.DataHub()
                old = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-01-01", "2026-01-02"]))
                new = pd.Series([20.0, 30.0], index=pd.to_datetime(["2026-01-02", "2026-01-03"]))
                hub.add("SSE", old, "cache")
                hub.add("SSE", new, "primary", merge=True, prefer_new=True)

                merged = hub.get("SSE")
                self.assertEqual(3, len(merged))
                self.assertEqual(20.0, float(merged.loc["2026-01-02"]))
                self.assertEqual(1.0, float(merged.loc["2026-01-01"]))

                restored = eng.DataHub()
                restored.load_series_cache()
                self.assertEqual(3, len(restored.get("SSE")))


if __name__ == "__main__":
    unittest.main()
