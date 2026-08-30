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
        end=now - pd.offsets.MonthEnd(1), periods=60, freq="M"
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
        self.assertEqual("SELL_NOW", alert.action)
        self.assertLessEqual(alert.gap, 0)
        self.assertGreater(alert.previous_completed_gap, 0)

    def test_positive_but_narrow_gap_triggers_advance_warning(self):
        daily, now = _synthetic_monthly_series(20.0)
        alert = eng.evaluate_monthly_macd("SSE", daily, "synthetic", now=now)

        self.assertEqual("PRE_DEATH_CROSS_CRITICAL", alert.level)
        self.assertGreater(alert.gap, 0)
        self.assertIsNotNone(alert.cross_price)
        self.assertLessEqual(alert.distance_to_cross_pct, 2.5)

    def test_sse_death_cross_overrides_low_composite_confidence(self):
        daily, now = _synthetic_monthly_series(0.0)
        alert = eng.evaluate_monthly_macd("SSE", daily, "synthetic", now=now)
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state"
            with patch.object(eng, "STATE_DIR", state), patch.object(eng, "CACHE_DIR", state / "series_cache"):
                hub = eng.DataHub()
                result = eng.score_engine([], {}, hub, [alert])

        self.assertEqual("SELL / 上证月线MACD死叉", result.action)
        self.assertIn("最高优先级", result.decision_path[0])


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
