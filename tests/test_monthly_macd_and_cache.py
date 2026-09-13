import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import a_share_risk_engine as eng


def _synthetic_monthly_series(base_values: np.ndarray, current_change: float,
                              live: bool = True) -> tuple[pd.Series, pd.Timestamp]:
    now = pd.Timestamp.now().normalize()
    if live:
        prior_months = pd.date_range(
            end=now - pd.offsets.MonthEnd(1),
            periods=len(base_values),
            freq=pd.offsets.MonthEnd(),
        )
        series = pd.Series(base_values, index=prior_months)
        series.loc[now] = float(base_values[-1] + current_change)
    else:
        month_ends = pd.date_range(
            end=now - pd.offsets.MonthEnd(1),
            periods=len(base_values),
            freq=pd.offsets.MonthEnd(),
        )
        series = pd.Series(base_values, index=month_ends)
        series.iloc[-1] = float(series.iloc[-1] + current_change)
    return series, now


def _find_alert(target: str, base_values: np.ndarray, deltas: np.ndarray,
                live: bool = True) -> eng.MonthlyMACDAlert:
    for delta in deltas:
        daily, now = _synthetic_monthly_series(base_values, float(delta), live=live)
        alert = eng.evaluate_monthly_macd("SSE", daily, "synthetic", now=now)
        if alert.level == target:
            return alert
    raise AssertionError(f"Unable to synthesize monthly MACD level {target}")


def _alert(level: str, index_key: str = "SSE", action: str = "RISK_UP") -> eng.MonthlyMACDAlert:
    return eng.MonthlyMACDAlert(
        index_key=index_key,
        index_name=eng.INDEX_NAMES.get(index_key, index_key),
        as_of="2026-09-13",
        source="synthetic",
        level=level,
        action=action,
        is_live_month=False,
        close=100.0,
        dif=1.0,
        dea=0.5,
        gap=0.5,
        previous_completed_gap=0.8,
        gap_daily_slope=None,
        estimated_trading_days_to_cross=None,
        cross_price=None,
        distance_to_cross_pct=None,
        reason=level,
    )


class MonthlyMACDTests(unittest.TestCase):
    def test_live_month_death_cross_is_detected_before_month_end(self):
        daily, now = _synthetic_monthly_series(np.linspace(2000.0, 4000.0, 60), 0.0)
        alert = eng.evaluate_monthly_macd("SSE", daily, "synthetic", now=now)

        self.assertTrue(alert.is_live_month)
        self.assertEqual("DEATH_CROSS_LIVE", alert.level)
        self.assertEqual("RISK_UP", alert.action)
        self.assertLessEqual(alert.gap, 0)
        self.assertGreater(alert.previous_completed_gap, 0)

    def test_positive_but_narrow_gap_triggers_advance_warning(self):
        daily, now = _synthetic_monthly_series(np.linspace(2000.0, 4000.0, 60), 20.0)
        alert = eng.evaluate_monthly_macd("SSE", daily, "synthetic", now=now)

        self.assertEqual("PRE_DEATH_CROSS_CRITICAL", alert.level)
        self.assertGreater(alert.gap, 0)
        self.assertIsNotNone(alert.cross_price)
        self.assertLessEqual(alert.distance_to_cross_pct, 2.5)

    def test_sse_death_cross_becomes_factor_not_override(self):
        daily, now = _synthetic_monthly_series(np.linspace(2000.0, 4000.0, 60), 0.0)
        alert = eng.evaluate_monthly_macd("SSE", daily, "synthetic", now=now)
        factors = eng.build_monthly_macd_factors([alert])
        sse_factor = next(x for x in factors if x.name == "上证指数月线MACD")

        self.assertGreater(sse_factor.signal, 0)
        self.assertEqual(0.8, sse_factor.signal)

    def test_sse_death_cross_does_not_force_sell(self):
        daily, now = _synthetic_monthly_series(np.linspace(2000.0, 4000.0, 60), 0.0)
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

    def test_build_monthly_macd_factors_maps_levels_and_stale(self):
        factors = eng.build_monthly_macd_factors([
            _alert("DEATH_CROSS_CONFIRMED", "SSE", "RISK_UP"),
            _alert("GOLDEN_CROSS_CONFIRMED", "CSI300", "RISK_DOWN"),
            _alert("DATA_STALE", "CSI1000", "VERIFY_DATA"),
            _alert("INSUFFICIENT_HISTORY", "CHINEXT", "NO_SIGNAL"),
        ])

        by_name = {factor.name: factor for factor in factors}
        self.assertEqual(1.0, by_name["上证指数月线MACD"].signal)
        self.assertEqual(-0.75, by_name["沪深300月线MACD"].signal)
        self.assertTrue(by_name["中证1000月线MACD"].stale)
        self.assertTrue(by_name["创业板指月线MACD"].missing)

    def test_bullish_and_golden_cross_states_are_supported(self):
        bullish = _find_alert("BULLISH", np.linspace(2000.0, 4000.0, 60), np.linspace(40.0, 400.0, 40))
        golden_live = _find_alert("GOLDEN_CROSS_LIVE", np.linspace(4000.0, 2000.0, 60), np.linspace(200.0, 3000.0, 80))
        golden_confirmed = _find_alert("GOLDEN_CROSS_CONFIRMED", np.linspace(4000.0, 2000.0, 60), np.linspace(200.0, 3000.0, 80), live=False)

        self.assertEqual("RISK_DOWN", bullish.action)
        self.assertEqual("RISK_DOWN", golden_live.action)
        self.assertEqual("RISK_DOWN", golden_confirmed.action)

    def test_negative_gap_recovery_stays_bearish_with_reduced_risk(self):
        recovering = _find_alert(
            "BEARISH_RECOVERING",
            np.linspace(4000.0, 2000.0, 60),
            np.linspace(-30.0, -5.0, 6),
        )
        factor = eng.build_monthly_macd_factors([recovering])[0]

        self.assertLess(recovering.gap, 0)
        self.assertGreater(recovering.gap, recovering.previous_completed_gap)
        self.assertEqual("RISK_UP", recovering.action)
        self.assertEqual(0.20, factor.signal)

    def test_stale_factor_does_not_affect_score(self):
        factors = [
            eng.FactorResult("fresh_bull", "test", 100.0, -1.0, 1.0, "", "test"),
            eng.FactorResult("stale_bear", "test", 100.0, 1.0, 1.0, "", "test", stale=True),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state"
            with patch.object(eng, "STATE_DIR", state), patch.object(eng, "CACHE_DIR", state / "series_cache"):
                hub = eng.DataHub()
                result = eng.score_engine(factors, {}, hub)

        self.assertEqual(0.0, result.sell_score)
        self.assertEqual(100.0, result.buy_score)

    def test_resonance_ignores_stale_inputs(self):
        features = {"US10Y_20D_BP": 45.0, "DXY_5D": 1.5, "USDCNH_5D": 1.2}

        fresh_adj, _ = eng.compute_resonance(features, stale_keys=set())
        stale_adj, _ = eng.compute_resonance(features, stale_keys={"USDCNH"})

        self.assertEqual(8.0, fresh_adj)
        self.assertEqual(0.0, stale_adj)

    def test_monthly_macd_multi_index_resonance_is_additive_only(self):
        alerts = [
            _alert("DEATH_CROSS_LIVE", "SSE"),
            _alert("BEARISH", "CSI300"),
            _alert("BEARISH", "CSI1000"),
            _alert("DEATH_CROSS_CONFIRMED", "CHINEXT"),
        ]

        adj4, _ = eng.compute_resonance({}, alerts, stale_keys=set())
        adj3, _ = eng.compute_resonance({}, alerts, stale_keys={"CHINEXT"})

        self.assertEqual(5.0, adj4)
        self.assertEqual(3.0, adj3)


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
