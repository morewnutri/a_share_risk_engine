import unittest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import a_share_risk_engine as eng


class _FakeYF:
    def __init__(self, mapping):
        self.mapping = mapping

    def download(self, ticker, **kwargs):
        return self.mapping.get(ticker, pd.DataFrame())


class _FakeResponse:
    text = "observation_date,DGS10\n2026-08-27,4.20\n2026-08-28,4.25\n"

    def raise_for_status(self):
        return None


class DataLoaderRobustnessTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        state = Path(self._tmp.name) / "state"
        self._state_patch = patch.object(eng, "STATE_DIR", state)
        self._cache_patch = patch.object(eng, "CACHE_DIR", state / "series_cache")
        self._state_patch.start()
        self._cache_patch.start()

    def tearDown(self):
        self._cache_patch.stop()
        self._state_patch.stop()
        self._tmp.cleanup()

    def test_yfinance_fallback_ticker_is_used(self):
        idx = pd.date_range("2026-01-01", periods=10, freq="D")
        fallback_df = pd.DataFrame({"Close": range(10)}, index=idx)
        fake = _FakeYF({"^HSTECH": pd.DataFrame(), "3033.HK": fallback_df})

        hub = eng.DataHub(history_days=30)
        with patch.object(eng, "yf", fake), patch.object(eng, "YF_TICKER_CANDIDATES", {"HSTECH": ["^HSTECH", "3033.HK"]}):
            hub.fetch_yfinance()

        self.assertIsNotNone(hub.get("HSTECH"))
        self.assertIn("3033.HK", hub.source("HSTECH"))

    def test_stale_critical_reduces_confidence(self):
        now = pd.Timestamp.now().normalize()
        stale_date = now - timedelta(days=20)
        fresh_date = now - timedelta(days=1)

        fresh_hub = eng.DataHub()
        stale_hub = eng.DataHub()
        for key in eng.CRITICAL_KEYS:
            fresh_hub.add(key, pd.Series([1.0], index=[fresh_date]), "Yahoo Finance via yfinance (dummy)")
            stale_hub.add(key, pd.Series([1.0], index=[stale_date]), "Yahoo Finance via yfinance (dummy)")

        factors = [eng.make_factor("dummy", "g", 1.0, 0.0, 0.0, "ok", "manual")]
        features = {k: 1.0 for k in eng.CRITICAL_KEYS}

        fresh_res = eng.score_engine(factors, features, fresh_hub)
        stale_res = eng.score_engine(factors, features, stale_hub)

        self.assertGreater(fresh_res.confidence, stale_res.confidence)
        self.assertTrue(stale_res.stale_critical)
        self.assertFalse(stale_res.missing_critical)
        self.assertEqual("DATA_INCOMPLETE / 不根据信号交易", stale_res.action)

    def test_stale_factors_are_excluded_from_score(self):
        hub = eng.DataHub()
        factors = [
            eng.FactorResult("fresh", "g", 10.0, 0.0, 1.0, "", "manual", False, False),
            eng.FactorResult("stale_bear", "g", 90.0, 1.0, 1.0, "", "manual", False, True),
        ]
        res = eng.score_engine(factors, {}, hub, [])
        self.assertAlmostEqual(50.0, res.sell_score, places=6)

    def test_resonance_ignores_stale_inputs(self):
        features = {"US10Y_20D_BP": 50.0, "DXY_5D": 2.0, "USDCNH_5D": 2.0}
        fresh_adj, _ = eng.compute_resonance(features, set(), [])
        stale_adj, _ = eng.compute_resonance(features, {"US10Y_20D_BP"}, [])
        self.assertGreater(fresh_adj, stale_adj)
        self.assertEqual(0.0, stale_adj)

    def test_fred_public_csv_fallback_does_not_require_api_key(self):
        with patch.dict(eng.os.environ, {"FRED_API_KEY": ""}), \
             patch.object(eng, "FRED_SERIES", {"US10Y": "DGS10"}), \
             patch.object(eng.requests, "get", return_value=_FakeResponse()):
            hub = eng.DataHub(history_days=30)
            hub.fetch_fred()

        self.assertEqual(4.25, eng.latest(hub.get("US10Y")))
        self.assertIn("FRED public CSV", hub.source("US10Y"))

    def test_nat_index_is_removed_during_normalisation(self):
        s = pd.Series([1.0, 2.0], index=[pd.Timestamp("2026-01-01"), pd.NaT])
        norm = eng.DataHub._normalise_series(s)

        self.assertEqual(1, len(norm))
        self.assertEqual(pd.Timestamp("2026-01-01"), norm.index[0])

    def test_unparseable_turnover_does_not_create_zero_series(self):
        class _FakeAK:
            @staticmethod
            def stock_zh_a_spot_em():
                return pd.DataFrame({"涨跌幅": [1.0, -1.0], "成交额": ["--", "abc"]})

        with patch.object(eng, "ak", _FakeAK()):
            hub = eng.DataHub()
            hub.fetch_a_share_snapshot()
        self.assertIsNone(hub.get("A_TURNOVER"))

    def test_margin_balance_requires_both_legs(self):
        class _FakeAK:
            @staticmethod
            def stock_margin_sse(start_date, end_date):
                return pd.DataFrame({
                    "信用交易日期": ["2026-01-01", "2026-01-02"],
                    "融资余额": [100.0, 110.0],
                })

        with patch.object(eng, "ak", _FakeAK()):
            hub = eng.DataHub()
            hub.fetch_margin()
        self.assertIsNone(hub.get("MARGIN_BALANCE"))
        self.assertTrue(any("MARGIN_BALANCE 保持缺失" in w for w in hub.warnings))

    def test_turnover_ratio_uses_previous_20_day_average(self):
        now = pd.Timestamp.now().normalize()
        idx = pd.date_range(end=now, periods=22, freq="D")
        vals = [100.0] * 20 + [210.0, 2000.0]
        hub = eng.DataHub()
        hub.add("A_TURNOVER_HIST", pd.Series(vals, index=idx), "snapshot")
        fe = eng.FactorEngine(hub)
        features = fe.build_features()
        self.assertAlmostEqual(2.1, features["A_TURNOVER_MA20_RATIO"], places=6)


if __name__ == "__main__":
    unittest.main()
