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


class _FakeAK:
    def __init__(self, snapshot=None, sse_margin=None, sz_margin=None):
        self._snapshot = snapshot
        self._sse_margin = sse_margin
        self._sz_margin = sz_margin

    def stock_zh_a_spot_em(self):
        return self._snapshot if self._snapshot is not None else pd.DataFrame()

    def stock_margin_sse(self, **kwargs):
        return self._sse_margin if self._sse_margin is not None else pd.DataFrame()

    def macro_china_market_margin_sz(self):
        return self._sz_margin if self._sz_margin is not None else pd.DataFrame()


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

    def test_stale_critical_inputs_can_trigger_data_incomplete(self):
        now = pd.Timestamp.now().normalize()
        stale_date = now - timedelta(days=20)
        hub = eng.DataHub()
        stale_keys = sorted(list(eng.CRITICAL_KEYS))[:3]
        for key in stale_keys:
            hub.add(key, pd.Series([1.0], index=[stale_date]), "Yahoo Finance via yfinance (dummy)")
        for key in eng.CRITICAL_KEYS - set(stale_keys):
            hub.add(key, pd.Series([1.0], index=[now]), "Yahoo Finance via yfinance (dummy)")

        result = eng.score_engine(
            [eng.make_factor("dummy", "g", 1.0, 0.0, 0.0, "ok", "manual")],
            {k: 1.0 for k in eng.CRITICAL_KEYS},
            hub,
        )

        self.assertEqual("DATA_INCOMPLETE / 不根据信号交易", result.action)
        self.assertEqual(sorted(stale_keys), result.stale_critical)

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

    def test_all_nan_snapshot_amount_does_not_become_zero_turnover(self):
        snapshot = pd.DataFrame({"涨跌幅": [1.0, -1.0], "成交额": ["bad", None]})

        with patch.object(eng, "ak", _FakeAK(snapshot=snapshot)):
            hub = eng.DataHub(history_days=30)
            hub.fetch_a_share_snapshot()

        self.assertIsNone(hub.get("A_TURNOVER"))

    def test_margin_balance_requires_both_legs(self):
        sse_margin = pd.DataFrame({"信用交易日期": ["2026-09-12"], "融资余额": [100.0]})

        with patch.object(eng, "ak", _FakeAK(sse_margin=sse_margin)):
            hub = eng.DataHub(history_days=30)
            hub.fetch_margin()

        self.assertIsNone(hub.get("MARGIN_BALANCE"))
        self.assertTrue(any("未合成 MARGIN_BALANCE" in warning for warning in hub.warnings))

    def test_turnover_ratio_uses_previous_20_day_average(self):
        hub = eng.DataHub(history_days=30)
        idx = pd.date_range("2026-08-01", periods=21, freq="B")
        hub.add("A_TURNOVER_HIST", pd.Series(range(1, 22), index=idx), "local snapshot history")

        features = eng.FactorEngine(hub).build_features()

        self.assertAlmostEqual(2.0, features["A_TURNOVER_MA20_RATIO"])

    def test_manual_none_values_are_skipped(self):
        manual = Path(self._tmp.name) / "manual_overrides.json"
        manual.write_text(
            '{"ETF_FLOW_5D_BN":{"value":null,"date":"2026-09-13"},"IF_BASIS_PCT":{"value":1.5,"date":"2026-09-13"}}',
            encoding="utf-8",
        )

        hub = eng.DataHub(history_days=30)
        hub.load_manual_overrides(manual)

        self.assertIsNone(hub.get("ETF_FLOW_5D_BN"))
        self.assertEqual(1.5, eng.latest(hub.get("IF_BASIS_PCT")))


if __name__ == "__main__":
    unittest.main()
