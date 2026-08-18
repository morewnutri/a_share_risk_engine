import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd

import a_share_risk_engine as eng


class _FakeYF:
    def __init__(self, mapping):
        self.mapping = mapping

    def download(self, ticker, **kwargs):
        return self.mapping.get(ticker, pd.DataFrame())


class DataLoaderRobustnessTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
