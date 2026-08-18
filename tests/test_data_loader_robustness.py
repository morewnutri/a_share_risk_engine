import os
import tempfile
import unittest
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

    def test_hstech_ticker_order_prefers_hk_etf(self):
        """3067.HK should be the first candidate for HSTECH; ^HSTECH should be last."""
        candidates = eng.YF_TICKER_CANDIDATES["HSTECH"]
        self.assertEqual(candidates[0], "3067.HK", "3067.HK (Hang Seng TECH ETF) should be first")
        self.assertIn("^HSTECH", candidates, "^HSTECH should still be present as fallback")
        self.assertGreater(candidates.index("^HSTECH"), 0, "^HSTECH should not be first")

    def test_a50_ticker_order_prefers_2823(self):
        """2823.HK should be the first candidate for A50; XIN9.SI should follow."""
        candidates = eng.YF_TICKER_CANDIDATES["A50"]
        self.assertEqual(candidates[0], "2823.HK", "2823.HK (iShares FTSE A50 ETF) should be first")

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

    def test_stale_snapshot_breadth_injection(self):
        """When live fetch fails, _inject_stale_snapshot_breadth should load snapshot data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Patch STATE_DIR to tmpdir
            snap_path = Path(tmpdir) / "a_market_snapshot.csv"
            snap_df = pd.DataFrame([{
                "date": "2026-01-01",
                "breadth": 0.55,
                "decliners": 0.30,
                "strong3": 0.10,
                "weak3": 0.08,
                "limit_down_approx": 5,
                "limit_down_ratio": 0.001,
                "turnover": 1e12,
            }])
            snap_df.to_csv(snap_path, index=False)

            hub = eng.DataHub()
            with patch.object(eng, "STATE_DIR", Path(tmpdir)):
                hub._inject_stale_snapshot_breadth()

            self.assertIsNotNone(hub.get("A_BREADTH"), "A_BREADTH should be populated from stale snapshot")
            self.assertAlmostEqual(float(hub.get("A_BREADTH").iloc[-1]), 0.55)
            self.assertTrue(any("快照" in w or "stale" in w.lower() for w in hub.warnings),
                            "A warning about stale data should be emitted")

    def test_parse_ak_index_df_stores_close_and_amount(self):
        """_parse_ak_index_df should populate price and amount series."""
        idx = pd.date_range("2026-01-01", periods=5, freq="D")
        df = pd.DataFrame({
            "date": idx.strftime("%Y-%m-%d"),
            "close": [100.0, 101, 102, 103, 104],
            "amount": [1e9, 2e9, 1.5e9, 1.8e9, 2.1e9],
        })
        hub = eng.DataHub()
        result = hub._parse_ak_index_df(df, "CSI300", "test_source")
        self.assertTrue(result)
        self.assertIsNotNone(hub.get("CSI300"))
        self.assertIsNotNone(hub.get("CSI300_INDEX_AMOUNT"))


if __name__ == "__main__":
    unittest.main()
