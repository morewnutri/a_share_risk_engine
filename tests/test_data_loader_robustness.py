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

    def test_fred_public_csv_fallback_does_not_require_api_key(self):
        with patch.dict(eng.os.environ, {"FRED_API_KEY": ""}), \
             patch.object(eng, "FRED_SERIES", {"US10Y": "DGS10"}), \
             patch.object(eng.requests, "get", return_value=_FakeResponse()):
            hub = eng.DataHub(history_days=30)
            hub.fetch_fred()

        self.assertEqual(4.25, eng.latest(hub.get("US10Y")))
        self.assertIn("FRED public CSV", hub.source("US10Y"))


if __name__ == "__main__":
    unittest.main()
