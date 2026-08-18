#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股多因子外部风险评分引擎
================================
用途：
1) 自动/半自动采集 A股、港股、美股、日韩、汇率、美债、信用利差、商品、融资余额等数据；
2) 同时考虑“绝对值 + 5日/20日变化速度 + 相对位置/共振”；
3) 输出 0-100 买入分、卖出分、数据置信度；
4) 对缺失/过期数据报警；
5) 输出透明的规则型决策树（DOT，可选渲染 PNG）；
6) 保存 A 股每日横截面快照，以便后续计算成交额 MA20、市场宽度历史。

重要：
- 这是研究/风控框架，不是自动交易指令。
- 阈值是“初始启发式参数”，应使用你自己的历史数据回测后校准。
- 默认不会因为单个指标达到某个绝对值就直接发出买卖信号。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import akshare as ak
except Exception:
    ak = None

DEFAULT_HISTORY_DAYS = 220
STATE_DIR = Path("state")
OUTPUT_DIR = Path("output")
MANUAL_FILE = Path("manual_overrides.json")

YF_TICKERS = {
    "SSE": "000001.SS",
    "CSI300": "000300.SS",
    "CSI1000": "000852.SS",
    "CHINEXT": "399006.SZ",
    "STAR50": "000688.SS",
    "HSI": "^HSI",
    "HSTECH": "^HSTECH",
    "A50": "XIN9.SI",
    "VIX": "^VIX",
    "NASDAQ100": "^NDX",
    "SOX": "^SOX",
    "DXY": "DX-Y.NYB",
    "NIKKEI": "^N225",
    "KOSPI": "^KS11",
    "USDCNH": "CNH=X",
    "USDJPY": "JPY=X",
    "COPPER": "HG=F",
    "OIL": "CL=F",
    "IRON_ORE": "TIO=F",
}

AK_INDEX_SYMBOLS = {
    "SSE": "sh000001",
    "CSI300": "sh000300",
    "CSI1000": "sh000852",
    "CHINEXT": "sz399006",
    "STAR50": "sh000688",
}

FRED_SERIES = {
    "US10Y": "DGS10",
    "US10Y_REAL": "DFII10",
    "US2Y": "DGS2",
    "HY_OAS": "BAMLH0A0HYM2",
    "FED_FUNDS": "DFF",
}

CRITICAL_KEYS = {
    "US10Y", "US10Y_REAL", "USDCNH", "VIX",
    "HSTECH", "CSI300", "A_BREADTH"
}

MAX_AGE_DAYS = {
    "market": 5,
    "fred": 10,
    "ak_macro": 15,
    "snapshot": 5,
    "manual": 7,
}

@dataclass
class DataSeries:
    key: str
    values: pd.Series
    source: str
    last_date: Optional[pd.Timestamp] = None
    note: str = ""

@dataclass
class FactorResult:
    name: str
    group: str
    weight: float
    signal: Optional[float]   # -1=偏多, +1=偏空
    value: Optional[float]
    detail: str
    source: str
    missing: bool = False

    @property
    def contribution(self) -> Optional[float]:
        if self.signal is None:
            return None
        return self.weight * self.signal

@dataclass
class EngineResult:
    timestamp: str
    buy_score: float
    sell_score: float
    confidence: float
    action: str
    risk_level: str
    resonance_adjustment: float
    missing_critical: List[str]
    warnings: List[str]
    factors: List[FactorResult]
    decision_path: List[str]

def clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))

def latest(s: Optional[pd.Series]) -> Optional[float]:
    if s is None or len(s.dropna()) == 0:
        return None
    return float(s.dropna().iloc[-1])

def pct_change_n(s: Optional[pd.Series], n: int) -> Optional[float]:
    if s is None:
        return None
    s = s.dropna()
    if len(s) <= n:
        return None
    old, new = float(s.iloc[-1-n]), float(s.iloc[-1])
    if old == 0:
        return None
    return (new / old - 1.0) * 100.0

def bp_change_n(s: Optional[pd.Series], n: int) -> Optional[float]:
    if s is None:
        return None
    s = s.dropna()
    if len(s) <= n:
        return None
    return (float(s.iloc[-1]) - float(s.iloc[-1-n])) * 100.0

def zscore_last(s: Optional[pd.Series], window: int = 60) -> Optional[float]:
    if s is None:
        return None
    s = s.dropna().tail(window)
    if len(s) < max(15, window // 3):
        return None
    std = float(s.std(ddof=0))
    if std == 0 or math.isnan(std):
        return 0.0
    return float((s.iloc[-1] - s.mean()) / std)

def linear_slope_per_day(s: Optional[pd.Series], window: int = 20) -> Optional[float]:
    if s is None:
        return None
    s = s.dropna().tail(window)
    if len(s) < max(8, window // 2):
        return None
    y = s.astype(float).values
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0])

def piecewise_risk(x: float, points: List[Tuple[float, float]]) -> float:
    points = sorted(points)
    if x <= points[0][0]:
        return clip(points[0][1])
    if x >= points[-1][0]:
        return clip(points[-1][1])
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0)
            return clip(y0 + t * (y1 - y0))
    return 0.0

def trend_risk_from_return(ret: Optional[float],
                           bullish_at: float,
                           bearish_at: float) -> Optional[float]:
    if ret is None:
        return None
    if bearish_at == bullish_at:
        return 0.0
    x = (ret - bullish_at) / (bearish_at - bullish_at)
    return clip(2.0 * x - 1.0)

class DataHub:
    def __init__(self, history_days: int = DEFAULT_HISTORY_DAYS):
        self.history_days = history_days
        self.series: Dict[str, DataSeries] = {}
        self.warnings: List[str] = []
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def add(self, key: str, s: pd.Series, source: str, note: str = "") -> None:
        if s is None:
            return
        s = pd.Series(s).copy()
        try:
            s.index = pd.to_datetime(s.index)
            s = s[~s.index.duplicated(keep="last")].sort_index()
        except Exception:
            pass
        s = pd.to_numeric(s, errors="coerce").dropna()
        if len(s) == 0:
            return
        last_date = s.index[-1] if isinstance(s.index, pd.DatetimeIndex) else None
        self.series[key] = DataSeries(key, s, source, last_date, note)

    def get(self, key: str) -> Optional[pd.Series]:
        obj = self.series.get(key)
        return obj.values if obj else None

    def source(self, key: str) -> str:
        return self.series[key].source if key in self.series else "MISSING"

    def fetch_yfinance(self) -> None:
        if yf is None:
            self.warnings.append("未安装 yfinance：海外指数/汇率/商品等自动行情将缺失。")
            return
        end = datetime.now().date() + timedelta(days=1)
        start = end - timedelta(days=self.history_days * 2)
        for key, ticker in YF_TICKERS.items():
            try:
                df = yf.download(
                    ticker,
                    start=start.isoformat(),
                    end=end.isoformat(),
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
                if df is None or df.empty:
                    self.warnings.append(f"yfinance 无数据: {key} ({ticker})")
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    close = df["Close"] if "Close" in df.columns.get_level_values(0) else df.iloc[:, 0]
                    if isinstance(close, pd.DataFrame):
                        close = close.iloc[:, 0]
                else:
                    close = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
                self.add(key, close, f"Yahoo Finance via yfinance ({ticker})")
            except Exception as e:
                self.warnings.append(f"yfinance 获取失败 {key}({ticker}): {e}")

    def fetch_fred(self) -> None:
        api_key = os.getenv("FRED_API_KEY", "").strip()
        if not api_key:
            self.warnings.append(
                "未设置 FRED_API_KEY：US10Y_REAL、HY_OAS 等官方宏观序列可能缺失；"
                "程序会降低置信度，而不是假装数据存在。"
            )
            return

        start = (datetime.now().date() - timedelta(days=self.history_days * 2)).isoformat()
        url = "https://api.stlouisfed.org/fred/series/observations"
        for key, sid in FRED_SERIES.items():
            try:
                r = requests.get(
                    url,
                    params={
                        "series_id": sid,
                        "api_key": api_key,
                        "file_type": "json",
                        "observation_start": start,
                        "sort_order": "asc",
                    },
                    timeout=20,
                )
                r.raise_for_status()
                obs = r.json().get("observations", [])
                rows = [(x["date"], float(x["value"])) for x in obs if x.get("value") not in (None, ".")]
                if rows:
                    self.add(
                        key,
                        pd.Series([v for _, v in rows], index=pd.to_datetime([d for d, _ in rows])),
                        f"FRED:{sid}"
                    )
                else:
                    self.warnings.append(f"FRED 无有效数据: {sid}")
            except Exception as e:
                self.warnings.append(f"FRED 获取失败 {sid}: {e}")

    def fetch_ak_bond_yields(self) -> None:
        if ak is None:
            self.warnings.append("未安装 AKShare：A股横截面、融资余额、中国10Y、BOJ等数据将缺失。")
            return
        try:
            start = (datetime.now().date() - timedelta(days=self.history_days * 2)).strftime("%Y%m%d")
            df = ak.bond_zh_us_rate(start_date=start)
            if df is None or df.empty:
                return
            date_col = self._find_col(df, ["日期", "date"])
            if not date_col:
                self.warnings.append("bond_zh_us_rate 找不到日期列。")
                return
            idx = pd.to_datetime(df[date_col], errors="coerce")
            cn10 = self._find_col(df, ["中国国债收益率10年", "中国10年", "中国国债10年"])
            us10 = self._find_col(df, ["美国国债收益率10年", "美国10年", "美国国债10年"])
            if cn10:
                self.add("CN10Y", pd.Series(pd.to_numeric(df[cn10], errors="coerce").values, index=idx),
                         "AKShare:bond_zh_us_rate")
            if us10 and "US10Y" not in self.series:
                self.add("US10Y", pd.Series(pd.to_numeric(df[us10], errors="coerce").values, index=idx),
                         "AKShare:bond_zh_us_rate (fallback)")
        except Exception as e:
            self.warnings.append(f"AKShare 中美国债收益率获取失败: {e}")

    def fetch_a_index_history(self) -> None:
        """用 AKShare 获取重要A股指数的日线成交额/成交量；用于相对MA20确认。"""
        if ak is None:
            return
        for key, symbol in AK_INDEX_SYMBOLS.items():
            try:
                df = ak.stock_zh_index_daily_em(symbol=symbol)
                if df is None or df.empty:
                    self.warnings.append(f"A股指数历史为空: {key}({symbol})")
                    continue
                date_col = self._find_col(df, ["date", "日期"])
                close_col = self._find_col(df, ["close", "收盘"])
                amount_col = self._find_col(df, ["amount", "成交额"])
                volume_col = self._find_col(df, ["volume", "成交量"])
                if not date_col:
                    self.warnings.append(f"A股指数历史缺日期列: {key}")
                    continue
                idx = pd.to_datetime(df[date_col], errors="coerce")
                if close_col and key not in self.series:
                    self.add(key, pd.Series(pd.to_numeric(df[close_col], errors="coerce").values, index=idx),
                             f"AKShare:stock_zh_index_daily_em({symbol})")
                if amount_col:
                    self.add(key+"_INDEX_AMOUNT",
                             pd.Series(pd.to_numeric(df[amount_col], errors="coerce").values, index=idx),
                             f"AKShare:stock_zh_index_daily_em({symbol})")
                if volume_col:
                    self.add(key+"_INDEX_VOLUME",
                             pd.Series(pd.to_numeric(df[volume_col], errors="coerce").values, index=idx),
                             f"AKShare:stock_zh_index_daily_em({symbol})")
            except Exception as e:
                self.warnings.append(f"A股指数历史获取失败 {key}({symbol}): {e}")

    def fetch_a_share_snapshot(self) -> None:
        if ak is None:
            return
        try:
            df = ak.stock_zh_a_spot_em()
            if df is None or df.empty:
                self.warnings.append("A股实时横截面为空。")
                return

            pct_col = self._find_col(df, ["涨跌幅"])
            amount_col = self._find_col(df, ["成交额"])
            if not pct_col:
                self.warnings.append("A股横截面缺少涨跌幅列，无法计算市场宽度。")
                return

            pct = pd.to_numeric(df[pct_col], errors="coerce").dropna()
            n = len(pct)
            if n == 0:
                return

            adv = float((pct > 0).sum() / n)
            dec = float((pct < 0).sum() / n)
            strong = float((pct >= 3).sum() / n)
            weak = float((pct <= -3).sum() / n)
            approx_limit_down = int((pct <= -9.5).sum())
            approx_limit_down_ratio = float(approx_limit_down / n)

            total_amount = np.nan
            if amount_col:
                total_amount = float(pd.to_numeric(df[amount_col], errors="coerce").sum())

            now = pd.Timestamp.now().normalize()
            self.add("A_BREADTH", pd.Series([adv], index=[now]), "AKShare:stock_zh_a_spot_em")
            self.add("A_DECLINERS", pd.Series([dec], index=[now]), "AKShare:stock_zh_a_spot_em")
            self.add("A_STRONG3", pd.Series([strong], index=[now]), "AKShare:stock_zh_a_spot_em")
            self.add("A_WEAK3", pd.Series([weak], index=[now]), "AKShare:stock_zh_a_spot_em")
            self.add("A_LIMIT_DOWN_APPROX", pd.Series([approx_limit_down], index=[now]), "AKShare:stock_zh_a_spot_em")
            self.add("A_LIMIT_DOWN_RATIO", pd.Series([approx_limit_down_ratio], index=[now]), "AKShare:stock_zh_a_spot_em")
            if not math.isnan(total_amount):
                self.add("A_TURNOVER", pd.Series([total_amount], index=[now]), "AKShare:stock_zh_a_spot_em")

            self._save_snapshot({
                "date": now.strftime("%Y-%m-%d"),
                "breadth": adv,
                "decliners": dec,
                "strong3": strong,
                "weak3": weak,
                "limit_down_approx": approx_limit_down,
                "limit_down_ratio": approx_limit_down_ratio,
                "turnover": None if math.isnan(total_amount) else total_amount,
            })

            hist = self._load_snapshot_history()
            if not hist.empty:
                hist.index = pd.to_datetime(hist["date"])
                if "breadth" in hist:
                    self.add("A_BREADTH_HIST", hist["breadth"], "local snapshot history")
                if "turnover" in hist:
                    self.add("A_TURNOVER_HIST", hist["turnover"], "local snapshot history")
        except Exception as e:
            self.warnings.append(f"A股横截面获取失败: {e}")

    def fetch_margin(self) -> None:
        if ak is None:
            return

        series_list = []
        try:
            end = datetime.now().date()
            start = end - timedelta(days=max(90, self.history_days))
            df = ak.stock_margin_sse(
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d")
            )
            if df is not None and not df.empty:
                date_col = self._find_col(df, ["信用交易日期", "日期"])
                bal_col = self._find_col(df, ["融资余额"])
                if date_col and bal_col:
                    s = pd.Series(
                        pd.to_numeric(df[bal_col], errors="coerce").values,
                        index=pd.to_datetime(df[date_col], errors="coerce"),
                    ).dropna()
                    series_list.append(s.rename("SH"))
        except Exception as e:
            self.warnings.append(f"上交所融资余额获取失败: {e}")

        try:
            if hasattr(ak, "macro_china_market_margin_sz"):
                df = ak.macro_china_market_margin_sz()
                if df is not None and not df.empty:
                    date_col = self._find_col(df, ["日期", "date"])
                    bal_col = self._find_col(df, ["融资余额"])
                    if date_col and bal_col:
                        s = pd.Series(
                            pd.to_numeric(df[bal_col], errors="coerce").values,
                            index=pd.to_datetime(df[date_col], errors="coerce"),
                        ).dropna()
                        series_list.append(s.rename("SZ"))
        except Exception as e:
            self.warnings.append(f"深市融资余额获取失败: {e}")

        if series_list:
            merged = pd.concat(series_list, axis=1).sort_index().ffill()
            merged["TOTAL"] = merged.sum(axis=1, min_count=1)
            self.add("MARGIN_BALANCE", merged["TOTAL"], "AKShare:SSE+SZ margin")

    def fetch_boj_policy(self) -> None:
        if ak is None:
            return
        try:
            if not hasattr(ak, "macro_bank_japan_interest_rate"):
                return
            df = ak.macro_bank_japan_interest_rate()
            if df is None or df.empty:
                return
            date_col = self._find_col(df, ["日期", "date"])
            val_col = self._find_col(df, ["今值", "实际值", "利率"])
            if date_col and val_col:
                idx = pd.to_datetime(df[date_col], errors="coerce")
                vals = df[val_col].astype(str).str.replace("%", "", regex=False).replace({"--": np.nan})
                self.add("BOJ_RATE",
                         pd.Series(pd.to_numeric(vals, errors="coerce").values, index=idx).dropna(),
                         "AKShare:macro_bank_japan_interest_rate")
        except Exception as e:
            self.warnings.append(f"日本央行利率获取失败: {e}")

    def load_manual_overrides(self, path: Path = MANUAL_FILE) -> None:
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            self.warnings.append(f"manual_overrides.json 解析失败: {e}")
            return

        now = pd.Timestamp.now().normalize()
        for key, obj in data.items():
            try:
                if isinstance(obj, (int, float)):
                    self.add(key, pd.Series([float(obj)], index=[now]), "manual")
                elif isinstance(obj, dict) and "value" in obj:
                    date = pd.to_datetime(obj.get("date", now))
                    self.add(key, pd.Series([float(obj["value"])], index=[date]), "manual")
                elif isinstance(obj, list):
                    idx, vals = [], []
                    for row in obj:
                        idx.append(pd.to_datetime(row["date"]))
                        vals.append(float(row["value"]))
                    self.add(key, pd.Series(vals, index=idx), "manual")
            except Exception as e:
                self.warnings.append(f"手工覆盖 {key} 读取失败: {e}")

    @staticmethod
    def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        cols = [str(c) for c in df.columns]
        for cand in candidates:
            for c in cols:
                if cand.lower() == c.lower():
                    return c
        for cand in candidates:
            for c in cols:
                if cand.lower() in c.lower():
                    return c
        return None

    def _save_snapshot(self, row: Dict[str, Any]) -> None:
        path = STATE_DIR / "a_market_snapshot.csv"
        new = pd.DataFrame([row])
        if path.exists():
            old = pd.read_csv(path)
            all_df = pd.concat([old, new], ignore_index=True)
            all_df = all_df.drop_duplicates(subset=["date"], keep="last").sort_values("date")
        else:
            all_df = new
        all_df.to_csv(path, index=False, encoding="utf-8-sig")

    def _load_snapshot_history(self) -> pd.DataFrame:
        path = STATE_DIR / "a_market_snapshot.csv"
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    def data_quality_warnings(self) -> List[str]:
        out = []
        now = pd.Timestamp.now().normalize()
        for key, ds in self.series.items():
            if ds.last_date is None:
                continue
            last = pd.Timestamp(ds.last_date)
            if last.tzinfo is not None:
                last = last.tz_convert(None)
            age = (now - last.normalize()).days
            src = ds.source.lower()
            if "fred" in src:
                max_age = MAX_AGE_DAYS["fred"]
            elif "manual" in src:
                max_age = MAX_AGE_DAYS["manual"]
            elif "snapshot" in src:
                max_age = MAX_AGE_DAYS["snapshot"]
            elif "akshare" in src and ("macro" in src or "bond" in src):
                max_age = MAX_AGE_DAYS["ak_macro"]
            else:
                max_age = MAX_AGE_DAYS["market"]
            if age > max_age:
                out.append(f"数据可能过期: {key} 最后日期={last.date()}, 已滞后 {age} 天, 来源={ds.source}")
        return out

def factor_missing(name, group, weight, source="MISSING", detail="缺失") -> FactorResult:
    return FactorResult(name, group, weight, None, None, detail, source, True)

def make_factor(name: str, group: str, weight: float, signal: Optional[float],
                value: Optional[float], detail: str, source: str) -> FactorResult:
    if signal is None or value is None:
        return factor_missing(name, group, weight, source, detail)
    return FactorResult(name, group, weight, clip(signal), float(value), detail, source, False)

class FactorEngine:
    def __init__(self, hub: DataHub):
        self.hub = hub
        self.features: Dict[str, Optional[float]] = {}

    def build_features(self) -> Dict[str, Optional[float]]:
        h = self.hub
        f: Dict[str, Optional[float]] = {}

        for key in ["US10Y", "US10Y_REAL", "US2Y", "HY_OAS", "FED_FUNDS", "CN10Y", "BOJ_RATE"]:
            s = h.get(key)
            f[key] = latest(s)
            f[key+"_5D_BP"] = bp_change_n(s, 5)
            f[key+"_20D_BP"] = bp_change_n(s, 20)
            f[key+"_Z60"] = zscore_last(s, 60)

        if f.get("CN10Y") is not None and f.get("US10Y") is not None:
            f["CN_US_10Y_SPREAD_BP"] = (f["CN10Y"] - f["US10Y"]) * 100.0
            cn, us = h.get("CN10Y"), h.get("US10Y")
            x = pd.concat([cn.rename("cn"), us.rename("us")], axis=1).ffill().dropna()
            spread = (x["cn"] - x["us"]) * 100.0
            f["CN_US_SPREAD_20D_CHG_BP"] = float(spread.iloc[-1] - spread.iloc[-21]) if len(spread) > 21 else None
        else:
            f["CN_US_10Y_SPREAD_BP"] = None
            f["CN_US_SPREAD_20D_CHG_BP"] = None

        for key in [
            "SSE","CSI300","CSI1000","CHINEXT","STAR50","HSI","HSTECH","A50",
            "VIX","NASDAQ100","SOX","DXY","NIKKEI","KOSPI","USDCNH","USDJPY",
            "COPPER","OIL","IRON_ORE"
        ]:
            s = h.get(key)
            f[key] = latest(s)
            f[key+"_1D"] = pct_change_n(s, 1)
            f[key+"_5D"] = pct_change_n(s, 5)
            f[key+"_20D"] = pct_change_n(s, 20)
            f[key+"_Z60"] = zscore_last(s, 60)
            f[key+"_SLOPE20"] = linear_slope_per_day(s, 20)

        for key in ["A_BREADTH","A_DECLINERS","A_STRONG3","A_WEAK3","A_LIMIT_DOWN_APPROX","A_LIMIT_DOWN_RATIO","A_TURNOVER"]:
            f[key] = latest(h.get(key))
        if f.get("A_WEAK3") is not None and f.get("A_STRONG3") is not None:
            f["A_WEAK_STRONG_GAP"] = f["A_WEAK3"] - f["A_STRONG3"]
        else:
            f["A_WEAK_STRONG_GAP"] = None

        # 重要指数成交额/成交量相对20日均值。这里不用静态“多少亿”阈值。
        for key in ["SSE","CSI300","CSI1000","CHINEXT","STAR50"]:
            amt = h.get(key+"_INDEX_AMOUNT")
            vol = h.get(key+"_INDEX_VOLUME")
            if amt is not None and len(amt.dropna()) >= 10:
                a = amt.dropna().astype(float)
                ma20 = a.tail(20).mean()
                f[key+"_AMOUNT_RATIO20"] = float(a.iloc[-1] / ma20) if ma20 > 0 else None
                f[key+"_AMOUNT_5D"] = pct_change_n(a, 5)
            else:
                f[key+"_AMOUNT_RATIO20"] = None
                f[key+"_AMOUNT_5D"] = None
            if vol is not None and len(vol.dropna()) >= 10:
                v = vol.dropna().astype(float)
                ma20v = v.tail(20).mean()
                f[key+"_VOLUME_RATIO20"] = float(v.iloc[-1] / ma20v) if ma20v > 0 else None
            else:
                f[key+"_VOLUME_RATIO20"] = None

        turnover_hist = h.get("A_TURNOVER_HIST")
        if turnover_hist is not None and len(turnover_hist.dropna()) >= 10:
            th = turnover_hist.dropna().astype(float)
            base = th.tail(20).mean()
            f["A_TURNOVER_MA20_RATIO"] = float(th.iloc[-1] / base) if base > 0 else None
        else:
            f["A_TURNOVER_MA20_RATIO"] = None

        margin = h.get("MARGIN_BALANCE")
        f["MARGIN_BALANCE"] = latest(margin)
        f["MARGIN_5D"] = pct_change_n(margin, 5)
        f["MARGIN_20D"] = pct_change_n(margin, 20)

        for key in ["ETF_FLOW_5D_BN", "IF_BASIS_PCT", "IC_BASIS_PCT", "IM_BASIS_PCT"]:
            f[key] = latest(h.get(key))

        self.features = f
        return f

    def evaluate(self) -> List[FactorResult]:
        if not self.features:
            self.build_features()
        f, h, R = self.features, self.hub, []

        x = f.get("US10Y")
        sig = None if x is None else piecewise_risk(
            x, [(3.5,-0.4),(4.0,-0.1),(4.3,0.15),(4.5,0.35),(4.8,0.70),(5.0,0.90),(5.2,1.0)]
        )
        R.append(make_factor("美债10Y绝对水平","全球利率",5.0,sig,x,
            f"10Y={x:.2f}%（绝对值仅低权重使用）" if x is not None else "缺失", h.source("US10Y")))

        d5, d20 = f.get("US10Y_5D_BP"), f.get("US10Y_20D_BP")
        if d5 is None and d20 is None:
            sig, val = None, x
        else:
            s5 = 0 if d5 is None else piecewise_risk(d5, [(-30,-1),(-10,-0.4),(0,0),(15,0.35),(30,0.75),(45,1)])
            s20 = 0 if d20 is None else piecewise_risk(d20, [(-50,-1),(-20,-0.4),(0,0),(25,0.35),(40,0.70),(55,1)])
            sig, val = clip(0.45*s5 + 0.55*s20), d20 if d20 is not None else d5
        R.append(make_factor("美债10Y斜率/变化速度","全球利率",8.0,sig,val,
            f"5日={d5}bp, 20日={d20}bp；快速上行权重高于绝对值", h.source("US10Y")))

        x = f.get("US10Y_REAL")
        sig = None if x is None else piecewise_risk(
            x, [(0.5,-0.8),(1.0,-0.45),(1.5,0.0),(2.0,0.45),(2.4,0.80),(2.5,0.92),(2.8,1.0)]
        )
        R.append(make_factor("美债10Y实际利率","全球利率",8.5,sig,x,
            f"实际利率={x:.2f}%；对高估值成长股尤敏感" if x is not None else "缺失", h.source("US10Y_REAL")))

        d5, d20 = f.get("US2Y_5D_BP"), f.get("US2Y_20D_BP")
        if d5 is None and d20 is None:
            sig, val = None, f.get("US2Y")
        else:
            s5 = 0 if d5 is None else piecewise_risk(d5, [(-30,-1),(-10,-0.3),(0,0),(15,0.4),(25,0.75),(40,1)])
            s20 = 0 if d20 is None else piecewise_risk(d20, [(-50,-1),(-20,-0.4),(0,0),(30,0.5),(50,1)])
            sig, val = clip(0.6*s5 + 0.4*s20), d5 if d5 is not None else d20
        R.append(make_factor("美债2Y/Fed预期重定价","全球利率",5.0,sig,val,
            f"2Y: 5日={d5}bp, 20日={d20}bp", h.source("US2Y")))

        d20 = f.get("FED_FUNDS_20D_BP")
        sig = None if d20 is None else piecewise_risk(d20, [(-50,-0.5),(-25,-0.25),(0,0),(25,0.25),(50,0.5)])
        R.append(make_factor("美联储政策利率方向","央行政策",2.0,sig,f.get("FED_FUNDS"),
            f"Fed Funds={f.get('FED_FUNDS')}%, 20日变化={d20}bp；低权重", h.source("FED_FUNDS")))

        d20 = f.get("BOJ_RATE_20D_BP")
        sig = None if d20 is None else piecewise_risk(d20, [(-25,-0.2),(0,0),(10,0.25),(25,0.6),(50,1)])
        R.append(make_factor("日本央行政策利率方向","央行政策",2.0,sig,f.get("BOJ_RATE"),
            f"BOJ={f.get('BOJ_RATE')}%, 20日变化={d20}bp", h.source("BOJ_RATE")))

        x, d5 = f.get("HY_OAS"), f.get("HY_OAS_5D_BP")
        if x is None:
            sig = None
        else:
            level = piecewise_risk(x, [(2.5,-0.6),(3.5,-0.1),(4.0,0.15),(4.5,0.4),(5.5,0.75),(7.0,1)])
            speed = 0 if d5 is None else piecewise_risk(d5, [(-50,-0.5),(0,0),(20,0.3),(50,0.7),(100,1)])
            sig = clip(0.65*level + 0.35*speed)
        R.append(make_factor("美国高收益债OAS","全球信用",4.0,sig,x,
            f"HY OAS={x}%, 5日变化={d5}bp", h.source("HY_OAS")))

        d5, d20 = f.get("USDCNH_5D"), f.get("USDCNH_20D")
        if d5 is None and d20 is None:
            sig, val = None, f.get("USDCNH")
        else:
            s5 = 0 if d5 is None else trend_risk_from_return(d5, -1.0, 1.2)
            s20 = 0 if d20 is None else trend_risk_from_return(d20, -2.0, 2.5)
            sig, val = clip(0.55*s5 + 0.45*s20), d5 if d5 is not None else d20
        R.append(make_factor("离岸人民币USD/CNH","汇率",8.5,sig,val,
            f"现值={f.get('USDCNH')}, 5日={d5}%, 20日={d20}%；上涨=人民币走弱", h.source("USDCNH")))

        d5, d20 = f.get("DXY_5D"), f.get("DXY_20D")
        if d5 is None and d20 is None:
            sig, val = None, f.get("DXY")
        else:
            s5 = 0 if d5 is None else trend_risk_from_return(d5, -1.5, 1.5)
            s20 = 0 if d20 is None else trend_risk_from_return(d20, -3.0, 3.0)
            sig, val = clip(0.55*s5 + 0.45*s20), d5 if d5 is not None else d20
        R.append(make_factor("美元指数DXY","汇率",4.5,sig,val,
            f"DXY={f.get('DXY')}, 5日={d5}%, 20日={d20}%", h.source("DXY")))

        spread, spread_chg = f.get("CN_US_10Y_SPREAD_BP"), f.get("CN_US_SPREAD_20D_CHG_BP")
        if spread is None:
            sig = None
        else:
            level = piecewise_risk(spread, [(-300,0.9),(-200,0.55),(-100,0.2),(0,-0.1),(100,-0.3)])
            chg = 0 if spread_chg is None else piecewise_risk(spread_chg, [(-60,0.8),(-30,0.4),(0,0),(30,-0.3),(60,-0.6)])
            sig = clip(0.65*level + 0.35*chg)
        R.append(make_factor("中美10Y国债利差","汇率",3.0,sig,spread,
            f"CN-US={spread}bp, 20日变化={spread_chg}bp", f"{h.source('CN10Y')} + {h.source('US10Y')}"))

        x, v5 = f.get("VIX"), f.get("VIX_5D")
        if x is None:
            sig = None
        else:
            level = piecewise_risk(x, [(12,-0.7),(15,-0.4),(20,0.0),(25,0.45),(30,0.7),(40,0.9),(50,1.0)])
            speed = 0 if v5 is None else trend_risk_from_return(v5, -20, 40)
            sig = clip(0.75*level + 0.25*speed)
        R.append(make_factor("VIX恐慌指数","全球风险",7.5,sig,x,
            f"VIX={x}, 5日={v5}%", h.source("VIX")))

        equity_specs = [
            ("HSTECH","恒生科技","中国资产外盘",6.0,-6.0,6.0),
            ("HSI","恒生指数","中国资产外盘",3.0,-5.0,5.0),
            ("A50","富时中国A50","中国资产外盘",4.0,-5.0,5.0),
            ("NASDAQ100","纳斯达克100","全球科技",3.5,-5.0,5.0),
            ("SOX","费城半导体SOX","全球科技",4.5,-7.0,7.0),
            ("NIKKEI","日经225","亚洲风险",2.0,-6.0,6.0),
            ("KOSPI","韩国KOSPI","亚洲/半导体周期",1.5,-6.0,6.0),
        ]
        for key, name, group, weight, bull, bear in equity_specs:
            r1, r5, r20 = f.get(key+"_1D"), f.get(key+"_5D"), f.get(key+"_20D")
            if r5 is None and r20 is None:
                sig, val = None, f.get(key)
            else:
                s1 = 0 if r1 is None else trend_risk_from_return(r1, bull/5, bear/5)
                s5 = 0 if r5 is None else trend_risk_from_return(r5, bull, bear)
                s20 = 0 if r20 is None else trend_risk_from_return(r20, bull*2, bear*2)
                sig, val = clip(-(0.20*s1 + 0.50*s5 + 0.30*s20)), r5 if r5 is not None else r20
            R.append(make_factor(name, group, weight, sig, val,
                f"1日={r1}%, 5日={r5}%, 20日={r20}%；看变化速度", h.source(key)))

        r1, r5 = f.get("USDJPY_1D"), f.get("USDJPY_5D")
        if r1 is None and r5 is None:
            sig, val = None, f.get("USDJPY")
        else:
            s1 = 0 if r1 is None else piecewise_risk(-r1, [(-2,-0.4),(0,0),(1,0.25),(2,0.6),(3.5,1)])
            s5 = 0 if r5 is None else piecewise_risk(-r5, [(-4,-0.5),(0,0),(2,0.3),(4,0.7),(6,1)])
            sig, val = clip(0.4*s1 + 0.6*s5), r5 if r5 is not None else r1
        R.append(make_factor("日元套息平仓风险","亚洲流动性",3.0,sig,val,
            f"USDJPY 1日={r1}%, 5日={r5}%；负值越大=日元升值越快", h.source("USDJPY")))

        c20 = f.get("COPPER_20D")
        sig = None if c20 is None else piecewise_risk(c20, [(-20,1),(-10,0.6),(-3,0.15),(0,0),(8,-0.3),(15,-0.4),(25,-0.2)])
        R.append(make_factor("铜价趋势","商品/增长",1.5,sig,c20,
            f"铜20日={c20}%；大跌偏风险，大涨只给有限利多", h.source("COPPER")))

        o20 = f.get("OIL_20D")
        if o20 is None:
            sig = None
        elif o20 >= 0:
            sig = piecewise_risk(o20, [(0,0),(8,0.15),(15,0.5),(25,0.9),(35,1)])
        else:
            sig = piecewise_risk(-o20, [(0,0),(10,0.15),(20,0.45),(30,0.75),(40,1)])
        R.append(make_factor("原油趋势","商品/通胀",1.5,sig,o20,
            f"原油20日={o20}%；暴涨=通胀风险，暴跌=需求/信用风险", h.source("OIL")))

        i20 = f.get("IRON_ORE_20D")
        sig = None if i20 is None else piecewise_risk(i20, [(-25,0.8),(-12,0.45),(-3,0.1),(0,0),(10,-0.2),(20,-0.25)])
        R.append(make_factor("铁矿石趋势","商品/中国需求",1.0,sig,i20,
            f"铁矿20日={i20}%；仅作为中国周期温度计", h.source("IRON_ORE")))

        r5, r20 = f.get("CSI300_5D"), f.get("CSI300_20D")
        if r5 is None and r20 is None:
            sig, val = None, f.get("CSI300")
        else:
            s5 = 0 if r5 is None else trend_risk_from_return(r5, -5, 5)
            s20 = 0 if r20 is None else trend_risk_from_return(r20, -10, 10)
            sig, val = clip(-(0.6*s5 + 0.4*s20)), r5 if r5 is not None else r20
        R.append(make_factor("沪深300自身趋势","A股内部",5.0,sig,val,
            f"5日={r5}%, 20日={r20}%；确认外部冲击是否传入", h.source("CSI300")))

        for key, name, w in [("CHINEXT","创业板趋势",2.0),("STAR50","科创50趋势",2.0),("CSI1000","中证1000趋势",2.0)]:
            r5, r20 = f.get(key+"_5D"), f.get(key+"_20D")
            if r5 is None and r20 is None:
                sig, val = None, f.get(key)
            else:
                s5 = 0 if r5 is None else trend_risk_from_return(r5, -6, 6)
                s20 = 0 if r20 is None else trend_risk_from_return(r20, -12, 12)
                sig, val = clip(-(0.6*s5 + 0.4*s20)), r5 if r5 is not None else r20
            R.append(make_factor(name,"A股内部",w,sig,val,f"5日={r5}%, 20日={r20}%",h.source(key)))

        b = f.get("A_BREADTH")
        sig = None if b is None else piecewise_risk(
            b, [(0.15,1.0),(0.25,0.75),(0.35,0.35),(0.50,0.0),(0.60,-0.35),(0.70,-0.65),(0.80,-0.9)]
        )
        R.append(make_factor("A股上涨家数比例","A股内部",5.5,sig,b,
            f"上涨比例={None if b is None else round(b*100,1)}%；<30%偏空，>60%偏多", h.source("A_BREADTH")))

        gap = f.get("A_WEAK_STRONG_GAP")
        sig = None if gap is None else piecewise_risk(
            gap, [(-0.20,-0.8),(-0.10,-0.45),(0,0),(0.10,0.45),(0.20,0.8),(0.30,1.0)]
        )
        R.append(make_factor("A股强弱扩散差","A股内部",2.5,sig,gap,
            f"跌幅>=3%占比 - 涨幅>=3%占比 = {gap}；正值越大说明弱股扩散", h.source("A_BREADTH")))

        ldr = f.get("A_LIMIT_DOWN_RATIO")
        sig = None if ldr is None else piecewise_risk(
            ldr, [(0.0002,-0.1),(0.001,0.1),(0.003,0.35),(0.006,0.65),(0.012,0.9),(0.02,1.0)]
        )
        R.append(make_factor("A股近似跌停比例","A股内部",2.0,sig,ldr,
            f"近似跌停股占全部A股={None if ldr is None else round(ldr*100,3)}%；使用比例而非固定家数",
            h.source("A_LIMIT_DOWN_RATIO")))

        # 重要指数：方向 × 放量确认。下跌放量 -> 风险增加；上涨放量 -> 偏多。
        for key, name, w in [
            ("CSI300","沪深300成交额确认",1.5),
            ("CSI1000","中证1000成交额确认",1.2),
            ("CHINEXT","创业板成交额确认",1.2),
            ("STAR50","科创50成交额确认",1.2),
        ]:
            ratio = f.get(key+"_AMOUNT_RATIO20")
            r5 = f.get(key+"_5D")
            if ratio is None or r5 is None:
                sig = None
                val = ratio
            else:
                direction = clip(-r5 / 5.0)  # 跌5%约 +1，涨5%约 -1
                volume_conviction = float(np.clip((ratio - 0.75) / 0.65, 0, 1))
                sig = clip(direction * volume_conviction)
                val = ratio
            R.append(make_factor(name,"A股成交结构",w,sig,val,
                f"成交额/MA20={ratio}, 指数5日={r5}%；放量只用于确认方向",
                h.source(key+"_INDEX_AMOUNT")))

        tr = f.get("A_TURNOVER_MA20_RATIO")
        sig = None if tr is None else piecewise_risk(tr, [(0.55,0.35),(0.70,0.2),(0.9,0.05),(1.0,0),(1.2,-0.05),(1.5,0)])
        R.append(make_factor("A股成交额/MA20","A股内部",2.0,sig,tr,
            f"成交额比={tr}；量本身无方向，主要用于共振规则", h.source("A_TURNOVER_HIST")))

        m5, m20 = f.get("MARGIN_5D"), f.get("MARGIN_20D")
        if m5 is None and m20 is None:
            sig, val = None, f.get("MARGIN_BALANCE")
        else:
            s5 = 0 if m5 is None else trend_risk_from_return(m5, -2.0, 2.0)
            s20 = 0 if m20 is None else trend_risk_from_return(m20, -4.0, 4.0)
            sig, val = clip(-(0.6*s5 + 0.4*s20)), m5 if m5 is not None else m20
        R.append(make_factor("融资余额变化","A股内部",2.5,sig,val,
            f"5日={m5}%, 20日={m20}%；持续下降偏风险", h.source("MARGIN_BALANCE")))

        etf = f.get("ETF_FLOW_5D_BN")
        sig = None if etf is None else piecewise_risk(etf, [(-500,1),(-200,0.6),(-50,0.2),(0,0),(100,-0.3),(300,-0.7),(600,-1)])
        R.append(make_factor("主要宽基ETF 5日净流入","A股资金",2.0,sig,etf,
            f"5日净流入={etf}亿元；正值偏多", h.source("ETF_FLOW_5D_BN")))

        for key, name, w in [("IF_BASIS_PCT","IF基差",1.0),("IC_BASIS_PCT","IC基差",0.8),("IM_BASIS_PCT","IM基差",0.8)]:
            x = f.get(key)
            sig = None if x is None else piecewise_risk(x, [(-3,1),(-1.5,0.6),(-0.5,0.2),(0,0),(0.5,-0.1),(1.5,-0.2)])
            R.append(make_factor(name,"A股衍生品",w,sig,x,f"{key}={x}%；负值贴水偏风险",h.source(key)))

        return R

def compute_resonance(f: Dict[str, Optional[float]]) -> Tuple[float, List[str]]:
    adj, notes = 0.0, []
    us10_20, dxy5, cnh5 = f.get("US10Y_20D_BP"), f.get("DXY_5D"), f.get("USDCNH_5D")
    real10, vix = f.get("US10Y_REAL"), f.get("VIX")
    hstech1, sox1 = f.get("HSTECH_1D"), f.get("SOX_1D")
    jpy5, nik5 = f.get("USDJPY_5D"), f.get("NIKKEI_5D")
    breadth, turnover = f.get("A_BREADTH"), f.get("A_TURNOVER_MA20_RATIO")
    hs5, csi5 = f.get("HSTECH_5D"), f.get("CSI300_5D")

    if us10_20 is not None and us10_20 >= 40 and dxy5 is not None and dxy5 >= 1.0 and cnh5 is not None and cnh5 >= 1.0:
        adj += 8; notes.append("红色共振：10Y美债20日+40bp以上 + DXY走强 + CNH贬值。")
    if real10 is not None and real10 >= 2.0 and cnh5 is not None and cnh5 >= 1.0 and vix is not None and vix >= 25:
        adj += 7; notes.append("红色共振：实际利率>=2% + CNH一周明显走弱 + VIX>=25。")
    if vix is not None and vix >= 25 and hstech1 is not None and hstech1 <= -3 and sox1 is not None and sox1 <= -3:
        adj += 7; notes.append("红色共振：VIX>=25 + 恒生科技单日<-3% + SOX单日<-3%。")
    if jpy5 is not None and jpy5 <= -4 and nik5 is not None and nik5 <= -5:
        adj += 6; notes.append("红色共振：5日日元快速升值 + 日经大跌。")
    if breadth is not None and breadth < 0.30 and turnover is not None and turnover >= 1.20:
        adj += 6; notes.append("内部确认：上涨家数<30% 且成交额>=MA20×1.2。")
    if us10_20 is not None and us10_20 <= -30 and dxy5 is not None and dxy5 <= -1.0 and cnh5 is not None and cnh5 <= -1.0:
        adj -= 7; notes.append("绿色共振：美债快速下行 + 美元走弱 + 人民币升值。")
    if hs5 is not None and hs5 >= 5 and breadth is not None and breadth >= 0.60 and turnover is not None and turnover >= 1.20 and csi5 is not None and csi5 > 0:
        adj -= 7; notes.append("绿色共振：恒生科技强 + A股宽度>60% + 放量 + 沪深300上涨。")
    return float(adj), notes

def rule_decision_tree(buy: float, sell: float, confidence: float,
                       missing_critical: List[str],
                       f: Dict[str, Optional[float]]) -> Tuple[str, List[str]]:
    path = []
    if confidence < 65 or len(missing_critical) >= 3:
        path.append(f"数据置信度={confidence:.1f}，或关键数据缺失过多 -> DATA_INCOMPLETE")
        return "DATA_INCOMPLETE / 不根据信号交易", path

    vix, cnh5 = f.get("VIX"), f.get("USDCNH_5D")
    real10, us10_20 = f.get("US10Y_REAL"), f.get("US10Y_20D_BP")
    breadth, turnover = f.get("A_BREADTH"), f.get("A_TURNOVER_MA20_RATIO")

    if vix is not None and vix >= 30 and cnh5 is not None and cnh5 >= 1.0:
        path.append("VIX>=30 且 CNH 5日贬值>=1% -> 强风险规避")
        return "RISK_OFF / 显著降低仓位", path

    if real10 is not None and real10 >= 2.0 and us10_20 is not None and us10_20 >= 40 and cnh5 is not None and cnh5 >= 1.0:
        path.append("实际利率>=2% + 10Y美债20日+40bp + CNH走弱 -> 利率/汇率三杀")
        return "REDUCE / 偏卖出", path

    if sell >= 68:
        path.append(f"综合卖出分={sell:.1f}>=68 -> 偏卖出")
        return "REDUCE / 偏卖出", path

    if buy >= 65 and breadth is not None and breadth >= 0.60 and (turnover is None or turnover >= 0.9):
        path.append(f"买入分={buy:.1f}>=65 + 市场宽度>=60% -> 偏买入")
        return "BUY_BIAS / 分批偏买入", path

    if buy >= 60:
        path.append(f"买入分={buy:.1f}>=60，但内部确认不足 -> 观察/小仓试错")
        return "WATCH_BUY / 观察偏多", path

    path.append("买卖分均未达到强阈值 -> HOLD")
    return "HOLD / 中性等待", path

def score_engine(factors: List[FactorResult],
                 features: Dict[str, Optional[float]],
                 hub: DataHub) -> EngineResult:
    valid = [x for x in factors if not x.missing and x.signal is not None]
    total_weight = sum(x.weight for x in factors)
    valid_weight = sum(x.weight for x in valid)

    base_risk = 50.0 if valid_weight <= 0 else (
        50.0 + 50.0 * sum(x.weight*x.signal for x in valid) / valid_weight
    )
    resonance, resonance_notes = compute_resonance(features)
    risk = float(np.clip(base_risk + resonance, 0, 100))
    sell, buy = risk, 100.0-risk

    weight_coverage = valid_weight / total_weight if total_weight else 0
    missing_critical = [k for k in sorted(CRITICAL_KEYS) if features.get(k) is None]
    critical_coverage = 1.0 - len(missing_critical) / len(CRITICAL_KEYS)
    confidence = float(np.clip(100.0*(0.65*weight_coverage + 0.35*critical_coverage), 0, 100))

    warnings = list(hub.warnings) + hub.data_quality_warnings()
    if missing_critical:
        warnings.append("关键数据缺失: " + ", ".join(missing_critical))

    action, path = rule_decision_tree(buy, sell, confidence, missing_critical, features)
    path = resonance_notes + path

    if sell >= 75: level = "极高"
    elif sell >= 65: level = "高"
    elif sell >= 55: level = "中高"
    elif sell >= 45: level = "中性"
    elif sell >= 35: level = "中低"
    else: level = "低"

    return EngineResult(
        datetime.now().astimezone().isoformat(timespec="seconds"),
        round(buy,1), round(sell,1), round(confidence,1), action, level,
        round(resonance,1), missing_critical, warnings, factors, path
    )

def decision_tree_dot() -> str:
    # Font fallback: use a cross-platform safe font stack so that Chinese
    # characters render on systems without Microsoft YaHei.  Graphviz will
    # try each font in the fontname value; if none is available the labels
    # may appear as boxes, but the ASCII text version (decision_tree_text.txt)
    # will always be readable regardless of font support.
    return r"""digraph AShareRiskTree {
    rankdir=TB;
    graph [fontname="Arial Unicode MS,WenQuanYi Micro Hei,DejaVu Sans,sans-serif"];
    node [shape=box, style="rounded", fontname="Arial Unicode MS,WenQuanYi Micro Hei,DejaVu Sans,sans-serif"];
    edge [fontname="Arial Unicode MS,WenQuanYi Micro Hei,DejaVu Sans,sans-serif"];

    A [label="Confidence>=65% & missing_critical<3?\n(数据置信度>=65% 且关键缺失<3)"];
    B [label="DATA_INCOMPLETE\n(no trade / 不根据信号交易)"];
    C [label="VIX>=30 AND CNH 5d depreciation>=1%?\n(VIX>=30 且 CNH 5日贬值>=1%)"];
    D [label="RISK_OFF\n(reduce position / 显著降低仓位)"];
    E [label="Real yield>=2% AND 10Y +40bp/20d AND CNH 5d>=1%?\n(实际利率>=2% 且10Y 20日+40bp 且CNH走弱)"];
    F [label="REDUCE\n(lean sell / 偏卖出)"];
    G [label="sell_score >= 68?\n(卖出分>=68)"];
    H [label="buy_score>=65 AND breadth>=60%?\n(买入分>=65 且上涨家数>=60%)"];
    I [label="BUY_BIAS\n(scale in / 分批偏买入)"];
    J [label="buy_score >= 60?\n(买入分>=60)"];
    K [label="WATCH_BUY\n(observe lean long / 观察偏多)"];
    L [label="HOLD\n(neutral wait / 中性等待)"];

    A -> B [label="No/否"];
    A -> C [label="Yes/是"];
    C -> D [label="Yes/是"];
    C -> E [label="No/否"];
    E -> F [label="Yes/是"];
    E -> G [label="No/否"];
    G -> F [label="Yes/是"];
    G -> H [label="No/否"];
    H -> I [label="Yes/是"];
    H -> J [label="No/否"];
    J -> K [label="Yes/是"];
    J -> L [label="No/否"];
}"""

def decision_tree_text() -> str:
    """
    Plain-text / ASCII representation of the decision tree.
    Always readable regardless of font or graphviz availability.
    """
    return """\
A-Share Risk Engine – Decision Tree (ASCII / text version)
============================================================

[ROOT] Confidence >= 65% AND missing_critical < 3?
  |
  +--[No/否]--> DATA_INCOMPLETE  (不根据信号交易)
  |             Action: do not trade based on signals.
  |             → Check directional bias in terminal summary for tendency.
  |
  +--[Yes/是]-> VIX >= 30 AND CNH 5-day depreciation >= 1%?
                |
                +--[Yes/是]--> RISK_OFF  (显著降低仓位)
                |              Action: significantly reduce positions.
                |
                +--[No/否]--> Real yield >= 2%
                              AND 10Y up +40 bp/20d
                              AND CNH 5d depreciation >= 1%?
                              |
                              +--[Yes/是]--> REDUCE  (偏卖出)
                              |              Action: lean sell.
                              |
                              +--[No/否]--> sell_score >= 68?
                                            |
                                            +--[Yes/是]--> REDUCE  (偏卖出)
                                            |
                                            +--[No/否]--> buy_score >= 65
                                                          AND breadth >= 60%?
                                                          |
                                                          +--[Yes/是]--> BUY_BIAS  (分批偏买入)
                                                          |              Action: scale in.
                                                          |
                                                          +--[No/否]--> buy_score >= 60?
                                                                        |
                                                                        +--[Yes/是]--> WATCH_BUY  (观察偏多)
                                                                        |
                                                                        +--[No/否]--> HOLD  (中性等待)

Notes:
  - Signal convention: signal > 0 → bearish contribution; signal < 0 → bullish.
  - buy_score + sell_score = 100.
  - Resonance adjustments (共振) can shift sell_score by ±6–8 pts per rule.
  - See output/factor_report.csv for per-factor details.
"""

def save_decision_tree() -> Tuple[Path, Optional[Path]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dot_path = OUTPUT_DIR / "decision_tree.dot"
    dot_path.write_text(decision_tree_dot(), encoding="utf-8")
    # Always write a portable plain-text version (readable even if PNG fonts are garbled)
    text_path = OUTPUT_DIR / "decision_tree_text.txt"
    text_path.write_text(decision_tree_text(), encoding="utf-8")
    try:
        import graphviz
        src = graphviz.Source(decision_tree_dot())
        rendered = Path(src.render(filename="decision_tree", directory=str(OUTPUT_DIR), format="png", cleanup=True))
        return dot_path, rendered
    except Exception:
        return dot_path, None

def factor_dataframe(factors: List[FactorResult]) -> pd.DataFrame:
    return pd.DataFrame([{
        "group": x.group,
        "factor": x.name,
        "weight": x.weight,
        "signal_-1bull_+1bear": None if x.signal is None else round(x.signal,3),
        "weighted_contribution": None if x.contribution is None else round(x.contribution,3),
        "value": x.value,
        "missing": x.missing,
        "source": x.source,
        "detail": x.detail,
    } for x in factors])

def compute_directional_bias(result: EngineResult) -> dict:
    """
    Compute a human-readable directional bias breakdown even when the
    executable action is DATA_INCOMPLETE.

    Mapping logic (documented):
    ─────────────────────────────────────────────────────────────────────
    The engine already produces buy_score (0-100) and sell_score (0-100)
    where buy_score + sell_score == 100.  We further split the "middle"
    band into a WATCH bucket so users get three categories:

      SELL  = sell_score  (higher → more bearish)
      BUY   = buy_score   (higher → more bullish)
      WATCH = abs(buy_score - sell_score) < 10 bonus allocation

    Normalised bias weights are derived as follows:
    1. raw_buy  = buy_score / 100
    2. raw_sell = sell_score / 100
    3. watch_mass = max(0, 0.30 - abs(raw_buy - raw_sell)) * (1/0.30)
       i.e. when the spread is very small (< 30 pp) some weight shifts
       to WATCH, up to a maximum of ~30% of total.
    4. Remaining weight is split proportionally between buy and sell.
    5. All three are renormalised to sum to 1.

    This is a *directional bias score*, NOT a calibrated probability.
    ─────────────────────────────────────────────────────────────────────
    """
    raw_buy  = result.buy_score  / 100.0
    raw_sell = result.sell_score / 100.0
    spread   = abs(raw_buy - raw_sell)

    # Watch mass: maximum 0.30, tapers to 0 at spread ≥ 0.30
    watch_mass = max(0.0, 0.30 - spread)          # in [0, 0.30]
    remaining  = 1.0 - watch_mass
    total_bs   = raw_buy + raw_sell if (raw_buy + raw_sell) > 0 else 1.0
    w_buy  = remaining * raw_buy  / total_bs
    w_sell = remaining * raw_sell / total_bs

    total  = w_buy + w_sell + watch_mass
    return {
        "BUY":   round(w_buy   / total * 100, 1),
        "WATCH": round(watch_mass / total * 100, 1),
        "SELL":  round(w_sell  / total * 100, 1),
    }


def bias_label(bias: dict) -> str:
    """Return a short English bias label from the bias dict."""
    if bias["BUY"] >= 50:
        return "BULLISH"
    if bias["SELL"] >= 50:
        return "BEARISH"
    return "NEUTRAL/WATCH"


def print_console(result: EngineResult) -> None:
    """
    Print a concise, human-readable summary to stdout.

    Verbose factor tables are suppressed here; they are saved to
    output/factor_report.csv and output/latest_score.json instead.
    """
    W = 72
    SEP = "─" * W

    bias   = compute_directional_bias(result)
    b_lbl  = bias_label(bias)
    is_incomplete = "DATA_INCOMPLETE" in result.action

    print("\n" + "=" * W)
    print("  A股多因子外部风险评分引擎  /  A-Share Risk Engine")
    print("=" * W)
    print(f"  时间 / Time      : {result.timestamp}")
    print(SEP)

    # ── 1. Executable action ──────────────────────────────────────────
    print(f"  最终动作  ACTION : {result.action}")
    print(f"  风险等级  LEVEL  : {result.risk_level}")
    print(SEP)

    # ── 2. Data quality ───────────────────────────────────────────────
    print(f"  数据置信度 CONF  : {result.confidence:5.1f}%  "
          f"({'⚠ 偏低，建议补全关键数据' if result.confidence < 65 else '✓ 可执行'})")
    if result.missing_critical:
        print(f"  关键缺失  MISSING: {', '.join(result.missing_critical)}")
    print(SEP)

    # ── 3. Buy/sell scores ────────────────────────────────────────────
    print(f"  买入分  BUY SCORE: {result.buy_score:5.1f} / 100")
    print(f"  卖出分 SELL SCORE: {result.sell_score:5.1f} / 100")
    print(f"  共振调整 RESONANC: {result.resonance_adjustment:+.1f} pt")
    print(SEP)

    # ── 4. Directional bias (always shown, even under DATA_INCOMPLETE) ─
    # NOTE: This is a directional bias score derived from factor weights;
    #       it is NOT a statistically calibrated probability.
    print("  方向倾向 BIAS SCORES (非统计概率 / not calibrated probabilities):")
    bar_w = 30
    for lbl, key in [("做多 BUY ", "BUY"), ("观望 WATCH", "WATCH"), ("做空 SELL", "SELL")]:
        pct   = bias[key]
        filled = int(round(pct / 100 * bar_w))
        bar    = "█" * filled + "░" * (bar_w - filled)
        print(f"    {lbl}: {bar} {pct:5.1f}%")
    print(f"  → 倾向方向: {b_lbl}")
    print(SEP)

    # ── 5. DATA_INCOMPLETE plain-English explanation ──────────────────
    if is_incomplete:
        if b_lbl == "BEARISH":
            bias_cn = "偏空 (mildly / moderately bearish)"
        elif b_lbl == "BULLISH":
            bias_cn = "偏多 (mildly / moderately bullish)"
        else:
            bias_cn = "中性/观望 (neutral / watch)"
        print(f"  ⓘ DATA_INCOMPLETE 说明:")
        print(f"    当前方向倾向 {bias_cn}，")
        print(f"    但因关键数据缺失（置信度 {result.confidence:.1f}% < 65%），")
        print(f"    动作不可执行，不建议据此直接交易。")
        print(f"    补全 {', '.join(result.missing_critical)} 后可获取可执行信号。")
        print(SEP)

    # ── 6. Decision path ─────────────────────────────────────────────
    print("  决策路径 DECISION PATH:")
    for step in result.decision_path:
        print(f"    → {step}")
    print(SEP)

    # ── 7. Top contributors (bullish / bearish) ───────────────────────
    available = [f for f in result.factors if not f.missing and f.contribution is not None]
    bullish = sorted(available, key=lambda x: x.contribution)[:3]   # most negative = most bullish
    bearish = sorted(available, key=lambda x: -x.contribution)[:3]  # most positive = most bearish

    if bullish:
        print("  ▲ 主要多头驱动 TOP BULLISH CONTRIBUTORS:")
        for f in bullish:
            print(f"    + {f.name:<22s} wt={f.weight:4.1f}  contrib={f.contribution:+.2f}")
    if bearish:
        print("  ▼ 主要空头驱动 TOP BEARISH CONTRIBUTORS:")
        for f in bearish:
            print(f"    - {f.name:<22s} wt={f.weight:4.1f}  contrib={f.contribution:+.2f}")
    print(SEP)

    # ── 8. Important warnings (deduplicated, capped at 5) ────────────
    # Filter to show only meaningful / actionable warnings in terminal.
    important_warnings = [
        w for w in result.warnings
        if any(kw in w for kw in ["关键数据缺失", "FRED_API_KEY", "过期", "yfinance 无数据"])
    ][:5]
    if important_warnings:
        print("  ⚠ 重要提醒 IMPORTANT WARNINGS:")
        for w in important_warnings:
            print(f"    ! {w}")
        if len(result.warnings) > len(important_warnings):
            print(f"    … 详见 output/latest_score.json (warnings 字段，共 {len(result.warnings)} 条)")
        print(SEP)

    print("  详细因子报告 → output/factor_report.csv")
    print("  完整快照     → output/latest_score.json")
    print("  决策树图     → output/decision_tree.png  (文字版见 output/decision_tree_text.txt)")
    print("=" * W + "\n")

def save_outputs(result: EngineResult, features: Dict[str, Optional[float]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    obj = asdict(result)
    obj["factors"] = [asdict(x) for x in result.factors]
    # Embed directional bias into the JSON for downstream consumers
    bias = compute_directional_bias(result)
    obj["directional_bias"] = bias
    obj["directional_bias_label"] = bias_label(bias)
    obj["note_bias"] = (
        "directional_bias is a normalized decision-weight score (NOT a calibrated probability). "
        "BUY+WATCH+SELL = 100. Useful for understanding tendency even when action=DATA_INCOMPLETE."
    )
    (OUTPUT_DIR / "latest_score.json").write_text(
        json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    factor_dataframe(result.factors).to_csv(
        OUTPUT_DIR / "factor_report.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([features]).to_csv(
        OUTPUT_DIR / "feature_snapshot.csv", index=False, encoding="utf-8-sig"
    )
    save_decision_tree()

def calibration_notes() -> str:
    return """
建议回测校准方式：
1. 保存每天 feature_snapshot.csv / latest_score.json；
2. 使用至少2~3年日频样本，最好覆盖2015、2018、2020、2021、2022、2023、2024等不同环境；
3. 目标变量可设为 CSI300 未来1日/5日/20日收益、最大回撤、是否跌破-3%/-5%；
4. 对 US10Y_20D_BP、US10Y_REAL、USDCNH_5D、VIX 等阈值逐一做 walk-forward 校准；
5. 若训练 sklearn 决策树，必须采用 expanding-window / walk-forward，避免未来数据泄漏。
"""

def run(history_days: int = DEFAULT_HISTORY_DAYS, no_live: bool = False) -> EngineResult:
    hub = DataHub(history_days)
    if not no_live:
        hub.fetch_yfinance()
        hub.fetch_fred()
        hub.fetch_ak_bond_yields()
        hub.fetch_a_index_history()
        hub.fetch_a_share_snapshot()
        hub.fetch_margin()
        hub.fetch_boj_policy()
    hub.load_manual_overrides()

    fe = FactorEngine(hub)
    features = fe.build_features()
    factors = fe.evaluate()
    result = score_engine(factors, features, hub)
    print_console(result)
    save_outputs(result, features)
    return result

def build_manual_template() -> None:
    if MANUAL_FILE.exists():
        print(f"{MANUAL_FILE} 已存在，不覆盖。")
        return
    today = datetime.now().date().isoformat()
    template = {
        "ETF_FLOW_5D_BN": {"value": 0, "date": today},
        "IF_BASIS_PCT": {"value": 0, "date": today},
        "IC_BASIS_PCT": {"value": 0, "date": today},
        "IM_BASIS_PCT": {"value": 0, "date": today}
    }
    MANUAL_FILE.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已生成 {MANUAL_FILE}")

def main():
    p = argparse.ArgumentParser(description="A股多因子外部风险评分引擎")
    p.add_argument("--history-days", type=int, default=DEFAULT_HISTORY_DAYS)
    p.add_argument("--no-live", action="store_true", help="不联网，只使用手工/本地数据")
    p.add_argument("--make-manual-template", action="store_true")
    p.add_argument("--show-calibration-notes", action="store_true")
    args = p.parse_args()

    if args.make_manual_template:
        build_manual_template()
        return
    if args.show_calibration_notes:
        print(calibration_notes())
        return
    run(args.history_days, args.no_live)

if __name__ == "__main__":
    main()
