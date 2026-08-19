#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股风险引擎 · 仪表盘
运行方式：  streamlit run app.py
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── paths ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("output")
STATE_DIR = Path("state")
LATEST_SCORE = OUTPUT_DIR / "latest_score.json"
FACTOR_REPORT = OUTPUT_DIR / "factor_report.csv"
RUN_HISTORY = OUTPUT_DIR / "run_history.csv"
MARKET_SNAPSHOT = STATE_DIR / "a_market_snapshot.csv"

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="A股风险引擎仪表盘",
    page_icon="📊",
    layout="wide",
)

# ── helpers ───────────────────────────────────────────────────────────────────

def load_latest_score() -> dict | None:
    if LATEST_SCORE.exists():
        try:
            return json.loads(LATEST_SCORE.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def load_run_history(days: int = 14) -> pd.DataFrame:
    if not RUN_HISTORY.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(RUN_HISTORY, encoding="utf-8-sig")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days)
        df = df[df["timestamp"] >= cutoff]
        return df
    except Exception:
        return pd.DataFrame()


def load_factor_report() -> pd.DataFrame:
    if not FACTOR_REPORT.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(FACTOR_REPORT, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()


def load_market_snapshot(days: int = 14) -> pd.DataFrame:
    if not MARKET_SNAPSHOT.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(MARKET_SNAPSHOT, encoding="utf-8-sig")
        # find the date column
        date_col = next((c for c in df.columns if "date" in c.lower()), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            cutoff = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=days)
            df = df[df[date_col] >= cutoff].sort_values(date_col)
        return df
    except Exception:
        return pd.DataFrame()


def action_color(action: str) -> str:
    a = action.upper()
    if "BUY" in a:
        return "🟢"
    if "REDUCE" in a or "RISK_OFF" in a:
        return "🔴"
    if "WATCH" in a:
        return "🟡"
    return "⚪"


def signal_badge(v: float | None) -> str:
    if v is None:
        return "—"
    if v <= -0.3:
        return f"🟢 {v:+.2f}"
    if v >= 0.3:
        return f"🔴 {v:+.2f}"
    return f"⚪ {v:+.2f}"


# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 设置")
    history_days = st.slider("历史天数", 7, 90, 14)
    if st.button("🔄 刷新数据"):
        st.rerun()
    st.markdown("---")
    st.markdown("**运行 CLI 更新数据:**")
    st.code("python a_share_risk_engine.py")
    st.markdown("**启动仪表盘:**")
    st.code("streamlit run app.py")

# ── main ───────────────────────────────────────────────────────────────────────
st.title("📊 A股多因子外部风险引擎 · 仪表盘")

score = load_latest_score()

if score is None:
    st.warning("⚠️ 尚无评分数据。请先运行：`python a_share_risk_engine.py`")
    st.stop()

# ── top summary ───────────────────────────────────────────────────────────────
ts = score.get("timestamp", "—")
st.caption(f"最新运行时间：{ts}")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    action = score.get("action", "—")
    st.metric("最终建议", f"{action_color(action)} {action}")
with col2:
    st.metric("买入分 (0-100)", f"{score.get('buy_score', '—')}")
with col3:
    st.metric("卖出分 (0-100)", f"{score.get('sell_score', '—')}")
with col4:
    st.metric("数据置信度", f"{score.get('confidence', '—')}%")
with col5:
    st.metric("风险等级", score.get("risk_level", "—"))

# ── buy/watch/sell visual bar ─────────────────────────────────────────────────
buy = float(score.get("buy_score", 50))
sell = float(score.get("sell_score", 50))
conf = float(score.get("confidence", 0))

st.markdown("### 📈 买入 / 中性 / 卖出 可视化")
bar_df = pd.DataFrame(
    {"分数": [buy, 100 - buy - sell, sell]},
    index=["买入分", "中性区间", "卖出分"],
)
# simple horizontal bar chart using st.progress-like approach
col_b, col_s = st.columns(2)
with col_b:
    st.markdown(f"**买入分** `{buy:.1f}`")
    st.progress(int(buy))
with col_s:
    st.markdown(f"**卖出分** `{sell:.1f}`")
    st.progress(int(sell))

resonance = score.get("resonance_adjustment", 0)
bias_label = "偏多 🟢" if resonance < -3 else ("偏空 🔴" if resonance > 3 else "中性 ⚪")
st.info(f"**共振调整** `{resonance:+.1f}`  ·  **方向偏向** {bias_label}")

# ── decision path ─────────────────────────────────────────────────────────────
with st.expander("📋 决策路径", expanded=False):
    for step in score.get("decision_path", []):
        st.markdown(f"- {step}")

# ── factor table ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔍 因子明细")

df_factors = load_factor_report()
if not df_factors.empty:
    # rename for display
    display_cols = {
        "group": "分组",
        "factor": "因子",
        "weight": "权重",
        "signal_-1bull_+1bear": "信号 (-1多/+1空)",
        "weighted_contribution": "加权贡献",
        "value": "当前值",
        "missing": "缺失",
        "source": "数据源",
        "detail": "详情",
    }
    df_show = df_factors.rename(columns=display_cols)
    # highlight missing
    def highlight_missing(row):
        if row.get("缺失", False):
            return ["background-color: #fff3cd"] * len(row)
        sig = row.get("信号 (-1多/+1空)")
        if sig is not None and not pd.isna(sig):
            if float(sig) >= 0.3:
                return ["background-color: #ffd7d7"] * len(row)
            if float(sig) <= -0.3:
                return ["background-color: #d4edda"] * len(row)
        return [""] * len(row)

    available = [c for c in display_cols.values() if c in df_show.columns]
    st.dataframe(
        df_show[available].style.apply(highlight_missing, axis=1),
        use_container_width=True,
        height=500,
    )
else:
    st.info("暂无因子报告，请运行 CLI 获取最新数据。")

# ── run history charts ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"### 📅 最近 {history_days} 天评分趋势（基于历史运行快照）")

df_hist = load_run_history(history_days)
if df_hist.empty:
    st.info(
        "尚无历史运行记录。每次运行 `python a_share_risk_engine.py` 后，数据将自动追加到 "
        "`output/run_history.csv`，之后仪表盘会显示趋势图。"
    )
else:
    st.caption("⚠️ 图表基于程序运行历史快照，而非连续市场数据。每次运行会追加一条记录。")
    df_hist = df_hist.set_index("timestamp")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**买入分 / 卖出分**")
        score_cols = [c for c in ["buy_score", "sell_score"] if c in df_hist.columns]
        if score_cols:
            st.line_chart(df_hist[score_cols].rename(columns={"buy_score": "买入分", "sell_score": "卖出分"}))
    with col_r:
        st.markdown("**数据置信度**")
        if "confidence" in df_hist.columns:
            st.line_chart(df_hist[["confidence"]].rename(columns={"confidence": "置信度"}))

    # key factor time series
    st.markdown("**关键因子时序（最近运行历史）**")
    factor_pairs = [
        (["VIX"], "VIX 恐慌指数"),
        (["US10Y", "US10Y_REAL"], "美债收益率 (%)"),
        (["A_BREADTH"], "A股上涨家数比例"),
        (["CSI300_5D", "HSTECH_5D"], "沪深300 / 恒生科技 5日涨跌 (%)"),
        (["A_TURNOVER_MA20_RATIO"], "A股成交额/MA20"),
    ]
    ncols = 2
    pairs_iter = iter(factor_pairs)
    for _ in range((len(factor_pairs) + 1) // ncols):
        cols = st.columns(ncols)
        for col in cols:
            try:
                keys, label = next(pairs_iter)
            except StopIteration:
                break
            available_keys = [k for k in keys if k in df_hist.columns]
            if available_keys:
                with col:
                    st.markdown(f"**{label}**")
                    st.line_chart(df_hist[available_keys])

# ── market snapshot charts ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"### 🏪 A股每日快照（最近 {history_days} 天，来自 state/a_market_snapshot.csv）")

df_mkt = load_market_snapshot(history_days)
if df_mkt.empty:
    st.info("暂无市场快照数据或数据不足，请先运行 CLI 积累历史。")
else:
    st.caption("⚠️ 图表基于实际市场数据采集快照。")
    date_col = next((c for c in df_mkt.columns if "date" in c.lower()), None)
    if date_col:
        df_mkt = df_mkt.set_index(date_col)
    numeric_cols = df_mkt.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        selected = st.multiselect(
            "选择显示列",
            options=numeric_cols,
            default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols,
        )
        if selected:
            st.line_chart(df_mkt[selected])

# ── warnings ───────────────────────────────────────────────────────────────────
warnings = score.get("warnings", [])
missing_critical = score.get("missing_critical", [])

if warnings or missing_critical:
    st.markdown("---")
    st.markdown("### ⚠️ 数据预警")
    if missing_critical:
        st.error(f"**关键数据缺失：** {', '.join(missing_critical)}")
    for w in warnings:
        st.warning(w)

st.markdown("---")
st.caption(
    "本仪表盘为研究/风控工具，不构成投资建议。"
    "阈值为初始启发式参数，实盘前须使用历史数据回测校准。"
)
