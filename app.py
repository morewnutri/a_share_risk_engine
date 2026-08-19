#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit dashboard for A股多因子外部风险评分引擎.

Usage (local):
    streamlit run app.py

Usage (Colab without ngrok – preferred):
    Run a_share_risk_engine.py first, then open output/dashboard_report.html directly.
    See README.md for details.
"""

import json
import os
from pathlib import Path

import pandas as pd

try:
    import streamlit as st
except ImportError:
    raise SystemExit("Please install streamlit: pip install streamlit")

OUTPUT_DIR = Path("output")
STATE_DIR = Path("state")

st.set_page_config(
    page_title="A股风险评分仪表盘",
    page_icon="📊",
    layout="wide",
)

st.title("📊 A股多因子外部风险评分引擎")

# ── Load latest score ────────────────────────────────────────────────────────

score_path = OUTPUT_DIR / "latest_score.json"
if not score_path.exists():
    st.error(
        "找不到 output/latest_score.json。请先运行：\n\n"
        "```\npython a_share_risk_engine.py\n```"
    )
    st.stop()

with open(score_path, encoding="utf-8") as f:
    data = json.load(f)

timestamp = data.get("timestamp", "未知")
buy_score = float(data.get("buy_score", 0))
sell_score = float(data.get("sell_score", 0))
confidence = float(data.get("confidence", 0))
action = data.get("action", "未知")
risk_level = data.get("risk_level", "未知")
resonance = float(data.get("resonance_adjustment", 0))
missing_critical = data.get("missing_critical", [])
warnings = data.get("warnings", [])
decision_path = data.get("decision_path", [])

# ── Determine directional bias label ─────────────────────────────────────────
# buy_score and sell_score are complementary (sell = risk, buy = 100 - risk).
# The engine's rule_decision_tree already returns a clear action string.
if "BUY" in action.upper():
    bias_label = "📈 偏买入 / BUY BIAS"
    bias_color = "🟢"
elif any(k in action.upper() for k in ("SELL", "REDUCE", "RISK_OFF")):
    bias_label = "📉 偏卖出 / SELL BIAS"
    bias_color = "🔴"
elif "WATCH" in action.upper():
    bias_label = "👀 观察 / WATCH"
    bias_color = "🟡"
elif "INCOMPLETE" in action.upper():
    bias_label = "⚠️ 数据不完整 / DATA INCOMPLETE"
    bias_color = "⚠️"
else:
    bias_label = "⏸ 中性等待 / HOLD"
    bias_color = "⚪"

# ── Top summary row ───────────────────────────────────────────────────────────
st.caption(f"最后更新: {timestamp}")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("买入分 (偏多) 🟢", f"{buy_score:.1f}", help="0→100，越高越偏多头。与卖出分互补，两者之和=100。")
col2.metric("卖出分 (偏空) 🔴", f"{sell_score:.1f}", help="0→100，越高越偏空头。")
col3.metric("风险等级", risk_level)
col4.metric("数据置信度", f"{confidence:.1f}%")
col5.metric("共振调整", f"{resonance:+.1f}")

st.markdown("---")

# ── Directional bias ─────────────────────────────────────────────────────────
st.subheader("综合方向偏好")
st.markdown(f"### {bias_color} {bias_label}")
st.markdown(f"**最终动作建议:** `{action}`")

# Progress bars
st.markdown(f"**买入分** `{buy_score:.1f}` / 100")
st.progress(int(buy_score))
st.markdown(f"**卖出分** `{sell_score:.1f}` / 100")
st.progress(int(sell_score))

# ── Decision path ─────────────────────────────────────────────────────────────
if decision_path:
    with st.expander("📋 决策路径"):
        for step in decision_path:
            st.markdown(f"- {step}")

# ── Run history chart ─────────────────────────────────────────────────────────
history_path = OUTPUT_DIR / "run_history.csv"
if history_path.exists():
    try:
        hist = pd.read_csv(history_path)
        hist["timestamp"] = pd.to_datetime(hist["timestamp"], errors="coerce")
        hist = hist.dropna(subset=["timestamp"]).sort_values("timestamp").tail(14)
        if not hist.empty:
            st.subheader("📈 近14次评分历史")
            chart_df = hist.set_index("timestamp")[["buy_score", "sell_score", "confidence"]].copy()
            chart_df.columns = ["买入分", "卖出分", "置信度"]
            st.line_chart(chart_df)
            with st.expander("查看评分历史表格"):
                disp = hist[["timestamp", "buy_score", "sell_score", "confidence", "action", "risk_level"]].copy()
                disp.columns = ["时间", "买入分", "卖出分", "置信度", "动作", "风险等级"]
                st.dataframe(disp.set_index("时间"), use_container_width=True)
    except Exception as e:
        st.warning(f"无法读取历史记录: {e}")

# ── Factor details ─────────────────────────────────────────────────────────────
factor_path = OUTPUT_DIR / "factor_report.csv"
if factor_path.exists():
    try:
        fdf = pd.read_csv(factor_path)
        st.subheader("🔍 因子明细")
        # Rename columns for display
        fdf_disp = fdf[["group", "factor", "weight", "signal_-1bull_+1bear",
                         "weighted_contribution", "value", "missing", "source"]].copy()
        fdf_disp.columns = ["组别", "因子", "权重", "信号(-1多/+1空)", "加权贡献", "数值", "缺失", "来源"]
        st.dataframe(fdf_disp, use_container_width=True, height=500)
    except Exception as e:
        st.warning(f"无法读取因子报告: {e}")

# ── Missing critical ──────────────────────────────────────────────────────────
if missing_critical:
    st.warning("⚠️ **关键数据缺失:** " + ", ".join(missing_critical))

# ── Warnings ──────────────────────────────────────────────────────────────────
if warnings:
    with st.expander(f"⚠️ 数据提醒 ({len(warnings)} 条)"):
        for w in warnings:
            st.markdown(f"- {w}")

# ── Snapshot ──────────────────────────────────────────────────────────────────
snap_path = STATE_DIR / "a_market_snapshot.csv"
if snap_path.exists():
    try:
        snap = pd.read_csv(snap_path).tail(14)
        with st.expander("📊 A股市场横截面快照（最近14天）"):
            st.dataframe(snap, use_container_width=True)
    except Exception:
        pass

st.markdown("---")
st.caption(
    "⚠️ 本工具仅供研究参考，不构成投资建议。阈值为初始启发式参数，请基于历史数据自行校准。"
)
