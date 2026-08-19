#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit dashboard for A-Share Risk Engine.

Run:
    streamlit run app.py

Or in Colab (with pyngrok):
    See README.md for Colab instructions.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

OUTPUT_DIR = Path("output")
STATE_DIR = Path("state")

st.set_page_config(page_title="A股风险评分仪表板", layout="wide")
st.title("A股多因子外部风险评分仪表板")

# ── Helper ──────────────────────────────────────────────────────────────────

def load_latest_score():
    path = OUTPUT_DIR / "latest_score.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_run_history(days: int = 14) -> pd.DataFrame:
    path = OUTPUT_DIR / "run_history.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        df = df[df["timestamp"] >= cutoff]
        return df
    except Exception:
        return pd.DataFrame()


def load_factor_report() -> pd.DataFrame:
    path = OUTPUT_DIR / "factor_report.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_snapshot_history() -> pd.DataFrame:
    path = STATE_DIR / "a_market_snapshot.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df.dropna(subset=["date"]).sort_values("date")
    except Exception:
        return pd.DataFrame()


# ── Reload button ────────────────────────────────────────────────────────────

if st.button("🔄 刷新数据（重新运行引擎）"):
    import subprocess, sys
    with st.spinner("运行引擎中..."):
        result = subprocess.run(
            [sys.executable, "a_share_risk_engine.py"],
            capture_output=True, text=True,
        )
    if result.returncode == 0:
        st.success("引擎运行完成！")
    else:
        st.error("引擎运行出错")
        st.code(result.stderr[-3000:] if result.stderr else "(无错误输出)")

# ── Latest scores ─────────────────────────────────────────────────────────────

score = load_latest_score()

if score is None:
    st.warning("暂无输出数据。请先运行 `python a_share_risk_engine.py`，或点击上方刷新按钮。")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
buy = score.get("buy_score", 0)
sell = score.get("sell_score", 0)
conf = score.get("confidence", 0)
action = score.get("action", "—")
risk_level = score.get("risk_level", "—")
ts = score.get("timestamp", "—")

# Derive directional bias directly from buy/sell scores (explicit, not arithmetic middle)
if buy >= sell + 10:
    bias_label = "偏多 (BUY BIAS)"
    bias_color = "green"
elif sell >= buy + 10:
    bias_label = "偏空 (SELL BIAS)"
    bias_color = "red"
else:
    bias_label = "中性 (NEUTRAL)"
    bias_color = "gray"

col1.metric("买入分 (Buy)", f"{buy:.1f}")
col2.metric("卖出分 (Sell)", f"{sell:.1f}")
col3.metric("置信度", f"{conf:.1f}%")
col4.metric("数据时间", ts[:19] if ts != "—" else "—")

st.markdown(f"**建议动作:** `{action}`")
st.markdown(f"**风险等级:** `{risk_level}`")
st.markdown(f"**方向偏向:** :{bias_color}[{bias_label}]")

# ── Action summary bar ────────────────────────────────────────────────────────

st.subheader("买入 / 观察 / 卖出 信号摘要")
action_upper = action.upper()
if "BUY_BIAS" in action_upper or "偏买" in action:
    signal_class = "🟢 BUY BIAS（分批偏买入）"
elif "WATCH" in action_upper or "观察" in action:
    signal_class = "🟡 WATCH BUY（观察偏多）"
elif "REDUCE" in action_upper or "卖出" in action or "RISK_OFF" in action_upper:
    signal_class = "🔴 SELL / REDUCE（偏卖出 / 降仓）"
elif "DATA_INCOMPLETE" in action_upper:
    signal_class = "⚪ DATA INCOMPLETE（数据不足，暂不操作）"
else:
    signal_class = "⚪ HOLD（中性等待）"

st.info(signal_class)

resonance = score.get("resonance_adjustment", 0)
st.caption(f"共振调整: {resonance:+.1f}  |  "
           f"关键缺失: {', '.join(score.get('missing_critical', [])) or '无'}")

# ── Decision path ─────────────────────────────────────────────────────────────

with st.expander("决策路径"):
    for step in score.get("decision_path", []):
        st.write("•", step)

# ── 14-day score history chart ────────────────────────────────────────────────

st.subheader("近14天评分走势")
hist = load_run_history(14)
if hist.empty:
    st.info("暂无历史记录（需多次运行引擎后才有）。")
else:
    chart_df = hist.set_index("timestamp")[["buy_score", "sell_score", "confidence"]].copy()
    chart_df.index = chart_df.index.tz_convert("Asia/Shanghai").strftime("%m-%d %H:%M")
    st.line_chart(chart_df)

    # Action history table
    tbl = hist[["timestamp", "buy_score", "sell_score", "confidence", "action", "risk_level"]].copy()
    tbl["timestamp"] = tbl["timestamp"].dt.tz_convert("Asia/Shanghai").dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(tbl.sort_values("timestamp", ascending=False).reset_index(drop=True),
                 use_container_width=True)

# ── Market breadth history ────────────────────────────────────────────────────

st.subheader("A股市场宽度历史（近14天）")
snap = load_snapshot_history()
if snap.empty:
    st.info("暂无快照历史。")
else:
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=14)
    snap_recent = snap[snap["date"] >= cutoff].set_index("date")
    cols_to_show = [c for c in ["breadth", "decliners", "strong3", "weak3"] if c in snap_recent.columns]
    if cols_to_show:
        st.line_chart(snap_recent[cols_to_show])
    if "turnover" in snap_recent.columns:
        st.caption("成交额历史（亿元）")
        st.bar_chart(snap_recent[["turnover"]])

# ── Factor detail ─────────────────────────────────────────────────────────────

st.subheader("因子明细")
factors_df = load_factor_report()
if factors_df.empty:
    st.info("无因子报告。")
else:
    st.dataframe(factors_df, use_container_width=True)

# ── Warnings ──────────────────────────────────────────────────────────────────

warnings = score.get("warnings", [])
if warnings:
    with st.expander(f"⚠ 数据提醒 ({len(warnings)} 条)"):
        for w in warnings:
            st.write("•", w)
