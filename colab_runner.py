"""Complete Google Colab runner for morewnutri/a_share_risk_engine.

Copy this entire file into one fresh Colab cell and run it. Dependencies are
installed into a repository-local target directory so the notebook kernel's
preloaded NumPy/pandas binaries are never replaced in place. This also avoids
depending on ``venv``/``ensurepip``, which is unavailable in some Colab images.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_URL = "https://github.com/morewnutri/a_share_risk_engine.git"
REPO_DIR = Path("/content/a_share_risk_engine")
DEPS_DIR = REPO_DIR / ".deps"
BRANCH = "main"
FRED_API_KEY = ""  # Optional; blank uses the official public FRED CSV.


def check_kernel_numeric_stack() -> None:
    """Fail fast if an earlier in-place pip install already broke this runtime."""
    try:
        import numpy as kernel_numpy
        import pandas as kernel_pandas
    except Exception as exc:
        raise RuntimeError(
            "The current Colab runtime already has a mixed NumPy/pandas ABI. "
            "Choose Runtime > Restart session, then run this corrected cell once."
        ) from exc
    print(
        "Colab kernel remains untouched:",
        f"numpy={kernel_numpy.__version__}",
        f"pandas={kernel_pandas.__version__}",
    )


def run(
    command: list[str],
    cwd: Path | None = None,
    check: bool = True,
    env: dict[str, str] | None = None,
):
    printable = " ".join(map(str, command))
    print(f"\n>>> {printable}")
    result = subprocess.run(
        command,
        cwd=None if cwd is None else str(cwd),
        text=True,
        capture_output=True,
        env=os.environ.copy() if env is None else env,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {printable}")
    return result


# ---------- 1. Clone into a clean directory ----------
check_kernel_numeric_stack()

if REPO_DIR.exists():
    shutil.rmtree(REPO_DIR)

run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(REPO_DIR)])

# Older remote revisions pin NumPy below 2, which is incompatible with the
# current Python 3.13 Colab image. Normalize only that requirement so this
# runner remains usable before the repository-side fix is merged.
requirements_path = REPO_DIR / "requirements.txt"
requirement_lines = requirements_path.read_text(encoding="utf-8").splitlines()
normalized_lines = [
    "numpy>=1.26.4" if line.strip().lower().startswith("numpy") else line
    for line in requirement_lines
]
if normalized_lines != requirement_lines:
    requirements_path.write_text("\n".join(normalized_lines) + "\n", encoding="utf-8")
    print("Adjusted the cloned NumPy requirement for Colab compatibility.")


# ---------- 2. Install into an isolated dependency directory ----------
# ``pip --target`` leaves the kernel packages alone and does not need the
# stdlib ``venv`` module's ensurepip bootstrap.
run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--no-cache-dir",
        "--upgrade",
        "--ignore-installed",
        "--target",
        str(DEPS_DIR),
        "-r",
        "requirements.txt",
    ],
    cwd=REPO_DIR,
)

isolated_env = os.environ.copy()
existing_pythonpath = isolated_env.get("PYTHONPATH")
isolated_env["PYTHONPATH"] = str(DEPS_DIR) + (
    os.pathsep + existing_pythonpath if existing_pythonpath else ""
)
run(
    [
        sys.executable,
        "-c",
        "import numpy, pandas; print('isolated numpy=', numpy.__version__, "
        "'pandas=', pandas.__version__)",
    ],
    cwd=REPO_DIR,
    env=isolated_env,
)

if shutil.which("dot") is None:
    run(["apt-get", "update", "-y"])
    run(["apt-get", "install", "-y", "graphviz"])


# ---------- 3. Run the engine in the isolated environment ----------
engine_env = isolated_env.copy()
engine_env["PYTHONIOENCODING"] = "utf-8"
if FRED_API_KEY.strip():
    engine_env["FRED_API_KEY"] = FRED_API_KEY.strip()
else:
    engine_env.pop("FRED_API_KEY", None)
    print("FRED_API_KEY is blank; using the official FRED public CSV endpoint.")

run([sys.executable, "a_share_risk_engine.py"], cwd=REPO_DIR, env=engine_env)


# ---------- 4. Render outputs with the untouched Colab kernel ----------
import pandas as pd
from IPython.display import Markdown, display


output_dir = REPO_DIR / "output"
state_dir = REPO_DIR / "state"


def safe_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame()
    except Exception as exc:
        print(f"Unable to read {path.name}: {exc}")
        return pd.DataFrame()


def show_md(text: str) -> None:
    display(Markdown(text))


def pct_bar(value: float, width: int = 24) -> str:
    value = max(0.0, min(100.0, float(value)))
    count = int(round(width * value / 100.0))
    return "█" * count + "░" * (width - count)


def fmt(value, digits: int = 2, suffix: str = "") -> str:
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return str(value) if value not in (None, "") else "—"


score_path = output_dir / "latest_score.json"
if not score_path.exists():
    raise RuntimeError("Engine did not create output/latest_score.json; inspect the log above.")

score = json.loads(score_path.read_text(encoding="utf-8"))
macd = safe_csv(output_dir / "monthly_macd_alerts.csv")
factors = safe_csv(output_dir / "factor_report.csv")
health = safe_csv(output_dir / "data_source_health.csv")
history = safe_csv(output_dir / "run_history.csv")
snapshot = safe_csv(state_dir / "a_market_snapshot.csv")

if macd.empty or "index_key" not in macd.columns:
    raise RuntimeError("monthly_macd_alerts.csv is missing or malformed.")
sse_rows = macd.loc[macd["index_key"].astype(str).eq("SSE")]
if sse_rows.empty:
    raise RuntimeError("monthly_macd_alerts.csv does not contain the SSE row.")
sse = sse_rows.iloc[0]

level = str(sse.get("level", "MISSING"))
if level in {"DEATH_CROSS_LIVE", "DEATH_CROSS_CONFIRMED", "BEARISH"}:
    icon = "🔴"
elif level in {
    "BEARISH_RECOVERING",
    "PRE_DEATH_CROSS_CRITICAL",
    "PRE_DEATH_CROSS_WARNING",
}:
    icon = "🟠"
elif level in {"WATCH", "DATA_STALE"}:
    icon = "🟡"
else:
    icon = "🟢"

show_md(
    f"""
# {icon} 上证指数月线 MACD（高权重因子，不单独覆盖综合评分）

- **状态**：`{level}`
- **规则动作**：`{sse.get('action', '—')}`
- **行情日期**：`{sse.get('as_of', '—')}`
- **当前月是否未完成**：`{sse.get('is_live_month', '—')}`
- **收盘**：`{fmt(sse.get('close'))}`
- **DIF**：`{fmt(sse.get('dif'), 4)}`
- **DEA**：`{fmt(sse.get('dea'), 4)}`
- **DIF - DEA**：`{fmt(sse.get('gap'), 4)}`
- **最近交易日正差斜率**：`{fmt(sse.get('gap_daily_slope'), 4)}`
- **预计死叉交易日数**：`{fmt(sse.get('estimated_trading_days_to_cross'), 1)}`
- **死叉临界收盘价**：`{fmt(sse.get('cross_price'))}`
- **距临界价跌幅**：`{fmt(sse.get('distance_to_cross_pct'), 2, '%')}`

> {sse.get('reason', '')}
"""
)

buy = float(score.get("buy_score", 0))
sell = float(score.get("sell_score", 0))
confidence = float(score.get("confidence", 0))
if buy >= 60:
    bias = "偏多 / BUY-BIAS"
elif sell >= 60:
    bias = "偏空 / SELL-BIAS"
else:
    bias = "中性 / HOLD-WATCH"

show_md(
    f"""
# A股多因子外部风险评分引擎

**时间**：`{score.get('timestamp', '—')}`  
**最终动作**：**{score.get('action', '—')}**  
**风险等级**：**{score.get('risk_level', '—')}**  
**方向倾向**：**{bias}**

### 分数概览
- **买入分**：`{buy:.1f}` `{pct_bar(buy)}`
- **卖出分**：`{sell:.1f}` `{pct_bar(sell)}`
- **数据置信度**：`{confidence:.1f}%` `{pct_bar(confidence)}`

### 关键缺失
{', '.join(score.get('missing_critical', [])) or '无'}

### 关键过期
{', '.join(score.get('stale_critical', [])) or '无'}
"""
)

if score.get("decision_path"):
    show_md("### 决策路径\n" + "\n".join(f"- {item}" for item in score["decision_path"]))

show_md("## 重要指数月线 MACD 全表")
macd_columns = [
    column
    for column in [
        "index_name", "as_of", "level", "action", "close", "dif", "dea", "gap",
        "gap_daily_slope", "estimated_trading_days_to_cross", "cross_price",
        "distance_to_cross_pct", "source", "reason",
    ]
    if column in macd.columns
]
display(macd[macd_columns])

if not factors.empty and "missing" in factors.columns:
    missing_mask = factors["missing"].astype(str).str.lower().eq("true")
    missing_factors = factors.loc[missing_mask].copy()
    optional_names = {"主要宽基ETF 5日净流入", "IF基差", "IC基差", "IM基差"}
    optional = missing_factors.loc[missing_factors["factor"].isin(optional_names)]
    automatic = missing_factors.loc[~missing_factors["factor"].isin(optional_names)]

    show_md(f"## 自动数据仍缺失（{len(automatic)} 项）")
    if automatic.empty:
        show_md("自动数据源对应因子均已取得。")
    else:
        display(automatic[["group", "factor", "weight", "source", "detail"]])

    show_md(f"## 可选手工数据缺失（{len(optional)} 项）")
    if not optional.empty:
        display(optional[["group", "factor", "weight", "source", "detail"]])

if not health.empty:
    stale_mask = health.get("stale", pd.Series(False, index=health.index)).astype(str).str.lower().eq("true")
    observations = pd.to_numeric(
        health.get("observations", pd.Series(0, index=health.index)), errors="coerce"
    ).fillna(0)
    health_problems = health.loc[stale_mask | observations.eq(0)]
    show_md("## 数据源健康检查")
    if health_problems.empty:
        show_md("没有发现已加载序列过期或空序列。")
    else:
        columns = [
            column
            for column in [
                "key", "source_chain", "observations", "first_date", "last_date",
                "age_days", "stale", "note",
            ]
            if column in health_problems.columns
        ]
        display(health_problems[columns])

if not factors.empty:
    factors["weighted_contribution"] = pd.to_numeric(
        factors.get("weighted_contribution"), errors="coerce"
    )
    ranked = factors.dropna(subset=["weighted_contribution"])
    columns = [
        column
        for column in [
            "group", "factor", "weight", "signal_-1bull_+1bear",
            "weighted_contribution", "value", "source",
        ]
        if column in factors.columns
    ]
    show_md("## 最偏空因子 Top 8")
    display(ranked.sort_values("weighted_contribution", ascending=False).head(8)[columns])
    show_md("## 最偏多因子 Top 8")
    display(ranked.sort_values("weighted_contribution").head(8)[columns])

if not history.empty:
    if "timestamp" in history.columns:
        history["timestamp"] = pd.to_datetime(history["timestamp"], errors="coerce", utc=True)
        history = history.sort_values("timestamp").tail(14)
    show_md("## 最近 14 次运行历史")
    display(history)

if not snapshot.empty:
    show_md("## 最近 14 条 A股快照")
    display(snapshot.tail(14))

warnings = score.get("warnings", [])
if warnings:
    show_md("## 数据提醒\n" + "\n".join(f"- {warning}" for warning in warnings))

show_md(
    "## 输出目录\n"
    f"- `{output_dir}`\n"
    f"- `{state_dir}`"
)
