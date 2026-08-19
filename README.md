# A股多因子外部风险评分引擎

## 安装

```bash
python -m pip install -r requirements.txt
```

Graphviz PNG 可视化还需要系统安装 Graphviz；没有也没关系，程序仍输出 `output/decision_tree.dot`。

## FRED API KEY

程序通过 FRED 官方 API 获取：
- DGS10：美国10Y国债收益率
- DFII10：美国10Y实际利率
- DGS2：美国2Y国债收益率
- BAMLH0A0HYM2：美国高收益债OAS
- DFF：有效联邦基金利率

macOS/Linux:
```bash
export FRED_API_KEY="你的key"
```

Windows PowerShell:
```powershell
$env:FRED_API_KEY="你的key"
```

未设置时程序会报警并降低置信度，不会补假数据。

## 运行

```bash
python a_share_risk_engine.py
```

输出：
- `output/latest_score.json`
- `output/factor_report.csv`
- `output/feature_snapshot.csv`
- `output/run_history.csv`（历史评分，最多保留30次）
- `output/dashboard_report.html`（**自包含 HTML 仪表盘，可直接下载在本地浏览器查看**）
- `output/decision_tree.dot`
- `output/decision_tree.png`（Graphviz可用时）
- `state/a_market_snapshot.csv`

## Streamlit 仪表盘（本地运行）

```bash
pip install streamlit
python a_share_risk_engine.py   # 先生成数据
streamlit run app.py
```

浏览器访问 `http://localhost:8501`，仪表盘会显示：
- 买入分 / 卖出分 / 风险等级 / 置信度
- 方向偏好（买入/观察/卖出/中性）及进度条
- 近14次运行历史折线图
- 因子明细表
- 数据警告

## Google Colab 使用指南（**无需 ngrok，无需账号**）

### 方法 A：直接下载 HTML 仪表盘（推荐，零依赖）

1. 打开新的 Colab Notebook
2. 在单元格中粘贴 `colab_run.py` 的内容（或上传后 `exec(open('colab_run.py').read())`）
3. 运行完毕后，左侧文件面板找到 `/content/a_share_risk_engine/output/dashboard_report.html`
4. 右键 → **Download**，用本地浏览器打开即可查看完整仪表盘

或直接使用以下最简单的 Colab 单元格：

```python
import os, subprocess, sys
from pathlib import Path

REPO_URL = "https://github.com/morewnutri/a_share_risk_engine.git"
REPO_DIR = "/content/a_share_risk_engine"
BRANCH   = "copilot/fix-data-and-dashboard-issues"
FRED_API_KEY = ""  # 可选

subprocess.run("apt-get install -y -q graphviz", shell=True)
subprocess.run(f"git clone --depth=1 {REPO_URL} {REPO_DIR} && cd {REPO_DIR} && git checkout {BRANCH}", shell=True)
subprocess.run(f"{sys.executable} -m pip install -r requirements.txt -q", shell=True, cwd=REPO_DIR)

if FRED_API_KEY:
    os.environ["FRED_API_KEY"] = FRED_API_KEY

subprocess.run(f"{sys.executable} a_share_risk_engine.py", shell=True, cwd=REPO_DIR)

# Download HTML report
from google.colab import files
files.download(f"{REPO_DIR}/output/dashboard_report.html")
```

### 方法 B：Streamlit + ngrok（可选，需要 ngrok 账号）

```python
import subprocess, sys, time, os
from pathlib import Path

REPO_DIR = "/content/a_share_risk_engine"
subprocess.run(f"{sys.executable} -m pip install pyngrok streamlit -q", shell=True)

from pyngrok import ngrok
# 确保已用 ngrok.set_auth_token("你的token") 设置 token

# 生成数据
subprocess.run(f"{sys.executable} a_share_risk_engine.py", shell=True, cwd=REPO_DIR)

# 启动 Streamlit
proc = subprocess.Popen(
    f"{sys.executable} -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0",
    shell=True, cwd=REPO_DIR,
)
time.sleep(6)
public_url = ngrok.connect(8501)
print("Dashboard URL:", public_url)
```

> **注意**：`http://127.0.0.1:8501` 在 Colab 中无效；必须使用 ngrok 或其他隧道。
> 建议直接使用方法 A（HTML 下载）避免此问题。

## 数据来源健壮性

| 数据 | 主来源 | 回退来源 |
|------|--------|---------|
| HSTECH | `^HSTECH` (yfinance) | `3033.HK`, `3067.HK`, `2838.HK` (港股ETF代理) |
| A50 | `XIN9.SI` (yfinance) | `2823.HK`, `CNYA`, `FXI` |
| A股指数历史 | `ak.stock_zh_index_daily_em` | `ak.stock_zh_index_daily` |
| A股横截面宽度 | `ak.stock_zh_a_spot_em` | `state/a_market_snapshot.csv` 本地历史快照（带标注） |
| US10Y_REAL | FRED:DFII10 | N/A（官方数据，不补假数据；置信度降低） |

回退数据来源会在输出中明确标注（如 `[代理/fallback]`、`[STALE: Xd old]`）。

## 评分逻辑

每个因子映射为：
- -1：偏多
- 0：中性
- +1：偏空

核心不仅看绝对值，也看：
- 1日/5日/20日收益率
- 美债5日/20日bp变化
- 60日z-score
- 20日线性斜率
- A股全市场成交额/MA20
- 沪深300/中证1000/创业板/科创50各自成交额相对MA20
- 跌停股占全部A股比例、强弱扩散差
- A股上涨家数比例
- 多市场共振

## 第一次运行

`A股成交额 / MA20` 依赖程序自己逐日保存 `state/a_market_snapshot.csv`。  
历史不足时该项会明确标记缺失，不会伪造。

## 可选手工数据

```bash
python a_share_risk_engine.py --make-manual-template
```

支持补充：
- ETF_FLOW_5D_BN
- IF_BASIS_PCT
- IC_BASIS_PCT
- IM_BASIS_PCT

## 重要

当前阈值是研究用初始值。实盘前应对 2015~现在做 walk-forward 回测并重新校准权重/阈值。

