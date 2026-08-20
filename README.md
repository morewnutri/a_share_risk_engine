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
- `output/run_history.csv`（每日追加，用于仪表板历史图表）
- `output/decision_tree.dot`
- `output/decision_tree.png`（Graphviz可用时）
- `state/a_market_snapshot.csv`
- `state/fred_cache.csv`（FRED 本地缓存，每次成功获取后自动更新）
- `state/index_history_cache.csv`（A股指数历史本地缓存，每次成功获取后自动更新）
- `state/derived_cache.json`（ETF流量/基差本地缓存，每次成功填充后自动更新）

## 仪表板

```bash
streamlit run app.py
```

仪表板功能：
- 买入分 / 卖出分 / 置信度指标卡
- Buy / Watch / Sell 信号摘要（明确区分偏多、观察偏多、偏卖出、中性等待）
- 近14天评分走势折线图 + 历史表格
- A股市场宽度历史图（上涨家数/下跌家数/强/弱比例）
- 因子明细表
- 一键重新运行引擎按钮

### Colab 运行仪表板

```python
import os, subprocess, sys, time
from pathlib import Path

REPO_URL = "https://github.com/morewnutri/a_share_risk_engine.git"
REPO_DIR = "/content/a_share_risk_engine"
BRANCH = "copilot/robust-data-fallbacks"
FRED_API_KEY = ""  # 可选

def run(cmd, cwd=None, check=True):
    r = subprocess.run(cmd, shell=True, cwd=cwd, text=True, capture_output=True)
    print(r.stdout or ""); print(r.stderr or "", file=sys.stderr)
    if check and r.returncode != 0: raise RuntimeError(cmd)

run("apt-get update -y && apt-get install -y graphviz")
if Path(REPO_DIR).exists(): run(f"rm -rf {REPO_DIR}")
run(f"git clone {REPO_URL} {REPO_DIR}")
run(f"git checkout {BRANCH}", cwd=REPO_DIR)
run(f"{sys.executable} -m pip install -r requirements.txt", cwd=REPO_DIR)
run(f"{sys.executable} -m pip install pyngrok", cwd=REPO_DIR)

if FRED_API_KEY: os.environ["FRED_API_KEY"] = FRED_API_KEY
run(f"{sys.executable} a_share_risk_engine.py", cwd=REPO_DIR)  # generate data first

from pyngrok import ngrok
proc = subprocess.Popen(
    f"{sys.executable} -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0",
    shell=True, cwd=REPO_DIR, env=os.environ.copy(),
)
time.sleep(8)
print("Dashboard URL:", ngrok.connect(8501))
```

如果 Streamlit 在 Colab 中无法访问，可以直接查看输出文件：
- `output/latest_score.json` — 最新评分
- `output/factor_report.csv` — 因子明细
- `output/run_history.csv` — 历史记录

## 数据回退机制（Fallback）

### HSTECH（恒生科技）
依次尝试：
1. yfinance: `3067.HK`（ETF proxy）
2. yfinance: `3033.HK`（ETF proxy）
3. yfinance: `^HSTECH`（直接指数，但历史上经常失败）
4. AKShare: `stock_hk_index_daily_em`（如可用）
5. 所有来源失败时留空并发出明确警告

`source` 字段明确标注来源类型（direct index / ETF proxy / fallback）。

### A50（富时中国A50）
依次尝试：
1. yfinance: `2823.HK`（ETF proxy）
2. yfinance: `XIN9.SI`（ETF proxy）
3. AKShare: `sh000016`（SSE50 作为代理，明确标注 `proxy=SSE50, fallback`）
4. 所有来源失败时留空并发出明确警告

### A股指数历史
依次尝试：
1. `ak.stock_zh_index_daily_em(symbol=...)` （含成交额/量）
2. `ak.index_zh_a_hist(symbol=..., period="daily")` （更新版 AKShare 路径）
3. **yfinance** 价格代理（`source` 标注 `yfinance:... (A股指数价格代理，无成交额)`）
4. **本地指数历史缓存** `state/index_history_cache.csv`（上次成功获取的历史，`source` 标注 `local-index-cache (缓存日期=..., 已滞后Xd)`）

每次成功获取后自动更新本地缓存，确保 AKShare 临时失联时仍有历史数据用于 MA20 计算。

### A股横截面快照（市场宽度）
依次尝试：
1. `ak.stock_zh_a_spot_em()`（实时）
2. `ak.stock_zh_a_spot()`（备用实时接口）
3. 本地 `state/a_market_snapshot.csv` 的最后一行（过期降级，**明确警告包含日期和滞后天数**）

**重要：** 使用过期快照时，`source` 字段会注明 `stale snapshot (date=..., age=Xd)`，不会静默当作实时数据。

### 宏观数据（FRED）
降级梯队：
1. **FRED 官方 API**（`FRED_API_KEY` 环境变量已设置时使用，返回完整历史）
2. **FRED 公共 CSV 端点**（免密钥，约1年历史，`source` 标注 `FRED-public-csv`）
3. **本地 FRED 缓存** `state/fred_cache.csv`（上次成功获取的数据，`source` 标注 `FRED-local-cache (date=..., 已滞后Xd)`）
4. 三层均失败时数据缺失，置信度降低

每次成功获取后会自动更新本地缓存，确保下次离线运行也有参考值。

**配置方式（仍推荐设置 API Key 以获取最完整历史）：**

### ETF 净流入 / 股指期货基差（ETF_FLOW_5D_BN / IF/IC/IM_BASIS_PCT）

这些字段没有可靠的免费公开实时数据源。降级梯队：
1. **手工覆盖** `manual_overrides.json`（`source` 标注 `manual`）
2. **本地 derived 缓存** `state/derived_cache.json`（上次手工或成功填充的值，`source` 标注 `derived-cache (日期=..., 已滞后Xd)`）
3. 两层均失败时该字段缺失（低权重，不影响核心信号）

**使用方式：** 运行 `python a_share_risk_engine.py --make-manual-template`，填写 `manual_overrides.json` 后再运行引擎；值会自动写入 derived 缓存供后续运行使用。



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

## 方向偏向（Buy/Watch/Sell）

仪表板中方向偏向由买入分和卖出分直接推导，**不使用 `100-buy-sell` 计算中间桶**：
- `buy_score >= sell_score + 10` → 偏多 (BUY BIAS)
- `sell_score >= buy_score + 10` → 偏空 (SELL BIAS)
- 否则 → 中性 (NEUTRAL)

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
