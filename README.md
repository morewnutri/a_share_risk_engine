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

## 运行（CLI）

```bash
python a_share_risk_engine.py
```

输出：
- `output/latest_score.json`
- `output/factor_report.csv`
- `output/feature_snapshot.csv`
- `output/run_history.csv`（每次运行自动追加一行，供仪表盘绘图）
- `output/decision_tree.dot`
- `output/decision_tree.png`（Graphviz可用时）
- `state/a_market_snapshot.csv`

## 仪表盘

每次 CLI 运行后，打开仪表盘查看最新评分与趋势：

```bash
streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`，展示：

- **顶部摘要**：最终建议、买入分、卖出分、置信度、风险等级
- **买入/卖出可视化条**：直观显示多空倾向
- **因子明细表**：所有因子的信号、权重、当前值，绿色=偏多，红色=偏空，黄色=数据缺失
- **评分趋势图**：最近 N 天（默认14天）的买入分、卖出分和置信度变化
- **关键因子时序图**：VIX、美债、A股宽度、沪深300等关键指标历史走势
- **A股每日快照图**：来自 `state/a_market_snapshot.csv` 的历史数据
- **数据预警**：缺失/过期数据警告

### 两周历史数据说明

仪表盘的趋势图基于**每次 CLI 运行**自动追加到 `output/run_history.csv` 的快照。
只要定期（每日/每周）运行一次 CLI，历史就会积累，仪表盘即可展示完整趋势。
A股每日快照（`state/a_market_snapshot.csv`）在 CLI 运行时直接从市场数据源拉取历史，可立即显示多日走势。

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
