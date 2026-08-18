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
- `output/decision_tree.dot`
- `output/decision_tree.png`（Graphviz可用时）
- `state/a_market_snapshot.csv`

## 数据源与回退链（关键因子）

- A股指数/成交额：优先 `AKShare:stock_zh_index_daily_em`，失败时回退 `baostock:query_history_k_data_plus`
- A股横截面宽度：优先 `AKShare:stock_zh_a_spot_em`，失败时回退 `AKShare:stock_zh_a_spot`
- 海外市场/汇率/商品：`yfinance`，并对 HSTECH/A50 使用多 ticker 回退链
- 美债/信用/Fed：`FRED`（未配置 `FRED_API_KEY` 时会明确告警并降低置信度）

程序会区分：
- `missing`：数据缺失（无值）
- `stale`：数据存在但超出最大允许滞后天数（会告警并下调置信度）

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
