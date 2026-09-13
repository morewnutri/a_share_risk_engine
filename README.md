# A股多因子外部风险评分引擎

## 安装

```bash
python -m pip install -r requirements.txt
```

Graphviz PNG 可视化还需要系统安装 Graphviz；没有也没关系，程序仍输出 `output/decision_tree.dot`。

## FRED API KEY（可选）

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

未设置时程序自动回退到 FRED 官方公开 CSV（不需要 Key）；若公开端点也不可用，才会标记缺失并降低置信度，不会补假数据。

## 运行

```bash
python a_share_risk_engine.py
```

输出：
- `output/latest_score.json`
- `output/factor_report.csv`
- `output/feature_snapshot.csv`
- `output/monthly_macd_alerts.csv`
- `output/monthly_macd_alerts.json`
- `output/data_source_health.csv`（每个序列的来源链、样本数、首末日期和过期状态）
- `output/decision_tree.dot`
- `output/decision_tree.png`（Graphviz可用时）
- `state/a_market_snapshot.csv`
- `state/series_cache/*.csv`（成功数据的本地缓存）

## 数据源与回退链（关键因子）

- A股指数/成交额：`AKShare:stock_zh_index_daily_em` 为主，`AKShare:index_zh_a_hist` 为接口级回退，`baostock:query_history_k_data_plus` 独立拉取并补齐缺失日期
- A股横截面宽度：优先 `AKShare:stock_zh_a_spot_em`，失败时回退 `AKShare:stock_zh_a_spot`
- 海外市场/汇率/商品：`yfinance`，并对 HSTECH/A50 使用多 ticker 回退链
- 美债/信用/Fed：优先 `FRED API`；未配置 `FRED_API_KEY` 时自动使用 `FRED public CSV`
- 所有成功序列都会落入 `state/series_cache`；下次先加载缓存，再由实时源覆盖同日值。短暂断网或单个接口失效不会把历史清空

注意：Baostock 很适合补 A 股指数日线，但不能替代实时全市场横截面，也不能提供 FRED 的实际利率/信用利差。因此这里采用按数据类型拆分的多源链，而不是把所有数据强行换成一个源。

程序会区分：
- `missing`：数据缺失（无值）
- `stale`：数据存在但超出最大允许滞后天数（会告警、降低置信度，并从评分/共振中排除）

## 月线 MACD 实时预警与因子化评分

监控指数：上证指数、沪深300、中证1000、创业板指、科创50。

MACD 使用标准月线参数 `12/26/9`。最后一根月线不是等到月末才生成，而是每天用当月最新收盘价更新，因此盘中月份一旦出现实时死叉就会立即识别。每次运行还会计算：

- `gap = DIF - DEA`
- 本月每天的 gap 收窄斜率
- 按当前斜率估计的死叉交易日数
- 令本月 `DIF = DEA` 的临界收盘价
- 当前价格距离临界价的百分比

告警级别：

- `DEATH_CROSS_LIVE`：本月未收盘，但实时月线已死叉
- `DEATH_CROSS_CONFIRMED`：最近已完成月线确认死叉
- `PRE_DEATH_CROSS_CRITICAL`：尚未死叉，但预计不超过 5 个交易日、距离临界价不超过 2.5%，或正差已收窄至少 75%
- `PRE_DEATH_CROSS_WARNING`：预计不超过 15 个交易日、距离临界价不超过 6%，或正差已收窄至少 45%
- `WATCH`：正差正在收窄
- `GOLDEN_CROSS_LIVE`：本月未收盘，但实时月线已金叉
- `GOLDEN_CROSS_CONFIRMED`：最近已完成月线确认金叉
- `BULLISH`：金叉后/多头扩张区间
- `BEARISH`：死叉后的空头区间
- `SAFE`：暂未临近死叉，也未形成更强多头扩张

这些月线 MACD 告警不会直接变成顶层 `SELL` / `RISK_OFF` 指令，而是先转成标准风险因子并进入总分：

- 上证：权重 `8.0`
- 沪深300：权重 `4.0`
- 中证1000 / 创业板 / 科创50：各 `2.0`

信号映射（`+` 为增风险，`-` 为降风险）：

- `DEATH_CROSS_CONFIRMED`=`+1.00`
- `DEATH_CROSS_LIVE`=`+0.80`
- `BEARISH`=`+0.55`
- `PRE_DEATH_CROSS_CRITICAL`=`+0.50`
- `PRE_DEATH_CROSS_WARNING`=`+0.30`
- `WATCH`=`+0.10`
- `SAFE`=`0.00`
- `BULLISH`=`-0.20`
- `GOLDEN_CROSS_LIVE`=`-0.50`
- `GOLDEN_CROSS_CONFIRMED`=`-0.75`

若月线 MACD 数据缺失/历史不足，则该因子记为 missing；若数据 stale，则记为 stale 因子并完全排除，不参与评分或共振。

要做到“及时”，程序仍需在每个交易日收盘后运行一次；算法提前预警不能替代调度器。Windows 任务计划程序或 CI 定时任务均可执行：

```bash
python a_share_risk_engine.py
```

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
- 核心A股指数的月线 MACD 因子与多指数共振

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

可选手工数据若暂时没有，请保留为 `null`，程序会跳过，不会把 `0` 当成真实中性值。

## 重要

当前阈值是研究用初始值。实盘前应对 2015~现在做 walk-forward 回测并重新校准权重/阈值。
