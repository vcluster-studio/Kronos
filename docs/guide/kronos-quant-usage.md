# Kronos 量化使用手册

本文档面向已有量化平台的用户，说明如何将 Kronos 作为信号引擎接入现有系统。

---

## 一、Kronos 能提供什么

Kronos 做且只做一件事：**给定历史 K 线，预测未来 N 期的完整 OHLCV 序列**。

```
输入: 过去 lookback 天的 OHLCV + 时间戳
输出: 未来 pred_len 天的 OHLCV + 时间戳
```

从这一件事中，你可以提取五类量化信息：

| 信息类型 | 提取方式 | 量化用途 |
|---------|---------|---------|
| 方向信号 | 预测终点价 - 当前价 | 涨跌判断、横截面排名 |
| 路径形态 | 预测序列的走势形状 | 判断上涨/下跌的方式和节奏 |
| 概率分布 | 多次采样统计 | 置信度、不确定性 |
| 波动率 | 预测序列的波动特征 | 风险评估、仓位缩放 |
| 横截面信息 | 多股票信号对比 | 选股排名、行业分布 |

---

## 二、基础调用

### 2.1 加载模型

```python
from model import Kronos, KronosTokenizer, KronosPredictor

tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")  # 或 Kronos-base
predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)
```

**模型选择**：

| 模型 | 参数量 | 推理速度（pred_len=10） | 适合场景 |
|------|--------|----------------------|---------|
| Kronos-mini | 4.1M | ~0.5s/只 | 快速扫描、小盘股池 |
| Kronos-small | 24.7M | ~1.5s/只 | 日常选股 |
| Kronos-base | 102.3M | ~3s/只 | 高精度场景 |

**分词器必须匹配模型**：

| 模型 | 分词器 |
|------|--------|
| Kronos-mini | `NeoQuasar/Kronos-Tokenizer-2k` |
| Kronos-small | `NeoQuasar/Kronos-Tokenizer-base` |
| Kronos-base | `NeoQuasar/Kronos-Tokenizer-base` |

### 2.2 单只股票预测

```python
import pandas as pd

# 准备数据（DataFrame 必须包含 open/high/low/close，volume/amount 可选）
df = pd.read_csv("your_data.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

lookback = 400
pred_len = 10

x_df = df.iloc[-lookback:][['open', 'high', 'low', 'close', 'volume', 'amount']]
x_timestamp = df.iloc[-lookback:]['timestamps']
y_timestamp = pd.bdate_range(
    start=df['timestamps'].iloc[-1] + pd.Timedelta(days=1),
    periods=pred_len
)

# 执行预测
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=pd.Series(y_timestamp),
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=False
)

# pred_df: DataFrame, index=y_timestamp
# 列: open, high, low, close, volume, amount
# 行: pred_len 行，每行一个未来时间点
```

### 2.3 批量预测

```python
# 所有 DataFrame 必须等长（lookback 相同，pred_len 相同）
pred_df_list = predictor.predict_batch(
    df_list=[df1, df2, df3],
    x_timestamp_list=[ts1, ts2, ts3],
    y_timestamp_list=[yts1, yts2, yts3],
    pred_len=10,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=True
)
# 返回 List[DataFrame]，顺序与输入一致
```

### 2.4 数据格式要求

**必需列**：`open`, `high`, `low`, `close`

**可选列**：`volume`, `amount`（缺失时自动填零，但建议提供以获得更准确的预测）

**时间戳**：必须转为 `datetime` 类型

**数据质量**：不允许 NaN 值，缺失数据需提前填充

**归一化**：KronosPredictor 内部自动做 instance-level 归一化（减均值除标准差），无需手动处理

---

## 三、信号提取

### 3.1 方向信号

最基础的用法，判断涨跌方向：

```python
last_close = df['close'].iloc[-1]

# 终点信号：预测期末价格变化
end_signal = pred_df['close'].iloc[-1] - last_close

# 均值信号：预测期内平均价格变化
mean_signal = pred_df['close'].mean() - last_close

# 最大涨幅信号
max_signal = pred_df['high'].max() - last_close

# 最大跌幅信号
min_signal = pred_df['low'].min() - last_close
```

**信号选择建议**：

| 信号 | 特点 | 适用场景 |
|------|------|---------|
| `end_signal` | 关注终点，噪声小 | 中长期持仓 |
| `mean_signal` | 关注整体趋势，更平滑 | 稳健型策略 |
| `max_signal` | 关注上行空间 | 激进型策略 |
| `min_signal` | 关注下行风险 | 风控辅助 |

### 3.2 路径形态

从预测序列的走势形状中提取信息：

```python
close_path = pred_df['close'].values

# 最高点出现的时间（前半段 vs 后半段）
peak_idx = np.argmax(close_path)
peak_timing = peak_idx / len(close_path)  # 0~1, <0.5=前半段见顶

# 路径内最大回撤
cummax = np.maximum.accumulate(close_path)
drawdowns = (close_path - cummax) / cummax
path_max_drawdown = drawdowns.min()

# 路径波动率
path_returns = np.diff(close_path) / close_path[:-1]
path_volatility = np.std(path_returns)

# 是否 V 型（先跌后涨）
first_half_min = close_path[:len(close_path)//2].min()
second_half_max = close_path[len(close_path)//2:].max()
is_v_shape = (first_half_min < last_close) and (second_half_max > last_close)

# 是否倒 V 型（先涨后跌）
is_inverse_v = (peak_timing < 0.5) and (close_path[-1] < last_close)
```

**形态与交易含义**：

| 形态 | 特征 | 持仓策略 |
|------|------|---------|
| 稳步上涨 | 单调递增 | 满仓持有 |
| 先跌后涨(V型) | 前半段低点 < 当前价 | 逢低加仓 |
| 先涨后跌(倒V型) | 前半段高点 > 当前价 | 快进快出，设紧止盈 |
| 震荡无方向 | 终点≈起点，中间波动 | 不参与 |
| 持续下跌 | 单调递减 | 不买入 |

### 3.3 概率分布（多次采样）

Kronos 是生成式模型，每次采样产生不同的预测路径。利用这一特性获取概率信息：

```python
# 多次采样
sample_count = 5
samples = []
for _ in range(sample_count):
    pred = predictor.predict(
        df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
        pred_len=10, T=1.0, top_p=0.9, sample_count=1, verbose=False
    )
    samples.append(pred)

# === 从采样结果提取概率信息 ===

end_prices = [s['close'].iloc[-1] for s in samples]

# 1. 方向一致性（几次看涨/几次看跌）
direction_consistency = sum(p > last_close for p in end_prices) / sample_count
# 0.8 = 80%采样看涨 → 高置信度
# 0.6 = 60%采样看涨 → 中等置信度
# 0.4 = 40%采样看涨 → 方向不确定

# 2. 期望收益
expected_return = np.mean(end_prices) - last_close

# 3. 不确定性（采样标准差）
uncertainty = np.std(end_prices)

# 4. 置信区间
ci_90 = np.percentile(end_prices, [5, 95])

# 5. 涨跌范围
upside = max(end_prices) - last_close    # 最大上行空间
downside = min(end_prices) - last_close  # 最大下行风险
```

**注意**：`predictor.predict(sample_count=5)` 内部已经做了5次采样并取均值返回。为了获取各次采样的独立路径，需要循环调用 `sample_count=1`。

**不确定性解读**：

| uncertainty / last_close | 含义 | 操作建议 |
|-------------------------|------|---------|
| < 1% | 非常确定 | 可按信号正常仓位操作 |
| 1% ~ 3% | 中等不确定 | 正常仓位 |
| 3% ~ 5% | 较不确定 | 减半仓位 |
| > 5% | 非常不确定 | 轻仓或观望 |

### 3.4 波动率指标

从预测路径中直接提取风险指标：

```python
# 1. 预测波动率（年化）
pred_returns = pred_df['close'].pct_change().dropna()
predicted_vol = pred_returns.std() * np.sqrt(252)

# 2. 预测路径内最大回撤
cummax = pred_df['close'].cummax()
drawdown = (pred_df['close'] - cummax) / cummax
predicted_max_drawdown = drawdown.min()

# 3. 从多次采样估计 VaR
all_end_returns = [(s['close'].iloc[-1] / last_close - 1) for s in samples]
var_95 = np.percentile(all_end_returns, 5)  # 95% VaR

# 4. 预测振幅
predicted_range = (pred_df['high'].max() - pred_df['low'].min()) / last_close
```

### 3.5 横截面信号

对股票池批量预测后，提取横截面信息：

```python
# 批量预测全股票池
stock_signals = {}
for stock in stock_pool:
    hist = get_history(stock, lookback=400)
    pred = predictor.predict(hist, ..., pred_len=10, sample_count=1)
    stock_signals[stock] = pred['close'].iloc[-1] - hist['close'].iloc[-1]

signal_series = pd.Series(stock_signals)

# 1. 横截面排名（百分位）
rank_pct = signal_series.rank(pct=True)

# 2. Z-Score 标准化
zscore = (signal_series - signal_series.mean()) / signal_series.std()

# 3. 行业分布
industry_signals = signal_series.groupby(lambda s: get_industry(s)).mean()
# 检查是否有行业集中偏向
```

---

## 四、采样参数调优

### 4.1 温度 T

控制采样随机性。**值越低，预测越确定性；值越高，预测越多样**。

```python
# 保守/确定性预测
pred = predictor.predict(..., T=0.5)

# 正常预测
pred = predictor.predict(..., T=1.0)

# 探索性预测
pred = predictor.predict(..., T=1.5)
```

| T值 | 效果 | 适用场景 |
|-----|------|---------|
| 0.3 ~ 0.6 | 高确定性，低多样性 | 信号确认、止损判断 |
| 0.7 ~ 1.0 | 平衡 | 日常选股 |
| 1.0 ~ 1.5 | 低确定性，高多样性 | 探索潜在机会、压力测试 |

### 4.2 top_p（核采样）

控制采样范围。只从累计概率达到 top_p 的候选token中采样。

```python
# 更集中（从概率最高的少数token中采样）
pred = predictor.predict(..., top_p=0.8)

# 更分散
pred = predictor.predict(..., top_p=0.95)
```

| top_p | 效果 | 适用场景 |
|-------|------|---------|
| 0.8 | 集中，预测更稳定 | 信号确认 |
| 0.9 | 平衡（默认） | 日常使用 |
| 0.95 | 分散，预测更多样 | 探索场景 |

### 4.3 sample_count

多次采样取平均，**降低随机性，提高信号稳定性**。

```python
# 单次采样（最快，但噪声大）
pred = predictor.predict(..., sample_count=1)

# 5次采样取平均（推荐）
pred = predictor.predict(..., sample_count=5)

# 10次采样取平均（最稳定，但慢5-10倍）
pred = predictor.predict(..., sample_count=10)
```

| sample_count | 推理时间 | 信号质量 | 适用场景 |
|-------------|---------|---------|---------|
| 1 | 基准 | 噪声较大 | 批量初筛 |
| 3 | 3x | 较好 | 日常选股 |
| 5 | 5x | 稳定 | 信号确认 |
| 10 | 10x | 很稳定 | 高精度场景 |

**重要**：`predictor.predict(sample_count=5)` 返回的是5次采样的**均值**，不是5条独立路径。如需独立路径，需循环调用 `sample_count=1`。

### 4.4 推荐参数组合

| 场景 | T | top_p | sample_count | 理由 |
|------|---|-------|-------------|------|
| 日频选股（批量） | 0.8 | 0.9 | 1 | 速度优先，批量做均值等效 |
| 选股信号确认 | 0.8 | 0.9 | 3 | 平衡速度和稳定性 |
| 监控增量推理 | 0.6 | 0.85 | 5 | 需要确定性判断 |
| 置信度评估 | 1.0 | 0.9 | 5次独立 | 获取概率分布 |

---

## 五、接入选股模块

### 5.1 基本流程

```
股票池 → 过滤(流动性/ST) → 批量Kronos预测 → 信号提取 → 排序 → 输出
```

```python
def kronos_stock_selection(predictor, stock_pool, date, config):
    """Kronos选股：输出信号供外部选股模块使用"""

    lookback = config.get('lookback', 400)
    pred_len = config.get('pred_len', 10)
    sample_count = config.get('sample_count', 1)

    results = {}
    for stock in stock_pool:
        # 获取历史数据
        hist = get_history(stock, date=date, lookback=lookback)
        if hist is None or len(hist) < lookback:
            continue

        # 准备时间戳
        x_timestamp = hist['timestamps']
        y_timestamp = pd.bdate_range(
            start=date + pd.Timedelta(days=1), periods=pred_len
        )

        # Kronos预测
        pred_df = predictor.predict(
            df=hist[['open', 'high', 'low', 'close', 'volume', 'amount']],
            x_timestamp=x_timestamp,
            y_timestamp=pd.Series(y_timestamp),
            pred_len=pred_len,
            T=config.get('T', 0.8),
            top_p=config.get('top_p', 0.9),
            sample_count=sample_count,
            verbose=False
        )

        # 提取信号
        last_close = hist['close'].iloc[-1]
        results[stock] = {
            'signal_end': pred_df['close'].iloc[-1] - last_close,
            'signal_mean': pred_df['close'].mean() - last_close,
            'pred_path': pred_df,                    # 保存路径供监控使用
            'entry_price': last_close,
            'predicted_vol': pred_df['close'].pct_change().std() * np.sqrt(252),
            'predicted_max_dd': _calc_path_max_drawdown(pred_df),
        }

    return results
```

### 5.2 带置信度的选股

```python
def kronos_selection_with_confidence(predictor, stock_pool, date, config):
    """带置信度评估的选股"""

    sample_count = config.get('confidence_samples', 5)
    results = {}

    for stock in stock_pool:
        hist = get_history(stock, date=date, lookback=config['lookback'])
        if hist is None:
            continue

        last_close = hist['close'].iloc[-1]

        # 多次独立采样
        end_prices = []
        pred_paths = []
        for _ in range(sample_count):
            pred = predictor.predict(
                df=hist[['open', 'high', 'low', 'close', 'volume', 'amount']],
                x_timestamp=hist['timestamps'],
                y_timestamp=pd.Series(pd.bdate_range(
                    start=date + pd.Timedelta(days=1), periods=config['pred_len']
                )),
                pred_len=config['pred_len'],
                T=1.0, top_p=0.9, sample_count=1, verbose=False
            )
            end_prices.append(pred['close'].iloc[-1])
            pred_paths.append(pred)

        # 统计
        end_prices = np.array(end_prices)
        direction_consistency = np.mean(end_prices > last_close)
        expected_return = np.mean(end_prices) - last_close
        uncertainty = np.std(end_prices) / last_close

        results[stock] = {
            'signal': expected_return,
            'confidence': direction_consistency,
            'uncertainty': uncertainty,
            'pred_path_mean': _average_paths(pred_paths),
            'pred_path_samples': pred_paths,
        }

    return results
```

### 5.3 输出格式建议

选股模块向外部系统输出的标准格式：

```python
# 每只股票的选股信号
stock_signal = {
    'stock': '600580',
    'date': '2026-05-05',
    'signal': 0.032,               # Alpha信号值（预测涨跌幅）
    'confidence': 0.80,            # 方向一致性
    'uncertainty': 0.015,          # 不确定性
    'predicted_vol': 0.22,         # 预测年化波动率
    'predicted_max_dd': -0.04,     # 预测路径内最大回撤
    'pred_path': pred_df,          # 完整预测路径（供监控用）
    'entry_price': 15.20,          # 当前收盘价
    'stop_loss': 13.98,            # 建议止损价(-8%)
    'take_profit': 16.72,          # 建议止盈触发价(+10%)
}
```

---

## 六、接入监控模块

### 6.1 核心前提：保存预测路径

选股时**必须保存完整的预测路径**，监控才能工作：

```python
# 选股时保存
pred_cache[stock] = {
    'pred_path': pred_df,           # 完整OHLCV预测
    'entry_price': last_close,      # 买入价
    'entry_date': date,             # 买入日期
    'signal_value': signal,         # 信号强度
    'confidence': confidence,       # 置信度
    'stop_loss': last_close * 0.92, # 止损价
    'take_profit': last_close * 1.10,# 止盈触发价
}
```

### 6.2 规则监控（零推理成本）

纯价格比较，不需要调用Kronos：

```python
def rule_monitor(pred_cache_entry, current_price, hold_days):
    """基于规则的持仓监控"""

    entry = pred_cache_entry['entry_price']
    actions = []

    # 硬止损
    if current_price <= pred_cache_entry['stop_loss']:
        actions.append(('SELL', '硬止损', f'价格{current_price}低于止损线'))

    # 跟踪止盈
    unrealized = (current_price - entry) / entry
    if unrealized > 0.05:
        highest = max(pred_cache_entry.get('highest', entry), current_price)
        pred_cache_entry['highest'] = highest
        drawdown = (current_price - highest) / highest
        if drawdown < -0.05:
            actions.append(('SELL', '跟踪止盈', f'盈利{unrealized:.1%}，回撤{drawdown:.1%}'))

    # 时间止损
    pred_len = len(pred_cache_entry['pred_path'])
    if hold_days >= pred_len and unrealized < 0.02:
        actions.append(('SELL', '时间止损', f'持有{hold_days}天信号未兑现'))

    return actions
```

### 6.3 路径偏离监控（零推理成本）

将实际价格与选股时保存的预测路径对比：

```python
def path_deviation_monitor(pred_cache_entry, current_price, hold_days):
    """预测路径偏离检测"""

    pred_path = pred_cache_entry['pred_path']
    actions = []

    if hold_days <= len(pred_path):
        expected = pred_path['close'].iloc[hold_days - 1]
        deviation = (current_price - expected) / expected

        # 大幅低于预测 → 异常
        if deviation < -0.03:
            actions.append(('ALERT', '路径偏离下行', f'实际{current_price}低于预测{expected:.2f}，偏离{deviation:.1%}'))

        # 大幅高于预测 → 超预期
        elif deviation > 0.05:
            actions.append(('INFO', '超预期上行', f'实际{current_price}高于预测{expected:.2f}，偏离{deviation:.1%}'))

    return actions
```

### 6.4 增量推理监控（需要推理，按需触发）

仅在规则监控或路径偏离触发告警后使用：

```python
def incremental_monitor(predictor, pred_cache_entry, current_hist, date, config):
    """增量推理：重新预测，判断信号是否反转"""

    # 用最新数据重新预测
    new_pred = predictor.predict(
        df=current_hist[['open', 'high', 'low', 'close', 'volume', 'amount']],
        x_timestamp=current_hist['timestamps'],
        y_timestamp=pd.Series(pd.bdate_range(
            start=date + pd.Timedelta(days=1), periods=config['pred_len']
        )),
        pred_len=config['pred_len'],
        T=0.6,   # 较低温度，获取确定性判断
        top_p=0.85,
        sample_count=1,
        verbose=False
    )

    # 判断信号是否反转
    last_close = current_hist['close'].iloc[-1]
    old_signal = pred_cache_entry['signal_value']
    new_signal = new_pred['close'].iloc[-1] - last_close

    if old_signal > 0 and new_signal < 0:
        return [('SELL', '信号反转', f'原信号{old_signal:.3f}，新信号{new_signal:.3f}，由涨转跌')]

    if old_signal > 0 and new_signal > 0 and new_signal < old_signal * 0.3:
        return [('REDUCE', '信号大幅衰减', f'原信号{old_signal:.3f}，新信号{new_signal:.3f}')]

    return [('HOLD', '信号维持', f'新信号{new_signal:.3f}')]
```

---

## 七、接入回测模块

### 7.1 回测中的Kronos调用

回测需要在历史日期上模拟Kronos预测。核心是**只用该日期之前的数据**：

```python
def backtest_kronos_predict(predictor, stock, date, config):
    """回测中的Kronos预测：严格使用date之前的数据"""

    # 获取截止到date的历史数据（不能用到未来数据！）
    hist = get_history_before(stock, date=date, lookback=config['lookback'])
    if hist is None or len(hist) < config['lookback']:
        return None

    x_timestamp = hist['timestamps']
    y_timestamp = pd.bdate_range(
        start=date + pd.Timedelta(days=1), periods=config['pred_len']
    )

    pred = predictor.predict(
        df=hist[['open', 'high', 'low', 'close', 'volume', 'amount']],
        x_timestamp=x_timestamp,
        y_timestamp=pd.Series(y_timestamp),
        pred_len=config['pred_len'],
        T=config.get('T', 0.8),
        top_p=config.get('top_p', 0.9),
        sample_count=1,    # 回测中用1次采样加速
        verbose=False
    )

    return pred
```

### 7.2 回测加速策略

Kronos推理是回测的最大瓶颈。加速方法：

| 方法 | 加速效果 | 实现方式 |
|------|---------|---------|
| sample_count=1 | 5x | 回测中只需相对排名，不需要精确信号 |
| 缓存预测结果 | 避免重复推理 | 同一股票同一lookback窗口只推理一次 |
| 降低pred_len | 线性加速 | 用pred_len=5代替10，信号质量略降 |
| 使用Kronos-mini | 3-6x | 牺牲精度换速度 |
| 批量推理 | 2-3x | 使用predict_batch批量处理 |

```python
# 回测缓存
pred_cache = {}

def cached_backtest_predict(predictor, stock, date, config):
    cache_key = f"{stock}_{date.strftime('%Y%m%d')}"
    if cache_key in pred_cache:
        return pred_cache[cache_key]

    pred = backtest_kronos_predict(predictor, stock, date, config)
    pred_cache[cache_key] = pred
    return pred
```

### 7.3 共用推理接口

如果你的回测模块已有基于公式的策略，Kronos可以通过统一接口接入：

```python
class KronosSignalProvider:
    """Kronos信号提供者：向回测引擎提供与公式策略格式一致的信号"""

    def __init__(self, predictor, config):
        self.predictor = predictor
        self.config = config

    def get_signals(self, date, stock_pool):
        """
        返回格式与公式策略一致的信号

        Returns:
            pd.Series: {stock: signal_value}
        """
        signals = {}
        for stock in stock_pool:
            pred = backtest_kronos_predict(
                self.predictor, stock, date, self.config
            )
            if pred is not None:
                hist = get_history_before(stock, date, self.config['lookback'])
                last_close = hist['close'].iloc[-1]
                signals[stock] = pred['close'].iloc[-1] - last_close

        return pd.Series(signals, name='kronos_signal')
```

---

## 八、完整信号提取参考

以下是一个完整的信号提取函数，覆盖所有可获取的信息：

```python
def extract_all_signals(predictor, hist_df, date, config):
    """
    从Kronos提取全部可量化信息

    Args:
        predictor: KronosPredictor实例
        hist_df: 历史OHLCV数据（至少lookback行）
        date: 当前日期
        config: 参数配置

    Returns:
        dict: 全部可提取的信号和指标
    """
    lookback = config.get('lookback', 400)
    pred_len = config.get('pred_len', 10)
    last_close = hist_df['close'].iloc[-1]

    # === 1. 单次预测（快速获取路径和方向） ===
    x_df = hist_df.iloc[-lookback:][['open','high','low','close','volume','amount']]
    x_timestamp = hist_df.iloc[-lookback:]['timestamps']
    y_timestamp = pd.bdate_range(date + pd.Timedelta(days=1), periods=pred_len)

    pred_df = predictor.predict(
        df=x_df, x_timestamp=x_timestamp,
        y_timestamp=pd.Series(y_timestamp),
        pred_len=pred_len, T=0.8, top_p=0.9, sample_count=1, verbose=False
    )

    # --- 方向信号 ---
    signal_end = pred_df['close'].iloc[-1] - last_close
    signal_mean = pred_df['close'].mean() - last_close
    signal_max = pred_df['high'].max() - last_close
    signal_min = pred_df['low'].min() - last_close

    # --- 路径形态 ---
    close_path = pred_df['close'].values
    peak_idx = np.argmax(close_path)
    peak_timing = peak_idx / len(close_path)
    path_max_dd = _calc_path_max_drawdown(pred_df)
    path_vol = pred_df['close'].pct_change().std() * np.sqrt(252)

    # === 2. 多次采样（获取概率分布，可选） ===
    confidence = None
    uncertainty = None
    if config.get('with_confidence', False):
        end_prices = []
        for _ in range(config.get('confidence_samples', 5)):
            p = predictor.predict(
                df=x_df, x_timestamp=x_timestamp,
                y_timestamp=pd.Series(y_timestamp),
                pred_len=pred_len, T=1.0, top_p=0.9,
                sample_count=1, verbose=False
            )
            end_prices.append(p['close'].iloc[-1])

        end_prices = np.array(end_prices)
        confidence = np.mean(end_prices > last_close)
        uncertainty = np.std(end_prices) / last_close

    # === 3. 汇总输出 ===
    return {
        # 方向信号
        'signal_end': signal_end,
        'signal_mean': signal_mean,
        'signal_max': signal_max,
        'signal_min': signal_min,

        # 路径形态
        'peak_timing': peak_timing,
        'path_max_drawdown': path_max_dd,
        'path_volatility': path_vol,

        # 概率分布（仅多次采样时可用）
        'confidence': confidence,
        'uncertainty': uncertainty,

        # 预测路径
        'pred_path': pred_df,

        # 辅助信息
        'entry_price': last_close,
        'stop_loss': last_close * (1 - config.get('stop_loss_rate', 0.08)),
        'take_profit': last_close * (1 + config.get('take_profit_rate', 0.10)),
    }


def _calc_path_max_drawdown(pred_df):
    close = pred_df['close']
    cummax = close.cummax()
    dd = (close - cummax) / cummax
    return dd.min()
```

---

## 九、注意事项

### 9.1 前视偏差

回测中必须严格使用 `date` 之前的数据。KronosPredictor 内部做归一化时用的均值和标准差来自输入数据，只要输入数据不包含未来信息，就不会有前视偏差。

### 9.2 预测衰减

Kronos预测的是未来 `pred_len` 期的K线。信号有效性与预测距离成反比：

```
第1-3天: 信号最强
第4-7天: 信号中等
第8-10天: 信号较弱
第10天后: 信号可能失效
```

建议：预测路径的前半段比后半段更可靠。

### 9.3 归一化边界

KronosPredictor 使用 instance-level 归一化（单条序列自身减均值除标准差）。这意味着：
- 预测值在归一化空间中是合理的
- 反归一化回原始价格时，如果输入数据波动极大，可能产生偏离
- 对极端行情（涨停/跌停/停牌复牌）的预测可靠性降低

### 9.4 成交量预测的局限性

Kronos对 `volume` 和 `amount` 的预测精度远低于价格预测。建议：
- 主要使用价格信号
- 成交量信号仅作为辅助参考
- 无成交量数据时，提供 `volume=0, amount=0` 也能正常预测价格

### 9.5 模型微调

预训练模型是通用模型。如需针对特定市场（如A股日频）获得更好效果，建议使用 `finetune/` 目录的微调管道。详见 [finetuning.md](../tutorial/finetuning.md)。
