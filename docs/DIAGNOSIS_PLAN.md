# Kronos A股日线选股微调思考（2026-06-09）

## 背景

当前目标并非构建一个传统量化 Alpha 模型。

目标是：

对于普通投资者，以可理解的方式展示：

> 为什么选择这只股票？

回答形式为：

> 因为模型预测未来 5~10 个交易日的 K 线整体处于上涨趋势。

因此：

模型最终输出应当是：

```text
未来K线轨迹
```

而不是：

```text
因子分数
收益率预测值
RankIC
```

RankIC 等指标仅作为内部评估指标。

---

## 当前观察

在现阶段实验中发现：

```text
vol / amt 学习效果极好
OHLC 学习效果极差
```

经验记录：

```text
vol/amt IC → 0.99+
OHLC IC → 接近0
```

说明训练资源被大量分配到成交量相关信息。

暂时将该现象称为：

```text
梯度绑架（Gradient Hijacking）
```

---

## 关于问题位置的判断

目前不能确定问题一定发生在 Tokenizer。

存在两种可能：

### 假设1：Tokenizer阶段发生问题

Tokenizer 将大部分码本容量用于表示：

```text
vol
amt
```

而价格相关信息编码不足。

表现：

```text
token 对 volume 极其敏感
token 对 close 不敏感
```

---

### 假设2：Predictor阶段发生问题

Tokenizer 实际已经编码了价格信息。

但 Predictor 的训练目标：

```text
Cross Entropy
```

天然倾向学习：

```text
更稳定
更容易预测
更容易降低Loss
```

的信息。

对于日线市场：

```text
成交量
成交额
```

通常比：

```text
未来价格方向
```

更容易预测。

因此 Predictor 优先优化量能。

---

## 当前观点

目前倾向：

```text
Tokenizer问题：30%
Predictor问题：70%
```

但缺少实验证据。

因此不建议立即大规模修改模型结构。

---

## 建议优先实验

### 实验1：Token敏感性分析

目标：

确认 Tokenizer 主要编码什么信息。

方法：

构造三组数据：

```text
原始数据
close扰动数据
vol扰动数据
```

分别编码：

```python
token_a = tokenizer(original)
token_b = tokenizer(close_shuffle)
token_c = tokenizer(vol_shuffle)
```

统计：

```python
distance(token_a, token_b)
distance(token_a, token_c)
```

---

### 结果解释

情况A：

```text
close扰动影响很小
vol扰动影响很大
```

说明：

```text
Tokenizer容量主要编码量能
```

此时应考虑：

```text
Tokenizer Loss重加权
```

---

情况B：

```text
close扰动影响很大
vol扰动影响类似
```

说明：

```text
Tokenizer其实学到了价格
```

问题主要发生在 Predictor。

---

## 关于Loss重加权

目前不建议直接实施。

原因：

尚未证明问题发生在 Tokenizer。

如果未来证明确实存在：

```text
Tokenizer容量偏向量能
```

则可尝试：

```python
weights = [
    open  = 2.0
    high  = 2.0
    low   = 2.0
    close = 3.0
    vol   = 0.5
    amt   = 0.5
]
```

仅作为实验方向。

---

## 关于Return Loss

目前不建议把项目转型为：

```text
收益率预测模型
```

原因：

项目目标不是：

```text
预测收益率
```

而是：

```text
预测未来K线
```

收益率属于派生结果。

未来K线属于直接结果。

因此：

```text
未来K线预测
```

仍应作为第一目标。

---

## 关于Direction Head

根据历史经验：

```text
hidden[:, -1, :]
```

并不包含未来目标信息。

直接：

```text
hidden → BCE
```

已经验证无效。

未来不建议继续投入大量时间。

---

## 关于滑动窗口

当前设计：

```text
lookback + predict
```

合理。

不建议放弃。

问题核心不在滑动窗口。

而在：

```text
Token表示
+
Predictor训练目标
```

之间的信息分配。

---

## Phase 1 结果（2026-06-09）

**Token敏感性实验已完成**

结果：
```text
close 扰动 → Hamming 0.92（最高）
vol 扰动 → Hamming 0.74
amt 扰动 → Hamming 0.73
```

Price avg Hamming: 0.83
Volume avg Hamming: 0.74
Ratio: 0.88

**结论**：Tokenizer 对价格敏感度更高，容量分配合理。

**问题位置确认**：
```text
Tokenizer问题：10%（已排除）
Predictor问题：90%（CE目标偏向量能）
```

---

## 下一阶段计划

Phase 1

```text
完成 Token敏感性实验 ✅ DONE
```

回答：

```text
Tokenizer到底在编码什么？ → 价格信息已编码，问题不在Tokenizer
```

---

Phase 2

根据实验结果决定：

```text
改Predictor（主要方向）
```

具体方案：

1. **Feature-weighted CE Loss**
   - 给价格特征更高权重，抑制 vol/amt 学习速度

2. **两阶段训练**
   - 第一阶段：只训练价格部分
   - 第二阶段：放开全部特征

---

Phase 3

重新评估：

```text
未来K线预测质量
```

而不是单纯关注：

```text
Price IC
Return IC
```

---

最终目标始终保持不变：

```text
让模型能够生成未来5~10日
具有实际预测意义的K线轨迹
并能够向普通投资者解释
为什么选择这只股票。
```