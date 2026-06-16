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

### 1.1 Token敏感性实验

**结论**：Tokenizer 对价格敏感度更高，容量分配合理。

| 扰动特征 | Hamming Distance | Decode MAE |
|----------|------------------|------------|
| close | **0.92**（最高） | 0.43 |
| vol | 0.74 | 0.31 |
| amt | 0.73 | 0.31 |

**问题位置确认**：
```text
Tokenizer问题：10%（已排除）
Predictor问题：90%（CE目标偏向量能）
```

---

### 1.2 误差分解诊断

**新增评估指标**：

```python
close_delta_mae = mean(abs(pred_return - actual_return))
close_delta_bias = mean(pred_return - actual_return)
pred_std vs actual_std  # 均值回归程度
direction_acc = DA
```

**close 特征诊断结果**：

| Step | IC | MAE | MAE/act_std | std_ratio | bias |
|------|-----|-----|-------------|-----------|------|
| +1 | 0.0065 | 0.0296 | **0.98** | 0.91 | 0.0064 |
| +2 | 0.0205 | 0.0374 | 0.92 | 0.84 | -0.0088 |
| +3 | 0.0113 | 0.0489 | 0.92 | 0.81 | -0.0119 |
| +5 | 0.0039 | 0.0649 | 0.91 | 0.78 | -0.0319 |

**核心问题**：

> **MAE / actual_std ≈ 0.98 → 预测误差等于市场波动本身，方向预测基本随机**

这不是均值回归（std_ratio ≈ 0.9），也不是系统偏差（bias很小），而是**方向预测能力缺失**。

**vol/amt 问题**：
- IC = 0.99+（方向正确）
- std_ratio = 26+（幅度失控）

---

## Phase 2 实施方案

### 2.1 核心改动

**训练目标改为相对位移**：

```python
pred_return = (pred_close - baseline_close) / baseline_close
actual_return = (actual_close - baseline_close) / baseline_close
```

而不是绝对价格 `future_close`。

**Loss 结构**：

```python
total_loss =
    prediction_only_CE        # 主损失：token 生成能力
    + λ1 * close_delta_loss   # 辅助：价格方向
    + λ2 * range_loss         # 辅助：波动范围
```

**初始参数**：

```python
λ1 = 0.1   # 不要太大，避免破坏预训练
λ2 = 0.03  # range 为次要目标
```

---

### 2.2 close_delta_loss 实现要点

**正确路径**（不绕 token）：

```python
# 生成 tokens → decode → denormalize → delta_loss
pred_tokens = argmax(logits[:, -pred_len:, :])
z = tokenizer.decode(pred_tokens, half=True)
pred_raw = z * stds + means
pred_delta = (pred_raw.close - baseline.close) / baseline.close
loss = SmoothL1(pred_delta, actual_delta, beta=0.005)
```

**错误路径**（已验证无效）：

```python
hidden[:, -1, :] → Linear → delta_loss  # ❌ 绕开 token 生成
```

---

### 2.3 SmoothL1 参数

```python
beta = 0.005  # 约 0.5% 误差阈值
# 网格搜索范围：0.003 ~ 0.01（0.3% ~ 1%）
```

A股低波动，MSE 被少数跳变样本拖偏，SmoothL1 更稳定。

---

### 2.4 Horizon 加权

越近越重：

```python
weights = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3]
# 或简化：gamma=0.75 → [1.0, 0.75, 0.56, ...]
```

---

### 2.5 实验步骤

| Step | 改动 | 目标 |
|------|------|------|
| 1 | 加 close_delta_loss（λ1=0.1） | 观察 close IC 变化 |
| 2 | 网格 λ1 ∈ [0.05, 0.1, 0.2, 0.3] | 找最优权重 |
| 3 | 加 range_loss（λ2=0.03） | K线结构稳定性 |
| 4 | 网格 SmoothL1 beta | 找最优阈值 |

---

### 2.6 预期效果

| 指标 | 当前 | 目标 |
|------|------|------|
| close IC | 0.006 | 0.10+ |
| close MAE/act_std | 0.98 | <0.5 |
| vol IC | 0.99 | 0.50~0.70 |
| DA (close) | 0.48 | 0.55+ |

---

## Phase 1 完成状态

```text
✅ Token敏感性实验 → 问题在 Predictor
✅ 误差分解诊断 → close 方向预测失败，MAE ≈ actual_std
```

---

## Phase 2 待执行

```text
⏳ 实现 train_single_step_v2.py
⏳ decoded close_delta SmoothL1
⏳ horizon 加权
```

---

## 文件索引

| 文件 | 用途 |
|------|------|
| `docs/DIAGNOSIS_PLAN.md` | 主诊断文档（本文件） |
| `docs/PHASE2_PLAN.md` | Phase 2 详细技术方案 |
| `finetune/token_sensitivity_analysis.py` | Token敏感性实验脚本 |
| `finetune/error_decomposition_diagnosis.py` | 误差分解诊断脚本 |
| `finetune/EXPERIENCE.md` | 经验教训记录 |