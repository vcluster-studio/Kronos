# Kronos 微调经验教训

本文档记录微调过程中的关键错误和教训，避免重复踩坑。

---

## 1. 数据泄露

**问题**：`window_size = lookback + predict + 1` 导致泄露（commit 4c569aa 修复）

**影响**：所有 pre-4c569aa 模型的 IC 指标均无效（mini_v5 close_ic=0.33 是虚假的）

**正确做法**：`window_size = lookback + predict`（无 +1）

---

## 2. 评估时的 Ground Truth Leakage

**问题**：EvalCallback 用 ground truth tokens 做 forward，导致虚假高 IC

**原始代码**：
```python
t0, t1 = tokenizer.encode(norm)  # 包含预测位置的 GT tokens
s1_logits, s2_logits = model(t0[:, :-p], t1[:, :-p], ...)  # 输入去掉最后 p 步
s1_p = argmax(s1_logits[:, -p, :])  # 取 logits
```

**问题分析**：模型 forward 时实际上看到了预测位置的 ground truth 信息。

**修复方案**：改用 `auto_regressive_inference`
```python
pred = auto_regressive_inference(tokenizer, model, x_tensor, x_stamp, y_stamp, ...)
pred_norm = pred[0, lookback:lookback+pred_len, :]  # 只取预测部分
```

**影响**：
- eval_results.jsonl 中的指标（vol IC=0.99, close IC=0.089）是虚假的
- 真实 auto_regressive IC：open=0.64, close/vol/amt≈0

---

## 3. hidden[:, -1, :] 直接预测无效

**问题**：hidden[:, -1, :] 是处理完最后一个**输入 token** 后的状态，不编码预测目标信息

**失败尝试**：
| 方法 | 描述 | 结果 |
|------|------|------|
| direction_head | hidden[:, -1, :] → Linear → BCE 预测涨跌 | 无效，已移除 (commit 9c7d27b) |
| decoder/MLP head | hidden[:, -1, :] → MLP → 6维收益率 | 无效，绕过 token 生成 |
| probe_diagnosis | hidden[:, -1, :] → Linear Probe 检验编码质量 | 无效，探测的是输入状态 |

**正确流程**：
```
输入 tokens → forward → hidden[:, -1, :] → 预测下一个 token →
生成的 token → 自回归继续 → tokenizer.decode(tokens) → 连续值
```

**结论**：任何绕过 token 生成、直接从 hidden state 预测的方法都是错误的。

---

## 4. 梯度绑架

**现象**：CE loss 中 vol/amt 拿走几乎所有学习能力

**表现**：
- vol/amt IC → 0.99+（几乎完美）
- OHLC IC → ≈0（接近随机）

**尝试**：
- 调高 LR (0.01) → 反而加剧 vol/amt 学习
- close_loss 辅助损失 → 梯度冲突，效果更差

**现状**：梯度绑架是核心未解决问题

---

## 5. Loss 函数演进

| 版本 | Loss 设计 | 结果 |
|------|----------|------|
| 原始 | 全位置均匀 CE | context 占 97.6% 权重，prediction 不学习 |
| V5 | direction_loss (BCE) + CE | direction_loss 无效 |
| V6 | prediction-only CE + horizon 衰减 + close_loss | close_loss 梯度冲突 |
| V6b | prediction-only CE + horizon 衰减（无 close_loss） | IC 更稳定，但梯度绑架依然存在 |

**结论**：close_loss 目前不起正面作用，纯 CE + horizon 衰减是当前最佳实践。

---

## 6. Price IC vs Return IC

**定义**：
- Price IC：corr(pred_value, actual_value)
- Return IC：corr(pred_return, actual_return)

**陷阱**：
- 当 pred ≈ baseline, actual ≈ baseline 时，Price IC 会很高（两者都与 baseline 强相关）
- Return IC 才是真正的预测能力衡量

**实际计算**：
```python
pred_return = (pred_value - baseline) / baseline
actual_return = (actual_value - baseline) / baseline
IC = corr(pred_return, actual_return)
```

---

## 7. IC 计算必须 decode + denormalize

**错误做法**：直接用 token indices 计算 IC

**正确做法**：
```python
token = argmax(logits)
z = tokenizer.decode([s1, s2], half=True)  # decode 到 normalized 值
value = z * std + mean  # denormalize 到原始值
return = (value - baseline) / baseline
IC = corr(pred_return, actual_return)
```

---

## 8. 梯度绑架根因诊断（2026-06-09）

**实验**：Token敏感性分析

**方法**：对单个特征扰动，观察 token 变化程度（Hamming distance）

**结果**：
| Feature | Shuffle Hamming | Noise Hamming |
|---------|-----------------|---------------|
| close | 0.92 | 0.32 |
| vol | 0.74 | 0.19 |
| amt | 0.73 | 0.17 |

**结论**：
- Tokenizer 对 close 最敏感（Hamming 最高）
- Tokenizer 已正确编码价格信息
- **问题不在 Tokenizer，在 Predictor CE 目标**

**根因**：CE loss 天然倾向优化更容易预测的特征（vol/amt），而非价格方向

---

## 文档更新记录

- 2026-06-09：初始版本，记录数据泄露、GT leakage、hidden state 预测错误等
- 2026-06-09：Token敏感性实验完成，确认问题在 Predictor（CE目标偏向量能），不在 Tokenizer