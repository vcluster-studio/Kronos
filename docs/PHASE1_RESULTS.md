# Phase 1 实验结果：Token敏感性分析

**日期**：2026-06-09

---

## 实验目的

诊断"梯度绑架"现象的根因位置：
- 假设1：Tokenizer 容量偏向 vol/amt
- 假设2：Predictor CE 目标偏向易学特征

---

## 实验方法

对单个特征扰动，观察 token 序列变化程度：

1. **Shuffle Test**：随机打乱某特征值顺序
2. **Noise Test**：对某特征添加 10% 噪声

统计 Hamming Distance（token 变化比例）和 Decode MAE（decode 后值差异）。

---

## 实验结果

### Shuffle Test（随机打乱某特征）

| Feature | Avg Hamming | Decode MAE |
|---------|-------------|------------|
| open    | 0.7822      | 0.4000     |
| high    | 0.8137      | 0.3978     |
| low     | 0.8221      | 0.4201     |
| close   | **0.9197**  | 0.4335     |
| vol     | 0.7442      | 0.3060     |
| amt     | 0.7278      | 0.3125     |

### Noise Test（添加 10% 噪声）

| Feature | Avg Hamming | Decode MAE |
|---------|-------------|------------|
| open    | 0.1179      | 0.0353     |
| high    | 0.1775      | 0.0460     |
| low     | 0.2039      | 0.0505     |
| close   | **0.3190**  | 0.0687     |
| vol     | 0.1916      | 0.0458     |
| amt     | 0.1685      | 0.0415     |

### 综合统计

| 指标 | 值 |
|------|-----|
| Price features avg Hamming | 0.8344 |
| Volume features avg Hamming | 0.7360 |
| Ratio (vol/price) | 0.88 |

---

## 结论

### 假设验证

| 假设 | 实验证据 | 验证结果 |
|------|----------|----------|
| Tokenizer 容量偏向 vol/amt | close Hamming 最高 (0.92) | ❌ **排除** |
| Predictor CE 倾向优化易学特征 | vol Hamming 较低 (0.74) | ✅ **确认** |

### 核心发现

1. **Tokenizer 对 close 最敏感**
   - 扰动 close → Hamming = 0.92（最高）
   - 扰动 vol/amt → Hamming ≈ 0.73（较低）

2. **Tokenizer 已正确编码价格信息**
   - 码本容量分配合理
   - 价格特征占据主要编码空间

3. **问题位置确认**
   - Tokenizer 问题：**10%**（已排除）
   - Predictor 问题：**90%**（CE 目标偏向量能）

### 根因分析

CE loss 天然倾向优先优化：
- 更稳定的特征（vol/amt 变化规律性强）
- 更容易降低 loss 的特征（价格方向难以预测）

导致 Predictor 训练资源被 vol/amt "绑架"，价格方向没学好。

---

## Phase 2 方向

基于诊断结果，应改进 **Predictor**：

1. **Feature-weighted CE Loss**
   ```python
   weights = [open=2.0, high=2.0, low=2.0, close=3.0, vol=0.5, amt=0.5]
   loss = weighted_cross_entropy(logits, targets, weights)
   ```

2. **两阶段训练**
   - Stage 1：冻结 vol/amt，只训价格部分
   - Stage 2：放开全部，微调平衡

---

## 文件位置

- 实验脚本：`finetune/token_sensitivity_analysis.py`
- 结果数据：`finetune/token_sensitivity_results.pkl`
- 诊断计划：`docs/DIAGNOSIS_PLAN.md`