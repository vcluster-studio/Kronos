# Kronos MA60 微调实验经验总结

> 更新时间：2026-05-28
> 本文档记录 MA60 归一化微调实验的关键发现、踩过的坑和教训。

---

## 一、核心发现：归一化策略 > 模型参数量

| 模型 | 参数量 | 归一化 | 全市场 IC |
|------|--------|--------|-----------|
| Kronos-mini | 4.1M | full_window | 0.0163 |
| Kronos-small | 24.7M | full_window | -0.1024 |
| Kronos-base | 102.3M | full_window | -0.1820 |
| **MA60-mini** | **4.1M** | **sliding MA60** | **0.1719** |
| MA60-small | 24.7M | sliding MA60 | 0.1646 |

**结论**：正确的归一化远比增加参数量重要。MA60-mini (4.1M) 碾压 pretrained-base (102.3M)。

---

## 二、模型匹配关系（绝对不能搞错）

| 模型 | Tokenizer | group_size | max_context |
|------|-----------|------------|-------------|
| Kronos-mini | Kronos-Tokenizer-2k | 5 | 2048 |
| Kronos-small | Kronos-Tokenizer-base | 4 | 512 |
| Kronos-base | Kronos-Tokenizer-base | 4 | 512 |

**教训**：Tokenizer 和 Predictor 必须 group_size 一致，不匹配训练无效。之前用 base tokenizer 训练 mini 导致结果崩溃。

---

## 三、训练流程中的关键教训

### 3.1 IC 评估必须用 val_data，不用 test_data

**错误做法**：训练过程中每轮用 test_data 算 IC，基于此保存 best_ic_model、调整 VL-Adaptive LR。
**问题**：test_data 应只用于最终评估，参与训练决策会导致信息泄露。
**正确做法**：IC 评估用 val_data，test_data 留给训练结束后的全市场测试。

### 3.2 IC 样本数 200 太少，必须 500+

**错误做法**：`list(test_data.keys())[:200]` 固定取字典序前 200 只股票。
**问题**：
1. 200 个点算相关系数，一个极端值就能大幅拉偏结果
2. 字典序前 200 只偏大盘股（600000.SH 起），样本有偏差
3. 固定不轮换，VL-Adaptive 和 best_ic 保存都基于噪声

**正确做法**：每轮随机采样 500 只股票（`rng.choice(symbols, 500)`），减少统计噪声。

### 3.3 IC 决策用滑动均值，不用单轮值

**问题**：单轮 IC 波动极大（-0.05~0.22），基于单轮 IC 保存 best_ic_model 可能保存的是噪声峰值。
**正确做法**：用最近 3 轮 IC 均值做决策（`ic_smoothed = mean(ic[-3:])`）。

### 3.4 学习率策略

| 场景 | 推荐 LR | 原因 |
|------|---------|------|
| 从 pretrained 开始训练 | 0.02 | 归一化方式变了，需要充分探索 |
| 从 checkpoint 恢复 | 0.003~0.008 | 已有基础，小步精调 |
| 小模型 (4.1M) | 0.02 | 容量小，需要大步探索 |
| 大模型 (24.7M+) | 0.003~0.008 | 容量大，大步过冲后 IC 崩塌 |

**教训**：V1 small 用 LR=0.02，Ep3 IC=0.18 后崩塌；V2 用 0.008 仍不稳定；V3 用 0.003 波动变小但 best IC 不高。大模型训练需要更保守的 LR。

### 3.5 Quick IC vs 全市场 IC 不能直接比较

| 指标 | 样本数 | 用途 | 典型值 |
|------|--------|------|--------|
| Quick IC | 200~500 | 训练中快速反馈 | 波动大 |
| 全市场 IC | 2555 | 最终评估 | 更稳定 |

**教训**：V2 训练 Quick IC=0.2065（200 样本），全市场 IC=0.1646（2555 样本），差距显著。Quick IC 只能看趋势，不能直接对比。

---

## 四、有价值的模型文件

### 4.1 最终可用模型（`final_models/`）

| 文件 | 说明 | 全市场 IC |
|------|------|-----------|
| `Kronos-Tokenizer-2k-MA60` | MA60 tokenizer (group_size=5, mini 用) | - |
| `Kronos-mini-MA60` | MA60 predictor (mini, best IC) | 0.1719 |
| `Kronos-Tokenizer-2k` | 原始 2k tokenizer | - |
| `Kronos-mini` | 原始 mini 模型 | 0.0163 |

### 4.2 训练产出（`outputs/models/`）

| 路径 | 说明 | 价值 |
|------|------|------|
| `ma60_tokenizer_v1/` | MA60 tokenizer (mini, gs=5) | **已复制到 final_models** |
| `ma60_tokenizer_base_v1/` | MA60 tokenizer (base, gs=4) | **small/base 训练必须** |
| `ma60_predictor_v1/` | MA60 predictor V1 (mini) | best IC，已复制到 final_models |
| `ma60_predictor_v2/` | MA60 predictor V2 (mini) | 严重过拟合，参考价值低 |
| `ma60_predictor_small_v1/` | MA60 predictor V1 (small) | IC 不稳定 |
| `ma60_predictor_small_v2/` | MA60 predictor V2 (small) | best quick IC=0.21，全市场 0.16 |
| `ma60_predictor_small_v3/` | MA60 predictor V3 (small, LR=0.003) | 未完成，best IC=0.22(200样本) |
| `ma60_predictor_mini_v2/` | 新逻辑重训 mini | **进行中** |
| `full_tokenizer_2k_v1/` | Full A-share tokenizer (2k) | 早期实验 |
| `full_predictor_v8/` | Full A-share predictor | 早期实验，无 MA60 |

### 4.3 可以清理的文件

以下为中间产物，best_ic_model / best_model / latest_model 重复：
- `ma60_predictor_small_v1/`、`v2/`、`v3/` 的 `latest_model` 和 `final_model`
- `ma60_predictor_v2/` 的所有 checkpoint（过拟合严重）
- `full_predictor_v8/`、`full_tokenizer_2k_v1/`（早期 full_window 实验）

---

## 五、推理产出

| 文件 | 记录数 | 说明 |
|------|--------|------|
| `predictions_ma60_v1ic_2025.sql` | 1,144,455 | 2025 年 MA60-mini 推理 |
| `predictions_ma60_v1ic_2026.sql` | 430,694 | 2026 年推理（含增量 0519-0522） |
| `predictions_ma60_v1ic_2026_0519_0524.sql` | - | 增量推理 0519-0524 |

**注意**：推理结果只有 OHLCV 数据，没有涨跌幅字段。如需要可后续补充。

---

## 六、Python 环境

- **环境名**：`criticality` (conda)
- **路径**：`C:/Users/junhu/.conda/envs/criticality/python.exe`
- **注意**：bash 中默认 `python` 指向 WindowsApps 无效路径，必须用完整路径或 `conda run`
- **GPU 推理**：必须加 `HF_HUB_OFFLINE=1` 避免尝试连接 HuggingFace 超时

---

## 七、训练脚本对照表

| 脚本 | 用途 | Tokenizer | 模型 |
|------|------|-----------|------|
| `train_tokenizer.py` | 原始 tokenizer 微调 (2k) | Kronos-Tokenizer-2k | - |
| `train_tokenizer_ma60_base.py` | MA60 tokenizer (gs=4) | Kronos-Tokenizer-base | - |
| `train_predictor.py` | 原始 predictor 微调 | 按 config | 按 config |
| `train_predictor_ma60.py` | MA60 predictor (mini) | ma60_tokenizer_v1 | Kronos-mini |
| `train_predictor_ma60_small.py` | MA60 predictor (small) | ma60_tokenizer_base_v1 | Kronos-small |

所有三个 predictor 训练脚本已统一：
- IC 评估 → val_data
- 随机采样 500 只
- IC 决策用 3 轮滑动均值
