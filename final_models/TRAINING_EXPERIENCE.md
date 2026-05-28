# Kronos 微调经验总结

## 重要发现：Tokenizer 匹配问题（2026-05-11）

### 问题背景

之前的测试使用了错误的 tokenizer 配置：
- **mini 模型**应该使用 `Kronos-Tokenizer-2k`（group_size=5）
- **small/base 模型**应该使用 `Kronos-Tokenizer-base`（group_size=4）

之前用 base tokenizer 测试 mini，导致结果偏差。

### Tokenizer 区别

| Tokenizer | group_size | 压缩比例 | 适用模型 |
|-----------|------------|----------|----------|
| Kronos-Tokenizer-2k | 5 | 5:1 | mini |
| Kronos-Tokenizer-base | 4 | 4:1 | small, base |

**压缩效率**：2k tokenizer 每 token 包含更多信息（5个时间步 vs 4个），序列更短。

---

## 一、正确配置下的模型性能对比（Seed=42）

### 1.1 原始预训练模型效果（正确匹配 tokenizer）

| 数据集 | mini (4.1M, 2k tok) | small (24.7M, base tok) | base (102.3M, base tok) |
|---------|---------------------|-------------------------|-------------------------|
| Full | **0.7207** | 0.7057 | 0.6993 |
| Mid | **0.7796** | 0.7737 | 0.7451 |
| Small | 0.6872 | **0.6910** | 0.6541 |
| Mid+Small | **0.7897** | 0.7599 | 0.7326 |

### 1.2 惊人发现

1. **mini (4.1M) 效果最好！** — 参数量最小，但 IC 最高
2. **参数量与效果反向** — mini > small > base
3. **mini 比参数量大 25 倍的 base 还好** — 高压缩率策略胜出

### 1.3 核心结论

**高压缩率（group_size=5）比大参数量更重要**

mini 用更聪明的方式（更高压缩）弥补了参数量劣势：
- 90 步输入 → 2k tokenizer → 18 tokens
- 90 步输入 → base tokenizer → 22-23 tokens
- 小模型处理紧凑序列时，注意力更集中

---

## 二、Tokenizer 匹配验证实验

### 2.1 测试配置

| 组合 | Tokenizer | Predictor | 说明 |
|------|-----------|-----------|------|
| mini-2k-orig | 2k (原始) | mini (原始) | 正确匹配 |
| mini-2k-trained | 2k (原始) | mini (训练后) | predictor 不兼容 |
| mini-base-orig | base (原始) | mini (原始) | 错误匹配 |
| mini-base-trained | base (训练后) | mini (训练后) | 训练组合 |
| small-base-orig | base (原始) | small (原始) | 正确匹配 |
| small-midtok-orig | base (训练后) | small (原始) | 微调 tokenizer |
| base-base-orig | base (原始) | base (原始) | 正确匹配 |
| base-midtok-orig | base (训练后) | base (原始) | 微调 tokenizer |

### 2.2 完整测试结果

| Dataset | mini-2k-orig | mini-2k-trained | mini-base-orig | mini-base-trained | small-base-orig | small-midtok | base-base-orig | base-midtok |
|---------|--------------|-----------------|----------------|-------------------|-----------------|--------------|----------------|-------------|
| Full | **0.7207** | 0.0688 | 0.1487 | 0.4147 | 0.7057 | 0.7239 | 0.6993 | 0.6949 |
| Mid | **0.7796** | 0.1376 | 0.1717 | 0.4741 | 0.7737 | 0.7799 | 0.7451 | 0.7677 |
| Small | 0.6872 | 0.0916 | 0.0535 | 0.4349 | **0.6910** | 0.6565 | 0.6541 | 0.6719 |
| Mid+Small | **0.7897** | 0.1274 | 0.1178 | 0.4508 | 0.7599 | 0.7565 | 0.7326 | 0.7482 |

### 2.3 关键发现

1. **mini + 2k tokenizer 效果最好**（IC≈0.72-0.79）
2. **错误匹配导致效果下降 70-90%**
   - mini (base tok + orig pred): IC≈0.15（下降 80%）
   - mini (2k tok + trained pred): IC≈0.07（下降 90%）
3. **训练后的 predictor 与错误 tokenizer 不兼容**
4. **微调 tokenizer 对 small/base 收益有限**（±3%）

---

## 三、之前的训练结论需要推翻

### 3.1 错误的训练方向

之前用 base tokenizer 训练 mini predictor：
- 训练时：base tokenizer + mini predictor
- 测试时：用 2k tokenizer + trained predictor → 效果崩溃（IC≈0.07）

**结论：之前的训练完全错误，tokenizer/predictor 不匹配导致训练无效**

### 3.2 正确的训练策略

如果要微调 mini：
1. 必须基于 Kronos-Tokenizer-2k（不是 base）
2. 用 2k tokenizer 训练 predictor
3. 测试时保持 tokenizer 一致

如果要微调 small/base：
1. 使用 Kronos-Tokenizer-base
2. 已训练的 mid tokenizer 可直接使用（基于 base）

---

## 四、最终建议

### 4.1 直接使用原始模型

**原始预训练效果已经很好（IC≈0.70），无需微调**

| 场景 | 推荐模型 | 原因 |
|------|----------|------|
| 快速原型/资源受限 | mini (4.1M) | 效果最好，参数最小 |
| 生产环境 | mini 或 small | mini 效果略好，small 更稳定 |
| 研究用途 | mini | 便于实验，速度快 |

### 4.2 微调的适用场景

**只有在以下情况才考虑微调：**
1. 目标数据与预训练数据差异大（新市场、新数据类型）
2. 有大量高质量特定领域数据
3. 需要针对特定股票类型优化

### 4.3 模型选择结论

```
效果排序：mini > small > base
参数排序：mini (4.1M) < small (24.7M) < base (102.3M)
性价比：mini 最高
```

---

## 二、微调步骤

### 2.1 数据准备

**数据路径**：
```
finetune/data/
├── processed_datasets/          # 全A股
├── processed_datasets_mid/      # 中盘股
├── processed_datasets_small/    # 小盘股
└── processed_datasets_mid_small/ # 中小盘混合
```

**数据格式**：每个数据集包含
- `train_data.pkl` - 训练数据
- `val_data.pkl` - 验证数据
- `test_data.pkl` - 测试数据
- `stock_categories.pkl` - 股票分类信息

**特征列**：`open`, `high`, `low`, `close`, `vol`, `amt`

### 2.2 Tokenizer 微调

**脚本**：`finetune/train_tokenizer_mini.py --config mid`

**配置文件**：`finetune/config_mid.py`

**运行命令**：
```bash
# 后台运行，日志输出到文件
python -u finetune/train_tokenizer_mini.py --config mid > logs/tokenizer_mid.log 2>&1 &

# 查看进度
tail -f logs/tokenizer_mid.log
```

**训练参数**：
| 参数 | 值 | 说明 |
|------|------|------|
| epochs | 30 | 训练轮数 |
| batch_size | 16 | 批次大小 |
| learning_rate | 0.0002 | 学习率 |
| lr_scheduler | OneCycleLR | 学习率调度 |
| patience | 10 | 早停耐心值 |
| n_train_iter | 2000 × batch_size | 每轮训练样本数 |

**输出路径**：`outputs/models/mid_tokenizer_v1/checkpoints/best_model`

### 2.3 Predictor 微调

**脚本**：`finetune/train_predictor_mini.py --config mid`

**配置文件**：`finetune/config_mid.py`

**运行命令**：
```bash
# 后台运行
python -u finetune/train_predictor_mini.py --config mid > logs/predictor_mid.log 2>&1 &

# 查看进度
tail -f logs/predictor_mid.log
```

**训练参数**：
| 参数 | 值 | 说明 |
|------|------|------|
| epochs | 30（早停） | 最大训练轮数 |
| batch_size | 16 | 批次大小 |
| learning_rate | 0.01 | 初始学习率 |
| lr_scheduler | vl_adaptive | VL-IC自适应调度 |
| ic_improve_threshold | 0.05 | IC改善阈值(5%) |
| patience | 10 | 早停耐心值 |
| grace_period | 5 | 前5轮不早停 |

### 2.4 VL-IC Adaptive 学习率策略

**核心逻辑**（代码位置：`train_predictor_mini.py`）：
1. 每轮训练后先计算 IC，再调整学习率
2. IC 改善 > 5% → 保持 LR（不衰减）
3. IC + Val Loss 同时改善 → 提升 LR（继续探索）
4. 都无改善 → 衰减 LR（精细收敛）

---

## 三、最终模型

**保存路径**：
```
final_models/
├── Kronos-Mid-Tokenizer-v1/     # 中盘 tokenizer
├── Kronos-Mid-Predictor-v1/     # 中盘 predictor
├── TRAINING_EXPERIENCE.md       # 本文档
└── training_scripts/            # 训练脚本备份
```

**使用方法**：
```python
from model import KronosTokenizer, Kronos

tokenizer = KronosTokenizer.from_pretrained("final_models/Kronos-Mid-Tokenizer-v1")
predictor = Kronos.from_pretrained("final_models/Kronos-Mid-Predictor-v1")
```

---

## 四、训练技巧

### 4.1 后台运行与日志

```bash
# 方法：使用 & 和日志重定向
python -u train_script.py > output.log 2>&1 &

# 实时查看日志
tail -f output.log

# 查看最后100行
tail -100 output.log

# 搜索关键信息
grep "IC:" output.log
grep "Epoch" output.log
```

### 4.2 监控训练状态

```bash
# GPU使用
nvidia-smi -l 1

# 关键日志标记
[VAL LOSS SAVED]   # Val Loss改善，保存模型
[IC SAVED]         # IC改善，保存模型
[EARLY STOP]       # 早停触发
[PATIENCE] x/10    # 当前耐心计数
```

### 4.3 常见问题处理

| 问题 | 解决方法 |
|------|----------|
| CUDA内存不足 | 减小 batch_size |
| 训练不稳定 | 降低 learning_rate |
| IC 波动大 | 增加测试样本数（100→300） |
| 早停过早 | 增加 grace_period |

---

## 五、脚本目录结构

```
finetune/
├── train_tokenizer_mini.py   # Tokenizer训练（主脚本）
├── train_predictor_mini.py   # Predictor训练（主脚本）
├── config_mid.py             # 中盘配置
├── dataset.py                # 数据集定义
├── unified_test.py           # 统一测试脚本（推荐使用）
│
├── config.py                 # 默认配置
├── config_mini.py            # 全量配置
├── config_small.py           # 小盘配置
├── config_mid_small.py       # 中小盘配置
│
└── [其他脚本]                 # 历史/临时脚本，可清理
```

**推荐使用**：
- 训练：`train_tokenizer_mini.py`、`train_predictor_mini.py`
- 测试：`unified_test.py`（固定随机种子，结果可重复）

---

## 六、测试信息

- **测试时间**：2026-05-11
- **测试脚本**：`finetune/unified_test.py`
- **随机种子**：42（固定）
- **测试样本**：300个股票/数据集
- **IC计算**：Spearman 相关系数