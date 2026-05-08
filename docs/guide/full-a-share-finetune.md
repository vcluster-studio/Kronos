# 全 A 股微调训练方案

本文档面向拥有全 A 股数据的用户，提供从基线验证到最终微调的完整训练方案。

---

## 一、方案总览

```
阶段 0: 环境准备                ── 0.5~1 天
阶段 1: 跑通基线管道（CSI300）   ── 1~2 天    ← 路径 A
阶段 2: 全 A 股数据准备          ── 2~3 天    ← 路径 B（与阶段 1 并行）
阶段 3: 全 A 股微调训练          ── 2~4 天
阶段 4: 回测对比评估             ── 0.5~1 天
```

**核心原则：先拿基线数字，再用最优数据微调，对比回测结果说话。**

阶段 1 和阶段 2 可并行推进——阶段 1 用 Qlib 默认 CSI300 数据验证管道，阶段 2 同时准备全 A 股数据。管道跑通后，替换数据即可进入阶段 3。

---

## 二、为什么需要 A 股微调

Kronos 预训练模型覆盖 45+ 全球交易所，是通用模型。A 股市场有以下独特性需要微调适配：

| A 股特征 | 与全球市场的差异 | 微调改善 |
|---------|----------------|---------|
| 涨跌停板 | ±10%（主板）/ ±20%（创业板/科创板）| 分词器码本覆盖涨跌停模式 |
| T+1 交易 | 无法日内回转，价格路径与 T+0 不同 | 预测器学习 T+1 路径规律 |
| 成交量分布 | 散户主导，量价关系与机构市场不同 | 分词器编码更准确的量价特征 |
| 政策驱动 | 政策行情频发 | 预测器适配 A 股特有的脉冲模式 |

---

## 三、数据规范

### 3.1 时间切分

| 数据集 | 时间范围 | 用途 | 说明 |
|--------|---------|------|------|
| 训练集 | 2018-01-02 ~ 2024-12-31 | 模型训练 | 7 年，覆盖两种熊市 + 两轮牛市 |
| 验证集 | 2025-01-02 ~ 2025-06-30 | 早停/模型选择 | 6 个月 |
| 测试集 | 2025-07-01 ~ 2026-04-30 | 最终评估 | 10 个月 out-of-sample |

验证集和训练集尾部会有 `lookback_window`（90 天）的数据重叠，这是滑窗机制的正常行为，评估时只看各自时间范围内的预测结果。

### 3.2 为什么是 2018 而不是更早

A 股市场在 2018 年前后发生了结构性变化：

| 时间节点 | 变化 | 对 K 线模式的影响 |
|---------|------|-----------------|
| 2017 | 北向资金起步 | 定价权尚未转移，量价模式与现在差异大 |
| 2018 | 贸易战 + 去杠杆 | **唯一的地缘冲突型熊市**，有不可替代的信息价值 |
| 2019 | 科创板开板 | ±20% 涨跌停首次出现 |
| 2020 | 创业板注册制 | 创业板从 ±10% 变为 ±20% |
| 2021 | 量化私募爆发 | 日内微观结构剧变 |
| 2023 | 全面注册制 | 壳价值归零，小盘股定价逻辑重构 |

**2017 年及更早的数据，市场微观结构与当前差异过大，可能引入负向干扰。2018 年包含唯一的地缘冲突型熊市样本，不应排除。**

### 3.3 训练集覆盖的市场状态

| 市场状态 | 对应年份 | 微调价值 |
|---------|---------|---------|
| 地缘冲突型熊市 | 2018 | 唯一样本，不可替代 |
| 流动性收紧型熊市 | 2022 | 与 2018 模式互补 |
| 核心资产牛市 | 2019 | 大盘蓝筹定价模式 |
| 疫情 V 型反转 | 2020 | 极端事件后的恢复模式 |
| 赛道结构性行情 | 2021 | 行业轮动模式 |
| 概念驱动行情 | 2023 | 主题投资模式 |
| 政策驱动反转 | 2024 Q4 | 政策市特征 |
| 小微盘流动性危机 | 2024 Q1 | 极端尾部风险 |

### 3.4 股票池过滤规则

| 过滤条件 | 阈值 | 理由 |
|---------|------|------|
| ST / *ST | 排除 | K 线模式异常，涨跌幅 ±5% |
| 上市不足 1 年 | 排除 | 次新股波动无规律，缺乏 lookback 数据 |
| 日均成交额 | < 1000 万 | 流动性不足，滑点大，回测不真实 |
| 日线数据缺失率 | > 5% | 数据质量差 |
| 停牌超过 20 个交易日 | 排除 | 复牌跳空无法预测 |

预计过滤后剩余约 3500~4000 只股票。

### 3.5 涨跌停处理

```
分词器微调阶段：
  - 保留涨跌停日数据（码本需要覆盖这些极端模式）

预测器微调阶段：
  - 排除"次日仍涨跌停"的样本（连续涨跌停不可预测）
  - 保留"首日涨跌停"的样本（模型需要识别涨跌停信号）
```

### 3.6 特征列

与 `config.py` 一致，6 列特征 + 5 列时间特征：

```python
feature_list = ['open', 'high', 'low', 'close', 'vol', 'amt']
time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']
# 日频数据中 minute=0, hour=15（收盘时刻）
```

---

## 四、阶段 0：环境准备

### 4.1 硬件需求

| 组件 | 最低要求 | 推荐 |
|------|---------|------|
| GPU | 1× RTX 3090 (24GB) | 2× A100 (40GB+) |
| 内存 | 64 GB | 128 GB |
| 存储 | 100 GB SSD | 500 GB SSD |

单卡可运行全部流程，双卡通过 DDP 训练速度约翻倍。

### 4.2 软件环境

```bash
# 创建环境
conda create -n kronos python=3.10 -y
conda activate kronos

# 安装 PyTorch（根据 CUDA 版本选择）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装项目依赖
pip install pyqlib einops matplotlib tqdm comet-ml huggingface_hub

# 项目安装
cd Kronos
pip install -e .
```

### 4.3 下载预训练模型

```python
from model import Kronos, KronosTokenizer

KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").save_pretrained("./pretrained/Kronos-Tokenizer-base")
Kronos.from_pretrained("NeoQuasar/Kronos-small").save_pretrained("./pretrained/Kronos-small")
# 如需使用 base 模型
Kronos.from_pretrained("NeoQuasar/Kronos-base").save_pretrained("./pretrained/Kronos-base")
```

---

## 五、阶段 1：跑通基线管道

**目标：用 Qlib 默认 CSI300 数据验证全流程可运行，记录基线指标。**

### 5.1 准备 Qlib 默认数据

```bash
python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
```

### 5.2 修改 config.py

仅需修改预训练模型路径：

```python
self.pretrained_tokenizer_path = "./pretrained/Kronos-Tokenizer-base"
self.pretrained_predictor_path = "./pretrained/Kronos-small"
```

其余参数使用默认值（CSI300、2011–2025 时间范围）。

### 5.3 执行训练与回测

```bash
cd finetune

# 数据预处理
python qlib_data_preprocess.py

# 微调分词器
torchrun --standalone --nproc_per_node=1 train_tokenizer.py

# 微调预测器
torchrun --standalone --nproc_per_node=1 train_predictor.py

# 回测
python qlib_test.py --device cuda:0
```

### 5.4 记录基线指标

完成回测后，记录以下指标作为后续对比的基线：

| 指标 | 含义 | 基线值 |
|------|------|--------|
| val_loss (tokenizer) | 分词器重建误差 | |
| val_loss (predictor) | 预测器交叉熵 | |
| IC | 预测信号与真实收益的 Pearson 相关 | |
| Rank IC | Spearman 秩相关 | |
| 年化超额收益（含成本） | 策略收益 - 基准 - 交易成本 | |
| 最大回撤 | 策略净值峰谷差 | |
| 夏普比率 | 超额收益 / 波动率 | |

---

## 六、阶段 2：全 A 股数据准备

本阶段与阶段 1 并行推进。

### 6.1 数据源接入

两种方式任选：

| 方式 | 数据源 | 适用场景 |
|------|--------|---------|
| Qlib 格式 | 自有数据转为 Qlib 格式，使用 `finetune/` 管道 | 推荐方式，改动最小 |
| CSV 格式 | 直接使用 CSV 文件，使用 `finetune_csv/` 管道 | 数据量较小或快速验证 |

**推荐使用 Qlib 格式**，与现有训练管道无缝衔接。

### 6.2 自有数据转 Qlib 格式

参考 [Qlib 官方数据转换文档](https://qlib.readthedocs.io/en/latest/component/data.html#converting-csv-format-into-qlib-format)，需要准备：

1. **日历文件** (`calendars/day.txt`) — 交易日列表
2. **特征文件** (`instruments/`) — 每只股票的 OHLCV 数据
3. **股票列表** (`instruments/all.txt`) — 全 A 股股票池定义

### 6.3 数据质量检查

完成数据转换后，建议执行以下检查：

- 股票数量和时间范围是否符合预期
- 各特征列是否存在缺失值或异常值
- 涨跌停日标记是否正确
- 数据缺失率是否在可接受范围内（<5%）

### 6.4 修改 config.py（全 A 股配置）

```python
class Config:
    def __init__(self):
        # === 数据路径 ===
        self.qlib_data_path = "~/.qlib/qlib_data/cn_data"  # 或自定义全 A 股数据路径
        self.instrument = 'all'  # 全 A 股股票池

        # === 时间范围 ===
        self.dataset_begin_time = "2017-04-01"   # 提前 lookback 天数以支持 2018 年初的滑窗
        self.dataset_end_time = "2026-04-30"

        self.train_time_range = ["2018-01-02", "2024-12-31"]
        self.val_time_range = ["2025-01-02", "2025-06-30"]
        self.test_time_range = ["2025-07-01", "2026-04-30"]
        self.backtest_time_range = ["2025-07-01", "2026-04-30"]

        # === 模型路径 ===
        self.pretrained_tokenizer_path = "./pretrained/Kronos-Tokenizer-base"
        self.pretrained_predictor_path = "./pretrained/Kronos-small"

        # === 训练参数（与基线一致） ===
        self.epochs = 30
        self.batch_size = 50
        self.tokenizer_learning_rate = 2e-4
        self.predictor_learning_rate = 4e-5
```

---

## 七、阶段 3：全 A 股微调训练

### 7.1 分词器微调

```bash
cd finetune

# 数据预处理（使用全 A 股配置）
python qlib_data_preprocess.py

# 微调分词器（双卡）
torchrun --standalone --nproc_per_node=2 train_tokenizer.py
# 或单卡
torchrun --standalone --nproc_per_node=1 train_tokenizer.py
```

**监控要点：**

| 指标 | 正常表现 | 异常信号 |
|------|---------|---------|
| `bsq_loss` | 逐渐下降或稳定 | 持续上升 → 码本坍缩 |
| `recon_loss` | 持续下降 | 停止下降 → 学习率过高或数据问题 |
| val_loss | 逐渐下降 | 上升 → 过拟合，需早停 |

### 7.2 预测器微调（分层学习率）

当前代码使用单一学习率，建议改为分层学习率以保护预训练知识：

```python
# 修改 train_predictor.py 中的 optimizer 定义

n_layers = len(model.module.transformer)
param_groups = [
    # 底层 Transformer（保护预训练通用特征）
    {'params': [p for name, p in model.named_parameters()
                if 'transformer' in name
                and any(f'transformer.{i}.' in name for i in range(n_layers // 2))],
     'lr': 1e-5},
    # 顶层 Transformer
    {'params': [p for name, p in model.named_parameters()
                if 'transformer' in name
                and any(f'transformer.{i}.' in name for i in range(n_layers // 2, n_layers))],
     'lr': 2e-5},
    # head + embedding + 其他
    {'params': [p for name, p in model.named_parameters()
                if 'transformer' not in name],
     'lr': 4e-5},
]
optimizer = torch.optim.AdamW(
    param_groups,
    betas=(config['adam_beta1'], config['adam_beta2']),
    weight_decay=config['adam_weight_decay']
)
```

**分层学习率的逻辑：**

- **底层**（1e-5）：学习 K 线通用规律（趋势、均值回归等），几乎不需要改
- **中层**（2e-5）：适度适配 A 股特征
- **顶层 + head**（4e-5）：重点适配 A 股预测任务

### 7.3 早停机制

当前代码只保存 best_model，建议添加早停防止过拟合：

```python
# 在 train_predictor.py 的 train_model() 中添加

patience = 5  # 连续 5 个 epoch val_loss 不降则停止
patience_counter = 0

# 在每个 epoch 结束后的 checkpointing 逻辑中：
if avg_val_loss < best_val_loss:
    best_val_loss = avg_val_loss
    patience_counter = 0
    save_path = f"{save_dir}/checkpoints/best_model"
    model.module.save_pretrained(save_path)
    print(f"Best model saved (Val Loss: {best_val_loss:.4f})")
else:
    patience_counter += 1
    if rank == 0:
        print(f"No improvement for {patience_counter} epoch(s) (patience={patience})")
    if patience_counter >= patience:
        if rank == 0:
            print(f"Early stopping at epoch {epoch_idx + 1}")
        break
```

### 7.4 训练命令汇总

```bash
cd finetune

# Step 1: 数据预处理
python qlib_data_preprocess.py

# Step 2: 微调分词器
torchrun --standalone --nproc_per_node=2 train_tokenizer.py

# Step 3: 微调预测器（自动加载 Step 2 产出的分词器）
torchrun --standalone --nproc_per_node=2 train_predictor.py

# Step 4: 回测评估
python qlib_test.py --device cuda:0
```

---

## 八、阶段 4：回测对比评估

### 8.1 对比实验设计

| 实验名称 | 训练数据 | 股票池 | 模型 |
|---------|---------|--------|------|
| baseline_pretrained | 无微调 | CSI300 | Kronos-small 通用 |
| baseline_csi300 | 2011–2025 | CSI300 | Kronos-small 微调 |
| **full_a_share** | **2018–2024** | **全 A 股** | **Kronos-small 微调** |
| full_a_share_base | 2018–2024 | 全 A 股 | Kronos-base 微调 |

### 8.2 关键评估指标

| 指标 | 计算方式 | 判断标准 |
|------|---------|---------|
| IC | 预测信号与真实收益的 Pearson 相关 | >0.03 有实用价值 |
| Rank IC | Spearman 秩相关 | 比 IC 更稳健 |
| ICIR | IC 均值 / IC 标准差 | >0.5 为优秀 |
| 年化超额收益 | 策略收益 - 基准收益 - 交易成本 | 正值即有效 |
| 最大回撤 | 策略净值峰谷差 | <15% 可接受 |
| 夏普比率 | 超额收益 / 波动率 | >1.0 可接受 |
| 换手率 | 日均换仓比例 | 过高说明信号不稳定 |

### 8.3 结果判断标准

| 对比项 | 判断规则 | 结论 |
|--------|---------|------|
| full_a_share vs baseline_pretrained | IC 提升 > 10% | 微调本身有收益 |
| full_a_share vs baseline_csi300 | IC 提升 > 20% | 全 A 股数据有实质价值 |
| full_a_share_base vs full_a_share | IC 提升 < 10% | small 够用，不值得用 base |

---

## 九、风险与预案

| 风险 | 表现 | 预案 |
|------|------|------|
| 码本坍缩 | `bsq_loss` 持续上升，recon_loss 停止下降 | 降低 tokenizer_lr 到 1e-4；增大 gamma0 正则化 |
| 预测器过拟合 | train_loss 下降但 val_loss 上升 | 启用早停；降低 predictor_lr |
| 显存不足 | OOM 错误 | 减小 batch_size 到 16；增大 accumulation_steps 到 4 |
| 训练时间过长 | 预处理或训练耗时过久 | 缩短训练集至 2020–2024；减少 n_train_iter |
| 微调效果退化 | IC 低于预训练模型 | 降低学习率；冻结底层 Transformer |
| 小盘股主导采样 | 码本偏向高波动模式 | 实施分层采样：大盘/中盘/小盘按比例均衡 |

---

## 十、时间估算

| 阶段 | 单卡 RTX 3090 | 双卡 A100 |
|------|--------------|-----------|
| 阶段 0：环境准备 | 0.5 天 | 0.5 天 |
| 阶段 1：基线管道 | 1~2 天 | 0.5~1 天 |
| 阶段 2：数据准备 | 2~3 天 | 2~3 天 |
| 阶段 3：全 A 股微调 | 3~4 天 | 1.5~2 天 |
| 阶段 4：回测对比 | 0.5 天 | 0.5 天 |
| **总计** | **7~10 天** | **5~7 天** |

阶段 1 和阶段 2 并行后，实际耗时约为 5~7 天（单卡）或 4~5 天（双卡）。
