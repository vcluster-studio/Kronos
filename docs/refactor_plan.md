# Kronos 模块重构方案（修订版）

**修订日期**: 2026-06-19  
**修订依据**: `docs/refactor_plan_review.md` 评审意见

---

## 变更日志

| 日期 | 章节 | 变更 | 原因 | 详见 review |
|------|------|------|------|-------------|
| 2026-06-19 | §1.5.2 / §1.5.6 | tokenizer 微调改为步数驱动放回采样（`n_train_iter_multiplier`，默认 32000 步/epoch），删除臆造的 `sample_ratio=0.1`；补微调效果检验（val 重建损失 + 与预训练对比 + 验收标准表） | 原方案杜撰了不存在的采样率与 `KronosTokenizer.train()` API；核实现有 `deprecated/finetune/train_tokenizer.py` 真实做法为步数驱动采样 + 重建损失 early stopping | §21 |
| 2026-06-19 | §1.2 / §6.2 / §8.3 | 三项口径决策确认：① I2 SlidingMANormalizer **保持含当前点**（不加 shift(1)，归一化需含当前点信息）；② I3 eval.py 用 **DDP 框架统一控卡数**（单/多卡同一套代码，`torchrun --nproc_per_node=N`）；③ P4 excess DA 已写入 training_info（确认已修）。I2 口径与旧代码不同 → Phase 4 等价性改为"新口径自洽 + 可换算对照"，不强求与旧数值对齐 | 用户拍板口径；核实 train.py:627/169 excess_da 已记录、normalization.py:112 含当前点、eval.py 待按 DDP 改造 | §22.6 |
| 2026-06-19 | §10 | epoch 输出与 best 跟踪完善（ABCD+X）：① epoch 打印各关键指标并列 `current \| best @ep`，不再 IC 主导；② best 跟踪扩展到 val_loss/ic/da_score/excess_da/combined 五项，各记 `{value, epoch}`；③ 可懂指标三件套（振幅误差率/涨跌停命中率）只在产生 best 时算并记录，平凡 epoch 不算（无额外 forward）；④ checkpoint 仍只 3 个（val_loss/ic/combined），DA/可懂指标 best 仅记录值+epoch 不存独立 ckpt（X）；⑤ `update_training_info` 签名改为 `best_updates` dict，epoch_record 补 da_score/excess_da/可懂指标 | 原 epoch 输出 IC 主导，DA/excess_da 无 best 跟踪，可懂指标完全缺失；用户要求各关键指标补 current+best、best 时算可懂指标 | §10 |

---

## 一、核心设计决策

### 1.1 归一化策略：Runtime 归一化

**选择**: Runtime 归一化（运行时归一化）

**理由**:
- pkl 保存原始 OHLCV 窗口，无需预处理多版本
- Dataset 统一处理归一化，保证训练/评估/回测一致
- 新增 norm_mode 只需扩展 normalizer，无需重新预处理数据

**实现**:
```python
# pkl 保存原始数据
window_data = {
    'values': original_ohlcv,      # 原始值
    'index': timestamps,
    'symbol': symbol,
    'window_start': start,
}

# Dataset __getitem__ 时归一化
class KronosDataset:
    def __getitem__(self, idx):
        x_raw = self.windows[idx]['values'][:self.lookback]
        x_norm = self.normalizer.normalize(x_raw)
        return x_norm
```

### 1.2 norm_mode 命名规范

统一命名，禁止混用：

| norm_mode | 含义 | 归一化方式 |
|-----------|------|-----------|
| `full_window` | 全窗口归一化 | 基于 lookback 窗口的 mean/std |
| `sliding_ma20` | MA20 滑动归一化 | 每点基于前 20 步的 mean/std（min_periods=1） |
| `sliding_ma60` | MA60 滑动归一化 | 每点基于前 60 步的 mean/std（min_periods=1） |
| `sliding_ma120` | MA120 滑动归一化 | 每点基于前 120 步的 mean/std（min_periods=1） |

**注意**: `min_periods=1` 与现有 `dataset.py:185` `rolling(window=ma_window, min_periods=1)` 对齐。窗口前若干点用 expanding window（数据不足时逐步扩大），保证新旧数据输出一致。

**解析函数**:
```python
def parse_norm_mode(norm_mode: str) -> dict:
    """解析 norm_mode 返回归一化配置"""
    if norm_mode == 'full_window':
        return {'method': 'full_window', 'window': None}
    elif norm_mode.startswith('sliding_ma'):
        window = int(norm_mode.split('_ma')[-1])
        return {'method': 'sliding_ma', 'window': window}
    else:
        raise ValueError(f"Unknown norm_mode: {norm_mode}")
```

### 1.3 数据分割：Target-based Split

**核心原则**: 按 target 区间分割，不是按 window_start 分割

**window_size 定义**:
```python
window_size = lookback + predict  # 禁止 +1，防止未来一步泄露
```

**样本 schema**:
```python
SampleSchema = {
    'symbol': str,
    'window_start': int,           # 窗口起点
    'lookback_start': int,         # lookback 区间起点
    'lookback_end': int,           # lookback 区间终点 (= window_start + lookback)
    'target_start': int,           # target 区间起点 (= lookback_end)
    'target_end': int,             # target 区间终点 (= target_start + predict)
    'split': str,                  # 'train' / 'val' / 'test'
}
```

#### Time Split（时间分割）

**语义（重要）**：按**每只股票序列内的样本位置序号**切分，不是按日期。`train_end`/`val_end` 是整数位置，表示"该股票序列内第 N 个样本"。因各股票数据按交易日升序排列，位置序号≈交易日序号（A 股每年约 244 个交易日）。

- 每只股票独立切：长股票切 train/val/test 三份，短股票（序列长度 < train_end）可能全归 train。
- 同一位置序号在不同股票对应不同日期（因各股票上市时间/停牌不同），但这不影响 target 区间不相交（每只股票内部位置单调）。
- **用户估算**：train_end ≈ 训练期交易日数。如 train 用 2018-01~2023-06（约 5.5 年 ≈ 1340 交易日），可设 train_end≈1340；val 至 2024-12（再 1.5 年 ≈ 370 交易日），val_end≈1710。实际按数据调整。

按 target_end 切分：
```python
def time_split(samples, train_end, val_end):
    """
    train_end/val_end: 整数位置（每只股票序列内样本序号）

    train: target_end <= train_end
    val:   target_start >= train_end and target_end <= val_end  
    test:  target_start >= val_end
    """
    train_samples = [s for s in samples if s['target_end'] <= train_end]
    val_samples = [s for s in samples if s['target_start'] >= train_end and s['target_end'] <= val_end]
    test_samples = [s for s in samples if s['target_start'] >= val_end]
    return train_samples, val_samples, test_samples
```

位置边界示例（A 股约 244 交易日/年）：
- train_end: 1340（≈ 2018-01 ~ 2023-06，5.5 年）
- val_end: 1710（≈ 至 2024-12，再 1.5 年）

#### Block Split（分层抽样）

以 target 时间块为单位分配，不是以 window index 为单位：

```python
def create_target_blocks(sym_samples, block_size=50):
    """
    创建 target 时间块（关键：块之间 target 时间区间不相交）
    
    算法：
    1. 取该股票 target 时间轴 [t_min, t_max]
    2. 按 block_size 切成不重叠的时间区间块
    3. 每个窗口按其 target 落在哪个时间块，归入该块
    4. 返回 block 列表，每个 block 是窗口列表
    
    注意：不是按样本列表顺序切分（那会退化为 window-index splitting → 泄露）
    """
    if not sym_samples:
        return []
    
    # 按 target_start 排序
    sorted_samples = sorted(sym_samples, key=lambda s: s['target_start'])
    
    # 取 target 时间范围
    t_min = sorted_samples[0]['target_start']
    t_max = sorted_samples[-1]['target_end']
    
    # 切成时间块
    blocks = []
    current_block_start = t_min
    
    while current_block_start < t_max:
        current_block_end = min(current_block_start + block_size, t_max)  # 末端块防超界
        
        # 收集 target 落在 [current_block_start, current_block_end) 的窗口
        block_windows = []
        for s in sorted_samples:
            if s['target_start'] >= current_block_start and s['target_end'] <= current_block_end:
                block_windows.append(s)
        
        if block_windows:
            blocks.append(block_windows)
        
        current_block_start = current_block_end
    
    # 断言：块之间 target 区间不相交
    for i, block_a in enumerate(blocks):
        for j, block_b in enumerate(blocks):
            if i != j:
                intervals_a = [(s['target_start'], s['target_end']) for s in block_a]
                intervals_b = [(s['target_start'], s['target_end']) for s in block_b]
                assert not intervals_overlap(intervals_a, intervals_b), f"block {i}/{j} overlap!"
    
    return blocks


def block_split(samples, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    """
    以 target 时间块为单位分配 train/val/test
    
    关键：同一股票相邻窗口可能跨 split（因为以 block 为单位）
    """
    rng = np.random.RandomState(seed)
    
    # 按股票分组
    by_symbol = defaultdict(list)
    for s in samples:
        by_symbol[s['symbol']].append(s)
    
    train, val, test = [], [], []
    
    for symbol, sym_samples in by_symbol.items():
        # 创建 target 时间块（块之间不相交）
        blocks = create_target_blocks(sym_samples, block_size=50)
        
        # 随机分配 blocks
        rng.shuffle(blocks)
        n_train = int(len(blocks) * train_ratio)
        n_val = int(len(blocks) * val_ratio)
        
        for block in blocks[:n_train]:
            train.extend(block)
        for block in blocks[n_train:n_train+n_val]:
            val.extend(block)
        for block in blocks[n_train+n_val:]:
            test.extend(block)
    
    return train, val, test
```

#### No-Leakage 检查（区间语义）

**关键**: 检查 target 时间区间不相交，而非元组相等。

```python
def validate_no_leakage(train_samples, val_samples, test_samples):
    """
    验证 target 时间区间无重叠（区间语义，非元组语义）
    
    区间 [s, e) 不相交: e1 <= s2 或 e2 <= s1
    """
    def get_symbols(samples):
        return set(s['symbol'] for s in samples)
    
    def get_intervals(samples, symbol):
        return [(s['target_start'], s['target_end']) for s in samples if s['symbol'] == symbol]
    
    def intervals_overlap(intervals_a, intervals_b):
        """检查两组区间是否有相交"""
        for s1, e1 in intervals_a:
            for s2, e2 in intervals_b:
                # 区间相交: NOT (e1 <= s2 OR e2 <= s1)
                if not (e1 <= s2 or e2 <= s1):
                    return True
        return False
    
    all_symbols = get_symbols(train_samples) | get_symbols(val_samples) | get_symbols(test_samples)
    
    pairs = [
        (train_samples, val_samples, "train/val"),
        (train_samples, test_samples, "train/test"),
        (val_samples, test_samples, "val/test"),
    ]
    
    for samples_a, samples_b, pair_name in pairs:
        for sym in all_symbols:
            intervals_a = get_intervals(samples_a, sym)
            intervals_b = get_intervals(samples_b, sym)
            
            if intervals_overlap(intervals_a, intervals_b):
                raise AssertionError(f"{pair_name} target overlap for {sym}: intervals intersect!")
    
    print("No-leakage check passed (interval semantics)")
```

### 1.4 Backtest 设计

backtest 不是 split_mode，是 dataset_role：

```python
BacktestSchema = {
    'symbol': str,
    'context_start': int,          # lookback 区间起点
    'context_end': int,            # lookback 区间终点
    'target_start': int,           # predict 区间起点
    'target_end': int,             # predict 区间终点
    'x_context': np.ndarray,       # lookback 原始数据
    'y_timestamp': np.ndarray,     # predict 时间戳
    'actual_target': np.ndarray,   # predict 原始数据（只用于评估，不进入模型）
    'normalization_meta': dict,    # 归一化元数据
}

# 断言
assert context_end < target_start, "lookback/target 边界错误"
assert normalizer.fit_range <= context_end, "normalizer 使用了未来数据"
```

### 1.5 Tokenizer 训练与一致性

#### 1.5.1 Tokenizer 设计澄清（关键）

本项目的 tokenizer 是 **VQ-VAE 式分词器**（`KronosTokenizer`，`model/kronos.py`），其设计要点：

1. **Tokenizer 是「微调」而非「从零训练」**。`KronosTokenizer` 无 `.train()` 类方法，只有 `from_pretrained`（继承 `PyTorchModelHubMixin`）。训练流程是：加载预训练 tokenizer → 用目标数据微调编码器/解码器权重 → `save_pretrained`。

2. **Tokenizer 架构由 model_type 决定，不由 norm_mode 决定**。预训练 tokenizer 只有两个：
   - `Kronos-Tokenizer-2k`（group_size=5，2048 context）→ **mini** 模型
   - `Kronos-Tokenizer-base`（group_size=4，512 context）→ **small / base** 模型

   `group_size`、context 长度是**架构超参**，预训练时定死，微调不改。tokenizer 与 model_type 绑定是因为 **predictor 的 token 维度必须与 tokenizer 输出匹配**（架构耦合），不是「vocab 三选一」。

3. **vocab_size 不是微调参数**。codebook/vocab 在预训练时已定（2k tokenizer 的 vocab 由 `s1_bits/s2_bits/group_size` 决定）。微调只调编码器/解码器权重，不改 vocab。因此 `mini=2048/small=4096/base=8192` 的映射是**错误的**——实际是「mini 用 2k tokenizer，small 与 base 共用 base tokenizer」，不存在 small=4096。

4. **norm_mode 影响 tokenizer，故不同 norm_mode 必须各自微调 tokenizer，不可复用**。原因：tokenizer 微调时，QlibDataset 按 norm_mode 归一化 OHLCV 后喂给 tokenizer；不同 norm_mode（如 full_window vs sliding_ma60）的归一化分布不同（均值/方差尺度、自相关结构不同），微调后的 tokenizer 编码器权重是针对该分布优化的。若把 sliding_ma60 微调的 tokenizer 用于 full_window 数据，编码失真。「tokenizer 与 predictor 必须同 norm_mode」的本质是**两者吃同一归一化分布的数据**——架构（model_type）匹配决定能对接，分布（norm_mode）匹配决定编码有效，两者缺一不可。因此 `{norm_mode}` 是 tokenizer 路径的必要键：同 model_type 下不同 norm_mode 的 tokenizer **架构相同、权重不同**，必须分别微调、分别存储、分别匹配。

5. **微调损失**：`recon_loss = mse(z_pre, x) + mse(z, x)`（重建）+ `bsq_loss`（码本 commit），见 `deprecated/finetune/train_tokenizer.py:194-195`。这是权重微调，不是重新学习码本。

> **对早期方案的纠正**：早期 §1.5 写的 `KronosTokenizer.train(data=, vocab_size=)` 是不存在的 API；`vocab_map={mini:2048, small:4096, base:8192}` 是杜撰映射；「每个 norm_mode 单独训练 tokenizer」的表述把「按 norm_mode 微调」误解为「按 norm_mode 从零建架构」。均已纠正。

#### 1.5.2 Tokenizer 微调入口

**采样机制（关键，对齐现有实现）**：tokenizer 微调**不用「按比例抽样」**（早期方案臆造的 `sample_ratio=0.1` 是错的，已删除），而是**步数驱动放回采样**——每 epoch 用 `RandomSampler(replacement=True, num_samples=n_train_iter)` 采样固定步数，跨 epoch 覆盖数据。这是现有 `deprecated/finetune/train_tokenizer.py:94-95,132` 的真实做法：

```python
n_train_iter = n_train_iter_multiplier * batch_size   # 默认 2000 * 16 = 32000 步/epoch
n_val_iter   = n_val_iter_multiplier   * batch_size   # 默认 400  * 16 = 6400 步/epoch
sampler = RandomSampler(dataset, replacement=True, num_samples=n_train_iter)  # 放回采样
```

步数驱动而非比例驱动的原因：数据量大时按比例抽样每 epoch 太长；放回采样 + 固定步数可控制每 epoch 时长，多 epoch 仍能覆盖全量。

```python
# finetune/tokenizer/train.py

def finetune_tokenizer(
    norm_mode: str,
    model_type: str,                     # mini → Kronos-Tokenizer-2k; small/base → Kronos-Tokenizer-base
    data_path: str,                      # 预处理后的训练数据（已含 norm_mode 归一化信息）
    output_dir: str,
    epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    n_train_iter_multiplier: int = 2000, # 每 epoch 采样步数 = multiplier × batch_size
    n_val_iter_multiplier: int = 400,    # val 步数 = multiplier × batch_size
    seed: int = 42
):
    """
    微调 tokenizer（非从零训练）

    步骤：
    1. 按 model_type 选预训练 tokenizer：
       - mini → pretrained/Kronos-Tokenizer-2k
       - small/base → pretrained/Kronos-Tokenizer-base
    2. 加载数据，KronosDataset 按 norm_mode 归一化
    3. 步数驱动放回采样（非比例抽样），每 epoch n_train_iter 步
    4. 微调循环：recon_loss + bsq_loss，标准 backward
    5. 每 epoch 用 val split 算重建损失，early stopping
    6. 保存到 outputs/tokenizers/{norm_mode}/{model_type}/
    """
    set_seed(seed)

    # 1. 选预训练 tokenizer（架构由 model_type 决定）
    pretrained_map = {
        'mini': 'pretrained/Kronos-Tokenizer-2k',
        'small': 'pretrained/Kronos-Tokenizer-base',
        'base': 'pretrained/Kronos-Tokenizer-base',
    }
    tokenizer = KronosTokenizer.from_pretrained(pretrained_map[model_type])
    tokenizer.to(device)

    # 2. 数据按 norm_mode 归一化（KronosDataset 内部处理）
    train_dataset = KronosDataset(train_data, train_indices, config, mode='train')
    val_dataset   = KronosDataset(val_data,   val_indices,   config, mode='val')

    # 3. 步数驱动放回采样（关键：非比例抽样）
    n_train_iter = n_train_iter_multiplier * batch_size
    n_val_iter   = n_val_iter_multiplier   * batch_size
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=RandomSampler(train_dataset, replacement=True, num_samples=n_train_iter),
        num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        sampler=RandomSampler(val_dataset, replacement=True, num_samples=n_val_iter),
        num_workers=0, drop_last=False,
    )

    # 4. 微调循环
    optimizer = torch.optim.AdamW(
        tokenizer.parameters(), lr=learning_rate, weight_decay=0.1,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate,
        steps_per_epoch=len(train_loader), epochs=epochs, pct_start=0.03, div_factor=10,
    )

    best_val_loss = float('inf')
    for epoch in range(epochs):
        tokenizer.train()
        for batch_x, _, _ in train_loader:
            batch_x = batch_x.to(device)
            zs, bsq_loss, _, _ = tokenizer(batch_x)
            z_pre, z = zs
            recon_loss = F.mse_loss(z_pre, batch_x) + F.mse_loss(z, batch_x)
            loss = (recon_loss + bsq_loss) / 2
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()

        # 5. val 重建损失 + early stopping
        tokenizer.eval()
        val_loss_sum, val_count = 0.0, 0
        with torch.no_grad():
            for batch_x, _, _ in val_loader:
                batch_x = batch_x.to(device)
                zs, _, _, _ = tokenizer(batch_x)
                _, z = zs
                vl = F.mse_loss(z, batch_x)
                val_loss_sum += vl.item() * batch_x.size(0)
                val_count += batch_x.size(0)
        avg_val_loss = val_loss_sum / val_count

        if avg_val_loss < best_val_loss - early_stopping_min_delta:
            best_val_loss = avg_val_loss
            tokenizer.save_pretrained(f"{output_path}/checkpoints/best_model")
        # ... early stopping 逻辑（patience + grace period，参考 deprecated 实现）

    # 6. 保存最终 + 元数据
    output_path = f"outputs/tokenizers/{norm_mode}/{model_type}"
    tokenizer.save_pretrained(f"{output_path}/checkpoints/final_model")

    meta = {
        'norm_mode': norm_mode,
        'model_type': model_type,
        'pretrained_base': pretrained_map[model_type],
        'data_fingerprint': compute_fingerprint(data_path),
        'learning_rate': learning_rate,
        'epochs': epochs,
        'n_train_iter': n_train_iter,        # 实际采样步数
        'n_val_iter': n_val_iter,
        'best_val_loss': best_val_loss,      # 重建损失（验收用）
    }
    with open(f"{output_path}/meta.json", 'w') as f:
        json.dump(meta, f, indent=2)
```

**CLI 命令**:
```bash
python finetune/tokenizer/train.py \
    --norm-mode sliding_ma60 \
    --model mini \
    --epochs 30 \
    --n-train-iter-multiplier 2000
```


（数据来自 `finetune/data/processed/{norm_mode}/...`，已按 norm_mode 预处理）

#### 1.5.3 路径键控含义（澄清）

`outputs/tokenizers/{norm_mode}/{model_type}/` 的含义：
- `{model_type}` 决定**架构**（2k vs base，mini/small/base 三选其基础）；
- `{norm_mode}` 标记**该 tokenizer 是用哪种归一化分布的数据微调的**。

即同一 model_type 下，不同 norm_mode 的 tokenizer **架构相同、权重不同**（因微调数据分布不同）。predictor 训练时必须加载与自身 norm_mode + model_type 都匹配的 tokenizer（架构匹配 + 分布匹配）。

#### 1.5.4 一致性校验

训练 predictor 前必须验证：

```python
def validate_tokenizer_consistency(tokenizer_meta, predictor_config):
    """验证 tokenizer 和 predictor 数据/架构一致性"""
    # 架构匹配：model_type 决定 tokenizer 基础架构
    assert tokenizer_meta['model_type'] == predictor_config.model_type, \
        f"tokenizer model_type {tokenizer_meta['model_type']} != predictor {predictor_config.model_type}"
    # 分布匹配：tokenizer 微调时的 norm_mode 必须与 predictor 训练 norm_mode 一致
    assert tokenizer_meta['norm_mode'] == predictor_config.norm_mode, \
        f"tokenizer norm_mode {tokenizer_meta['norm_mode']} != predictor {predictor_config.norm_mode}"
    print("Tokenizer consistency check passed")
```

checkpoint 必须记录 tokenizer 信息：
```json
{
  "data": {
    "norm_mode": "sliding_ma60",
    "data_fingerprint": "sha256:..."
  },
  "tokenizer": {
    "path": "outputs/tokenizers/sliding_ma60/mini",
    "pretrained_base": "pretrained/Kronos-Tokenizer-2k",
    "data_fingerprint": "sha256:..."
  }
}
```

#### 1.5.5 执行计划位置

Phase 2 新增 `finetune/tokenizer/train.py` 微调入口，确保 Phase 4 训练 predictor 时有「架构匹配 + 分布匹配」的 tokenizer。微调而非从零训练，故无需大规模数据/长训练，epochs≈30 即可（参考 `deprecated/finetune/train_tokenizer.py`）。

#### 1.5.6 微调效果检验

**采样机制**（§1.5.2 已述）：步数驱动放回采样，非比例抽样。`n_train_iter = n_train_iter_multiplier × batch_size`（默认 32000 步/epoch），跨多 epoch 覆盖全量。

**检验方式**（两层，对齐现有 `deprecated/finetune/train_tokenizer.py` + `validate_tokenizer.py`）：

**层 1 — 训练中（自动）**：每 epoch 用 val split 算重建损失 `mse(z, batch_x)`（z = 分词后解码的重建），驱动 early stopping：
```python
zs, _, _, _ = tokenizer(batch_x)
_, z = zs
val_loss = F.mse_loss(z, batch_x)   # 重建损失
```
记录 `best_val_loss` 到 meta.json。

**层 2 — 训练后（脚本 `finetune/tokenizer/validate.py`）**：对比「预训练 tokenizer vs 微调后」在同一 val 数据上的重建误差 mean/std：
```
预trained Kronos-Tokenizer-2k:  0.001234 +/- 0.000456
微调后 (sliding_ma60):          0.000987 +/- 0.000321   # 应更低
```

**验收标准**：

| 标准 | 级别 | 说明 |
|------|------|------|
| 微调后 val 重建损失 < 预训练在同数据重建损失 | 必要 | 证明 tokenizer 适应了目标 norm_mode 分布 |
| 重建损失收敛（early stopping 触发，非暴力训满） | 必要 | 证明非过拟合式下降 |
| 短 predictor 预训练 loss 下降、IC 非零 | 可选 | 下游验证分词对预测有效；本质属 Phase 4，非 tokenizer 阶段阻断项 |

**重建损失的合理性**：重建损失是 tokenizer 微调的**直接目标**（编解码忠实重建该分布数据），微调后低于预训练即证明适应了分布。它是必要的微调质量指标。

**重建损失的局限**：重建损失低 ≠ 分出来的 token 对预测有用（间接性）。一个完美重建但 token 冗余的 tokenizer 仍可能让 predictor 学不到东西。故「可选」层用短 predictor 训练做下游验证——但这是 Phase 4 的事，tokenizer 微调阶段用重建损失 + 与预训练对比即可，下游效果在 Phase 4 自然体现。

---

### 1.6 度量口径定稿（关键：train + eval 共用）

> **背景**：现有 `eval_ddp.py:226-241` 与 `train_ddp.py:365-378` 同源，trajectory IC 均用**反归一化后的原始价格序列**计算。原始价格序列强自相关，即使预测完全持平也会因与实际趋势同向拿到高 IC，导致 IC 被价格趋势成分夸大。该问题在 eval 中是「报告失真」，在 train 中更严重——`train_ddp.py:731` 的 `close_trajectory_ic` 经 `calculate_combined_score`（`shared/eval.py:74`，ic_weight=0.6）驱动 **checkpoint 选择与 early stopping**，即用错误目标选模型。
>
> **核心约束**：重构后 `train.py` 与 `eval.py` 必须共用 `core/metrics.py` 的同一组度量函数，禁止训练用一套口径、评估用另一套。

#### 1.6.1 Trajectory IC：去趋势口径

trajectory IC 的设计意图（见项目 memory）是「同一股票内预测轨迹**形状**与实际轨迹**形状**的相似度」。原始价格序列不满足此意图，必须对**去趋势序列**算相关。

**去趋势定义**：对每个样本的预测段与实际段，相对 lookback 末根 K 线（baseline）归一化为相对序列：

```python
# core/metrics.py

def detrend_to_baseline(series: np.ndarray, baseline: float) -> np.ndarray:
    """
    去趋势：相对 baseline 的相对序列
    series: (pred_len,) 或 (pred_len, n_features)
    baseline: 标量或 (n_features,)，取 lookback 最后一根对应特征值
    返回: (series - baseline) / (|baseline| + eps)
    """
    return (series - baseline) / (np.abs(baseline) + 1e-8)


def safe_trajectory_ic(pred_traj, actual_traj, min_len=3):
    """
    安全计算轨迹 IC（去趋势序列上的 Pearson + Spearman）

    Args:
        pred_traj: 去趋势后的预测序列（相对 baseline）
        actual_traj: 去趋势后的实际序列（相对 baseline）
    Returns:
        (ic, rank_ic)，无法计算时返回 (None, None)

    约束: 调用方必须先 detrend_to_baseline，禁止传入原始价格。
    """
    if len(pred_traj) < min_len:
        return None, None
    pred_std = np.std(pred_traj)
    actual_std = np.std(actual_traj)
    if pred_std < 1e-8 or actual_std < 1e-8:
        return None, None
    ic = np.corrcoef(pred_traj, actual_traj)[0, 1]
    rank_ic, _ = spearmanr(pred_traj, actual_traj)
    ic = ic if np.isfinite(ic) else None
    rank_ic = rank_ic if np.isfinite(rank_ic) else None
    return ic, rank_ic
```

**口径规则**：
- `pred_traj` / `actual_traj` 必须是 `detrend_to_baseline` 的输出，**禁止传入原始价格**。
- baseline 取 lookback 末根（与现有 DA 基线一致：`orig[lookback-1]`）。
- 训练 checkpoint 选择的 `close_trajectory_ic` 与评估报告的 `close_trajectory_ic` **调用同一函数**，保证口径一致。

#### 1.6.2 Direction Accuracy：excess DA

DA 基线为 lookback 末根（`baseline = values[lookback-1]`），与现有实现一致。新增 **excess DA** 以使数字可解读：

```python
def safe_corrcoef(x, y, default=0.0): ...   # 保持现有实现
def safe_spearmanr(x, y, default=0.0): ...  # 保持现有实现

def excess_da(model_da: float, naive_da: float) -> float:
    """
    excess DA = model DA − naive DA
    naive_da: 持平预测（pred == baseline）的 DA，即实际方向中「与 baseline 同向」的比例
    让 DA 可解读: excess > 0 才是真 alpha，excess ≈ 0 等同朴素预测
    """
    return model_da - naive_da
```

**规则**：eval 主输出与训练监控均报 `model_da` 与 `excess_da` 两个数；`naive_da` 由实际序列统计得出（不依赖模型）。

#### 1.6.3 聚合：保留分布，不只传均值

现有 `eval_ddp.py:256-332` / `train_ddp.py:660-729` 跨 GPU 只聚合 `mean×n`，丢失 std/分位数。重构后聚合须保留分布：

```python
def aggregate_ic(local_ic_lists, world_size, device):
    """
    聚合各 GPU 的 IC 列表，保留分布
    返回: {mean, std, p25, p50, p75, n}
    """
    # 收集 sum / sum_sq / n 用于 mean 与 std
    # 分位数需先 all_gather 完整列表（或用近似分位数算法）再算
    return {
        'mean': ..., 'std': ...,
        'p25': ..., 'p50': ..., 'p75': ...,
        'n': total_n,
    }
```

**规则**：IC 报告必须含 mean + std + 至少 p25/p50/p75。仅 mean 不足以判断稳健性（IC 0.21 的 std 是 0.05 还是 0.3 决定数字是否可信）。

#### 1.6.4 Backtest IC：不跨股票混算

现有 `simple_backtest.py:464-480` 把所有股票、所有时间步的 `pred_gain`/`actual_gain` 混成一个列表算单一 IC，受 beta/时期效应污染。

**规则**：backtest IC 改为**逐股票计算再聚合**：

```python
# 每只股票先算自己的 IC，再对股票间 IC 取均值/分布
per_stock_ic = [safe_corrcoef(stock_pred_gains, stock_actual_gains) for stock in stocks]
backtest_ic_mean = np.mean([ic for ic in per_stock_ic if ic is not None])
backtest_ic_std = np.std(...)
```

结合项目目标（预测 K 线轨迹，非选股 alpha），backtest IC 不作为主输出，仅作附录参考；主输出用 1.6.5 的可懂指标。

**因子打分参数（E7）**：现有 `simple_backtest.py:46-47` 的 `SIGNAL_CENTER=0.084` / `SIGNAL_STEEPNESS=21` 硬编码无注释。重构后移入 `BacktestConfig`，并注释来源（经验标定的 sigmoid 中心/陡度，用于把涨幅映射到 [0,1] 分值）：

```python
@dataclass
class BacktestConfig:
    signal_center: float = 0.084      # sigmoid 中心：涨幅 8.4% 处分值=0.5（经验标定）
    signal_steepness: float = 21.0    # sigmoid 陡度：控制分值对涨幅的敏感度（经验标定）
```

#### 1.6.5 可懂指标三件套（eval 主输出）

契合项目目标（指标要股市人群可懂、能对标尺），eval 主输出采用三个股市人秒懂的标尺，IC/RankIC 降为附录：

| 指标 | 含义 | 标尺 | 能力维度 |
|------|------|------|----------|
| 方向胜率 | 每步涨跌对错（= DA） | 50%=随机，55%=微弱，60%+=强 | 方向 |
| 振幅误差率 | 预测振幅 ÷ 实际振幅 | 1.0=完美，0.8-1.2=可用，偏离>30%=失真 | 振幅 |
| 涨跌停命中率 | 预测涨停的命中比例 | 随机≈1-3%，10%+=有信号，30%+=强 | 极端事件 |

```python
def amplitude_error_rate(pred_high_low, actual_high_low):
    """预测振幅 / 实际振幅，1.0 为完美"""
    return (pred_high_low) / (actual_high_low + 1e-8)

def limit_hit_rate(pred_limit_flags, actual_limit_flags):
    """预测涨停的样本中实际涨停比例（A 股 ±10% / 创业板 ±20%）"""
    # pred_limit_flags: 模型预测触及涨停的样本
    # actual_limit_flags: 实际涨停的样本
    return actual_limit_flags[pred_limit_flags].mean()
```

**反向警告（纳入）**：涨跌停命中率若低（<10%）会刺破「模型能预测」的幻觉，但这正是诚实度量所需。0.21 IC 谁都不懂意味什么，「涨停命中 8%」所有人都懂意味「勉强比随机好」。可懂性优先于学术指标。

#### 1.6.6 combined_score 迁移与权重审查

`calculate_combined_score`（`shared/eval.py:74`，`ic_weight=0.6, da_weight=0.4`）迁入 `core/metrics.py`。

**权重审查要求**：去趋势后 IC 量级可能变化（原始价格口径的虚高 IC 消失），0.6/0.4 权重可能不再合适。Phase -1 须用正确口径重新标定权重，并在 `summary.json` 记录最终权重与标定依据。

#### 1.6.7 Early Stopping patience 审视（train 专用）

> **背景**：现有 `train_ddp.py:1095-1114` 的 `patience_counter` 归零有**两条独立途径**——`avg_val_loss < best_val_loss`，**或** `ic_smoothed > best_ic`。后者 `ic_smoothed` 是错误口径 `close_trajectory_ic` 的 3-epoch 滑动均值（`train_ddp.py:1035/1077`）。即 early stopping 的**停止时机**不只由 checkpoint 选择决定，更由 patience 累积节奏决定。IC 被价格趋势托底在虚高水平时，`ic_smoothed > best_ic` 频繁触发 → patience 频繁归零 → 训练在该停时不停。E1 修正为去趋势口径后，IC 量级下降、波动结构变化，patience 累积节奏随之偏移，停止时机必然改变。这是 §1.6.6 combined_score 权重审查的同源问题——既然权重需重标定，early stopping 的归零条件与 patience 阈值理应同步审视。

**Phase -1 须定稿**（写入 `core/metrics.py` 旁的训练逻辑说明或 `TrainConfig` 注释）：

1. **patience 归零条件**：去趋势口径下，`patience_counter` 归零是否仍应包含「`ic_smoothed > best_ic`」这条途径？两种选择需明确记录：
   - 保留：IC 改进也归零 patience —— 需在去趋势口径下重新确认 `early_stopping_patience=12` 是否仍合适（去趋势后 IC 波动更大，原阈值可能过松或过紧）。
   - 移除：patience 仅由 val_loss 驱动，IC 只用于 checkpoint 选择 —— 更干净，但需确认 val_loss 单独能否及时停止。
2. **patience 阈值**：无论保留或移除 IC 途径，`early_stopping_patience` 需在去趋势口径的 IC 曲线上重新确认，记录最终值与依据。
3. **Phase 4 不要求停止 epoch 一致**：口径变更必然改变 early stopping 停止时机，等价性验证**不应**要求新训练停止 epoch 与旧训练一致，应改为「loss 收敛曲线形态可比」（见 §8.3 / Phase 4）。

#### 1.6.8 使用规则

所有新代码（`core/metrics.py`、`train.py`、`eval.py`、`backtest.py`）禁止直接调用 `np.corrcoef` / `spearmanr`，必须：

```python
from core.metrics import (
    safe_corrcoef, safe_spearmanr, safe_trajectory_ic,
    detrend_to_baseline, excess_da, aggregate_ic,
    amplitude_error_rate, limit_hit_rate,
)
```

旧入口（`mode*/train.py`、`shared/eval*.py`、`simple_backtest.py`）随 Phase 6 deprecated 自然失效，无需逐个修改。但 **Phase 0 baseline 必须用新口径重算**（见 §7 Phase -1），不能用旧入口的错误口径 IC 作为基准。

---

## 二、目录结构

### 2.1 finetune 目录

```
finetune/
├── data/
│   ├── processed/
│   │   ├── {norm_mode}/
│   │   │   ├── lb{lookback}_pd{predict}/
│   │   │   │   ├── time/
│   │   │   │   │   ├── train.pkl     # 样本索引 + 原始窗口
│   │   │   │   │   ├── val.pkl
│   │   │   │   │   ├── test.pkl
│   │   │   │   │   ├── meta.pkl      # fingerprint, split 边界
│   │   │   │   ├── block/
│   │   │   │   │   ├── train.pkl
│   │   │   │   │   ├── val.pkl
│   │   │   │   │   ├── test.pkl
│   │   │   │   │   ├── meta.pkl
│   │   │   │   ├── backtest/
│   │   │   │   │   ├── samples.pkl   # backtest 样本
│   │   │   │   │   ├── meta.pkl

**说明**: processed/ 按 norm_mode 键控的原因：
- sliding_ma{N} 需要窗口起点前有 N 步历史，不同 N 的可用样本集不同
- 例如 sliding_ma60 需要每个样本前 60 步数据存在，sliding_ma120 需要 120 步
- 因此样本集随 norm_mode 变化，需分目录存储
- full_window 不需要额外历史，但为统一目录结构也保留 norm_mode 键
│
├── predictor/
│   ├── core/
│   │   ├── config.py                 # 配置对象（拆分 DataConfig/TrainConfig）
│   │   ├── paths.py                  # 路径构建函数
│   │   ├── schema.py                 # 样本 schema 定义
│   │   ├── normalization.py          # Normalizer 类
│   │   ├── splitting.py              # time/block/backtest split 算法
│   │   ├── dataset.py                # KronosDataset
│   │   ├── modeling.py               # model/tokenizer 加载
│   │   ├── metrics.py                # IC/trajectory IC/combined score
│   │   ├── checkpoints.py            # checkpoint 管理
│   │   ├── utils.py                  # 通用工具
│   │
│   ├── train.py                      # 训练入口
│   ├── eval.py                       # 评估入口
│   ├── backtest.py                   # 回测入口
│   ├── preprocess.py                 # 预处理入口
│
├── preprocess/
│   ├── generate_raw_windows.py
│   ├── split_time.py
│   ├── split_block.py
│   ├── generate_backtest.py
│
├── deprecated/                       # 废弃代码（不入 git，重构完成后移入）
```

### 2.2 outputs 目录

```
outputs/
├── models/
│   ├── {norm_mode}/
│   │   ├── lb{lookback}_pd{predict}/
│   │   │   ├── {split_mode}/
│   │   │   │   ├── mini/
│   │   │   │   │   ├── checkpoints/
│   │   │   │   │   │   ├── best_ic.safetensors
│   │   │   │   │   │   ├── best_combined.safetensors
│   │   │   │   │   ├── config.json       # 训练配置 + data fingerprint
│   │   │   │   │   ├── summary.json      # 结果 + 完整元数据
│   │   │   │   ├── small/
│   │   │   │   ├── base/
│
├── tokenizers/
│   ├── {norm_mode}/
│   │   ├── mini/                      # vocab_size=2048
│   │   ├── small/                     # vocab_size=4096
│   │   ├── base/                      # vocab_size=8192
│   │   │   ├── vocab.json
│   │   │   ├── merges.txt
│   │   │   ├── meta.json              # norm_mode, data_fingerprint
```

**注意**: tokenizer 路径统一使用 model_type（mini/small/base）作为键，vocab_size 隐含映射：
- mini → 2048
- small → 4096
- base → 8192

### 2.3 根目录

```
Kronos/
├── deprecated/                       # 废弃代码（不入 git）
├── finetune/
├── outputs/
├── data/
│   └── kline_daily_raw.pkl          # 原始数据（完整历史 + 训练验证测试期）
├── pretrained/
├── model/
├── docs/
```

**原始数据说明**:
- `data/kline_daily_raw.pkl`: 核心原始数据，供预处理使用
- 包含所有股票完整历史，时间覆盖训练/验证/测试期
- preprocess.py 从此文件读取并生成 processed/ 目录下的样本

**数据迁移策略**:
- 现有 `finetune/data/ma60_norm/block_lb400_pd10/` 约 1.5GB 旧格式 pkl 不删除
- Phase 3 用小样本从 raw 生成新 schema 数据验证一致性
- 验证通过后，新链路使用 processed/ 目录数据

---

## 三、路径构建函数

```python
# core/paths.py

def get_raw_path() -> str:
    """获取原始数据路径（统一位置）"""
    return "data/kline_daily_raw.pkl"


def get_split_data_path(norm_mode: str, lookback: int, predict: int, 
                        split_mode: str, split_name: str) -> str:
    """
    split_mode: time | block
    split_name: train | val | test
    """
    return f"finetune/data/processed/{norm_mode}/lb{lookback}_pd{predict}/{split_mode}/{split_name}.pkl"


def get_backtest_data_path(norm_mode: str, lookback: int, predict: int) -> str:
    """获取回测数据路径"""
    return f"finetune/data/processed/{norm_mode}/lb{lookback}_pd{predict}/backtest/samples.pkl"


def get_meta_path(norm_mode: str, lookback: int, predict: int,
                  split_mode: str = None, role: str = "train") -> str:
    """获取 meta 文件路径"""
    base = f"finetune/data/processed/{norm_mode}/lb{lookback}_pd{predict}"
    if split_mode:
        return f"{base}/{split_mode}/meta.pkl"
    elif role == "backtest":
        return f"{base}/backtest/meta.pkl"
    return f"{base}/meta.pkl"


def get_model_path(norm_mode: str, lookback: int, predict: int,
                   split_mode: str, model_type: str) -> str:
    """获取模型保存路径"""
    return f"outputs/models/{norm_mode}/lb{lookback}_pd{predict}/{split_mode}/{model_type}"


def get_tokenizer_path(norm_mode: str, model_type: str) -> str:
    """获取 tokenizer 路径"""
    return f"outputs/tokenizers/{norm_mode}/{model_type}"
```

---

## 四、配置对象

```python
# core/config.py

from dataclasses import dataclass, field
from typing import List

@dataclass
class DataConfig:
    """数据配置 - 不变量"""
    norm_mode: str = 'sliding_ma60'
    lookback: int = 400
    predict: int = 10
    split_mode: str = 'block'
    min_samples: int = field(init=False)
    features: List[str] = field(default_factory=lambda: ['open', 'high', 'low', 'close', 'vol', 'amt'])
    time_features: List[str] = field(default_factory=lambda: ['minute', 'hour', 'weekday', 'day', 'month'])
    clip: float = 5.0
    seed: int = 42
    
    def __post_init__(self):
        self.min_samples = self.lookback + self.predict  # 禁止 +1
        self.validate()  # 触发 norm_mode 白名单检查
    
    def validate(self):
        assert self.min_samples == self.lookback + self.predict
        assert self.norm_mode in ['full_window', 'sliding_ma20', 'sliding_ma60', 'sliding_ma120']


@dataclass
class TrainConfig:
    """训练超参"""
    model_type: str = 'mini'
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 0.01  # 从大值起步
    weight_decay: float = 0.01
    early_stopping_patience: int = 12
    warmup_epochs: int = 2


@dataclass
class ArtifactConfig:
    """模型/Tokenizer 路径"""
    model_type: str = 'mini'  # mini/small/base，用于 tokenizer 路径键与预训练模型选择
    pretrained_model_path: str = 'pretrained/Kronos-mini'
    tokenizer_path: str = None
    output_dir: str = None

    def resolve_tokenizer_path(self, data_config: DataConfig):
        """根据 norm_mode 和 model_type 自动选择 tokenizer"""
        if self.tokenizer_path:
            return self.tokenizer_path
        # 统一使用 model_type 作为键（mini/small/base）
        return f"outputs/tokenizers/{data_config.norm_mode}/{self.model_type}"
```

---

## 五、核心模块职责

| 模块 | 职责 |
|------|------|
| `config.py` | 配置对象定义与校验 |
| `paths.py` | 所有路径构建 |
| `schema.py` | SampleSchema, BacktestSchema, MetaSchema 定义 |
| `normalization.py` | FullWindowNormalizer, SlidingMANormalizer |
| `splitting.py` | time_split, block_split, validate_no_leakage |
| `dataset.py` | KronosDataset（runtime 归一化） |
| `modeling.py` | load_tokenizer, load_model, validate_consistency |
| `metrics.py` | trajectory_ic, combined_score, safe_corrcoef, safe_spearmanr |
| `checkpoints.py` | save_checkpoint, load_checkpoint, select_best |
| `utils.py` | set_seed, get_device, format_timestamp |

---

## 六、命令行接口

### 6.1 训练

```bash
# 单卡
python finetune/predictor/train.py \
    --norm-mode sliding_ma60 \
    --lookback 400 \
    --predict 10 \
    --split-mode block \
    --model mini \
    --epochs 50 \
    --lr 0.01

# 多卡
torchrun --nproc_per_node=4 \
    finetune/predictor/train.py \
    --norm-mode sliding_ma60 \
    --lookback 400 \
    --split-mode block \
    --model mini
```

### 6.2 评估

**功能范围**:

- **DDP 分布式评估**: 支持 `torchrun --nproc_per_node=N`（迁移自 `shared/eval_ddp.py`）
- **多模型批量对比**: `--models mini,small,base`（迁移自 `shared/eval_ddp.py`）
- **checkpoint 选择**: `--checkpoint best_ic` 或 `--checkpoint best_combined`（迁移自 `shared/eval_ddp.py`）
- **分层评估**: `--strata large,mid,small` 按市值分层评估 —— **新建**（旧 `eval_ddp.py` 无此能力，需在 `core/` 中实现）

**主输出**（契合可懂性目标，见 §1.6.5）:

eval 默认输出可懂指标三件套（方向胜率 / 振幅误差率 / 涨跌停命中率）+ excess DA；IC/RankIC/trajectory IC 作为附录给量化人。

```bash
# 单卡单模型
python finetune/predictor/eval.py \
    --norm-mode sliding_ma60 \
    --lookback 400 \
    --split-mode block \
    --model mini \
    --n-samples 1000

# 多卡 DDP 评估
torchrun --nproc_per_node=4 \
    finetune/predictor/eval.py \
    --norm-mode sliding_ma60 \
    --model mini \
    --n-samples -1

# 多模型批量对比
python finetune/predictor/eval.py \
    --norm-mode sliding_ma60 \
    --models mini,small,base \
    --checkpoint best_combined

# 分层评估（新建能力）
python finetune/predictor/eval.py \
    --norm-mode sliding_ma60 \
    --model mini \
    --strata large,mid,small
```

### 6.3 回测

**主输出**: 可懂指标三件套（§1.6.5）。**backtest IC 不作为主输出**（§1.6.4：逐股票算再聚合，仅作附录参考，因项目目标为轨迹预测非选股 alpha）。

```bash
python finetune/predictor/backtest.py \
    --norm-mode sliding_ma60 \
    --lookback 400 \
    --model mini \
    --n-samples 1000
```

### 6.4 预处理

```bash
python finetune/predictor/preprocess.py \
    --norm-mode sliding_ma60 \
    --lookback 400 \
    --split-mode block \
    --validate
```

---

## 七、执行计划（修订后）

### Phase -1: 度量口径定稿（关键前置）

> **为什么需要**：现有 train/eval 共用的 trajectory IC 用原始价格序列计算（见 §1.6），口径错误。Phase 0 baseline 与 Phase 4 等价性验证都依赖度量口径，若口径未定稿，baseline 是「被夸大的数字」、等价性验证是「对齐错误基准」。必须先定稿口径，再建 baseline。

1. 编写 `core/metrics.py` 全部函数（§1.6）：
   - `detrend_to_baseline` + `safe_trajectory_ic`（去趋势口径）
   - `safe_corrcoef` / `safe_spearmanr`
   - `excess_da`（model DA − naive DA）
   - `aggregate_ic`（保留 std/分位数）
   - `amplitude_error_rate` / `limit_hit_rate`（可懂指标三件套）
   - `calculate_combined_score` 迁入 + 权重重标定
2. 明确 train.py 与 eval.py 共用 `core/metrics.py` 同一函数，禁止两端口径不一致。
3. 用现有 checkpoint 在小样本上跑新旧口径对照：记录原始价格口径 IC vs 去趋势口径 IC 的差异，作为后续等价性验证的换算参照（见 Phase 4）。
4. combined_score 权重用正确口径重新标定，记录最终权重与依据。
5. **early stopping 审视（§1.6.7）**：定稿 `patience_counter` 归零条件（是否保留「IC 改进归零 patience」途径）与 `early_stopping_patience` 阈值，在去趋势口径的 IC 曲线上重新确认，记录最终选择与依据。
6. **不移动任何旧代码，不修改旧入口**。

### Phase 0: 冻结与基线

1. 选定基线配置：`sliding_ma60 + lb400 + pd10 + block + mini`
2. 固定 tokenizer、checkpoint、数据样本数、seed=42
3. 运行旧入口，记录训练/评估输出作为 baseline —— **但 IC 度量用 Phase -1 定稿的正确口径重算**（不能用旧入口的错误口径 IC 作基准）
4. baseline 记录内容：loss、去趋势 trajectory IC（mean/std/分位数）、excess DA、可懂指标三件套、checkpoint 路径
5. **不移动任何旧代码**

### Phase 1: 定义数据契约

1. 编写 `core/schema.py`：SampleSchema, BacktestSchema, MetaSchema
2. 编写 `core/normalization.py`：Normalizer 类
3. 编写 `core/splitting.py`：split 算法 + validate_no_leakage
4. 选择归一化策略：runtime

### Phase 2: 抽取核心模块（无副作用）

1. `core/config.py`：拆分 DataConfig/TrainConfig/ArtifactConfig（含 S2 修正：ArtifactConfig 补 `model_type` 字段）
2. `core/paths.py`：路径函数
3. `core/metrics.py`：指标计算（Phase -1 已完成函数定义，本阶段集成到 core/ 并补 unit test）
4. `core/utils.py`：工具函数
5. `finetune/tokenizer/train.py`：tokenizer 训练入口（新增）

旧入口继续可运行。

### Phase 3: 新 Dataset / Preprocess

1. `core/dataset.py`：KronosDataset（runtime 归一化）
2. `preprocess.py`：预处理入口
3. 用小样本验证新旧数据输出一致
4. 运行 validate_no_leakage

### Phase 4: 新训练/评估/回测入口

1. `train.py`：统一训练入口（支持单卡/多卡），checkpoint 选择 IC 用 `core/metrics.py` 去趋势口径
2. `eval.py`：评估入口，主输出可懂指标三件套 + excess DA，附录 IC/RankIC
3. `backtest.py`：回测入口，backtest IC 逐股票算再聚合（附录），主输出可懂指标
4. **等价性验证基准（口径变更下重定义）**：
   - 旧入口 baseline 用的是**原始价格口径 IC**（错误），新入口用**去趋势口径 IC**（正确），两者数值必然不同，**不能要求新入口复现旧 IC 数值**。
   - 验证标准改为：(a) 新入口在正确口径下自洽（同 seed 跑两次落在噪声地板内）；(b) 新入口用原始价格口径换算后，与旧入口 IC 可对照（验证数据流一致，而非口径一致）；(c) loss 必须可对齐（loss 不受 IC 口径影响）；(d) early stopping 停止 epoch **不要求与旧训练一致**（口径变更改变 patience 累积节奏，见 §1.6.7），改为 loss 收敛曲线形态可比。
   - 见 §8.3 详细步骤。

### Phase 5: 切换默认入口

1. 文档更新为新 CLI
2. CI/checklist 使用新入口
3. 旧入口标记 legacy（保留在 git）

### Phase 6: 归档旧代码

**前置条件**: 新链路已验证通过（Phase 5 complete）

**执行步骤**:

1. **打 git tag 保留 provenance**:
   ```bash
   git tag pre-refactor-archive -m "Snapshot before deprecated migration"
   git push origin pre-refactor-archive
   ```
   此 tag 记录旧代码在 git 中的最后状态，日后排查新代码 bug 时可追溯参照。

2. **移动旧代码**:
   - 移动 `mode1~10/`、`shared/`、`simple_backtest.py` 等到 `deprecated/`
   - `deprecated/` 不纳入 git（用户要求）
   - **注意**: `git mv` 进 gitignored 目录会将文件从版本控制中删除；确认已在 tag 中保存

3. **确认新链路运行**:
   ```bash
   python finetune/predictor/train.py --norm-mode sliding_ma60 --model mini --epochs 1
   python finetune/predictor/eval.py --norm-mode sliding_ma60 --model mini --n-samples 100
   ```

---

## 八、验证清单

### 8.1 数据验证

- [ ] `min_samples == lookback + predict`（无 +1）
- [ ] 每个样本记录完整 target 区间
- [ ] train/val/test target 无重叠
- [ ] block split 以 target block 为单位
- [ ] backtest `context_end < target_start`
- [ ] normalizer 不使用 target 区间数据
- [ ] validate_no_leakage 通过

### 8.2 Tokenizer 一致性

- [ ] tokenizer.norm_mode == predictor.norm_mode
- [ ] tokenizer.data_fingerprint == predictor.data_fingerprint
- [ ] checkpoint config 记录 tokenizer path

### 8.3 等价性验证

**容差设定原则**:

重构是结构性改动，不涉及算法。同数据、同 norm、同 split、同 seed，结果应近乎一致。宽松容差会掩盖微妙 bug（如 min_periods 差异、split 边界差一步）。

**口径变更下的特殊处理**（关键）：

本次重构同时变更了 IC 度量口径（原始价格 → 去趋势，见 §1.6）。旧入口 baseline 用错误口径，新入口用正确口径，**IC 数值必然不同，不能要求新入口复现旧 IC 数值**。等价性验证标准改为：

1. **loss 可对齐**（loss 不受 IC 口径影响）：新入口 loss 落在 old-vs-old 噪声地板内。
2. **数据流一致性**：新入口用「原始价格口径」换算后算 IC，与旧入口 IC 可对照（验证数据流正确，而非口径一致）。
3. **新入口正确口径自洽**：同 seed 跑两次，去趋势 IC 落在噪声地板内。
4. **去趋势口径换算关系**：Phase -1 已记录的「原始价格口径 IC vs 去趋势口径 IC」差异，新入口符合该换算关系。

**步骤**:

1. **先跑 old-vs-old 测噪声地板**:
   ```bash
   # 固定 seed，跑旧入口两次
   python finetune/predictor/mode2_ma60_t0/train_ddp.py --seed 42 --epochs 5
   # 记录 loss、IC（原始价格口径）
   python finetune/predictor/mode2_ma60_t0/train_ddp.py --seed 42 --epochs 5
   # 计算 old-vs-old 偏差（DDP 非确定性、数据加载顺序等固有方差）
   ```

2. **据此设定容差**:
   - loss 偏差 < 5%（落在噪声地板内）
   - IC 不设「复现旧值」容差，改用上述口径变更四标准

3. **任何超出容差的偏差必须 root-cause**，不得"在容差内就放过"

**验证清单**:

- [ ] old-vs-old 噪声地板已测量（loss）
- [ ] 旧入口 baseline 已保存（含原始价格口径 IC）
- [ ] Phase -1 口径换算关系已记录（原始价格 IC vs 去趋势 IC）
- [ ] 新入口 loss 可对齐（在噪声地板内）
- [ ] 新入口原始价格口径 IC 与旧入口可对照（数据流一致）
- [ ] 新入口去趋势口径 IC 自洽（同 seed 两次落在噪声地板内）
- [ ] trajectory IC 使用去趋势序列计算（detrend_to_baseline，禁止原始价格）
- [ ] 超容差偏差已 root-cause

### 8.4 迁移验证

- [ ] Phase 0-5 旧入口未被移动
- [ ] Phase 6 前新链路已验证可用
- [ ] deprecated 移动后不影响新链路

---

## 九、废弃代码迁移（Phase 6）

| 原路径 | 移入 deprecated/ |
|--------|------------------|
| `finetune/predictor/mode1_original/` | `deprecated/finetune/predictor/mode1_original/` |
| `finetune/predictor/mode2_ma60_t0/` | `deprecated/finetune/predictor/mode2_ma60_t0/` |
| `finetune/predictor/mode3~mode10/` | `deprecated/finetune/predictor/mode3~mode10/` |
| `finetune/predictor/shared/` | `deprecated/finetune/predictor/shared/` |
| `finetune/predictor/simple_backtest.py` | `deprecated/finetune/predictor/simple_backtest.py` |
| `finetune/batch_inference*.py` | `deprecated/finetune/batch_inference*.py` |
| `finetune/check_*.py` | `deprecated/finetune/check_*.py` |

**保留代码**（不移入 deprecated）：
- `finetune/data/` - 数据目录
- `finetune/preprocess/` - 预处理脚本（新建）

---

## 十、训练信息实时记录

训练开始时创建 `training_info.json`，每个 epoch 完成时更新：

```python
# train.py 中

def init_training_info(output_dir, config):
    """训练开始时创建 info 文件"""
    info = {
        "status": "running",
        "start_time": datetime.now().isoformat(),
        "config": {
            "norm_mode": config.norm_mode,
            "lookback": config.lookback,
            "predict": config.predict,
            "split_mode": config.split_mode,
            "model_type": config.model_type,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
        },
        "data": {
            "train_samples": 0,
            "val_samples": 0,
            "data_fingerprint": "待更新",
            "target_leakage_check": "待更新",  # Phase 3 validate_no_leakage 结果
        },
        "tokenizer": {
            "path": config.tokenizer_path,
            "data_fingerprint": "待更新",
        },
        "epochs": [],
        "best": {
            "ic": 0.0,
            "combined": 0.0,
            "epoch": 0,
        },
    }
    
    info_path = os.path.join(output_dir, 'training_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    return info_path


def update_training_info(info_path, epoch, metrics, best_updates=None):
    """
    每个 epoch 完成时更新 training_info

    Args:
        metrics: 本 epoch 的全部指标（含 da_score、excess_da、可懂指标，可懂指标在非 best 时为 None）
        best_updates: 本 epoch 刷新的 best 列表，如
            {'val_loss': (2.28, 7), 'ic': (0.2105, 7), 'combined': (0.478, 7)}
            可懂指标（amplitude_error_rate, limit_hit_rate）仅在某 best 刷新时才填入
    """
    with open(info_path, 'r', encoding='utf-8') as f:
        info = json.load(f)

    # epoch 记录：所有关键指标并列（不再 IC 主导）
    epoch_record = {
        "epoch": epoch,
        "train_loss": metrics['train_loss'],
        "val_loss": metrics['val_loss'],
        "ic": metrics['ic'],
        "da_score": metrics['da_score'],
        "excess_da": metrics['excess_da'],
        "combined": metrics['combined'],
        "lr": metrics['lr'],
        # 可懂指标：仅本 epoch 产生 best 时才有值，否则 None（平凡 epoch 不算，省时）
        "amplitude_error_rate": metrics.get('amplitude_error_rate'),  # None 或 {mean, std, usable_pct}
        "limit_hit_rate": metrics.get('limit_hit_rate'),              # None 或 {hit_rate, n_pred, n_actual}
        "time": datetime.now().isoformat(),
    }
    info['epochs'].append(epoch_record)

    # 更新 best：每个关键指标独立记 best 值 + best epoch（checkpoint X：只记录，不存独立 ckpt）
    if best_updates:
        for metric_name, (value, best_epoch) in best_updates.items():
            info['best'][metric_name] = {"value": value, "epoch": best_epoch}

    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)
```

**epoch 打印（并列显示 current + best，不再 IC 主导）**：
```
Epoch 12/50  LR: 0.003000
  IC:        current 0.1823  | best 0.2105 @ep8
  DA_score:  current 0.5410  | best 0.5520 @ep10
  Excess DA: current +4.1%   | best +6.2% @ep10
  Combined:  current 0.4521  | best 0.4780 @ep8
  Val_loss:  current 2.31    | best 2.28 @ep7
  Train: 2.35, Time: 03:12
```

**best 跟踪与可懂指标三件套的设计（ABCD + checkpoint X）**：

- **A. epoch 打印并列**：各关键指标并列显示 `current | best @ep`，不以 IC 为主标题。
- **B. best 跟踪扩展**：每个关键指标（val_loss / ic / da_score / excess_da / combined）独立记 `best.{metric} = {value, epoch}`，写入 training_info 的 `best` 块。
- **C. checkpoint 不泛滥（X）**：仍只存 3 个 checkpoint——`best_model`(val_loss) / `best_ic_model`(ic) / `best_combined_model`(combined)。da_score / excess_da / 可懂指标的 best **只记录值+epoch，不存独立 checkpoint**；需取某指标 best 时的模型，查 training_info 该指标 best epoch，对应取该 epoch 的 checkpoint。
- **D. 可懂指标三件套只在产生 best 时算**：振幅误差率 + 涨跌停命中率 在本 epoch 刷新任一 best（val_loss / ic / combined）时才收集计算，记入该 best 的 epoch_record 与 `best` 块；平凡 epoch 不算（省时，无额外 forward，pred_raw 已有，仅多两个 numpy 运算）。方向胜率 = DA（已有，不算额外）。

**training_info.json 结构**：
```json
{
  "status": "running",
  "start_time": "2026-06-19T10:00:00",
  "config": {
    "norm_mode": "sliding_ma60",
    "lookback": 400,
    "predict": 10,
    "split_mode": "block",
    "model_type": "mini",
    "epochs": 50,
    "batch_size": 16,
    "learning_rate": 0.01
  },
  "data": {
    "train_samples": 50000,
    "val_samples": 10000,
    "data_fingerprint": "sha256:abc123",
    "target_leakage_check": "passed"
  },
  "tokenizer": {
    "path": "outputs/tokenizers/sliding_ma60/mini",
    "data_fingerprint": "sha256:def456"
  },
  "epochs": [
    {
      "epoch": 1,
      "train_loss": 2.5,
      "val_loss": 2.3,
      "ic": 0.05,
      "da_score": 0.51,
      "excess_da": 0.01,
      "combined": 0.03,
      "lr": 0.01,
      "amplitude_error_rate": null,
      "limit_hit_rate": null,
      "time": "2026-06-19T10:05:00"
    },
    {
      "epoch": 8,
      "train_loss": 2.1,
      "val_loss": 2.28,
      "ic": 0.2105,
      "da_score": 0.55,
      "excess_da": 0.05,
      "combined": 0.478,
      "lr": 0.006,
      "amplitude_error_rate": {"mean_rate": 1.05, "std_rate": 0.18, "usable_pct": 0.72},
      "limit_hit_rate": {"hit_rate": 0.12, "n_pred_limit": 340, "n_actual_limit": 280},
      "time": "2026-06-19T10:40:00"
    }
  ],
  "best": {
    "val_loss": {"value": 2.28, "epoch": 7},
    "ic": {"value": 0.2105, "epoch": 8},
    "da_score": {"value": 0.5520, "epoch": 10},
    "excess_da": {"value": 0.062, "epoch": 10},
    "combined": {"value": 0.4780, "epoch": 8},
    "amplitude_error_rate": {"value": 1.05, "epoch": 8},
    "limit_hit_rate": {"value": 0.12, "epoch": 8}
  }
}
```

训练结束时更新 status 为 "completed" 或 "early_stopped"。

> **对代码现状的说明**：现有 train.py 的 `update_training_info`（行 152-184）已是 9 字段（含 da_score、excess_da），但 `best` 块仍只跟 ic/combined（旧版），未扩展到 da_score/excess_da/可懂指标；epoch 打印仍 IC 主导（行 629-630）；可懂指标三件套未在 train 路径收集。本节为待实现目标，开发人员按上述 ABCD + X 改造 train.py 的 best 跟踪与 epoch 输出。

---

## 十一、检查清单

- [ ] Phase -1: 度量口径定稿（去趋势 traj IC、excess DA、聚合 std/分位数、backtest IC 不混算、可懂指标三件套、combined_score 权重重标定）
- [ ] Phase -1: early stopping 审视（patience 归零条件 + 阈值，见 §1.6.7）
- [ ] Phase -1: train.py 与 eval.py 共用 core/metrics.py 同一函数（口径一致）
- [ ] Phase 0: baseline 已保存（IC 用正确口径重算）
- [ ] Phase 1: 数据契约定义完成
- [ ] Phase 2: 核心模块编写完成（含 S2 ArtifactConfig.model_type）
- [ ] Phase 3: Dataset/Preprocess 验证通过
- [ ] Phase 4: 新入口等价性验证通过（口径变更四标准，见 §8.3）
- [ ] Phase 5: 文档切换到新 CLI
- [ ] Phase 6: 废弃代码移入 deprecated/
- [ ] 最终: 新链路可运行，旧代码已清理

**度量口径验证**（贯穿 Phase -1 至 Phase 4）:
- [ ] trajectory IC 用去趋势序列（detrend_to_baseline），禁止原始价格
- [ ] checkpoint 选择 IC 与 eval 报告 IC 同口径
- [ ] aggregate 保留 mean + std + p25/p50/p75
- [ ] backtest IC 逐股票算再聚合，不跨股票混算
- [ ] eval 主输出含 excess DA
- [ ] eval 主输出含可懂指标三件套（方向胜率/振幅误差率/涨跌停命中率）
- [ ] combined_score 权重用正确口径重标定并记录