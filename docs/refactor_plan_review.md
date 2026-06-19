# 重构方案评审报告（v2 复审）

**复审日期**: 2026-06-19  
**评审对象**: `docs/refactor_plan.md`（修订版）  
**评审结论**: ⚠️ 大幅改善，但存在 1 个关键残留必须在执行前补上；另有 2 个重要残留需明确

---

## 一、复审总评

修订版对 v1 评审意见响应良好，多数阻断级问题已解决。方案已从“目录搬家”升级为有数据契约、有归一化决策、有分割算法、有等价性验证的重构设计。**方向上可以进入执行前的最后修订。**

但复审发现一个关键问题：**泄露检查与 block 分割仍建立在“窗口元组/索引”语义上，而非“target 时间区间”语义**。本项目历史上正是栽在这里（lb60 window-index splitting 100% target 泄露）。这是执行前必须修正的最后一道关卡。

---

## 二、v1 问题处置追踪

| v1 编号 | 问题 | v2 状态 | 说明 |
|---------|------|---------|------|
| B1 | 迁移顺序错误 | ✅ 已解决 | 新 Phase 0–6 顺序正确，Phase 6 明确“最后执行”，Phase 0–5 不动旧代码 |
| B2 | deprecated/ 不入 git 与迁移冲突 | ⚠️ 部分解决 | 见 R4：仍把已跟踪源码移入 gitignored 目录，provenance 丢失风险残留 |
| B3 | 数据分割算法未定义 | ⚠️ 部分解决 | 骨架已写出（target-based、no-leakage 检查），但检查逻辑有缺陷，见 R1/R2 |
| B4 | 归一化数据流未定型 | ✅ 已解决 | §1.1 明确选择 runtime 归一化，pkl 存原始 OHLCV |
| B5 | norm_mode 命名不统一 | ✅ 已解决 | §1.2 统一为 full_window/sliding_ma{20,60,120} + parse_norm_mode，与现有 dataset.py 兼容 |
| B6 | Tokenizer 训练链路缺失 | ⚠️ 部分解决 | 一致性校验已加，但训练入口仍缺，见 R3 |
| M1 | 路径函数语义混乱 | ✅ 已解决 | §3 拆分为 split_data/backtest/meta 三类函数 |
| M2 | Config 类过粗 | ✅ 已解决 | §4 拆分 DataConfig/TrainConfig/ArtifactConfig，min_samples 不变量已加，LR=0.01 |
| M3 | core/ 职责不清 | ✅ 已解决 | §5 列出 10 个模块及职责 |
| M4 | 回测数据设计不完整 | ✅ 已解决 | §1.4 BacktestSchema + 边界断言 |
| M5 | 输出元数据缺失 | ✅ 已解决 | §1.5 + §10 training_info.json |

**结论**：11 项中 8 项已解决，3 项部分解决。新增内容（§1.6 safe metrics、§10 training_info.json）方向正确。

---

## 三、残留关键问题（执行前必须修正）

### R1. `validate_no_leakage` 检查过弱，无法捕获重叠窗口的 target 区间重叠

**方案位置**: §1.3 No-Leakage 检查。

**问题**: 当前检查对 `(symbol, target_start, target_end)` 元组做集合 disjoint：

```python
train_targets = set((s['symbol'], s['target_start'], s['target_end']) for s in train_samples)
assert train_targets.isdisjoint(val_targets)
```

这只验证“没有两个样本拥有**完全相同**的 target 元组”。但当窗口有重叠（stride < predict，本项目正是如此）时，train 样本 A 的 target = [100,110]，val 样本 B 的 target = [101,111]。两者元组不同 → disjoint 检查通过；但 target **时间区间**在 [101,110] 重叠 → 模型在 train 时已学到 B 要预测的未来。**这正是 lb60 window-index splitting 100% 泄露的机制**。

元组 disjoint 只能抓“完全重复的 target 窗口”，抓不到“时间区间相交”。对本项目的重叠窗口而言，这个检查几乎形同虚设。

**修订要求**: 改为按 symbol 做 target 区间不相交检查：

```python
def validate_no_leakage(train, val, test):
    # 按 symbol 聚合每个 split 的 target 区间
    # 对每对 split，验证同一 symbol 的任意两个 target 区间不相交
    # 区间 [s,e) 不相交: e1 <= s2 或 e2 <= s1
    for s1, s2 in [(train, val), (train, test), (val, test)]:
        for sym in symbols(s1) | symbols(s2):
            intervals_a = [(s['target_start'], s['target_end']) for s in s1 if s['symbol'] == sym]
            intervals_b = [(s['target_start'], s['target_end']) for s in s2 if s['symbol'] == sym]
            assert no_overlap(intervals_a, intervals_b), f"{sym} target overlap across splits"
```

`no_overlap` 可用排序扫描实现。这才是 target-based splitting 的正确守卫。

---

### R2. `block_split` 的 `create_target_blocks` 未定义，正确性完全依赖该函数

**方案位置**: §1.3 Block Split。

**问题**: block_split 的安全性完全押在 `create_target_blocks(sym_samples, block_size=50)` 上，但该函数未定义。两种实现天差地别：

- **若按时间轴分区**：把每只股票的 target 时间轴切成不重叠的 50 长度区间，整块分配给一个 split，块内所有 target 落在该区间的窗口随块走 → 安全。
- **若按样本列表分块**：把样本列表按顺序切 50 个一批 → 退化为 window-index splitting → 泄露（同 lb60 旧 bug）。

方案没说 create_target_blocks 是哪一种。这是 R1 的同源问题：**分割与检查都必须基于 target 时间区间，而非样本元组/索引**。

**修订要求**: 在方案中写明 create_target_blocks 的语义：

```python
def create_target_blocks(sym_samples, block_size):
    """
    1. 取该股票 target 时间轴 [t_min, t_max]。
    2. 按 block_size 切成不重叠的时间区间块。
    3. 每个窗口按其 target 落在哪个时间块，归入该块。
    4. 返回 block 列表，每个 block 是窗口列表。
    关键：块之间 target 时间区间不相交。
    """
```

并加断言：同一股票任意两个 block 的 target 区间不相交。

---

## 四、残留重要问题

### R3. Tokenizer 训练链路仍缺失

**方案位置**: §2.1、§2.2、§6、§1.5。

**问题**: 方案按 `outputs/tokenizers/{norm_mode}/{model_type}` 键控 tokenizer 路径，并要求 `tokenizer.norm_mode == predictor.norm_mode`。这隐含一个强假设：**每个 norm_mode 需要单独训练一个 tokenizer**（因为 tokenizer 编码的是归一化后的值，full_window 与 sliding_ma60 归一化分布不同，tokenizer 不可互换）。

但方案全文没有 tokenizer 训练入口：

- §6 CLI 只有 train/eval/backtest/preprocess，无 train_tokenizer；
- §2.1 `modeling.py` 只有 `load_tokenizer`，无训练；
- `finetune/preprocess/` 无 tokenizer 生成脚本；
- 现有 `finetune/train_tokenizer.py` / `train_tokenizer_ma60_base.py` 的去留未提。

结果：§1.5 的 `validate_tokenizer_consistency` 没有对象可校验——根本没有产生 per-norm_mode tokenizer 的流程。

**修订要求**:

1. 新增 `finetune/predictor/train_tokenizer.py`（或 `finetune/tokenizer/train.py`）入口。
2. 明确 tokenizer 训练用哪份数据、哪个 norm_mode、输出到 `outputs/tokenizers/{norm_mode}/{model_type}`。
3. 在执行计划 Phase 1/2 中补 tokenizer 训练模块的位置。
4. 否则 norm_mode 键控 tokenizer 路径的设计就是空中楼阁。

---

### R4. Phase 6 仍把已跟踪源码移入 gitignored `deprecated/`，丢失 provenance

**方案位置**: §2.1、§2.3、§7 Phase 6、§9。

**问题**: 用户要求 `deprecated/` 不入 git，方案照办。但 Phase 6 仍把 `mode1~10/`、`shared/`、`simple_backtest.py` 等已 git 跟踪的源码移动进去。一旦移入 gitignored 目录并提交，这些源码从版本控制中消失——等价于删除唯一可对照的旧实现。新 `core/` 是从这些旧代码抽取的，日后排查新代码 bug 时无 git 可追溯的参照。

**修订要求**: 尊重用户“不入 git”决定，但补一个 provenance 保险：

1. **Phase 6 执行前**打 git tag 或分支 `pre-refactor-archive`，记录旧代码在 git 中的最后状态。
2. 明确 Phase 6 的 git 操作机制（`git rm` vs `git mv` 进 ignore 目录的行为不同），避免误操作。
3. 或：Phase 6 只移动确实不再需要的，`shared/` 中被 core/ 直接抽取的关键文件先保留到新链路稳定一段时间再归档。

---

### R5. 等价性容差 ±10% IC 过松，可能掩盖回归

**方案位置**: §8.3。

**问题**: 重构是结构性改动，不涉及算法。新 pipeline 同样是 runtime 归一化（与旧 `dataset.py` 一致），同数据、同 norm、同 split、同 seed，结果应**近乎一致**。±10% IC 容差足以让一个有微妙 bug 的重构（如 R7 的 min_periods 差异、或 split 边界差一步）通过验证。本项目历史上泄露曾制造虚假高 IC，宽松容差会再次掩盖问题。

**修订要求**:

1. 先跑 **old-vs-old**（旧入口跑两次，固定 seed）测出噪声地板（DDP 非确定性、数据加载顺序等造成的固有方差）。
2. 据此设定容差，默认收紧到 IC 偏差 < 2% 或落在噪声地板内。
3. 任何超出容差的偏差必须 root-cause，不得“在容差内就放过”。

---

### R6. 新 `eval.py` 功能范围未定义，可能丢失现有评估能力

**方案位置**: §6.2。

**问题**: 现有 `shared/eval_ddp.py` 支持：多模型批量评估（`--models`）、DDP 分布式评估、指定 checkpoint（`--best_combined_model`）、分层评估。新 `eval.py` CLI 只有 `--norm-mode/--lookback/--split-mode/--model/--n-samples`，未提：

- 是否支持 DDP 评估（评估计算量大，现有专门做了 eval_ddp）；
- 是否支持多模型批量对比；
- 是否支持分层（large/mid/small）评估；
- checkpoint 选择（best_ic vs best_combined）。

若新 eval.py 是单进程单模型，相对现有是能力回归。

**修订要求**: 明确新 eval.py 的功能边界，至少声明 DDP 评估与分层评估是否保留；若保留，CLI 与 core 模块需相应设计。

---

## 五、残留次要问题

### R7. SlidingMANormalizer 边界行为（min_periods）未指定

**方案位置**: §1.1、§4。

现有 `dataset.py:185` 用 `rolling(window=ma_window, min_periods=1)`，即窗口前若干点用 expanding window（数据不足时逐步扩大）。这使 lookback 窗口前几个点的归一化统计量与“完整 60 步 MA”不同。新 `SlidingMANormalizer` 必须逐字复刻此行为，否则新旧数据输出不一致，R5 等价性必失败。方案未指定 min_periods。**要求**：normalizer 显式声明 min_periods=1 并加注释说明这是与旧实现的对齐点。

### R8. Tokenizer 路径键不一致

**方案位置**: §2.2 vs §4 `resolve_tokenizer_path`。

§2.2 目录与 `get_tokenizer_path(norm_mode, model_type)` 用 `mini/small/base`；但 `resolve_tokenizer_path` 返回 `outputs/tokenizers/{norm_mode}/2k`（mini→2k，其余→base）。同一 tokenizer 路径两种键。**要求**：统一为 model_type 键，或在目录结构中明确 2k/base 子目录并让 get_tokenizer_path 与之一致。

### R9. `processed/` 按 norm_mode 键控 与 runtime 归一化存在张力

**方案位置**: §1.1 vs §2.1。

runtime 归一化意味着 pkl 存原始 OHLCV，原始窗口本身与 norm_mode 无关。但 `processed/{norm_mode}/...` 仍按 norm_mode 分目录，会导致同一批原始窗口在不同 norm_mode 下重复存储，或 norm_mode 键形同虚设。可能的正当理由：sliding_ma{N} 需要窗口起点前有 N 步历史，不同 N 的可用样本集不同 → 样本集随 norm_mode 变化。**要求**：方案明确说明这一点，否则去掉 processed/ 上的 norm_mode 键，改为只按 lookback/predict/split_mode 键控。

### R10. §1.6 “受影响文件”包含即将 deprecated 的 mode* 文件

**方案位置**: §1.6。

列表含 `mode*/train.py`、`mode*/eval.py`、`shared/eval*.py` 等，但这些在 Phase 6 会被移入 deprecated。改它们是浪费，且与 Phase 6 矛盾。**要求**：§1.6 的替换范围只限新文件（`core/metrics.py`、新 `train.py/eval.py/backtest.py`），旧文件的 corrcoef 调用随 deprecated 自然失效。

### R11. `DataConfig.validate()` 定义但从未调用

**方案位置**: §4。

`__post_init__` 设了 min_samples，但没调 validate()；validate() 含 norm_mode 白名单检查却无人触发。**要求**：在 `__post_init__` 末尾调用 `self.validate()`，否则校验形同虚设。

### R12. 数据迁移策略缺失，raw 路径不一致

**方案位置**: §2.1 vs §2.3 vs §7 Phase 3。

- §2.1 写 `finetune/data/raw/kline_daily.pkl`；§2.3 写根 `data/kline_daily_raw.pkl`。raw 位置自相矛盾。
- 现有 `block_lb400_pd10` 有约 1.5GB 旧格式 pkl（train/val/test/final_test）。Phase 3 “用小样本验证新旧一致”未说明：是从 raw 全量重新生成新 schema 数据，还是转换现有 pkl？结合 no-deletion 规则与数据量，必须明确。

**要求**：(1) 统一 raw 路径；(2) 明确现有大数据集的去留与迁移方式。

---

## 六、新内容评价

| 新内容 | 评价 |
|--------|------|
| §1.6 safe_corrcoef/safe_spearmanr/safe_trajectory_ic | ✅ 方向正确，但实为现有 `shared/utils.py` 的整合（`calc_trajectory_ic` 被改名 `safe_trajectory_ic`）。建议直接迁移而非重写，并同步现有调用点 |
| §10 training_info.json | ✅ 符合监控原则，结构合理。建议补 `data.target_leakage_check: passed` 字段，把 R1 的检查结果记入 |
| §1.4 BacktestSchema | ✅ context/target 分离、actual_target 不入模型的设计正确 |

---

## 七、执行前必须完成的最小修订集

按优先级排序，完成这几项后方案可进入执行：

1. **[关键] R1 + R2**：泄露检查与 block 分割改为 target 时间区间语义。这是项目历史教训的防线，不可跳过。
2. **[重要] R3**：补 tokenizer 训练入口，否则 norm_mode 键控 tokenizer 无意义。
3. **[重要] R4**：Phase 6 前打 `pre-refactor-archive` tag 保留 provenance。
4. **[重要] R5**：等价性容差改为基于 old-vs-old 噪声地板的 data-driven 容差。
5. **[重要] R6**：明确新 eval.py 是否保留 DDP/分层评估能力。
6. **[次要] R7/R8/R11**：normalizer min_periods、tokenizer 路径键、validate 调用——影响等价性与一致性，顺手修正。
7. **[次要] R9/R10/R12**：文档一致性与迁移策略澄清。

---

## 八、二次评审记录

**评审日期**: 2026-06-19 (第二轮)
**评审人**: Claude Code（最终负责整合）

### 8.1 审核报告质量评价

**状态**: ✅ 审核报告质量达标，可作为修订依据

本报告经历两轮：v1 用 B(阻断级)/M(重要级) 框架提出 11 个问题；v2 复审确认其中 8 个已解决，并用 R1–R12（关键/重要/次要）框架记录残留问题。当前正文为 v2 复审内容，本节及第九节为版本追踪。

| 方面 | 评价 |
|------|------|
| 问题分级 | v1 的 B/M 与 v2 的 R 分级双轨清晰，优先级可追溯 |
| 历史教训融合 | 正确引用 feedback 中的泄露、IC计算、tokenizer 一致性、min_samples 不变量等教训 |
| 建议可执行 | 每个问题都有具体代码/架构修订建议，不只是批评 |
| 新执行计划 | Phase 0–6 解决了原方案的执行顺序根本错误 |
| 验证清单 | 数据/Tokenizer/训练/迁移四维度覆盖完整 |

### 8.2 待改进点

**1. 报告缺少决策摘要**

建议在开头增加「修订决策表」，一眼看出哪些章节需重写：

| 原章节 | 是否需重写 | 关键改动 |
|--------|------------|----------|
| 1 目录结构 | 否 | 补充 core/ 各模块职责 |
| 2 命名规则 | 是 | 统一 norm_mode（full_window/sliding_ma{N}） |
| 3 数据分割 | 是 | 写出 target-based 算法 + 区间不相交检查 |
| 6 迁移 | 是 | 改为最后执行，保留 provenance |
| 7 执行步骤 | 是 | 改为 Phase 0–6 顺序 |

**2. Phase 0 基线需更具体**

当前写「选定最小可运行配置」，但未明确：

- 用哪个旧入口作为 baseline（`finetune/predictor/mode2_ma60_t0/train_ddp.py`）
- 如何记录 baseline（loss 值、IC 值、checkpoint 路径、training_info）
- 对齐容差（建议先跑 old-vs-old 测噪声地板，见 R5）

**3. 残留问题需在执行前定稿**

R1/R2（泄露检查与 block 分割的 target 区间语义）是执行前必须解决的最后一关，不能等到 Phase 3 再处理——应在方案文档中先把算法和检查函数定稿，再进入 Phase 0。

### 8.3 后续行动建议

1. 将此报告同步给方案修订人
2. 按报告建议的 Phase 0–6 顺序修订 `docs/refactor_plan.md`
3. 优先修订 R1/R2/R3，其余可并行
4. 修订完成后重新提交审核

---

## 九、审核记录

| 版本 | 日期 | 评审结果 | 主要变动 |
|------|------|----------|----------|
| v1 | 2026-06-19 | ⚠️ 需修订 | 初版系统性评审，B1–B6 阻断级 + M1–M5 重要级，共 11 个问题 |
| v2 | 2026-06-19 | ⚠️ 大幅改善，需补关键残留 | 复审确认 8 项已解决；残留 R1/R2（关键，区间语义）+ R3–R6（重要）+ R7–R12（次要） |
| v3 | 2026-06-19 | ✅ 报告达标 | 二次评审确认报告质量，补决策摘要、Phase 0 细化、版本追踪 |

> 注：v2 评审结论为「⚠️ 大幅改善，但存在 1 个关键残留必须在执行前补上」——R1（泄露检查过弱）与 R2（create_target_blocks 未定义）。

---

## 十、结论

修订版质量显著提升，v1 的 11 项问题已解决 8 项。**剩余关键风险集中在数据分割的区间语义（R1/R2）**——这恰好是项目最敏感、曾实际踩坑的领域，建议在动任何代码前先把这两项的算法和检查函数定稿。

完成第七节的最小修订集后，方案可进入 Phase 0 执行。

---

## 十一、v3 复审记录（修订版复核）

**复核日期**: 2026-06-19
**复核对象**: `docs/refactor_plan.md`（修订版，响应 v2 复审意见）
**复核结论**: ✅ 所有阻断级和重要问题均已解决，方案可进入执行阶段

### 11.1 问题处置追踪表

| 编号 | 问题 | 修订状态 | 方案位置 |
|------|------|----------|----------|
| **R1** | validate_no_leakage 检查过弱 | ✅ 已解决 | §1.3 改为区间相交检查（intervals_overlap），不再用元组 disjoint |
| **R2** | create_target_blocks 未定义 | ✅ 已解决 | §1.3 写明算法：按 target 时间轴切块，断言块间不相交 |
| **R3** | Tokenizer 训练链路缺失 | ✅ 已解决 | §1.5 新增 train_tokenizer 函数 + CLI；Phase 2 包含 tokenizer 模块 |
| **R4** | deprecated provenance 丢失 | ✅ 已解决 | Phase 6 前打 git tag `pre-refactor-archive`，保留追溯参照 |
| **R5** | 等价性容差过松 | ✅ 已解决 | §8.3：先跑 old-vs-old 测噪声地板，收紧到 IC < 2%，超容差必须 root-cause |
| **R6** | eval.py 功能范围未定义 | ✅ 已解决 | §6.2 明确：DDP、多模型批量、checkpoint 选择、分层评估 |
| **R7** | min_periods 未指定 | ✅ 已解决 | §1.2 明确 `min_periods=1` 与现有 dataset.py:185 对齐 |
| **R8** | Tokenizer 路径键不一致 | ✅ 已解决 | §2.2 统一使用 model_type（mini/small/base）作为键 |
| **R9** | norm_mode 键控张力 | ✅ 已解决 | §2.1 说明：sliding_ma{N} 需窗口前 N 步历史，样本集随 norm_mode 变化 |
| **R10** | §1.6 包含 deprecated 文件 | ✅ 已解决 | 已说明旧入口随 Phase 6 deprecated 自然失效，无需逐个修改 |
| **R11** | validate() 未调用 | ✅ 已解决 | §4 DataConfig `__post_init__` 末尾调用 `self.validate()` |
| **R12** | raw 路径不一致 | ✅ 已解决 | §2.3 统一为 `data/kline_daily_raw.pkl`，说明迁移策略 |

**结论**：v2 复审提出的全部 12 个问题均已解决。

### 11.2 亮点改进

1. **§1.3 create_target_blocks** 算法完整，包含断言验证块间不相交
2. **§1.5 Tokenizer 训练** 设计合理，含 fingerprint 校验与 data_fingerprint 记录
3. **Phase 6 provenance 保留** 通过 git tag `pre-refactor-archive` 解决追溯问题
4. **§8.3 等价性验证** 采用 old-vs-old 噪声地板法，科学严谨
5. **§6.2 eval.py** 功能继承完整（DDP、多模型、分层），无能力回归

### 11.3 建议追加的内容（非阻断）

以下两项不影响执行，但建议在 Phase 1/2 实现时补充：

#### S1. §1.3 create_target_blocks 边界处理

当前算法假设 `block_size` 固定，但 target 时间轴末端可能不满 block_size：

```python
# 建议在实现时补充
while current_block_start < t_max:
    current_block_end = min(current_block_start + block_size, t_max)  # 防止超界
```

方案中的写法 `current_block_end = current_block_start + block_size` 在最后一块可能超出 `t_max`，导致空块或遗漏末端样本。

**建议**: Phase 2 实现 `splitting.py` 时修正。

#### S2. §4 ArtifactConfig.model_type 字段缺失

`ArtifactConfig.resolve_tokenizer_path` 使用了 `self.model_type`，但类定义未声明该字段：

```python
# 当前定义（§4）
@dataclass  
class ArtifactConfig:
    """模型/Tokenizer 路径"""
    pretrained_model_path: str = 'pretrained/Kronos-mini'
    tokenizer_path: str = None
    output_dir: str = None
    
    def resolve_tokenizer_path(self, data_config: DataConfig):
        # 使用了 self.model_type，但未定义
        return f"outputs/tokenizers/{data_config.norm_mode}/{self.model_type}"
```

**建议修正**:

```python
@dataclass  
class ArtifactConfig:
    """模型/Tokenizer 路径"""
    model_type: str = 'mini'  # 新增
    pretrained_model_path: str = 'pretrained/Kronos-mini'
    tokenizer_path: str = None
    output_dir: str = None
```

或在 `resolve_tokenizer_path` 中显式传入 model_type 参数。

**建议**: Phase 2 实现 `config.py` 时修正。

### 11.4 执行批准

修订版方案已解决审核报告 v2 复审提出的全部 12 个问题。方案设计完整、执行顺序正确（Phase 0–6）、验证机制健全（等价性、泄露检查、tokenizer 一致性）。

**批准进入执行阶段**。建议在 Phase 1/2 实现核心模块时顺手修正 S1/S2。

---

## 十二、审核记录（完整）

| 版本 | 日期 | 评审结果 | 主要变动 |
|------|------|----------|----------|
| v1 | 2026-06-19 | ⚠️ 需修订 | 初版系统性评审，B1–B6 阻断级 + M1–M5 重要级，共 11 个问题 |
| v2 | 2026-06-19 | ⚠️ 大幅改善，需补关键残留 | 复审确认 8 项已解决；残留 R1/R2（关键）+ R3–R6（重要）+ R7–R12（次要） |
| v3 | 2026-06-19 | ✅ 报告达标 | 二次评审确认报告质量，补决策摘要、版本追踪 |
| v4 | 2026-06-19 | ✅ 执行批准 | 复核修订版，全部 12 问题已解决，追加 S1/S2 非阻断建议 |
| v5 | 2026-06-19 | ⚠️ 有条件批准 | 外部度量口径意见核实：E1/E8（关键，影响 baseline 与等价性基准）+ E2/E4（重要）+ E5/E7（可选/次要）；§6.2 分层评估实为新建非继承；建议插入 Phase -1 度量口径定稿。**追加核实：E1/E2/E4 在训练脚本 `train_ddp.py` 同源存在，E1 驱动 checkpoint 选择与 early stopping，严重性升级；`core/metrics.py` 必须一次性修 train+eval 两端** |

---

## 十三、最终结论

`docs/refactor_plan.md` 修订版已完成全部架构评审意见（R1–R12）的修订，架构质量达标。

**但 v5 外部度量口径核实发现 E1/E8 关键问题**（见 §14）：trajectory IC 用原始价格序列、backtest IC 跨股票混算，二者口径错误会污染 Phase 0 baseline 与 Phase 4 等价性基准。因此执行批准从「无条件」修正为「**有条件**」。

执行顺序：
1. **Phase -1（新增）：度量口径定稿** — 定义 trajectory IC 去趋势口径、backtest IC 聚合口径、excess DA、可懂指标三件套（§14.6）
2. Phase 0 用正确口径保存 baseline
3. Phase 1/2 实现 S1/S2 非阻断修正 + E2/E4 度量可解读性
4. Phase 3 严格运行 validate_no_leakage
5. Phase 4 等价性验证按 §8.3 old-vs-old 噪声地板法执行（基准为正确口径）

执行时注意：
1. §6.2 分层评估为新建（旧 eval_ddp 无此能力），非继承
2. 可懂指标（方向胜率 / 振幅误差率 / 涨跌停命中率）纳入 eval 主输出，契合项目可懂性目标

---

## 十四、外部度量口径意见核实（v5）

**核实日期**: 2026-06-19
**意见来源**: 外部审核（针对评估脚本 `eval_ddp.py` / `simple_backtest.py` 的度量口径）
**核实方式**: 逐条对照代码行号验证，确认成立性

> 边界声明：本节意见均限于「度量计算是否正确 / 可解读 / 口径是否合理」，不涉及策略与交易系统职责（遵循 eval scope 边界）。数据独立性前提已由用户确认，本节不涉及泄露验证。

### 14.1 意见核实表

| 编号 | 意见 | 代码依据 | 核实结论 | 优先级 |
|------|------|----------|----------|--------|
| E1 | Trajectory IC 用原始价格序列算，口径错误 | `eval_ddp.py:226-241` `traj_ic = np.corrcoef(pred_traj, actual_traj)`，`pred_traj` 来自 `pred_raw`（反归一化原始价格） | ✅ **成立** | 🔴 关键 |
| E2 | DA 基线是 lookback 最后一根，未报 excess DA | `eval_ddp.py:246-248` `baseline[fi]` 为 lookback 末值，无 naive baseline 对照 | ✅ 成立 | 🟡 重要 |
| E4 | 聚合丢分布，只传 mean×n | `eval_ddp.py:256-332` 仅 `local_ic_mean * n_local`，无 std/分位数 | ✅ 成立 | 🟡 重要 |
| E5 | eval 缺 cross-sectional Rank IC | 项目目标为轨迹预测，非选股 alpha | ✅ 降级为可选 | ⚪ 可选 |
| E7 | sigmoid 魔数 0.084/21 硬编码无注释 | `simple_backtest.py:46-47` `SIGNAL_CENTER=0.084` / `SIGNAL_STEEPNESS=21` | ✅ 成立 | ⚪ 次要 |
| E8 | backtest IC 跨股票跨时间混算 | `simple_backtest.py:464-480` `pred_gains`/`actual_gains` 跨股票聚合算单一 IC | ✅ **成立** | 🔴 关键 |

### 14.2 关键发现：E1 违反既有设计意图

E1 不仅是度量口径错误，更是**对 trajectory IC 设计意图的实现偏差**。

项目 memory（`feedback_trajectory_ic.md`）明确：trajectory IC 的设计目标是「同一股票内，预测轨迹**形状**与实际轨迹**形状**的相似度」。但 `eval_ddp.py:227` 用 `pred_raw`（原始价格）算相关——原始价格序列强自相关，即使预测完全持平（形状为常数），也会因与实际价格趋势同向而拿到高 IC。

**影响**：当前报出的 trajectory IC（如 0.21）被价格趋势成分夸大，不反映「轨迹形状预测能力」。正确口径应为对**去趋势序列**（收益率 / 一阶差分 / 对 lookback 末值归一化后的相对序列）算相关，才能测形状。

**与方案的关系**：方案 §1.6 的 `safe_trajectory_ic` 仍接收 `pred_traj`/`actual_traj`，未规定输入是原始价格还是去趋势序列。重构后若直接迁移，**E1 会原样带入新 `core/metrics.py`**。

### 14.3 E1/E8 对「执行批准」前提的影响

v4（§11.4）已给出「执行批准」。但 E1/E8 暴露的是**度量基准本身的口径错误**，会侵蚀执行批准的两个前提：

1. **Phase 0 baseline**：若 baseline 的 trajectory IC 用错误口径记录，则 baseline 是「被夸大的数字」，无对照价值。
2. **Phase 4 等价性验证**：§8.3 的 old-vs-old 噪声地板法对齐的是「IC 数值」，若 IC 口径错误，等价性验证变成「对齐一个错误基准」——新代码复现了错误口径，反而判为通过。

**结论**：执行批准应从「无条件批准」修正为「**有条件批准**」——条件是：在 Phase 0 之前（或至少 Phase 0 内）先定稿 E1/E8 的度量口径，使 baseline 与等价性基准建立在正确口径上。

### 14.4 方案 §6.2 「继承」措辞核实

方案 §6.2 称 eval.py「继承现有 `shared/eval_ddp.py` 能力」，列出 DDP / 多模型批量 / checkpoint 选择 / **分层评估 `--strata large,mid,small`**。

核实：现有 `eval_ddp.py`（`evaluate_model_ddp`，行 335+）**无分层评估实现**——只有按 model 逐个评估、采样、聚合。分层（large/mid/small）在现有 eval 链路中不存在。

**问题**：§6.2 用「继承」一词暗示该能力已存在于旧代码、新代码照搬即可，但实际需新建。这是 R6「功能范围未定义」的残留——方案给了 CLI 形状，但未说明分层评估是「迁移」还是「新建」。

**建议**：§6.2 将「分层评估」标注为「新建（旧 eval_ddp 无此能力）」，避免执行时误以为有参照实现。

### 14.5 可懂指标建议（契合项目目标）

外部建议的主输出三指标与项目目标 memory（`project_understandable_metrics.md`：指标要股市人群可懂、能对标尺）完全一致，且均在 eval 度量边界内（非交易系统职责）：

| 指标 | 含义 | 标尺 | 现状 | 能力维度 |
|------|------|------|------|----------|
| 方向胜率 | 每步涨跌对错 | 50%=随机，55%=微弱，60%+=强 | 已有 DA，埋在 per-step 表 | 方向 |
| 振幅误差率 | 预测振幅÷实际振幅 | 1.0=完美，0.8-1.2=可用 | 缺 | 振幅 |
| 涨跌停命中率 | 预测涨停命中 | 随机≈1-3%，10%+=有信号 | 缺 | 极端事件 |

**评估**：这三项各自对应模型一种能力（方向 / 振幅 / 极端事件），且是股市人群秒懂的标尺。建议方案在 `core/metrics.py` 或 eval 主输出中采纳，IC/RankIC 降为附录给量化人。

**反向警告（采纳）**：涨跌停命中率若低（如 <10%）会刺破「模型能预测」的幻觉——但这正是诚实度量所需。0.21 IC 谁都不懂意味什么，「涨停命中 8%」所有人都懂意味「勉强比随机好」。这一条符合项目「可懂性优先」目标，应纳入。

### 14.6 对方案的修订要求（按优先级）

1. **[关键] E1**：§1.6 `safe_trajectory_ic` 须规定输入为**去趋势序列**（收益率或对 baseline 归一化），禁止用原始价格。在方案中写明口径定义，Phase 0 baseline 即用正确口径。
2. **[关键] E8**：§6.3 backtest 或 `core/metrics.py` 须规定 backtest IC 的聚合口径——是逐股票算再聚合（不混算），还是明确声明 backtest IC 因跨股票混算仅作粗略参考。结合项目目标（轨迹预测，非选股），应评估 backtest IC 是否该作为主输出。
3. **[重要] E2**：eval 主输出补 excess DA（model DA − naive DA），naive baseline = 持平预测或 persistence 预测。让 DA 数字可解读。
4. **[重要] E4**：`aggregate_results` 聚合时保留 std 与分位数（至少 p25/p50/p75），不只传 mean×n。IC 0.21 的稳健性才可判断。
5. **[重要] §6.2 措辞**：分层评估标注为「新建」，不写「继承」。
6. **[可选] E5**：cross-sectional Rank IC 不纳入主输出（项目目标非选股），可留附录。
7. **[次要] E7**：sigmoid 魔数 0.084/21 加注释说明来源，或移入配置。
8. **[建议] 14.5 可懂指标**：eval 主输出采用方向胜率 / 振幅误差率 / 涨跌停命中率三指标，IC/RankIC 降为附录。

### 14.7 修订后的执行批准状态

| 项 | v4 状态 | v5 修正 |
|----|---------|---------|
| 架构（R1–R12） | ✅ 全部解决 | 维持 |
| 度量口径（E1/E8） | 未评审 | 🔴 需在 Phase 0 前定稿 |
| 度量可解读性（E2/E4） | 未评审 | 🟡 Phase 1/2 补 |
| 执行批准 | 无条件批准 | **有条件批准**：先定稿 E1/E8 口径，再建 baseline |

**执行顺序调整建议**：在 Phase 0 之前插入「**Phase -1：度量口径定稿**」——定义 trajectory IC 的去趋势口径、backtest IC 的聚合口径、excess DA、可懂指标三件套。否则 Phase 0 baseline 与 Phase 4 等价性验证均建立在错误基准上。

### 14.8 训练脚本同源问题核实（关键扩展）

**核实动机**：eval 的度量口径问题是否也存在于训练脚本——训练中的 IC/DA 计算若同源错误，则问题不只是「报告失真」，而是「训练决策被错误指标驱动」。

**核实对象**：`finetune/predictor/mode2_ma60_t0/train_ddp.py`（当前主训练入口）+ `finetune/predictor/shared/eval.py`（被训练脚本 import 的指标函数）。

**核实结论：E1/E2/E4 在训练脚本中完全同源存在，且严重性升级。**

| 编号 | eval 中 | 训练脚本中 | 训练脚本位置 | 严重性 |
|------|---------|-----------|--------------|--------|
| E1 | 原始价格算 traj IC | ✅ 同源 | `train_ddp.py:365-378` `pred_raw = pred_norm*std+mean` → `pred_traj=pred_raw[:,fi]` → `corrcoef(pred_traj, actual_traj)` | 🔴 升级 |
| E2 | DA 基线=lookback 末值，无 excess DA | ✅ 同源 | `train_ddp.py:389-390` `baseline=orig_full[lookback-1]`；`shared/eval.py:135/278/406` 统一 `baseline=values[lookback-1]` | 🟡 同级 |
| E4 | 聚合只传 mean×n，丢 std | ✅ 同源 | `train_ddp.py:660-729` `aggregate_ic_results` 只聚合 `sum/n`，无 std/分位数 | 🟡 同级 |

#### E1 在训练中的后果（严重性升级原因）

训练脚本中被夸大的 trajectory IC **不只是报告数字**，它直接驱动两项训练决策：

1. **Checkpoint 选择**（`train_ddp.py:731`）：
   ```python
   close_ic = result.get('close_trajectory_ic', 0)  # E1 错误口径的 IC
   combined_score = calculate_combined_score(close_ic, da_score)  # shared/eval.py:74
   ```
   `calculate_combined_score`（`shared/eval.py:74`）以 `ic_weight=0.6` 把 60% 权重押在 `close_trajectory_ic` 上。即 **best_ic / best_combined checkpoint 的选择主要依据一个被价格趋势夸大的指标**。

2. **Early stopping 判断**：`best_ic`（`train_ddp.py:826`）驱动 patience 计数与停止时机。若 IC 被趋势成分托底在一个虚高水平，early stopping 可能在模型尚未真正学到轨迹形状时就停止，或在趋势巧合时误判为「进步」。

**核心结论**：E1 在 eval 中是「报告失真」，在训练中是「**用错误目标驱动 checkpoint 选择与停止时机**」。后者影响模型本身的质量，远比报告失真严重。重构若只修 eval 不修 train，则训练仍按错误指标选模型。

#### E4 在训练中的后果

`aggregate_ic_results`（`train_ddp.py:660-729`）跨 GPU 只聚合 `sum(local_ic_list)/n`，丢弃 std/分位数。后果：训练监控看到的 IC 是单一均值，无法判断该 IC 是稳健（std 小）还是由少数极端样本撑起（std 大）。被价格趋势托底的虚高 IC 若同时伴随大 std，更难被察觉。

#### E2 在训练中的后果

DA 基线统一为 `lookback-1`（末根 K 线），与 eval 同源。训练中 DA_score 占 combined_score 的 40%，同样无 excess DA，无法判断 DA 是真 alpha 还是「持平朴素预测」的自然水平。

### 14.9 修订后的严重性与批准状态（含训练同源）

| 项 | v5(仅 eval) | v5+训练同源 |
|----|-------------|-------------|
| E1 严重性 | 🔴 关键（报告失真） | 🔴🔴 **关键+**（驱动 checkpoint 选择与 early stopping） |
| E4 严重性 | 🟡 重要 | 🟡 重要（训练监控盲区） |
| E2 严重性 | 🟡 重要 | 🟡 重要（combined_score 40% 权重） |
| Phase -1 范围 | 仅 eval 度量口径 | **eval + train 度量口径同时定稿** |

---

## 十五、v6 复核结论（最终批准）

**复核日期**: 2026-06-19
**复核对象**: `docs/refactor_plan.md`（修订版，响应 v5 度量口径审核意见）
**复核结论**: ✅ 全部问题已解决，无条件批准执行

### 15.1 问题处置追踪表（v5 → v6）

| 编号 | 问题 | 修订状态 | 方案位置 |
|------|------|----------|----------|
| **E1** | Trajectory IC 用原始价格（eval+train 同源） | ✅ 已解决 | §1.6.1 定义 `detrend_to_baseline` + `safe_trajectory_ic`，明确禁止原始价格；§7 Phase -1 定稿；§8.3 等价性验证改口径变更四标准 |
| **E8** | backtest IC 跨股票混算 | ✅ 已解决 | §1.6.4 改为逐股票算再聚合，不作为主输出 |
| **E2** | 无 excess DA | ✅ 已解决 | §1.6.2 定义 `excess_da`，eval 主输出含 model_da + excess_da |
| **E4** | 聚合丢分布 | ✅ 已解决 | §1.6.3 `aggregate_ic` 保留 mean/std/p25/p50/p75 |
| **E5** | 缺 cross-sectional Rank IC | ✅ 已处理 | 降级为可选（项目目标非选股 alpha），不纳入主输出 |
| **E7** | sigmoid 魔数无注释 | ✅ 已解决 | §1.6.4 移入 `BacktestConfig`，注释来源（经验标定） |
| **§6.2 措辞** | 分层评估误标「继承」 | ✅ 已解决 | §6.2 明确标注「新建（旧 eval_ddp 无此能力）」 |
| **Phase -1** | 度量口径需前置定稿 | ✅ 已解决 | §7 新增 Phase -1，明确 train/eval 共用 `core/metrics.py` |
| **可懂指标** | eval 主输出建议 | ✅ 已采纳 | §1.6.5 定义三件套（方向胜率/振幅误差率/涨跌停命中率），§6.2 eval 主输出采用 |
| **combined_score 权重** | 去趋势后需重标定 | ✅ 已解决 | §1.6.6 明确 Phase -1 重新标定权重并记录依据 |
| **等价性验证** | 口径变更下旧基准无效 | ✅ 已解决 | §8.3 重定义验证标准：loss 对齐 + 数据流一致 + 新口径自洽 + 换算关系 |

### 15.2 亮点改进

1. **§1.6 度量口径定稿** 完整解决 train+eval 同源问题，明确禁止原始价格、强制去趋势
2. **Phase -1 前置** 正确识别度量口径是 baseline 与等价性验证的前提
3. **§8.3 口径变更四标准** 科学处理「新口径无法复现旧值」问题，改用数据流一致性验证
4. **可懂指标三件套** 契合项目目标（股市人群可懂），诚实度量优先于学术指标
5. **§1.6.4 BacktestConfig** 魔数配置化并注释来源

### 15.3 执行批准状态

| 项 | v5 状态 | v6 修正 |
|----|---------|---------|
| 架构（R1–R12） | ✅ 全部解决 | 维持 |
| 度量口径（E1/E8） | 🔴 需前置定稿 | ✅ §1.6 + Phase -1 解决 |
| 度量可解读性（E2/E4） | 🟡 Phase 1/2 补 | ✅ §1.6 解决 |
| train 同源问题 | 🔴 未评审 | ✅ §1.6 明确 train/eval 共用 |
| 等价性验证基准 | 🔴 口径变更失效 | ✅ §8.3 重定义四标准 |
| 执行批准 | 有条件批准 | **无条件批准** |

### 15.4 执行顺序确认

1. **Phase -1**: 度量口径定稿（`core/metrics.py` 全部函数 + combined_score 权重标定）
2. **Phase 0**: baseline 保存（IC 用正确口径重算，记录换算关系）
3. **Phase 1**: 数据契约定义
4. **Phase 2**: 核心模块抽取（含 ArtifactConfig.model_type 修正）
5. **Phase 3**: Dataset/Preprocess + validate_no_leakage
6. **Phase 4**: 新入口 + 等价性验证（口径变更四标准）
7. **Phase 5**: 切换默认入口
8. **Phase 6**: 归档旧代码（git tag provenance）

### 15.5 最终批准

`docs/refactor_plan.md` 修订版已解决全部架构问题（R1–R12）与度量口径问题（E1–E8 + train 同源 + 等价性基准）。

**无条件批准进入执行阶段**。

执行关键点：
1. Phase -1 先定稿度量口径，再建 baseline
2. train.py 与 eval.py 共用 `core/metrics.py`，禁止两端口径不一致
3. trajectory IC 必须用去趋势序列（detrend_to_baseline），禁止原始价格
4. checkpoint 选择 IC 与 eval 报告 IC 同口径
5. 等价性验证用口径变更四标准（§8.3），不要求复现旧 IC 值

---

## 十六、审核记录（完整）

| 版本 | 日期 | 评审结果 | 主要变动 |
|------|------|----------|----------|
| v1 | 2026-06-19 | ⚠️ 需修订 | 初版系统性评审，B1–B6 阻断级 + M1–M5 重要级，共 11 个问题 |
| v2 | 2026-06-19 | ⚠️ 大幅改善，需补关键残留 | 复审确认 8 项已解决；残留 R1/R2（关键）+ R3–R6（重要）+ R7–R12（次要） |
| v3 | 2026-06-19 | ✅ 报告达标 | 二次评审确认报告质量，补决策摘要、版本追踪 |
| v4 | 2026-06-19 | ✅ 执行批准 | 复核修订版，全部 12 问题已解决，追加 S1/S2 非阻断建议 |
| v5 | 2026-06-19 | ⚠️ 有条件批准 | 外部度量口径核实：E1/E8（关键，train 同源升级）+ E2/E4（重要）；建议插入 Phase -1 |
| v6 | 2026-06-19 | ✅ 无条件批准 | 最终复核，全部问题已解决（架构 + 度量口径 + train 同源 + 等价性基准） |
| v7 | 2026-06-19 | ✅ 维持无条件批准 | 增量复核 early stopping patience 层（A1）；确认 A2/A3/A4 已覆盖；修复 v6 合并残留；**A1 已写入方案 §1.6.7/Phase -1/Phase 4/检查清单** |

---

## 十七、最终结论

`docs/refactor_plan.md` 修订版（响应 v1–v5 全部审核意见）已达到执行标准。

**批准状态**: ✅ 无条件批准

**执行顺序**: Phase -1 → Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6

**关键约束**（执行时必须遵守）：
1. trajectory IC 必须用去趋势序列，禁止原始价格
2. train.py 与 eval.py 共用 `core/metrics.py` 同一函数
3. checkpoint 选择 IC 与 eval 报告 IC 同口径
4. 等价性验证用口径变更四标准（loss 对齐 + 数据流一致 + 新口径自洽 + 换算关系）
5. Phase 6 前打 git tag `pre-refactor-archive` 保留 provenance

**审核完成**。方案可进入执行阶段。

---

## 十八、v7 增量审核（early stopping patience 残余缝隙）

**审核日期**: 2026-06-19
**审核人**: Claude Code（增量复核，针对 v6 未单独核实的 early stopping 层）
**审核性质**: 非阻断追加。v6 的"无条件批准"维持，本节补一处残余缝隙 + 修复一处文档合并残留。

### 18.1 背景：为何单独复核 early stopping

v5 §14.8 指出 E1（错误口径 trajectory IC）"驱动 checkpoint 选择与 early stopping"，v6 将其归入 §1.6 背景并标"已解决"。但 v5/v6 均**未单独核实 early stopping 的 patience 机制**——只笼统提到"early stopping"，未区分"checkpoint 选择"与"patience 计数"两层。本节补这层核实。

### 18.2 代码核实：patience 确由错误口径 IC 驱动

`train_ddp.py:1095-1114`：

```python
improved = False
if avg_val_loss < best_val_loss:          # 途径1: val_loss 改进 → patience 归零
    best_val_loss = avg_val_loss
    patience_counter = 0
    improved = True
    ...
if ic_smoothed > best_ic:                  # 途径2: 错误口径 IC 改进 → patience 也归零
    best_ic = ic_smoothed
    patience_counter = 0
    improved = True
    ...
# Early stopping
if patience_counter >= config.early_stopping_patience:
    break
```

**关键**：`patience_counter` 归零有两条独立途径——val_loss 改进，**或** `ic_smoothed > best_ic` 改进。`ic_smoothed` 是 `history['ic']` 的 3-epoch 滑动均值（行 1077），而 `history['ic']` 存的是错误口径的 `close_trajectory_ic`（行 1035）。

因此 early stopping 的**停止时机**不只由 best_ic_model 的选择决定，更由 patience 的累积节奏决定。IC 被价格趋势托底在虚高水平时，`ic_smoothed > best_ic` 容易频繁触发 → patience 频繁归零 → 训练**在该停时不停**。这比"选错 checkpoint"更根本，因为它改变模型何时固化。

### 18.3 方案覆盖核查（A1–A4）

| 增量项 | 内容 | 方案覆盖 | 核查结论 |
|--------|------|----------|----------|
| A1 | early stopping patience 也绑错误口径 IC，口径变更后停止时机需重评 | §1.6 背景提"early stopping"，但**未单独约束 patience 层**，未要求口径变更后重评停止时机 | ⚠️ 残余缝隙 |
| A2 | 等价性基准在口径变更下失效 | §8.3 四标准（loss 对齐 + 数据流一致 + 新口径自洽 + 换算关系），§917/981 明确 | ✅ 已覆盖 |
| A3 | §6.2 分层评估误标"继承" | §6.2 行 801 已标注"新建（旧 eval_ddp.py 无此能力）" | ✅ 已覆盖 |
| A4 | combined_score 0.6/0.4 权重在去趋势后失配 | §1.6.6 明确 Phase -1 用正确口径重标定并记录依据 | ✅ 已覆盖 |

### 18.4 A1 残余缝隙的处理（已写入方案）

E1 修正为去趋势口径后，IC 量级下降（虚高成分消失）、波动结构变化，`ic_smoothed > best_ic` 的触发频率与 patience 累积节奏都会改变，early stopping 停止时机随之偏移。方案 §1.6 原未对此显式约束。

**已订正**（2026-06-19，方案补丁）：

1. 方案新增 **§1.6.7 Early Stopping patience 审视**，明确要求 Phase -1 定稿：(a) `patience_counter` 归零条件是否保留「IC 改进归零 patience」途径；(b) `early_stopping_patience=12` 阈值在去趋势口径 IC 曲线上重新确认；(c) Phase 4 不要求停止 epoch 与旧训练一致，改 loss 收敛曲线形态可比。
2. **Phase -1** 第 5 条新增 early stopping 审视任务。
3. **Phase 4** 等价性标准第 (d) 条补 early stopping 停止 epoch 不要求一致。
4. **§11 检查清单**新增 Phase -1 early stopping 审视项。

**严重性**：非阻断（与 §1.6.6 combined_score 权重审查同源，已一并纳入 Phase -1）。经此订正，A1 不再是"两不管"的执行时注意项，而是方案显式任务。

### 18.5 文档合并残留修复

本审核发现并修复一处文档合并残留：v6 插入 §15–§17 后，原 v5 §14.9 的结尾内容（表格行"有条件批准"及三条要求）未删除，残留在 §17 之后，与 §15.3"无条件批准"矛盾。已删除该残留段落。该残留仅为编辑遗留，不影响审核结论。

### 18.6 v7 批准状态

| 项 | v6 状态 | v7 核查 |
|----|---------|---------|
| 架构（R1–R12） | ✅ | 维持 |
| 度量口径（E1/E8） | ✅ | 维持 |
| train 同源 | ✅ | 维持 |
| 等价性基准（A2） | ✅ | 维持 |
| §6.2 措辞（A3） | ✅ | 维持 |
| combined_score 权重（A4） | ✅ | 维持 |
| early stopping patience（A1） | 未单独核实 | ✅ 已写入方案 §1.6.7 + Phase -1 第5条 + Phase 4 (d) + 检查清单 |
| 文档合并残留 | — | ✅ 已修复 |
| **执行批准** | 无条件批准 | **维持无条件批准**（A1 已纳入方案显式任务） |

**结论**：v6 的无条件批准维持。A1 已从"执行时注意项"升级为方案显式任务（§1.6.7 等），不再是两不管缝隙。本节为最终增量审核，无新增阻断问题。

---

## 十九、代码预审（v8，代码初步完成后）

**预审日期**: 2026-06-19
**预审对象**: 已落盘的新代码 `finetune/predictor/core/` + 入口脚本（train/eval/backtest/preprocess.py）
**预审性质**: 代码仍在修补细节，本节为预审快照，核实方案硬约束的实际落实情况，不替代正式验收。

### 19.1 已落盘代码盘点

| 模块 | 文件 | 状态 |
|------|------|------|
| core/ | config.py, paths.py, schema.py, normalization.py, splitting.py, dataset.py, metrics.py, utils.py, test_metrics.py | ✅ 齐全 |
| 入口 | train.py, eval.py, backtest.py, preprocess.py | ✅ 齐全 |
| tokenizer 训练入口 | `finetune/tokenizer/train.py` | ❌ 不存在 |
| preprocess 子目录 | `finetune/preprocess/` | ❌ 不存在（仅有 predictor/preprocess.py 单文件） |

### 19.2 硬约束落实核实

| 硬约束 | 核实结果 | 证据 |
|--------|----------|------|
| train/eval 共用 core/metrics | ✅ | train.py:55-69、eval.py:48-63、backtest.py:37-45 均 import core.metrics |
| 禁直接 np.corrcoef/spearmanr | ✅ | 三入口均无直接调用 |
| traj IC 用去趋势序列 | ✅ | train.py:307-311、eval.py:227-230 均 `detrend_to_baseline` 后再 `safe_trajectory_ic` |
| backtest IC 不混算 | ✅ | backtest.py:200-211 逐股票算再聚合 |
| checkpoint 用去趋势 IC | ✅ | train.py:531 `ic_smoothed > best_ic`，ic 来自去趋势口径 |
| R1 区间语义 no-leakage | ✅ | splitting.py 扫描线区间相交检查（行279） |
| R2 create_target_blocks 按时间轴 | ✅ | splitting.py:114,126 按 target_start 排序切块 + min 防超界 |
| R7 min_periods=1 | ✅ | normalization.py:75 默认 1，注释对齐 dataset.py:185 |
| R11 validate 调用 | ✅ | config.py:41 `__post_init__` 调 validate() |
| S2 ArtifactConfig.model_type | ✅ | config.py:133 |
| min_samples 无 +1 | ✅ | config.py:40 |
| LR=0.01 | ✅ | config.py:84 |

**架构与度量口径主体已落实**，与方案 §1.6 高度一致。以下为预审发现的具体问题。

### 19.3 预审发现的问题

#### P1. 🔴 `compute_naive_da` 硬编码返回 0.5，E2 excess DA 失去意义

**位置**: `core/metrics.py:204-240`、`format_metrics_report:669`

`compute_naive_da` 的 docstring 写了一大段纠结（"持平预测认为不变""threshold 很难定"），最终：
```python
# 简化定义：naive DA = 50%（随机二分类的期望）
return 0.5
```
`format_metrics_report:669` 也写死 `naive_da = 0.5`。

**问题**：E2 的设计意图是「model DA − naive DA 让数字可解读」，naive baseline 应是**持平预测（persistence: pred == baseline）在实际数据上的 DA**，即实际方向中「与 baseline 同向」的比例。这可从 actual 序列直接统计，不需要 threshold。硬编码 0.5 使 `excess_da = model_da - 0.5` 退化为 DA 的平移，既不是"持平朴素预测"对照，也无法判断 model DA 是否真优于朴素预测——A 股多数日子涨，naive DA（看多）本就 >50%，用 0.5 会高估 excess。

**要求**：实现真实的 naive DA——对每个样本，naive 预测 = baseline 持平，naive DA = (actual_dir == 持平方向) 的比例。或明确声明 naive = persistence 并实现 persistence 预测的 DA。禁止用 0.5 占位。

#### P2. 🔴 tokenizer 训练入口缺失（R3 未落地）

**位置**: `finetune/tokenizer/train.py` 不存在

方案 §1.5 定义了 `train_tokenizer` 函数 + CLI，Phase 2 第 5 条要求新增该入口。实际未建。后果：`outputs/tokenizers/{norm_mode}/{model_type}` 路径键控设计无产生流程，`validate_tokenizer_consistency`（§1.5）无对象可校验。预测器训练时 `resolve_tokenizer_path` 会指向不存在的 tokenizer。

**要求**：补 `finetune/tokenizer/train.py`，或若暂时复用现有 `finetune/train_tokenizer.py`，需在方案中说明迁移路径与 per-norm_mode 适配。

#### P3. 🟡 eval.py IC 聚合未走 `aggregate_ic`，DDP 场景丢分布

**位置**: `eval.py:256-265`

eval.py 单卡 IC 聚合用本地 `np.mean/np.std/p50`，未调用 `core/metrics.py:aggregate_ic`（该函数含 all_gather 分布聚合）。后果：
- 若 eval.py 支持 DDP（方案 §6.2 宣称支持 `torchrun`），多卡时各 rank 只算本地，未 all_gather 完整列表 → 分布统计错误。
- eval.py 未报 p25/p75（只 mean/std/p50），低于 §1.6.3 要求。

**要求**：确认 eval.py 是否 DDP；若是，改用 `aggregate_ic`；补 p25/p75。

#### P4. 🟡 train.py 未输出 excess DA（仅 import 未调用）

**位置**: train.py:61 import `excess_da` 但训练循环未调用

训练监控只报 DA_score，看不到 excess DA。E2 的可解读性在训练阶段缺失。

**要求**：训练日志补 excess DA（需先修 P1 的 naive_da）。

#### P5. 🟡 Phase -1 待办项在代码中"占位未定稿"

**位置**: config.py:86,96,101-102

- `early_stopping_patience=12`（注释"需重新确认"，未确认）
- `ic_patience_reset=True`（v7 §1.6.7 要求 Phase -1 定稿保留/移除，代码默认 True 但无审视记录）
- `combined_ic_weight=0.6/da_weight=0.4`（注释"需重新标定"，未标定）

这些是 Phase -1 度量口径定稿的产物，代码先占了默认值。**可接受**（代码先行），但 Phase -1 必须用去趋势口径实际跑 IC 曲线后确认这些值，否则 v7 §1.6.7 的审视要求未真正完成。

**要求**：Phase -1 执行时记录这些参数的最终值与依据，写入 summary.json。

#### P6. ⚪ BacktestSchema / amplitude 用 pred_raw[0] 单根

**位置**: eval.py:242-243 `pred_amp = pred_raw[0,1] - pred_raw[0,2]`

振幅误差率只用 predict 段第 1 根的 high-low，未用整段。振幅是"日内"概念，单根合理，但 §1.6.5 描述为"预测振幅÷实际振幅"未明确单根 vs 整段。非缺陷，建议在文档或注释明确口径。

### 19.4 预审结论

| 类别 | 状态 |
|------|------|
| 架构落实（R1/R2/R7/R11/S2/B4） | ✅ 全部落地 |
| 度量口径主体（E1/E8/可懂指标） | ✅ 落地 |
| E2 excess DA | 🔴 naive_da 占位 0.5，需实修 |
| R3 tokenizer 入口 | 🔴 缺失 |
| E4 聚合分布 | 🟡 eval.py 未用 aggregate_ic |
| Phase -1 定稿项 | 🟡 代码占位，待实际确认 |

**预审判断**：代码主体质量良好，方案硬约束大部分已真正落实（非文档空话）。**阻断正式验收的硬伤是 P1（naive_da 占位）与 P2（tokenizer 入口缺失）**；P3/P4 影响度量完整性；P5 是 Phase -1 待办。代码修补细节时应优先处理 P1/P2。

**不改变 v7 的执行批准**：方案层面已批准，本节为代码预审，问题在实现层，需在代码修补阶段解决 P1/P2 后再做正式验收。

---

## 二十、全面代码评审（v9，代码修补后）

**评审日期**: 2026-06-19
**评审范围**: `core/` 全部 9 模块 + train/eval/backtest/preprocess.py
**评审方法**: 逐文件通读 + DDP 集体操作对称性逐个核查 + 归一化/splitting 泄露防线核实；agent 初筛后由评审人亲自核实关键指控，剔除误报。

> 注：§19 预审后代码已修补——train.py 的 early stopping break（改 broadcast 同步）、DDP forward（改 `model(...)`）、naive DA（改真实 up_ratio 计算）均已修复。本节反映修补后现状。

### 20.1 致命问题（🔴，阻断多卡训练）

#### F1. `train.py:357-358` naive DA 的 all_gather 在 `if rank == 0:` 块内 → NCCL 死锁

**证据**:
```python
# 行 338
if world_size > 1:
    ic_result = aggregate_ic(...)   # 行 339 所有 rank 执行 ✅
    da_result = aggregate_da(...)   # 行 340 所有 rank 执行 ✅
    naive_da_by_step = {}           # 行 343
    if rank == 0:                   # 行 344 ← 只有 rank 0 进
        for step_idx in range(config.predict):
            ...
            dist.all_gather(gathered_up, up_count_tensor)   # 行 357 ← 仅 rank 0
            dist.all_gather(gathered_n, n_tensor)           # 行 358 ← 仅 rank 0
```

**后果**: 修 naive DA 时把 all_gather 放进了 `if rank == 0:` 块。rank 0 在行 357 等待所有 rank 参与 all_gather，但 rank 1/2/3 不进该块、已 return 出函数 → rank 0 永久阻塞 → **NCCL 超时**。与刚修的 early stopping 死锁同类（集体操作在 rank 条件分支内）。

**修复**: all_gather 必须所有 rank 执行，只 rank 0 处理结果（仿 `aggregate_ic` 模式）：
```python
# 所有 rank 执行 all_gather
for step_idx in range(config.predict):
    ...  # 准备 tensor
    dist.all_gather(gathered_up, up_count_tensor)   # 所有 rank
    dist.all_gather(gathered_n, n_tensor)           # 所有 rank
    if rank == 0:                                    # 只 rank 0 组装
        naive_da_by_step[step_idx] = ...
```

#### F2. `train.py:519` validation loss 未实现，直接用 train loss

**证据**: `avg_val_loss = avg_train_loss  # TODO: 实现 validation loss 计算`

**后果**: `best_val_loss` 驱动的 `best_model` checkpoint 选择基于训练 loss（非验证 loss）→ best_model 是训练集拟合最好的，过拟合风险；early stopping 的 val_loss 改进途径实际是 train loss 改进，语义错位。

**修复**: 实现真实 validation loss（在 val 集前向算 recon_loss，不反传）。

#### F3. `train.py:786` val_indices 每股只取最后 1 个窗口 → val 集过小

**证据**:
```python
# 行 784-789（val_indices 构建）
elif hasattr(d, 'columns'):
    if len(d) >= config.lookback + config.predict:
        val_indices.append((symbol, len(d) - config.lookback - config.predict))  # 每股 1 个
```

**后果**: 每只股票 val 只评估最后 1 个窗口。若 5000 股则 val 仅 5000 样本，但 IC 统计需足够样本且跨股票分布；且只评最后窗口 = 只评最近时点，IC 不稳、不代表整体泛化。

**修复**: val 应评该股票所有合法窗口（或按 preprocess 已切好的 val split 取，而非 main 里重新只取末尾）。

### 20.2 重要问题（🟡）

#### I1. `dataset.py:92-94` + `train.py:801` + `utils.py set_seed` 三者叠加 → DDP 各卡数据相同

**证据**:
- `dataset.py:92`: `if self.mode=='train': rand_idx = self.py_rng.randint(0, len); symbol, start = self.indices[rand_idx]` —— __getitem__ 忽略 DataLoader 传入的 idx，自己用 `self.py_rng` 随机采样。
- `dataset.py:71`: `self.py_rng = np.random.RandomState(config.seed)` —— 所有 rank 同 seed。
- `utils.py set_seed`: 所有 rank 同 seed。

**后果**: DDP 各 rank 的 KronosDataset 用相同 seed → 每个 step 采到相同 (symbol, start) → 各卡训练数据完全相同 → DDP 数据并行失效（退化为梯度平均的同数据多卡）。加上 `__getitem__` 忽略 idx，DataLoader 的 RandomSampler 形同虚设。

**修复**: (a) __getitem__ 应尊重传入 idx（用 DataLoader 的 sampler 控制顺序），不要自随机；(b) DDP 应用 `DistributedSampler` 或各 rank seed = config.seed + rank。

#### I2. `normalization.py:112` SlidingMANormalizer 无 `shift(1)`，与旧 `dataset.py:183` 不一致

**证据**:
- 旧 `dataset.py:183`: `df_shifted = df.shift(1); rolling_mean = df_shifted.rolling(...).mean()` —— 排除当前点。
- 新 `normalization.py:112`: `s.rolling(window, min_periods=1).mean()` —— **包含当前点**。

**后果**: 旧实现"每个点用之前 N 步"（不含自身），新实现"用含自身的 N 步"。这不是未来泄露（rolling 是因果的，不含未来），但：(a) 与旧实现不一致 → Phase 4 等价性验证（与旧入口对齐）必失败；(b) 归一化当前点时用了当前点自身，归一化值"知道"当前水平，轻微降低预测难度。方案 §1.2 只承诺 min_periods=1 对齐，未提 shift(1)，但 feedback 强调"新旧数据输出一致"。

**修复**: 明确选择——若要与旧实现完全对齐，加 `shift(1)`；若刻意改为含当前点，需在方案说明并放弃 Phase 4 的"与旧入口对齐"基准（改为新口径自洽）。

#### I3. `eval.py` 完全无 DDP 支持，方案 §6.2 承诺未落地

**证据**: `eval.py:348 device = get_device()`（无 local_rank），main() 无 `init_process_group`，evaluate 函数无 world_size/rank 参数。IC/DA 聚合用本地 `np.mean`（行 263-297）——单进程下正确，但无法 `torchrun` 多卡。

**后果**: 方案 §6.2 宣称 eval 支持 `torchrun --nproc_per_node=N`，实际未实现。R6 残留。评估全量慢。

**修复**: 若保留 DDP 评估承诺，eval.py 需加 DDP 初始化 + 用 `aggregate_ic`（含 all_gather）；若放弃，修正方案 §6.2 措辞。

#### I4. `backtest.py:249` excess_da 硬编码 0.5，与 train.py 不一致

**证据**: `result['excess_da'] = excess_da(model_da, 0.5)` —— 而 train.py 已改为真实 `max(up_ratio, 1-up_ratio)`。

**后果**: train 与 backtest 的 excess DA 口径不一致；backtest 的 excess DA 仍是平移非真 baseline。

**修复**: backtest 也算真实 naive DA。

#### I5. `backtest.py:236-243` IC 聚合丢 p25/p75

**证据**: 只算 mean/std/p50，无 p25/p75。

**后果**: 违反 §1.6.3"聚合保留 mean/std/p25/p50/p75"。无法判断 backtest IC 稳健性。

**修复**: 补 p25/p75。

#### I6. `splitting.py:132` create_target_blocks 跨块窗口被丢弃

**证据**: `if s.target_start >= current_block_start and s.target_end <= current_block_end:` —— 要求 target 完全在块内。target 跨块边界（block_size=50, predict=10 时常见）的窗口被丢弃。

**后果**: 样本丢失（非泄露），数据利用率下降，块间 target 时间轴出现空洞。

**修复**: 跨块窗口按 target_start 归入起始块（或 target 中点归入对应块），保证不丢样本且块间不相交。

#### I7. `splitting.py:70-78` time_split 跨边界样本强分到 train → validate_no_leakage 崩溃

**证据**: 行 57 `if s.target_end <= train_end: train`；跨边界样本（target_start<train_end<target_end）落 else 行 70 分到 train，其 target 跨入 val 区间。

**后果**: validate_no_leakage 扫描线会检测到 train/val target 相交 → AssertionError 中止 preprocess（非静默泄露，但 preprocess 会崩）。

**修复**: time_split 应丢弃跨 split 边界的样本，而非强分。

#### I8. `train.py:740-743` tokenizer 静默 fallback，norm_mode 不匹配风险

**证据**:
```python
tokenizer_path = get_tokenizer_path(config.norm_mode, train_config.model_type)
if not os.path.exists(tokenizer_path):
    tokenizer_path = 'outputs/tokenizers/final/2k-MA60' if args.model == 'mini' else '...'
```

**后果**: 新路径体系（`outputs/tokenizers/{norm_mode}/{model_type}`）不存在 → 总是 fallback 到 legacy tokenizer。legacy tokenizer 的 norm_mode 可能与 config 不匹配，违反 tokenizer-predictor 数据一致性（§1.5）。且静默 fallback 无警告。

**修复**: R3 tokenizer 训练入口落地前，明确 fallback 规则并警告 norm_mode 不匹配；或修复 R3。

#### I9. `train.py` 无 `--resume` 断点续训

**证据**: CLI 无 --resume，model 直接 `Kronos.from_pretrained(pretrained)` 不加载 checkpoint。旧 `train_ddp.py` 有 --resume。

**后果**: 训练中断无法续训，长训练风险。

**修复**: 加 --resume 加载 latest checkpoint + optimizer/scheduler 状态。

### 20.3 次要问题（⚪）

| # | 位置 | 问题 |
|---|------|------|
| M1 | `metrics.py:204-240` `compute_naive_da` | 仍 `return 0.5`（train.py 绕过它自算，但该函数是死代码/误导，且 backtest 仍可能误用） |
| M2 | `train.py:156` `update_training_info` | `safe_save_json.__wrapped__` 逻辑混乱，每次重读全 json 再写，频繁 IO |
| M3 | `train.py:530` | `seed=config.seed + epoch_idx*9999` 但 `n_samples=-1` 全量，seed 无效 |
| M4 | `train.py:537` | `da_by_step` 把聚合后标量包成单元素列表传 `calculate_da_score`，语义变形（设计是接原始 DA 列表） |
| M5 | `splitting.py:141-147` | create_target_blocks 块间 assert O(块数²)，块数多时慢；可改扫描线 |
| M6 | `utils.py safe_save_*` | 非原子写入，训练中 crash 可能损坏 training_info.json |
| M7 | `dataset.py:92` | __getitem__ 忽略 idx 自随机，破坏 DataLoader sampler 语义（与 I1 同源） |

### 20.4 agent 误报澄清（避免开发人员走弯路）

| agent 指控 | 实际 | 结论 |
|-----------|------|------|
| `metrics.py aggregate_ic` padding 零值污染统计 | 重组时 `for i in range(lengths[rank_idx])` 只取有效长度，padding 零不入统计 | ❌ 误报 |
| `calculate_combined_score` (ic+1)/2 丢失负相关 | 负 IC → 较低分（被惩罚），是合理归一化 | ❌ 误报（设计选择） |
| `sigmoid_score` overflow | float64 上限 ~1e308，steepness=21 实际不溢出 | ❌ 误报 |
| `normalization.py` 未来泄露 | rolling 是因果的（不含未来）；真实问题是 shift(1) 缺失（I2） | ⚠️ 定性过重，问题真实但非"未来泄露" |
| `validate_no_leakage` O(n²) | 实为扫描线 O(N log N)；O(n²) 是 create_target_blocks 块间 assert（M5） | ❌ 误报（张冠李戴） |
| `splitting.py:70` feature leakage | 实为 target 跨 split，被 no-leakage 拦截崩溃（I7），非静默泄露 | ⚠️ 现象真实，定性不准 |

### 20.5 已正确落实（肯定）

- train/eval/backtest 共用 `core/metrics`，无直接 corrcoef/spearmanr
- traj IC 全走 `detrend_to_baseline` → `safe_trajectory_ic`（train.py:313、eval.py:227）
- backtest IC 逐股票算再聚合（backtest.py:200）
- checkpoint 选择用去趋势 IC
- early stopping break 用 `dist.broadcast` 同步所有 rank（已修）
- DDP forward 用 `model(...)`（已修）
- naive DA 真实计算 `max(up_ratio, 1-up_ratio)`（train.py:365，已修；eval.py:303）
- validate_no_leakage 扫描线区间语义正确
- R1/R2/R7/R11/S2/min_samples/LR=0.01 落地
- aggregate_ic/aggregate_da 的 all_gather 对称（行 339-340）

### 20.6 修复优先级

1. **F1**（all_gather 死锁）——多卡训练直接卡死，必须最先修。
2. **I1**（DDP 各卡数据相同）——多卡训练正确性，与 F1 同优先级。
3. **F2/F3**（val_loss 未实现 / val 集过小）——训练有效性，单卡也受影响。
4. **I2**（shift(1)）——决定 Phase 4 等价性能否通过，需先定口径。
5. **I3-I9**——评估回测完整性与一致性。
6. **M1-M7**——代码质量。

### 20.7 评审结论

代码主体架构正确，方案硬约束大部分落实，DDP early stopping/forward/naive DA 已修补。**但 F1（naive DA all_gather 死锁）是修补时新引入的致命 bug，会使多卡训练在评估阶段卡死**——这是当前最紧迫的修复项。I1（DDP 数据同质）次之，使多卡训练即使不卡死也退化为同数据多卡。

**单卡训练**受 F2/F3（val 失效）影响，best_model 选择失真，但不崩溃。

**建议**: 修 F1 + I1 + F2/F3 后，多卡与单卡训练才可进入正式验收。I2 需先与方案对齐口径再修。

---

## 二十一、Tokenizer 训练方案纠正（v10）

**纠正日期**: 2026-06-19
**触发**: 用户指出 §1.5 tokenizer 训练方案有误，「没有理解这个项目关于 tokenizer 的设计」。并补充确认：「不同的 norm_mode 影响 tokenizer 模型，所以需要微调」。

### 21.1 核实项目 tokenizer 真实设计

核实对象：`model/kronos.py`（KronosTokenizer 类）、`deprecated/finetune/train_tokenizer.py`、`final_models/training_scripts/train_tokenizer_mini.py`、`pretrained/`、`outputs/tokenizers/`。

**核实结论**：

1. **`KronosTokenizer` 无 `.train()` 类方法**（`grep "def train"` kronos.py 无结果）。只有 `from_pretrained`（继承 `PyTorchModelHubMixin`，kronos.py:13）。训练是「加载预训练 → 标准训练循环微调 → save_pretrained」。
2. **预训练 tokenizer 只有两个**：`pretrained/Kronos-Tokenizer-2k`（mini 用，group_size=5，2048 context）和 `pretrained/Kronos-Tokenizer-base`（small/base 用，group_size=4，512 context）。
3. **微调损失**：`recon_loss = mse(z_pre,x)+mse(z,x)` + `bsq_loss`（deprecated train_tokenizer.py:194-195）。权重微调，非重新学码本。
4. **group_size/context 是架构超参**（kronos.py:40 `__init__` 参数），预训练定死，微调不改。
5. **norm_mode 影响微调数据分布**（QlibDataset 按 norm_mode 归一化后喂 tokenizer），故不同 norm_mode 需各自微调——与用户补充一致。

### 21.2 早期 §1.5 的三处错误（已纠正）

| 错误 | 实际 | 严重性 |
|------|------|--------|
| 写 `KronosTokenizer.train(data=, vocab_size=)` 从零训练 | 该 API 不存在；实际 `from_pretrained` + 微调循环 | 🔴 致命（按此实现会直接报错） |
| `vocab_map={mini:2048, small:4096, base:8192}` | 杜撰映射；实际 mini→2k，small/base 共用 base tokenizer，无 small=4096 | 🔴 致命（误导架构选择） |
| 「每个 norm_mode 单独训练 tokenizer」表述 | 把「按 norm_mode 微调」误解为「按 norm_mode 从零建架构」；实际是同架构、不同权重微调 | 🟡 表述误导 |

### 21.3 纠正后的设计要点（已写入方案 §1.5.1-1.5.5）

- tokenizer 是**微调**（from_pretrained + 权重训练），非从零训练；
- **架构由 model_type 决定**（mini→2k，small/base→base），架构耦合 predictor token 维度；
- **vocab_size 不是微调参数**，预训练时定；
- **norm_mode 决定微调数据分布**，不同 norm_mode 必须各自微调、不可复用（响应用户补充）；
- 路径 `outputs/tokenizers/{norm_mode}/{model_type}/` 含义：model_type 决定架构，norm_mode 标记微调分布；同 model_type 不同 norm_mode 架构相同、权重不同。

### 21.4 代码层遗留错误（供开发人员修）

方案已纠正，但代码中仍有杜撰的 vocab 映射残留：

1. **`train.py:87-91` `TOKENIZER_PATHS`**：
   ```python
   TOKENIZER_PATHS = {
       'mini': {'vocab': 2048},
       'small': {'vocab': 4096},   # ← 杜撰
       'base': {'vocab': 8192},    # ← 杜撰
   }
   ```
   该常量实际未被使用（行 740 用 `get_tokenizer_path`），是死代码，但误导。**建议删除或改为架构映射**：`{'mini':'Kronos-Tokenizer-2k', 'small':'Kronos-Tokenizer-base', 'base':'Kronos-Tokenizer-base'}`。

2. **`config.py:143` docstring**：「vocab_size 映射：mini→2048, small→4096, base→8192」——同样错误。**建议改为**：「架构映射：mini→Kronos-Tokenizer-2k，small/base→Kronos-Tokenizer-base」。

3. **`train.py:743` fallback** `'outputs/tokenizers/final/2k-MA60' if mini else '...base-MA60'`——2k/base 区分正确（与架构一致），但 `final/` 是旧路径。**建议**：R3（tokenizer 微调入口）落地后，fallback 改为新路径或直接报错而非静默 fallback（见 §20 I8）。

### 21.5 待办

- **R3 落地**：`finetune/tokenizer/train.py` 仍为空目录。按纠正后的 §1.5.2 实现微调入口（from_pretrained + 微调循环，非从零训练）。
- 修代码层 vocab 映射残留（§21.4）。
- Phase 2 执行时按 §1.5.2 实现，vocab_size 不作为参数。

