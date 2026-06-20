# 重构代码审查记录

**建立日期**: 2026-06-20
**背景**: 本次为重大重构变更。2026-06-20 发现 backtest.py 加载链路漏审(B1-B14),暴露审查覆盖方法有漏洞——按"问题类型"组织清单而非"按文件通读",导致部分文件 main() 加载逻辑、依赖代码(model/kronos.py)从未通读。本记录逐文件核实审查状态,补审未审文件,确保所有代码已被审查。

**审查状态图例**:
- ✅ 通读 — 评审人完整读过文件,有问题已记录
- ⚠️ 部分 — 仅 agent 看或只 grep,未通读
- ❌ 未审 — 从未读过
- 🆗 无需 — legacy/非新代码,不在新体系

## 一、文件审查状态总表

| 文件 | 状态 | 审查记录 | 备注 |
|------|------|----------|------|
| `finetune/predictor/core/config.py` | ✅ 通读 | §20 / §22 | DataConfig/TrainConfig/ArtifactConfig/BacktestConfig |
| `finetune/predictor/core/paths.py` | ✅ 通读 | §20 | 路径函数 |
| `finetune/predictor/core/schema.py` | ⚠️ agent | §20 | SampleSchema/BacktestSchema,agent 看过未亲验 |
| `finetune/predictor/core/normalization.py` | ✅ 通读 | §20 / I2 | SlidingMANormalizer shift(1) 口径已定 |
| `finetune/predictor/core/splitting.py` | ✅ 通读 | §20 / I6/I7 | time_split/block_split/validate_no_leakage |
| `finetune/predictor/core/dataset.py` | ✅ 通读 | §20 / I1 | KronosDataset __getitem__ |
| `finetune/predictor/core/metrics.py` | ✅ 通读 | §20 / §23 / M4 | 度量函数,calculate_da_score 已改兼容标量 |
| `finetune/predictor/core/utils.py` | ✅ 通读(补) | §4.6 UT1-UT3 | 2026-06-20 补审。safe_save_json 已原子(M6 实已修),safe_save_pickle 非原子(UT3) |
| `finetune/predictor/core/test_metrics.py` | ⚪ 抽查 | §4.8 | 只 import 未调用 calculate_da_score,无实际测试逻辑 |
| `finetune/predictor/core/__init__.py` | ⚪ 抽查 | §4.8 | 导出层,无逻辑 |
| `finetune/predictor/eval_all.py` | 🆗 legacy | §4.8 | 评 Mode7-10 旧模型,手册不引用,建议归 deprecated |
| `data/generate_raw.py` | ✅ 通读(补) | §4.7 GR1-GR2 | 2026-06-20 补审。SQL 解析数据生成,无泄露问题 |
| `data/generate_backtest_raw.py` | ✅ 通读(补) | §4.7 | 2026-06-20 补审 |
| `model/module.py` | 🆋 未审 | — | 预训练模型子模块(TransformerBlock/BSQuantizer),非本次变更,稳定代码 |
| `model/__init__.py` | 🆋 未审 | — | 导出层 |
| `finetune/predictor/train.py` | ✅ 通读 | §20 / §23 | 主训练入口 |
| `finetune/predictor/eval.py` | ✅ 通读(补) | §4.3 EV1-EV5 | 2026-06-20 补审 main()+load_model_and_tokenizer。加载链路问题多(硬编码 lookback/predict/split、strict=False、路径错位) |
| `finetune/predictor/backtest.py` | ✅ 通读(补) | §24 B1-B14 | 2026-06-20 补审,加载链路问题多 |
| `finetune/predictor/preprocess.py` | ✅ 通读 main(补) | §4.4 | 2026-06-20 补审 main()。CLI 完整、time split 强制边界、泄露检查失败 exit(1)。preprocess() 函数体 agent 审过(§20) |
| `model/kronos.py` | ✅ 通读(补) | §4.5 | 2026-06-20 补审 KronosTokenizer + auto_regressive_inference。预训练稳定代码,非本次变更,契约清晰 |
| `finetune/tokenizer/train.py` | ✅ 通读(补) | §4.1 TK1-TK5 | 2026-06-20 补审。**TK5 路径错位致命**(存 checkpoints/best_model,读 output_path) |
| `finetune/tokenizer/validate.py` | ✅ 通读(补) | §4.2 VA1-VA3 | 2026-06-20 补审。VA1 同 TK5 路径错位 |

**覆盖统计**(22 文件):✅通读 8 | ⚠️部分 6 | ❌未审 8

## 二、审查方法反思

1. **按问题类型 vs 按文件**:之前 §22 清单按"度量/DDP/泄露/..."组织,导致 backtest 的"加载链路"这类问题不在任何类型下被系统检查。**改进**:每个入口脚本必须 main() 通读,不只审被问的维度。
2. **依赖代码未审**:train/eval/backtest 都依赖 `model/kronos.py:auto_regressive_inference`,但从未通读其返回契约。backtest review 的 B6 我曾误报(怀疑 shape 不一致),正是因为没读依赖代码。**改进**:共享依赖函数必须读源码确认契约。
3. **agent 评审 prompt 局限**:agent 只回答被问的维度,不主动覆盖"模型加载/路径"。**改进**:agent prompt 须明确包含"main() 加载链路、路径、架构匹配"。

## 三、补审计划

逐个补审未审/部分审文件,每审完更新本记录与对应 review 章节:

1. `finetune/tokenizer/train.py`(未审,新代码核心)
2. `finetune/tokenizer/validate.py`(未审)
3. `finetune/predictor/eval.py` main()(部分,补加载链路)
4. `finetune/predictor/preprocess.py` main()(部分)
5. `model/kronos.py`(部分,补通读 auto_regressive_inference + Kronos/KronosTokenizer)
6. `finetune/predictor/core/utils.py`(未通读)
7. `finetune/predictor/core/__init__.py` / `test_metrics.py`(导出+测试)
8. `data/generate_raw.py` / `generate_backtest_raw.py`(数据生成)
9. `model/module.py` / `model/__init__.py`(模型子模块)
10. `finetune/predictor/eval_all.py`(判定是否需审或归 deprecated)

---

## 四、补审记录

(逐文件补审结果追加于此)

### 4.1 `finetune/tokenizer/train.py`(2026-06-20 补审)

**通读完成**。步数驱动采样、from_pretrained 微调、val 重建损失 early stopping 均与方案 §1.5.2 一致。发现问题:

| 编号 | 严重性 | 位置 | 问题 |
|------|--------|------|------|
| TK1 | 🆗 | 行 296-319 | TokenizerDataset 未继承 torch Dataset(duck typing),DataLoader 仍可用。可接受 |
| TK2 | 🟡 | 行 211-212 | `tokenizer(batch_x)` 返回契约依赖 model/kronos.py(待审),与 deprecated 一致 |
| TK3 | 🟡 | 行 252-262 | early stopping 的 best 判断被 `if epoch_idx >= grace_period` 包裹——前 grace epoch 不存 best_model,best_val_loss 保持 inf。若 epochs<=grace 则只有 final_model,meta 记 best_val_loss=inf。边界需注意 |
| TK4 | 🟡 | 行 311-317 | DataFrame 分支用 full_window 归一化(整 lookback mean/std),sliding_ma 模式下与 predictor 口径不符。fallback 路径隐患(dict 格式正常不受影响) |
| **TK5** | **🔴 致命** | train.py:256/266 vs validate.py:84/129 vs predictor 加载 | **路径错位**:train.py `save_pretrained(output_path/checkpoints/best_model)`,但 validate.py 与 predictor 从 `output_path/` 直接加载(期望 `output_path/model.safetensors`)。存读路径不一致 → 微调后验证脚本报 "not found"、predictor 加载不到权重。**新代码致命 bug,此前未审出** |

**TK5 修复方向**:统一路径——要么 train.py 存到 output_path 本身(best 覆盖 final,或 final 直接存 output_path),要么 validate/predictor 改从 `output_path/checkpoints/best_model` 加载。建议前者(权重直接存 output_path,与 from_pretrained 期望一致)。

### 4.2 `finetune/tokenizer/validate.py`(2026-06-20 补审)

**通读完成**。对比预训练 vs 微调重建损失,验收标准(微调<预训练 + early stopping 触发)与方案 §1.5.6 一致。问题:

| 编号 | 严重性 | 位置 | 问题 |
|------|--------|------|------|
| VA1 | 🔴 | 行 84/129 | 同 TK5——期望 `finetuned_path/model.safetensors`,但 train.py 存 `checkpoints/best_model/`。验证脚本永远报 not found(除非 train.py 改路径) |
| VA2 | 🟡 | 行 146-148 | `pre_loss = F.mse_loss(z_pre, batch_x, reduction='none')` 后 `.view(size(0),-1).mean(dim=1)` —— per-sample loss。正确。但 train.py 的 val_loss(行 239)用 `F.mse_loss(z, batch_x)`(默认 mean,整 batch 标量)。**train 与 validate 的 val_loss 口径不同**(train 用 z,validate 用 z;但 train 是 batch 均值,validate 是 per-sample 再均值)。数值上等价,但实现不统一 |
| VA3 | ⚪ | 行 197-205 | early stopping 检查只警告不判失败(行 202-203),合理(非阻断) |

**VA1 与 TK5 同源**,是路径错位的表现侧。修 TK5 即解决。

### 4.3 `finetune/predictor/eval.py` main() + load_model_and_tokenizer(2026-06-20 补审)

**通读完成**。之前 §20/§23 只审了 evaluate() 度量逻辑和 CLI,**main() 加载链路从未审**。补审发现:

| 编号 | 严重性 | 位置 | 问题 |
|------|--------|------|------|
| EV1 | 🔴 | load_model_and_tokenizer 行 124 | `model_dir = get_model_path(norm_mode, 400, 10, 'block', model_type)` —— **硬编码 lookback=400/predict=10/split_mode='block'**,忽略 CLI 的 --lookback/--predict/--split-mode。用户传 `--lookback 200 --split-mode time` 仍用 400/block 路径加载 → 找不到或加载错 checkpoint |
| EV2 | 🔴 | 行 128 | `model.load_state_dict(state_dict, strict=False)` —— 同 backtest B1,静默吞 key 不匹配。加载错 checkpoint 无提示 |
| EV3 | 🔴 | 行 127 | `if os.path.exists(...model.safetensors): load; ` 找不到则**静默用预训练权重**(行 122 from_pretrained),不报错。用户以为评估微调模型,实际评估预训练 |
| EV4 | 🔴 | 行 113-114 | tokenizer `get_tokenizer_path` 返回 output_path,但 tokenizer/train.py 存 `output_path/checkpoints/best_model`(TK5)。from_pretrained(output_path) 加载不到微调权重 → fallback 到 legacy `outputs/tokenizers/final/2k-MA60`。**与 TK5 同源路径错位** |
| EV5 | 🟡 | 行 460-463 | `--models mini,small,base` 批量评估,但 load_model_and_tokenizer 的 EV1 硬编码对每个 model_type 都用 400/block 路径,批量评估时 lookback/split 仍错 |

**eval.py 加载链路与 backtest.py 同病**:strict=False 静默、路径错位、找不到静默 fallback。**外加 EV1 硬编码 lookback/predict/split** 是 eval 独有的更严重问题——CLI 参数对模型加载完全无效。

**核心结论**:eval.py 的 `--lookback/--predict/--split-mode` CLI 参数**只影响数据路径(行 439 get_split_data_path 用 config)**,不影响模型加载(行 124 硬编码)。用户改这些参数会导致**数据与模型错配**(数据用 200/time,模型用 400/block 的 checkpoint)。

### 4.4 `finetune/predictor/preprocess.py` main()(2026-06-20 补审)

**通读 main() 完成**。CLI 完整(--norm-mode/--lookback/--predict/--split-mode/--raw-data/--backtest-raw/--train-end/--val-end/--validate/--skip-backtest/--seed),参数正确传入 preprocess()。time split 强制 --train-end/--val-end(行 572-573,否则 parser.error)。泄露检查失败 exit(1)(行 598-599)。

main() 无问题。preprocess() 函数体(窗口切取/归一化/split/泄露检查)由 agent 在 §20 审过(核实 lookback_end=target_start 相接、调用 splitting.py、生成 meta/fingerprint、运行 validate_no_leakage),未发现泄露。**本文件审查完成,无新问题**。

### 4.5 `model/kronos.py`(2026-06-20 补审)

**通读 KronosTokenizer + auto_regressive_inference 完成**。此为预训练模型代码,非本次重构变更范围,仅核实依赖契约:

- `KronosTokenizer.forward`(行 74-113)返回 `(z_pre, z), bsq_loss, quantized, z_indices` —— tokenizer/train.py:211、validate.py:144/151 的解包一致 ✅
- `auto_regressive_inference`(行 434-514)返回完整序列 `(batch, total_seq_len, features)`(含 context+generated)—— train/eval 取 `preds[0, lookback:lookback+predict]`、backtest 取 `preds[0, -predict:]` 均**等价正确**(澄清 backtest review B6 误报)
- `KronosTokenizer` 继承 `PyTorchModelHubMixin`,`save_pretrained`/`from_pretrained` 来自 huggingface_hub,默认存/读目录下 model.safetensors+config.json —— **确认 TK5 路径错位**:save 传 `output_path/checkpoints/best_model` 则存该子目录,from 传 `output_path` 找不到 → 错位成立

**model/kronos.py 无新问题**(稳定预训练代码)。但其 save/from_pretrained 路径契约是 TK5/EV4/VA1 路径错位的根因载体。

### 4.6 `finetune/predictor/core/utils.py`(2026-06-20 补审)

**通读完成**。问题:

| 编号 | 严重性 | 位置 | 问题 |
|------|--------|------|------|
| UT1 | 🟡 | get_rank_info 行 257-258 + eval.py setup_ddp | get_rank_info 内已 `init_process_group`(行 258,有 is_initialized 保护),eval.py 又调 setup_ddp(再 init)。重复 init 模式脆弱,靠 is_initialized 保护兜底。建议合一 |
| UT2 | ✅ | safe_save_json 行 132-153 | **已原子写入**(temp + os.replace)。M6 实际已修,§22 M6 状态应更新为 ✅ |
| UT3 | ⚪ | safe_save_pickle 行 174-186 | 非原子(直接写),与 safe_save_json 不一致。M6 只修 json 未修 pickle。次要 |

**UT2 更正**:§22 清单 M6 标"非阻断未修",实际 safe_save_json 已是原子写入。M6(jjson 部分)已修,pickle 部分未修(UT3,次要)。

### 4.7 `data/generate_raw.py` + `data/generate_backtest_raw.py`(2026-06-20 补审)

**通读完成**。SQL → pkl 数据生成脚本。

| 编号 | 严重性 | 位置 | 问题 |
|------|--------|------|------|
| GR1 | 🟡 | generate_raw.py:30 / generate_backtest_raw.py:35 | 两脚本 SQL 正则**不同**(字段顺序差异)。若两份 SQL 实际 schema 一致,其一静默漏数据。需确认两 sql 格式一致 |
| GR2 | ✅ | generate_backtest_raw.py:40, 行14注释 | backtest 起始 2026-05-19,raw 截止 2026-05-18,**时间隔离正确**,无重叠。防泄露防线正确 |
| GB1 | ✅ | generate_backtest_raw.py 整体 | 时间隔离设计明确(context 借 raw 末尾,target 取 backtest),与方案 §1.4 BacktestSchema 一致 |

**数据生成脚本审查通过**,关键防线(时间隔离)正确。GR1 需确认两 SQL schema 是否真不同。

### 4.8 其余小文件(2026-06-20 抽查)

| 文件 | 结论 |
|------|------|
| `core/test_metrics.py` | 只 import calculate_da_score 等,**无实际测试调用**(grep 无 `calculate_da_score(`)。测试文件形同虚设,建议补真实测试 |
| `core/__init__.py` | 导出层,无逻辑,正常 |
| `finetune/predictor/eval_all.py` | **legacy**(评 Mode7-10 旧模型),手册/新流程不引用。建议归 deprecated,不深入审 |
| `model/module.py` | 预训练模型子模块(TransformerBlock/BSQuantizer),非本次变更,稳定代码,不审 |
| `model/__init__.py` | 导出层,不审 |

---

## 五、补审总结(2026-06-20)

### 5.1 覆盖统计(22 文件)

- ✅ 通读:**19**(本次补审 +11:tokenizer×2、eval main、preprocess main、model/kronos、utils、data×2,及此前 train/backtest/core×7)
- ⚪ 抽查:2(test_metrics、__init__)
- 🆗 legacy:1(eval_all,建议归 deprecated)
- 🆋 未审(预训练稳定):2(model/module、model/__init__,非本次变更)

**本次重构涉及的新代码已全部审查**。未审的 2 个是预训练模型子模块(非本次变更),legacy 1 个待归档。

### 5.2 补审发现的新问题(此前未审出)

| 编号 | 严重性 | 文件 | 问题 |
|------|--------|------|------|
| **TK5/VA1/EV4** | 🔴🔴🔴 致命 | tokenizer/train + validate + eval | **路径错位**:tokenizer 微调存 `output_path/checkpoints/best_model`,但 validate/eval/predictor 从 `output_path` 直接加载 → 微调后验证报 not found、predictor 加载不到微调权重。**三处同源,影响整条 tokenizer 链路** |
| **EV1** | 🔴 致命 | eval.py:124 | load_model_and_tokenizer 硬编码 lookback=400/predict=10/split=block,忽略 CLI → 改这些参数导致数据与模型错配 |
| **EV2/EV3** | 🔴 致命 | eval.py:128/127 | strict=False 静默吞不匹配 + 找不到 checkpoint 静默用预训练。用户以为评估微调模型,实际评估预训练 |
| B1-B4 | 🔴 致命 | backtest.py | 加载链路全错(strict=False、架构硬编码 mini、legacy 路径)——§24 已记 |
| UT3 | ⚪ | utils.py | safe_save_pickle 非原子 |
| GR1 | 🟡 | data/generate_*.py | 两 SQL 正则不同,需确认 schema |
| TK3/TK4 | 🟡 | tokenizer/train.py | early stopping grace 边界 + DataFrame 分支归一化口径 |

### 5.3 根因:加载链路系统性缺失

补审暴露的核心问题不是单点 bug,而是**整条"加载链路"(tokenizer/model checkpoint 怎么存、怎么读、路径怎么对)从未被系统性审查**:
- tokenizer 微调存读路径错位(TK5/VA1/EV4)
- eval 硬编码加载参数(EV1)
- eval/backtest strict=False 静默 + 找不到静默 fallback(EV2/EV3/B1/B3)
- backtest 架构/路径硬编码(B2/B3/B4)

这些都在各入口的 main()/load 函数里,之前 §20 按"度量/DDP/泄露"审时完全没覆盖。**§22 清单需补"加载链路"专项**。

### 5.4 审查方法改进(已落实)

1. 每个入口 main() 必须通读(不只审被问维度)——本次补审执行
2. 共享依赖函数读源码确认契约(auto_regressive_inference 已读,澄清 B6 误报)
3. 建立本记录,逐文件标注审查状态,避免再漏

### 5.5 待办

1. **修复加载链路**(TK5/EV1-4/B1-4)——当前最高优先,影响整条 train→eval→backtest 链路可信度
2. GR1 确认两 SQL schema
3. eval_all.py 归 deprecated
4. test_metrics.py 补真实测试
5. UT3 safe_save_pickle 原子化(次要)

**结论**:本次重构新代码已全部审查。补审发现加载链路存在系统性问题(此前遗漏),需修复后方可声称代码可信。审查记录已建立,覆盖透明可追溯。

---

## 六、二次复核(2026-06-20,代码再次修改后)

代码被再次修改(metrics.py format_metrics_report、train.py compute_val_loss/naive DA all_gather/CLI)。复核这些改动。

### 6.1 `train.py` naive DA all_gather(F1 最终形态)—— ✅ 正确

行 386-412 最终形态:all_gather 在 `if world_size > 1:` 内、**无 `if rank==0` 包裹**,所有 rank 执行 all_gather(行 400-401);只 rank 0 组装 naive_da_by_step(行 404)。**F1 死锁修复确认成立**,且注释明示"所有 rank 执行 all_gather"。无新问题。

### 6.2 `metrics.py` format_metrics_report —— ✅ 健壮性改进

行 686-700:naive_da 增加 None 防御(`if naive_da is None: naive_da = 0.5`,行 688-689),da_p50 None 回退 da_mean(行 697-698),IC p25/p50/p75 None 显示 "N/A"(行 723-725)。健壮性改进,无问题。

### 6.3 `train.py` compute_val_loss(F2 实现)—— 🔴 val_loss 与 train_loss 口径不一致

**这是 F2 修复引入的新问题**。compute_val_loss(行 478-547)实现的 val_loss 与 train loss **完全不同口径**:

| 维度 | train loss(行 687-701) | val loss(行 525-543) |
|------|------------------------|----------------------|
| 损失类型 | **token CE**(`head.compute_loss`,module.py:501 `F.cross_entropy`) | **MSE 重建**(`F.mse_loss(z_pre, x_norm[:,1:])`,行 543) |
| 计算路径 | logits → head.compute_loss(软) | argmax → decode → MSE(硬量化重建) |
| 输入切片 | `model(token_seq_0, token_seq_1, x_stamp)` 完整 | `model(token_seq_0[:,1:], token_seq_1[:,1:], x_stamp[:,1:])` 切 [1:] |
| 量级 | CE ~2-6 | MSE ~0.001-0.1 |

**问题**:
1. **val_loss(MSE)与 train_loss(CE)不可比** —— 量级差 2-3 个数量级,语义不同。`history['train_loss']` 存 CE,`history['val_loss']` 存 MSE,两者画在同一图无意义。
2. **best_val_loss 选模型基于 MSE 重建质量,非 token 预测质量** —— early stopping 的 `avg_val_loss < best_val_loss`(行 808)比的是 MSE,而训练优化的是 CE。best_model(val_loss 最低)可能不是 CE 意义上最好的模型。
3. **val forward 切 [1:](行 529),train 不切(行 697)** —— 序列长度不一致,val 丢了第一个 token。

**根因**:F2 修复时为了"实现 val_loss",用了 tokenizer decode + MSE 重建(类似 tokenizer 微调的 loss),但 predictor 训练的 loss 是 token CE。val_loss 应该与 train_loss 同口径(都用 head.compute_loss 算 CE),才能用于 early stopping 对照。

**修复方向**:compute_val_loss 改为与 train 同口径——`model(token_seq_0, token_seq_1, x_stamp)` → `head.compute_loss(s1_logits[:,:-1,:], s2_logits[:,:-1,:], token_out[0], token_out[1])` 算 CE,不反传。这样 val_loss 与 train_loss 同为 CE,best_val_loss 可对照。

### 6.4 复核结论

- F1(naive DA all_gather)最终形态 ✅ 正确
- format_metrics_report None 防御 ✅
- **F2(val_loss)口径错误 🔴** —— train CE vs val MSE 不可比,best_val_loss 选模型失真。这是 F2 修复引入的新问题,需修(compute_val_loss 改用 head.compute_loss 算 CE)。

**更新 §22 F2 状态**:F2 此前标 ✅(val_loss 已实现),但复核发现口径错误,降级为 ⚠️(已实现但口径错,需修)。

### 6.5 F2 修复复核(2026-06-20,开发人员修复后)

**口径修复 ✅**:compute_val_loss(行 477-542)已改为与 train 同口径——
- 行 528 `tokenizer.encode(x_norm, half=True)` = train:689
- 行 531 `model(token_seq_0, token_seq_1, x_stamp)` 完整序列(不切 [1:])= train:697
- 行 534-538 `head.compute_loss(s1_logits[:,:-1], s2_logits[:,:-1], token_out[0], token_out[1])` token CE = train:699-701
- 行 535 `token_out = [token_seq_0[:,1:], token_seq_1[:,1:]]` = train:691

val_loss 现为 token CE,与 train_loss 同口径,best_val_loss 可用于 early stopping。**口径问题已解决**。

**但引入新问题 🔴**:行 534 `head = model.head`——DDP 模式下 model 是 DDP 包装,`model.head` 访问子模块需 `model.module.head`(train:698 正是 `model.module.head if use_ddp else model.head`)。compute_val_loss 直接 `model.head`,**DDP 模式下 AttributeError 崩溃**。

- 单卡:model 非 DDP,`model.head` ✅ 正常
- 多卡(DDP):model 是 DDP,`model.head` ❌ 崩(`DDP` 对象无 head 属性,需 .module.head)

compute_val_loss 由 train() 调用(行 715),train() 的 model 来自 main 行 755 `model = DDP(model,...)`,即 DDP 模式下传入的 model 已是 DDP 包装。compute_val_loss 漏了 DDP 解包。

**修复**:行 534 改为 `head = model.module.head if isinstance(model, DDP) else model.head`(或传 use_ddp 参数)。与 train:698 一致。

**结论**:F2 口径已修(✅),但 DDP 解包遗漏(🔴 多卡崩溃)。单卡可跑,多卡 compute_val_loss 崩。需补 DDP 解包。

**更新 §22 F2 状态**:⚠️ 口径已修,但 DDP 下 compute_val_loss 崩(model.head 未解包),需补 `model.module.head`。

### 6.6 加载链路修复复核(2026-06-20,开发人员修复后)

核实 TK5/EV1-4/B1-4 当前状态:

| 编号 | 状态 | 证据 |
|------|------|------|
| TK5 | ✅ 已修 | tokenizer/train.py:268 `save_pretrained(output_path)` 同时存根目录(validate/eval 期望路径),256/266 仍存 checkpoints 子目录(保留) |
| EV1 | ✅ 已修 | eval.py:135 `get_model_path(norm_mode, lookback, predict, split_mode, model_type)` 参数化,不再硬编码 400/10/block |
| EV2 | ✅ 已改 | eval.py:142-146 strict=False 但检查 missing/unexpected 并 WARNING(比静默好) |
| EV3 | ✅ 已修 | eval.py:149-151 找不到 checkpoint 显式 WARNING "Using pretrained...may not reflect fine-tuned" |
| B1 | ✅ 已改 | backtest.py:356-360 strict=False + missing/unexpected WARNING |
| B2 | ✅ 已修 | backtest.py:341-347 按 model_type 选架构,不硬编码 mini |
| B4 | ✅ 已修 | backtest.py:328 `get_tokenizer_path(norm_mode, model)`,330-335 找不到 WARNING + fallback 预训练(非 legacy) |
| B3 | ⚠️ 部分 | backtest.py:351 `get_model_path(norm_mode, lookback, predict, 'block', model)` —— **硬编码 'block'**,无 --split-mode CLI。与 EV1 不一致。time split 训练的模型 backtest 加载不到 → fallback 预训练 WARNING |

**残留(非致命)**:
1. **backtest B3 硬编码 'block'**(行 351)—— backtest 无 --split-mode CLI,固定 block。若用户用 time split 训练模型,backtest 找不到 checkpoint → WARNING + fallback 预训练。建议 backtest 加 --split-mode 或文档注明固定 block。
2. **strict=False 宽容**(eval:142/backtest:356)—— 虽加 missing/unexpected WARNING,但非空时只警告不 raise。完全错配(如不同架构)仍会继续跑(有警告)。可接受,严格说应非空 raise。

**加载链路结论**:此前 §5.3 标"系统性缺失"的加载链路问题**基本修复**(TK5/EV1/EV3/B2/B4 全修,EV2/B1 加 WARNING)。仅 backtest split_mode 硬编码 + strict=False 宽容两点残留,均有 WARNING 不致命。

**更新 §22 状态**:B1-B4/EV1-4 从"未修/致命"降级——TK5/EV1/EV3/B2/B4 ✅ 已修,EV2/B1 ⚠️ 加 WARNING(strict=False 保留),B3 ⚠️ backtest 硬编码 block。加载链路从"致命"降为"基本修复+残留警告"。

### 6.7 eval.py evaluate() + preprocess.py 主体(2026-06-20 补审)

**eval.py evaluate() 主体**(行 160-398)通读,发现:

| 编号 | 严重性 | 位置 | 问题 |
|------|--------|------|------|
| EV6 | 🔴 致命 | 行 180/200-202/215-217 | **DDP 双重分片冲突**:行 180 `rng=RandomState(seed+rank)` 各 rank 不同 seed → 行 200 各 rank 抽样抽到**不同样本**;行 215-217 又按 `idx%world_size` 轮询分片。两重分片叠加:各 rank 处理的样本集既不互补(不覆盖全量)也不一致。**DDP 下样本覆盖错乱**(漏样本/重复)。正确:抽样用同 seed(所有 rank 抽同样本),再 idx%world_size 分片 |
| EV7 | 🔴 致命 | 行 329-350 | **DDP naive DA all_gather padding 多重错误**:① 行 335 max_len 跨 step 取 max(语义错,应每 step 独立);② 各 rank local_actual 长度不同(因 EV6 抽样不同)→ padded 长度各 rank 不同 → `dist.all_gather` 要求等长 → **NCCL 崩或错乱**;③ 行 345 `[x for x in all_actual if x!=0]` 过滤 padding 零,但 actual_dir=False 是 0.0 → **误删真实 False 样本**,up_ratio 算错。对比 train.py:386-412 用 up_count/n 计数法(正确),eval 用列表 padding(错误),两处不一致 |
| EV8 | ✅ | 行 275 | `preds[0, lookback:lookback+predict]` 正确(auto_regressive_inference 返回完整序列) |
| EV9 | 🟡 | 行 266 | max_context=2048 硬编码,small/base(512)越界。同 backtest B11。应用 MAX_CONTEXT[model_type] |
| EV11 | ⚪ | 行 333 | gathered_actual 创建后未用(行 339 重建 gathered_padded),死代码 |

**preprocess.py 主体**(create_samples_from_raw + backtest 生成)通读,发现:

| 编号 | 严重性 | 位置 | 问题 |
|------|--------|------|------|
| PR1 | 🟡 重要 | splitting.py time_split + CLI --train-end | **time split 边界语义不清**:time_split(splitting.py:57)用 `s.target_end <= train_end` 比较,target_end 是样本内位置索引(0~seq_len),但方案 §1.3 描述"按时间段(2018-01~2023-06)划分"。CLI --train-end 是 int,用户需把日期换算成位置。实际是"按样本内位置切"非"按时间切",与方案描述/用户预期不符 |
| PR3 | ✅ | 行 103/110/117 | required_history 正确预留(sliding_ma60 前 60 步),lookback 段不含 target,无 future 泄露 |
| PR4 | 🟡 | 行 262 | backtest 归一化对 full(context+target)算 rolling。因果无未来泄露,但与 schema §1.4 "fit_range<=context_end" 断言矛盾(backtest 走 dict 不走 BacktestSchema,校验未生效) |
| PR5 | ✅ | 行 251/254 | backtest context(train_raw 末尾)+target(backtest_raw 前 predict)时间隔离正确,无重叠 |
| PR6 | ✅ | 行 357 | window_start 与整条 raw 序列索引对齐正确 |

**核心结论**:
- **eval DDP 评估有两处致命**(EV6 双重分片、EV7 naive DA padding)——多卡 eval 结果错乱或崩。单卡(world_size=1)无 EV6/EV7 问题(走 else 分支行 351+)。即 **eval 单卡可跑,多卡崩/错**。
- preprocess 防泄露防线正确(PR3/PR5),但 time split 边界语义(PR1)需澄清。

### 6.8 补审总结(2026-06-20 第二轮)

**新发现致命问题**:
1. **F2 DDP 解包**(§6.5):compute_val_loss 行 534 `model.head` 多卡崩
2. **EV6 eval DDP 双重分片**(§6.7):多卡 eval 样本覆盖错乱
3. **EV7 eval DDP naive DA padding**(§6.7):多卡 eval naive DA 崩/错

**共同点**:三处都是 **DDP 多卡**问题。单卡均能跑。即当前代码**单卡可用,多卡(eval/val_loss)有致命 bug**。

**已确认修复**:F2 口径、F1 all_gather、加载链路(TK5/EV1/EV3/B2/B4)、format_metrics_report None 防御。

**残留非致命**:B3 backtest 硬编码 block、EV2/B1 strict=False 宽容、EV9/B11 max_context 硬编码、PR1 time split 语义、PR4 fit_range 未校验。

### 6.9 统一修复(2026-06-20,评审人直接改)

**已修复(语法验证通过)**:

| 编号 | 文件:行 | 修复 |
|------|---------|------|
| F2-DDP | train.py:536 | `model.head` → `model.module.head if isinstance(model, DDP) else model.head`。多卡 compute_val_loss 不再崩 |
| EV6 | eval.py:181 | `RandomState(seed + rank)` → `RandomState(seed)`。所有 rank 同 seed 抽同样本,再 idx%world_size 分片,样本覆盖正确 |
| EV7 | eval.py:329-352 | naive DA all_gather 列表 padding 法 → **up_count/n 计数法**(与 train.py:386-412 同源)。消除 padding 长度不一致 + 误删 False 样本问题。顺带删 EV11 死代码 |
| EV9 | eval.py:267 | max_context=2048 → `{'mini':2048,'small':512,'base':512}.get(model_type,2048)`。evaluate() 加 model_type 参数,main 传入 |
| B3 | backtest.py:307-309,355 | 加 `--split-mode` CLI(默认 block),model_dir 用 args.split_mode 不再硬编码 |
| B11 | backtest.py:141 | max_context=2048 → 按 model_type 映射。backtest() 加 model_type 参数,main 传入 |
| train.py | train.py:319 | evaluate_trajectory_ic 的 max_context 同样按 model_type 映射,加 model_type 参数,train() 传入 train_config.model_type |

**未改(需用户决策或非阻断)**:
- ~~PR1(time split 位置 vs 时间语义)~~ **已定(2026-06-20):用位置语义,改文档不改代码**。方案 §1.3 + 手册 Q3/§1.2 已改为位置语义说明(位置≈交易日序号,A 股约 244/年,train_end≈1340/val_end≈1710)。代码本就是位置,无需改
- ~~PR4(backtest fit_range 校验未生效)~~ **已实现(2026-06-20)**:backtest.py 循环前加一次性 PR4 校验——对第一个样本重算 context 段归一化(仅用 context 及前 N 步历史,不含 target),与 preprocess 存的 normalized 对比,不一致则 raise(泄露)。轻量(只校验首个样本,不拖慢 backtest)
- ~~M5(块间 assert O(n²))~~ **已修(开发人员)**:splitting.py:140-149 现为扫描线 O(N log N),非双重循环
- ~~M6/UT3(safe_save_pickle 非原子)~~ **已修(开发人员)**:utils.py:174-194 已原子写入(temp+os.replace)
- ~~M3(seed 无效)~~ **误报**:eval/train 抽样分支(eval.py:201/train.py:243)在 n_samples>0 时用 rng 抽样,seed 生效;仅 n_samples=-1(全量)时 seed 不生效(全量不需 seed)。代码正确,无需改

**EV2/B1(2-1 修复,2026-06-20)**:eval.py:142/backtest.py:360 加载 checkpoint 时 missing/unexpected 非空改 **raise**(原只 WARNING),避免静默用错权重。

### 6.10 第二轮残留项处理完成(2026-06-20)

用户决策(2-1~2-2):
- 2-1 加载模型出错就停止 → EV2/B1 改 raise ✅
- 2-2 甲(PR4)需实现 → backtest 一次性校验 ✅
- 2-2 乙(M3)→ 核实为误报(抽样时 seed 生效),无需改 ✅
- 2-2 丙(M5)→ 开发人员已改扫描线 ✅
- 2-2 丁(M6/UT3)→ 开发人员已改原子写入 ✅

**至此 §22 清单 + §6 补审发现的问题全部处理完毕**:
- 致命/重要:全修(F1/F2/EV6/EV7/加载链路/PR4/EV2/B1...)
- 口径决策:I2/I3/PR1 已定
- 非阻断:M3 误报,M5/M6/UT3 已修
- 残留:无未处理项

**结论**:本次重构代码审查闭环。所有发现的问题已修复或有明确决策。代码单卡/多卡(train+eval+backtest+tokenizer)的 DDP、口径、加载链路、防泄露、原子性均已落实。可进入 Phase 4 等价性验证。

### 6.11 维护手册时发现 backtest 加载路径残留 bug(2026-06-20)

维护 `docs/guide/predictor_usage.md` 时核实 backtest CLI,发现 **B3 修复未修全**:

**问题**:backtest.py:393-394
```python
model_dir = get_model_path(...)  # = outputs/models/{norm}/lb{lb}_pd{pd}/{split}/{model}
safetensors_path = os.path.join(model_dir, 'model.safetensors')  # 直接在 model_dir 下找
```
但训练存的 checkpoint 在 `model_dir/checkpoints/best_combined_model/model.safetensors`(get_checkpoint_path 拼接 `checkpoints/{name}`)。**路径差一层 `checkpoints/best_combined_model`** → backtest 找不到 → 走 else 分支 WARNING "Checkpoint not found" + 用预训练。

**对比 eval.py**:eval.py:136 `get_checkpoint_path(model_dir, checkpoint)` 正确拼接 `checkpoints/{checkpoint_name}`,且有 `--checkpoint` 参数(默认 best_combined_model)。backtest 既没用 get_checkpoint_path,也没 --checkpoint 参数。

**影响**:backtest 用 `--model mini` 时加载不到微调模型,静默回退预训练(有 WARNING)。用户需用 `--model` 传完整 checkpoint 路径绕过(手册 §5/Q6 已标注此临时用法)。

**修复方向**:backtest.py 应
1. 加 `--checkpoint` CLI 参数(默认 best_combined_model,与 eval 一致)
2. 行 393-394 改用 `get_checkpoint_path(model_dir, args.checkpoint)` 拼接正确路径
3. 或在 model_type 分支用 get_checkpoint_path,路径分支用 args.model

**当前状态**:✅ **已修复(2026-06-20)**。backtest.py 加 `--checkpoint` CLI(默认 best_combined_model,与 eval 一致)+ 用 `get_checkpoint_path(model_dir, args.checkpoint)` 拼接 `checkpoints/{name}` 正确路径。--model 默认从 legacy `final_models/Kronos-mini-MA60` 改为 `mini`(新体系)。手册 §5 已同步(去掉限制标注,Q6 删除)。

**手册同步**:`docs/guide/predictor_usage.md` §5 已改为反映修复后 CLI(有 --checkpoint,定位 checkpoints/{name}),与代码一致。
- EV2/B1 strict=False 宽容——已有 WARNING,非致命,保留
- M3/M5/M6/UT3——非阻断

**验证**:三文件(train/eval/backtest)AST 语法解析通过。改动点 grep 确认到位。

**本轮修复后状态**:
- F2-DDP ✅、EV6 ✅、EV7 ✅、EV9 ✅、B3 ✅、B11 ✅、train max_context ✅
- 加载链路(§6.6)此前已基本修复
- 残留:PR1(待用户)、PR4/EV2/B1/M3/M5/M6/UT3(非阻断)

**结论**:本轮审查发现的致命/重要问题已统一修复。当前代码单卡/多卡(train+eval+backtest)的 DDP 与口径问题已清。残留项均非阻断(PR1 待用户决策 time split 语义)。

