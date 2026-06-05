# Kronos 微调模块

本目录包含 Kronos 模型的微调训练、评估、推理和数据预处理脚本。

## 目录

- [脚本索引](#脚本索引)
- [模型版本表](#模型版本表)
- [数据格式](#数据格式)
- [评估指标](#评估指标)
- [已知问题](#已知问题)

## 脚本索引

### 数据准备

| 脚本 | 用途 |
|------|------|
| `csv_data_preprocess.py` | CSV 格式 K 线数据预处理 |
| `clean_sql_data.py` | SQL 数据源清洗 |
| `split_by_quarter.py` | 按季度拆分数据 |
| `merge_quarters.py` | 合并季度数据 |
| `merge_all_quarters.py` | 合并全部季度 |
| `merge_full_data.py` | 合并全量数据 |
| `final_merge_quarters.py` | 最终季度合并 |
| `reorganize_2025.py` | 重组 2025 年数据 |
| `verify_new_data.py` | 验证新数据完整性 |
| `filter_stocks.py` | 按条件筛选股票 |
| `precompute_ma60_normalization.py` | 预计算 MA60 滑动归一化统计量 |
| `preprocess_kline_daily_ma60.py` | 日 K 线 MA60 归一化预处理 |
| `preprocess_windowed.py` | 窗口化预处理（支持 `--lookback` 参数，密集采样 stride=10，时间外推测验证/测试集划分） |
| `incremental_update_ma60.py` | MA60 数据增量更新 |

### 分词器训练

| 脚本 | 用途 |
|------|------|
| `train_tokenizer.py` | 原始分词器训练（内置配置，支持 mid/full/small/mid_small 数据集） |
| `train_tokenizer_ma60_base.py` | MA60 分词器训练（small/base 模型用） |
| `validate_tokenizer.py` | 分词器重建误差验证 |
| `test_ma60_tokenizer.py` | MA60 分词器测试 |

### 预测器训练

| 脚本 | 用途 |
|------|------|
| `train_predictor.py` | 原始预测器训练（mini 模型，整体窗口归一化） |
| `train_predictor_ma60.py` | MA60 预测器训练（mini 模型，lookback=400） |
| `train_predictor_ma60_small.py` | MA60 预测器训练（small 模型，lookback=200，6 维全特征评估，即将添加 PCGrad 梯度手术） |

### 评估与基准

| 脚本 | 用途 |
|------|------|
| `benchmark_models.py` | 多模型基准测试（每模型独立 lookback/数据集/分词器，6 维全特征指标对比） |
| `unified_test.py` | 统一测试脚本（固定随机种子） |
| `full_comparison_test.py` | 完整对比测试 |
| `full_market_test.py` | 全市场测试 |
| `model_comparison_test.py` | 模型对比测试 |
| `test_complete_comparison.py` | 完整对比测试 |
| `compare_ma60_vs_orig.py` | MA60 与原始归一化对比 |
| `validate_prediction.py` | 预测结果验证 |
| `check_format.py` | 数据格式检查 |
| `check_stocks_info.py` | 股票信息检查 |

### 批量推理

| 脚本 | 用途 |
|------|------|
| `batch_inference_sql.py` | SQL 数据源批量推理 |
| `batch_inference_sql_part2.py` | SQL 推理 Part 2 |
| `batch_inference_sql_part2_cont.py` | SQL 推理 Part 2 续 |
| `batch_inference_sql_part3.py` | SQL 推理 Part 3 |
| `batch_inference_sql_part4.py` | SQL 推理 Part 4 |
| `batch_inference_from_preprocessed.py` | 从预处理数据批量推理 |
| `batch_inference_incremental.py` | 增量推理 |
| `batch_inference_ma60_best.py` | MA60 最佳模型推理 |
| `batch_inference_ma60_optimized.py` | MA60 优化推理 |
| `high_throughput_inference.py` | 高吞吐推理 |

### 诊断与测试

| 脚本 | 用途 |
|------|------|
| `analyze_data_distribution.py` | 6 维特征数据分布分析（归一化统计、原始收益率、量级、熵、MA60 局部统计） |
| `test_time_extrapolation.py` | 时间外推测试 |
| `test_clip.py` | Clip 范围测试 |
| `test_bit_mask.py` | 位掩码测试 |
| `test_inference.py` | 推理功能测试 |

### 核心模块

| 文件 | 用途 |
|------|------|
| `config.py` | 配置管理（路径、超参数） |
| `dataset.py` | 数据集类定义 |

## 模型版本表

| 版本 | 模型 | 分词器 | 数据集 | lookback | freeze | 方向损失权重 | close IC@step3 | 状态 |
|------|------|--------|--------|----------|--------|------------|---------------|------|
| mini-v5 | Kronos-mini (4.1M) | ma60_tokenizer_v1 | v3 (400) | 400 | layer 4 + emb | 0.3 | ~0.20 | 稳定 |
| small-v5a | Kronos-small (24.7M) | ma60_tokenizer_base_v1 | v3 (400) | 400 | layer 4 + emb | 0.3 | 崩溃 | 废弃 |
| small-v5b | Kronos-small (24.7M) | ma60_tokenizer_base_v1 | v3_small (200) | 200 | layer 4 + emb | 0.3 | -0.218 | IC 崩溃 |

**关键发现**: small 模型 CE loss 持续下降但 IC 崩溃，原因是 vol/amt 梯度主导训练（vol IC=0.74 vs close IC=0.20）。下一步采用 PCGrad 梯度手术解决特征间梯度冲突。

## 数据格式

### 预处理数据 (pkl)

每个股票一个字典：
```python
{
    'normalized': np.ndarray,  # shape (T, 6), 归一化后的 OHLCVA
    'original':  np.ndarray,  # shape (T, 6), 原始数据
    'means':     np.ndarray,  # shape (T, 6), MA60 滑动均值
    'stds':      np.ndarray,  # shape (T, 6), MA60 滑动标准差
    'dates':     np.ndarray,  # shape (T,),  日期
}
```

6 维特征顺序: `[open, high, low, close, vol, amt]`

### 数据集版本

| 数据集 | lookback | 路径 | 说明 |
|--------|----------|------|------|
| v3 | 400 | `processed_datasets_ma60_windowed_v3` | mini 模型用 |
| v3_small | 200 | `processed_datasets_ma60_windowed_v3_small` | small 模型用（max_context=512 限制） |

## 评估指标

| 指标 | 含义 | 理想值 |
|------|------|--------|
| IC | Pearson 相关系数（预测值 vs 真实值） | > 0.05 |
| RankIC | Spearman 秩相关 | > 0.05 |
| ICIR | IC / IC标准差 | > 0.5 |
| DA | 方向准确率 (Direction Accuracy) | > 50% |
| DDA | 方向变化准确率 | > 50% |
| NMSE | 归一化均方误差 | 越小越好 |
| pred_bias | 预测偏差 (均值差) | 接近 0 |
| var_ratio | 预测方差 / 真实方差 | 接近 1 |

**注意**: IC 计算必须经过 decode + denormalize，不能直接用 token indices。

## 已知问题

### IC-VL 解耦（small 模型）

small 模型训练中 CE loss 持续下降，但 IC 崩溃至负值。根因是 vol/amt 梯度主导训练：

- vol/amt 统计特性更易预测（原始收益率峰度 3.45 vs 价格 6.7）
- 归一化后各特征分布相似，但原始尺度差异导致梯度量级不均
- 形成梯度冲突：vol/amt 的梯度方向与 price 特征不一致

**解决方案**: PCGrad（Yu et al., NeurIPS 2020）— 投影冲突梯度到法平面。参考论文: `docs/reference/2001.06782v4.pdf`

### small 模型 lookback 限制

Kronos-small max_context=512，lookback=200 时总 token 数=420（6 特征 × 200 + 20 步预测），已达上限。无法使用 lookback=400。

## 输出路径

- 训练模型: `outputs/models/`
- 最终模型: `final_models/`
- 训练日志: `outputs/training_logs/`
- 经验文档: `final_models/TRAINING_EXPERIENCE.md`
