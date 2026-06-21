"""
Kronos Predictor Training Entry

统一训练入口，支持：
- 多种 norm_mode（full_window/sliding_ma{N}）
- 多种 split_mode（time/block）
- 多种 model_type（mini/small/base）
- DDP 多卡训练
- 正确度量口径（detrended trajectory IC）

使用：
    # 单卡
    python finetune/predictor/train.py --norm-mode sliding_ma60 --model mini --epochs 50

    # 多卡
    torchrun --nproc_per_node=4 finetune/predictor/train.py --norm-mode sliding_ma60 --model mini

关键：
- 与 eval.py 共用 core/metrics.py
- trajectory IC 用去趋势序列（禁止原始价格）
- checkpoint 选择用正确口径 IC
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

from model.kronos import Kronos, KronosTokenizer

from finetune.predictor.core.config import DataConfig, TrainConfig, ArtifactConfig
from finetune.predictor.core.paths import (
    get_tokenizer_path,
    get_model_path,
    get_checkpoint_path,
    get_training_info_path,
    get_summary_path,
    get_split_data_path,
    ensure_dir,
)
from finetune.predictor.core.dataset import KronosDataset, KronosWindowedDataset, collate_fn
from finetune.predictor.core.metrics import (
    FEATURE_NAMES,
    detrend_to_baseline,
    safe_trajectory_ic,
    safe_corrcoef,
    safe_spearmanr,
    excess_da,
    aggregate_ic,
    aggregate_da,
    calculate_combined_score,
    calculate_da_score,
    get_log_step_weights,
    get_feature_weights,
    format_metrics_report,
    # §10 新增：可懂指标
    amplitude_error_rate,
    detect_limit,
    limit_hit_rate,
)
from finetune.predictor.core.utils import (
    set_seed,
    get_device,
    format_time,
    get_model_size,
    get_rank_info,
    cleanup_ddp,
    debug_print,
    safe_save_json,
    safe_save_pickle,
)


# ============================================================================
# 配置
# ============================================================================

# T2 修复：删除杜撰的 vocab 映射，改为正确的架构映射
# 实际 mini→Kronos-Tokenizer-2k，small/base→Kronos-Tokenizer-base
# vocab_size 由预训练架构决定，不是配置参数
TOKENIZER_ARCH = {
    'mini': 'Kronos-Tokenizer-2k',
    'small': 'Kronos-Tokenizer-base',
    'base': 'Kronos-Tokenizer-base',
}

MODEL_PATHS = {
    'mini': 'pretrained/Kronos-mini',
    'small': 'pretrained/Kronos-small',
    'base': 'pretrained/Kronos-base',
}

MAX_CONTEXT = {
    'mini': 2048,
    'small': 512,
    'base': 512,
}


# ============================================================================
# 训练信息记录
# ============================================================================

def init_training_info(output_dir: str, config: DataConfig, train_config: TrainConfig) -> str:
    """
    初始化 training_info.json

    §10 更新：best 跟踪扩展到 5 项 + 可懂指标
    """
    info = {
        'status': 'running',
        'start_time': datetime.now().isoformat(),
        'config': {
            'norm_mode': config.norm_mode,
            'lookback': config.lookback,
            'predict': config.predict,
            'split_mode': config.split_mode,
            'model_type': train_config.model_type,
            'epochs': train_config.epochs,
            'batch_size': train_config.batch_size,
            'learning_rate': train_config.learning_rate,
        },
        'data': {
            'train_samples': 0,
            'val_samples': 0,
            'data_fingerprint': '待更新',
            'target_leakage_check': '待更新',
        },
        'tokenizer': {
            'path': '',
            'data_fingerprint': '待更新',
        },
        'epochs': [],
        # §10 更新：best 跟踪扩展到 5 项 + 可懂指标
        'best': {
            'val_loss': {'value': float('inf'), 'epoch': 0},
            'ic': {'value': -999, 'epoch': 0},
            'da_score': {'value': 0.0, 'epoch': 0},
            'excess_da': {'value': 0.0, 'epoch': 0},
            'combined': {'value': 0.0, 'epoch': 0},
            'amplitude_error_rate': {'value': None, 'epoch': 0},
            'limit_hit_rate': {'value': None, 'epoch': 0},
        },
    }

    info_path = get_training_info_path(output_dir)
    ensure_dir(info_path)
    safe_save_json(info, info_path)

    return info_path


def update_training_info(info_path: str, epoch: int, metrics: dict, best_updates: dict = None):
    """
    更新 training_info.json

    §10 更新：
    - metrics 含 da_score/excess_da/可懂指标（可懂指标在非 best 时为 None）
    - best_updates 为本 epoch 刷新的 best 列表，如 {'val_loss': (2.28, 7), 'ic': (0.21, 7)}
    """
    # 读取现有 info
    info = {}
    try:
        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        info = {}

    # epoch 记录：所有关键指标并列
    epoch_record = {
        'epoch': epoch,
        'train_loss': metrics.get('train_loss', 0),
        'val_loss': metrics.get('val_loss', 0),
        'ic': metrics.get('ic', 0),
        'da_score': metrics.get('da_score', 0),
        'excess_da': metrics.get('excess_da', 0),
        'combined': metrics.get('combined', 0),
        'lr': metrics.get('lr', 0),
        # 可懂指标：仅本 epoch 产生 best 时才有值，否则 None
        'amplitude_error_rate': metrics.get('amplitude_error_rate'),
        'limit_hit_rate': metrics.get('limit_hit_rate'),
        'time': datetime.now().isoformat(),
    }
    info.setdefault('epochs', []).append(epoch_record)

    # 更新 best：每个关键指标独立记 best 值 + best epoch
    if best_updates:
        info.setdefault('best', {})
        for metric_name, (value, best_epoch) in best_updates.items():
            info['best'][metric_name] = {'value': value, 'epoch': best_epoch}

    safe_save_json(info, info_path)


# ============================================================================
# 评估函数（使用正确度量口径）
# ============================================================================

def evaluate_trajectory_ic(
    model,
    tokenizer,
    val_data,
    indices,
    config: DataConfig,
    device: torch.device,
    world_size: int = 1,
    rank: int = 0,
    n_samples: int = 500,
    seed: int = 42,
    model_type: str = 'mini'
) -> dict:
    """
    Trajectory IC 评估（正确口径：去趋势序列）

    §10 更新：添加可懂指标收集（amplitude_error_rate / limit_hit_rate）

    关键：
    - pred/actual 必须先 detrend_to_baseline
    - 禁止直接用原始价格
    - 聚合保留分布（mean/std/p25/p50/p75）
    """
    from model.kronos import auto_regressive_inference

    model.eval()
    tokenizer.eval()

    # DDP 解包：auto_regressive_inference 内部调 model.decode_s1/decode_s2，
    # DDP 包装对象无这些方法，需用 model.module
    raw_model = model.module if isinstance(model, DDP) else model

    rng = np.random.RandomState(seed)

    # 抽样窗口
    if n_samples > 0 and n_samples < len(indices):
        sample_idx = rng.choice(len(indices), size=n_samples, replace=False)
        eval_indices = [indices[i] for i in sample_idx]
    else:
        eval_indices = indices

    # 分片：每个 rank 处理一部分
    per_rank = len(eval_indices) // world_size
    start_idx = rank * per_rank
    end_idx = start_idx + per_rank if rank < world_size - 1 else len(eval_indices)
    local_indices = eval_indices[start_idx:end_idx]

    # 本地 IC 收集
    local_ics = {f: [] for f in FEATURE_NAMES}
    local_rics = {f: [] for f in FEATURE_NAMES}
    local_da = [{f: [] for f in FEATURE_NAMES} for _ in range(config.predict)]

    # 收集 actual_dir 统计（用于计算 naive DA）
    local_actual_dir = [[] for _ in range(config.predict)]  # 仅 close

    # §10 新增：可懂指标收集
    local_amplitude_rates = []  # 振幅误差率
    local_pred_limit = []       # 预测涨跌停
    local_actual_limit = []     # 实际涨跌停
    limit_pct = 0.10  # 主板涨跌停阈值（默认 10%）

    for (symbol, window_start) in local_indices:
        d = val_data[symbol]
        window_end = window_start + config.lookback + config.predict

        try:
            # 取数：统一走 extract_window，支持 DataFrame / dict time / dict block 三种格式
            from finetune.predictor.core.dataset import extract_window
            normalized, original_vals, means, stds, timestamps = extract_window(d, window_start, window_end)

            if normalized is None:
                # DataFrame 格式（full_window runtime 归一化）
                x_raw = original_vals[:config.lookback]
                x_mean = np.mean(x_raw, axis=0)
                x_std = np.std(x_raw, axis=0) + 1e-5
                x_norm = np.clip((x_raw - x_mean) / x_std, -config.clip, config.clip)
                original = original_vals
            else:
                # dict 格式（time / block，已预归一化）
                x_norm = normalized[:config.lookback].astype(np.float32)
                original = original_vals

            baseline = original[config.lookback - 1]

            # 时间戳特征
            x_stamp = np.stack([
                timestamps[:config.lookback].minute.values,
                timestamps[:config.lookback].hour.values,
                timestamps[:config.lookback].weekday.values,
                timestamps[:config.lookback].day.values,
                timestamps[:config.lookback].month.values,
            ], axis=1).astype(np.float32)

            y_stamp = np.stack([
                timestamps[config.lookback:].minute.values,
                timestamps[config.lookback:].hour.values,
                timestamps[config.lookback:].weekday.values,
                timestamps[config.lookback:].day.values,
                timestamps[config.lookback:].month.values,
            ], axis=1).astype(np.float32)

            with torch.no_grad():
                x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(device)
                x_stamp_tensor = torch.from_numpy(x_stamp).unsqueeze(0).to(device)
                y_stamp_tensor = torch.from_numpy(y_stamp).unsqueeze(0).to(device)

                preds = auto_regressive_inference(
                    tokenizer, raw_model,
                    x_tensor, x_stamp_tensor, y_stamp_tensor,
                    max_context={'mini': 2048, 'small': 512, 'base': 512}.get(model_type, 2048),  # 按 model_type
                    pred_len=config.predict,
                    clip=config.clip,
                    T=1.0,
                    top_p=0.9,
                    sample_count=1,
                    verbose=False
                )

                pred_norm = preds[0, config.lookback:config.lookback + config.predict, :]

                # 反归一化：preds 已是 numpy（auto_regressive_inference 返回 np.ndarray）
                # dict 格式用预计算的 means/stds（block 模式已是窗口内相对切片），
                # DataFrame 格式用 runtime 的 x_std/x_mean
                if normalized is not None:
                    pred_raw = pred_norm * stds[config.lookback:] + means[config.lookback:]
                else:
                    pred_raw = pred_norm * x_std + x_mean

                actual = original[config.lookback:config.lookback + config.predict]

            # Trajectory IC（去趋势口径）
            for fi, fn in enumerate(FEATURE_NAMES):
                baseline_val = baseline[fi]

                # 去趋势
                pred_detrend = detrend_to_baseline(pred_raw[:, fi], baseline_val)
                actual_detrend = detrend_to_baseline(actual[:, fi], baseline_val)

                ic, rank_ic = safe_trajectory_ic(pred_detrend, actual_detrend)

                if ic is not None:
                    local_ics[fn].append(ic)
                if rank_ic is not None:
                    local_rics[fn].append(rank_ic)

            # DA
            for step_idx in range(config.predict):
                for fi, fn in enumerate(FEATURE_NAMES):
                    pred_dir = (pred_raw[step_idx, fi] - baseline[fi]) > 0
                    actual_dir = (actual[step_idx, fi] - baseline[fi]) > 0
                    # 存 float（非 bool）：聚合时 np.percentile 对 bool 做插值会崩溃
                    local_da[step_idx][fn].append(float(pred_dir == actual_dir))

                    # 收集 close 的 actual_dir（用于 naive DA）
                    if fn == 'close':
                        local_actual_dir[step_idx].append(float(actual_dir))

            # §10 新增：可懂指标收集
            baseline_close = baseline[3]  # close 特征

            # 振幅误差率（第一根 predict 的 high - low）
            pred_amp = pred_raw[0, 1] - pred_raw[0, 2]  # high - low
            actual_amp = actual[0, 1] - actual[0, 2]
            amp_rate = amplitude_error_rate(pred_amp, actual_amp)
            local_amplitude_rates.append(amp_rate)

            # 涨跌停检测
            pred_limit = detect_limit(pred_raw, baseline_close, limit_pct)
            actual_limit = detect_limit(actual, baseline_close, limit_pct)
            local_pred_limit.append(pred_limit.any())
            local_actual_limit.append(actual_limit.any())

        except Exception as e:
            # 评估循环不应静默吞异常（曾因 block 格式未支持，全部样本异常被吞 → IC/DA 全 0）
            # 打印首个异常供诊断，仍 continue 不中断整个评估
            if rank == 0:
                print(f"[WARN eval] {symbol}@{window_start}: {type(e).__name__}: {e}", flush=True)
            continue

    # 聚合（保留分布）
    if world_size > 1:
        ic_result = aggregate_ic(local_ics, world_size, device, rank == 0)
        da_result = aggregate_da(local_da, world_size, device, config.predict, rank == 0)

        # 聚合 actual_dir 统计计算 naive DA（所有 rank 执行 all_gather）
        # 注意：dist.all_gather 必须所有 rank 同时调用，否则死锁
        naive_da_by_step = {}
        for step_idx in range(config.predict):
            local_up_count = sum(local_actual_dir[step_idx])
            local_n = len(local_actual_dir[step_idx])

            up_count_tensor = torch.tensor([local_up_count], device=device)
            n_tensor = torch.tensor([local_n], device=device)

            gathered_up = [torch.zeros_like(up_count_tensor) for _ in range(world_size)]
            gathered_n = [torch.zeros_like(n_tensor) for _ in range(world_size)]

            # 所有 rank 执行 all_gather
            dist.all_gather(gathered_up, up_count_tensor)
            dist.all_gather(gathered_n, n_tensor)

            # 只有 rank 0 组装结果
            if rank == 0:
                total_up = sum(t.item() for t in gathered_up)
                total_n = sum(t.item() for t in gathered_n)

                if total_n > 0:
                    up_ratio = total_up / total_n
                    naive_da_by_step[step_idx] = float(max(up_ratio, 1 - up_ratio))
                else:
                    naive_da_by_step[step_idx] = 0.5
    else:
        # 单卡直接聚合
        ic_result = {}
        for fn in FEATURE_NAMES:
            ics = local_ics[fn]
            if ics:
                ic_result[fn] = {
                    'mean': float(np.mean(ics)),
                    'std': float(np.std(ics)) if len(ics) >= 2 else 0.0,
                    'p50': float(np.percentile(ics, 50)),
                    'n': len(ics),
                }
            else:
                ic_result[fn] = {'mean': 0.0, 'std': 0.0, 'p50': None, 'n': 0}

        da_result = {}
        for step_idx in range(config.predict):
            step_result = {}
            for fn in FEATURE_NAMES:
                da_list = local_da[step_idx][fn]
                if da_list:
                    step_result[fn] = {
                        'mean': float(np.mean(da_list)),
                        'std': float(np.std(da_list)),
                        'n': len(da_list),
                    }
                else:
                    step_result[fn] = {'mean': 0.0, 'std': 0.0, 'n': 0}
            da_result[f'step{step_idx + 1}'] = step_result

        # 计算 naive DA（多数方向比例）
        naive_da_by_step = {}
        for step_idx in range(config.predict):
            actual_dirs = local_actual_dir[step_idx]
            if actual_dirs:
                up_ratio = np.mean(actual_dirs)
                naive_da_by_step[step_idx] = float(max(up_ratio, 1 - up_ratio))
            else:
                naive_da_by_step[step_idx] = 0.5

    # §10 新增：可懂指标聚合
    amplitude_result = None
    limit_result = None

    if rank == 0:
        # 振幅误差率统计
        if local_amplitude_rates:
            amp_arr = np.array(local_amplitude_rates)
            amplitude_result = {
                'mean_rate': float(np.mean(amp_arr)),
                'std_rate': float(np.std(amp_arr)),
                'usable_pct': float(np.mean(np.abs(amp_arr - 1.0) < 0.3)),
            }

        # 涨跌停命中率
        if local_pred_limit and local_actual_limit:
            limit_result = limit_hit_rate(
                np.array(local_pred_limit),
                np.array(local_actual_limit)
            )

    return ic_result, da_result, naive_da_by_step, amplitude_result, limit_result


def compute_val_loss(
    model,
    tokenizer,
    val_data,
    val_indices,
    config: DataConfig,
    device: torch.device,
    batch_size: int = 16,
    world_size: int = 1,
    rank: int = 0,
    use_ddp: bool = False,
) -> float:
    """
    计算 validation 子集上的损失（与 train 同口径 token CE）

    F2 口径修复：使用 head.compute_loss 算 token CE，与 train 同口径。
    之前用 MSE 重建，量级差 2-3 个数量级，不可用于 early stopping。

    val_indices 已是本 epoch 抽定的同源子集（与 IC 同源），此处全量遍历、不再二次抽样。
    DDP：各 rank 按 idx%world_size 分片处理（与 IC 评估一致），val_loss all_reduce 聚合
    保证各 rank 拿到同一值 → early-stop 决定一致（防死锁）。

    Args:
        model: Kronos 模型
        tokenizer: KronosTokenizer
        val_data: 验证数据 dict
        val_indices: 验证索引列表（已是 epoch 同源子集）
        config: 数据配置
        device: 设备
        batch_size: 批大小
        world_size: DDP 进程数
        rank: DDP rank
        use_ddp: 是否 DDP

    Returns:
        avg_val_loss: 平均 token CE 损失（与 train 同口径，DDP 下跨 rank 聚合）
    """
    model.eval()

    # DDP 解包：val 前向用裸 model，不走 DDP forward。
    # DDP forward 会注册反向同步 hook 且假设各 rank 调用次数一致，
    # 但本函数各 rank 按 batch_idx%world_size 分片处理（调用次数不同）→ DDP ALLREDUCE 错位死锁。
    # eval 在 no_grad 下不需梯度同步，聚合靠下方手写 all_reduce。
    raw_model = model.module if isinstance(model, DDP) else model

    # 创建临时 val dataset 和 loader（顺序遍历整个同源子集，不抽样）
    val_dataset = KronosDataset(val_data, val_indices, config, mode='val')
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    val_loss_sum = 0.0
    val_count = 0

    with torch.no_grad():
        for batch_idx, (x_norm, x_stamp, y_stamp, meta) in enumerate(val_loader):
            # DDP 分片：与 IC 评估一致，各 rank 按 idx%world_size 处理
            if use_ddp and (batch_idx % world_size != rank):
                continue

            x_norm = x_norm.to(device)
            x_stamp = x_stamp.to(device)

            # Tokenize
            token_seq_0, token_seq_1 = tokenizer.encode(x_norm, half=True)

            # Forward（与 train 同口径）—— 用裸 model，不走 DDP forward（防分片不均死锁）
            s1_logits, s2_logits = raw_model(token_seq_0, token_seq_1, x_stamp)

            # CE loss（与 train 同口径，用 head.compute_loss）
            head = raw_model.head
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]
            ce_loss, _, _ = head.compute_loss(
                s1_logits[:, :-1, :], s2_logits[:, :-1, :], token_out[0], token_out[1]
            )
            n = x_norm.size(0)
            val_loss_sum += ce_loss.item() * n
            val_count += n

    # DDP：跨 rank 聚合 val_loss（与 tokenizer 一致），保证 early-stop 各 rank 一致
    if use_ddp:
        sum_tensor = torch.tensor([val_loss_sum, val_count], device=device, dtype=torch.float64)
        dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
        val_loss_sum = sum_tensor[0].item()
        val_count = sum_tensor[1].item()

    model.train()
    return val_loss_sum / val_count if val_count > 0 else 0.0


# ============================================================================
# 主训练函数
# ============================================================================

def train(
    model,
    tokenizer,
    train_loader,
    val_data,
    val_indices,
    config: DataConfig,
    train_config: TrainConfig,
    save_dir: str,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
    use_ddp: bool = False,
    resume_checkpoint: str = None,
    val_samples: int = -1,
):
    """
    主训练循环

    关键：
    - IC 用去趋势口径（E1 解决）
    - checkpoint 选择用正确口径
    - early stopping patience 需审视（§1.6.7）
    - val_loss 与 IC 同源采样（每 epoch 抽定一份 val 子集共用），防过拟评估
    - 训练无放回子集采样（防过拟，由 train_loader 的 sampler 控制）

    I9 修复：支持 --resume 断点续训
    """
    is_main = (rank == 0)
    start_time = time.time()
    start_epoch = 0

    # 初始化 training_info
    info_path = init_training_info(save_dir, config, train_config) if is_main else None

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay
    )

    # Cosine Annealing LR
    total_steps = train_config.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=train_config.lr_min
    )

    # I9 修复：Resume 加载
    start_epoch = 0
    best_val_loss = float('inf')
    best_val_loss_epoch = 0
    best_ic = -999
    best_ic_epoch = 0
    best_da_score = 0.0
    best_da_score_epoch = 0
    best_excess_da = 0.0
    best_excess_da_epoch = 0
    best_combined = 0.0
    best_combined_epoch = 0
    # §10 新增：可懂指标 best
    best_amplitude_error_rate = 1.0  # 越接近 1.0 越好，初始设为最差
    best_amplitude_error_rate_epoch = 0
    best_limit_hit_rate = 0.0  # 越高越好
    best_limit_hit_rate_epoch = 0
    patience_counter = 0

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        if is_main:
            print(f"\n[RESUME] Loading from {resume_checkpoint}")
        # 加载模型权重
        unwrapped = model.module if use_ddp else model
        state_dict = load_file(os.path.join(resume_checkpoint, 'model.safetensors'))
        unwrapped.load_state_dict(state_dict, strict=False)

        # 加载 optimizer/scheduler 状态
        opt_path = os.path.join(resume_checkpoint, 'optimizer.pt')
        sch_path = os.path.join(resume_checkpoint, 'scheduler.pt')
        meta_path = os.path.join(resume_checkpoint, 'resume_meta.json')

        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
        if os.path.exists(sch_path):
            scheduler.load_state_dict(torch.load(sch_path, map_location=device))
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                resume_meta = json.load(f)
            start_epoch = resume_meta.get('epoch', 0)
            best_val_loss = resume_meta.get('best_val_loss', float('inf'))
            best_val_loss_epoch = resume_meta.get('best_val_loss_epoch', 0)
            best_ic = resume_meta.get('best_ic', -999)
            best_ic_epoch = resume_meta.get('best_ic_epoch', 0)
            best_da_score = resume_meta.get('best_da_score', 0.0)
            best_da_score_epoch = resume_meta.get('best_da_score_epoch', 0)
            best_excess_da = resume_meta.get('best_excess_da', 0.0)
            best_excess_da_epoch = resume_meta.get('best_excess_da_epoch', 0)
            best_combined = resume_meta.get('best_combined', 0.0)
            best_combined_epoch = resume_meta.get('best_combined_epoch', 0)
            # §10 新增：可懂指标 resume
            best_amplitude_error_rate = resume_meta.get('best_amplitude_error_rate', 1.0)
            best_amplitude_error_rate_epoch = resume_meta.get('best_amplitude_error_rate_epoch', 0)
            best_limit_hit_rate = resume_meta.get('best_limit_hit_rate', 0.0)
            best_limit_hit_rate_epoch = resume_meta.get('best_limit_hit_rate_epoch', 0)
            patience_counter = resume_meta.get('patience_counter', 0)
            # epoch 从 resume_meta 推算（或从 training_info.json 读取）
            if is_main:
                print(f"[RESUME] Starting from epoch {start_epoch + 1}")
                print(f"[RESUME] Best: val_loss={best_val_loss:.4f}@{best_val_loss_epoch}, IC={best_ic:.4f}@{best_ic_epoch}, DA={best_da_score:.4f}@{best_da_score_epoch}, Combined={best_combined:.4f}@{best_combined_epoch}")

    # §10 更新：best 跟踪已在上面的初始化/resume 中完成

    history = {
        'train_loss': [],
        'val_loss': [],
        'ic': [],
        'da_score': [],
        'excess_da': [],
        'combined': [],
        'lr': [],
    }

    # I9 修复：从 resume 的 epoch 开始
    for epoch_idx in range(start_epoch, train_config.epochs):
        epoch_start = time.time()
        model.train()

        epoch_losses = []
        current_lr = optimizer.param_groups[0]['lr']

        if is_main:
            print(f"\n=== Epoch {epoch_idx + 1}/{train_config.epochs} ===")
            print(f"LR: {current_lr:.6f}")

        total_batches = len(train_loader)
        for batch_idx, (x_norm, x_stamp, y_stamp, meta) in enumerate(train_loader):
            x_norm = x_norm.to(device, non_blocking=True)
            x_stamp = x_stamp.to(device, non_blocking=True)

            # Tokenize
            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(x_norm, half=True)

            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

            # Forward
            # DDP 模式必须走 model(...) 触发梯度 all-reduce hook；
            # model.module(...) 绕过 DDP 会导致各卡梯度不同步。
            # head.compute_loss 仍需 model.module.head 访问子模块。
            s1_logits, s2_logits = model(token_seq_0, token_seq_1, x_stamp)
            head = model.module.head if use_ddp else model.head
            recon_loss, _, _ = head.compute_loss(
                s1_logits[:, :-1, :], s2_logits[:, :-1, :], token_out[0], token_out[1]
            )

            optimizer.zero_grad()
            recon_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(recon_loss.item())

            if is_main and (batch_idx % 50 == 0 or batch_idx == 0):
                avg_loss = sum(epoch_losses[-50:]) / min(len(epoch_losses[-50:]), 50)
                progress = (batch_idx + 1) / total_batches * 100
                elapsed = time.time() - epoch_start
                print(f"  Batch {batch_idx + 1}/{total_batches} ({progress:.1f}%) - Loss: {recon_loss.item():.4f}, Avg: {avg_loss:.4f} [{elapsed:.0f}s]", flush=True)

        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        current_lr = optimizer.param_groups[0]['lr']

        # 同源采样：本 epoch 抽定一份 val 子集，val_loss 与 IC 共用（修两处独立采样的 bug）
        # 每 epoch 随机重抽（评估更全面，early-stop 靠 patience 容噪）
        # rank0 抽样 → broadcast 给所有 rank，保证各 rank 评估同一子集（IC 分片聚合正确）
        D_val = len(val_indices)
        n_eval = val_samples if val_samples > 0 else D_val
        n_eval = min(n_eval, D_val)
        rng_eval = np.random.RandomState(config.seed + epoch_idx)
        sampled_pos = rng_eval.choice(D_val, size=n_eval, replace=False)
        if use_ddp:
            # broadcast 采样位置（int64 数组）给所有 rank
            pos_tensor = torch.from_numpy(sampled_pos.astype(np.int64)).to(device)
            gathered = [torch.zeros_like(pos_tensor) for _ in range(world_size)]
            dist.all_gather(gathered, pos_tensor)
            sampled_pos = gathered[0].cpu().numpy()
        epoch_val_indices = [val_indices[i] for i in sampled_pos]

        # Validation loss（在 val 子集上前向计算，不反传，与 IC 同源子集）
        avg_val_loss = compute_val_loss(
            model, tokenizer, val_data, epoch_val_indices, config, device,
            batch_size=train_config.batch_size,
            world_size=world_size, rank=rank, use_ddp=use_ddp,
        )

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(current_lr)

        # Trajectory IC 评估（§10 更新：返回可懂指标）—— 与 val_loss 同源（同一 epoch_val_indices）
        ic_result, da_result, naive_da_by_step, amplitude_result, limit_result = evaluate_trajectory_ic(
            model, tokenizer, val_data, epoch_val_indices, config,
            device, world_size, rank,
            n_samples=n_eval,  # 已是子集，全量评估这份子集
            seed=config.seed + epoch_idx,
            model_type=train_config.model_type
        )

        # 关键同步：eval 集体通信全部完成后，所有 rank 在此 barrier 对齐，
        # 然后才让 rank0 进重活（save_pretrained/torch.save/写 JSON/epoch 输出）。
        # 否则 non-zero rank 跳过整段 is_main 块、直奔下一 epoch 的 DDP train forward，
        # 在 backward allreduce 等 rank0 → others 领先一整个 epoch → NCCL 600s 超时死锁。
        # barrier 把 rank0 与 others 的偏差控制在单 epoch 内，不跨 epoch 累积。
        if use_ddp:
            dist.barrier()

        if is_main:
            current_ic = ic_result.get('close', {}).get('mean', 0)

            # 计算 DA_score 和 excess_da
            # M4 修复：直接传标量（已聚合的均值），calculate_da_score 兼容标量/列表
            da_by_step = [{f: da_result[f'step{s+1}'].get(f, {}).get('mean', 0) for f in FEATURE_NAMES} for s in range(config.predict)]
            da_score = calculate_da_score(da_by_step, config.predict)

            # 计算 excess DA（平均 excess）
            excess_da_avg = 0.0
            for step_idx in range(config.predict):
                da_mean = da_result.get(f'step{step_idx + 1}', {}).get('close', {}).get('mean', 0)
                naive_da = naive_da_by_step.get(step_idx, 0.5)
                excess_da_avg += (da_mean - naive_da)
            excess_da_avg /= config.predict

            # Combined score（权重 0.6/0.4）
            current_combined = calculate_combined_score(current_ic, da_score)

            history['ic'].append(current_ic)
            history['da_score'].append(da_score)
            history['excess_da'].append(excess_da_avg)
            history['combined'].append(current_combined)

            # 6 维指标：IC / DA 按 6 feature 展开（close 为主指标带 *），best 仍只追 close + Combined
            # n = 有效样本数：IC 因 safe_trajectory_ic 对近平坦序列返回 None 会丢样本，
            #   n<总数 说明部分样本被丢；n=0 说明全丢 → mean 兜底 0（即 IC=0 的常见根因）
            epoch_time = time.time() - epoch_start
            print(f"\n  Epoch {epoch_idx + 1}/{train_config.epochs}  LR: {current_lr:.6f}")

            # 每 feature 的 IC mean 与 n
            feat_ic_mean = {f: ic_result.get(f, {}).get('mean', 0.0) for f in FEATURE_NAMES}
            feat_ic_n = {f: ic_result.get(f, {}).get('n', 0) for f in FEATURE_NAMES}
            # 每 feature 的 DA（步间简单平均，用于展示；headline DA_score 仍为加权版）
            feat_da_mean = {
                f: float(np.mean([da_result.get(f'step{s+1}', {}).get(f, {}).get('mean', 0.0)
                                  for s in range(config.predict)]))
                for f in FEATURE_NAMES
            }
            # DA 每步每 feature 都 append，n 各 feature 一致；取 step1 close 的 n 代表已处理样本数
            feat_da_n = da_result.get('step1', {}).get('close', {}).get('n', 0)

            ic_vals = "  ".join(f"{f}{'*' if f=='close' else ''}{feat_ic_mean[f]:+.3f}" for f in FEATURE_NAMES)
            ic_ns = "[" + ", ".join(str(feat_ic_n[f]) for f in FEATURE_NAMES) + "]"  # 顺序同上: open,high,low,close,vol,amt
            da_vals = "  ".join(f"{f}{'*' if f=='close' else ''} {feat_da_mean[f]:.3f}" for f in FEATURE_NAMES)

            # 哨兵显示：best_* 仍为初始哨兵值（best_ic=-999 / best_val_loss=inf / 其余 0.0@ep0）说明从未更新过，
            # 显示 "—" 而非裸露哨兵。resume 时 best 从 resume_meta 读入真实值，不触发此分支。
            def _fmt_best(val, epoch, is_sentinel):
                return "—" if is_sentinel else f"{val:.4f} @ep{epoch}"
            def _fmt_best_pct(val, epoch, is_sentinel):
                return "—" if is_sentinel else f"{val:+.1%} @ep{epoch}"
            ic_best_str = _fmt_best(best_ic, best_ic_epoch, best_ic == -999)
            da_best_str = _fmt_best(best_da_score, best_da_score_epoch,
                                    best_da_score == 0.0 and best_da_score_epoch == 0)
            exc_best_str = _fmt_best_pct(best_excess_da, best_excess_da_epoch,
                                         best_excess_da == 0.0 and best_excess_da_epoch == 0)
            comb_best_str = _fmt_best(best_combined, best_combined_epoch,
                                      best_combined == 0.0 and best_combined_epoch == 0)
            vl_best_str = _fmt_best(best_val_loss, best_val_loss_epoch,
                                    best_val_loss == float('inf'))

            print(f"    IC:    {ic_vals}")
            print(f"           n: {ic_ns}   | best close {ic_best_str}")
            print(f"    DA:    {da_vals}  (n={feat_da_n})")
            print(f"           best close {da_best_str}")
            print(f"    ExcDA: close* {excess_da_avg:+.1%}  | best {exc_best_str}  [close-only]")
            print(f"    Combined:  current {current_combined:.4f}  | best {comb_best_str}")
            print(f"    Val_loss:  current {avg_val_loss:.4f}  | best {vl_best_str}")
            print(f"    Train: {avg_train_loss:.4f}, Time: {format_time(epoch_time)}")

            # IC 滑动均值（用于 early stopping）
            ic_window = 3
            if len(history['ic']) >= ic_window:
                ic_smoothed = np.mean(history['ic'][-ic_window:])
            else:
                ic_smoothed = current_ic

            # 保存 latest（I9 修复：同时保存 optimizer/scheduler 状态）
            latest_path = get_checkpoint_path(save_dir, 'latest_model')
            unwrapped = model.module if use_ddp else model
            unwrapped.save_pretrained(latest_path)

            # 保存 optimizer/scheduler 状态（用于 resume）
            torch.save(optimizer.state_dict(), os.path.join(latest_path, 'optimizer.pt'))
            torch.save(scheduler.state_dict(), os.path.join(latest_path, 'scheduler.pt'))
            resume_meta = {
                'epoch': epoch_idx,
                'best_val_loss': best_val_loss,
                'best_val_loss_epoch': best_val_loss_epoch,
                'best_ic': best_ic,
                'best_ic_epoch': best_ic_epoch,
                'best_da_score': best_da_score,
                'best_da_score_epoch': best_da_score_epoch,
                'best_excess_da': best_excess_da,
                'best_excess_da_epoch': best_excess_da_epoch,
                'best_combined': best_combined,
                'best_combined_epoch': best_combined_epoch,
                # §10 新增：可懂指标
                'best_amplitude_error_rate': best_amplitude_error_rate,
                'best_amplitude_error_rate_epoch': best_amplitude_error_rate_epoch,
                'best_limit_hit_rate': best_limit_hit_rate,
                'best_limit_hit_rate_epoch': best_limit_hit_rate_epoch,
                'patience_counter': patience_counter,
            }
            safe_save_json(resume_meta, os.path.join(latest_path, 'resume_meta.json'))

            # §10 更新：best 跟踪扩展 + 可懂指标只在 best 时算
            best_updates = {}
            improved = False
            compute_understandable = False  # 标记是否需要算可懂指标

            # val_loss best
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_loss_epoch = epoch_idx + 1
                improved = True
                compute_understandable = True
                best_updates['val_loss'] = (best_val_loss, best_val_loss_epoch)

                best_path = get_checkpoint_path(save_dir, 'best_model')
                unwrapped.save_pretrained(best_path)

            # IC best
            if train_config.ic_patience_reset and ic_smoothed > best_ic:
                best_ic = ic_smoothed
                best_ic_epoch = epoch_idx + 1
                patience_counter = 0
                improved = True
                compute_understandable = True
                best_updates['ic'] = (best_ic, best_ic_epoch)

                ic_path = get_checkpoint_path(save_dir, 'best_ic_model')
                unwrapped.save_pretrained(ic_path)

            # DA_score best（只记录，不存 checkpoint）
            if da_score > best_da_score:
                best_da_score = da_score
                best_da_score_epoch = epoch_idx + 1
                best_updates['da_score'] = (best_da_score, best_da_score_epoch)

            # Excess DA best（只记录，不存 checkpoint）
            if excess_da_avg > best_excess_da:
                best_excess_da = excess_da_avg
                best_excess_da_epoch = epoch_idx + 1
                best_updates['excess_da'] = (best_excess_da, best_excess_da_epoch)

            # Combined best
            if current_combined > best_combined:
                best_combined = current_combined
                best_combined_epoch = epoch_idx + 1
                improved = True
                compute_understandable = True
                best_updates['combined'] = (best_combined, best_combined_epoch)

                combined_path = get_checkpoint_path(save_dir, 'best_combined_model')
                unwrapped.save_pretrained(combined_path)

            if not improved:
                patience_counter += 1

            # §10 D：可懂指标只在产生 best 时记录（已在 evaluate_trajectory_ic 中收集）
            # 注意：evaluate_trajectory_ic 总是返回这些值，但只在产生 best 时才写入 training_info
            epoch_amplitude_result = None
            epoch_limit_result = None
            if compute_understandable:
                # 使用 evaluate_trajectory_ic 的返回值
                epoch_amplitude_result = amplitude_result
                epoch_limit_result = limit_result

                # 如果可懂指标刷新 best，也记录到 best_updates
                if amplitude_result and amplitude_result.get('mean_rate'):
                    # 检查是否刷新 amplitude_error_rate best
                    current_amp = amplitude_result['mean_rate']
                    if current_amp < best_amplitude_error_rate:  # 越接近 1.0 越好，但我们记录 mean_rate
                        best_amplitude_error_rate = current_amp
                        best_amplitude_error_rate_epoch = epoch_idx + 1
                        best_updates['amplitude_error_rate'] = (best_amplitude_error_rate, best_amplitude_error_rate_epoch)

                if limit_result and limit_result.get('hit_rate'):
                    current_limit_hit = limit_result['hit_rate']
                    if current_limit_hit > best_limit_hit_rate:  # 越高越好
                        best_limit_hit_rate = current_limit_hit
                        best_limit_hit_rate_epoch = epoch_idx + 1
                        best_updates['limit_hit_rate'] = (best_limit_hit_rate, best_limit_hit_rate_epoch)

            # 更新 training_info（§10 更新）
            if info_path:
                metrics_dict = {
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'ic': current_ic,
                    'da_score': da_score,
                    'excess_da': excess_da_avg,
                    'combined': current_combined,
                    'lr': current_lr,
                    'amplitude_error_rate': epoch_amplitude_result,
                    'limit_hit_rate': epoch_limit_result,
                }
                update_training_info(info_path, epoch_idx + 1, metrics_dict, best_updates)

        # Early stopping（§1.6.7: patience=12 需审视）
        # DDP 同步：rank 0 算 stop 决定，broadcast 给所有 rank，一起 break。
        # 关键：此段必须在 is_main 块【外】，所有 rank 都执行 barrier/broadcast/break。
        # 之前的 bug：整段被缩进进 is_main 块，只有 r0 执行 → others 永不 break、
        # 永不参与 broadcast → r0 卡死、others 超前进下一 epoch → NCCL 死锁。
        # 此 barrier 让 others 等 rank0 做完 is_main 块（save/输出），再一起进 broadcast。
        if use_ddp:
            dist.barrier()
        stop_flag = torch.zeros(1, dtype=torch.float32, device=device)
        if is_main:
            if epoch_idx >= train_config.early_stopping_grace_period:
                if patience_counter >= train_config.early_stopping_patience:
                    print(f"\n[EARLY STOP] No improvement for {patience_counter} epochs")
                    stop_flag[0] = 1.0
        if use_ddp:
            dist.broadcast(stop_flag, src=0)
        if stop_flag[0] > 0:
            break

            # Final save
    if is_main:
        final_path = get_checkpoint_path(save_dir, 'final_model')
        unwrapped = model.module if use_ddp else model
        unwrapped.save_pretrained(final_path)

        # Summary
        summary = {
            'config': {
                'norm_mode': config.norm_mode,
                'lookback': config.lookback,
                'predict': config.predict,
                'split_mode': config.split_mode,
                'model_type': train_config.model_type,
            },
            'result': {
                'best_val_loss': best_val_loss,
                'best_ic': best_ic,
                'best_combined': best_combined,
                'epochs_trained': epoch_idx + 1,
            },
            'history': history,
        }
        summary_path = get_summary_path(save_dir)
        safe_save_json(summary, summary_path)

        total_time = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"Training completed in {format_time(total_time)}")
        print(f"Best IC: {best_ic:.4f}")
        print(f"Best Combined: {best_combined:.4f}")
        print(f"Saved to: {save_dir}")
        print(f"{'=' * 60}")

    # 训练结束同步：rank0 在上面 if is_main 块做 final save（save_pretrained + 写 JSON），
    # others 跳过直奔 return → cleanup_ddp → destroy_process_group。
    # 若 rank0 save 慢，others 先 destroy_process_group → 各 rank destroy 不对齐 → 进程挂。
    # 此 barrier 让 rank0 做完 save 后、return 前等齐所有 rank，再一起进 cleanup。
    if use_ddp:
        dist.barrier()

    return {
        'best_val_loss': best_val_loss,
        'best_ic': best_ic,
        'best_combined': best_combined,
        'epochs_trained': epoch_idx + 1,
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Kronos Predictor Training')
    parser.add_argument('--norm-mode', type=str, default='sliding_ma60',
                        choices=['full_window', 'sliding_ma20', 'sliding_ma60', 'sliding_ma120'])
    parser.add_argument('--lookback', type=int, default=400)
    parser.add_argument('--predict', type=int, default=10)
    parser.add_argument('--split-mode', type=str, default='block',
                        choices=['time', 'block'])
    parser.add_argument('--model', type=str, default='mini',
                        choices=['mini', 'small', 'base'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-folder', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path (e.g. outputs/models/.../checkpoints/latest_model)')
    # 采样参数（防过拟 + 快速验证）
    # train: --train-sample-ratio 控每 epoch 训练子集占比（防过拟，默认 0.5）；
    #        --train-samples 传绝对数覆盖 ratio（-1=用 ratio）。
    # val/IC: --n-sample-ratio 控验证子集占比（默认 0.5）；--n-samples 传绝对数覆盖。
    # val_loss 与 IC 同源（epoch 内抽定一份共用），子集大小统一由 --n-sample-ratio/--n-samples 控制。
    # ratio<1 会让 len(train_loader) 缩小 → CosineAnnealingLR total_steps 同比缩小 →
    # cosine 提前退火。调 ratio 后需配套调 --epochs（如 ratio=0.5 时 epochs×2 才等价全量步数）。
    parser.add_argument('--train-sample-ratio', type=float, default=0.5,
                        help='每 epoch 训练子集占总量比例（防过拟，默认 0.5）。无放回采样，多 epoch 轮换不同子集。'
                             'ratio<1 会同比缩短 cosine 退火周期，需配套调 --epochs')
    parser.add_argument('--train-samples', type=int, default=-1,
                        help='每 epoch 训练样本绝对数（-1=用 ratio，>0 覆盖 ratio）。快速验证/调试用')
    parser.add_argument('--n-sample-ratio', type=float, default=0.5,
                        help='验证(val_loss+IC)子集占 val 总量比例（默认 0.5）。每 epoch 随机重抽，val_loss 与 IC 同源')
    parser.add_argument('--n-samples', type=int, default=-1,
                        help='验证样本绝对数（-1=用 ratio，>0 覆盖 ratio）。快速验证用')
    args = parser.parse_args()

    # DDP setup
    rank, local_rank, world_size, use_ddp = get_rank_info()
    device = get_device(local_rank)
    is_main = (rank == 0)

    # Config（DDP 各 rank 用不同 seed，保证数据不同）
    # 注意：seed 偏移量用 rank，不是 local_rank（多机场景 rank 全局唯一）
    config = DataConfig(
        norm_mode=args.norm_mode,
        lookback=args.lookback,
        predict=args.predict,
        split_mode=args.split_mode,
        seed=args.seed + rank,  # 各 rank seed 不同
    )

    train_config = TrainConfig(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )

    set_seed(config.seed)

    # 数据路径（由 preprocess.py 预先生成）
    train_path = get_split_data_path(config.norm_mode, config.lookback, config.predict, config.split_mode, 'train')
    val_path = get_split_data_path(config.norm_mode, config.lookback, config.predict, config.split_mode, 'val')

    if is_main:
        print(f"\n{'=' * 60}")
        print(f"Kronos Predictor Training")
        print(f"{'=' * 60}")
        print(f"norm_mode: {config.norm_mode}")
        print(f"lookback: {config.lookback}, predict: {config.predict}")
        print(f"split_mode: {config.split_mode}")
        print(f"model: {train_config.model_type}")
        print(f"batch_size: {train_config.batch_size}")
        print(f"lr: {train_config.learning_rate}")
        print(f"world_size: {world_size}")
        print(f"{'=' * 60}")

    # 加载 tokenizer
    tokenizer_path = get_tokenizer_path(config.norm_mode, train_config.model_type)
    if not os.path.exists(os.path.join(tokenizer_path, 'model.safetensors')):
        # I8 修复：显式警告而非静默 fallback
        # fallback 到 legacy tokenizer（注意 norm_mode 可能不匹配）
        legacy_path = 'outputs/tokenizers/final/2k-MA60' if args.model == 'mini' else 'outputs/tokenizers/final/base-MA60'
        if os.path.exists(legacy_path):
            if is_main:
                print(f"[WARNING] Tokenizer not found at {tokenizer_path}")
                print(f"[WARNING] Using legacy fallback: {legacy_path}")
                print(f"[WARNING] norm_mode mismatch: tokenizer may not match data distribution")
            tokenizer_path = legacy_path
        else:
            raise FileNotFoundError(
                f"Tokenizer not found at {tokenizer_path}. "
                f"Please run tokenizer training first: "
                f"python finetune/tokenizer/train.py --norm-mode {config.norm_mode} --model {train_config.model_type}"
            )

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    tokenizer.eval().to(device)
    if is_main:
        print(f"Tokenizer: {tokenizer_path}")

    # 加载模型
    model_path = MODEL_PATHS[train_config.model_type]
    model = Kronos.from_pretrained(model_path)
    model.to(device)
    if use_ddp:
        # broadcast_buffers=False：Kronos 的 buffer（inv_freq/basis/group_basis/group_codebook）
        # 都是确定性常量，预训练权重 + 相同 init → 各卡已一致，无需每 forward 广播。
        # 关闭后消除 DDP 每 forward 的 BROADCAST，避免 buffer 广播时机错位导致的死锁。
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)
    if is_main:
        print(f"Model: {model_path}, Size: {get_model_size(model):.2f}M")

    # 加载数据
    import pickle
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(val_path, 'rb') as f:
        val_data = pickle.load(f)

    # 构建索引
    train_indices = []
    for symbol, d in train_data.items():
        mode = d.get('mode', 'time') if isinstance(d, dict) else None
        if mode == 'block':
            for b_start, blk in d['blocks'].items():
                for w in blk['windows']:
                    train_indices.append((symbol, int(w)))
        elif 'windows' in d:
            for w in d['windows']:
                train_indices.append((symbol, int(w)))
        elif hasattr(d, 'columns'):
            for i in range(len(d) - config.lookback - config.predict + 1):
                train_indices.append((symbol, i))
        else:
            for i in range(len(d['normalized']) - config.lookback - config.predict + 1):
                train_indices.append((symbol, i))

    val_indices = []
    for symbol, d in val_data.items():
        mode = d.get('mode', 'time') if isinstance(d, dict) else None
        if mode == 'block':
            for b_start, blk in d['blocks'].items():
                for w in blk['windows']:
                    val_indices.append((symbol, int(w)))
        elif 'windows' in d:
            # 已预处理的窗口列表
            for w in d['windows']:
                val_indices.append((symbol, int(w)))
        elif hasattr(d, 'columns'):
            # DataFrame 格式：取所有合法窗口（非仅末尾 1 个）
            for i in range(len(d) - config.lookback - config.predict + 1):
                val_indices.append((symbol, i))
        else:
            # dict 格式：取所有合法窗口
            for i in range(len(d['normalized']) - config.lookback - config.predict + 1):
                val_indices.append((symbol, i))

    if is_main:
        print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

    # Dataset
    train_dataset = KronosDataset(train_data, train_indices, config, mode='train')
    val_dataset = KronosDataset(val_data, val_indices, config, mode='val')

    # 训练采样数：--train-samples(绝对值) 优先，否则用 --train-sample-ratio(比例)
    # replacement=False：无放回，每 epoch 抽真子集，多 epoch 轮换不同样本 → 防过拟
    D_train = len(train_dataset)
    if args.train_samples > 0:
        train_samples = min(args.train_samples, D_train)
    else:
        train_samples = max(1, int(D_train * args.train_sample_ratio))

    # 验证采样数：--n-samples(绝对值) 优先，否则用 --n-sample-ratio(比例)
    # val_loss 与 IC 同源（epoch 内抽定一份共用），每 epoch 随机重抽
    D_val = len(val_indices)
    if args.n_samples > 0:
        val_samples = min(args.n_samples, D_val)
    else:
        val_samples = max(1, int(D_val * args.n_sample_ratio))

    if is_main:
        print(f"Train: {D_train} windows, sampling {train_samples}/epoch (ratio={args.train_sample_ratio if args.train_samples <= 0 else 'n/a'})")
        print(f"Val: {D_val} windows, sampling {val_samples}/epoch for val_loss+IC (ratio={args.n_sample_ratio if args.n_samples <= 0 else 'n/a'})")

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        sampler=RandomSampler(train_dataset, replacement=False, num_samples=train_samples),
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    # 输出目录
    if args.output_folder:
        save_dir = os.path.join(project_root, 'outputs/models', args.output_folder)
    else:
        save_dir = get_model_path(config.norm_mode, config.lookback, config.predict, config.split_mode, train_config.model_type)

    if is_main:
        ensure_dir(save_dir)
        ensure_dir(get_checkpoint_path(save_dir, 'checkpoints'))

    # 训练（I9 修复：传入 resume_checkpoint）
    result = train(
        model, tokenizer, train_loader, val_data, val_indices,
        config, train_config, save_dir, device,
        rank, world_size, use_ddp,
        resume_checkpoint=args.resume,
        val_samples=val_samples,
    )

    cleanup_ddp()


if __name__ == '__main__':
    main()