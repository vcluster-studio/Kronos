"""
Kronos Predictor Core Modules

重构后的核心模块，提供：
- config: 配置对象（DataConfig/TrainConfig/ArtifactConfig）
- paths: 路径构建函数
- schema: 数据 schema 定义（SampleSchema/BacktestSchema/MetaSchema）
- normalization: 归一化器（FullWindowNormalizer/SlidingMANormalizer）
- splitting: 数据分割算法（time_split/block_split/validate_no_leakage）
- metrics: 度量计算（trajectory IC/DA/可懂指标）
- utils: 工具函数
"""

from .config import (
    DataConfig,
    TrainConfig,
    ArtifactConfig,
    BacktestConfig,
    parse_norm_mode,
)

from .paths import (
    get_raw_path,
    get_backtest_raw_path,
    get_split_data_path,
    get_backtest_data_path,
    get_meta_path,
    get_model_path,
    get_tokenizer_path,
    get_checkpoint_path,
    get_training_info_path,
    get_summary_path,
    ensure_dir,
)

from .schema import (
    SampleSchema,
    BacktestSchema,
    MetaSchema,
    TrainingInfoSchema,
    sample_to_dict,
    dict_to_sample,
    intervals_overlap,
    compute_fingerprint,
)

from .normalization import (
    FullWindowNormalizer,
    SlidingMANormalizer,
    NormalizerFactory,
    get_normalizer,
)

from .splitting import (
    time_split,
    create_target_blocks,
    block_split,
    validate_no_leakage,
    create_backtest_samples,
    get_split_stats,
)

from .metrics import (
    detrend_to_baseline,
    safe_corrcoef,
    safe_spearmanr,
    safe_trajectory_ic,
    excess_da,
    aggregate_ic,
    aggregate_da,
    amplitude_error_rate,
    compute_amplitude_stats,
    limit_hit_rate,
    detect_limit,
    calculate_combined_score,
    calculate_da_score,
    get_log_step_weights,
    get_feature_weights,
    format_metrics_report,
    FEATURE_NAMES,
)

from .utils import (
    set_seed,
    get_device,
    format_time,
    format_timestamp,
    get_model_size,
    count_parameters,
    ensure_dir,
    safe_save_json,
    safe_load_json,
    safe_save_pickle,
    safe_load_pickle,
    extract_time_features,
    get_rank_info,
    cleanup_ddp,
    debug_print,
    verbose_print,
)

from .dataset import (
    KronosDataset,
    KronosWindowedDataset,
    load_split_data,
    collate_fn,
)


__all__ = [
    # Config
    'DataConfig',
    'TrainConfig',
    'ArtifactConfig',
    'BacktestConfig',
    'parse_norm_mode',

    # Paths
    'get_raw_path',
    'get_backtest_raw_path',
    'get_split_data_path',
    'get_backtest_data_path',
    'get_meta_path',
    'get_model_path',
    'get_tokenizer_path',
    'get_checkpoint_path',
    'get_training_info_path',
    'get_summary_path',
    'ensure_dir',

    # Schema
    'SampleSchema',
    'BacktestSchema',
    'MetaSchema',
    'TrainingInfoSchema',
    'sample_to_dict',
    'dict_to_sample',
    'intervals_overlap',
    'compute_fingerprint',

    # Normalization
    'FullWindowNormalizer',
    'SlidingMANormalizer',
    'NormalizerFactory',
    'get_normalizer',

    # Splitting
    'time_split',
    'create_target_blocks',
    'block_split',
    'validate_no_leakage',
    'create_backtest_samples',
    'get_split_stats',

    # Metrics
    'detrend_to_baseline',
    'safe_corrcoef',
    'safe_spearmanr',
    'safe_trajectory_ic',
    'excess_da',
    'aggregate_ic',
    'aggregate_da',
    'amplitude_error_rate',
    'compute_amplitude_stats',
    'limit_hit_rate',
    'detect_limit',
    'calculate_combined_score',
    'calculate_da_score',
    'get_log_step_weights',
    'get_feature_weights',
    'format_metrics_report',
    'FEATURE_NAMES',

    # Utils
    'set_seed',
    'get_device',
    'format_time',
    'format_timestamp',
    'get_model_size',
    'count_parameters',
    'ensure_dir',
    'safe_save_json',
    'safe_load_json',
    'safe_save_pickle',
    'safe_load_pickle',
    'extract_time_features',
    'get_rank_info',
    'cleanup_ddp',
    'debug_print',
    'verbose_print',

    # Dataset
    'KronosDataset',
    'KronosWindowedDataset',
    'load_split_data',
    'collate_fn',
]