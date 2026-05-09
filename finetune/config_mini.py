"""
Kronos-mini 微调配置

mini 版本特点：
- 参数量小（4.1M），训练快
- 适合快速实验和调参
- 可作为 small 版本的参数搜索基础

用法：
    from finetune.config_mini import ConfigMini
"""

import os

class ConfigMini:
    """
    Kronos-mini 微调配置类
    """

    def __init__(self):
        # Project root
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # =================================================================
        # Model Selection - MINI VERSION
        # =================================================================
        self.model_version = "mini"  # mini / small / base

        # =================================================================
        # Data & Feature Parameters
        # =================================================================
        self.qlib_data_path = "~/.qlib/qlib_data/cn_data"
        self.instrument = 'main_board'

        self.dataset_begin_time = "2017-10-01"
        self.dataset_end_time = '2026-05-07'

        self.lookback_window = 90
        self.predict_window = 10
        self.max_context = 512  # mini 支持 2048，但保持 512 便于对比

        self.feature_list = ['open', 'high', 'low', 'close', 'vol', 'amt']
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']

        # =================================================================
        # Dataset Splitting
        # =================================================================
        self.train_time_range = ["2018-01-02", "2024-12-31"]
        self.val_time_range = ["2025-01-02", "2025-06-30"]
        self.test_time_range = ["2025-07-01", "2026-05-07"]
        self.backtest_time_range = ["2025-07-01", "2026-05-07"]

        self.dataset_path = os.path.join(self.project_root, "finetune", "data", "processed_datasets")

        # =================================================================
        # Training Hyperparameters - MINI OPTIMIZED
        # =================================================================
        self.clip = 5.0

        # === 快速验证配置 ===
        self.epochs = 20
        self.log_interval = 50  # 更频繁日志
        self.batch_size = 16    # mini 更小，可以更大 batch

        # 样本量（基于 V1 经验）
        self.n_train_iter = 2000 * self.batch_size   # 32,000 样本
        self.n_val_iter = 400 * self.batch_size      # 6,400 样本

        # 学习率（基于 V1 成功经验）
        self.tokenizer_learning_rate = 2e-4
        self.predictor_learning_rate = 1e-2

        self.accumulation_steps = 1
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_weight_decay = 0.1
        self.seed = 100

        # === 改进的早停配置 ===
        self.early_stopping_patience = 5       # 至少5轮负增长才停止
        self.early_stopping_min_delta = 0.0001 # 最小改善阈值（相对于0.01级别的val loss）
        self.early_stopping_grace_period = 3   # 前 3 轮不早停
        self.early_stopping_check_ic = True    # 同时检查 IC
        self.early_stopping_window = 5         # 看最近 5 轮趋势

        # === 学习率调度配置 ===
        # 使用周期性学习率跳出局部最优
        self.lr_scheduler = 'cosine_warmup'    # cosine_warmup / onecycle / plateau
        self.lr_T_0 = 5                        # 第一个周期长度（epochs）
        self.lr_T_mult = 2                     # 每次周期翻倍
        self.lr_eta_min = 1e-6                 # 最小学习率

        # =================================================================
        # Paths - MINI VERSION
        # =================================================================
        self.save_path = os.path.join(self.project_root, "outputs", "models")

        self.tokenizer_save_folder_name = f'mini_tokenizer_v1'
        self.predictor_save_folder_name = f'mini_predictor_v1'

        self.backtest_save_folder_name = 'mini_backtest_v1'
        self.backtest_result_path = os.path.join(self.project_root, "outputs", "backtest_results")

        # Pretrained models - MINI
        self.pretrained_tokenizer_path = os.path.join(self.project_root, "pretrained", "Kronos-Tokenizer-base")
        self.pretrained_predictor_path = os.path.join(self.project_root, "pretrained", "Kronos-mini")

        # Fine-tuned models (use latest_model since final val_loss 0.00897 < best saved 0.00958)
        self.finetuned_tokenizer_path = os.path.join(self.save_path, self.tokenizer_save_folder_name, "checkpoints", "latest_model")
        self.finetuned_predictor_path = os.path.join(self.save_path, self.predictor_save_folder_name, "checkpoints", "best_model")

        # =================================================================
        # Experiment Tracking
        # =================================================================
        self.use_comet = False
        self.comet_config = {
            "api_key": "YOUR_COMET_API_KEY",
            "project_name": "Kronos-Mini-Finetune",
            "workspace": "your_workspace"
        }
        self.comet_tag = 'mini_v1'
        self.comet_name = 'mini_finetune_v1'

        # =================================================================
        # Backtesting
        # =================================================================
        self.backtest_n_symbol_hold = 50
        self.backtest_n_symbol_drop = 5
        self.backtest_hold_thresh = 5
        self.inference_T = 0.6
        self.inference_top_p = 0.9
        self.inference_top_k = 0
        self.inference_sample_count = 5
        self.backtest_batch_size = 1000
        self.backtest_benchmark = "SH000300"

    def set_experiment(self, exp_name, lr=None, batch_size=None, n_train=None):
        """
        快速设置实验参数

        Args:
            exp_name: 实验名称
            lr: 学习率（可选）
            batch_size: 批次大小（可选）
            n_train: 训练样本数（可选）
        """
        self.tokenizer_save_folder_name = f'mini_tokenizer_{exp_name}'
        self.predictor_save_folder_name = f'mini_predictor_{exp_name}'
        self.comet_tag = f'mini_{exp_name}'
        self.comet_name = f'mini_finetune_{exp_name}'

        if lr is not None:
            self.predictor_learning_rate = lr
        if batch_size is not None:
            self.batch_size = batch_size
            self.n_train_iter = 2000 * batch_size
            self.n_val_iter = 400 * batch_size
        if n_train is not None:
            self.n_train_iter = n_train

        # 更新路径
        self.finetuned_tokenizer_path = os.path.join(self.save_path, self.tokenizer_save_folder_name, "checkpoints", "best_model")
        self.finetuned_predictor_path = os.path.join(self.save_path, self.predictor_save_folder_name, "checkpoints", "best_model")

        print(f"Experiment: {exp_name}")
        print(f"  LR: {self.predictor_learning_rate}")
        print(f"  Batch: {self.batch_size}")
        print(f"  Train samples: {self.n_train_iter}")
