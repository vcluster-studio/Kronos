import os

class Config:
    """
    Configuration class for the entire project.
    All paths are relative to project root - run scripts from project root directory.
    """

    def __init__(self):
        # Project root directory (Kronos/)
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # =================================================================
        # Data & Feature Parameters
        # =================================================================
        self.qlib_data_path = "~/.qlib/qlib_data/cn_data"  # Not used for CSV preprocessing
        self.instrument = 'main_board'

        # Overall time range
        self.dataset_begin_time = "2017-10-01"
        self.dataset_end_time = '2026-05-07'

        # Sliding window parameters
        self.lookback_window = 90
        self.predict_window = 10
        self.max_context = 512

        # Features
        self.feature_list = ['open', 'high', 'low', 'close', 'vol', 'amt']
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']

        # =================================================================
        # Dataset Splitting & Paths
        # =================================================================
        self.train_time_range = ["2018-01-02", "2024-12-31"]
        self.val_time_range = ["2025-01-02", "2025-06-30"]
        self.test_time_range = ["2025-07-01", "2026-05-07"]
        self.backtest_time_range = ["2025-07-01", "2026-05-07"]

        # Paths (relative to project root)
        self.dataset_path = os.path.join(self.project_root, "finetune", "data", "processed_datasets")

        # =================================================================
        # Training Hyperparameters (OPTIMIZED CONFIG v2)
        # =================================================================
        self.clip = 5.0

        # === 优化后的训练配置 ===
        self.epochs = 30                   # 正常训练轮数
        self.log_interval = 100            # 每100步打印日志
        self.batch_size = 8                # 适应 2GB 显存

        # === 优化：增加训练样本量 ===
        self.n_train_iter = 5000 * self.batch_size   # 原值 3000，增加到 5000（每轮 40000 样本）
        self.n_val_iter = 800 * self.batch_size      # 原值 500，增加到 800（每轮 6400 样本）

        # Learning rates
        self.tokenizer_learning_rate = 2e-4
        # === 优化：降低预测器学习率，防止过拟合 ===
        self.predictor_learning_rate = 1e-5          # 原值 4e-5，降低 4 倍

        self.accumulation_steps = 1
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_weight_decay = 0.1
        self.seed = 100

        # =================================================================
        # Experiment Logging & Saving
        # =================================================================
        self.use_comet = False
        self.comet_config = {
            "api_key": "YOUR_COMET_API_KEY",
            "project_name": "Kronos-A-Share-Finetune",
            "workspace": "your_comet_workspace"
        }
        self.comet_tag = 'a_share_main_board'
        self.comet_name = 'a_share_main_board_finetune'

        # Save paths
        self.save_path = os.path.join(self.project_root, "outputs", "models")
        self.tokenizer_save_folder_name = 'a_share_tokenizer_v2'    # v2 版本，优化后
        self.predictor_save_folder_name = 'a_share_predictor_v2'    # v2 版本，优化后
        self.backtest_save_folder_name = 'a_share_backtest_v2'
        self.backtest_result_path = os.path.join(self.project_root, "outputs", "backtest_results")

        # =================================================================
        # Model & Checkpoint Paths
        # =================================================================
        # Pretrained models
        self.pretrained_tokenizer_path = os.path.join(self.project_root, "pretrained", "Kronos-Tokenizer-base")
        self.pretrained_predictor_path = os.path.join(self.project_root, "pretrained", "Kronos-small")

        # Fine-tuned models (output)
        # 分词器用之前训练好的版本（已收敛）
        self.finetuned_tokenizer_path = os.path.join(self.save_path, "a_share_tokenizer", "checkpoints", "best_model")
        # 预测器是新版本（优化后）
        self.finetuned_predictor_path = os.path.join(self.save_path, self.predictor_save_folder_name, "checkpoints", "best_model")

        # =================================================================
        # Backtesting Parameters
        # =================================================================
        self.backtest_n_symbol_hold = 50
        self.backtest_n_symbol_drop = 5
        self.backtest_hold_thresh = 5
        self.inference_T = 0.6
        self.inference_top_p = 0.9
        self.inference_top_k = 0
        self.inference_sample_count = 5
        self.backtest_batch_size = 1000
        self.backtest_benchmark = self._set_benchmark(self.instrument)

    def _set_benchmark(self, instrument):
        dt_benchmark = {
            'csi800': "SH000906",
            'csi1000': "SH000852",
            'csi300': "SH000300",
            'main_board': "SH000300",
        }
        return dt_benchmark.get(instrument, "SH000300")
