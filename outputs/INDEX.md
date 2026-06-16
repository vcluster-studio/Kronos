# Model Output Directory Index

## Current Generation (60-step lookback, HF Trainer via train_single_step.py, NO data leakage)

| Directory | Predict | Samples | LR | Status | Key Results |
|-----------|---------|---------|-----|--------|-------------|
| `ma60_predictor_lb60_pd1/` | 1 | Full (~5.5M) | 0.003 | Completed 30 epochs | open_ic=0.597, close_ic=0.089, vol_ic=0.991 |
| `ma60_predictor_lb60_pd5_samp50000/` | 5 | 50k | 0.003 | Killed at epoch 5 | close_ic_step5=0.059, open_ic_step3=0.100, vol_ic_step5=0.95 |
| `ma60_predictor_lb60_pd5_lr01/` | 5 | 50k | 0.01 | Training (2026-06-08) | eval_loss=7.82 (epoch 1), close_ic_step3=-0.007 |

## Previous Generation (400-step lookback, old training pipeline, ALL RESULTS INVALID)

**WARNING: All models below were trained BEFORE commit `4c569aa` which fixed the `window_size = lookback + predict + 1` data leakage bug. Their IC metrics are inflated and unreliable.**

| Directory | Model | Notes |
|-----------|-------|-------|
| `ma60_predictor_v1/` | Kronos-base | INVALID — data leakage |
| `ma60_predictor_v2/` | Kronos-base | INVALID — data leakage |
| `ma60_predictor_mini_v5/` | Kronos-mini | INVALID — checkpoints removed, only summary.json kept as lesson |
| `ma60_predictor_mini_v6c/` | Kronos-mini | INVALID — data leakage |
| `ma60_predictor_mini_v6d/` | Kronos-mini | INVALID — data leakage |
| `ma60_predictor_mini_v6e/` | Kronos-mini | INVALID — data leakage |
| `ma60_predictor_small_v1/` ~ `v6/` | Kronos-small | INVALID — data leakage |

## Key Findings (updated 2026-06-09)

1. **Data leakage invalidated all pre-4c569aa results**: `window_size = lookback + predict + 1` leaked 1 future step. mini_v5's close_ic=0.33 was not real.
2. **EvalCallback ground truth leakage bug**: eval_results.jsonl metrics are fake (vol IC=0.99 was from GT tokens in forward). Fixed in train_single_step.py.
3. **True auto_regressive IC for lb60_pd1**: open IC=0.64, close/vol/amt IC≈0 (real performance).
4. **direction_loss/decoder/probe are dead ends**: hidden[:, -1, :] doesn't encode feature-specific info. Must use token generation.
5. **Gradient kidnapping persists**: vol/amt dominate CE loss regardless of LR. This is the core unsolved problem.
6. **Higher LR (0.01) made it worse**: Accelerated vol/amt learning, further starving price features.
