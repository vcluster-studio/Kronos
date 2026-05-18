# =============================================================================
# Kronos Training - Entrypoint Script
# =============================================================================
# Pretrained weights must be mounted from host via volume:
#   -v /path/to/pretrained:/app/pretrained
#
# Required model files in /app/pretrained:
#   Kronos-Tokenizer-2k/   (from NeoQuasar/Kronos-Tokenizer-2k)
#   Kronos-mini/           (from NeoQuasar/Kronos-mini)
#
# Usage:
#   bash entrypoint.sh --task tokenizer --dataset mid
#   bash entrypoint.sh --task predictor --dataset mid
#   bash entrypoint.sh --task preprocess --dataset mid
#   bash entrypoint.sh --task evaluate --dataset mid
#   bash entrypoint.sh --task pipeline --dataset mid
#   bash entrypoint.sh --help
# =============================================================================

set -e

# ---- Default Values ----
TASK=""
DATASET="mid"
EPOCHS=""
BATCH_SIZE=""
LR=""
PREDICTOR_EPOCHS=""
PREDICTOR_LR=""

# ---- Color Output ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "${BLUE}[STEP]${NC} $*"; }

# ---- Parse Arguments ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)       TASK="$2"; shift 2 ;;
        --dataset)    DATASET="$2"; shift 2 ;;
        --epochs)     EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --lr)         LR="$2"; shift 2 ;;
        --predictor-epochs) PREDICTOR_EPOCHS="$2"; shift 2 ;;
        --predictor-lr)     PREDICTOR_LR="$2"; shift 2 ;;
        --help)       show_help; exit 0 ;;
        *)            log_error "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

show_help() {
    cat << 'EOF'
Kronos Training Entrypoint

Usage:
    bash entrypoint.sh --task <task> [options]

Tasks:
    tokenizer        Train tokenizer only
    predictor        Train predictor only
    preprocess       Run data preprocessing only
    evaluate         Run unified evaluation only
    pipeline         Full pipeline: preprocess -> tokenizer -> predictor -> evaluate
    shell            Drop into bash shell

Options:
    --dataset <name>         Dataset name: mid, small, mid_small, full (default: mid)
    --epochs <n>             Tokenizer training epochs (default: from script)
    --batch-size <n>         Batch size override
    --lr <float>             Learning rate override
    --predictor-epochs <n>   Predictor training epochs
    --predictor-lr <float>   Predictor learning rate

Volume Mounts (required):
    /app/pretrained      Pretrained model weights (MUST be mounted from host)
                         Required files:
                           - Kronos-Tokenizer-2k/
                           - Kronos-mini/
    /app/finetune_csv    Raw CSV data for preprocessing
    /app/output          Training output (models, checkpoints)
    /app/logs            Training logs

Examples:
    # Train tokenizer on mid-cap data
    docker run --gpus all \
        -v /root/kronos/pretrained:/app/pretrained \
        -v /root/kronos/csv:/app/finetune_csv \
        -v /root/kronos/output:/app/output \
        kronos-train bash entrypoint.sh --task tokenizer --dataset mid

    # Full pipeline
    docker run --gpus all \
        -v /root/kronos/pretrained:/app/pretrained \
        -v /root/kronos/csv:/app/finetune_csv \
        -v /root/kronos/output:/app/output \
        -v /root/kronos/logs:/app/logs \
        kronos-train bash entrypoint.sh --task pipeline --dataset mid --epochs 20
EOF
}

# ---- Verify Pretrained Models ----
verify_models() {
    log_step "Verifying pretrained models..."

    local missing=0

    if [ ! -d "/app/pretrained/Kronos-Tokenizer-2k" ]; then
        log_error "Missing: /app/pretrained/Kronos-Tokenizer-2k"
        log_error "  Download from: https://huggingface.co/NeoQuasar/Kronos-Tokenizer-2k"
        missing=1
    else
        log_info "Found: Kronos-Tokenizer-2k"
    fi

    if [ ! -d "/app/pretrained/Kronos-mini" ]; then
        log_error "Missing: /app/pretrained/Kronos-mini"
        log_error "  Download from: https://huggingface.co/NeoQuasar/Kronos-mini"
        missing=1
    else
        log_info "Found: Kronos-mini"
    fi

    if [ "$missing" -eq 1 ]; then
        log_error "Pretrained models not found! Mount them via: -v /path/to/pretrained:/app/pretrained"
        log_error "Or download manually:"
        log_error "  git clone https://huggingface.co/NeoQuasar/Kronos-Tokenizer-2k pretrained/Kronos-Tokenizer-2k"
        log_error "  git clone https://huggingface.co/NeoQuasar/Kronos-mini pretrained/Kronos-mini"
        exit 1
    fi

    log_info "All pretrained models verified"
}

# ---- Data Preprocessing ----
run_preprocess() {
    log_step "Running data preprocessing for dataset: ${DATASET}"

    # Determine exclude categories
    EXCLUDE_ARG=""
    OUTPUT_ARG=""
    case $DATASET in
        mid)       EXCLUDE_ARG="--exclude large small"; OUTPUT_ARG="--output processed_datasets_mid" ;;
        small)     EXCLUDE_ARG="--exclude large mid";   OUTPUT_ARG="--output processed_datasets_small" ;;
        mid_small) EXCLUDE_ARG="--exclude large";       OUTPUT_ARG="--output processed_datasets_mid_small" ;;
        full)      EXCLUDE_ARG="";                      OUTPUT_ARG="" ;;
        *)         log_error "Unknown dataset: ${DATASET}"; exit 1 ;;
    esac

    # Check if data already processed
    DATA_DIR="/app/finetune/data/processed_datasets_${DATASET}"
    if [ "$DATASET" = "full" ]; then
        DATA_DIR="/app/finetune/data/processed_datasets"
    fi

    if [ -f "${DATA_DIR}/train_data.pkl" ] && [ -f "${DATA_DIR}/val_data.pkl" ] && [ -f "${DATA_DIR}/test_data.pkl" ]; then
        log_info "Processed data already exists at ${DATA_DIR}, skipping preprocessing"
        return
    fi

    # Verify CSV data exists
    if [ ! -d "/app/finetune_csv/exported_kline_data/stocks" ]; then
        log_error "CSV data not found! Mount via: -v /path/to/csv:/app/finetune_csv"
        exit 1
    fi

    log_info "Preprocessing data..."
    python -u finetune/csv_data_preprocess.py $EXCLUDE_ARG $OUTPUT_ARG

    log_info "Data preprocessing complete"
}

# ---- Train Tokenizer ----
train_tokenizer() {
    log_step "Training tokenizer on dataset: ${DATASET}"

    CMD="python -u finetune/train_tokenizer.py --dataset ${DATASET}"
    [ -n "$EPOCHS" ]     && CMD="$CMD --epochs $EPOCHS"
    [ -n "$BATCH_SIZE" ] && CMD="$CMD --batch-size $BATCH_SIZE"
    [ -n "$LR" ]         && CMD="$CMD --lr $LR"

    log_info "Command: $CMD"
    eval $CMD

    log_info "Tokenizer training complete"
}

# ---- Train Predictor ----
train_predictor() {
    log_step "Training predictor on dataset: ${DATASET}"

    CMD="python -u finetune/train_predictor.py --dataset ${DATASET}"
    [ -n "$PREDICTOR_EPOCHS" ] && CMD="$CMD --epochs $PREDICTOR_EPOCHS"
    [ -n "$PREDICTOR_LR" ]     && CMD="$CMD --lr $PREDICTOR_LR"
    [ -n "$BATCH_SIZE" ]       && CMD="$CMD --batch-size $BATCH_SIZE"

    log_info "Command: $CMD"
    eval $CMD

    log_info "Predictor training complete"
}

# ---- Run Evaluation ----
run_evaluate() {
    log_step "Running unified evaluation..."

    python -u finetune/unified_test.py

    log_info "Evaluation complete"
}

# ---- Print System Info ----
print_system_info() {
    log_step "System Information"
    echo "============================================"
    python -c "
import torch
print(f'PyTorch:  {torch.__version__}')
print(f'CUDA:     {torch.version.cuda}')
print(f'GPU:      {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
if torch.cuda.is_available():
    print(f'GPU Mem:  {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print(f'GPUs:     {torch.cuda.device_count()}')
import numpy as np; print(f'NumPy:    {np.__version__}')
import pandas as pd; print(f'Pandas:   {pd.__version__}')
"
    echo "============================================"
}

# ---- Main ----
log_info "Kronos Training Container"
log_info "Task: ${TASK:-not specified}"
log_info "Dataset: ${DATASET}"

print_system_info

case "$TASK" in
    tokenizer)
        verify_models
        run_preprocess
        train_tokenizer
        ;;
    predictor)
        verify_models
        run_preprocess
        train_predictor
        ;;
    preprocess)
        run_preprocess
        ;;
    evaluate)
        verify_models
        run_evaluate
        ;;
    pipeline)
        log_step "Running FULL pipeline: preprocess -> tokenizer -> predictor -> evaluate"
        verify_models
        run_preprocess
        train_tokenizer
        train_predictor
        run_evaluate
        log_info "Full pipeline complete!"
        ;;
    shell)
        log_info "Dropping into shell..."
        exec /bin/bash
        ;;
    "")
        log_error "No task specified. Use --task <task>"
        show_help
        exit 1
        ;;
    *)
        log_error "Unknown task: $TASK"
        show_help
        exit 1
        ;;
esac
