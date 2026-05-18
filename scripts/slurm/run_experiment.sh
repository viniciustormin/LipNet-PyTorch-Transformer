#!/bin/bash
#SBATCH --job-name=lipnet-gru-vs-transformer
#SBATCH --partition=h100n3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=UNLIMITED
#SBATCH --output=/raid/user_viniciustormin/logs/%x_%j.log
#SBATCH --error=/raid/user_viniciustormin/logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
COMMON_SH="${PROJECT_ROOT}/scripts/slurm/common.sh"

[[ -f "${COMMON_SH}" ]] || { echo "ERRO: rode 'sbatch scripts/slurm/run_experiment.sh' a partir da raiz do repositório." >&2; exit 1; }
export PROJECT_ROOT="$(cd "$(dirname "${COMMON_SH}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${COMMON_SH}"

detect_container_runtime
build_container_args

log "=== Experimento LipNet GRU vs Transformer ==="
print_job_context

mkdir -p checkpoints/gru checkpoints/transformer results

setup_extra_deps

# ---------------------------------------------------------------------------
# Step 1: Train GRU
# ---------------------------------------------------------------------------
log "STEP 1/3 — Treinando LipNet-GRU..."
container_exec python train_transformer.py \
    --model         gru \
    --video_path    "${LIP_DIR}" \
    --anno_path     "${ALIGN_DIR}" \
    --train_list    data/overlap_train.txt \
    --val_list      data/overlap_val.txt \
    --batch_size    "${BATCH_SIZE}" \
    --lr            "${LR}" \
    --max_epoch     "${MAX_EPOCH}" \
    --seed          "${SEED}" \
    --dropout       "${DROPOUT}" \
    --num_workers   "${NUM_WORKERS}" \
    --vid_padding   "${VID_PADDING}" \
    --txt_padding   "${TXT_PADDING}" \
    --num_classes   "${NUM_CLASSES}" \
    --save_dir      checkpoints/gru \
    --display       100 \
    --gpu           0

log "GRU concluído."

# ---------------------------------------------------------------------------
# Step 2: Train Transformer
# ---------------------------------------------------------------------------
log "STEP 2/3 — Treinando LipNet-Transformer..."
container_exec python train_transformer.py \
    --model           transformer \
    --video_path      "${LIP_DIR}" \
    --anno_path       "${ALIGN_DIR}" \
    --train_list      data/overlap_train.txt \
    --val_list        data/overlap_val.txt \
    --batch_size      "${BATCH_SIZE}" \
    --lr              "${LR}" \
    --max_epoch       "${MAX_EPOCH}" \
    --seed            "${SEED}" \
    --dropout         "${DROPOUT}" \
    --num_workers     "${NUM_WORKERS}" \
    --vid_padding     "${VID_PADDING}" \
    --txt_padding     "${TXT_PADDING}" \
    --num_classes     "${NUM_CLASSES}" \
    --d_model         512 \
    --nhead           8 \
    --num_layers      2 \
    --dim_feedforward 2048 \
    --attn_dropout    0.1 \
    --save_dir        checkpoints/transformer \
    --display         100 \
    --gpu             0

log "Transformer concluído."

# ---------------------------------------------------------------------------
# Step 3: Compare
# ---------------------------------------------------------------------------
log "STEP 3/3 — Comparando modelos..."
container_exec python compare_models.py \
    --gru_ckpt   checkpoints/gru/gru_best.pt \
    --gru_hist   checkpoints/gru/history.json \
    --trans_ckpt checkpoints/transformer/transformer_best.pt \
    --trans_hist checkpoints/transformer/history.json \
    --video_path "${LIP_DIR}" \
    --anno_path  "${ALIGN_DIR}" \
    --val_list   data/overlap_val.txt \
    --vid_padding  "${VID_PADDING}" \
    --txt_padding  "${TXT_PADDING}" \
    --num_classes  "${NUM_CLASSES}" \
    --batch_size   "${BATCH_SIZE}" \
    --num_workers  4 \
    --d_model         512 \
    --nhead           8 \
    --num_layers      2 \
    --dim_feedforward 2048 \
    --out_dir    results \
    --gpu        0

log "=== Experimento concluído. Resultados em: results/ ==="
