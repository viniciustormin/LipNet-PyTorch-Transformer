#!/bin/bash
#SBATCH --job-name=lipnet-gru-all
#SBATCH --partition=h100n3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=UNLIMITED
#SBATCH --output=/raid/user_viniciustormin/logs/%x_%j.log
#SBATCH --error=/raid/user_viniciustormin/logs/%x_%j.err

# Retreina LipNet-GRU em todos os speakers com batch maior e mais épocas.
# Resume automático: rode de novo se interrompido.
#
# Uso:
#   sbatch scripts/slurm/run_gru_all.sh
#
# Variáveis de ambiente opcionais:
#   SPEAKERS=all   BATCH_SIZE=32   MAX_EPOCH=200

set -euo pipefail

SPEAKERS="${SPEAKERS:-all}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-16}"
MAX_EPOCH="${MAX_EPOCH:-200}"

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
COMMON_SH="${PROJECT_ROOT}/scripts/slurm/common.sh"

[[ -f "${COMMON_SH}" ]] || { echo "ERRO: rode 'sbatch scripts/slurm/run_gru_all.sh' a partir da raiz do repositório." >&2; exit 1; }
export PROJECT_ROOT="$(cd "$(dirname "${COMMON_SH}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${COMMON_SH}"

detect_container_runtime
build_container_args

log "=== LipNet-GRU (todos os speakers) | speakers=${SPEAKERS} (${RUN_TAG}) ==="
print_job_context
log "BATCH_SIZE : ${BATCH_SIZE}  |  NUM_WORKERS: ${NUM_WORKERS}  |  MAX_EPOCH: ${MAX_EPOCH}"

GRU_DIR="${CKPT_ROOT}/gru_v2"
mkdir -p "${GRU_DIR}" "${RESULTS_DIR}"

setup_extra_deps

make_data_lists

log "=== Treinando LipNet-GRU ==="
log "Save dir: ${GRU_DIR}"

container_exec python train_transformer.py \
    --model           gru \
    --video_path      "${LIP_DIR}" \
    --anno_path       "${ALIGN_DIR}" \
    --train_list      "${TRAIN_LIST}" \
    --val_list        "${VAL_LIST}" \
    --batch_size      "${BATCH_SIZE}" \
    --lr              "${LR}" \
    --max_epoch       "${MAX_EPOCH}" \
    --seed            "${SEED}" \
    --dropout         "${DROPOUT}" \
    --num_workers     "${NUM_WORKERS}" \
    --vid_padding     "${VID_PADDING}" \
    --txt_padding     "${TXT_PADDING}" \
    --num_classes     "${NUM_CLASSES}" \
    --save_dir        "${GRU_DIR}" \
    --display         100 \
    --gpu             0

log "=== GRU concluído ==="
log "Checkpoint : ${GRU_DIR}/gru_best.pt"
log "History    : ${GRU_DIR}/history.json"
