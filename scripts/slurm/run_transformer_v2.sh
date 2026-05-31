#!/bin/bash
#SBATCH --job-name=lipnet-transformer-v2
#SBATCH --partition=h100n3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=UNLIMITED
#SBATCH --output=/raid/user_viniciustormin/logs/%x_%j.log
#SBATCH --error=/raid/user_viniciustormin/logs/%x_%j.err

# Treina Transformer v2 (beam search CTC + warmup cosine scheduler) com todos os speakers.
# Resume automático: rode de novo se interrompido — retoma do último epoch salvo.
#
# Uso:
#   sbatch scripts/slurm/run_transformer_v2.sh
#
# Variáveis de ambiente opcionais:
#   SPEAKERS=all   BATCH_SIZE=64   WARMUP_STEPS=4000   MAX_EPOCH=100

set -euo pipefail

# Override de padrões antes de sourciar common.sh (que usa ${VAR:-default})
SPEAKERS="${SPEAKERS:-all}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-16}"
WARMUP_STEPS="${WARMUP_STEPS:-4000}"
MAX_EPOCH="${MAX_EPOCH:-100}"

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
COMMON_SH="${PROJECT_ROOT}/scripts/slurm/common.sh"

[[ -f "${COMMON_SH}" ]] || { echo "ERRO: rode 'sbatch scripts/slurm/run_transformer_v2.sh' a partir da raiz do repositório." >&2; exit 1; }
export PROJECT_ROOT="$(cd "$(dirname "${COMMON_SH}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${COMMON_SH}"

detect_container_runtime
build_container_args

log "=== LipNet Transformer v2 (beam search + warmup ${WARMUP_STEPS} steps) | speakers=${SPEAKERS} (${RUN_TAG}) ==="
print_job_context
log "BATCH_SIZE : ${BATCH_SIZE}  |  NUM_WORKERS: ${NUM_WORKERS}  |  MAX_EPOCH: ${MAX_EPOCH}"

V2_DIR="${CKPT_ROOT}/transformer_v2"
mkdir -p "${V2_DIR}" "${RESULTS_DIR}"

setup_extra_deps

make_data_lists

log "=== Treinando Transformer v2 ==="
log "Save dir: ${V2_DIR}"

container_exec python train_transformer.py \
    --model           transformer \
    --use_warmup \
    --warmup_steps    "${WARMUP_STEPS}" \
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
    --save_dir        "${V2_DIR}" \
    --display         100 \
    --gpu             0 \
    --d_model         512 \
    --nhead           8 \
    --num_layers      4 \
    --freeze_frontend_epochs 20 \
    --dim_feedforward 2048 \
    --attn_dropout    0.1

log "=== Transformer v2 concluído ==="
log "Checkpoint : ${V2_DIR}/transformer_best.pt"
log "History    : ${V2_DIR}/history.json"
log ""
log "Para gerar a comparação final no Mac:"
log "  rsync -av <cluster>:${V2_DIR}/ checkpoints/${RUN_TAG}/transformer_v2/"
log "  python compare_final.py --v2_ckpt checkpoints/${RUN_TAG}/transformer_v2/transformer_best.pt ..."
