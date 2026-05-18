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

log "=== Experimento LipNet: ${TRAIN_MODELS} | speakers=${SPEAKERS} (${RUN_TAG}) ==="
print_job_context

mkdir -p "${CKPT_ROOT}" "${RESULTS_DIR}"

setup_extra_deps

# ---------------------------------------------------------------------------
# Gera data/active_train.txt e data/active_val.txt para os speakers ativos
# ---------------------------------------------------------------------------
make_data_lists

# ---------------------------------------------------------------------------
# Treina cada modelo listado em TRAIN_MODELS
# ---------------------------------------------------------------------------
step=1
total_models=$(echo "${TRAIN_MODELS}" | wc -w)

for model in ${TRAIN_MODELS}; do
    log "STEP ${step}/$((total_models + 1)) — Treinando LipNet-${model^^}..."
    mkdir -p "${CKPT_ROOT}/${model}"

    common_args=(
        --video_path    "${LIP_DIR}"
        --anno_path     "${ALIGN_DIR}"
        --train_list    "${TRAIN_LIST}"
        --val_list      "${VAL_LIST}"
        --batch_size    "${BATCH_SIZE}"
        --lr            "${LR}"
        --max_epoch     "${MAX_EPOCH}"
        --seed          "${SEED}"
        --dropout       "${DROPOUT}"
        --num_workers   "${NUM_WORKERS}"
        --vid_padding   "${VID_PADDING}"
        --txt_padding   "${TXT_PADDING}"
        --num_classes   "${NUM_CLASSES}"
        --save_dir      "${CKPT_ROOT}/${model}"
        --display       100
        --gpu           0
    )

    if [[ "${model}" == "transformer" ]]; then
        container_exec python train_transformer.py \
            --model           transformer \
            "${common_args[@]}" \
            --d_model         512 \
            --nhead           8 \
            --num_layers      2 \
            --dim_feedforward 2048 \
            --attn_dropout    0.1
    else
        container_exec python train_transformer.py \
            --model gru \
            "${common_args[@]}"
    fi

    log "${model^^} concluído."
    step=$((step + 1))
done

# ---------------------------------------------------------------------------
# Comparação — só executa se ambos gru e transformer foram treinados
# ---------------------------------------------------------------------------
gru_ckpt="${CKPT_ROOT}/gru/gru_best.pt"
trans_ckpt="${CKPT_ROOT}/transformer/transformer_best.pt"

if [[ -f "${gru_ckpt}" ]] && [[ -f "${trans_ckpt}" ]]; then
    log "STEP ${step}/$((total_models + 1)) — Comparando modelos..."
    container_exec python compare_models.py \
        --gru_ckpt   "${gru_ckpt}" \
        --gru_hist   "${CKPT_ROOT}/gru/history.json" \
        --trans_ckpt "${trans_ckpt}" \
        --trans_hist "${CKPT_ROOT}/transformer/history.json" \
        --video_path "${LIP_DIR}" \
        --anno_path  "${ALIGN_DIR}" \
        --val_list   "${VAL_LIST}" \
        --vid_padding  "${VID_PADDING}" \
        --txt_padding  "${TXT_PADDING}" \
        --num_classes  "${NUM_CLASSES}" \
        --batch_size   "${BATCH_SIZE}" \
        --num_workers  4 \
        --d_model         512 \
        --nhead           8 \
        --num_layers      2 \
        --dim_feedforward 2048 \
        --out_dir    "${RESULTS_DIR}" \
        --gpu        0
    log "Comparação concluída."
else
    log "Pulando comparação (requer checkpoints de gru E transformer)."
    [[ -f "${gru_ckpt}" ]]   || log "  Faltando: ${gru_ckpt}"
    [[ -f "${trans_ckpt}" ]] || log "  Faltando: ${trans_ckpt}"
fi

log "=== Experimento concluído. Resultados em: ${RESULTS_DIR}/ ==="
