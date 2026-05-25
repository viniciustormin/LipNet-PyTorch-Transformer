#!/bin/bash
# Treina apenas o Transformer v2 (beam search + warmup cosine) e gera comparação final.
#
# Pré-requisito: GRU e Transformer v1 já treinados em checkpoints/${RUN_TAG}/
#   checkpoints/s1/gru/gru_best.pt
#   checkpoints/s1/transformer/transformer_best.pt
#
# Uso:
#   bash scripts/macos/run_transformer_v2.sh
#
# Resume automático: se interrompido, rode de novo — retoma do último epoch salvo.
# Variáveis de ambiente opcionais:
#   MAX_EPOCH=50  BATCH_SIZE=4  DEVICE=auto  WARMUP_STEPS=1000

set -euo pipefail

# ─── Relança sob caffeinate ───────────────────────────────────────────────────
if [[ -z "${CAFFEINATED:-}" ]]; then
    echo "[lipnet-macos] Iniciando com caffeinate -si (impede hibernação)..."
    echo "[lipnet-macos] Mantenha o carregador conectado para fechar a tampa."
    exec caffeinate -si env CAFFEINATED=1 bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

upper() { echo "$1" | tr '[:lower:]' '[:upper:]'; }

command -v uv >/dev/null 2>&1 || die "uv não encontrado. Instale: brew install uv"

# ─── Log ──────────────────────────────────────────────────────────────────────
LOG_DIR="${PROJECT_ROOT}/logs/macos"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/transformer_v2_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "${LOG_FILE}") 2>&1
log "Log: ${LOG_FILE}"
log "Para acompanhar: tail -f ${LOG_FILE}"

# ─── Setup venv ───────────────────────────────────────────────────────────────
log "=== Setup: verificando dependências ==="
cd "${PROJECT_ROOT}"
setup_venv

# ─── Verifica pré-requisitos ─────────────────────────────────────────────────
GRU_CKPT="${CKPT_ROOT}/gru/gru_best.pt"
GRU_HIST="${CKPT_ROOT}/gru/history.json"
V1_CKPT="${CKPT_ROOT}/transformer/transformer_best.pt"
V1_HIST="${CKPT_ROOT}/transformer/history.json"

[[ -f "${GRU_CKPT}" ]]  || die "GRU checkpoint não encontrado: ${GRU_CKPT}"
[[ -f "${GRU_HIST}" ]]  || die "GRU history não encontrado: ${GRU_HIST}"
[[ -f "${V1_CKPT}" ]]   || die "Transformer v1 checkpoint não encontrado: ${V1_CKPT}"
[[ -f "${V1_HIST}" ]]   || die "Transformer v1 history não encontrado: ${V1_HIST}"

make_data_lists

WARMUP_STEPS="${WARMUP_STEPS:-1000}"
V2_DIR="${CKPT_ROOT}/transformer_v2"
mkdir -p "${V2_DIR}" "${RESULTS_DIR}"

log "=== Treinando Transformer v2 (beam search + warmup ${WARMUP_STEPS} steps) ==="
log "Device: ${DEVICE}  |  Epochs: ${MAX_EPOCH}  |  Batch: ${BATCH_SIZE}  |  LR: ${LR}"
log "Save dir: ${V2_DIR}"

py train_transformer.py \
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
    --device          "${DEVICE}" \
    --d_model         512 \
    --nhead           8 \
    --num_layers      2 \
    --dim_feedforward 2048 \
    --attn_dropout    0.1

log "=== Transformer v2 concluído ==="

# ─── Comparação final dos 3 modelos ──────────────────────────────────────────
V2_CKPT="${V2_DIR}/transformer_best.pt"
V2_HIST="${V2_DIR}/history.json"

if [[ -f "${V2_CKPT}" ]]; then
    log "=== Gerando comparação final (GRU | Transformer v1 | Transformer v2) ==="
    py compare_final.py \
        --gru_ckpt  "${GRU_CKPT}" \
        --gru_hist  "${GRU_HIST}" \
        --v1_ckpt   "${V1_CKPT}" \
        --v1_hist   "${V1_HIST}" \
        --v2_ckpt   "${V2_CKPT}" \
        --v2_hist   "${V2_HIST}" \
        --video_path  "${LIP_DIR}" \
        --anno_path   "${ALIGN_DIR}" \
        --val_list    "${VAL_LIST}" \
        --vid_padding "${VID_PADDING}" \
        --txt_padding "${TXT_PADDING}" \
        --num_classes "${NUM_CLASSES}" \
        --batch_size  "${BATCH_SIZE}" \
        --num_workers 2 \
        --d_model         512 \
        --nhead           8 \
        --num_layers      2 \
        --dim_feedforward 2048 \
        --out_dir   "${RESULTS_DIR}" \
        --device    "${DEVICE}"
    log "Comparação concluída. Resultados em: ${RESULTS_DIR}/"
else
    log "AVISO: ${V2_CKPT} não encontrado — comparação ignorada."
fi

log "=== Pipeline completo! ==="
log "Checkpoints v2 : ${V2_DIR}/"
log "Resultados     : ${RESULTS_DIR}/"
log "Log            : ${LOG_FILE}"
