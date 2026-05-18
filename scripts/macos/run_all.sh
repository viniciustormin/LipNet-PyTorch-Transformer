#!/bin/bash
# Pipeline completo no macOS: instala deps → baixa dados → preprocess → treina → compara
#
# Uso — rode no terminal e pode fechar a tampa (com carregador conectado):
#   bash scripts/macos/run_all.sh
#
# Variáveis de ambiente opcionais:
#   SPEAKERS="1 2 3"    → processa speakers específicos   (padrão: "1")
#   TRAIN_MODELS="gru"  → treina só GRU                  (padrão: "gru transformer")
#   SKIP_DOWNLOAD=1     → pula download (dados já existem)
#   SKIP_PREPROCESS=1   → pula extração de lips
#   DEVICE=cpu          → força CPU em vez de MPS         (padrão: auto)
#   MAX_EPOCH=10        → treino curto para teste rápido  (padrão: 50)
#
# IMPORTANTE: mantenha o Mac conectado ao carregador.
# caffeinate -s impede a hibernação apenas quando na tomada.

set -euo pipefail

# ─── Relança sob caffeinate se necessário ─────────────────────────────────────
if [[ -z "${CAFFEINATED:-}" ]]; then
    echo "[lipnet-macos] Iniciando com caffeinate -si (impede hibernação)..."
    echo "[lipnet-macos] Mantenha o carregador conectado para fechar a tampa."
    exec caffeinate -si env CAFFEINATED=1 bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# Maiúsculas portável (bash 3.2 do macOS não tem ${var^^})
upper() { echo "$1" | tr '[:lower:]' '[:upper:]'; }

# ─── Verificações iniciais ────────────────────────────────────────────────────
command -v uv     >/dev/null 2>&1 || die "uv não encontrado. Instale: brew install uv"
command -v ffmpeg >/dev/null 2>&1 || die "ffmpeg não encontrado. Instale: brew install ffmpeg"

# ─── Log para arquivo ─────────────────────────────────────────────────────────
LOG_DIR="${PROJECT_ROOT}/logs/macos"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "${LOG_FILE}") 2>&1
log "Log salvo em: ${LOG_FILE}"
log "Para acompanhar em outro terminal: tail -f ${LOG_FILE}"

# ─── Setup do venv ────────────────────────────────────────────────────────────
log "=== Setup: instalando dependências ==="
cd "${PROJECT_ROOT}"
setup_venv

log "Device: ${DEVICE}  |  Speakers: ${SPEAKERS}  |  Models: ${TRAIN_MODELS}"
log "Epochs: ${MAX_EPOCH}  |  Batch: ${BATCH_SIZE}  |  LR: ${LR}"

# ─── Etapa 0: Download do GRID Corpus ────────────────────────────────────────
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

if [[ "${SKIP_DOWNLOAD}" == "1" ]]; then
    log "=== Etapa 0: Download IGNORADO (SKIP_DOWNLOAD=1) ==="
else
    log "=== Etapa 0: Baixando GRID Corpus (~2.4 GB por speaker) ==="
    bash "${SCRIPT_DIR}/download_grid.sh"
    log "=== Etapa 0 concluída ==="
fi

# ─── Etapa 1: Pré-processamento de lips ───────────────────────────────────────
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"

if [[ "${SKIP_PREPROCESS}" == "1" ]]; then
    log "=== Etapa 1: Pré-processamento IGNORADO (SKIP_PREPROCESS=1) ==="
else
    log "=== Etapa 1: Extraindo frames e recortando lips (device=cpu) ==="

    for n in $(effective_speakers); do
        spk="s${n}"
        spk_raw="${RAW_VIDEO_DIR}/${spk}"
        spk_lip="${LIP_DIR}/${spk}"

        if [[ -d "${spk_lip}" ]] && [[ $(find "${spk_lip}" -name "*.jpg" | wc -l | tr -d ' ') -gt 0 ]]; then
            log "  [SKIP] ${spk} já processado"
            continue
        fi

        if [[ ! -d "${spk_raw}" ]]; then
            log "  ERRO: ${spk_raw} não encontrado após download — abortando"
            exit 1
        fi

        log "  Processando ${spk} (pode levar horas na primeira vez)..."
        mkdir -p "${spk_lip}"
        py scripts/preprocess_lips.py \
            --raw_dir   "${spk_raw}" \
            --out_dir   "${spk_lip}" \
            --n_workers 1 \
            --device    cpu
        log "  [OK] ${spk}"
    done

    log "=== Etapa 1 concluída ==="
fi

# ─── Gera listas de dados ─────────────────────────────────────────────────────
make_data_lists
mkdir -p "${CKPT_ROOT}" "${RESULTS_DIR}"

# ─── Etapas de treinamento ────────────────────────────────────────────────────
step=1
total_models=$(echo "${TRAIN_MODELS}" | wc -w | tr -d ' ')

for model in ${TRAIN_MODELS}; do
    model_upper="$(upper "${model}")"
    log "=== Etapa $((step + 1))/$((total_models + 1)) — Treinando LipNet-${model_upper} (device=${DEVICE}) ==="
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
        --device        "${DEVICE}"
    )

    if [[ "${model}" == "transformer" ]]; then
        py train_transformer.py \
            --model           transformer \
            "${common_args[@]}" \
            --d_model         512 \
            --nhead           8 \
            --num_layers      2 \
            --dim_feedforward 2048 \
            --attn_dropout    0.1
    else
        py train_transformer.py \
            --model gru \
            "${common_args[@]}"
    fi

    log "$(upper "${model}") concluído."
    step=$((step + 1))
done

# ─── Etapa final: Comparação ──────────────────────────────────────────────────
gru_ckpt="${CKPT_ROOT}/gru/gru_best.pt"
trans_ckpt="${CKPT_ROOT}/transformer/transformer_best.pt"

if [[ -f "${gru_ckpt}" ]] && [[ -f "${trans_ckpt}" ]]; then
    log "=== Etapa final — Comparando GRU vs Transformer ==="
    py compare_models.py \
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
        --num_workers  2 \
        --d_model         512 \
        --nhead           8 \
        --num_layers      2 \
        --dim_feedforward 2048 \
        --out_dir    "${RESULTS_DIR}" \
        --device     "${DEVICE}"
    log "Comparação concluída. Resultados em: ${RESULTS_DIR}/"
else
    log "Pulando comparação (requer checkpoints de GRU E Transformer)."
    [[ -f "${gru_ckpt}" ]]   || log "  Faltando: ${gru_ckpt}"
    [[ -f "${trans_ckpt}" ]] || log "  Faltando: ${trans_ckpt}"
fi

log "=== Pipeline completo! ==="
log "Checkpoints : ${CKPT_ROOT}/"
log "Resultados  : ${RESULTS_DIR}/"
log "Log         : ${LOG_FILE}"
