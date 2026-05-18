#!/bin/bash
# Configuração compartilhada para macOS (M4 Pro, MPS/CPU).
# Não depende de Apptainer nem SLURM.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

# ─── Dados ───────────────────────────────────────────────────────────────────
GRID_ROOT="${GRID_ROOT:-${PROJECT_ROOT}/data/grid}"
RAW_VIDEO_DIR="${RAW_VIDEO_DIR:-${GRID_ROOT}/raw_videos}"
LIP_DIR="${LIP_DIR:-${GRID_ROOT}/lip}"
ALIGN_DIR="${ALIGN_DIR:-${GRID_ROOT}/GRID_align_txt}"

# ─── Escopo do experimento ────────────────────────────────────────────────────
# Exemplos:
#   SPEAKERS="1"           → só speaker 1 (rápido, recomendado para testar)
#   SPEAKERS="1 2 3"       → speakers 1, 2 e 3
#   SPEAKERS=all           → todos os 33 speakers
SPEAKERS="${SPEAKERS:-1}"

# Modelos a treinar:
#   "gru transformer"  → treina os dois e compara (padrão)
#   "gru"              → só GRU
#   "transformer"      → só Transformer
TRAIN_MODELS="${TRAIN_MODELS:-gru transformer}"

# ─── Hiperparâmetros ─────────────────────────────────────────────────────────
BATCH_SIZE="${BATCH_SIZE:-4}"       # menor que no cluster; aumente se tiver RAM
LR="${LR:-1e-4}"
MAX_EPOCH="${MAX_EPOCH:-50}"
SEED="${SEED:-42}"
DROPOUT="${DROPOUT:-0.5}"
NUM_WORKERS="${NUM_WORKERS:-2}"     # 0 se DataLoader travar no macOS
VID_PADDING="${VID_PADDING:-75}"
TXT_PADDING="${TXT_PADDING:-200}"
NUM_CLASSES="${NUM_CLASSES:-28}"

# ─── Device ──────────────────────────────────────────────────────────────────
# "auto" → detecta automaticamente: cuda → mps → cpu
# M4 Pro usa "mps" (Metal Performance Shaders)
DEVICE="${DEVICE:-auto}"

# aten::_ctc_loss não está implementado no MPS — habilita fallback para CPU
# só nas ops sem suporte (o restante continua no MPS).
export PYTORCH_ENABLE_MPS_FALLBACK=1

# ─── Paths derivados ─────────────────────────────────────────────────────────
_speakers_tag() {
    if [[ "${SPEAKERS}" == "all" ]]; then echo "all"; return; fi
    local tag=""
    for n in ${SPEAKERS}; do tag="${tag:+${tag}-}s${n}"; done
    echo "$tag"
}

RUN_TAG="${RUN_TAG:-$(_speakers_tag)}"
CKPT_ROOT="${CKPT_ROOT:-${PROJECT_ROOT}/checkpoints/${RUN_TAG}}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results/${RUN_TAG}}"

ALL_SPEAKERS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32 33 34"

effective_speakers() {
    if [[ "${SPEAKERS}" == "all" ]]; then echo "${ALL_SPEAKERS}"; else echo "${SPEAKERS}"; fi
}

# ─── Utilitários ─────────────────────────────────────────────────────────────
log() { echo "[lipnet-macos $(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die() { echo "[lipnet-macos] ERRO: $*" >&2; exit 1; }

# Gera data/active_train.txt e data/active_val.txt filtrados por SPEAKERS
make_data_lists() {
    local src_train="${PROJECT_ROOT}/data/overlap_train.txt"
    local src_val="${PROJECT_ROOT}/data/overlap_val.txt"

    if [[ "${SPEAKERS}" == "all" ]]; then
        TRAIN_LIST="${src_train}"
        VAL_LIST="${src_val}"
        log "Data lists: completos (todos os speakers)"
        return
    fi

    local pattern=""
    for n in ${SPEAKERS}; do
        pattern="${pattern}${pattern:+|}^s${n}/"
    done

    TRAIN_LIST="${PROJECT_ROOT}/data/active_train.txt"
    VAL_LIST="${PROJECT_ROOT}/data/active_val.txt"

    grep -E "${pattern}" "${src_train}" > "${TRAIN_LIST}"
    grep -E "${pattern}" "${src_val}"   > "${VAL_LIST}"

    local n_train n_val
    n_train=$(wc -l < "${TRAIN_LIST}" | tr -d ' ')
    n_val=$(wc -l < "${VAL_LIST}" | tr -d ' ')
    log "Data lists: ${n_train} treino / ${n_val} validação (speakers: ${SPEAKERS})"
}

# ─── Python via venv uv ───────────────────────────────────────────────────────
VENV="${PROJECT_ROOT}/.venv"

setup_venv() {
    if [[ ! -f "${VENV}/bin/python" ]]; then
        log "Criando venv com uv..."
        uv venv "${VENV}"
    fi
    log "Instalando/verificando dependências..."
    uv pip install --python "${VENV}/bin/python" -q -r "${PROJECT_ROOT}/requirements.txt"
    log "Dependências prontas."
}

py() { "${VENV}/bin/python" "$@"; }
