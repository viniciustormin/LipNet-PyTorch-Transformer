#!/bin/bash

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${THIS_DIR}/../.." && pwd)}"
PROJECT_NAME="${PROJECT_NAME:-$(basename "${PROJECT_ROOT}")}"
PROJECTS_ROOT="${PROJECTS_ROOT:-$(cd "${PROJECT_ROOT}/.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${PROJECTS_ROOT}/.." && pwd)}"

IMAGES_ROOT="${IMAGES_ROOT:-${WORKSPACE_ROOT}/images}"
GLOBAL_LOG_ROOT="${GLOBAL_LOG_ROOT:-${WORKSPACE_ROOT}/logs}"
IMAGE_PATH="${IMAGE_PATH:-${IMAGES_ROOT}/pytorch_2.8.0-cuda12.8-cudnn9-devel.sif}"

GRID_ROOT="${GRID_ROOT:-${PROJECT_ROOT}/data/grid}"
RAW_VIDEO_DIR="${RAW_VIDEO_DIR:-${GRID_ROOT}/raw_videos}"
LIP_DIR="${LIP_DIR:-${GRID_ROOT}/lip}"
ALIGN_DIR="${ALIGN_DIR:-${GRID_ROOT}/GRID_align_txt}"

# =============================================================================
# Configuração do experimento — altere aqui para mudar o escopo
# =============================================================================

# Speakers a usar. Exemplos:
#   "1"               → só o speaker 1  (experimento rápido)
#   "1 2 3"           → speakers 1, 2 e 3
#   "all"             → todos os 33 speakers do GRID
SPEAKERS="${SPEAKERS:-1}"

# Modelos a treinar. Exemplos:
#   "gru transformer" → treina os dois e compara  (padrão)
#   "transformer"     → só o Transformer          (dataset completo)
#   "gru"             → só o GRU
TRAIN_MODELS="${TRAIN_MODELS:-gru transformer}"

# =============================================================================
# Hiperparâmetros de treino
# =============================================================================
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-4}"
MAX_EPOCH="${MAX_EPOCH:-50}"
SEED="${SEED:-42}"
DROPOUT="${DROPOUT:-0.5}"
NUM_WORKERS="${NUM_WORKERS:-8}"
VID_PADDING="${VID_PADDING:-75}"
TXT_PADDING="${TXT_PADDING:-200}"
NUM_CLASSES="${NUM_CLASSES:-28}"

# =============================================================================
# Paths dinâmicos (derivados de SPEAKERS)
# =============================================================================

# Tag usada em checkpoints/logs: "s1", "s1-s2-s3", "all"
_speakers_tag() {
    if [[ "${SPEAKERS}" == "all" ]]; then
        echo "all"
    else
        local tag=""
        for n in ${SPEAKERS}; do tag="${tag:+${tag}-}s${n}"; done
        echo "$tag"
    fi
}

RUN_TAG="${RUN_TAG:-$(_speakers_tag)}"
CKPT_ROOT="${CKPT_ROOT:-${PROJECT_ROOT}/checkpoints/${RUN_TAG}}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results/${RUN_TAG}}"

# Lista de todos os speakers válidos do GRID (s21 não tem vídeo)
ALL_SPEAKERS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32 33 34"

# Speakers efetivos como lista numérica
effective_speakers() {
    if [[ "${SPEAKERS}" == "all" ]]; then
        echo "${ALL_SPEAKERS}"
    else
        echo "${SPEAKERS}"
    fi
}

# =============================================================================
# Container
# =============================================================================
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-}"
declare -a CONTAINER_ARGS=()

log()  { echo "[lipnet $(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die()  { echo "[lipnet] ERRO: $*" >&2; exit 1; }

detect_container_runtime() {
    if [[ -n "${CONTAINER_RUNTIME}" ]]; then
        command -v "${CONTAINER_RUNTIME}" >/dev/null 2>&1 || die "Runtime não encontrado: ${CONTAINER_RUNTIME}"
        return
    fi
    if command -v apptainer  >/dev/null 2>&1; then CONTAINER_RUNTIME="apptainer";  return; fi
    if command -v singularity >/dev/null 2>&1; then CONTAINER_RUNTIME="singularity"; return; fi
    die "apptainer/singularity não encontrado."
}

build_container_args() {
    CONTAINER_ARGS=(
        "${CONTAINER_RUNTIME}" exec --nv
        --bind "${WORKSPACE_ROOT}:${WORKSPACE_ROOT}"
        --pwd  "${PROJECT_ROOT}"
        "${IMAGE_PATH}"
    )
}

container_exec() { "${CONTAINER_ARGS[@]}" "$@"; }

setup_extra_deps() {
    log "Instalando dependências extras..."
    "${CONTAINER_RUNTIME}" exec --nv "${IMAGE_PATH}" \
        pip install --user --quiet \
            jiwer editdistance matplotlib tensorboardX tqdm opencv-python-headless
}

# =============================================================================
# Data lists — gerados dinamicamente a partir de SPEAKERS
# =============================================================================

# Gera data/active_train.txt e data/active_val.txt filtrados por SPEAKERS.
# Se SPEAKERS="all", aponta direto para os lists completos (sem copiar).
make_data_lists() {
    local src_train="${PROJECT_ROOT}/data/overlap_train.txt"
    local src_val="${PROJECT_ROOT}/data/overlap_val.txt"

    if [[ "${SPEAKERS}" == "all" ]]; then
        TRAIN_LIST="${src_train}"
        VAL_LIST="${src_val}"
        log "Data lists: completos (todos os speakers)"
        return
    fi

    # Monta pattern grep: ^s1/|^s2/|...
    local pattern=""
    for n in ${SPEAKERS}; do
        pattern="${pattern}${pattern:+|}^s${n}/"
    done

    TRAIN_LIST="${PROJECT_ROOT}/data/active_train.txt"
    VAL_LIST="${PROJECT_ROOT}/data/active_val.txt"

    grep -E "${pattern}" "${src_train}" > "${TRAIN_LIST}"
    grep -E "${pattern}" "${src_val}"   > "${VAL_LIST}"

    local n_train n_val
    n_train=$(wc -l < "${TRAIN_LIST}")
    n_val=$(wc -l < "${VAL_LIST}")
    log "Data lists: ${n_train} treino / ${n_val} validação (speakers: ${SPEAKERS})"
}

# =============================================================================
# Diagnóstico
# =============================================================================
print_job_context() {
    log "PROJECT_ROOT  : ${PROJECT_ROOT}"
    log "WORKSPACE_ROOT: ${WORKSPACE_ROOT}"
    log "IMAGE_PATH    : ${IMAGE_PATH}"
    log "GRID_ROOT     : ${GRID_ROOT}"
    log "SPEAKERS      : ${SPEAKERS}  →  tag=${RUN_TAG}"
    log "TRAIN_MODELS  : ${TRAIN_MODELS}"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
}
