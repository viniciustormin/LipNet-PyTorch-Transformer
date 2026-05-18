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

GRID_ROOT="${GRID_ROOT:-/data/grid}"
RAW_VIDEO_DIR="${RAW_VIDEO_DIR:-${GRID_ROOT}/raw_videos}"
LIP_DIR="${LIP_DIR:-${GRID_ROOT}/lip}"
ALIGN_DIR="${ALIGN_DIR:-${GRID_ROOT}/GRID_align_txt}"

BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-4}"
MAX_EPOCH="${MAX_EPOCH:-50}"
SEED="${SEED:-42}"
DROPOUT="${DROPOUT:-0.5}"
NUM_WORKERS="${NUM_WORKERS:-8}"
VID_PADDING="${VID_PADDING:-75}"
TXT_PADDING="${TXT_PADDING:-200}"
NUM_CLASSES="${NUM_CLASSES:-28}"

CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-}"
declare -a CONTAINER_ARGS=()

log() { echo "[lipnet $(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die() { echo "[lipnet] ERRO: $*" >&2; exit 1; }

detect_container_runtime() {
    if [[ -n "${CONTAINER_RUNTIME}" ]]; then
        command -v "${CONTAINER_RUNTIME}" >/dev/null 2>&1 || die "Runtime não encontrado: ${CONTAINER_RUNTIME}"
        return
    fi
    if command -v apptainer >/dev/null 2>&1; then CONTAINER_RUNTIME="apptainer"; return; fi
    if command -v singularity >/dev/null 2>&1; then CONTAINER_RUNTIME="singularity"; return; fi
    die "apptainer/singularity não encontrado."
}

build_container_args() {
    CONTAINER_ARGS=(
        "${CONTAINER_RUNTIME}" exec --nv
        --bind "${WORKSPACE_ROOT}:${WORKSPACE_ROOT}"
        --bind "${GRID_ROOT}:${GRID_ROOT}"
        --pwd "${PROJECT_ROOT}"
        "${IMAGE_PATH}"
    )
}

container_exec() { "${CONTAINER_ARGS[@]}" "$@"; }

setup_extra_deps() {
    log "Instalando dependências extras..."
    "${CONTAINER_RUNTIME}" exec --nv "${IMAGE_PATH}" \
        pip install --user --quiet \
            jiwer editdistance matplotlib tensorboardX tqdm opencv-python-headless
    log "Dependências prontas."
}

print_job_context() {
    log "PROJECT_ROOT   : ${PROJECT_ROOT}"
    log "WORKSPACE_ROOT : ${WORKSPACE_ROOT}"
    log "IMAGE_PATH     : ${IMAGE_PATH}"
    log "GLOBAL_LOG_ROOT: ${GLOBAL_LOG_ROOT}"
    log "GRID_ROOT      : ${GRID_ROOT}"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
}
