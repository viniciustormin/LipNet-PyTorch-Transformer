#!/bin/bash
#SBATCH --job-name=grid-preprocess
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

[[ -f "${COMMON_SH}" ]] || { echo "ERRO: rode 'sbatch scripts/slurm/run_preprocessing.sh' a partir da raiz do repositório." >&2; exit 1; }
export PROJECT_ROOT="$(cd "$(dirname "${COMMON_SH}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${COMMON_SH}"

detect_container_runtime
build_container_args

log "=== Preprocessing GRID iniciado ==="
print_job_context

mkdir -p "${LIP_DIR}"

log "Instalando face-alignment..."
"${CONTAINER_RUNTIME}" exec --nv "${IMAGE_PATH}" \
    pip install --user --quiet face-alignment

log "Extraindo frames e recortando lábios..."
container_exec python scripts/preprocess_lips.py \
    --raw_dir   "${RAW_VIDEO_DIR}" \
    --out_dir   "${LIP_DIR}" \
    --n_workers 1 \
    --device    cuda

log "=== Preprocessing concluído ==="
log "Próximo passo: sbatch scripts/slurm/run_experiment.sh"
