#!/bin/bash
#SBATCH --job-name=grid-download
#SBATCH --partition=h100n3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=UNLIMITED
#SBATCH --output=../../logs/%x_%j.log
#SBATCH --error=../../logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
COMMON_SH="${PROJECT_ROOT}/scripts/slurm/common.sh"

[[ -f "${COMMON_SH}" ]] || { echo "ERRO: rode 'sbatch scripts/slurm/download_grid.sh' a partir da raiz do repositório." >&2; exit 1; }
export PROJECT_ROOT="$(cd "$(dirname "${COMMON_SH}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${COMMON_SH}"

log "=== Download GRID Corpus iniciado ==="
log "Node: ${SLURMD_NODENAME:-local}"
print_job_context

mkdir -p "${RAW_VIDEO_DIR}" "${ALIGN_DIR}"

BASE_URL="http://spandh.dcs.shef.ac.uk/gridcorpus"
SPEAKERS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32 33 34)

# --- Alignment files (~20 MB total) ---
log "Baixando arquivos de alinhamento..."
for n in "${SPEAKERS[@]}"; do
    spk="s${n}"
    if [[ -d "${ALIGN_DIR}/${spk}" ]]; then
        log "  [SKIP] ${spk} já extraído"
        continue
    fi
    log "  ${spk} alignments..."
    wget -q --show-progress --continue \
        "${BASE_URL}/${spk}/align/${spk}_alignments.tar" \
        -O "${ALIGN_DIR}/${spk}_alignments.tar"
    tar -xf "${ALIGN_DIR}/${spk}_alignments.tar" -C "${ALIGN_DIR}/"
    rm -f "${ALIGN_DIR}/${spk}_alignments.tar"
done

# --- Videos (~1 GB por speaker, ~33 GB total) ---
log "Baixando vídeos..."
for n in "${SPEAKERS[@]}"; do
    spk="s${n}"
    if [[ -d "${RAW_VIDEO_DIR}/${spk}" ]]; then
        log "  [SKIP] ${spk} já extraído"
        continue
    fi
    log "  ${spk} videos (~1 GB)..."
    wget -q --show-progress --continue \
        "${BASE_URL}/${spk}/video/${spk}_v1_mpg_6000.zip" \
        -O "${RAW_VIDEO_DIR}/${spk}_video.zip"
    unzip -q "${RAW_VIDEO_DIR}/${spk}_video.zip" -d "${RAW_VIDEO_DIR}/"
    rm -f "${RAW_VIDEO_DIR}/${spk}_video.zip"
done

log "=== Download concluído ==="
log "Próximo passo: sbatch scripts/slurm/run_preprocessing.sh"
