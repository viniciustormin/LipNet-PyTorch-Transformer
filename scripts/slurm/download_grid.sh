#!/bin/bash
# Download do GRID Corpus — rode direto no terminal, sem SLURM (não precisa de GPU):
#   bash scripts/slurm/download_grid.sh
#
# O download é só wget; não faz sentido alocar nó GPU para isso.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/common.sh"

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
