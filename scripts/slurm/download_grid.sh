#!/bin/bash
# Download do GRID Corpus — rode direto no terminal, sem SLURM:
#   bash scripts/slurm/download_grid.sh
#
# Fonte: https://spandh.dcs.shef.ac.uk/gridcorpus/#downloads
# Tamanho total: ~6 MB (alinhamentos) + ~82 GB (vídeos 6000kbps, 2 partes por speaker)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/common.sh"

log "=== Download GRID Corpus iniciado ==="
log "Node: ${SLURMD_NODENAME:-local}"
print_job_context

mkdir -p "${RAW_VIDEO_DIR}" "${ALIGN_DIR}"

BASE_URL="https://spandh.dcs.shef.ac.uk/gridcorpus"
# s21 não tem vídeo no GRID
SPEAKERS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32 33 34)

_wget() {
    wget --no-check-certificate -L --show-progress --continue "$@"
}

_check() {
    local f="$1" min="${2:-1024}"
    local sz; sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
    if [[ $sz -lt $min ]]; then
        log "  ERRO: '$f' vazio ou inválido (${sz} bytes) — verifique conexão"
        rm -f "$f"
        return 1
    fi
}

# ---------------------------------------------------------------------------
# 1. Word alignments  — s{n}/align/s{n}.tar  (~190 KB cada)
# ---------------------------------------------------------------------------
log "Baixando word alignments..."
for n in "${SPEAKERS[@]}"; do
    spk="s${n}"
    out_dir="${ALIGN_DIR}/${spk}/align"

    if [[ -d "$out_dir" ]] && [[ $(ls "$out_dir"/*.align 2>/dev/null | wc -l) -gt 0 ]]; then
        log "  [SKIP] ${spk} alignments já existem"
        continue
    fi

    log "  ${spk} alignments (~190 KB)..."
    tmp="${ALIGN_DIR}/${spk}.tar"
    _wget "${BASE_URL}/${spk}/align/${spk}.tar" -O "$tmp"
    _check "$tmp" 1024

    mkdir -p "${ALIGN_DIR}/${spk}/align"
    tar -xf "$tmp" -C "${ALIGN_DIR}/${spk}/align/" --strip-components=1 2>/dev/null \
        || tar -xf "$tmp" -C "${ALIGN_DIR}/${spk}/align/"
    rm -f "$tmp"
    log "  [OK] ${spk}"
done

# ---------------------------------------------------------------------------
# 2. Videos (high quality 6000kbps) — duas partes por speaker (~1.2 GB cada)
#    part1 + part2 são partes de um único tar — concatenar antes de extrair
# ---------------------------------------------------------------------------
log "Baixando vídeos (6000kbps, ~2.4 GB por speaker, ~82 GB total)..."
for n in "${SPEAKERS[@]}"; do
    spk="s${n}"
    out_dir="${RAW_VIDEO_DIR}/${spk}"

    if [[ -d "$out_dir" ]] && [[ $(find "$out_dir" -name "*.mpg" | wc -l) -gt 0 ]]; then
        log "  [SKIP] ${spk} vídeos já existem"
        continue
    fi

    mkdir -p "$out_dir"

    log "  ${spk} part1 (~1.2 GB)..."
    p1="${out_dir}/${spk}.mpg_6000.part1.tar"
    _wget "${BASE_URL}/${spk}/video/${spk}.mpg_6000.part1.tar" -O "$p1"
    _check "$p1" 104857600   # mínimo 100 MB

    log "  ${spk} part2 (~1.2 GB)..."
    p2="${out_dir}/${spk}.mpg_6000.part2.tar"
    _wget "${BASE_URL}/${spk}/video/${spk}.mpg_6000.part2.tar" -O "$p2"
    _check "$p2" 104857600

    log "  ${spk} extraindo..."
    # partes são um tar dividido — concatenar e extrair direto para o diretório do speaker
    cat "$p1" "$p2" | tar -xf - -C "${RAW_VIDEO_DIR}/"
    rm -f "$p1" "$p2"
    log "  [OK] ${spk}"
done

log "=== Download concluído ==="
log "Próximo passo: sbatch scripts/slurm/run_preprocessing.sh"
