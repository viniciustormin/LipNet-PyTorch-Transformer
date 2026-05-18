#!/bin/bash
# Download do GRID Corpus no macOS — rode no terminal:
#   bash scripts/macos/download_grid.sh
#
# Controle via variáveis de ambiente (padrão: só speaker 1):
#   SPEAKERS="1 2 3" bash scripts/macos/download_grid.sh
#   SPEAKERS=all     bash scripts/macos/download_grid.sh
#
# Fonte: https://spandh.dcs.shef.ac.uk/gridcorpus/#downloads
# Tamanho: ~190 KB (alinhamentos) + ~2.4 GB (vídeos) por speaker

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

log "=== Download GRID Corpus iniciado ==="
log "Speakers: ${SPEAKERS}  →  tag=${RUN_TAG}"

mkdir -p "${RAW_VIDEO_DIR}" "${ALIGN_DIR}"

BASE_URL="https://spandh.dcs.shef.ac.uk/gridcorpus"

_wget() {
    # Chamada: _wget URL -O destino
    local url="$1" out="$3"
    if command -v wget >/dev/null 2>&1; then
        wget --no-check-certificate -L --show-progress --continue "$url" -O "$out"
    else
        curl -L --insecure -C - --progress-bar -o "$out" "$url"
    fi
}

_check() {
    local f="$1" min="${2:-1024}"
    # wc -c funciona em macOS e Linux
    local sz
    sz=$(wc -c < "$f" 2>/dev/null | tr -d ' ') || sz=0
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
for n in $(effective_speakers); do
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

    mkdir -p "${ALIGN_DIR}/${spk}"
    tar -xf "$tmp" -C "${ALIGN_DIR}/${spk}/"
    rm -f "$tmp"
    log "  [OK] ${spk}"
done

# ---------------------------------------------------------------------------
# 2. Videos (high quality 6000kbps) — duas partes INDEPENDENTES por speaker
#    (~1.2 GB cada); extrair separadamente (não é split archive)
# ---------------------------------------------------------------------------
log "Baixando vídeos (6000kbps, ~2.4 GB por speaker)..."
for n in $(effective_speakers); do
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

    log "  ${spk} extraindo part1..."
    tar -xf "$p1" -C "${RAW_VIDEO_DIR}/"
    rm -f "$p1"

    log "  ${spk} extraindo part2..."
    tar -xf "$p2" -C "${RAW_VIDEO_DIR}/"
    rm -f "$p2"
    log "  [OK] ${spk}"
done

log "=== Download concluído ==="
log "Próximo passo: bash scripts/macos/run_all.sh"
