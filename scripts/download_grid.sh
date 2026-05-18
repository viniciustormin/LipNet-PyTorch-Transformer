#!/bin/bash
# =============================================================================
# Download GRID Corpus raw videos + alignment files
#
# Official site: http://spandh.dcs.shef.ac.uk/gridcorpus/
# Total size: ~33 GB (videos) + ~20 MB (alignments)
#
# Usage (run on the cluster, NOT via SLURM — just an interactive download):
#   bash scripts/download_grid.sh
#
# Or submit as a SLURM job with enough time (allow ~3-6h depending on bandwidth):
#   sbatch --job-name=grid-download --ntasks=1 --cpus-per-task=4 \
#          --mem=8G --time=06:00:00 \
#          --output=logs/download_%j.log \
#          scripts/download_grid.sh
# =============================================================================

set -euo pipefail

GRID_ROOT="/data/grid"
RAW_VIDEO_DIR="${GRID_ROOT}/raw_videos"
ALIGN_DIR="${GRID_ROOT}/GRID_align_txt"

mkdir -p "$RAW_VIDEO_DIR" "$ALIGN_DIR" logs

# Speakers s1–s34; s21 does not exist in GRID
SPEAKERS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32 33 34)

BASE_URL="http://spandh.dcs.shef.ac.uk/gridcorpus"

echo "=== GRID Corpus download started at $(date) ==="
echo "Destination: $GRID_ROOT"

# ---------------------------------------------------------------------------
# 1. Download alignment files (small — ~20 MB total)
# ---------------------------------------------------------------------------
echo ""
echo "--- Downloading alignment files ---"
for n in "${SPEAKERS[@]}"; do
    spk="s${n}"
    out_tar="${ALIGN_DIR}/${spk}_alignments.tar"

    if [ -d "${ALIGN_DIR}/${spk}" ]; then
        echo "  [SKIP] ${spk} alignments already extracted"
        continue
    fi

    echo "  Downloading alignments: ${spk}"
    wget -q --show-progress --continue \
        "${BASE_URL}/${spk}/align/${spk}_alignments.tar" \
        -O "$out_tar"

    tar -xf "$out_tar" -C "$ALIGN_DIR/"
    rm -f "$out_tar"
    echo "  [OK] ${spk} alignments extracted to ${ALIGN_DIR}/${spk}/"
done

# ---------------------------------------------------------------------------
# 2. Download videos (heavy — ~1 GB per speaker, ~33 GB total)
# ---------------------------------------------------------------------------
echo ""
echo "--- Downloading videos ---"
for n in "${SPEAKERS[@]}"; do
    spk="s${n}"
    out_zip="${RAW_VIDEO_DIR}/${spk}_video.zip"
    out_dir="${RAW_VIDEO_DIR}/${spk}"

    if [ -d "$out_dir" ]; then
        echo "  [SKIP] ${spk} videos already extracted"
        continue
    fi

    echo "  Downloading videos: ${spk} (~1 GB)"
    wget -q --show-progress --continue \
        "${BASE_URL}/${spk}/video/${spk}_v1_mpg_6000.zip" \
        -O "$out_zip"

    echo "  Extracting ${spk}..."
    unzip -q "$out_zip" -d "${RAW_VIDEO_DIR}/"
    rm -f "$out_zip"
    echo "  [OK] ${spk} videos extracted to ${out_dir}/"
done

echo ""
echo "=== Download complete at $(date) ==="
echo ""
echo "Next step — extract frames and crop lips:"
echo "  sbatch scripts/run_preprocessing.sh"
