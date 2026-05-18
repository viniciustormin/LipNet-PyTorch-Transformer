#!/bin/bash
# =============================================================================
# SLURM job — GRID Corpus preprocessing
# Extracts lip crops from raw MPG videos using face_alignment + ffmpeg.
#
# Submit AFTER download_grid.sh has finished:
#   sbatch scripts/run_preprocessing.sh
# =============================================================================

#SBATCH --job-name=grid-preprocess
#SBATCH --partition=h100n3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/raid/%u/logs/preprocess_%j_stdout.log
#SBATCH --error=/raid/%u/logs/preprocess_%j_stderr.log

SIF="/raid/$USER/images/pytorch_2.8.0-cuda12.8-cudnn9-devel.sif"

GRID_ROOT="/data/grid"
RAW_VIDEO_DIR="${GRID_ROOT}/raw_videos"
LIP_DIR="${GRID_ROOT}/lip"

mkdir -p "$LIP_DIR" logs

cd "$SLURM_SUBMIT_DIR" || exit 1

echo "=== Preprocessing started at $(date) ==="
echo "SIF : $SIF"
echo "RAW : $RAW_VIDEO_DIR"
echo "OUT : $LIP_DIR"

# Install face-alignment inside the container (persists in ~/.local)
echo "Installing face-alignment..."
apptainer exec --nv "$SIF" pip install --user --quiet face-alignment

# Run preprocessing — 1 worker with GPU (face_alignment uses CUDA)
apptainer exec \
    --nv \
    --bind "${GRID_ROOT}:${GRID_ROOT}" \
    --bind "${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}" \
    "$SIF" \
    python scripts/preprocess_lips.py \
        --raw_dir   "$RAW_VIDEO_DIR" \
        --out_dir   "$LIP_DIR" \
        --n_workers 1 \
        --device    cuda

echo ""
echo "=== Preprocessing complete at $(date) ==="
echo ""
echo "Verify output:"
echo "  ls ${LIP_DIR}/s1/video/mpg_6000/ | head -5"
echo ""
echo "Next step — run the training experiment:"
echo "  sbatch run_experiment.sh"
