#!/bin/bash
# =============================================================================
# SLURM job script — LipNet GRU vs Transformer comparison experiment
#
# Submit:  sbatch run_experiment.sh
# Monitor: squeue -u $USER  |  tail -f logs/slurm_<JOB_ID>_stdout.log
# =============================================================================

#SBATCH --job-name=lipnet-gru-vs-transformer
#SBATCH --partition=h100n3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/raid/%u/logs/slurm_%j_stdout.log
#SBATCH --error=/raid/%u/logs/slurm_%j_stderr.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com   # <-- update this

# =============================================================================
# Apptainer image — PyTorch 2.8.0 + CUDA 12.8 + cuDNN 9
# =============================================================================

# Path to the SIF pulled with:
#   apptainer pull pytorch_2.8.0-cuda12.8-cudnn9-devel.sif \
#       docker://pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
SIF="/raid/$USER/images/pytorch_2.8.0-cuda12.8-cudnn9-devel.sif"

# Convenience wrapper: runs python inside the container with GPU + dataset access.
# $SLURM_SUBMIT_DIR is bind-mounted so the project files are visible inside.
PY="apptainer exec \
    --nv \
    --bind /data/grid:/data/grid \
    --bind ${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR} \
    ${SIF} python"

# =============================================================================
# Environment setup
# =============================================================================

echo "========================================"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "GPUs        : $CUDA_VISIBLE_DEVICES"
echo "Start time  : $(date)"
echo "Working dir : $SLURM_SUBMIT_DIR"
echo "SIF         : $SIF"
echo "========================================"

mkdir -p logs checkpoints/gru checkpoints/transformer results

cd "$SLURM_SUBMIT_DIR" || exit 1

# --- Install extra Python packages not present in the base image ------------
# pip --user installs into ~/.local (outside the read-only SIF) and persists
# across jobs — this step is fast after the first run (~30 s).
echo "Installing extra dependencies inside container..."
apptainer exec --nv "$SIF" pip install --user --quiet \
    jiwer \
    editdistance \
    matplotlib \
    tensorboardX \
    tqdm \
    opencv-python-headless
echo "Dependencies ready."

# --- Sanity-check GPU visibility -------------------------------------------
$PY -c "import torch; \
        print('PyTorch :', torch.__version__); \
        print('CUDA    :', torch.cuda.is_available()); \
        print('GPU     :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# =============================================================================
# Shared hyperparameters — identical for both models
# =============================================================================

DATASET_ROOT="/data/grid"
VIDEO_PATH="${DATASET_ROOT}/lip"
ANNO_PATH="${DATASET_ROOT}/GRID_align_txt"
TRAIN_LIST="data/overlap_train.txt"
VAL_LIST="data/overlap_val.txt"

BATCH_SIZE=8
LR=1e-4
MAX_EPOCH=50
SEED=42
DROPOUT=0.5
NUM_WORKERS=8
VID_PADDING=75
TXT_PADDING=200
NUM_CLASSES=28

# =============================================================================
# Step 1: Train GRU model
# =============================================================================

echo ""
echo "========================================"
echo "STEP 1/3 — Training LipNet-GRU"
echo "========================================"

$PY train_transformer.py \
    --model         gru \
    --video_path    "$VIDEO_PATH" \
    --anno_path     "$ANNO_PATH" \
    --train_list    "$TRAIN_LIST" \
    --val_list      "$VAL_LIST" \
    --batch_size    $BATCH_SIZE \
    --lr            $LR \
    --max_epoch     $MAX_EPOCH \
    --seed          $SEED \
    --dropout       $DROPOUT \
    --num_workers   $NUM_WORKERS \
    --vid_padding   $VID_PADDING \
    --txt_padding   $TXT_PADDING \
    --num_classes   $NUM_CLASSES \
    --save_dir      checkpoints/gru \
    --display       100 \
    --gpu           0

GRU_EXIT=$?
if [ $GRU_EXIT -ne 0 ]; then
    echo "ERROR: GRU training failed with exit code $GRU_EXIT" >&2
    exit $GRU_EXIT
fi
echo "GRU training completed at: $(date)"

# =============================================================================
# Step 2: Train Transformer model
# =============================================================================

echo ""
echo "========================================"
echo "STEP 2/3 — Training LipNet-Transformer"
echo "========================================"

$PY train_transformer.py \
    --model           transformer \
    --video_path      "$VIDEO_PATH" \
    --anno_path       "$ANNO_PATH" \
    --train_list      "$TRAIN_LIST" \
    --val_list        "$VAL_LIST" \
    --batch_size      $BATCH_SIZE \
    --lr              $LR \
    --max_epoch       $MAX_EPOCH \
    --seed            $SEED \
    --dropout         $DROPOUT \
    --num_workers     $NUM_WORKERS \
    --vid_padding     $VID_PADDING \
    --txt_padding     $TXT_PADDING \
    --num_classes     $NUM_CLASSES \
    --d_model         512 \
    --nhead           8 \
    --num_layers      2 \
    --dim_feedforward 2048 \
    --attn_dropout    0.1 \
    --save_dir        checkpoints/transformer \
    --display         100 \
    --gpu             0

TRANS_EXIT=$?
if [ $TRANS_EXIT -ne 0 ]; then
    echo "ERROR: Transformer training failed with exit code $TRANS_EXIT" >&2
    exit $TRANS_EXIT
fi
echo "Transformer training completed at: $(date)"

# =============================================================================
# Step 3: Compare models
# =============================================================================

echo ""
echo "========================================"
echo "STEP 3/3 — Comparing models"
echo "========================================"

$PY compare_models.py \
    --gru_ckpt   checkpoints/gru/gru_best.pt \
    --gru_hist   checkpoints/gru/history.json \
    --trans_ckpt checkpoints/transformer/transformer_best.pt \
    --trans_hist checkpoints/transformer/history.json \
    --video_path "$VIDEO_PATH" \
    --anno_path  "$ANNO_PATH" \
    --val_list   "$VAL_LIST" \
    --vid_padding  $VID_PADDING \
    --txt_padding  $TXT_PADDING \
    --num_classes  $NUM_CLASSES \
    --batch_size   $BATCH_SIZE \
    --num_workers  4 \
    --d_model         512 \
    --nhead           8 \
    --num_layers      2 \
    --dim_feedforward 2048 \
    --out_dir    results \
    --gpu        0

echo ""
echo "========================================"
echo "Experiment complete at: $(date)"
echo "Results saved to: results/"
echo "========================================"
