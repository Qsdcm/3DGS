#!/bin/bash
# train.sh - 3D Gaussian Splatting MRI Reconstruction Training Script
# Example run (pt1.10): GPU=1 DATA=/data/data54/wanghaobo/data/ksp_full.h5 OUT=./outputs ITERS=5000 bash train.sh
#
# Usage (single GPU):
#   bash train.sh
#
# Usage (multi-GPU):
#   MULTI_GPU=1 bash train.sh

# ============================================================================
# Configuration - Modify these variables
# ============================================================================

# GPU settings
GPU=${GPU:-0}                          # GPU ID for single-GPU training
NGPU=${NGPU:-4}                        # Number of GPUs for multi-GPU training
MULTI_GPU=${MULTI_GPU:-0}              # Set to 1 to enable multi-GPU

# Data paths
DATA=${DATA:-/path/to/data}            # H5 file or directory containing H5 files
OUT=${OUT:-./outputs}                  # Output directory

# Training settings
ITERS=${ITERS:-2000}                   # Maximum optimization iterations
N_GAUSSIANS=${N_GAUSSIANS:-1000}       # Initial number of Gaussians

# Data-related hyperparameters
ACC=${ACC:-4.0}                        # Acceleration factor
MASK_TYPE=${MASK_TYPE:-gaussian}       # Mask type: gaussian or uniform
CENTER_FRAC=${CENTER_FRAC:-0.08}       # Center fraction for ACS

# Optional: Additional arguments
EXTRA_ARGS=${EXTRA_ARGS:-""}           # Any additional arguments

# ============================================================================
# Run Training
# ============================================================================

echo "=============================================="
echo "3DGS MRI Reconstruction - Training"
echo "=============================================="
echo "Data: $DATA"
echo "Output: $OUT"
echo "Iterations: $ITERS"
echo "Initial Gaussians: $N_GAUSSIANS"
echo "Acceleration: ${ACC}x"
echo "Mask Type: $MASK_TYPE"
echo "=============================================="

if [ "$MULTI_GPU" -eq 1 ]; then
    echo "Mode: Multi-GPU ($NGPU GPUs)"
    echo "=============================================="
    
    CUDA_VISIBLE_DEVICES=$GPU torchrun \
        --nproc_per_node=$NGPU \
        train.py \
        --data_root "$DATA" \
        --out_root "$OUT" \
        --max_iters $ITERS \
        --n_gaussians $N_GAUSSIANS \
        --acceleration $ACC \
        --mask_type $MASK_TYPE \
        --center_fraction $CENTER_FRAC \
        --distributed 1 \
        $EXTRA_ARGS
else
    echo "Mode: Single-GPU (GPU $GPU)"
    echo "=============================================="
    
    CUDA_VISIBLE_DEVICES=$GPU python train.py \
        --data_root "$DATA" \
        --out_root "$OUT" \
        --max_iters $ITERS \
        --n_gaussians $N_GAUSSIANS \
        --acceleration $ACC \
        --mask_type $MASK_TYPE \
        --center_fraction $CENTER_FRAC \
        $EXTRA_ARGS
fi

echo ""
echo "Training complete!"
echo "Results saved to: $OUT"
