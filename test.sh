#!/bin/bash
# test.sh - 3D Gaussian Splatting MRI Reconstruction Test/Evaluation Script
#
# Usage (single GPU):
#   bash test.sh
#
# Usage (multi-GPU):
#   MULTI_GPU=1 bash test.sh

# ============================================================================
# Configuration - Modify these variables
# ============================================================================

# GPU settings
GPU=${GPU:-0}                          # GPU ID for single-GPU testing
NGPU=${NGPU:-2}                        # Number of GPUs for multi-GPU testing
MULTI_GPU=${MULTI_GPU:-0}              # Set to 1 to enable multi-GPU

# Paths
DATA=${DATA:-""}                       # H5 file or directory (optional, for GT comparison)
OUT=${OUT:-./outputs}                  # Output directory containing results

# Data-related hyperparameters (only used if DATA is provided)
ACC=${ACC:-4.0}                        # Acceleration factor
MASK_TYPE=${MASK_TYPE:-gaussian}       # Mask type: gaussian or uniform

# Output
OUTPUT_FILE=${OUTPUT_FILE:-results.json}

# Optional: Additional arguments
EXTRA_ARGS=${EXTRA_ARGS:-""}

# ============================================================================
# Run Evaluation
# ============================================================================

echo "=============================================="
echo "3DGS MRI Reconstruction - Evaluation"
echo "=============================================="
echo "Results directory: $OUT"
if [ -n "$DATA" ]; then
    echo "Data (for GT): $DATA"
    echo "Acceleration: ${ACC}x"
    echo "Mask Type: $MASK_TYPE"
fi
echo "Output file: $OUTPUT_FILE"
echo "=============================================="

# Build command
CMD="--out_root \"$OUT\" --output_file $OUTPUT_FILE"

if [ -n "$DATA" ]; then
    CMD="$CMD --data_root \"$DATA\" --acceleration $ACC --mask_type $MASK_TYPE"
fi

if [ "$MULTI_GPU" -eq 1 ]; then
    echo "Mode: Multi-GPU ($NGPU GPUs)"
    echo "=============================================="
    
    eval "CUDA_VISIBLE_DEVICES=$GPU torchrun \
        --nproc_per_node=$NGPU \
        test.py \
        $CMD \
        --distributed 1 \
        $EXTRA_ARGS"
else
    echo "Mode: Single-GPU (GPU $GPU)"
    echo "=============================================="
    
    eval "CUDA_VISIBLE_DEVICES=$GPU python test.py \
        $CMD \
        $EXTRA_ARGS"
fi

echo ""
echo "Evaluation complete!"
echo "Results saved to: $OUT/$OUTPUT_FILE"
