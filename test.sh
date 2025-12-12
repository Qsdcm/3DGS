#!/bin/bash
# 3DGSMR Testing Script
# 使用方法: bash test.sh <checkpoint_path>
# 示例: bash test.sh checkpoints/3dgsmr_acc4.0_20241210/checkpoint_final.pth

# ============================================
# 环境设置
# ============================================
echo "============================================"
echo "3DGSMR Testing Script"
echo "============================================"

# 激活 conda 环境
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate pt1.10

# 设置 GPU
export CUDA_VISIBLE_DEVICES=3

# PyTorch 内存管理优化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# ============================================
# 参数检查
# ============================================
if [ -z "$1" ]; then
    echo "Usage: bash test.sh <checkpoint_path>"
    echo "Example: bash test.sh checkpoints/3dgsmr_acc4.0_20241210/checkpoint_final.pth"
    
    # 如果没有提供参数，尝试找到最新的 checkpoint
    LATEST_CKPT=$(find checkpoints -name "checkpoint_final.pth" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    
    if [ -n "$LATEST_CKPT" ]; then
        echo ""
        echo "Found latest checkpoint: ${LATEST_CKPT}"
        echo "Using this checkpoint..."
        CHECKPOINT="${LATEST_CKPT}"
    else
        echo "No checkpoint found. Please train the model first."
        exit 1
    fi
else
    CHECKPOINT="$1"
fi

# 检查 checkpoint 是否存在
if [ ! -f "${CHECKPOINT}" ]; then
    echo "Error: Checkpoint file not found: ${CHECKPOINT}"
    exit 1
fi

# ============================================
# 配置参数
# ============================================
SAVE_DIR="results/$(basename $(dirname ${CHECKPOINT}))_$(date +%Y%m%d_%H%M%S)"
DEVICE="cuda:0"  # 因为 CUDA_VISIBLE_DEVICES=1，所以这里用 cuda:0
VISUALIZE="--visualize"
NUM_SLICES=5

# ============================================
# 运行测试
# ============================================
echo ""
echo "Configuration:"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Save Dir: ${SAVE_DIR}"
echo "  Device: GPU 1 (${DEVICE})"
echo ""

cd /data/data54/wanghaobo/3DGS

python test.py \
    --checkpoint "${CHECKPOINT}" \
    --save_dir "${SAVE_DIR}" \
    --device "${DEVICE}" \
    ${VISUALIZE} \
    --num_slices ${NUM_SLICES}

echo ""
echo "============================================"
echo "Testing completed!"
echo "Results saved to: ${SAVE_DIR}"
echo "============================================"
