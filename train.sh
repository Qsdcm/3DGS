#!/bin/bash
# 3DGSMR Training Script
# 使用方法: bash train.sh

# ============================================
# 环境设置
# ============================================
echo "============================================"
echo "3DGSMR Training Script"
echo "============================================"

# 激活 conda 环境
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate pt1.10

# 设置 GPU
export CUDA_VISIBLE_DEVICES=3

# PyTorch 内存管理优化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# ============================================
# 配置参数
# ============================================

# 数据路径
DATA_PATH="/data/data54/wanghaobo/data/ksp_full.h5"

# 欠采样参数
ACCELERATION=4.0        # 加速倍数: 4x, 8x, 等
CENTER_FRACTION=0.08    # 中心全采样比例
MASK_TYPE="gaussian"    # 掩码类型: gaussian, uniform

# 模型参数
INIT_NUM_GAUSSIANS=10000
MAX_GAUSSIANS=100000

# 训练参数
NUM_ITERATIONS=5000
LEARNING_RATE=0.001
DENSIFY_INTERVAL=100

# 日志参数
LOG_INTERVAL=100
SAVE_INTERVAL=1000
SAVE_DIR="checkpoints/3dgsmr_acc${ACCELERATION}_$(date +%Y%m%d_%H%M%S)"

# 设备
DEVICE="cuda:0"  # 因为 CUDA_VISIBLE_DEVICES=3，所以这里用 cuda:0
SEED=42

# ============================================
# 运行训练
# ============================================
echo ""
echo "Configuration:"
echo "  Data Path: ${DATA_PATH}"
echo "  Acceleration: ${ACCELERATION}x"
echo "  Mask Type: ${MASK_TYPE}"
echo "  Num Iterations: ${NUM_ITERATIONS}"
echo "  Save Dir: ${SAVE_DIR}"
echo "  Device: GPU 3 (${DEVICE})"
echo ""

cd /data/data54/wanghaobo/3DGS

python train.py \
    --data_path "${DATA_PATH}" \
    --acceleration ${ACCELERATION} \
    --center_fraction ${CENTER_FRACTION} \
    --mask_type "${MASK_TYPE}" \
    --init_num_gaussians ${INIT_NUM_GAUSSIANS} \
    --max_gaussians ${MAX_GAUSSIANS} \
    --num_iterations ${NUM_ITERATIONS} \
    --learning_rate ${LEARNING_RATE} \
    --densify_interval ${DENSIFY_INTERVAL} \
    --log_interval ${LOG_INTERVAL} \
    --save_interval ${SAVE_INTERVAL} \
    --save_dir "${SAVE_DIR}" \
    --device "${DEVICE}" \
    --seed ${SEED}

echo ""
echo "============================================"
echo "Training completed!"
echo "Checkpoints saved to: ${SAVE_DIR}"
echo "============================================"
