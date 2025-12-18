#!/bin/bash
# train.sh - 3D Gaussian Splatting MRI Reconstruction (Single GPU Version)
#
# Usage:
#   bash train.sh

# ============================================================================
# Configuration
# ============================================================================

# 1. GPU 设置 (指定单卡，例如使用第0号卡)
GPU=3

# 2. 数据路径 (已修改为您提供的真实路径)
DATA="/data/data54/wanghaobo/data/ksp_full.h5"

# 3. 输出目录
OUT="./results_single_gpu"

# 4. 关键超参数 (根据优化建议调整，确保能训练出图像)
# 初始高斯点数 (调大以覆盖主体)
N_GAUSSIANS=50000
# 初始亮度系数 (调大以防止初始全黑)
K_INIT=10.0
# 密度学习率 (调大以加快收敛)
LR_RHO=0.05
# 中心点学习率
LR_CENTERS=0.003
# 加速倍数
ACC=4.0

# ============================================================================
# Run Training
# ============================================================================

echo "=============================================="
echo "3DGS MRI Reconstruction - Single GPU Training"
echo "=============================================="
echo "GPU: $GPU"
echo "Data: $DATA"
echo "Output: $OUT"
echo "Params: N_Gaussians=$N_GAUSSIANS, K_Init=$K_INIT, LR_Rho=$LR_RHO"
echo "=============================================="

# 确保脚本遇到错误即停止
set -e

# 运行命令 (使用 python 直接运行，避开 torchrun 的复杂性)
CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --data_root "$DATA" \
    --out_root "$OUT" \
    --n_gaussians $N_GAUSSIANS \
    --k_init $K_INIT \
    --lr_rho $LR_RHO \
    --lr_centers $LR_CENTERS \
    --acceleration $ACC \
    --mask_type gaussian \
    --max_iters 3000 \
    --densify_every 300 \
    --vis_every 100 \
    --print_every 50

echo ""
echo "Training complete! Results saved to: $OUT"