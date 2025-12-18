#!/bin/bash
# 3DGSMR Training Script
#
# 用法:
#   bash scripts/train.sh                    # 使用默认配置
#   bash scripts/train.sh --acceleration 6  # 指定加速因子
#   bash scripts/train.sh --gpu 1           # 使用GPU 1
#
# 论文: Three-Dimensional MRI Reconstruction with Gaussian Representations

set -e

# ======================= 配置 =======================
# 项目根目录
PROJECT_ROOT="/data/data54/wanghaobo/3DGS/3dgsVC"

# 数据路径
DATA_PATH="/data/data54/wanghaobo/data/ksp_full.h5"

# 默认参数
CONFIG="${PROJECT_ROOT}/configs/default.yaml"
OUTPUT_DIR="${PROJECT_ROOT}/outputs"
GPU=0
ACCELERATION=4
MAX_ITERATIONS=2000
INITIAL_POINTS=500
SEED=42
RESUME=""

# ======================= 解析参数 =======================
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --acceleration)
            ACCELERATION="$2"
            shift 2
            ;;
        --max_iterations)
            MAX_ITERATIONS="$2"
            shift 2
            ;;
        --initial_points)
            INITIAL_POINTS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --help)
            echo "3DGSMR Training Script"
            echo ""
            echo "Usage: bash scripts/train.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config PATH          Path to config file (default: configs/default.yaml)"
            echo "  --output_dir PATH      Output directory (default: outputs/)"
            echo "  --gpu ID               GPU device ID (default: 0)"
            echo "  --acceleration FACTOR  Acceleration factor (default: 4)"
            echo "  --max_iterations N     Maximum iterations (default: 2000)"
            echo "  --initial_points N     Initial number of Gaussians (default: 500)"
            echo "  --seed N               Random seed (default: 42)"
            echo "  --resume PATH          Resume from checkpoint"
            echo "  --data_path PATH       Path to data file"
            echo "  --help                 Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ======================= 环境设置 =======================
# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=${GPU}

# 切换到项目目录
cd ${PROJECT_ROOT}

# 创建输出目录
OUTPUT_DIR_FULL="${OUTPUT_DIR}/acc${ACCELERATION}_pts${INITIAL_POINTS}_seed${SEED}"
mkdir -p ${OUTPUT_DIR_FULL}

# ======================= 打印配置 =======================
echo "=============================================="
echo "          3DGSMR Training Script             "
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Project Root:     ${PROJECT_ROOT}"
echo "  Config:           ${CONFIG}"
echo "  Data Path:        ${DATA_PATH}"
echo "  Output Dir:       ${OUTPUT_DIR_FULL}"
echo "  GPU:              ${GPU}"
echo "  Acceleration:     ${ACCELERATION}x"
echo "  Max Iterations:   ${MAX_ITERATIONS}"
echo "  Initial Points:   ${INITIAL_POINTS}"
echo "  Seed:             ${SEED}"
if [ -n "${RESUME}" ]; then
    echo "  Resume From:      ${RESUME}"
fi
echo ""
echo "=============================================="

# ======================= 检查文件 =======================
if [ ! -f "${DATA_PATH}" ]; then
    echo "Error: Data file not found: ${DATA_PATH}"
    exit 1
fi

if [ ! -f "${CONFIG}" ]; then
    echo "Error: Config file not found: ${CONFIG}"
    exit 1
fi

# ======================= 运行训练 =======================
echo ""
echo "Starting training..."
echo ""

# 构建命令
CMD="python train.py \
    --config ${CONFIG} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR_FULL} \
    --gpu 0 \
    --acceleration ${ACCELERATION} \
    --max_iterations ${MAX_ITERATIONS} \
    --initial_points ${INITIAL_POINTS} \
    --seed ${SEED}"

# 添加resume参数
if [ -n "${RESUME}" ]; then
    CMD="${CMD} --resume ${RESUME}"
fi

# 执行训练
echo "Command: ${CMD}"
echo ""

${CMD}

# ======================= 训练完成 =======================
echo ""
echo "=============================================="
echo "          Training Complete!                 "
echo "=============================================="
echo ""
echo "Results saved to: ${OUTPUT_DIR_FULL}"
echo ""
echo "To test the model, run:"
echo "  bash scripts/test.sh --checkpoint ${OUTPUT_DIR_FULL}/checkpoints/best.pth"
echo ""
