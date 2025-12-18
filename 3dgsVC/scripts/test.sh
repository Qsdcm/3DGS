#!/bin/bash
# 3DGSMR Testing Script
#
# 用法:
#   bash scripts/test.sh --checkpoint outputs/best.pth
#   bash scripts/test.sh --checkpoint outputs/best.pth --output_dir test_results/
#
# 论文: Three-Dimensional MRI Reconstruction with Gaussian Representations

set -e

# ======================= 配置 =======================
# 项目根目录
PROJECT_ROOT="/data/data54/wanghaobo/3DGS/3dgsVC"

# 数据路径
DATA_PATH="/data/data54/wanghaobo/data/ksp_full.h5"

# 默认参数
CHECKPOINT=""
OUTPUT_DIR="${PROJECT_ROOT}/test_results"
GPU=0
SAVE_VOLUME=true
SAVE_SLICES=false
CONFIG=""

# ======================= 解析参数 =======================
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
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
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --save_slices)
            SAVE_SLICES=true
            shift
            ;;
        --no_save_volume)
            SAVE_VOLUME=false
            shift
            ;;
        --help)
            echo "3DGSMR Testing Script"
            echo ""
            echo "Usage: bash scripts/test.sh --checkpoint PATH [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --checkpoint PATH      Path to checkpoint file"
            echo ""
            echo "Options:"
            echo "  --config PATH          Path to config file (optional, uses checkpoint config)"
            echo "  --output_dir PATH      Output directory (default: test_results/)"
            echo "  --gpu ID               GPU device ID (default: 0)"
            echo "  --data_path PATH       Path to data file"
            echo "  --save_slices          Save slice images as PNG"
            echo "  --no_save_volume       Don't save volume as .npy"
            echo "  --help                 Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ======================= 检查参数 =======================
if [ -z "${CHECKPOINT}" ]; then
    echo "Error: --checkpoint is required"
    echo "Usage: bash scripts/test.sh --checkpoint PATH"
    exit 1
fi

if [ ! -f "${CHECKPOINT}" ]; then
    echo "Error: Checkpoint file not found: ${CHECKPOINT}"
    exit 1
fi

# ======================= 环境设置 =======================
# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=${GPU}

# 切换到项目目录
cd ${PROJECT_ROOT}

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# ======================= 打印配置 =======================
echo "=============================================="
echo "          3DGSMR Testing Script              "
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Project Root:     ${PROJECT_ROOT}"
echo "  Checkpoint:       ${CHECKPOINT}"
echo "  Data Path:        ${DATA_PATH}"
echo "  Output Dir:       ${OUTPUT_DIR}"
echo "  GPU:              ${GPU}"
echo "  Save Volume:      ${SAVE_VOLUME}"
echo "  Save Slices:      ${SAVE_SLICES}"
if [ -n "${CONFIG}" ]; then
    echo "  Config:           ${CONFIG}"
fi
echo ""
echo "=============================================="

# ======================= 检查文件 =======================
if [ ! -f "${DATA_PATH}" ]; then
    echo "Error: Data file not found: ${DATA_PATH}"
    exit 1
fi

# ======================= 运行测试 =======================
echo ""
echo "Starting testing..."
echo ""

# 构建命令
CMD="python test.py \
    --checkpoint ${CHECKPOINT} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --gpu 0"

# 添加可选参数
if [ -n "${CONFIG}" ]; then
    CMD="${CMD} --config ${CONFIG}"
fi

if [ "${SAVE_VOLUME}" = true ]; then
    CMD="${CMD} --save_volume"
fi

if [ "${SAVE_SLICES}" = true ]; then
    CMD="${CMD} --save_slices"
fi

# 执行测试
echo "Command: ${CMD}"
echo ""

${CMD}

# ======================= 测试完成 =======================
echo ""
echo "=============================================="
echo "          Testing Complete!                  "
echo "=============================================="
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Output files:"
echo "  - metrics.yaml         : Evaluation metrics"
if [ "${SAVE_VOLUME}" = true ]; then
    echo "  - reconstruction.npy   : Reconstructed volume (complex)"
    echo "  - reconstruction_magnitude.npy : Magnitude image"
    echo "  - target.npy           : Ground truth"
    echo "  - zero_filled.npy      : Zero-filled baseline"
fi
if [ "${SAVE_SLICES}" = true ]; then
    echo "  - slices/              : Slice images"
fi
echo ""
