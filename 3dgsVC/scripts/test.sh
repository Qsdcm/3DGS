#!/bin/bash
# 3DGSMR Testing Script
#
# 用法:
#   bash scripts/test.sh --dataset /path/to/data --weights /path/to/checkpoint --acceleration 8

set -e

# ======================= 基础配置 =======================
PROJECT_ROOT="/data/data54/wanghaobo/3DGS/3dgsVC"
CONFIG="${PROJECT_ROOT}/configs/default.yaml"
GPU=0

# 必需参数初始化为空
DATASET=""
WEIGHTS=""
ACCELERATION=""
SAVE_VOLUME="true" # 默认开启
SAVE_SLICES="true" # 默认开启

# ======================= 解析参数 =======================
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --weights) WEIGHTS="$2"; shift 2 ;;
        --acceleration) ACCELERATION="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --no_save) SAVE_VOLUME="false"; SAVE_SLICES="false"; shift 1 ;; # 可选关闭保存
        --help)
            echo "Usage: bash scripts/test.sh --dataset PATH --weights PATH --acceleration N"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ======================= 检查必需参数 =======================
if [ -z "${DATASET}" ] || [ -z "${WEIGHTS}" ]; then
    echo "Error: --dataset and --weights are required."
    echo "Example: bash scripts/test.sh --dataset data.h5 --weights model.pth --acceleration 4"
    exit 1
fi

# ======================= 环境设置 =======================
export CUDA_VISIBLE_DEVICES=${GPU}
cd ${PROJECT_ROOT}

echo "=============================================="
echo "          3DGSMR Testing                     "
echo "=============================================="
echo "Dataset:      ${DATASET}"
echo "Weights:      ${WEIGHTS}"
echo "Acceleration: ${ACCELERATION:-'From Config'}"
echo "GPU:          ${GPU}"
echo "=============================================="

# ======================= 构建命令 =======================
CMD="python test.py --dataset ${DATASET} --weights ${WEIGHTS} --config ${CONFIG} --gpu 0"

if [ -n "${ACCELERATION}" ]; then CMD="${CMD} --acceleration ${ACCELERATION}"; fi
if [ "${SAVE_VOLUME}" = "true" ]; then CMD="${CMD} --save_volume"; fi
if [ "${SAVE_SLICES}" = "true" ]; then CMD="${CMD} --save_slices"; fi

# ======================= 运行 =======================
echo "Executing: ${CMD}"
echo ""

${CMD}