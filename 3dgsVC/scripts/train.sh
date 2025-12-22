#!/bin/bash
# 3DGSMR Training Script (Cleaned)
#
# 用法:
#   bash scripts/train.sh                    # 完全使用 default.yaml 的配置
#   bash scripts/train.sh --acceleration 8   # 仅覆盖加速倍数，其他用 yaml
#   bash scripts/train.sh --gpu 1            # 指定 GPU

set -e

# ======================= 基础配置 =======================
# 项目根目录 (根据实际情况调整)
PROJECT_ROOT="/data/data54/wanghaobo/3DGS/3dgsVC"

# 默认配置文件
CONFIG="${PROJECT_ROOT}/configs/default.yaml"

# 默认 GPU ID (仅此保留默认值，因为 shell 需要用来设置环境变量)
GPU=0

# 初始化变量为空 (依靠 default.yaml 为准)
DATA_PATH=""
OUTPUT_DIR=""
ACCELERATION=""
MAX_ITERATIONS=""
INITIAL_POINTS=""
SEED=""
RESUME=""

# ======================= 解析参数 =======================
while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --acceleration) ACCELERATION="$2"; shift 2 ;;
        --max_iterations) MAX_ITERATIONS="$2"; shift 2 ;;
        --initial_points) INITIAL_POINTS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --resume) RESUME="$2"; shift 2 ;;
        --data_path) DATA_PATH="$2"; shift 2 ;;
        --help)
            echo "Usage: bash scripts/train.sh [OPTIONS]"
            echo "Options correspond to train.py arguments."
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ======================= 环境设置 =======================
export CUDA_VISIBLE_DEVICES=${GPU}
cd ${PROJECT_ROOT}

echo "=============================================="
echo "          3DGSMR Training (Yaml Based)       "
echo "=============================================="
echo "Config: ${CONFIG}"
echo "GPU:    ${GPU}"
echo "=============================================="

# ======================= 构建命令 =======================
# 基础命令，始终使用 config
CMD="python train.py --config ${CONFIG} --gpu 0"

# 仅当用户在 shell 中指定了参数时，才添加到命令中覆盖 yaml
if [ -n "${DATA_PATH}" ]; then CMD="${CMD} --data_path ${DATA_PATH}"; fi
if [ -n "${OUTPUT_DIR}" ]; then CMD="${CMD} --output_dir ${OUTPUT_DIR}"; fi
if [ -n "${ACCELERATION}" ]; then CMD="${CMD} --acceleration ${ACCELERATION}"; fi
if [ -n "${MAX_ITERATIONS}" ]; then CMD="${CMD} --max_iterations ${MAX_ITERATIONS}"; fi
if [ -n "${INITIAL_POINTS}" ]; then CMD="${CMD} --initial_points ${INITIAL_POINTS}"; fi
if [ -n "${SEED}" ]; then CMD="${CMD} --seed ${SEED}"; fi
if [ -n "${RESUME}" ]; then CMD="${CMD} --resume ${RESUME}"; fi

# ======================= 运行 =======================
echo "Executing: ${CMD}"
echo ""

${CMD}