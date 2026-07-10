#!/bin/bash
# ===========================================================================
# MFAMambaIRv2 分布式训练启动脚本 (8卡 RTX 2080)
# 用法: bash scripts/train_mfamambaIRv2_8gpu.sh [config_name]
# 示例: bash scripts/train_mfamambaIRv2_8gpu.sh lightSR_x4
#       bash scripts/train_mfamambaIRv2_8gpu.sh SR_x2
# ===========================================================================

set -e

# ---- 参数解析 ----
CONFIG_NAME="${1:-lightSR_x4}"
NUM_GPUS=8
PORT=29500

# ---- 路径 ----
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_FILE="${ROOT_DIR}/options/train/mfamambaIRv2/train_MFAMambaIRv2_${CONFIG_NAME}.yml"

# ---- 检查配置文件是否存在 ----
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    echo "可用配置:"
    ls "${ROOT_DIR}/options/train/mfamambaIRv2/" | sed 's/train_MFAMambaIRv2_//;s/.yml//'
    exit 1
fi

echo "============================================================"
echo "  MFAMambaIRv2 分布式训练"
echo "  配置: ${CONFIG_NAME}"
echo "  GPU数: ${NUM_GPUS}"
echo "  端口: ${PORT}"
echo "  配置文件: ${CONFIG_FILE}"
echo "============================================================"

# ---- 检查可用的 GPU ----
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# ---- 启动分布式训练 ----
cd "$ROOT_DIR"
python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${PORT} \
    basicsr/train.py \
    -opt "${CONFIG_FILE}" \
    --launcher pytorch

echo "训练完成。"
