#!/bin/bash
# MFAMambaIRv2-light SR x4 训练脚本
# 使用 screen 后台运行, 8卡 RTX 2080Ti 分布式训练
# 仅保留 best + latest checkpoint

set -e

# ============ 配置 ============
PROJECT_DIR="/home/guoshuaile/project/MambaIR-main"
CONFIG="options/train/mfamambaIRv2/train_MFAMambaIRv2_lightSR_x4.yml"
NGPU=8
SCREEN_NAME="train_lightSR_x4"
CONDA_ENV="sr"
LOG_DIR="${PROJECT_DIR}/experiments/logs"
LOG_FILE="${LOG_DIR}/${SCREEN_NAME}_$(date +%Y%m%d_%H%M%S).log"
# =================================

# 创建日志目录
mkdir -p "$LOG_DIR"

# 检查是否已有同名 screen 会话
if screen -list | grep -q "$SCREEN_NAME"; then
    echo "[错误] screen 会话 '$SCREEN_NAME' 已存在"
    echo "  查看: screen -r $SCREEN_NAME"
    echo "  终止: screen -X -S $SCREEN_NAME quit"
    exit 1
fi

echo "============================================"
echo "  MFAMambaIRv2-light SR x4 训练"
echo "============================================"
echo "  配置文件: $CONFIG"
echo "  GPU数量:  $NGPU"
echo "  Conda环境: $CONDA_ENV"
echo "  Screen:   $SCREEN_NAME"
echo "  日志文件:  $LOG_FILE"
echo "============================================"
echo ""

# 启动 screen 会话
screen -dmS "$SCREEN_NAME" bash -c "
    source /home/guoshuaile/miniconda3/etc/profile.d/conda.sh
    conda activate $CONDA_ENV
    cd $PROJECT_DIR
    export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9
    echo '=== 训练开始: $(date) ==='
    echo '=== 配置: $CONFIG ==='
    python -m torch.distributed.launch --nproc_per_node=$NGPU basicsr/train.py -opt $CONFIG --launcher pytorch 2>&1 | tee $LOG_FILE
    echo '=== 训练结束: $(date) ==='
    echo '=== 按任意键关闭此窗口 ==='
    read -n 1
"

echo "训练已在 screen 会话 '$SCREEN_NAME' 中启动"
echo ""
echo "常用命令:"
echo "  进入会话:  screen -r $SCREEN_NAME"
echo "  查看日志:  tail -f $LOG_FILE"
echo "  退出会话:  Ctrl+A 然后按 D"
echo "  终止训练:  screen -X -S $SCREEN_NAME quit"
