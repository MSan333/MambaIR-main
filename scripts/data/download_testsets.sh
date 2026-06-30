#!/bin/bash
# 下载超分辨率标准测试集（Set5, Set14, BSD100, Urban100, Manga109）
# 用法: bash scripts/data/download_testsets.sh [目标目录]

set -e

DATA_DIR="${1:-$(dirname $(dirname $(dirname $(realpath $0))))/datasets}"
SR_DIR="$DATA_DIR/SR"
mkdir -p "$SR_DIR"

echo "[测试集] 开始下载..."

# 方法1: 从Google Drive下载预打包测试集（推荐）
# 如果无法访问Google Drive，使用方法2
GDRIVE_URL="https://drive.google.com/uc?export=download&id=1n-7pmwjP0isZBK7w3tx2y8CTastlABx1"

# 方法2: 从GitHub备份下载
GITHUB_BACKUP="https://github.com/jbhuang0604/SelfExSR/archive/refs/heads/master.zip"

# 尝试方法1
echo "  [下载] 尝试从Google Drive下载测试集合集..."
if curl -L -C - --retry 3 -o "$SR_DIR/benchmark.zip" "$GDRIVE_URL" 2>/dev/null; then
    # 检查是否真正下载成功（Google Drive可能返回HTML确认页）
    if file "$SR_DIR/benchmark.zip" | grep -q "Zip archive"; then
        echo "  [完成] Google Drive下载成功"
        echo "  [解压] benchmark.zip..."
        cd "$SR_DIR"
        unzip -q -o benchmark.zip
        echo "  [完成] 测试集解压完毕"
    else
        echo "  [警告] Google Drive下载可能需要确认，尝试备用方案..."
        rm -f "$SR_DIR/benchmark.zip"
    fi
else
    echo "  [警告] Google Drive下载失败"
fi

# 如果测试集目录不存在，手动创建结构并提示
if [ ! -d "$SR_DIR/Set5" ]; then
    echo ""
    echo "  [提示] 自动下载失败，请手动下载测试集："
    echo "  1. Google Drive: https://drive.google.com/file/d/1n-7pmwjP0isZBK7w3tx2y8CTastlABx1/"
    echo "  2. 解压到: $SR_DIR/"
    echo ""
    echo "  或从以下地址分别下载："
    echo "  - Set5: https://github.com/jbhuang0604/SelfExSR"
    echo "  - Set14: https://github.com/jbhuang0604/SelfExSR"
    echo "  - BSD100: https://github.com/jbhuang0604/SelfExSR"
    echo "  - Urban100: https://github.com/jbhuang0604/SelfExSR"
    echo "  - Manga109: http://www.manga109.org/ (需申请)"
    
    # 创建目录结构占位
    for dataset in Set5 Set14 BSD100 Urban100 Manga109; do
        mkdir -p "$SR_DIR/$dataset/HR"
        mkdir -p "$SR_DIR/$dataset/LR_bicubic/X2"
        mkdir -p "$SR_DIR/$dataset/LR_bicubic/X3"
        mkdir -p "$SR_DIR/$dataset/LR_bicubic/X4"
    done
    echo "  [完成] 已创建目录结构占位"
fi

echo ""
echo "[测试集] 处理完成！"
echo "  目录: $SR_DIR/"
ls -d "$SR_DIR"/*/ 2>/dev/null || true
