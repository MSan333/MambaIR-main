#!/bin/bash
# 下载Flickr2K数据集（2650张高清图像）
# 用法: bash scripts/data/download_flickr2k.sh [目标目录]
# 注意: 文件较大(~2.4GB)，下载时间较长

set -e

DATA_DIR="${1:-$(dirname $(dirname $(dirname $(realpath $0))))/datasets}"
FLICKR_DIR="$DATA_DIR/Flickr2K"
mkdir -p "$FLICKR_DIR"

echo "[Flickr2K] 开始下载（~2.4GB，请耐心等待）..."

FLICKR_URL="https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar"

if [ -f "$FLICKR_DIR/Flickr2K.tar" ]; then
    echo "  [跳过] Flickr2K.tar 已存在"
elif [ -d "$FLICKR_DIR/Flickr2K" ]; then
    echo "  [跳过] 已解压"
else
    echo "  [下载] Flickr2K.tar..."
    curl -L -C - --retry 5 --retry-delay 10 -o "$FLICKR_DIR/Flickr2K.tar" "$FLICKR_URL"
    echo "  [完成] Flickr2K.tar"
fi

# 解压
if [ -f "$FLICKR_DIR/Flickr2K.tar" ] && [ ! -d "$FLICKR_DIR/Flickr2K" ]; then
    echo "  [解压] Flickr2K.tar..."
    cd "$FLICKR_DIR"
    tar -xf Flickr2K.tar
    echo "  [完成] 解压完毕"
fi

echo ""
echo "[Flickr2K] 下载完成！"
echo "  目录: $FLICKR_DIR/Flickr2K/ (2650张)"
