#!/bin/bash
# 下载所有超分辨率数据集
# 用法: bash scripts/data/download_all.sh [目标目录]
# 默认下载到项目根目录的 datasets/ 下

set -e

# 目标目录（可通过参数指定）
DATA_DIR="${1:-$(dirname $(dirname $(dirname $(realpath $0))))/datasets}"
echo "============================================"
echo "  超分辨率数据集下载工具"
echo "  目标目录: $DATA_DIR"
echo "============================================"

mkdir -p "$DATA_DIR"

# 下载DIV2K
bash "$(dirname $0)/download_div2k.sh" "$DATA_DIR"

# 下载测试集
bash "$(dirname $0)/download_testsets.sh" "$DATA_DIR"

# 下载Flickr2K（可选，较大）
bash "$(dirname $0)/download_flickr2k.sh" "$DATA_DIR"

echo ""
echo "============================================"
echo "  全部下载完成！"
echo "  数据目录: $DATA_DIR"
echo "============================================"
