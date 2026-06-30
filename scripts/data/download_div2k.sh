#!/bin/bash
# 下载DIV2K数据集（训练集800张 + 验证集100张）
# 用法: bash scripts/data/download_div2k.sh [目标目录]

set -e

DATA_DIR="${1:-$(dirname $(dirname $(dirname $(realpath $0))))/datasets}"
DIV2K_DIR="$DATA_DIR/DIV2K"
mkdir -p "$DIV2K_DIR"

echo "[DIV2K] 开始下载..."

# DIV2K官方下载地址
BASE_URL="https://data.vision.ee.ethz.ch/cvl/DIV2K"

# 下载函数（支持断点续传）
download() {
    local url=$1
    local output=$2
    if [ -f "$output" ]; then
        echo "  [跳过] $(basename $output) 已存在"
        return 0
    fi
    echo "  [下载] $(basename $output)..."
    curl -L -C - --retry 5 --retry-delay 10 -o "$output" "$url"
    echo "  [完成] $(basename $output)"
}

# 训练集
echo ""
echo "[DIV2K] 下载训练集..."
download "$BASE_URL/DIV2K_train_HR.zip" "$DIV2K_DIR/DIV2K_train_HR.zip"
download "$BASE_URL/DIV2K_train_LR_bicubic_X2.zip" "$DIV2K_DIR/DIV2K_train_LR_bicubic_X2.zip"
download "$BASE_URL/DIV2K_train_LR_bicubic_X3.zip" "$DIV2K_DIR/DIV2K_train_LR_bicubic_X3.zip"
download "$BASE_URL/DIV2K_train_LR_bicubic_X4.zip" "$DIV2K_DIR/DIV2K_train_LR_bicubic_X4.zip"

# 验证集
echo ""
echo "[DIV2K] 下载验证集..."
download "$BASE_URL/DIV2K_valid_HR.zip" "$DIV2K_DIR/DIV2K_valid_HR.zip"
download "$BASE_URL/DIV2K_valid_LR_bicubic_X2.zip" "$DIV2K_DIR/DIV2K_valid_LR_bicubic_X2.zip"
download "$BASE_URL/DIV2K_valid_LR_bicubic_X3.zip" "$DIV2K_DIR/DIV2K_valid_LR_bicubic_X3.zip"
download "$BASE_URL/DIV2K_valid_LR_bicubic_X4.zip" "$DIV2K_DIR/DIV2K_valid_LR_bicubic_X4.zip"

# 解压
echo ""
echo "[DIV2K] 解压文件..."
cd "$DIV2K_DIR"
for f in *.zip; do
    if [ -f "$f" ]; then
        dir_name="${f%.zip}"
        if [ -d "$dir_name" ]; then
            echo "  [跳过] $dir_name/ 已解压"
        else
            echo "  [解压] $f..."
            unzip -q "$f"
            echo "  [完成] $f"
        fi
    fi
done

echo ""
echo "[DIV2K] 下载完成！"
echo "  训练HR: $DIV2K_DIR/DIV2K_train_HR/ (800张)"
echo "  训练LR: $DIV2K_DIR/DIV2K_train_LR_bicubic/X2,X3,X4/"
echo "  验证HR: $DIV2K_DIR/DIV2K_valid_HR/ (100张)"
echo "  验证LR: $DIV2K_DIR/DIV2K_valid_LR_bicubic/X2,X3,X4/"
