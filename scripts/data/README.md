# 数据集下载与管理脚本

## 使用方法

### 快速开始（服务器上）

```bash
# 1. 下载所有数据集（DIV2K + 测试集）
bash scripts/data/download_all.sh

# 2. 仅下载DIV2K训练集（最小必要数据）
bash scripts/data/download_div2k.sh

# 3. 仅下载测试集
bash scripts/data/download_testsets.sh

# 4. 从OSS下载（如果已上传到OSS）
pip install oss2
python scripts/data/download_from_oss.py
```

### 目录结构
下载后数据集会组织为以下结构：
```
datasets/
├── DF2K/
│   ├── HR/                    # DIV2K + Flickr2K 高分辨率
│   └── LR_bicubic/
│       ├── X2/
│       ├── X3/
│       └── X4/
├── DIV2K/
│   ├── DIV2K_train_HR/
│   ├── DIV2K_train_LR_bicubic/X2, X3, X4
│   ├── DIV2K_valid_HR/
│   └── DIV2K_valid_LR_bicubic/X2, X3, X4
└── SR/
    ├── Set5/HR, LR_bicubic/X2,X3,X4
    ├── Set14/...
    ├── BSD100/...
    ├── Urban100/...
    └── Manga109/...
```

### 配置说明
- 训练配置中的数据路径需要与实际下载位置对应
- 修改 `options/train/mfamambaIR/train_MFAMambaIR_SR_x4.yml` 中的 `dataroot_gt` 和 `dataroot_lq`
