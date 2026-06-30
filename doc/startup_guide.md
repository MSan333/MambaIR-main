# MFAMambaIR 项目启动方案

## 1. 环境要求

| 组件 | 版本要求 |
|------|----------|
| OS | Ubuntu 20.04+ |
| Python | 3.8+ |
| PyTorch | 2.0.1+ |
| CUDA | 11.7+ |
| GPU显存 | ≥16GB (推荐24GB+) |

## 2. 环境安装

```bash
# 创建conda环境
conda create -n mfamambaIR python=3.8 -y
conda activate mfamambaIR

# 安装PyTorch (CUDA 11.7)
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

# 安装核心依赖
pip install causal_conv1d==1.0.0
pip install mamba_ssm==1.0.1

# 安装项目依赖
pip install einops timm opencv-python scikit-image lmdb pyyaml tqdm addict future yapf h5py

# 安装训练日志
pip install swanlab tensorboard

# 安装OSS工具（可选，用于数据下载）
pip install oss2
```

## 3. 数据集准备

### 方式1：从源站下载
```bash
bash scripts/data/download_all.sh
```

### 方式2：从OSS下载
```bash
export OSS_ACCESS_ID="your_access_id"
export OSS_ACCESS_KEY="your_access_key"
python scripts/data/download_from_oss.py --data-dir ./datasets
```

### 数据目录结构
```
datasets/
├── DF2K/
│   ├── HR/                         # 高分辨率训练图像
│   └── LR_bicubic/
│       ├── X2/                     # 2倍下采样
│       ├── X3/                     # 3倍下采样
│       └── X4/                     # 4倍下采样
└── Set5/                           # 验证集
    ├── HR/
    └── LR_bicubic/
        └── X4/
```

**注意**：如果只用 DIV2K (800张) 调试，将 DIV2K_train_HR 链接到 datasets/DF2K/HR 即可：
```bash
mkdir -p datasets/DF2K/LR_bicubic
ln -s /path/to/DIV2K_train_HR datasets/DF2K/HR
ln -s /path/to/DIV2K_train_LR_bicubic/X4 datasets/DF2K/LR_bicubic/X4
```

## 4. 训练配置说明

配置文件：`options/train/mfamambaIR/train_MFAMambaIR_SR_x4.yml`

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| batch_size_per_gpu | 8 | 单卡batch大小，16GB显存改为4 |
| gt_size | 256 | 训练patch大小，显存不足改为128 |
| total_iter | 500000 | 总迭代次数 |
| val_freq | 5000 | 验证频率（每5000iter验证一次） |
| lr | 2e-4 | 初始学习率 |
| milestones | [250000, 400000, 450000, 475000] | 学习率衰减节点 |

### 训练规模估算

| 数据集 | 图像数 | 每epoch iter数 | 总epoch数 |
|--------|--------|---------------|-----------|
| DIV2K | 800 | 100 | ~5000 |
| DF2K | 3450 | ~431 | ~1160 |

## 5. 启动训练

### SwanLab 配置
```bash
# 设置API Key（必须）
export SWANLAB_API_KEY="your_swanlab_api_key"
```

### 单卡训练
```bash
# x4超分（推荐先跑这个验证代码正确性）
python basicsr/train.py -opt options/train/mfamambaIR/train_MFAMambaIR_SR_x4.yml

# x2超分
python basicsr/train.py -opt options/train/mfamambaIR/train_MFAMambaIR_SR_x2.yml

# x3超分
python basicsr/train.py -opt options/train/mfamambaIR/train_MFAMambaIR_SR_x3.yml
```

### 多卡训练
```bash
# 4卡并行
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 \
    basicsr/train.py -opt options/train/mfamambaIR/train_MFAMambaIR_SR_x4.yml \
    --launcher pytorch
```

### 断点续训
```bash
# 修改配置文件中的 resume_state 路径
# path:
#   resume_state: experiments/MFAMambaIR_SR_x4/training_states/xxxxx.state

python basicsr/train.py -opt options/train/mfamambaIR/train_MFAMambaIR_SR_x4.yml
```

## 6. 显存优化方案

| GPU显存 | batch_size | gt_size | 预估训练时间(x4) |
|---------|-----------|---------|-----------------|
| 24GB (A5000/3090) | 8 | 256 | ~2天 |
| 16GB (V100/A4000) | 4 | 128 | ~3天 |
| 12GB (3060Ti) | 2 | 128 | ~5天 |

## 7. 查看训练进度

### SwanLab（推荐）
- 访问 https://swanlab.cn 登录查看
- 横坐标为 epoch，纵坐标为 loss/PSNR
- 同时记录 iter 信息便于对照

### TensorBoard
```bash
tensorboard --logdir experiments/MFAMambaIR_SR_x4/tb_logger --port 6006
```

### 终端日志
```
[MFAMa..][epoch: 10, iter:   1,000, lr:(2.000e-04,)] [eta: 1:23:45, time (data): 0.123 (0.012)] l_pix: 3.1234e-02
```

## 8. 测试评估

```bash
# 训练完成后测试
python basicsr/test.py -opt options/test/mfamambaIR/test_MFAMambaIR_SR_x4.yml
```

## 9. 目标指标

基于 MambaIRv2 baseline (×4)：

| 测试集 | MambaIRv2 PSNR | MFAMambaIR 目标 |
|--------|---------------|----------------|
| Set5 | 33.04 | 33.10+ |
| Set14 | 29.38 | 29.45+ |
| BSD100 | 28.04 | 28.08+ |
| Urban100 | 27.23 | 27.43+ (+0.20dB) |
| Manga109 | 32.47 | 32.62+ (+0.15dB) |

重点涨点目标：Urban100 和 Manga109（纹理/结构丰富，频域增强优势明显）

## 10. 快速验证清单

在正式长时间训练前，建议先做以下验证：

- [ ] `python -c "from basicsr.archs.mfamambaIR_arch import MFAMambaIR; print('OK')"` 验证模型可导入
- [ ] 用小数据集跑100 iter验证训练流程无报错
- [ ] 检查SwanLab是否正常记录数据
- [ ] 检查GPU显存占用是否在预期范围内
