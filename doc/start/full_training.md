# MFAMambaIR 完整版（Classic SR）训练指南

## 模型信息

| 项目 | 值 |
|------|-----|
| 模型 | MFAMambaIR |
| 参数量 | ~26.5M |
| 类型 | Classic SR（大模型） |
| 对标论文 | MambaIRv2-B (22.9M), Table 4 |
| 训练数据 | DIV2K (800张) |
| 总迭代 | 500K iter |
| 单卡4090训练时间 | ~3天 |

---

## 需要超越的论文指标

### MambaIRv2-B (Table 4, DF2K训练)

> 注意：论文用 DF2K (3450张) 训练，你用 DIV2K (800张)。
> 绝对数值会比论文低 0.1-0.3 dB，需用**同配置 baseline 对比**。

#### x4 超分

| 测试集 | MambaIRv2-B PSNR | SSIM |
|--------|------------------|------|
| Set5 | 33.14 | 0.9057 |
| Set14 | 29.23 | 0.7975 |
| BSD100 | 28.00 | 0.7511 |
| Urban100 | 27.89 | 0.8344 |
| Manga109 | 32.57 | 0.9295 |

#### x2 超分

| 测试集 | MambaIRv2-B PSNR | SSIM |
|--------|------------------|------|
| Set5 | 38.65 | 0.9631 |
| Set14 | 34.89 | 0.9275 |
| BSD100 | 32.62 | 0.9053 |
| Urban100 | 34.49 | 0.9468 |
| Manga109 | 40.42 | 0.9810 |

#### x3 超分

| 测试集 | MambaIRv2-B PSNR | SSIM |
|--------|------------------|------|
| Set5 | 35.18 | 0.9334 |
| Set14 | 31.12 | 0.8557 |
| BSD100 | 29.55 | 0.8169 |
| Urban100 | 30.28 | 0.8905 |
| Manga109 | 35.61 | 0.9556 |

---

## 公平对比方法

由于训练数据不同，**正确的对比方式**是：

1. 用相同配置跑 MFAMambaIR → 得到结果 A
2. 用相同配置跑 MambaIRv2-B baseline → 得到结果 B
3. **A - B = 你的创新点贡献**（只要 > 0 即有效）

---

## 训练配置文件

| 尺度 | 配置文件 |
|------|---------|
| x2 | `options/train/mfamambaIR/train_MFAMambaIR_SR_x2.yml` |
| x3 | `options/train/mfamambaIR/train_MFAMambaIR_SR_x3.yml` |
| x4 | `options/train/mfamambaIR/train_MFAMambaIR_SR_x4.yml` |

### 当前 x4 配置概要

```yaml
name: MFAMambaIR_SR_x4
model_type: MambaIRv2Model
scale: 4
num_gpu: 1

network_g:
  type: MFAMambaIR
  embed_dim: 174
  depths: [6, 6, 6, 6, 6, 6]
  num_heads: [6, 6, 6, 6, 6, 6]
  # MFA 创新点参数
  use_freq: true
  num_freq_scales: 3
  cfsg_kernels: [7, 9, 11]
  freq_deploy_ratio: 0.75

datasets:
  train:
    name: DIV2K
    batch_size_per_gpu: 2
    gt_size: 256
    dataset_enlarge_ratio: 100
  val:
    name: Set14
    crop_border: 4
    test_y_channel: true

train:
  total_iter: 500000
  lr: 2e-4
  milestones: [250000, 400000, 450000, 475000]
```

---

## 启动命令

### x4 超分（推荐首先跑）

```bash
screen -S mfa_x4
conda activate sr
cd ~/pyproject/MambaIR-main
export SWANLAB_API_KEY="o4MGQAOSX8rGztH69Jj5P"
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/mfamambaIR/train_MFAMambaIR_SR_x4.yml 2>&1 | tee train_mfa_x4.log
```

### x2 超分

```bash
screen -S mfa_x2
conda activate sr
cd ~/pyproject/MambaIR-main
export SWANLAB_API_KEY="o4MGQAOSX8rGztH69Jj5P"
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/mfamambaIR/train_MFAMambaIR_SR_x2.yml 2>&1 | tee train_mfa_x2.log
```

### x3 超分

```bash
screen -S mfa_x3
conda activate sr
cd ~/pyproject/MambaIR-main
export SWANLAB_API_KEY="o4MGQAOSX8rGztH69Jj5P"
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/mfamambaIR/train_MFAMambaIR_SR_x3.yml 2>&1 | tee train_mfa_x3.log
```

---

## 验证正确性

启动后确认日志显示：

```
Network [MFAMambaIR] is created.
Network: MFAMambaIR, with parameters: 26,541,157
```

如果看到 `Network [MambaIR] is created` 或 `Network [MambaIRv2] is created`，说明用错了配置。

---

## 训练进度监控

### 终端日志

```bash
screen -r mfa_x4
# 或查看日志
tail -f train_mfa_x4.log
```

### SwanLab 面板

https://swanlab.cn/@sansan/MFAMambaIR

### 记录的指标（横坐标为 iter）

| 指标 | 频率 | 说明 |
|------|------|------|
| `l_pix` | 每 200 iter | 训练 loss（细粒度） |
| `train_5k/l_pix` | 每 5000 iter | 训练 loss（粗粒度） |
| `train_5k/lr` | 每 5000 iter | 学习率衰减 |
| `train/elapsed_sec` | 每 5000 iter | 累计训练时间 |
| `val/Set14/psnr` | 每 5000 iter | 验证 PSNR |
| `val/Set14/best_psnr` | 每 5000 iter | 历史最佳 PSNR |

---

## 学习率衰减策略

```
0 ~ 250K iter:     2e-4
250K ~ 400K iter:  1e-4   (×0.5)
400K ~ 450K iter:  5e-5   (×0.5)
450K ~ 475K iter:  2.5e-5 (×0.5)
475K ~ 500K iter:  1.25e-5 (×0.5)
```

---

## 断点续训

训练中断后，直接重新运行相同命令即可自动恢复：

```bash
# 确保配置中有 auto_resume 或 resume_state 指向最新 .state 文件
python basicsr/train.py -opt options/train/mfamambaIR/train_MFAMambaIR_SR_x4.yml
```

训练状态保存在 `experiments/MFAMambaIR_SR_x4/training_states/` 目录下。

---

## Baseline 对照实验

完成 MFAMambaIR 训练后，需用**相同配置**跑 MambaIRv2 baseline：

```bash
# 修改配置文件中 network_g.type 为 MambaIRv2 即可
# 或直接使用:
python basicsr/train.py -opt options/train/mambairv2/train_MambaIRv2_SR_x4.yml
```

注意：baseline 也需要修改为 DIV2K 数据路径和 num_gpu=1。

---

## 预期训练时间线

| 阶段 | iter | 预期 Set14 x4 PSNR | 时间 |
|------|------|-------------------|------|
| 初始 | 5K | ~26.5 | ~1.5h |
| 快速上升 | 50K | ~28.3 | ~7h |
| 趋于稳定 | 200K | ~28.9 | ~1天 |
| 接近收敛 | 400K | ~29.1 | ~2.5天 |
| **最终** | **500K** | **~29.2** | **~3天** |

---

## 实验完成后

1. 记录 MFAMambaIR 在 Set14 上的最终 PSNR
2. 同配置跑 MambaIRv2-B baseline
3. 计算差值 → 写入论文
4. 补充 x2、x3 实验
5. 做消融实验（关闭频域模块对比）
