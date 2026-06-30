# MFAMambaIR 实施报告
## 多尺度频域感知MambaIRv2（融合RFA级联思想）

---

## 一、方案概述

### 1.1 核心目标
在MambaIRv2的ASSM模块基础上，引入多尺度频域分支，通过级联频域融合门(Cascaded FSG)实现频域-空域信息的深度互补，显著提升高频纹理恢复能力。

### 1.2 预期涨点
| 数据集 | Baseline (MambaIRv2) | 预期提升 | 目标PSNR |
|--------|---------------------|---------|----------|
| Set5 (×4) | 32.92 | +0.10~0.15 | 33.02~33.07 |
| Set14 (×4) | 29.10 | +0.10~0.15 | 29.20~29.25 |
| BSD100 (×4) | 27.92 | +0.08~0.12 | 28.00~28.04 |
| Urban100 (×4) | 27.25 | +0.20~0.30 | 27.45~27.55 |
| Manga109 (×4) | 32.15 | +0.15~0.25 | 32.30~32.40 |

### 1.3 涨点原因分析
- Urban100/Manga109含大量高频纹理（建筑线条、漫画细节），频域信息对这类恢复至关重要
- MambaIRv2的ASSM擅长全局建模但对局部高频细节捕获不够充分
- 频域分支提供正交的特征空间，与空域信息互补性强
- RFA级联设计逐层放大有效信息，避免单层融合的信息损失

---

## 二、整体架构设计

### 2.1 网络结构图（文字描述）

```
输入LR图像
    ↓
浅层特征提取 (3×3 Conv)
    ↓
┌─────────────────────────────────────┐
│  RVSSM Block (×N)                    │
│  ┌─────────────────────────────────┐ │
│  │  输入特征 F_in                   │ │
│  │      ↓                          │ │
│  │  ┌────────┐    ┌─────────────┐  │ │
│  │  │空域路径 │    │ 频域路径     │  │ │
│  │  │ (ASSM) │    │ (MFS模块)   │  │ │
│  │  └────┬───┘    └──────┬──────┘  │ │
│  │       ↓               ↓         │ │
│  │  ┌────────────────────────────┐  │ │
│  │  │   级联频域融合门 (CFSG)     │  │ │
│  │  │   Stage1: Amp+Dis (7×7)   │  │ │
│  │  │   Stage2: Amp+Dis (9×9)   │  │ │
│  │  │   Stage3: Amp+Dis (11×11) │  │ │
│  │  └──────────┬─────────────────┘  │ │
│  │             ↓                    │ │
│  │  融合输出 F_out                   │ │
│  └─────────────────────────────────┘ │
│                                      │
│  重复N个Block...                      │
└─────────────────────────────────────┘
    ↓
深层特征提取 (3×3 Conv)
    ↓
上采样模块 (PixelShuffle)
    ↓
输出HR图像
```

### 2.2 关键模块定义

#### A. 多尺度频域特征提取模块 (MFS - Multi-scale Frequency Separator)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleFreqSeparator(nn.Module):
    """多尺度频域特征分离器
    
    使用DWT(离散小波变换)将特征分解为多尺度频域成分
    """
    def __init__(self, channels, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        
        # 小波分解核 (Haar小波)
        self.register_buffer('ll_filter', self._make_haar_filter('ll'))
        self.register_buffer('lh_filter', self._make_haar_filter('lh'))
        self.register_buffer('hl_filter', self._make_haar_filter('hl'))
        self.register_buffer('hh_filter', self._make_haar_filter('hh'))
        
        # 各尺度通道调整
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels * 4, channels, 1),  # 4个子带合并
                nn.GELU(),
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            ) for _ in range(num_scales)
        ])
        
        # 尺度聚合
        self.aggregate = nn.Sequential(
            nn.Conv2d(channels * num_scales, channels, 1),
            nn.GELU()
        )
    
    def _make_haar_filter(self, band):
        """生成Haar小波滤波器"""
        if band == 'll':
            kernel = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        elif band == 'lh':
            kernel = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        elif band == 'hl':
            kernel = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        else:  # hh
            kernel = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        return kernel.unsqueeze(0).unsqueeze(0)
    
    def dwt_2d(self, x):
        """2D离散小波变换"""
        B, C, H, W = x.shape
        # 对每个通道独立做DWT
        x_pad = F.pad(x, (0, W % 2, 0, H % 2))
        
        ll = F.conv2d(x_pad.reshape(B*C, 1, x_pad.shape[2], x_pad.shape[3]), 
                      self.ll_filter, stride=2).reshape(B, C, -1, x_pad.shape[3]//2)
        lh = F.conv2d(x_pad.reshape(B*C, 1, x_pad.shape[2], x_pad.shape[3]), 
                      self.lh_filter, stride=2).reshape(B, C, -1, x_pad.shape[3]//2)
        hl = F.conv2d(x_pad.reshape(B*C, 1, x_pad.shape[2], x_pad.shape[3]), 
                      self.hl_filter, stride=2).reshape(B, C, -1, x_pad.shape[3]//2)
        hh = F.conv2d(x_pad.reshape(B*C, 1, x_pad.shape[2], x_pad.shape[3]), 
                      self.hh_filter, stride=2).reshape(B, C, -1, x_pad.shape[3]//2)
        
        return torch.cat([ll, lh, hl, hh], dim=1)  # [B, 4C, H/2, W/2]
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 输入特征
        Returns:
            freq_features: [B, C, H, W] 多尺度频域特征
        """
        scale_features = []
        current = x
        
        for i in range(self.num_scales):
            # 逐级DWT分解
            decomposed = self.dwt_2d(current)  # [B, 4C, H/2^i, W/2^i]
            
            # 通道调整 + 上采样回原始分辨率
            feat = self.scale_convs[i](decomposed)
            feat = F.interpolate(feat, size=x.shape[2:], mode='bilinear', align_corners=False)
            scale_features.append(feat)
            
            # 下一级使用LL子带
            current = decomposed[:, :x.shape[1], :, :]  # LL子带
        
        # 聚合所有尺度
        multi_scale = torch.cat(scale_features, dim=1)
        freq_features = self.aggregate(multi_scale)
        
        return freq_features
```

#### B. 级联频域融合门 (CFSG - Cascaded Frequency-Spatial Gate)

```python
class AmpDisModule(nn.Module):
    """放大-判别模块 (借鉴UniConvNet RFA思想)
    
    Amp: 放大前层影响权重，扩大有效感受野
    Dis: 判别新尺度信息，保持AGD分布
    """
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        
        # Amplify分支：扩大前层感受野
        self.amp = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=channels),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        
        # Discriminate分支：判别细粒度信息
        self.dis = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        
        # 门控权重
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x_spatial, x_freq):
        """
        Args:
            x_spatial: 空域特征
            x_freq: 频域特征
        Returns:
            fused: 融合特征
        """
        # Amp: 放大空域特征的影响范围
        amp_out = self.amp(x_spatial)
        
        # Dis: 判别频域细节
        dis_out = self.dis(x_freq)
        
        # 门控融合
        gate_weight = self.gate(torch.cat([amp_out, dis_out], dim=1))
        fused = gate_weight * amp_out + (1 - gate_weight) * dis_out
        
        return fused


class CascadedFSG(nn.Module):
    """级联频域-空域融合门
    
    三级AmpDis逐步融合，核从7→9→11渐进扩大
    """
    def __init__(self, channels):
        super().__init__()
        self.stage1 = AmpDisModule(channels, kernel_size=7)
        self.stage2 = AmpDisModule(channels, kernel_size=9)
        self.stage3 = AmpDisModule(channels, kernel_size=11)
        
        # 残差缩放
        self.scale = nn.Parameter(torch.zeros(1))
        
        # AGD约束的可学习温度
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, f_spatial, f_freq):
        """
        Args:
            f_spatial: [B, C, H, W] ASSM输出的空域特征
            f_freq: [B, C, H, W] MFS输出的频域特征
        Returns:
            output: [B, C, H, W] 融合后特征
        """
        # 三级级联融合
        out1 = self.stage1(f_spatial, f_freq)
        out2 = self.stage2(out1, f_freq)
        out3 = self.stage3(out2, f_freq)
        
        # 残差连接 + 可学习缩放
        output = f_spatial + self.scale * out3
        
        return output
```

#### C. MFA-RVSSM Block (完整Block设计)

```python
class MFA_RVSSMBlock(nn.Module):
    """融合多尺度频域感知的RVSSM Block
    
    在原RVSSM Block基础上增加频域分支和CFSG融合
    """
    def __init__(self, dim, depth=1, d_state=16, 
                 use_freq=True, num_freq_scales=3):
        super().__init__()
        self.use_freq = use_freq
        
        # 原ASSM路径（保持不变）
        self.norm1 = nn.LayerNorm(dim)
        self.assm = ASSM(dim=dim, d_state=d_state)  # 原MambaIRv2的ASSM
        
        # 频域路径（新增）
        if use_freq:
            self.freq_separator = MultiScaleFreqSeparator(dim, num_scales=num_freq_scales)
            self.cfsg = CascadedFSG(dim)
        
        # FFN
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, H, W, C] 或 [B, C, H, W]
        Returns:
            out: 与输入相同shape
        """
        B, C, H, W = x.shape
        
        # ASSM路径
        residual = x
        x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        f_spatial = self.assm(x_norm)
        
        # 频域路径 + CFSG融合
        if self.use_freq:
            f_freq = self.freq_separator(x_norm)
            f_fused = self.cfsg(f_spatial, f_freq)
        else:
            f_fused = f_spatial
        
        # 残差连接
        x = residual + f_fused
        
        # FFN
        residual = x
        x_ffn = self.norm2(x.permute(0, 2, 3, 1))
        x_ffn = self.ffn(x_ffn).permute(0, 3, 1, 2)
        x = residual + x_ffn
        
        return x
```

---

## 三、实施步骤详细计划

### 第一阶段：基础模块实现（第1-2周）

#### Week 1: 频域特征提取
- [ ] 实现 `MultiScaleFreqSeparator` 类
- [ ] 单元测试：验证DWT正确性（与pywt库对比）
- [ ] 验证多尺度分解的特征图可视化
- [ ] 确认GPU内存占用（目标：<10%额外增长）

#### Week 2: 级联融合门
- [ ] 实现 `AmpDisModule` 类
- [ ] 实现 `CascadedFSG` 类
- [ ] 单元测试：前向传播shape正确性
- [ ] 梯度检查：确认所有参数有梯度流

### 第二阶段：集成与调试（第3-4周）

#### Week 3: Block级集成
- [ ] 实现 `MFA_RVSSMBlock`
- [ ] 将其替换MambaIRv2中的原RVSSM Block
- [ ] 修改 `mambairv2_arch.py`，增加频域开关配置
- [ ] 小规模验证（DIV2K子集100张，×4，训练10K iter）

#### Week 4: 网络级集成
- [ ] 创建新的网络配置文件 `train_MFAMambaIR_SR_x4.yml`
- [ ] 设置频域分支的部署策略（哪些层用、哪些层不用）
- [ ] 超参初步搜索：
  - num_freq_scales: [2, 3, 4]
  - cfsg_kernel_sizes: [(5,7,9), (7,9,11), (9,11,13)]
  - freq_branch_ratio: [0.25, 0.5, 0.75, 1.0]（频域分支在哪些Block启用）

### 第三阶段：训练与优化（第5-7周）

#### Week 5-6: 正式训练
- [ ] DIV2K + Flickr2K 联合训练
- [ ] 训练配置：
  ```yaml
  batch_size: 32
  patch_size: 64
  total_iter: 500000
  lr: 2e-4 (CosineAnnealing → 1e-6)
  loss: L1Loss
  optimizer: AdamW (weight_decay: 0.01)
  ```
- [ ] 监控训练曲线，检查频域分支是否有效学习
- [ ] 中间checkpoint评估（100K/200K/300K iter）

#### Week 7: 性能调优
- [ ] 如果涨点不足：
  - 尝试FFT替代DWT
  - 调整CFSG级联数量
  - 增加频域Loss（频域L1约束）
- [ ] 如果涨点明显：
  - 减少计算量（裁剪不必要的尺度）
  - 确认模型大小/FLOPs在合理范围

### 第四阶段：消融实验（第8-9周）

#### Week 8: 核心消融
- [ ] 消融实验列表：
  1. w/o 频域分支（纯MambaIRv2 baseline）
  2. w/ 单尺度频域（仅1层DWT）
  3. w/ 多尺度频域，无CFSG（简单加法融合）
  4. w/ 多尺度频域 + 简单Gate（非级联）
  5. w/ 多尺度频域 + CFSG（完整方案）
  6. w/ 多尺度频域 + CFSG + AGD约束
  7. 不同部署位置：仅浅层/仅深层/全部层
  8. 不同核大小：(5,7,9) vs (7,9,11) vs (9,11,13)

#### Week 9: 补充实验
- [ ] 不同超分倍数：×2, ×3, ×4
- [ ] 可视化分析：
  - 频域特征图热力图
  - 有效感受野(ERF)对比
  - 重建结果局部放大对比
- [ ] 计算量对比表（Params/FLOPs/推理时间）

---

## 四、代码修改清单

### 需要新建的文件
```
basicsr/archs/mfamambaIR_arch.py         # 主网络架构
basicsr/archs/modules/freq_module.py      # 频域模块
basicsr/archs/modules/cfsg.py             # 级联融合门
options/train/mfamambaIR/train_MFAMambaIR_SR_x2.yml
options/train/mfamambaIR/train_MFAMambaIR_SR_x3.yml
options/train/mfamambaIR/train_MFAMambaIR_SR_x4.yml
options/test/mfamambaIR/test_MFAMambaIR_SR_x2.yml
options/test/mfamambaIR/test_MFAMambaIR_SR_x3.yml
options/test/mfamambaIR/test_MFAMambaIR_SR_x4.yml
```

### 需要修改的文件
```
basicsr/archs/__init__.py                 # 注册新架构
basicsr/models/__init__.py                # 如需新model类
```

---

## 五、训练配置模板

```yaml
# train_MFAMambaIR_SR_x4.yml
name: MFAMambaIR_SR_x4
model_type: MambaIRv2Model
scale: 4
num_gpu: 1

datasets:
  train:
    name: DIV2K_Flickr2K
    type: PairedImageDataset
    dataroot_gt: datasets/DF2K/HR
    dataroot_lq: datasets/DF2K/LR_bicubic/X4
    io_backend:
      type: disk
    gt_size: 256
    use_hflip: true
    use_rot: true
    num_worker_per_gpu: 8
    batch_size_per_gpu: 8
    dataset_enlarge_ratio: 1

  val:
    name: Set5
    type: PairedImageDataset
    dataroot_gt: datasets/Set5/HR
    dataroot_lq: datasets/Set5/LR_bicubic/X4
    io_backend:
      type: disk

network_g:
  type: MFAMambaIR
  inp_channels: 3
  out_channels: 3
  dim: 180
  num_blocks: [6, 6, 6, 6, 6, 6]
  num_refinement_blocks: 4
  heads: [6, 6, 6, 6, 6, 6]
  ffn_expansion_factor: 2.66
  bias: false
  upscale: 4
  img_size: 64
  # MFA特有参数
  use_freq: true
  num_freq_scales: 3
  cfsg_kernels: [7, 9, 11]
  freq_deploy_ratio: 0.75  # 75%的Block使用频域分支

path:
  pretrain_network_g: ~
  strict_load_g: true

train:
  optim_g:
    type: AdamW
    lr: !!float 2e-4
    weight_decay: 0.01
    betas: [0.9, 0.999]

  scheduler:
    type: CosineAnnealingRestartLR
    periods: [500000]
    restart_weights: [1]
    eta_min: !!float 1e-6

  total_iter: 500000
  warmup_iter: 5000

  pixel_opt:
    type: L1Loss
    loss_weight: 1.0

val:
  val_freq: !!float 5e3
  save_img: false
  metrics:
    psnr:
      type: calculate_psnr
      crop_border: 4
      test_y_channel: true
```

---

## 六、风险与应对

| 风险 | 概率 | 应对措施 |
|------|------|---------|
| DWT计算导致GPU OOM | 中 | 使用gradient checkpointing；减少batch_size至4 |
| 频域分支学不到有用信息 | 中 | 增加频域L1 Loss监督；检查梯度流 |
| CFSG级联过深导致梯度消失 | 低 | 每级残差连接；使用PreNorm |
| 涨点不足(<0.05dB) | 中 | 尝试FFT替代DWT；增加频域注意力机制 |
| 训练不收敛 | 低 | 降低lr至1e-4；冻结ASSM先训练频域分支 |

---

## 七、成功标准

### 最低目标（Pass）
- Urban100 (×4) PSNR提升 ≥ 0.10 dB
- 参数量增加 ≤ 10%
- 消融实验证明各模块有效

### 期望目标（Good）
- Urban100 (×4) PSNR提升 ≥ 0.20 dB
- 所有测试集均有提升
- FLOPs增加 ≤ 15%

### 卓越目标（Excellent）
- Urban100 (×4) PSNR提升 ≥ 0.30 dB
- 超越HAT/DAT等SOTA方法
- 可视化结果具有明显优势
- 投稿顶会论文

---

## 八、关键参考

1. MambaIRv2 (CVPR 2025) - Baseline架构
2. UniConvNet (ICCV 2025) - RFA级联思想来源
3. SwinIR (ICCVW 2021) - 经典Transformer超分
4. HAT (CVPR 2023) - SOTA对标
5. Degradation-Aware Frequency-Separated Transformer (ICCV 2025) - 频域超分参考
6. HDW-SR - 小波超分参考
