# TSSM 创新点失败分析与替代决策

> 创建时间：2026年7月

## 一、背景

TSSM（纹理引导状态空间调制）是原规划中的创新点2，核心思想是通过纹理复杂度估计器（TCE）生成调制系数，对 SSM 的 C 矩阵进行乘法调制，使纹理区域获得更强的输出权重。

## 二、实验数据

### 2.1 各创新点组合的 Set14 PSNR 结果

| 实验 | MFA | TSSM | DSTA | Set14 PSNR (dB) | 训练进度 |
|------|-----|------|------|-----------------|----------|
| Baseline (MambaIRv2) | ✗ | ✗ | ✗ | 28.84 | 500k (完整) |
| +MFA | ✓ | ✗ | ✗ | **28.8633** | 500k (完整) |
| +MFA+DSTA | ✓ | ✗ | ✓ | **28.8601** | 380k (进行中) |
| +MFA+DSTA+TSSM (Full) | ✓ | ✓ | ✓ | **28.50** | 500k (完整) |

### 2.2 关键发现

1. **MFA 单独有效**：28.86 vs baseline 28.84，提升 +0.02 dB
2. **MFA+DSTA 有效**：28.86@380k，仍有 120k 迭代未完成，预计最终结果会继续提升
3. **加入 TSSM 后严重退化**：28.50 vs MFA+DSTA 的 28.86，**下降 0.36 dB**，甚至低于单独 MFA

### 2.3 TSSM 的训练曲线特征

- 收敛速度明显变慢
- 最终 PSNR 无法突破 28.50
- Loss 曲线在后期出现震荡，无法稳定收敛
- 最佳 PSNR 在训练早期就已达到，后续迭代未能进一步提升

## 三、失败原因深度分析

### 3.1 根本原因：破坏 SSM 状态动力学

SSM 的核心状态空间方程：

```
h[t] = A · h[t-1] + B[t] · x[t]     (状态转移)
y[t] = C · h[t]   + D · x[t]       (输出投影)
```

TSSM 将输出方程修改为：

```
y[t] = C'[t] · h[t] + D · x[t]
其中 C'[t] = C[t] × texture_mod[t]   (纹理调制)
```

这导致以下问题：

1. **A/B/C/D 协同学习被破坏**：SSM 的四个矩阵在训练中需要协同优化，A 矩阵控制状态转移的动力学特性，B/C 矩阵控制输入/输出的投影方式。外部强制调制 C 矩阵会打乱 A-B-C 之间已建立的平衡关系，导致状态传递不稳定。

2. **语义排序后的调制错位**：ASSM 的语义路由（semantic-guided neighborhood）会对输入序列进行重排序，TSSM 的 TCE 输出需要按排序索引重排才能对齐。这个重排过程引入了额外的对齐误差，尤其在语义边界处调制系数可能错位。

3. **调制范围不可控**：TCE 的调制系数范围为 [0.5, 1.5]，虽然零初始化时为 1.0（中性），但训练过程中调制系数会偏离 1.0。对于纹理区域调制系数 >1 会放大 C 矩阵输出，可能导致状态空间的数值不稳定。

### 3.2 次要原因

1. **梯度路径复杂化**：TCE 的梯度需要通过 selective_scan 的 C 矩阵反传，这条梯度路径经过 SSM 内部的非线性变换，容易产生梯度消失或不稳定。

2. **DDP 兼容性问题**：TSSM 引入了动态参数路径（TCE 的输出依赖输入），导致部分参数在某些迭代中未被使用，需要额外启用 `find_unused_parameters=True`，增加了通信开销。

3. **与 MFA 的潜在冲突**：MFA 在残差路径做频域增强，改变了输入到 ATTBlock 的特征分布。TSSM 的 TCE 对这个改变后的特征做纹理估计，但 TCE 是用 3x3+5x5 DWConv 设计的，可能无法正确估计频域增强后的特征纹理。

## 四、经验教训

### 4.1 核心原则

> **创新点不能修改 SSM 的内部机制（A/B/C/D 矩阵）**

SSM 的状态空间方程是一个数学上严格耦合的系统，A/B/C/D 矩阵之间存在隐式的协同关系。任何对其中某个矩阵的外部调制都可能破坏这种协同性。

### 4.2 安全的增强位置

在 MambaIRv2 的 ATTBlock 中，有以下安全的外部增强位置：

```
ATTBlock:
  ├── Part1: norm1 → wqkv → WindowAttention [DSTA安全✓] → convffn1 [CGFFN安全✓]
  └── Part2: norm3 → ASSM [不可修改✗] → convffn2 [CGFFN安全✓]

ASSB:
  ├── BasicBlock (含 ATTBlock)
  ├── Conv
  ├── MultiScaleFreqSeparator [MFA安全✓]
  └── CascadedFSG [MFA安全✓]
```

安全位置总结：
- **残差路径**：MFA 的频域增强（已验证有效）
- **注意力路径**：DSTA 的 Token 聚合（已验证有效）
- **前馈路径**：ConvFFN 的增强（CGFFN 方案）
- **不安全**：SSM 内部的 A/B/C/D 矩阵

### 4.3 创新点设计准则

1. **正交性**：每个创新点应作用于网络的不同路径/模块，避免相互干扰
2. **外部性**：增强应在外部添加，不修改基础模块的内部机制
3. **可消融**：通过独立开关控制，支持任意组合的消融实验
4. **轻量化**：参数量增加 <5%，不显著影响推理速度
5. **有依据**：基于近期高水平论文的方法论，而非凭空设计

## 五、替代方案决策

基于上述分析，决定弃用 TSSM，采用 **SSDPS（语义-空间双路径扫描）** 作为新的创新点2。

- 详细设计见：`doc/ssdps_innovation_design.md`
- 代码实现：`basicsr/archs/modules/ssdps_assm.py`（待实现）
- 灵感来源：PRISMamba (ICML 2026) + SP-MoMamba (ICML 2026) + MaIR (CVPR 2025)
- 核心思想：不修改 SSM 的 A/B/C/D 矩阵，而是在 hidden 通道维度分割为两路，
  一路保持语义排序扫描（原始 SGN），一路使用空间光栅排序扫描，
  互补融合以同时捕获语义关联和空间连续性

## 六、文件变更记录

| 操作 | 文件 | 说明 |
|------|------|------|
| 保留 | `basicsr/archs/modules/tssm_module.py` | 保留代码用于消融对比，默认关闭 |
| 保留 | `basicsr/archs/modules/tssm_assm.py` | 同上 |
| 新增 | `basicsr/archs/modules/ssdps_assm.py` | SSDPS_ASSM 模块实现（待实现） |
| 修改 | `basicsr/archs/mfamambaIRv2_arch.py` | ATTBlock 中新增 `use_ssdps` 开关 |
| 修改 | `options/train/mfamambaIRv2/train_MFAMambaIRv2_lightSR_x4.yml` | 配置更新 |

---

*文档结束*
