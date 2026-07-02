# 毕业设计文档

## 课题信息

| 项目 | 内容 |
|------|------|
| **课题方向** | 图像超分辨率（Image Super-Resolution） |
| **Baseline 模型** | MambaIRv2（CVPR 2025） |
| **核心技术** | 状态空间模型（SSM）、ASSM 注意力机制、非因果建模 |
| **学位类型** | 硕士毕业设计 |

---

## 文档索引

| 文档 | 说明 | 内容概览 |
|------|------|----------|
| [innovation_plan.md](./innovation_plan.md) | 三个创新点详细规划 | 动机、技术方案、预期效果、实现计划、风险评估 |
| [references.md](./references.md) | 2025-2026 参考论文列表 | 28 篇论文，按技术路线分类，标注相关度 |

---

## 创新点概览

### 创新点一：MFAMambaIR — 多尺度频域感知 MambaIRv2

- **核心问题**：MambaIRv2 在空间域建模，对高频纹理细节捕获能力有限
- **解决思路**：引入 DWT/FFT 频域分支 + 频域-空域融合门（FSG）
- **技术亮点**：频域感知门机制自适应融合频域/空域特征；语义引导邻域扩展至频域
- **预期提升**：PSNR +0.15~0.25 dB（高纹理数据集），参数增加 <3%
- **参考论文**：
  - UniConvNet: Cascaded Receptive Field Aggregator for Convolutional Vision (ICCV 2025) — 级联融合门 CFSG 的核心灵感来源
  - DAFST: Degradation-Aware Frequency-Separated Transformer for Blind SR (ICCV 2025) — 频率分离+退化感知结合
  - FedSR: Frequency-Aware Enhancement Framework for Diffusion-Based SR (ICLR 2025) — 频域感知增强框架
  - DMNet: Dual-domain Modulation Network for Lightweight SR (arXiv 2025) — 双域（频+空）交叉调制
  - HDW-SR: High-Frequency Guided Diffusion with Wavelet Decomposition (arXiv 2025) — 小波分解引导高频恢复

### 创新点二：TSSMambaIR — 纹理引导状态空间调制 MambaIRv2

- **核心问题**：MambaIRv2 的 ASSM 使用固定状态转移矩阵，对所有空间位置施加相同的状态衰减，无法区分纹理复杂区域与平滑区域
- **解决思路**：设计轻量纹理复杂度估计器（TCE），对 ASSM 的状态转移矩阵 A 进行空间自适应调制
- **技术亮点**：纹理区域保持强记忆（慢衰减），平滑区域快速遗忘（快衰减）；仅增加 <1% 参数
- **预期提升**：PSNR +0.05~0.15 dB
- **参考论文**：
  - TAMambaIR: Texture-Aware State Space Model for Image Restoration (IJCAI 2025) — 纹理感知SSM，调制转移矩阵的核心参考
  - PropMambaSR: Lightweight SR with Propagation State Space Model (TMM 2026) — 跨层隐藏状态路由
  - Rep-Mamba: Re-Parameterization in Vision Mamba for SR (TGRS 2025) — 跨尺度状态传播
  - DPMambaIR: All-in-One Restoration via Degradation-Aware Prompt SSM (arXiv 2025) — 条件化SSM参考

### 创新点三：DSTAMambaIR — 密度驱动选择性Token聚合增强 MambaIRv2

- **核心问题**：MambaIRv2 的窗口注意力对所有 Token 一视同仁计算 QKV，纹理区域的关键 Token 与平滑区域冗余 Token 获得相同计算资源
- **解决思路**：密度驱动的 Token 重要性评估 + 选择性 KV 聚合，Q 保持全分辨率
- **技术亮点**：关键 Token 获得精细注意力，冗余 Token 聚合后获得粗粒度注意力；同时提升效果和效率
- **预期提升**：PSNR +0.05~0.15 dB，FLOPs 降低 15~25%
- **参考论文**：
  - SAT: Selective Aggregation Transformer for Image Super-Resolution (CVPR 2026 Findings) — 密度驱动Token聚合，核心参考
  - CATANet: Efficient Content-Aware Token Aggregation for SR (CVPR 2025) — 内容感知Token聚合
  - TAMambaIR: Texture-Aware State Space Model (IJCAI 2025) — 纹理感知计算分配理念

---

## 技术关键词

`State Space Model` `Mamba` `ASSM` `非因果建模` `频域分析` `小波变换` `FFT` `纹理感知` `状态调制` `转移矩阵` `选择性聚合` `Token重要性` `密度驱动` `混合架构`

---

## 目录结构

```
doc/
├── README.md              # 本文档（索引与概览）
├── innovation_plan.md     # 三个创新点详细规划
└── references.md          # 参考论文列表（28篇）
```

---

> **最后更新**：2026 年 7 月
