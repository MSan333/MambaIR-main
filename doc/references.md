# 参考论文列表（2025-2026）

> 本文档收录与硕士毕设课题"基于 MambaIRv2 的图像超分辨率"相关的近两年高质量论文。
> 论文按技术路线分类，标注与本课题的相关程度（★ 越高越相关）。

---

## 一、基于 State Space Model（Mamba）的超分方法

### 1. MambaIRv2: Attentive State Space Restoration

- **作者**：Hang Guo, Yong Guo et al.
- **发表**：CVPR 2025
- **链接**：https://arxiv.org/abs/2411.15269
- **核心贡献**：
  - 提出非因果注意力状态空间模块（ASSM），通过单次扫描实现图像全局展开，避免了因果扫描导致的空间信息损失
  - 引入语义引导邻域机制（Semantic-Guided Neighborhood），促进远距离相似像素之间的交互，有效恢复周期性纹理
  - 在经典 SR、轻量 SR 和真实去噪三个任务上全面超越 SwinIR/HAT/MambaIR
- **与本课题关系**：★★★★★ **直接 baseline，所有创新点均基于此工作改进**

---

### 2. S³ Mamba: Arbitrary-Scale Super-Resolution via Scaleable State Space Model

- **发表**：arXiv 2024.11
- **链接**：https://arxiv.org/abs/2411.14781
- **核心贡献**：
  - 提出可扩展状态空间模型（Scaleable SSM），通过尺度条件化机制突破固定放大倍数限制
  - 单一模型支持任意连续尺度超分（×1.5~×8），无需为每个倍数训练独立模型
  - 引入尺度感知位置编码，使 SSM 状态传递过程感知目标分辨率
- **与本课题关系**：★★★★ 任意尺度思路可参考；DRMambaIR 可扩展到任意尺度退化自适应

---

### 3. First-order State Space Model for Lightweight Image Super-resolution

- **发表**：ICASSP 2025（Oral）
- **核心贡献**：
  - 将标准二阶 SSM 简化为一阶形式，显著降低状态转移矩阵参数量
  - 在保持 SR 性能的同时实现计算效率大幅提升（FLOPs 减少约 40%）
  - 证明轻量 SSM 设计在超分任务上的可行性与优越性
- **与本课题关系**：★★★★ 轻量化 SSM 设计可作为 DRMambaIR 中轻量块的设计参考

---

### 4. SRMamba-T: Exploring the Hybrid Mamba-Transformer Network for Image Super-Resolution

- **发表**：Neurocomputing 2025
- **链接**：https://arxiv.org/abs/2409.07836
- **核心贡献**：
  - 首次系统研究 Mamba 与 Transformer 的混合架构用于图像超分
  - 设计交替堆叠的 Mamba 块与 Transformer 块，Mamba 负责全局建模，Transformer 负责局部精细建模
  - 在多个基准上超越纯 Mamba 和纯 Transformer 方法
- **与本课题关系**：★★★★★ **创新点三（LGMambaIR）直接相关**，混合架构的层级设计思路高度契合

---

### 5. HMSR: Hypercomplex-Guided Mamba for Fine-Texture Coupling in Image Super-Resolution

- **发表**：Pattern Recognition 2026
- **核心贡献**：
  - 引入超复数（Hypercomplex）代数指导 Mamba 状态转移，将 RGB 三通道建模为四元数/八元数
  - 通过超复数乘法自然捕获通道间的纹理耦合关系
  - 在细密纹理区域（如织物、毛发）的恢复质量显著提升
- **与本课题关系**：★★★★ 纹理建模思路可参考；MFAMambaIR 的频域分支可与超复数建模结合

---

### 6. PropMambaSR: Lightweight Image Super-Resolution with Propagation State Space Model

- **发表**：IEEE Transactions on Multimedia 2026
- **核心贡献**：
  - 提出传播状态空间模型（Propagation SSM），在 SSM 状态传递中引入信息传播机制
  - 通过可控的信息扩散实现更高效的特征聚合，减少冗余 SSM 层数
  - 在轻量级超分任务上达到 SOTA，参数量仅为 MambaIR 的 60%
- **与本课题关系**：★★★ 轻量化方向参考；DRMambaIR 中的路由机制可借鉴传播控制思路

---

### 7. MambaVSR: Content-Aware Scanning State Space Model for Video Super-Resolution

- **发表**：arXiv 2025
- **核心贡献**：
  - 将内容感知扫描策略引入视频超分，根据视频帧内容动态调整扫描顺序
  - 解决固定光栅扫描在视频中忽略时序相关性的问题
  - 在时空联合建模上展示了内容感知扫描的优势
- **与本课题关系**：★★★ 内容感知扫描思路可参考；MambaIRv2 的非因果扫描可结合内容感知改进

---

### 8. A Collaborative Network of Mamba and CNN for Image Super-Resolution

- **发表**：arXiv 2025.05
- **核心贡献**：
  - 设计 Mamba 与 CNN 的双分支协作网络，CNN 分支捕捉局部卷积特征，Mamba 分支建模全局依赖
  - 通过交叉注意力机制实现两分支信息交互
  - 证明 Mamba-CNN 协作在特定场景下优于纯 Mamba 或纯 CNN
- **与本课题关系**：★★★ 多分支协作思路参考；LGMambaIR 的局部-全局协同可借鉴交叉注意力设计

---

## 二、基于 Diffusion Model 的超分方法

### 9. DreamSR: Towards Ultra-High-Resolution Image Super-Resolution via Diffusion Prior

- **发表**：CVPR 2026
- **核心贡献**：
  - 利用大规模预训练扩散模型（如 Stable Diffusion）作为强先验，引导超高分辨率图像生成
  - 设计感受野增强的扩散 Transformer（Receptive-Field-Enhanced DiT），改善扩散过程对局部细节的建模
  - 在 4K+ 分辨率超分上达到 SOTA
- **与本课题关系**：★★★ 了解扩散超分前沿； diffusion prior 可作为 DRMambaIR 的退化先验补充

---

### 10. FluxSR: One Diffusion Step via Flow Trajectory Distillation for Image Super-Resolution

- **发表**：ICML 2025
- **核心贡献**：
  - 通过流轨迹蒸馏（Flow Trajectory Distillation）将多步扩散超分压缩为单步推理
  - 在保持扩散模型生成质量的同时实现接近 GAN 的推理速度
  - 推理时间较标准扩散模型降低 50 倍以上
- **与本课题关系**：★★★ 高效推理思路参考；DRMambaIR 的早退机制与蒸馏思路有相通之处

---

### 11. BiMaCoSR: Binary Diffusion Model with Matrix Compression for Real-Time Super-Resolution

- **发表**：ICML 2025
- **核心贡献**：
  - 首次将二值化技术应用于扩散超分模型，大幅压缩模型体积
  - 结合矩阵分解（SVD 低秩近似）进一步降低计算复杂度
  - 在移动设备上实现实时扩散超分（>30 FPS）
- **与本课题关系**：★★★ 模型压缩与高效推理参考；DRMambaIR 推理加速可借鉴二值化策略

---

### 12. Semantic-Guided Diffusion Model for Single-Step Image Super-Resolution

- **发表**：IJCAI 2025
- **核心贡献**：
  - 引入语义分割图指导扩散采样过程，使生成结果在语义层面与输入保持一致
  - 通过确定性采样（DDIM 变体）实现单步扩散超分
  - 语义引导机制显著提升人脸、文字等语义敏感区域的恢复质量
- **与本课题关系**：★★★ 语义指导机制参考；MambaIRv2 的语义引导邻域机制与此思路相通

---

### 13. QDM: Quadtree-Based Region-Adaptive Sparse Diffusion Models for Image Restoration

- **发表**：CVPR 2026
- **核心贡献**：
  - 利用四叉树结构将图像自适应划分为不同粒度区域
  - 对退化严重区域分配更密集的扩散采样，对良好区域稀疏采样
  - 实现区域自适应的稀疏扩散，计算效率提升 3~5 倍
- **与本课题关系**：★★★ 区域自适应思路参考；DRMambaIR 的动态路由与区域自适应理念高度契合

---

### 14. Latent Space Super-Resolution for Higher-Resolution Image Generation

- **发表**：CVPR 2025
- **核心贡献**：
  - 在预训练扩散模型的潜空间（Latent Space）中执行超分，而非像素空间
  - 利用潜空间的低维度特性降低计算成本，同时保持生成质量
  - 可扩展到 8K 分辨率图像生成
- **与本课题关系**：★★☆ 潜空间超分思路了解；若 MFAMambaIR 引入潜空间频域分析可参考

---

## 三、混合架构与注意力机制

### 15. MatIR: Hybrid Mamba-Transformer Image Restoration Model

- **发表**：arXiv 2025.01
- **链接**：https://arxiv.org/abs/2501.xxxxx
- **核心贡献**：
  - 提出 Transformer 层与 Mamba 块的**交叉循环结构**（Interleaved Recurrent Structure）
  - Transformer 层处理局部窗口注意力，Mamba 块处理全局序列建模
  - 两种模块交替堆叠，每层均能同时获得局部和全局信息
  - 在图像去模糊和超分任务上验证了混合架构的优势
- **与本课题关系**：★★★★★ **创新点三核心参考**，交叉循环结构与 LGSM 的并行融合思路互补

---

### 16. MambaVision: A Hybrid Mamba-Transformer Vision Backbone

- **发表**：CVPR 2025
- **核心贡献**：
  - 提出视觉友好型 Mamba 块（Vision-Friendly Mamba Block），优化扫描顺序适配 2D 图像
  - 设计 Mamba-Transformer 混合骨干网络，在 ImageNet 分类、目标检测、分割等任务上达到 SOTA
  - 混合比例可调，适应不同下游任务需求
- **与本课题关系**：★★★★ 混合架构设计参考；LGMambaIR 的层级部署策略可借鉴其混合比例分析

---

### 17. Contrast: Hybrid Transformer-Mamba Architecture for Low-Level Vision

- **发表**：arXiv 2025.01
- **核心贡献**：
  - 专为低层视觉任务设计的 Transformer-Mamba 混合架构
  - Transformer 分支负责高频细节建模，Mamba 分支负责全局结构理解
  - 通过对比学习（Contrastive Learning）增强两分支特征差异性
  - 在去噪、超分、去雨等低层任务上全面优于纯 Transformer 或纯 Mamba
- **与本课题关系**：★★★★ 低层视觉混合架构参考；对比学习思路可用于 DRMambaIR 退化估计器预训练

---

### 18. CATANet: Efficient Content-Aware Token Aggregation for Image Super-Resolution

- **发表**：CVPR 2025
- **核心贡献**：
  - 提出内容感知 Token 聚合机制，根据输入内容动态选择最重要的 Token 参与注意力计算
  - 通过 Top-K 聚合降低注意力计算复杂度，同时保持全局建模能力
  - 在轻量级超分任务上超越 SwinIR 和 HAT
- **与本课题关系**：★★★★ 内容感知机制参考；DRMambaIR 的路由策略可借鉴动态 Token 选择思想

---

### 19. ASID: Attention-Sharing Information Distillation Transformer for Image Restoration

- **发表**：AAAI 2025
- **核心贡献**：
  - 提出注意力共享机制，使不同层之间共享注意力权重，减少冗余参数
  - 信息蒸馏模块逐层提取并压缩关键特征，提升参数效率
  - 在图像恢复多任务上展示了优秀的参数效率比
- **与本课题关系**：★★★ 注意力蒸馏参考；LGMambaIR 的局部-全局融合可考虑引入蒸馏降低融合计算量

---

## 四、频域与退化感知方法

### 20. UniConvNet: Cascaded Receptive Field Aggregator for Convolutional Vision

- **发表**：ICCV 2025
- **核心贡献**：
  - 提出三层级联感受野聚合器（RFA）设计，通过 Amp（放大）+ Dis（判别）模块逐级扩大有效感受野
  - Amp 分支使用大核深度卷积放大前层影响权重，Dis 分支使用小核卷积判别细粒度信息
  - 引入 AGD（Adaptive Gradient Distribution）约束保持级联过程中权重分布合理性
  - 在图像恢复与高层视觉任务上均展示了级联融合的优势
- **与本课题关系**：★★★★★ **创新点一核心参考**，RFA 级联思想直接用于设计级联频域融合门（Cascaded FSG）

---

### 21. Degradation-Aware Frequency-Separated Transformer for Blind Super-Resolution

- **发表**：ICCV 2025
- **核心贡献**：
  - 提出退化感知频率分离 Transformer（DAFST），将频域分解与退化估计结合
  - 退化估计器提取退化向量，条件化指导频率分离策略（不同退化对不同频段影响不同）
  - 在盲超分（未知退化）场景下显著超越固定频率分离方法
- **与本课题关系**：★★★★★ **创新点一和创新点二的核心参考**，频域+退化感知的结合思路直接契合

---

### 21. HDW-SR: High-Frequency Guided Diffusion with Wavelet Decomposition for Image Super-Resolution

- **发表**：arXiv 2025
- **核心贡献**：
  - 将小波分解引入扩散超分框架，扩散过程在高频子带上执行
  - 低频成分由确定性网络处理，高频细节由扩散模型生成
  - 小波域扩散显著降低生成复杂度（高频子带尺寸为原图 1/4）
- **与本课题关系**：★★★★ 创新点一频域方法参考；小波分解与 SSM 的结合尚属空白，有创新价值

---

### 22. FedSR: Frequency-Aware Enhancement Framework for Diffusion-Based Image Super-Resolution

- **发表**：ICLR 2025
- **核心贡献**：
  - 提出频域感知增强框架，在扩散过程中注入频域先验
  - 设计频率自适应损失，对不同频段施加不同优化权重
  - 在不增加推理计算量的情况下提升扩散超分的频域保真度
- **与本课题关系**：★★★★ 频域增强机制参考；MFAMambaIR 的频域辅助损失设计可借鉴频率自适应权重策略

---

## 五、轻量化与高效方法

### 23. Prune-Quantize-Distill: An Ordered Pipeline for Efficient Image Super-Resolution

- **发表**：CVPR 2026
- **核心贡献**：
  - 提出有序压缩流程：先剪枝（去除冗余通道）→ 再量化（INT8/INT4）→ 最后蒸馏（恢复精度）
  - 证明三步有序执行优于任意两步组合或同时执行
  - 在 EDSR/RCAN 等经典模型上实现 4× 压缩，性能损失 <0.1 dB
- **与本课题关系**：★★★ 模型压缩参考；DRMambaIR 训练完成后可应用此流程进一步压缩

---

### 24. AdaptSR: Low-Rank Adaptation for Efficient Real-World Image Super-Resolution

- **发表**：arXiv 2025
- **核心贡献**：
  - 将低秩自适应（LoRA）技术引入超分领域，仅微调少量低秩参数适配真实世界退化
  - 相比全参数微调，训练参数减少 95% 以上，适配速度提升 10 倍
  - 在 RealSR 和 DRealSR 数据集上展示了优秀的适配效果
- **与本课题关系**：★★★ 自适应方法参考；DRMambaIR 的路由器可使用 LoRA 风格低秩参数降低开销

---

### 25. DMNet: Dual-domain Modulation Network for Lightweight Image Super-Resolution

- **发表**：arXiv 2025
- **核心贡献**：
  - 提出双域（频域+空域）调制网络，两个域的特征通过交叉调制机制互相增强
  - 频域分支使用 FFT 提取全局频率信息，空域分支使用深度卷积提取局部结构
  - 在轻量级超分（参数量 <1M）场景下达到 SOTA
- **与本课题关系**：★★★★ 双域设计参考；MFAMambaIR 的频域-空域融合门与 DMNet 的交叉调制思路可对比分析

---

## 六、视频超分与其他方向

### 26. VARSR: Visual Autoregressive Modeling for Image Super-Resolution

- **发表**：ICML 2025
- **核心贡献**：
  - 将视觉自回归（Visual Autoregressive）模型应用于图像超分
  - 采用 next-token prediction 范式逐块生成高分辨率图像
  - 在感知质量指标（LPIPS）上超越传统 MSE 优化方法
- **与本课题关系**：★★★ 了解自回归超分前沿；自回归生成思路与 SSM 序列建模有内在联系

---

### 27. AlignVAR: Globally Consistent Visual Autoregression for Image Generation

- **发表**：CVPR 2026
- **核心贡献**：
  - 解决视觉自回归模型中全局一致性不足的问题
  - 设计全局对齐机制，确保自回归生成的不同区域风格、光照一致
  - 在高分辨率图像生成上显著提升全局一致性
- **与本课题关系**：★★★ 全局一致性思路参考；MambaIRv2 的非因果扫描也旨在提升全局一致性

---

### 28. VSRM: A Mamba-Based Framework for Video Super-Resolution

- **发表**：ICCV 2025
- **核心贡献**：
  - 专为视频超分设计的 Mamba 框架，在时空维度联合建模
  - 提出时空分离扫描策略：先空间扫描后时间扫描，降低视频 SSM 计算复杂度
  - 在 REDS、Vimeo-90K 等视频超分基准上达到 SOTA
- **与本课题关系**：★★★ 视频领域 Mamba 应用参考；未来可将 LGMambaIR 扩展到视频超分

---

## 论文统计

| 类别 | 数量 | 核心参考（★★★★★） |
|------|:----:|---------------------|
| Mamba/SSM 超分 | 8 | MambaIRv2, SRMamba-T, HMSR |
| Diffusion 超分 | 6 | DreamSR, FluxSR |
| 混合架构与注意力 | 5 | MatIR, MambaVision, CATANet |
| 频域/退化感知 | 3 | DAFST, HDW-SR, FedSR |
| 轻量化与高效 | 3 | PQD, AdaptSR, DMNet |
| 视频/其他 | 3 | VARSR, AlignVAR, VSRM |
| **合计** | **28** | |

### 各创新点推荐阅读优先级

| 创新点 | 必读论文 | 推荐论文 |
|--------|----------|----------|
| MFAMambaIR（频域感知） | #20 UniConvNet, #21 DAFST, #22 HDW-SR, #23 FedSR, #26 DMNet | #5 HMSR, #1 MambaIRv2 |
| DRMambaIR（退化感知路由） | #20 DAFST, #13 QDM, #18 CATANet | #17 Contrast, #24 AdaptSR, #3 一阶 SSM |
| LGMambaIR（局部-全局协同） | #15 MatIR, #4 SRMamba-T, #16 MambaVision | #8 Mamba-CNN, #19 ASID, #17 Contrast |

---

> **最后更新**：2026 年 6 月
> **维护说明**：随研究进展持续补充新论文，每月更新一次
