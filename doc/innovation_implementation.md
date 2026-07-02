# MFAMambaIR 三创新点技术实现文档

## 创新点一：多尺度频域感知增强 (MFA)

状态：已实现，正在训练

### 模块文件

- `basicsr/archs/modules/freq_module.py` — MultiScaleFreqSeparator（多尺度频域分离器）
- `basicsr/archs/modules/cfsg.py` — CascadedFSG（级联频域-空域融合门）
- `basicsr/archs/mfamambaIR_arch.py` — MFA_ASSB（集成频域模块的ASSB块）

### 核心原理

- Haar 小波 DWT 做 3 级分解，提取多尺度频域特征
- 级联融合门（CFSG）：3 级 AmpDisModule（7x7, 9x9, 11x11），每级有 Amp（大核DW Conv放大）+ Dis（3x3判别）+ Gate（门控融合）
- 数据流：ASSB 残差路径 → Conv → MultiScaleFreqSeparator → CascadedFSG → PatchEmbed + 残差

### 参考论文

- UniConvNet (ICCV 2025) — 级联RFA设计
- DAFST (ICCV 2025) — 频率分离
- FedSR (ICLR 2025) — 频域感知增强

### 配置参数

```yaml
use_freq: true
num_freq_scales: 3
cfsg_kernels: [7, 9, 11]
freq_deploy_ratio: 0.75
```

---

## 创新点二：纹理引导状态空间调制 (TSSM)

状态：模块已实现，待集成

### 模块文件

- `basicsr/archs/modules/tssm_module.py` — TextureComplexityEstimator (TCE)
- `basicsr/archs/modules/tssm_assm.py` — TSSM_Selective_Scan + TSSM_ASSM

### 核心原理

说明 TCE 的工作原理：

1. 多尺度 DW Conv (3x3 + 5x5) 提取局部梯度特征
2. 1x1 Conv 压缩到 d_state 维度
3. Sigmoid + Scale 到 [0.5, 1.5] 范围
4. 对 SSM 中的 C 矩阵施加乘法调制

状态空间方程：

```
h[t] = A * h[t-1] + B[t] * x[t]     (不受调制)
y[t] = C'[t] * h[t] + D * x[t]
其中 C'[t] = (C[t] + prompt) * texture_mod[t]
```

### 关键设计

- 零初始化策略：compress 卷积权重=0 → sigmoid(0)=0.5 → 调制系数=1.0，确保初始不改变原始行为
- 语义排序对齐：TCE 输出需按 ASSM 的语义排序索引 (x_sort_indices) 重排
- 消融开关：use_tssm=True/False

### 参数量

| embed_dim  | d_state | TCE参数量 |
|------------|---------|-----------|
| 48 (light) | 8       | 3.01K     |
| 174 (full) | 16      | ~12K      |

### 参考论文

- TAMambaIR (IJCAI 2025) — 纹理感知SSM，转移矩阵调制
- PropMambaSR (TMM 2026) — 跨层状态传播
- Rep-Mamba (TGRS 2025) — 跨尺度状态传播
- DPMambaIR (arXiv 2025) — 条件化SSM

---

## 创新点三：密度驱动选择性Token聚合 (DSTA)

状态：模块已实现，待集成

### 模块文件

- `basicsr/archs/modules/dsta_module.py` — TokenDensityEstimator + SpatialTokenAggregator + DSTAWindowAttention

### 核心原理

说明 DSTA 的完整流程：

1. Token密度评估：轻量MLP (Linear→GELU→Linear→Sigmoid) 对 Q 计算密度分数
2. Top-K选择：保留密度最高的 K 个Token的 KV 不变（默认 keep_ratio=0.5，即 128/256）
3. 空间聚合：剩余 Token 按空间位置排序后每 group_size=4 个一组做平均聚合 → (N-K)/4 个代表Token
4. 注意力计算：Q(全分辨率N=256) × KV(聚合后M=160) → 输出仍为全分辨率
5. 相对位置偏差适配：保留Token用原始位置，聚合Token用组代表位置

### KV 压缩比

```
N=256 (原始) → M=160 (聚合后)
  - Top-K保留: 128 tokens
  - 空间聚合: 128/4 = 32 tokens
  - 总计: 128 + 32 = 160 tokens
  - 压缩率: 37.5% 注意力计算量减少
```

### 关键设计

- Drop-in replacement：与原始 WindowAttention 接口完全兼容
- 消融开关：use_dsta=True/False
- Mask适配：shifted window的mask对聚合Token取组内最严格值(min)
- 新增参数：~7.5K（仅密度评估器MLP）

### 参考论文

- SAT (CVPR 2026 Findings) — 密度驱动Token聚合，减少97% KV
- CATANet (CVPR 2025) — 内容感知Token聚合
- TAMambaIR (IJCAI 2025) — 纹理感知计算分配

---

## 集成方式（待实现）

### 单创新点集成

每个创新点可独立启用：

```yaml
# 创新点1: 频域增强
use_freq: true

# 创新点2: 纹理调制
use_tssm: true

# 创新点3: Token聚合
use_dsta: true
```

### 完整三合一架构

三个创新点作用于 MambaIRv2 不同模块，可叠加使用：

```
AttentiveLayer:
  ├── DSTAWindowAttention (创新点3: 替换原 WindowAttention)
  ├── TSSM_ASSM (创新点2: 替换原 ASSM)
  └── ConvFFN (不变)

MFA_ASSB:
  ├── BasicBlock (含修改后的 AttentiveLayer)
  ├── Conv
  ├── MultiScaleFreqSeparator (创新点1)
  └── CascadedFSG (创新点1)
```

### 消融实验方案

| 实验        | MFA | TSSM | DSTA | 目的                          |
|-------------|-----|------|------|-------------------------------|
| Baseline    | ✗   | ✗    | ✗    | MambaIRv2原始结果             |
| +MFA        | ✓   | ✗    | ✗    | 验证频域增强独立有效          |
| +TSSM       | ✗   | ✓    | ✗    | 验证纹理调制独立有效          |
| +DSTA       | ✗   | ✗    | ✓    | 验证Token聚合独立有效         |
| +MFA+TSSM   | ✓   | ✓    | ✗    | 频域+纹理协同                 |
| +MFA+DSTA   | ✓   | ✗    | ✓    | 频域+聚合协同                 |
| Full        | ✓   | ✓    | ✓    | 三合一完整模型                |

---

## 文件结构总览

```
basicsr/archs/modules/
├── __init__.py
├── freq_module.py          # 创新点1: MultiScaleFreqSeparator
├── cfsg.py                 # 创新点1: CascadedFSG
├── tssm_module.py          # 创新点2: TextureComplexityEstimator
├── tssm_assm.py            # 创新点2: TSSM_Selective_Scan + TSSM_ASSM
└── dsta_module.py          # 创新点3: DSTAWindowAttention

basicsr/archs/
├── mambairv2_arch.py       # Baseline (不修改)
├── mambairv2light_arch.py  # Baseline Light (不修改)
└── mfamambaIR_arch.py      # MFA集成架构 (创新点1已集成)
```

---

*更新时间：2026年7月*
