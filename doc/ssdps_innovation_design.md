# SSDPS: 语义-空间双路径扫描 — 新创新点2设计文档

> 创建时间：2026年7月
> 状态：设计中，待实现
> 替代方案：TSSM（已失败，见 `doc/tssm_failure_analysis.md`）

## 一、研究动机

### 1.1 问题定位

MambaIRv2 的核心创新 ASSM（Attentive State Space Module）使用**语义引导邻域（SGN）**对 token 序列进行重排序：通过 Gumbel-Softmax 将每个像素分配到语义类别，然后按类别排序后送入 selective_scan。

这一设计的**关键缺陷**：

1. **破坏 2D 空间连续性**：语义排序后，空间相邻但语义不同的 token 在扫描序列中可能距离很远。SSM 的状态传递 h[t]→h[t+1] 是序列相邻的，语义排序导致 SSM 无法建立局部空间依赖。

2. **平滑区域退化**：对于天空、墙壁等大面积平滑区域，所有 token 被分配到同一语义类，在序列中聚集在一起。但它们在空间上跨度很大，SSM 的状态在这些 token 间传递时缺乏局部纹理的增量信息。

3. **单一视角局限**：MambaIRv2 仅使用语义排序这一种扫描策略，无法同时兼顾语义关联和空间连续性。

### 1.2 论文支撑（2026 ICML 顶会）

**PRISMamba: Partial Ring Scan (ICML 2026)**
- 明确证明：扫描顺序"critically affects performance by altering spatial adjacency, fracturing object continuity"
- 指出固定路径扫描在旋转变换下性能下降 1~2%
- 提出 ring scan + partial channel filtering 来缓解空间连续性问题

**SP-MoMamba (ICML 2026)**
- 指出："standard 1D scanning is a fundamental issue in applying Mamba-based SSMs to SR"
- 提出超像素驱动的多专家 SSM 来解决扫描顺序问题

两篇 ICML 2026 论文均证实：**扫描顺序是 Vision SSM 的核心瓶颈**，而 MambaIRv2 的语义排序恰恰牺牲了空间连续性。

### 1.3 与其他创新点的关系

| 创新点 | 作用位置 | 增强维度 | 状态 |
|--------|----------|----------|------|
| MFA（创新点1） | ASSB 残差路径 | 频率域感知 | 已验证 28.86 dB |
| **SSDPS（新创新点2）** | **ASSM SSM 扫描路径** | **扫描顺序优化** | 待实现 |
| DSTA（创新点3） | ATTBlock 注意力路径 | 计算效率 | 已验证 28.86 dB@380k |

三个创新点分别针对**残差路径、SSM 扫描路径、注意力路径**，完全正交，可独立消融。

## 二、参考论文

### 2.1 核心参考（2026 顶会）

| 论文 | 年份/会议 | 核心贡献 | 与 SSDPS 的关系 |
|------|----------|---------|----------------|
| **PRISMamba: Partial Ring Scan** | **ICML 2026** | 环形扫描 + 部分通道过滤，证明扫描顺序破坏物体连续性 | **动机支撑**：SGN 语义排序同样破坏空间连续性 |
| **SP-MoMamba** | **ICML 2026** | 超像素驱动多专家 SSM，指出 1D 扫描是 SR 的根本问题 | **动机支撑**：扫描顺序是 SR 中 SSM 的核心瓶颈 |

### 2.2 辅助参考（2025 顶会）

| 论文 | 年份/会议 | 核心贡献 | 与 SSDPS 的关系 |
|------|----------|---------|----------------|
| MaIR | CVPR 2025 | 嵌套 S 形扫描（NSS）保持局部性和连续性 | **方法支撑**：空间连续性扫描的可行性 |
| 2D-CrossScan Mamba | AAAI 2025 | 多路径 2D 空间扫描 + 隐藏状态传播 | **方法支撑**：双路径扫描聚合策略 |

### 2.3 新颖性分析

| 方法 | 语义排序 | 空间排序 | 双路径 | 适用场景 |
|------|---------|---------|--------|---------|
| MambaIRv2 (CVPR 2025) | ✓ | ✗ | ✗ | 通用图像恢复 |
| MaIR (CVPR 2025) | ✗ | ✓ (S形) | ✗ | 通用图像恢复 |
| 2D-CrossScan (AAAI 2025) | ✗ | ✓ (多方向) | ✗ | 通用视觉 |
| SP-MoMamba (ICML 2026) | ✗ | ✓ (超像素) | ✗ | 轻量 SR |
| PRISMamba (ICML 2026) | ✗ | ✓ (环形) | ✗ | 图像分类 |
| **SSDPS（本方案）** | **✓** | **✓ (光栅)** | **✓** | **轻量 SR** |

**SSDPS 是首个结合语义排序 + 空间排序的双路径扫描方案。**

## 三、技术方案

### 3.1 原始 ASSM 数据流（MambaIRv2）

```
输入 x [B, HW, C]
  │
  ├── 语义路由: route(x) → Gumbel-Softmax → cls_policy [B, HW, num_tokens]
  ├── 提示生成: prompt = cls_policy × full_embedding → [B, HW, d_state]
  ├── 排序索引: sort(cls_policy.argmax()) → x_sort_indices
  │
  ├── 特征投影: x → reshape(B,C,H,W) → in_proj(C→hidden) → CPE → reshape(B,HW,hidden)
  │
  ├── SGN-unfold: semantic_neighbor(x, x_sort_indices) → semantic_x [B, HW, hidden]
  ├── 选择性扫描: y = selectiveScan(semantic_x, prompt) → [B, HW, hidden]
  ├── SGN-fold: x = semantic_neighbor(y, x_sort_indices_reverse) → [B, HW, hidden]
  │
  └── 输出: out_norm(y) → out_proj(hidden→dim) → [B, HW, C]
```

**问题**：selective_scan 仅在语义排序的序列上运行一次，丢失空间连续性。

### 3.2 SSDPS-ASSM 数据流

```
输入 x [B, HW, C]
  │
  ├── 语义路由: route(x) → Gumbel-Softmax → cls_policy [B, HW, num_tokens]
  ├── 提示生成: prompt = cls_policy × full_embedding → [B, HW, d_state]
  ├── 排序索引: sort(cls_policy.argmax()) → x_sort_indices
  │
  ├── 特征投影: x → reshape(B,C,H,W) → in_proj(C→hidden) → CPE → reshape(B,HW,hidden)
  │
  ├── 通道分割: x_sem, x_spa = split(x, dim=-1, split_size=hidden//2)
  │
  ├── Path A (语义路径, hidden/2):
  │     ├── SGN-unfold: semantic_neighbor(x_sem, x_sort_indices) → semantic_x [B, HW, hidden/2]
  │     ├── 选择性扫描: y_sem = selectiveScan_A(semantic_x, prompt) → [B, HW, hidden/2]
  │     └── SGN-fold: y_sem = semantic_neighbor(y_sem, x_sort_indices_reverse) → [B, HW, hidden/2]
  │
  ├── Path B (空间路径, hidden/2):
  │     ├── 无需排序: x_spa 已是空间光栅顺序 [B, HW, hidden/2]
  │     ├── 选择性扫描: y_spa = selectiveScan_B(x_spa, zero_prompt) → [B, HW, hidden/2]
  │     └── 无需逆排序: y_spa 已是空间顺序 [B, HW, hidden/2]
  │
  ├── 通道融合: y = concat(y_sem, y_spa, dim=-1) → [B, HW, hidden]
  │
  └── 输出: out_norm(y) → out_proj(hidden→dim) → [B, HW, C]
```

### 3.3 关键设计决策

#### 3.3.1 为什么在 hidden 维度分割而非复制？

| 方案 | 通道操作 | 计算量 | 参数量 | 风险 |
|------|---------|--------|--------|------|
| 复制（双路全通道） | hidden → 2×hidden | 2×原始 | 2×原始 | 计算翻倍 |
| **分割（各半通道）** | **hidden → 2×(hidden/2)** | **≈原始** | **≈原始** | **信息分流** |

分割方案保持总计算量和参数量与原始 ASSM 相当。两个半尺寸 SSM 的参数量之和 ≈ 一个全尺寸 SSM。

#### 3.3.2 为什么 Path B 不使用语义提示（prompt）？

MambaIRv2 的 prompt（ASE 机制）是通过语义路由计算的，Path B 不走语义路由，因此：

- **方案1（推荐）**：Path B 的 prompt 设为全零，即 `Cs = Cs + 0 = Cs`，退化为标准 SSM（无 ASE）
- 方案2：复用 Path A 的 prompt（需按空间顺序重排），增加对齐复杂度

选择方案1的理由：
- Path A 已通过 ASE 捕获语义关联，Path B 专注空间连续性，两者职责分离
- 零 prompt 使 Path B 更接近原始 Mamba 行为，训练更稳定
- 实现简单，无需额外的 prompt 重排

#### 3.3.3 为什么 Path B 使用光栅顺序而非 S 形 / Hilbert / 环形？

- **光栅顺序**：最简单，天然保持 2D 空间连续性（H×W 展平），与 MambaIR 原始扫描一致
- S 形扫描（MaIR）：更优的空间连续性，但需要额外的 stripe 分割和 shift-stripe 机制
- Hilbert 曲线：理论最优的空间局部性，但索引计算复杂
- 环形扫描（PRISMamba）：为分类任务设计，SR 中不适用

光栅顺序足以解决核心问题（空间连续性），且实现成本最低。后续可消融对比不同空间扫描策略。

### 3.4 消融开关设计

```yaml
network_g:
  use_ssdps: true          # 启用 SSDPS 双路径扫描
  use_tssm: false          # TSSM 已弃用（保持 false）
  ssdps_ratio: 0.5         # 语义路径通道占比（默认 0.5 即各半）
```

- `use_ssdps=false`：使用原始 ASSM（baseline 行为）
- `use_ssdps=true`：使用 SSDPS_ASSM 替换 ASSM，通道对半分割
- `ssdps_ratio`：可调节语义/空间路径的通道分配比例

## 四、集成方案

### 4.1 ATTBlock 中的位置

```
MFAMambaIRv2_ATTBlock:
  Part1: x → norm1 → wqkv → DSTAWindowAttention → convffn1 → scale1
  Part2: x → norm3 → [SSDPS]ASSM → convffn2 → scale2
```

SSDPS 替换 ATTBlock 中的 ASSM 实例，接口完全兼容。

### 4.2 代码集成点

| 操作 | 文件 | 说明 |
|------|------|------|
| **新增** | `basicsr/archs/modules/ssdps_assm.py` | SSDPS_ASSM 模块实现 |
| **修改** | `basicsr/archs/modules/__init__.py` | 注册 SSDPS_ASSM |
| **修改** | `basicsr/archs/mfamambaIRv2_arch.py` | ATTBlock 新增 `use_ssdps` 开关 |
| **修改** | `options/train/mfamambaIRv2/train_MFAMambaIRv2_lightSR_x4.yml` | 配置更新 |

### 4.3 接口设计

```python
class SSDPS_ASSM(nn.Module):
    """Semantic-Spatial Dual-Path Scan ASSM

    替换原始 ASSM，在 hidden 通道维度分割为两路：
    - Path A (语义路径): 保持 SGN 语义排序扫描 + ASE 提示
    - Path B (空间路径): 光栅顺序扫描，无提示，保持 2D 空间连续性

    接口与原始 ASSM 完全兼容:
        forward(x, x_size, token) → [B, HW, C]

    Args:
        dim (int): 输入通道数 C
        d_state (int): SSM 状态维度
        input_resolution (tuple): 输入分辨率
        num_tokens (int): 语义 token 数
        inner_rank (int): 内部秩
        mlp_ratio (float): 隐藏维度扩展比（默认 2）
        ssdps_ratio (float): 语义路径通道占比（默认 0.5）
    """
```

### 4.4 SSDPS_ASSM 内部结构

```python
class SSDPS_ASSM(nn.Module):
    def __init__(self, dim, d_state, input_resolution, num_tokens=64,
                 inner_rank=128, mlp_ratio=2., ssdps_ratio=0.5):
        super().__init__()
        self.dim = dim
        hidden = int(dim * mlp_ratio)
        self.hidden_sem = int(hidden * ssdps_ratio)   # 语义路径通道数
        self.hidden_spa = hidden - self.hidden_sem     # 空间路径通道数

        # 共享组件（与原始 ASSM 一致）
        self.in_proj = nn.Conv2d(dim, hidden, 1, 1, 0)
        self.CPE = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)
        self.out_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, dim, bias=True)
        self.act = nn.SiLU()

        # 语义路由（与原始 ASSM 一致）
        self.embeddingB = nn.Embedding(num_tokens, inner_rank)
        self.route = nn.Sequential(
            nn.Linear(dim, dim // 3),
            nn.GELU(),
            nn.Linear(dim // 3, num_tokens),
            nn.LogSoftmax(dim=-1)
        )

        # 双路径 SSM（各半尺寸）
        self.selectiveScan_sem = Selective_Scan(
            d_model=self.hidden_sem, d_state=d_state, expand=1)
        self.selectiveScan_spa = Selective_Scan(
            d_model=self.hidden_spa, d_state=d_state, expand=1)

    def forward(self, x, x_size, token):
        B, n, C = x.shape
        H, W = x_size

        # === 语义路由（仅 Path A 使用）===
        full_embedding = self.embeddingB.weight @ token.weight
        pred_route = self.route(x)
        cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)
        prompt = torch.matmul(cls_policy, full_embedding).view(B, n, self.d_state)

        detached_index = torch.argmax(cls_policy.detach(), dim=-1, keepdim=False).view(B, n)
        x_sort_values, x_sort_indices = torch.sort(detached_index, dim=-1, stable=False)
        x_sort_indices_reverse = index_reverse(x_sort_indices)

        # === 特征投影 ===
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x = self.in_proj(x)
        x = x * torch.sigmoid(self.CPE(x))
        x = x.view(B, x.shape[1], -1).contiguous().permute(0, 2, 1)  # [B, HW, hidden]

        # === 通道分割 ===
        x_sem, x_spa = torch.split(x, [self.hidden_sem, self.hidden_spa], dim=-1)

        # === Path A: 语义排序扫描 ===
        semantic_x = semantic_neighbor(x_sem, x_sort_indices)      # SGN-unfold
        y_sem = self.selectiveScan_sem(semantic_x, prompt)          # selective scan
        y_sem = semantic_neighbor(y_sem, x_sort_indices_reverse)   # SGN-fold

        # === Path B: 空间光栅顺序扫描 ===
        zero_prompt = torch.zeros(B, n, self.d_state, device=x.device, dtype=x.dtype)
        y_spa = self.selectiveScan_spa(x_spa, zero_prompt)          # 直接光栅顺序扫描

        # === 通道融合 ===
        y = torch.cat([y_sem, y_spa], dim=-1)  # [B, HW, hidden]
        y = self.out_proj(self.out_norm(y))

        return y
```

### 4.5 初始化策略

- in_proj / CPE / out_proj / route / embeddingB：与原始 ASSM 完全一致（复用相同初始化）
- selectiveScan_sem / selectiveScan_spa：各自独立初始化（A_logs, Ds, dt_projs 等）
  - 与原始 ASSM 的 selectiveScan 使用相同的初始化函数
- 训练初期：两路 SSM 各自独立学习，由于 out_norm + out_proj 的存在，融合是平滑的
- 无需特殊零初始化策略（两路都是有效的 SSM 扫描，不存在"中性"需求）

## 五、参数量与复杂度分析

### 5.1 参数量对比

以 light 配置 (dim=48, mlp_ratio=2, hidden=96, d_state=16) 为例：

| 组件 | 原始 ASSM | SSDPS_ASSM | 差异 |
|------|-----------|------------|------|
| in_proj | 48×96 = 4,608 | 48×96 = 4,608 | 0 |
| CPE | 96×9 = 864 | 96×9 = 864 | 0 |
| selectiveScan | ~6,000 | ~6,000 (2×~3,000) | 0 |
| out_proj | 96×48 = 4,608 | 96×48 = 4,608 | 0 |
| route | ~1,200 | ~1,200 | 0 |
| embeddingB | 2,048 | 2,048 | 0 |
| **总计** | **~19,328** | **~19,328** | **≈0** |

注：Selective_Scan 参数与 d_model 线性相关，两个 d_model=48 的 SSM 参数量之和 ≈ 一个 d_model=96 的 SSM。

### 5.2 计算复杂度

| 操作 | 原始 ASSM | SSDPS_ASSM |
|------|-----------|------------|
| in_proj + CPE | O(C×hidden×HW) | 相同 |
| selective_scan | O(hidden×L×d_state) | O(hidden/2×L×d_state) × 2 = 相同 |
| out_proj | O(hidden×C×HW) | 相同 |
| **总计** | **基准** | **≈基准** |

语义排序/逆排序的开销在 Path A 中保持不变，Path B 省去了排序步骤。

### 5.3 显存对比

SSDPS 的两路 SSM 各处理半通道，中间激活的显存占用与原始 ASSM 相当。

## 六、消融实验设计

### 6.1 消融矩阵

| 实验 | MFA | SSDPS | DSTA | TSSM | Set14 PSNR | 目标 |
|------|-----|-------|------|------|------------|------|
| Baseline | ✗ | ✗ | ✗ | ✗ | 28.84 (已知) | 基准 |
| +MFA | ✓ | ✗ | ✗ | ✗ | 28.86 (已知) | MFA 独立效果 |
| +MFA+DSTA | ✓ | ✗ | ✓ | ✗ | ~28.88 (预期) | MFA+DSTA |
| +MFA+SSDPS | ✓ | ✓ | ✗ | ✗ | >28.88 (预期) | SSDPS 替代TSSM |
| +MFA+SSDPS+DSTA | ✓ | ✓ | ✓ | ✗ | **>28.90 (目标)** | 三合一 |
| +MFA+TSSM+DSTA | ✓ | ✗ | ✓ | ✓ | 28.50 (已知) | TSSM 对照（失败） |

### 6.2 SSDPS 内部消融

| 变体 | Path A (语义) | Path B (空间) | 验证目标 |
|------|-------------|-------------|---------|
| 仅语义 (原始) | ✓ | ✗ | 退化到 MambaIRv2 baseline |
| 仅空间 | ✗ | ✓ | 纯空间扫描的效果 |
| SSDPS (双路径) | ✓ | ✓ | 双路径互补增益 |

### 6.3 空间扫描策略消融（可选）

| Path B 扫描策略 | 说明 |
|----------------|------|
| 光栅顺序 (默认) | H×W 展平，最简单 |
| S 形顺序 | MaIR 风格，交替行方向 |
| Hilbert 曲线 | 理论最优空间局部性 |

## 七、实现计划

| 步骤 | 任务 | 优先级 | 预估工时 |
|------|------|--------|---------|
| 1 | 实现 `ssdps_assm.py` 模块 | 高 | 2h |
| 2 | 修改 ATTBlock 集成 `use_ssdps` 开关 | 高 | 1h |
| 3 | 更新 YAML 配置（use_ssdps=true, use_tssm=false） | 高 | 0.5h |
| 4 | 前向传播验证：shape + 梯度反传 | 中 | 1h |
| 5 | 启动 MFA+SSDPS+DSTA 三合一训练 | 高 | 0.5h |
| 6 | 训练监控与结果对比 | 中 | 持续 |

## 八、预期效果

| 指标 | 预期 |
|------|------|
| +MFA+SSDPS+DSTA vs Baseline | +0.08~0.15 dB |
| +MFA+SSDPS+DSTA vs +MFA+DSTA | +0.04~0.08 dB |
| +MFA+SSDPS+DSTA vs +MFA+TSSM+DSTA (失败) | +0.40 dB 以上 |
| 参数量变化 | ≈0%（与原始 ASSM 持平） |
| 推理速度影响 | <3% |
| 视觉效果 | 空间连续区域（天空、墙壁）更平滑，纹理边界更清晰 |

## 九、硕士论文故事线

### 9.1 三创新点叙事结构

> 本文基于 MambaIRv2 (CVPR 2025) 提出三个正交创新点，从频率感知、SSM 扫描策略、注意力效率三个维度全面增强轻量级超分辨率：
>
> **创新点1 MFA**（频率域感知）：Haar 小波多尺度频域分离 + 级联融合门，增强高频纹理恢复。作用于 ASSB 残差路径。
>
> **创新点2 SSDPS**（SSM 扫描优化）：针对 MambaIRv2 语义引导邻域（SGN）破坏 2D 空间连续性的问题，提出语义-空间双路径扫描。在 hidden 通道维度分割特征，语义路径保持 SGN 语义排序捕获语义关联，空间路径使用光栅排序保持 2D 空间连续性。两路各处理半通道，总计算量不变。**ICML 2026 两篇论文（PRISMamba, SP-MoMamba）均证实扫描顺序是 Vision SSM 的核心瓶颈。**
>
> **创新点3 DSTA**（注意力效率）：密度驱动 Token 聚合 + 空间分组聚合，减少窗口注意力的 KV 计算冗余。作用于 ATTBlock 注意力路径。

### 9.2 创新点2 的论文方法论

**问题分析**：MambaIRv2 的 SGN 按语义相似性重排序 token，虽然捕获了语义关系，但破坏了 2D 空间连续性。ICML 2026 的 PRISMamba 证明"扫描顺序 critical 影响性能，因为它破坏了物体连续性"，SP-MoMamba 指出"1D 扫描是 SSM 在 SR 中的根本问题"。

**解决方案**：SSDPS 在通道维度将 hidden 特征分割为两路：
- 语义路径（Path A）：保持 MambaIRv2 原始的 SGN 语义排序 + ASE 提示
- 空间路径（Path B）：使用空间光栅顺序，无 ASE 提示，保持 2D 空间连续性

**安全性保证**：SSDPS 不修改 SSM 的 A/B/C/D 矩阵（TSSM 的失败原因），仅改变送入 SSM 的 token 排列顺序。两路均为标准 selective_scan 调用。

**效率保证**：两路各处理半通道，总参数量和计算量与原始 ASSM 持平。

---

*文档结束*
