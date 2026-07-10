"""SSDPS: Semantic-Spatial Dual-Path Scan ASSM

在 MambaIRv2 的 ASSM 基础上, 将 hidden 通道维度分割为两路:
  - Path A (语义路径): 保持 SGN 语义排序扫描 + ASE 提示
  - Path B (空间路径): 光栅顺序扫描, 无提示, 保持 2D 空间连续性

不修改 SSM 的 A/B/C/D 矩阵, 仅改变送入 SSM 的 token 排列顺序。

消融开关:
    use_ssdps=True  (默认): 启用双路径扫描
    use_ssdps=False         : 退化为原始 ASSM

接口兼容:
    forward(self, x, x_size, token) → [B, HW, C]
    与原始 ASSM 完全相同, 可作为 drop-in 替换

依赖:
    - basicsr.archs.mambairv2_arch.Selective_Scan (基类)
    - basicsr.archs.mambairv2_arch.index_reverse, semantic_neighbor (语义排序)

参考论文:
    - PRISMamba: Partial Ring Scan (ICML 2026) — 证明扫描顺序 critical 影响性能
    - SP-MoMamba (ICML 2026) — 指出 1D 扫描是 SSM 在 SR 中的根本问题
    - MaIR (CVPR 2025) — S 形扫描保持空间连续性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 从 MambaIRv2 导入基础组件 (完全复用, 不做修改)
from basicsr.archs.mambairv2_arch import (
    Selective_Scan,
    index_reverse,
    semantic_neighbor,
)


class SSDPS_ASSM(nn.Module):
    """Semantic-Spatial Dual-Path Scan ASSM

    替换原始 ASSM, 在 hidden 通道维度分割为两路:
    - Path A (语义路径): SGN 语义排序扫描 + ASE 提示 (与原始 ASSM 一致)
    - Path B (空间路径): 光栅顺序扫描, 零提示, 保持 2D 空间连续性

    两路各处理半通道, 总参数量和计算量与原始 ASSM 相当。

    Args:
        dim (int): 输入通道数 C
        d_state (int): SSM 状态维度
        input_resolution (tuple): 输入分辨率 (H, W)
        num_tokens (int): 语义 token 数, 默认 64
        inner_rank (int): ASSM 内部秩, 默认 128
        mlp_ratio (float): 隐藏维度扩展比, 默认 2.0
        ssdps_ratio (float): 语义路径通道占比, 默认 0.5 (各半)
    """

    def __init__(self, dim, d_state, input_resolution, num_tokens=64,
                 inner_rank=128, mlp_ratio=2., ssdps_ratio=0.5):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank

        # Mamba params
        self.expand = mlp_ratio
        hidden = int(self.dim * self.expand)
        self.d_state = d_state

        # 通道分割
        self.hidden_sem = int(hidden * ssdps_ratio)   # 语义路径通道数
        self.hidden_spa = hidden - self.hidden_sem     # 空间路径通道数
        assert self.hidden_sem > 0 and self.hidden_spa > 0, \
            f"hidden_sem={self.hidden_sem}, hidden_spa={self.hidden_spa} must be > 0"

        # 共享组件 (与原始 ASSM 一致)
        self.out_norm = nn.LayerNorm(hidden)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden, dim, bias=True)

        self.in_proj = nn.Sequential(
            nn.Conv2d(self.dim, hidden, 1, 1, 0),
        )

        self.CPE = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden),
        )

        self.embeddingB = nn.Embedding(self.num_tokens, self.inner_rank)
        self.embeddingB.weight.data.uniform_(-1 / self.num_tokens, 1 / self.num_tokens)

        self.route = nn.Sequential(
            nn.Linear(self.dim, self.dim // 3),
            nn.GELU(),
            nn.Linear(self.dim // 3, self.num_tokens),
            nn.LogSoftmax(dim=-1)
        )

        # 双路径 SSM (各半尺寸)
        self.selectiveScan_sem = Selective_Scan(
            d_model=self.hidden_sem, d_state=self.d_state, expand=1)
        self.selectiveScan_spa = Selective_Scan(
            d_model=self.hidden_spa, d_state=self.d_state, expand=1)

    def forward(self, x, x_size, token):
        """
        Args:
            x: [B, HW, C] 输入序列 (空间光栅顺序)
            x_size: (H, W)
            token: nn.Embedding, token.weight 为 [inner_rank, d_state]

        Returns:
            [B, HW, C] 输出序列 (空间光栅顺序)
        """
        B, n, C = x.shape
        H, W = x_size

        # === 语义路由 (仅 Path A 使用) ===
        full_embedding = self.embeddingB.weight @ token.weight  # [num_tokens, d_state]

        pred_route = self.route(x)  # [B, HW, num_tokens]
        cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)  # [B, HW, num_tokens]

        prompt = torch.matmul(cls_policy, full_embedding).view(B, n, self.d_state)  # [B, HW, d_state]

        detached_index = torch.argmax(cls_policy.detach(), dim=-1, keepdim=False).view(B, n)  # [B, HW]
        x_sort_values, x_sort_indices = torch.sort(detached_index, dim=-1, stable=False)
        x_sort_indices_reverse = index_reverse(x_sort_indices)

        # === 特征投影 (与原始 ASSM 一致) ===
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x = self.in_proj(x)               # [B, hidden, H, W]
        x = x * torch.sigmoid(self.CPE(x))  # CPE gating
        cc = x.shape[1]                    # hidden
        x = x.view(B, cc, -1).contiguous().permute(0, 2, 1)  # [B, HW, hidden]

        # === 通道分割 ===
        x_sem, x_spa = torch.split(x, [self.hidden_sem, self.hidden_spa], dim=-1)

        # === Path A: 语义排序扫描 ===
        # SGN-unfold: 按语义排序重排
        semantic_x = semantic_neighbor(x_sem, x_sort_indices)  # [B, HW, hidden_sem]
        # 选择性扫描 (带 ASE 提示)
        y_sem = self.selectiveScan_sem(semantic_x, prompt)    # [B, HW, hidden_sem]
        # SGN-fold: 恢复空间顺序
        y_sem = semantic_neighbor(y_sem, x_sort_indices_reverse)  # [B, HW, hidden_sem]

        # === Path B: 空间光栅顺序扫描 ===
        # 无需排序, x_spa 已是空间光栅顺序
        zero_prompt = torch.zeros(B, n, self.d_state, device=x.device, dtype=x.dtype)
        y_spa = self.selectiveScan_spa(x_spa, zero_prompt)  # [B, HW, hidden_spa]
        # 无需逆排序, y_spa 已是空间顺序

        # === 通道融合 ===
        y = torch.cat([y_sem, y_spa], dim=-1)  # [B, HW, hidden]
        y = self.out_proj(self.out_norm(y))    # [B, HW, dim]

        return y
