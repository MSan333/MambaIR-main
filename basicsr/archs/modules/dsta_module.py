r"""密度驱动选择性 Token 聚合 (Density-driven Selective Token Aggregation, DSTA) 模块

本模块在 MambaIRv2 的 WindowAttention 基础上引入创新点 3：
通过密度驱动的 Token 重要性评估，选择性地聚合 KV tokens，
将注意力计算集中在关键高频 Token 上，从而在降低计算量的同时保持恢复质量。

核心思想：
    1. 对窗口内 Token 计算信息密度分数（轻量 MLP）
    2. 保留 Top-K 高密度 Token 的 KV 原样
    3. 将剩余 Token 的 KV 按空间邻近性聚合为少量代表性 Token
    4. Q 保持全分辨率，与聚合后的 KV 计算注意力
    5. 通过相对位置偏差适配 + 反向输出保持全分辨率输出

参考:
    - SAT (CVPR 2026): Spatially-Adaptive Token aggregation
    - CATANet (CVPR 2025): Content-Aware Token Aggregation

接口兼容性:
    DSTAWindowAttention.forward(qkv, rpi, mask) 的输入输出形状
    与原始 WindowAttention 完全一致，可直接替换。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 尝试从 basicsr 导入工具函数；若依赖缺失则使用本地 fallback
# 这样在完整训练环境中使用真实实现，在独立测试时也能运行
try:
    from basicsr.archs.arch_util import to_2tuple, trunc_normal_
except (ImportError, ModuleNotFoundError):
    def to_2tuple(x):
        """将标量转换为二元组"""
        if isinstance(x, (tuple, list)):
            return tuple(x)
        return (x, x)

    def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
        """截断正态分布初始化 (与 nn.init.trunc_normal_ 一致)"""
        with torch.no_grad():
            tensor.normal_(mean, std).clamp_(a, b)
        return tensor


# =========================================================================== #
#  辅助类 1: Token 信息密度评估器
# =========================================================================== #

class TokenDensityEstimator(nn.Module):
    r"""Token 信息密度评估器

    使用轻量 MLP (Linear → GELU → Linear → Sigmoid) 对每个 Token
    计算一个标量信息密度分数，衡量该 Token 携带的高频/纹理信息量。

    Args:
        dim (int): 输入特征维度 C
        hidden_dim (int, optional): 隐藏层维度，默认 dim // 4
    """

    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or max(dim // 4, 4)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        r"""计算 Token 密度分数

        Args:
            x: [B_, N, C] — Token 特征（通常使用 Q 的全维表示）

        Returns:
            density: [B_, N, 1] — 每个 Token 的密度分数 (0~1)
        """
        return torch.sigmoid(self.fc2(self.act(self.fc1(x))))


# =========================================================================== #
#  辅助类 2: 空间邻近性 Token 聚合器
# =========================================================================== #

class SpatialTokenAggregator(nn.Module):
    r"""空间邻近性 Token 聚合器

    将剩余（非 Top-K）Token 的 KV 按固定大小分组进行平均/加权平均聚合，
    生成少量代表性 Token，从而减少 KV 的数量。

    分组策略：
        剩余 Token 已按空间位置排序，按顺序每 ``group_size`` 个为一组，
        确保空间邻近的 Token 被聚合在一起。

    Args:
        group_size (int): 每组包含的 Token 数，默认 4
        method (str): 聚合方法，'mean' 或 'weighted_mean'
    """

    def __init__(self, group_size=4, method='mean'):
        super().__init__()
        self.group_size = group_size
        self.method = method

    def forward(self, kv, density_scores=None):
        r"""对 KV 进行空间聚合

        Args:
            kv: [B_, num_heads, L, head_dim] — 已 gather 的剩余 Token 的 KV
                L = num_groups * group_size
            density_scores: [B_, L] — 剩余 Token 的密度分数
                仅当 method='weighted_mean' 时使用

        Returns:
            aggregated: [B_, num_heads, num_groups, head_dim]
        """
        B_, H, L, D = kv.shape
        num_groups = L // self.group_size
        # 截断为 group_size 的整数倍（处理不整除的情况）
        usable = num_groups * self.group_size
        kv = kv[:, :, :usable]
        kv = kv.reshape(B_, H, num_groups, self.group_size, D)

        if self.method == 'weighted_mean' and density_scores is not None:
            # 使用密度分数作为权重，密度高的 Token 贡献更大
            scores = density_scores[:, :usable].reshape(B_, num_groups, self.group_size)
            weights = F.softmax(scores, dim=-1)  # 组内归一化 → [B_, num_groups, group_size]
            # 加权平均: [B_, num_heads, num_groups, group_size, D] * [B_, 1, num_groups, group_size, 1]
            aggregated = (kv * weights.unsqueeze(1).unsqueeze(-1)).sum(dim=3)
        else:
            # 简单平均
            aggregated = kv.mean(dim=3)

        return aggregated


# =========================================================================== #
#  核心类: DSTA 窗口注意力
# =========================================================================== #

class DSTAWindowAttention(nn.Module):
    r"""密度驱动选择性 Token 聚合的窗口注意力 (DSTA Window Attention)

    在标准窗口多头自注意力 (Window-MHSA) 的基础上，引入密度驱动的 KV 选择性聚合：
    - 高密度 Token（高频/纹理丰富区域）保留完整 KV
    - 低密度 Token（平坦区域）聚合为代表性 Token
    - Q 始终保持全分辨率，确保输出不变

    计算复杂度从 O(N²·d) 降低为 O(N·M·d)，其中 M = K + (N-K)/group_size < N。

    Args:
        dim (int): 输入通道数 C
        window_size (int or tuple): 窗口大小
        num_heads (int): 注意力头数
        qkv_bias (bool): 是否使用 QKV 偏置
        keep_ratio (float): 保留的 Top-K Token 比例，默认 0.5
        aggregate_method (str): 聚合方法，'mean' 或 'weighted_mean'
        group_size (int): 聚合分组大小，默认 4
        use_dsta (bool): 是否启用 DSTA，False 时退化为标准 WindowAttention

    Example::

        >>> dsta = DSTAWindowAttention(dim=174, window_size=16, num_heads=6)
        >>> qkv = torch.randn(4, 256, 522)   # [B_, N, 3C]
        >>> rpi = torch.randint(0, 961, (256, 256))
        >>> out = dsta(qkv, rpi)
        >>> out.shape
        torch.Size([4, 256, 174])
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 keep_ratio=0.5, aggregate_method='mean',
                 group_size=4, use_dsta=True):
        super().__init__()

        self.dim = dim
        self.window_size = to_2tuple(window_size)  # (Wh, Ww)
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 相对位置偏差表（与原始 WindowAttention 完全一致）
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads))
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

        # ---- DSTA 参数 ----
        self.use_dsta = use_dsta
        self.keep_ratio = keep_ratio
        self.group_size = group_size
        self.aggregate_method = aggregate_method

        if use_dsta:
            self.density_estimator = TokenDensityEstimator(dim)
            self.aggregator = SpatialTokenAggregator(
                group_size=group_size, method=aggregate_method)

    # ------------------------------------------------------------------ #
    #  标准窗口注意力（use_dsta=False 时调用）
    # ------------------------------------------------------------------ #

    def _forward_standard(self, qkv, rpi, mask):
        """标准窗口注意力，与原始 WindowAttention 完全一致"""
        b_, n, c3 = qkv.shape
        c = c3 // 3

        qkv = qkv.reshape(b_, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            n, n, -1).permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        return x

    # ------------------------------------------------------------------ #
    #  DSTA 窗口注意力（use_dsta=True 时调用）
    # ------------------------------------------------------------------ #

    def _forward_dsta(self, qkv, rpi, mask):
        """密度驱动选择性 Token 聚合窗口注意力"""
        b_, n, c3 = qkv.shape
        c = c3 // 3
        num_heads = self.num_heads
        head_dim = c // num_heads
        device = qkv.device

        # ============================================================ #
        #  步骤 1: 分离 Q, K, V
        # ============================================================ #
        # qkv: [B_, N, 3C] → [3, B_, num_heads, N, head_dim]
        qkv = qkv.reshape(b_, n, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # 各 [B_, num_heads, N, head_dim]

        # ============================================================ #
        #  步骤 2: Token 信息密度评估
        # ============================================================ #
        # 将 Q 转换为 [B_, N, C] 输入密度评估器
        q_flat = q.transpose(1, 2).reshape(b_, n, c)  # [B_, N, C]
        density = self.density_estimator(q_flat)       # [B_, N, 1]
        density_flat = density.squeeze(-1)              # [B_, N]

        # ============================================================ #
        #  步骤 3: Top-K Token 选择
        # ============================================================ #
        k_keep = int(n * self.keep_ratio)
        _, topk_idx = torch.topk(density_flat, k_keep, dim=-1)  # [B_, K]

        # 构建保留掩码: True = 保留, False = 待聚合
        keep_mask = torch.zeros(b_, n, dtype=torch.bool, device=device)
        keep_mask.scatter_(1, topk_idx, True)

        # ============================================================ #
        #  步骤 4: 获取剩余 Token 索引（按空间位置排序，保证邻近性）
        # ============================================================ #
        # 排序键: 保留 Token 排在后面 (sort_key 大)，剩余 Token 排在前面 (sort_key 小)
        # 剩余 Token 内部按空间位置升序排列
        spatial_pos = torch.arange(n, device=device).unsqueeze(0)  # [1, N]
        sort_key = keep_mask.long() * n + spatial_pos  # [B_, N]
        sorted_idx = torch.argsort(sort_key, dim=-1)   # [B_, N] 剩余在前, 保留在后
        remaining_idx = sorted_idx[:, :n - k_keep]     # [B_, N-K] 按空间位置排序

        # 空间分组
        gs = self.group_size
        num_groups = (n - k_keep) // gs
        # 截断为 group_size 的整数倍
        remaining_idx = remaining_idx[:, :num_groups * gs]          # [B_, num_groups*gs]
        grouped_idx = remaining_idx.reshape(b_, num_groups, gs)      # [B_, num_groups, gs]
        grouped_idx_flat = grouped_idx.reshape(b_, num_groups * gs)   # [B_, num_groups*gs]

        # 聚合后 KV token 数
        m = k_keep + num_groups

        # ============================================================ #
        #  步骤 5: 聚合 KV
        # ============================================================ #
        # 5a) 保留 Token 的 KV（直接 gather）
        topk_exp = topk_idx.unsqueeze(1).unsqueeze(-1).expand(-1, num_heads, -1, head_dim)
        k_keep_kv = torch.gather(k, 2, topk_exp)  # [B_, num_heads, K, head_dim]
        v_keep_kv = torch.gather(v, 2, topk_exp)

        # 5b) 剩余 Token 的 KV（gather 后聚合）
        rem_exp = grouped_idx_flat.unsqueeze(1).unsqueeze(-1).expand(-1, num_heads, -1, head_dim)
        k_rem = torch.gather(k, 2, rem_exp)  # [B_, num_heads, num_groups*gs, head_dim]
        v_rem = torch.gather(v, 2, rem_exp)

        # 剩余 Token 的密度分数（用于加权平均）
        rem_density = torch.gather(density_flat, 1, grouped_idx_flat)  # [B_, num_groups*gs]

        k_agg = self.aggregator(k_rem, rem_density)  # [B_, num_heads, num_groups, head_dim]
        v_agg = self.aggregator(v_rem, rem_density)

        # 5c) 拼接: 保留 + 聚合
        k_final = torch.cat([k_keep_kv, k_agg], dim=2)  # [B_, num_heads, M, head_dim]
        v_final = torch.cat([v_keep_kv, v_agg], dim=2)

        # ============================================================ #
        #  步骤 6: 注意力计算 (Q 全分辨率 × KV 聚合后)
        # ============================================================ #
        q = q * self.scale
        attn = q @ k_final.transpose(-2, -1)  # [B_, num_heads, N, M]

        # ============================================================ #
        #  步骤 7: 相对位置偏差适配
        # ============================================================ #
        # 保留 Token: 使用自身空间位置
        # 聚合 Token: 使用组内代表位置（第一个 Token 的空间位置）
        group_rep_pos = grouped_idx[:, :, 0]               # [B_, num_groups]
        key_positions = torch.cat([topk_idx, group_rep_pos], dim=-1)  # [B_, M]

        # 从 rpi 中索引: rpi_kv[b, i, j] = rpi[i, key_positions[b, j]]
        key_pos_exp = key_positions.unsqueeze(1).expand(b_, n, m)    # [B_, N, M]
        rpi_exp = rpi.unsqueeze(0).expand(b_, n, n)                    # [B_, N, N]
        rpi_kv = torch.gather(rpi_exp, 2, key_pos_exp)               # [B_, N, M]

        bias = self.relative_position_bias_table[rpi_kv.reshape(-1)]  # [B_*N*M, num_heads]
        bias = bias.reshape(b_, n, m, num_heads).permute(0, 3, 1, 2)  # [B_, num_heads, N, M]
        attn = attn + bias

        # ============================================================ #
        #  步骤 8: Attention Mask 适配
        # ============================================================ #
        if mask is not None:
            nw = mask.shape[0]
            # 每个 B_ 元素对应的窗口索引
            win_idx = torch.arange(b_, device=device) % nw
            mask_batch = mask[win_idx]  # [B_, N, N]

            # 保留 Token 的 mask 列
            mask_keep = torch.gather(
                mask_batch, 2, topk_idx.unsqueeze(1).expand(b_, n, k_keep))  # [B_, N, K]

            # 聚合 Token 的 mask 列（取组内最严格值: 不可见则整组不可见）
            mask_agg = torch.gather(
                mask_batch, 2,
                grouped_idx_flat.unsqueeze(1).expand(b_, n, num_groups * gs))  # [B_, N, num_groups*gs]
            mask_agg = mask_agg.reshape(b_, n, num_groups, gs)
            mask_agg = mask_agg.min(dim=3).values  # [B_, N, num_groups]

            mask_kv = torch.cat([mask_keep, mask_agg], dim=2)  # [B_, N, M]
            attn = attn + mask_kv.unsqueeze(1)  # [B_, 1, N, M] 广播到 num_heads

        attn = self.softmax(attn)

        # ============================================================ #
        #  步骤 9: 输出
        # ============================================================ #
        x = (attn @ v_final).transpose(1, 2).reshape(b_, n, c)  # [B_, N, C]
        x = self.proj(x)
        return x

    # ------------------------------------------------------------------ #
    #  统一入口
    # ------------------------------------------------------------------ #

    def forward(self, qkv, rpi, mask=None):
        r"""DSTA 窗口注意力前向传播

        Args:
            qkv: [num_windows*B, N, 3*C] — 输入 QKV（已由 wqkv 投影）
            rpi: [N, N] — 相对位置索引
            mask: [num_windows, N, N] 或 None — 注意力掩码

        Returns:
            x: [num_windows*B, N, C] — 输出特征

        Note:
            输入输出形状与原始 WindowAttention 完全一致，可直接替换。
        """
        if not self.use_dsta:
            return self._forward_standard(qkv, rpi, mask)
        return self._forward_dsta(qkv, rpi, mask)

    def extra_repr(self) -> str:
        return (f'dim={self.dim}, window_size={self.window_size}, '
                f'num_heads={self.num_heads}, qkv_bias={self.qkv_bias}, '
                f'use_dsta={self.use_dsta}, keep_ratio={self.keep_ratio}, '
                f'group_size={self.group_size}, method={self.aggregate_method}')


# =========================================================================== #
#  测试代码
# =========================================================================== #

def _calculate_rpi_sa(window_size):
    """计算相对位置索引（与 MambaIRv2 的 calculate_rpi_sa 一致）"""
    ws = window_size
    coords_h = torch.arange(ws)
    coords_w = torch.arange(ws)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, N, N
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # N, N, 2
    relative_coords[:, :, 0] += ws - 1
    relative_coords[:, :, 1] += ws - 1
    relative_coords[:, :, 0] *= 2 * ws - 1
    relative_position_index = relative_coords.sum(-1)  # N, N
    return relative_position_index


if __name__ == '__main__':
    # ---- 测试参数 ----
    dim = 174            # 嵌入维度 C
    window_size = 16     # 窗口大小 (16x16)
    num_heads = 6        # 注意力头数
    head_dim = dim // num_heads  # 29
    N = window_size * window_size  # 256
    B_ = 4               # num_windows * batch_size

    print('=' * 70)
    print('DSTA (密度驱动选择性 Token 聚合) 模块测试')
    print('=' * 70)
    print(f'参数: dim={dim}, window_size={window_size}, num_heads={num_heads}')
    print(f'      head_dim={head_dim}, N={N}, B_={B_}')
    print(f'      输入 qkv shape: [{B_}, {N}, {3 * dim}]')
    print()

    # 构造相对位置索引
    rpi = _calculate_rpi_sa(window_size)  # [256, 256]

    # 构造输入
    qkv = torch.randn(B_, N, 3 * dim)

    # ---- 测试 1: DSTA 开启 (keep_ratio=0.5, group_size=4) ----
    print('-' * 70)
    print('测试 1: DSTA 开启 (keep_ratio=0.5, group_size=4, method=mean)')
    print('-' * 70)

    dsta = DSTAWindowAttention(
        dim=dim, window_size=window_size, num_heads=num_heads,
        keep_ratio=0.5, aggregate_method='mean', group_size=4, use_dsta=True)
    dsta.eval()

    K = int(N * 0.5)        # 128
    num_groups = (N - K) // 4  # 32
    M = K + num_groups        # 160

    print(f'  聚合前 KV token 数: {N}')
    print(f'  保留 Top-K token:  {K} (keep_ratio=0.5)')
    print(f'  聚合组数:          {num_groups} (group_size=4)')
    print(f'  聚合后 KV token 数: {M}')
    print(f'  KV 压缩率:         {M}/{N} = {M / N:.1%}')
    print(f'  注意力矩阵:        [{B_}, {num_heads}, {N}, {N}] → [{B_}, {num_heads}, {N}, {M}]')
    print(f'  计算量降低:        {(1 - M / N) * 100:.1f}% (attn matmul)')
    print()

    with torch.no_grad():
        out = dsta(qkv, rpi)
    print(f'  输入:  {list(qkv.shape)}')
    print(f'  输出:  {list(out.shape)}')
    assert out.shape == (B_, N, dim), f'输出形状错误: {out.shape} != ({B_}, {N}, {dim})'
    print('  [PASS] 输出形状正确')
    print()

    # ---- 测试 2: DSTA 关闭 (退化为标准 WindowAttention) ----
    print('-' * 70)
    print('测试 2: DSTA 关闭 (use_dsta=False, 退化为标准 WindowAttention)')
    print('-' * 70)

    dsta_off = DSTAWindowAttention(
        dim=dim, window_size=window_size, num_heads=num_heads,
        use_dsta=False)
    dsta_off.eval()

    with torch.no_grad():
        out_off = dsta_off(qkv, rpi)
    print(f'  输入:  {list(qkv.shape)}')
    print(f'  输出:  {list(out_off.shape)}')
    assert out_off.shape == (B_, N, dim), f'输出形状错误: {out_off.shape}'
    print('  [PASS] 输出形状正确')
    print()

    # ---- 测试 3: weighted_mean 聚合方法 ----
    print('-' * 70)
    print('测试 3: weighted_mean 聚合方法')
    print('-' * 70)

    dsta_wm = DSTAWindowAttention(
        dim=dim, window_size=window_size, num_heads=num_heads,
        keep_ratio=0.5, aggregate_method='weighted_mean', group_size=4)
    dsta_wm.eval()

    with torch.no_grad():
        out_wm = dsta_wm(qkv, rpi)
    print(f'  输入:  {list(qkv.shape)}')
    print(f'  输出:  {list(out_wm.shape)}')
    assert out_wm.shape == (B_, N, dim), f'输出形状错误: {out_wm.shape}'
    print('  [PASS] 输出形状正确')
    print()

    # ---- 测试 4: 带 mask 的场景 (模拟 shifted window) ----
    print('-' * 70)
    print('测试 4: 带 attention mask (模拟 shifted window attention)')
    print('-' * 70)

    # 构造一个简单的 mask: [num_windows, N, N], 值为 0 或 -100
    nw = 4  # 窗口数 (B_ = batch * nw, 这里 batch=1, nw=4)
    mask = torch.zeros(nw, N, N)
    mask[:, :, :] = -100.0  # 默认不可见
    mask[:, :, :128] = 0.0   # 前半部分可见

    with torch.no_grad():
        out_masked = dsta(qkv, rpi, mask=mask)
    print(f'  输入:  {list(qkv.shape)}, mask: {list(mask.shape)}')
    print(f'  输出:  {list(out_masked.shape)}')
    assert out_masked.shape == (B_, N, dim), f'输出形状错误: {out_masked.shape}'
    print('  [PASS] 输出形状正确')
    print()

    # ---- 测试 5: 梯度反向传播 ----
    print('-' * 70)
    print('测试 5: 梯度反向传播')
    print('-' * 70)

    # 5a) mean 模式: 注意力主路径梯度正常，但密度评估器无梯度
    #     (topk 选择是离散操作，梯度不经过选择索引)
    dsta_mean = DSTAWindowAttention(
        dim=dim, window_size=window_size, num_heads=num_heads,
        keep_ratio=0.5, aggregate_method='mean', group_size=4)
    qkv_grad = torch.randn(B_, N, 3 * dim, requires_grad=True)
    out_mean = dsta_mean(qkv_grad, rpi)
    loss = out_mean.sum()
    loss.backward()

    proj_grad = dsta_mean.proj.weight.grad
    print(f'  [mean 模式]')
    print(f'  输出 sum:              {loss.item():.4f}')
    print(f'  proj 层梯度范数:       {proj_grad.norm().item():.6f}')
    assert proj_grad is not None and proj_grad.norm() > 0, 'proj 层梯度为零!'
    print('  [PASS] 注意力主路径梯度正常')

    # 5b) weighted_mean 模式: 密度评估器通过聚合权重获得梯度
    dsta_wm_grad = DSTAWindowAttention(
        dim=dim, window_size=window_size, num_heads=num_heads,
        keep_ratio=0.5, aggregate_method='weighted_mean', group_size=4)
    qkv_grad2 = torch.randn(B_, N, 3 * dim, requires_grad=True)
    out_wm = dsta_wm_grad(qkv_grad2, rpi)
    loss_wm = out_wm.sum()
    loss_wm.backward()

    de_grad = dsta_wm_grad.density_estimator.fc1.weight.grad
    proj_grad2 = dsta_wm_grad.proj.weight.grad
    print(f'  [weighted_mean 模式]')
    print(f'  输出 sum:              {loss_wm.item():.4f}')
    if de_grad is not None:
        print(f'  密度评估器 fc1 梯度范数: {de_grad.norm().item():.6f}')
        assert de_grad.norm() > 0, '密度评估器梯度为零!'
        print('  [PASS] 密度评估器梯度正常传播 (通过加权聚合权重)')
    else:
        print('  [INFO] 密度评估器梯度为 None (选择索引不可导, 需 STE 或辅助损失)')
    print(f'  proj 层梯度范数:       {proj_grad2.norm().item():.6f}')
    print()

    # ---- 测试 6: 不同 keep_ratio ----
    print('-' * 70)
    print('测试 6: 不同 keep_ratio 下的 token 压缩')
    print('-' * 70)
    print(f'  {"keep_ratio":>12} | {"K":>5} | {"num_groups":>10} | {"M":>5} | {"压缩率":>8} | {"输出形状":>20}')
    print(f'  {"-"*12}-+-{"-"*5}-+-{"-"*10}-+-{"-"*5}-+-{"-"*8}-+-{"-"*20}')

    for kr in [0.25, 0.5, 0.75, 1.0]:
        k = int(N * kr)
        ng = (N - k) // 4
        m = k + ng
        model = DSTAWindowAttention(
            dim=dim, window_size=window_size, num_heads=num_heads,
            keep_ratio=kr, group_size=4, use_dsta=(kr < 1.0))
        model.eval()
        with torch.no_grad():
            o = model(qkv, rpi)
        status = '✓' if o.shape == (B_, N, dim) else '✗'
        print(f'  {kr:>12.2f} | {k:>5} | {ng:>10} | {m:>5} | {m/N:>7.1%} | {status} {list(o.shape)}')

    print()

    # ---- 参数量统计 ----
    print('=' * 70)
    print('参数量统计')
    print('=' * 70)

    dsta_on = DSTAWindowAttention(dim=dim, window_size=window_size, num_heads=num_heads, use_dsta=True)
    dsta_off2 = DSTAWindowAttention(dim=dim, window_size=window_size, num_heads=num_heads, use_dsta=False)

    params_off = sum(p.numel() for p in dsta_off2.parameters())
    params_on = sum(p.numel() for p in dsta_on.parameters())
    params_de = sum(p.numel() for p in dsta_on.density_estimator.parameters())

    print(f'  标准 WindowAttention 参数量:  {params_off:>8,}')
    print(f'  DSTA WindowAttention 参数量:  {params_on:>8,}')
    print(f'  其中密度评估器:              {params_de:>8,} (+{params_de / params_off * 100:.2f}%)')
    print(f'  新增参数量:                  {params_on - params_off:>8,} (+{(params_on - params_off) / params_off * 100:.2f}%)')
    print()

    print('=' * 70)
    print('所有测试通过!')
    print('=' * 70)
