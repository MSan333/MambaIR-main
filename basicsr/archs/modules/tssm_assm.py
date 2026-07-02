"""TSSM_ASSM: 纹理引导状态空间调制的 ASSM

在 MambaIRv2 的 ASSM 基础上, 集成纹理复杂度估计器 (TCE),
对 Selective_Scan 中的 C 矩阵进行乘法调制, 使纹理区域获得更强的输出权重。

修改点 (相比原始 ASSM):
    1. 在 ASSM forward 中, 将 2D 特征图送入 TCE 估计纹理复杂度
    2. 将纹理调制系数按语义排序索引重排 (与特征排序一致)
    3. 在 TSSM_Selective_Scan 的 forward_core 中, 对 C 矩阵施加乘法调制

消融开关:
    use_tssm=True  (默认): 启用纹理调制
    use_tssm=False         : 退化为原始 ASSM (不使用 TCE, 使用原始 Selective_Scan)

接口兼容:
    forward(self, x, x_size, token) → [B, HW, C]
    与原始 ASSM 完全相同, 可作为 drop-in 替换

依赖:
    - basicsr.archs.mambairv2_arch.Selective_Scan (基类)
    - basicsr.archs.mambairv2_arch.index_reverse, semantic_neighbor (语义排序)
    - basicsr.archs.modules.tssm_module.TextureComplexityEstimator (TCE)
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
from basicsr.archs.modules.tssm_module import TextureComplexityEstimator


class TSSM_Selective_Scan(Selective_Scan):
    """带纹理引导调制的 Selective_Scan

    继承自原始 Selective_Scan, 仅修改 forward_core 和 forward,
    在 C 矩阵计算后施加纹理调制系数。其余所有参数和初始化逻辑完全继承父类。

    状态空间方程:
        h[t] = A * h[t-1] + B[t] * x[t]      (状态转移, 不受调制)
        y[t] = C'[t] * h[t] + D * x[t]       (输出投影)
        其中 C'[t] = (C[t] + prompt) * texture_mod[t]  (纹理引导调制)

    调制效果:
        - 纹理复杂区域: texture_mod > 1 → C 矩阵输出权重增强
        - 平滑区域: texture_mod ≈ 1 → 保持原始行为
    """

    def forward_core(self, x, prompt, texture_mod=None):
        """选择性扫描核心计算 (带纹理调制)

        与原始 Selective_Scan.forward_core 相比, 唯一区别:
        在 Cs + prompt 之后, 对 Cs 施加乘法纹理调制。

        Args:
            x: [B, L, C] 输入序列 (已按语义排序)
            prompt: [B, 1, d_state, L] 语义提示 (ASE)
            texture_mod: [B, 1, d_state, L] 纹理调制系数
                        None 时不调制 (退化为原始行为)

        Returns:
            out_y: [B, C, L] 输出序列
        """
        B, L, C = x.shape
        K = 1  # mambairV2 仅需 1 次扫描
        xs = x.permute(0, 2, 1).view(B, 1, C, L).contiguous()  # B, 1, C, L

        x_dbl = torch.einsum("b k d l, k c d -> b k c l",
                             xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l",
                           dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)

        # ASE: 语义提示加到 C 矩阵 (与原始代码一致)
        Cs = Cs.float().view(B, K, -1, L) + prompt  # (b, k, d_state, l)

        # ======== TSSM: 纹理引导调制 ========
        # 对 C 矩阵进行乘法调制, 使纹理区域获得更强的输出权重
        # Cs: [B, K=1, d_state, L]
        # texture_mod: [B, 1, d_state, L] → 广播乘法
        if texture_mod is not None:
            Cs = Cs * texture_mod.float()
        # =====================================

        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x, prompt, texture_mod=None, **kwargs):
        """前向传播

        Args:
            x: [B, L, C] 输入序列 (已按语义排序)
            prompt: [B, L, d_state] 语义提示
            texture_mod: [B, 1, d_state, L] 纹理调制系数
                        None 时不调制 (退化为原始行为)

        Returns:
            y: [B, L, C] 输出序列
        """
        b, l, c = prompt.shape
        prompt = prompt.permute(0, 2, 1).contiguous().view(b, 1, c, l)
        y = self.forward_core(x, prompt, texture_mod)  # [B, L, C]
        y = y.permute(0, 2, 1).contiguous()
        return y


class TSSM_ASSM(nn.Module):
    """纹理引导状态空间调制的 ASSM

    相比原始 ASSM, 在调用 selective_scan 前:
        1. 将序列特征转回 2D 空间 → [B, C, H, W]
        2. 用 TCE 估计纹理复杂度 → [B, 1, d_state, H*W]
        3. 将调制系数按语义排序索引重排 → [B, 1, d_state, L]
        4. 在 TSSM_Selective_Scan 中对 C 矩阵进行乘法调制

    消融开关:
        use_tssm=True  (默认): 启用纹理调制, 使用 TSSM_Selective_Scan
        use_tssm=False         : 退化为原始 ASSM, 使用 Selective_Scan

    接口:
        forward(self, x, x_size, token) → [B, HW, C]
        与原始 ASSM 完全兼容, 可作为 drop-in 替换

    数据流:
        x [B,HW,C]
          → 语义路由 (route + gumbel_softmax)
          → 2D 重塑 [B,C,H,W]
          → TCE 纹理调制系数 [B,1,d_state,HW]   ← TSSM 新增
          → 调制系数按语义排序重排                ← TSSM 新增
          → in_proj + CPE
          → semantic_neighbor 排序 (SGN-unfold)
          → TSSM_Selective_Scan (带纹理调制)     ← TSSM 修改
          → out_proj + out_norm
          → semantic_neighbor 逆排序 (SGN-fold)
          → 输出 [B,HW,C]
    """

    def __init__(self, dim, d_state, input_resolution, num_tokens=64,
                 inner_rank=128, mlp_ratio=2., use_tssm=True):
        """
        Args:
            dim (int): 输入通道维度
            d_state (int): SSM 状态维度
            input_resolution (tuple): 输入分辨率 (H, W)
            num_tokens (int): 语义令牌数
            inner_rank (int): ASSM 内部秩
            mlp_ratio (float): FFN 隐层扩展比例
            use_tssm (bool): 是否启用纹理引导调制 (False 时退化为原始 ASSM)
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank

        # Mamba 参数
        self.expand = mlp_ratio
        hidden = int(self.dim * self.expand)
        self.d_state = d_state

        # TSSM 开关
        self.use_tssm = use_tssm

        if use_tssm:
            # 纹理复杂度估计器 (TCE)
            self.tce = TextureComplexityEstimator(channels=dim, d_state=d_state)
            # 带纹理调制的 Selective_Scan
            self.selectiveScan = TSSM_Selective_Scan(
                d_model=hidden, d_state=self.d_state, expand=1)
        else:
            # 消融模式: 使用原始 Selective_Scan (无纹理调制)
            self.selectiveScan = Selective_Scan(
                d_model=hidden, d_state=self.d_state, expand=1)

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

    def forward(self, x, x_size, token):
        """前向传播

        Args:
            x: [B, HW, C] 输入特征序列
            x_size: (H, W) 空间尺寸
            token: nn.Embedding, 语义令牌字典 (即 AttentiveLayer 中的 embeddingA)

        Returns:
            x: [B, HW, C] 输出特征序列
        """
        B, n, C = x.shape
        H, W = x_size

        # ---- 语义路由 (与原始 ASSM 相同) ----
        full_embedding = self.embeddingB.weight @ token.weight  # [inner_rank, d_state]

        pred_route = self.route(x)  # [B, HW, num_token]
        cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)  # [B, HW, num_token]

        prompt = torch.matmul(cls_policy, full_embedding).view(B, n, self.d_state)

        detached_index = torch.argmax(cls_policy.detach(), dim=-1,
                                     keepdim=False).view(B, n)  # [B, HW]
        x_sort_values, x_sort_indices = torch.sort(
            detached_index, dim=-1, stable=False)
        x_sort_indices_reverse = index_reverse(x_sort_indices)

        # ---- 转换到 2D 空间 ----
        x_2d = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()  # [B, dim, H, W]

        # ======== TSSM: 纹理复杂度估计 ========
        if self.use_tssm:
            # 在 2D 空间上估计纹理复杂度 (原始空间顺序)
            texture_mod = self.tce(x_2d)  # [B, 1, d_state, H*W]

            # 将调制系数按语义排序索引重排 (与特征排序一致)
            # 这样调制系数的空间顺序与 semantic_x 对齐
            #
            # texture_mod: [B, 1, d_state, L]
            #   → squeeze(1): [B, d_state, L]
            #   → permute(0,2,1): [B, L, d_state]  (适配 semantic_neighbor 格式)
            #   → semantic_neighbor: [B, L, d_state]  (按 x_sort_indices 重排)
            #   → permute(0,2,1).unsqueeze(1): [B, 1, d_state, L]  (恢复格式)
            texture_mod_sorted = semantic_neighbor(
                texture_mod.squeeze(1).permute(0, 2, 1),  # [B, L, d_state]
                x_sort_indices  # [B, L]
            )  # [B, L, d_state]
            texture_mod_sorted = texture_mod_sorted.permute(
                0, 2, 1).unsqueeze(1).contiguous()  # [B, 1, d_state, L]
        # ========================================

        # ---- 特征投影 (与原始 ASSM 相同) ----
        x = self.in_proj(x_2d)  # [B, hidden, H, W]
        x = x * torch.sigmoid(self.CPE(x))  # CPE 调制
        cc = x.shape[1]
        x = x.view(B, cc, -1).contiguous().permute(0, 2, 1)  # [B, n, hidden]

        # ---- 语义排序 (SGN-unfold) ----
        semantic_x = semantic_neighbor(x, x_sort_indices)  # [B, n, hidden]

        # ---- 选择性扫描 (带纹理调制) ----
        if self.use_tssm:
            y = self.selectiveScan(semantic_x, prompt, texture_mod_sorted)
        else:
            y = self.selectiveScan(semantic_x, prompt)

        # ---- 输出投影 (与原始 ASSM 相同) ----
        y = self.out_proj(self.out_norm(y))

        # ---- 语义逆排序 (SGN-fold) ----
        x = semantic_neighbor(y, x_sort_indices_reverse)

        return x


if __name__ == '__main__':
    print("=" * 60)
    print("TSSM_ASSM 测试")
    print("=" * 60)

    # ==================== TCE 参数量 ====================
    print("\n--- TCE 参数量 ---")
    for ed, ds in [(48, 8), (60, 8), (180, 16)]:
        tce = TextureComplexityEstimator(channels=ed, d_state=ds)
        p = sum(pp.numel() for pp in tce.parameters())
        print(f"  embed_dim={ed:3d}, d_state={ds:2d}: {p:6d} params ({p / 1e3:.2f}K)")

    # ==================== TSSM_ASSM 参数量对比 ====================
    print("\n--- TSSM_ASSM 参数量对比 ---")
    B, H, W = 2, 16, 16
    dim = 48
    d_state = 8
    num_tokens = 64
    inner_rank = 32

    # 创建 embeddingA (token 字典, 与 AttentiveLayer 中一致)
    embeddingA = nn.Embedding(inner_rank, d_state)
    embeddingA.weight.data.uniform_(-1 / inner_rank, 1 / inner_rank)

    # TSSM_ASSM (启用 TSSM)
    assm_tssm = TSSM_ASSM(
        dim=dim, d_state=d_state, input_resolution=(H, W),
        num_tokens=num_tokens, inner_rank=inner_rank,
        mlp_ratio=2., use_tssm=True)

    # TSSM_ASSM (禁用 TSSM, 消融对照)
    assm_base = TSSM_ASSM(
        dim=dim, d_state=d_state, input_resolution=(H, W),
        num_tokens=num_tokens, inner_rank=inner_rank,
        mlp_ratio=2., use_tssm=False)

    params_tssm = sum(p.numel() for p in assm_tssm.parameters())
    params_base = sum(p.numel() for p in assm_base.parameters())
    tce_params = params_tssm - params_base
    print(f"  TSSM_ASSM (use_tssm=True):  {params_tssm / 1e3:.2f}K params")
    print(f"  TSSM_ASSM (use_tssm=False): {params_base / 1e3:.2f}K params")
    print(f"  TCE 额外参数: {tce_params / 1e3:.2f}K "
          f"({tce_params / params_base * 100:.2f}% 增量)")

    # ==================== 前向传播测试 ====================
    print("\n--- 前向传播测试 ---")
    x = torch.randn(B, H * W, dim)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  设备: {device}")

    if device == 'cuda':
        assm_tssm = assm_tssm.cuda()
        assm_base = assm_base.cuda()
        embeddingA = embeddingA.cuda()
        x = x.cuda()

    try:
        # TSSM 模式
        y_tssm = assm_tssm(x, (H, W), embeddingA)
        print(f"  TSSM 输入: {list(x.shape)}")
        print(f"  TSSM 输出: {list(y_tssm.shape)}")
        assert y_tssm.shape == x.shape, \
            f"输出形状 {y_tssm.shape} 应与输入 {x.shape} 一致"

        # 消融模式 (use_tssm=False)
        y_base = assm_base(x, (H, W), embeddingA)
        print(f"  Base 输入: {list(x.shape)}")
        print(f"  Base 输出: {list(y_base.shape)}")
        assert y_base.shape == x.shape, \
            f"输出形状 {y_base.shape} 应与输入 {x.shape} 一致"

        print("\n  [OK] 所有形状验证通过")

        # 梯度回传测试
        print("\n--- 梯度回传测试 ---")
        x_grad = torch.randn(B, H * W, dim, device=device, requires_grad=True)
        y_grad = assm_tssm(x_grad, (H, W), embeddingA)
        loss = y_grad.sum()
        loss.backward()
        assert x_grad.grad is not None, "输入梯度不应为 None"
        assert assm_tssm.tce.compress.weight.grad is not None, \
            "TCE compress 权重梯度不应为 None"
        print("  [OK] 梯度回传正常 (含 TCE 参数)")

    except Exception as e:
        print(f"\n  前向测试失败: {e}")
        print("  注意: TSSM_ASSM 前向传播需要 CUDA + mamba_ssm 环境")
        print("  在 CPU 环境下仅验证模块构建和参数量")
