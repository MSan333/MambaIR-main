"""MFAMambaIRv2: 三创新点集成架构

集成三个创新点到 MambaIRv2:
    创新点1 - MFA (多尺度频域感知增强):
        MultiScaleFreqSeparator + CascadedFSG, 在 ASSB 残差路径中引入频域增强
    创新点2 - TSSM (纹理引导状态空间调制):
        TSSM_ASSM 替换 ASSM, 用纹理复杂度调制 SSM 的 C 矩阵
    创新点3 - DSTA (密度驱动选择性Token聚合):
        DSTAWindowAttention 替换 WindowAttention, 减少注意力计算量

架构层次:
    AttentiveLayer (MFAMambaIRv2_ATTBlock):
      ├── DSTAWindowAttention (创新点3, 可选替换 WindowAttention)
      ├── TSSM_ASSM (创新点2, 可选替换 ASSM)
      └── ConvFFN / GatedMLP (不变)

    BasicBlock (MFAMambaIRv2_BasicBlock):
      └── MFAMambaIRv2_ATTBlock × depth

    ASSB (MFAMambaIRv2_ASSB):
      ├── MFAMambaIRv2_BasicBlock
      ├── Conv
      ├── MultiScaleFreqSeparator (创新点1, 可选)
      └── CascadedFSG (创新点1, 可选)

    MFAMambaIRv2 (主网络):
      ├── conv_first → PatchEmbed
      ├── MFAMambaIRv2_ASSB × N (频域增强层) / ASSB × N (原始层)
      └── norm → conv_after_body → upsample

消融控制:
    use_freq=True/False   启用/禁用 MFA
    use_tssm=True/False   启用/禁用 TSSM
    use_dsta=True/False   启用/禁用 DSTA
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import to_2tuple, trunc_normal_

# ---- 从 MambaIRv2 导入全部基础组件 (完全复用, 不做修改) ----
from basicsr.archs.mambairv2_arch import (
    window_partition, window_reverse,
    PatchEmbed, PatchUnEmbed, Upsample, UpsampleOneStep,
    WindowAttention, ConvFFN, Gate, GatedMLP,
    ASSM, Selective_Scan,
    AttentiveLayer, BasicBlock, ASSB,
    index_reverse, semantic_neighbor,
)

# ---- 导入频域增强模块 (创新点1) ----
from basicsr.archs.modules.freq_module import MultiScaleFreqSeparator
from basicsr.archs.modules.cfsg import CascadedFSG

# ---- 导入 TSSM 模块 (创新点2) ----
from basicsr.archs.modules.tssm_assm import TSSM_ASSM
from basicsr.archs.modules.ssdps_assm import SSDPS_ASSM

# ---- 导入 DSTA 模块 (创新点3) ----
from basicsr.archs.modules.dsta_module import DSTAWindowAttention


# ===========================================================================
#  MFAMambaIRv2_ATTBlock: 支持 DSTA + TSSM 的 AttentiveLayer
# ===========================================================================
class MFAMambaIRv2_ATTBlock(nn.Module):
    """增强的 AttentiveLayer, 支持可选的 DSTA (创新点3) 和 TSSM (创新点2)

    与 MambaIRv2 原始 AttentiveLayer 的区别:
        - use_dsta=True:  使用 DSTAWindowAttention 替换 WindowAttention
        - use_tssm=True:  使用 TSSM_ASSM 替换 ASSM (已弃用)
        - use_ssdps=True: 使用 SSDPS_ASSM 替换 ASSM (语义-空间双路径扫描)
        - use_tssm 和 use_ssdps 互斥, 同时为 True 时优先 use_ssdps
        - 全部为 False 时, 行为与原始 AttentiveLayer 完全一致

    数据流 (与原始 AttentiveLayer 相同):
        Part1: x → norm1 → wqkv → [DSTA]WindowAttention → +shortcut → convffn1 → scale1
        Part2: x → norm3 → [TSSM]ASSM → +x → convffn2 → scale2
    """

    def __init__(self,
                 dim,
                 d_state,
                 input_resolution,
                 num_heads,
                 window_size,
                 shift_size,
                 inner_rank,
                 num_tokens,
                 convffn_kernel_size,
                 mlp_ratio,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 is_last=False,
                 use_dsta=False,
                 dsta_keep_ratio=0.5,
                 dsta_group_size=4,
                 use_tssm=False,
                 use_ssdps=False,
                 ):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.convffn_kernel_size = convffn_kernel_size
        self.num_tokens = num_tokens
        self.softmax = nn.Softmax(dim=-1)
        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.is_last = is_last
        self.inner_rank = inner_rank
        self.use_ssdps = use_ssdps

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)

        layer_scale = 1e-4
        self.scale1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
        self.scale2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

        self.wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)

        # ---- 创新点3: DSTA WindowAttention (可选) ----
        if use_dsta:
            self.win_mhsa = DSTAWindowAttention(
                self.dim,
                window_size=to_2tuple(self.window_size),
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                keep_ratio=dsta_keep_ratio,
                group_size=dsta_group_size,
                use_dsta=True,
            )
        else:
            self.win_mhsa = WindowAttention(
                self.dim,
                window_size=to_2tuple(self.window_size),
                num_heads=num_heads,
                qkv_bias=qkv_bias,
            )

        # ---- 创新点2: SSDPS / TSSM ASSM (可选, 互斥) ----
        if use_ssdps:
            # SSDPS: 语义-空间双路径扫描 (新创新点2, 替代 TSSM)
            self.assm = SSDPS_ASSM(
                self.dim,
                d_state,
                input_resolution=input_resolution,
                num_tokens=num_tokens,
                inner_rank=inner_rank,
                mlp_ratio=mlp_ratio,
                ssdps_ratio=0.5,
            )
        elif use_tssm:
            # TSSM: 纹理引导状态空间调制 (已弃用, 保留用于消融对比)
            self.assm = TSSM_ASSM(
                self.dim,
                d_state,
                input_resolution=input_resolution,
                num_tokens=num_tokens,
                inner_rank=inner_rank,
                mlp_ratio=mlp_ratio,
                use_tssm=True,
            )
        else:
            # 原始 ASSM
            self.assm = ASSM(
                self.dim,
                d_state,
                input_resolution=input_resolution,
                num_tokens=num_tokens,
                inner_rank=inner_rank,
                mlp_ratio=mlp_ratio,
            )

        mlp_hidden_dim = int(dim * self.mlp_ratio)
        self.convffn1 = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim,
                                kernel_size=convffn_kernel_size)
        self.convffn2 = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim,
                                kernel_size=convffn_kernel_size)

        self.embeddingA = nn.Embedding(self.inner_rank, d_state)
        self.embeddingA.weight.data.uniform_(-1 / self.inner_rank, 1 / self.inner_rank)

    def forward(self, x, x_size, params):
        h, w = x_size
        b, n, c = x.shape
        c3 = 3 * c

        # part1: Window-MHSA (支持 DSTA)
        shortcut = x
        x = self.norm1(x)
        qkv = self.wqkv(x)
        qkv = qkv.reshape(b, h, w, c3)
        if self.shift_size > 0:
            shifted_qkv = torch.roll(qkv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = params['attn_mask']
        else:
            shifted_qkv = qkv
            attn_mask = None
        x_windows = window_partition(shifted_qkv, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c3)
        attn_windows = self.win_mhsa(x_windows, rpi=params['rpi_sa'], mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x
        x_win = attn_x.view(b, n, c) + shortcut
        x_win = self.convffn1(self.norm2(x_win), x_size) + x_win
        x = shortcut * self.scale1 + x_win

        # part2: Attentive State Space (支持 TSSM)
        shortcut = x
        x_aca = self.assm(self.norm3(x), x_size, self.embeddingA) + x
        x = x_aca + self.convffn2(self.norm4(x_aca), x_size)
        x = shortcut * self.scale2 + x

        return x


# ===========================================================================
#  MFAMambaIRv2_BasicBlock: 使用增强 AttentiveLayer 的 BasicBlock
# ===========================================================================
class MFAMambaIRv2_BasicBlock(nn.Module):
    """增强的 BasicBlock, 使用 MFAMambaIRv2_ATTBlock 替代原始 AttentiveLayer

    与 MambaIRv2 BasicBlock 的区别:
        - 传递 use_dsta / use_tssm / use_ssdps 参数到每个 AttentiveLayer
    """

    def __init__(self,
                 dim,
                 d_state,
                 input_resolution,
                 idx,
                 depth,
                 num_heads,
                 window_size,
                 inner_rank,
                 num_tokens,
                 convffn_kernel_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 use_dsta=False,
                 dsta_keep_ratio=0.5,
                 dsta_group_size=4,
                 use_tssm=False,
                 use_ssdps=False,
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.idx = idx

        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                MFAMambaIRv2_ATTBlock(
                    dim=dim,
                    d_state=d_state,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    inner_rank=inner_rank,
                    num_tokens=num_tokens,
                    convffn_kernel_size=convffn_kernel_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    is_last=i == depth - 1,
                    use_dsta=use_dsta,
                    dsta_keep_ratio=dsta_keep_ratio,
                    dsta_group_size=dsta_group_size,
                    use_tssm=use_tssm,
                    use_ssdps=use_ssdps,
                )
            )

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size, params):
        b, n, c = x.shape
        for layer in self.layers:
            x = layer(x, x_size, params)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'


# ===========================================================================
#  MFAMambaIRv2_ASSB: 增强 ASSB (MFA + TSSM + DSTA)
# ===========================================================================
class MFAMambaIRv2_ASSB(nn.Module):
    """三创新点集成的 ASSB

    与原始 ASSB 的区别:
        1. 残差组使用 MFAMambaIRv2_BasicBlock (含 DSTA + SSDPS/TSSM)
        2. 残差路径增加 MultiScaleFreqSeparator + CascadedFSG (MFA)

    数据流:
        x → MFAMambaIRv2_BasicBlock → PatchUnEmbed → Conv → [FreqSep + CFSG] → PatchEmbed → (+x)
    """

    def __init__(self,
                 dim,
                 d_state,
                 idx,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 inner_rank,
                 num_tokens,
                 convffn_kernel_size,
                 mlp_ratio,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=4,
                 resi_connection='1conv',
                 use_freq=True,
                 num_freq_scales=3,
                 cfsg_kernels=(7, 9, 11),
                 use_dsta=False,
                 dsta_keep_ratio=0.5,
                 dsta_group_size=4,
                 use_tssm=False,
                 use_ssdps=False,
                 ):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.use_freq = use_freq

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0,
            embed_dim=dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0,
            embed_dim=dim, norm_layer=None)

        # ---- 残差组: 使用增强的 BasicBlock (含 DSTA + SSDPS/TSSM) ----
        self.residual_group = MFAMambaIRv2_BasicBlock(
            dim=dim,
            d_state=d_state,
            input_resolution=input_resolution,
            idx=idx,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            num_tokens=num_tokens,
            inner_rank=inner_rank,
            convffn_kernel_size=convffn_kernel_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            use_dsta=use_dsta,
            dsta_keep_ratio=dsta_keep_ratio,
            dsta_group_size=dsta_group_size,
            use_tssm=use_tssm,
            use_ssdps=use_ssdps,
        )

        # ---- 残差卷积 ----
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )

        # ---- 频域增强模块 (创新点1, 可选) ----
        if use_freq:
            self.freq_separator = MultiScaleFreqSeparator(
                channels=dim, num_scales=num_freq_scales)
            self.cfsg = CascadedFSG(
                channels=dim, kernel_sizes=tuple(cfsg_kernels))

    def forward(self, x, x_size, params):
        residual = self.residual_group(x, x_size, params)
        x_2d = self.patch_unembed(residual, x_size)
        x_2d = self.conv(x_2d)

        if self.use_freq:
            f_freq = self.freq_separator(x_2d)
            x_2d = self.cfsg(x_2d, f_freq)

        return self.patch_embed(x_2d) + x


# ===========================================================================
#  MFAMambaIRv2: 三创新点集成主网络
# ===========================================================================
@ARCH_REGISTRY.register()
class MFAMambaIRv2(nn.Module):
    """MFAMambaIRv2: 三创新点集成 MambaIRv2

    在 MambaIRv2 基础上集成三个可选创新点:
        - MFA:  多尺度频域感知增强 (频域分离器 + 级联频域-空域融合门)
        - TSSM: 纹理引导状态空间调制 (TCE + 调制 C 矩阵)
        - DSTA: 密度驱动选择性Token聚合 (减少注意力计算量)

    每个创新点可通过独立开关控制, 支持任意组合的消融实验。

    Args:
        img_size (int): 输入图像大小, 默认 64
        patch_size (int): Patch 大小, 默认 1
        in_chans (int): 输入通道数, 默认 3
        embed_dim (int): 嵌入维度, 默认 180
        d_state (int): SSM 状态维度, 默认 16
        depths (tuple): 各层深度
        num_heads (tuple): 各层注意力头数
        window_size (int): 窗口大小, 默认 16
        inner_rank (int): ASSM 内部秩, 默认 64
        num_tokens (int): 语义令牌数, 默认 128
        convffn_kernel_size (int): ConvFFN 核大小, 默认 5
        mlp_ratio (float): FFN 隐层比例, 默认 2.0
        upscale (int): 上采样倍数, 默认 2
        img_range (float): 图像值范围, 默认 1.0
        upsampler (str): 上采样器类型
        resi_connection (str): 残差连接类型, 默认 '1conv'
        num_freq_scales (int): 频域分解尺度数, 默认 3
        freq_deploy_ratio (float): 使用频域增强的 Block 比例, 默认 0.75
        cfsg_kernels (tuple): CFSG 各级卷积核大小, 默认 (7, 9, 11)
        use_freq (bool): 是否启用频域增强 (创新点1), 默认 True
        use_tssm (bool): 是否启用纹理调制 (创新点2), 默认 True
        use_dsta (bool): 是否启用Token聚合 (创新点3), 默认 True
        dsta_keep_ratio (float): DSTA Top-K 保留比例, 默认 0.5
        dsta_group_size (int): DSTA 聚合分组大小, 默认 4
    """

    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=180,
                 d_state=16,
                 depths=(6, 6, 6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6, 6, 6),
                 window_size=16,
                 inner_rank=64,
                 num_tokens=128,
                 convffn_kernel_size=5,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 # ---- 创新点控制参数 ----
                 num_freq_scales=3,
                 freq_deploy_ratio=0.75,
                 use_freq=True,
                 cfsg_kernels=(7, 9, 11),
                 use_tssm=True,
                 use_dsta=True,
                 use_ssdps=False,
                 dsta_keep_ratio=0.5,
                 dsta_group_size=4,
                 **kwargs):
        super().__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        # ===================== 1, shallow feature extraction =====================
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ===================== 2, deep feature extraction =====================
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        relative_position_index_SA = self._calculate_rpi_sa()
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)

        # ---- 确定哪些层部署频域增强 (优先中间层) ----
        if use_freq:
            num_freq_layers = max(1, int(self.num_layers * freq_deploy_ratio))
            start = (self.num_layers - num_freq_layers) // 2
            freq_layer_indices = set(
                min(start + i, self.num_layers - 1) for i in range(num_freq_layers))
        else:
            freq_layer_indices = set()

        # ---- 构建各层 MFAMambaIRv2_ASSB / ASSB ----
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if i_layer in freq_layer_indices:
                # 频域增强层: 使用 MFAMambaIRv2_ASSB (含 MFA + SSDPS/TSSM + DSTA)
                layer = MFAMambaIRv2_ASSB(
                    dim=embed_dim,
                    d_state=d_state,
                    idx=i_layer,
                    input_resolution=(patches_resolution[0], patches_resolution[1]),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    inner_rank=inner_rank,
                    num_tokens=num_tokens,
                    convffn_kernel_size=convffn_kernel_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    downsample=None,
                    use_checkpoint=use_checkpoint,
                    img_size=img_size,
                    patch_size=patch_size,
                    resi_connection=resi_connection,
                    use_freq=True,
                    num_freq_scales=num_freq_scales,
                    cfsg_kernels=cfsg_kernels,
                    use_dsta=use_dsta,
                    dsta_keep_ratio=dsta_keep_ratio,
                    dsta_group_size=dsta_group_size,
                    use_tssm=use_tssm,
                    use_ssdps=use_ssdps,
                )
            else:
                # 非频域增强层: 使用原始 ASSB (但内部 AttentiveLayer 仍可含 SSDPS/TSSM/DSTA)
                if use_tssm or use_dsta or use_ssdps:
                    # 需要增强的 BasicBlock 但不含频域模块
                    layer = MFAMambaIRv2_ASSB(
                        dim=embed_dim,
                        d_state=d_state,
                        idx=i_layer,
                        input_resolution=(patches_resolution[0], patches_resolution[1]),
                        depth=depths[i_layer],
                        num_heads=num_heads[i_layer],
                        window_size=window_size,
                        inner_rank=inner_rank,
                        num_tokens=num_tokens,
                        convffn_kernel_size=convffn_kernel_size,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=qkv_bias,
                        norm_layer=norm_layer,
                        downsample=None,
                        use_checkpoint=use_checkpoint,
                        img_size=img_size,
                        patch_size=patch_size,
                        resi_connection=resi_connection,
                        use_freq=False,
                        use_dsta=use_dsta,
                        dsta_keep_ratio=dsta_keep_ratio,
                        dsta_group_size=dsta_group_size,
                        use_tssm=use_tssm,
                        use_ssdps=use_ssdps,
                    )
                else:
                    # 完全原始 ASSB (无任何创新点)
                    layer = ASSB(
                        dim=embed_dim,
                        d_state=d_state,
                        idx=i_layer,
                        input_resolution=(patches_resolution[0], patches_resolution[1]),
                        depth=depths[i_layer],
                        num_heads=num_heads[i_layer],
                        window_size=window_size,
                        inner_rank=inner_rank,
                        num_tokens=num_tokens,
                        convffn_kernel_size=convffn_kernel_size,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=qkv_bias,
                        norm_layer=norm_layer,
                        downsample=None,
                        use_checkpoint=use_checkpoint,
                        img_size=img_size,
                        patch_size=patch_size,
                        resi_connection=resi_connection,
                    )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # ===================== 3, reconstruction =====================
        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            self.upsample = UpsampleOneStep(
                upscale, embed_dim, num_out_ch,
                (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _calculate_rpi_sa(self):
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        return relative_coords.sum(-1)

    def _calculate_mask(self, x_size):
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))
        h_slices = (slice(0, -self.window_size),
                     slice(-self.window_size, -(self.window_size // 2)),
                     slice(-(self.window_size // 2), None))
        w_slices = (slice(0, -self.window_size),
                     slice(-self.window_size, -(self.window_size // 2)),
                     slice(-(self.window_size // 2), None))
        cnt = 0
        for h_s in h_slices:
            for w_s in w_slices:
                img_mask[:, h_s, w_s, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, params):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        for layer in self.layers:
            x = layer(x, x_size, params)
        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    def forward(self, x):
        # ---- padding ----
        h_ori, w_ori = x.size()[-2], x.size()[-1]
        mod = self.window_size
        h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
        w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori
        h, w = h_ori + h_pad, w_ori + w_pad
        x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h, :]
        x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w]

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        attn_mask = self._calculate_mask([h, w]).to(x.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}

        if self.upsampler == 'pixelshuffle':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(
                torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_up2(
                torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first, params)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        # ---- unpadding ----
        x = x[..., :h_ori * self.upscale, :w_ori * self.upscale]
        return x


if __name__ == '__main__':
    print("=" * 70)
    print("MFAMambaIRv2: 三创新点集成架构测试")
    print("=" * 70)

    upscale = 4
    model = MFAMambaIRv2(
        upscale=upscale,
        img_size=64,
        embed_dim=48,
        d_state=8,
        depths=[5, 5, 5, 5],
        num_heads=[4, 4, 4, 4],
        window_size=16,
        inner_rank=32,
        num_tokens=64,
        convffn_kernel_size=5,
        img_range=1.,
        mlp_ratio=1.,
        upsampler='pixelshuffledirect',
        # 三创新点开关
        use_freq=True,
        use_tssm=True,
        use_dsta=True,
        num_freq_scales=3,
        freq_deploy_ratio=0.75,
        cfsg_kernels=(7, 9, 11),
        dsta_keep_ratio=0.5,
        dsta_group_size=4,
    )

    # 参数量统计
    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total / 1e6:.3f}M")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable / 1e6:.3f}M")

    # 前向测试
    inp = torch.randn(1, 3, 64, 64)
    out = model(inp)
    print(f"Input:  {inp.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == (1, 3, 64 * upscale, 64 * upscale), f"Output shape mismatch: {out.shape}"
    print("[OK] 前向传播测试通过")

    # ---- 消融实验配置测试 ----
    print("\n" + "-" * 70)
    print("消融实验配置:")
    configs = [
        ("Baseline",          False, False, False),
        ("+MFA",              True,  False, False),
        ("+TSSM",             False, True,  False),
        ("+DSTA",             False, False, True),
        ("+MFA+TSSM",         True,  True,  False),
        ("+MFA+DSTA",         True,  False, True),
        ("Full (MFA+TSSM+DSTA)", True, True, True),
    ]
    print(f"{'Config':<25} | {'Params (M)':>10} | {'Freq':>5} | {'TSSM':>5} | {'DSTA':>5}")
    print(f"{'-'*25}-+-{'-'*10}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}")
    for name, freq, tssm, dsta in configs:
        m = MFAMambaIRv2(
            upscale=4, img_size=64, embed_dim=48, d_state=8,
            depths=[5, 5, 5, 5], num_heads=[4, 4, 4, 4],
            window_size=16, inner_rank=32, num_tokens=64,
            convffn_kernel_size=5, img_range=1., mlp_ratio=1.,
            upsampler='pixelshuffledirect',
            use_freq=freq, use_tssm=tssm, use_dsta=dsta,
        )
        p = sum(pp.numel() for pp in m.parameters())
        print(f"{name:<25} | {p/1e6:>10.3f} | {'✓' if freq else '✗':>5} | {'✓' if tssm else '✗':>5} | {'✓' if dsta else '✗':>5}")
