"""MFAMambaIR: Multi-scale Frequency-Aware MambaIR

基于 MambaIRv2, 在 ASSB 的残差路径中引入 CFSG (级联频域-空域融合门)
与多尺度频域分离器 (MultiScaleFreqSeparator), 其余组件完全复用 MambaIRv2 原始实现。

核心改动:
    ASSB 残差路径: BasicBlock → PatchUnEmbed → Conv → [FreqSep + CFSG] → PatchEmbed
    其余所有组件 (AttentiveLayer, BasicBlock, ASSM, WindowAttention, ConvFFN 等)
    均直接使用 MambaIRv2 原始代码, 不做任何修改。
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

# ---- 导入频域增强模块 ----
from basicsr.archs.modules.freq_module import MultiScaleFreqSeparator
from basicsr.archs.modules.cfsg import CascadedFSG


# ===========================================================================
#  MFA_ASSB: 在 ASSB 残差路径中加入频域增强
# ===========================================================================
class MFA_ASSB(nn.Module):
    """Attentive State Space Block with CFSG Frequency-Spatial Fusion

    结构与 MambaIRv2 的 ASSB 基本相同, 区别是在残差路径的 Conv 之后、
    PatchEmbed 之前插入频域分离器 (MultiScaleFreqSeparator) + 级联频域-空域
    融合门 (CascadedFSG)。

    数据流:
        x ──→ BasicBlock ──→ PatchUnEmbed ──→ Conv ──→ [FreqSep + CFSG] ──→ PatchEmbed ──→ (+x)
             (原 AttentiveLayer × depth)         (可选, use_freq=True 时启用)
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
                 cfsg_kernels=(7, 9, 11)):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.use_freq = use_freq

        # ---- Patch Embed / UnEmbed (与 MambaIRv2 相同) ----
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0,
            embed_dim=dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0,
            embed_dim=dim, norm_layer=None)

        # ---- 残差组: 直接复用 MambaIRv2 的 BasicBlock ----
        self.residual_group = BasicBlock(
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
        )

        # ---- 残差卷积 (与 MambaIRv2 相同) ----
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

        # ---- 频域增强模块: 频域分离器 + 级联频域-空域融合门 (CFSG) ----
        if use_freq:
            self.freq_separator = MultiScaleFreqSeparator(
                channels=dim, num_scales=num_freq_scales)
            self.cfsg = CascadedFSG(
                channels=dim, kernel_sizes=tuple(cfsg_kernels))

    def forward(self, x, x_size, params):
        # 与 MambaIRv2 ASSB 相同的残差路径, 仅多一步频域-空域融合
        residual = self.residual_group(x, x_size, params)   # [B, HW, C]
        x_2d = self.patch_unembed(residual, x_size)        # [B, C, H, W]
        x_2d = self.conv(x_2d)                             # [B, C, H, W]

        # 频域分离 + CFSG 级联融合 (可选)
        if self.use_freq:
            f_freq = self.freq_separator(x_2d)             # [B, C, H, W] 多尺度频域特征
            x_2d = self.cfsg(x_2d, f_freq)                # [B, C, H, W] 级联融合

        return self.patch_embed(x_2d) + x                  # [B, HW, C]


# ===========================================================================
#  MFAMambaIR: 主网络
# ===========================================================================
@ARCH_REGISTRY.register()
class MFAMambaIR(nn.Module):
    """MFAMambaIR: Multi-scale Frequency-Aware MambaIR

    整体结构与 MambaIRv2 完全一致:
        conv_first → PatchEmbed → MFA_ASSB × N → norm → conv_after_body → upsample

    唯一区别: 使用 MFA_ASSB 替代 ASSB, 在残差路径中增加多尺度频域增强。

    Args:
        img_size (int): 输入图像大小。默认 64
        patch_size (int): Patch 大小。默认 1
        in_chans (int): 输入通道数。默认 3
        embed_dim (int): 嵌入维度。默认 180
        d_state (int): SSM 状态维度。默认 16
        depths (tuple): 各层深度
        num_heads (tuple): 各层注意力头数
        window_size (int): 窗口大小。默认 16
        inner_rank (int): ASSM 内部秩。默认 64
        num_tokens (int): 语义令牌数。默认 128
        convffn_kernel_size (int): ConvFFN 核大小。默认 5
        mlp_ratio (float): FFN 隐层比例。默认 2.0
        upscale (int): 上采样倍数。默认 2
        img_range (float): 图像值范围。默认 1.0
        upsampler (str): 上采样器类型
        resi_connection (str): 残差连接类型。默认 '1conv'
        num_freq_scales (int): 频域分解尺度数。默认 3
        freq_deploy_ratio (float): 使用频域增强的 Block 比例。默认 0.75
        use_freq (bool): 是否启用频域增强分支。默认 True
        cfsg_kernels (tuple): CFSG 各级卷积核大小。默认 (7, 9, 11)
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
                 num_freq_scales=3,
                 freq_deploy_ratio=0.75,
                 use_freq=True,
                 cfsg_kernels=(7, 9, 11),
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

        # Patch Embed / UnEmbed (与 MambaIRv2 完全相同)
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

        # 绝对位置编码
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        # 相对位置索引 (与 MambaIRv2 相同)
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

        # ---- 构建各层 MFA_ASSB / ASSB ----
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if i_layer in freq_layer_indices:
                layer = MFA_ASSB(
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
                )
            else:
                # 不部署频域增强时, 直接使用原始 ASSB
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

        # conv_after_body (与 MambaIRv2 相同)
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

    # ------------------------------------------------------------------
    #  初始化权重 (与 MambaIRv2 相同)
    # ------------------------------------------------------------------
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ------------------------------------------------------------------
    #  相对位置索引 & 注意力掩码 (从 MambaIRv2 复制, 因它们是实例方法)
    # ------------------------------------------------------------------
    def _calculate_rpi_sa(self):
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
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

    # ------------------------------------------------------------------
    #  深层特征提取 (与 MambaIRv2 相同)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    #  前向传播 (与 MambaIRv2 完全相同的流程)
    # ------------------------------------------------------------------
    def forward(self, x):
        # ---- padding (与 MambaIRv2 相同的 reflect-flip 策略) ----
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
    upscale = 2
    model = MFAMambaIR(
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
        num_freq_scales=3,
        freq_deploy_ratio=0.75,
        use_freq=True,
        cfsg_kernels=(7, 9, 11),
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
