"""MFAMambaIR_v3: MFA + DSTA

MFA (1: freq) + DSTA (3: token aggregation)
AttentiveLayerV3 > BasicBlockV3 > MFA_ASSB_V3 > MFAMambaIR_v3
"""
import torch
import torch.nn as nn

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from basicsr.archs.mambairv2_arch import (
    window_partition, window_reverse,
    PatchEmbed, PatchUnEmbed, Upsample, UpsampleOneStep,
    WindowAttention, ConvFFN, Gate, GatedMLP,
    ASSM, Selective_Scan,
    AttentiveLayer, BasicBlock, ASSB,
    index_reverse, semantic_neighbor,
)
from basicsr.archs.modules.freq_module import MultiScaleFreqSeparator
from basicsr.archs.modules.cfsg import CascadedFSG
from basicsr.archs.modules.dsta_module import DSTAWindowAttention


class AttentiveLayerV3(AttentiveLayer):
    """AttentiveLayer with DSTA replacing WindowAttention."""

    def __init__(self, dim, d_state, input_resolution, num_heads, window_size,
                 shift_size, inner_rank, num_tokens, convffn_kernel_size,
                 mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm, is_last=False,
                 use_dsta=True, dsta_keep_ratio=0.5, dsta_group_size=4):
        super().__init__(
            dim=dim, d_state=d_state, input_resolution=input_resolution,
            num_heads=num_heads, window_size=window_size, shift_size=shift_size,
            inner_rank=inner_rank, num_tokens=num_tokens,
            convffn_kernel_size=convffn_kernel_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, norm_layer=norm_layer, is_last=is_last)
        if use_dsta:
            self.win_mhsa = DSTAWindowAttention(
                dim=self.dim,
                window_size=to_2tuple(self.window_size),
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                keep_ratio=dsta_keep_ratio,
                group_size=dsta_group_size,
                use_dsta=True)


class BasicBlockV3(nn.Module):

    def __init__(self, dim, d_state, input_resolution, idx, depth, num_heads,
                 window_size, inner_rank, num_tokens, convffn_kernel_size,
                 mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False,
                 use_dsta=True, dsta_keep_ratio=0.5, dsta_group_size=4):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.idx = idx
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                AttentiveLayerV3(
                    dim=dim, d_state=d_state, input_resolution=input_resolution,
                    num_heads=num_heads, window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    inner_rank=inner_rank, num_tokens=num_tokens,
                    convffn_kernel_size=convffn_kernel_size, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, norm_layer=norm_layer,
                    is_last=i == depth - 1,
                    use_dsta=use_dsta, dsta_keep_ratio=dsta_keep_ratio,
                    dsta_group_size=dsta_group_size))
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


class MFA_ASSB_V3(nn.Module):

    def __init__(self, dim, d_state, idx, input_resolution, depth, num_heads,
                 window_size, inner_rank, num_tokens, convffn_kernel_size,
                 mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False, img_size=224,
                 patch_size=4, resi_connection='1conv',
                 use_freq=True, num_freq_scales=3, cfsg_kernels=(7, 9, 11),
                 use_dsta=True, dsta_keep_ratio=0.5, dsta_group_size=4):
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
        self.residual_group = BasicBlockV3(
            dim=dim, d_state=d_state, input_resolution=input_resolution,
            idx=idx, depth=depth, num_heads=num_heads, window_size=window_size,
            num_tokens=num_tokens, inner_rank=inner_rank,
            convffn_kernel_size=convffn_kernel_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, norm_layer=norm_layer,
            downsample=downsample, use_checkpoint=use_checkpoint,
            use_dsta=use_dsta, dsta_keep_ratio=dsta_keep_ratio,
            dsta_group_size=dsta_group_size)
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))
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

@ARCH_REGISTRY.register()
class MFAMambaIR_v3(nn.Module):
    """MFAMambaIR_v3: MFA + DSTA

    conv_first -> PatchEmbed -> MFA_ASSB_V3 x N -> norm -> conv_after_body -> upsample
    All layers use MFA_ASSB_V3 (with DSTA). Freq layers also have MFA.
    """

    def __init__(self,
                 img_size=64, patch_size=1, in_chans=3, embed_dim=180,
                 d_state=16, depths=(6, 6, 6, 6, 6, 6), num_heads=(6, 6, 6, 6, 6, 6),
                 window_size=16, inner_rank=64, num_tokens=128,
                 convffn_kernel_size=5, mlp_ratio=2., qkv_bias=True,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1., upsampler='',
                 resi_connection='1conv',
                 num_freq_scales=3, freq_deploy_ratio=0.75, use_freq=True,
                 cfsg_kernels=(7, 9, 11),
                 use_dsta=True, dsta_keep_ratio=0.5, dsta_group_size=4,
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

        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

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

        if use_freq:
            num_freq_layers = max(1, int(self.num_layers * freq_deploy_ratio))
            start = (self.num_layers - num_freq_layers) // 2
            freq_layer_indices = set(
                min(start + i, self.num_layers - 1) for i in range(num_freq_layers))
        else:
            freq_layer_indices = set()

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = MFA_ASSB_V3(
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
                use_freq=(i_layer in freq_layer_indices) and use_freq,
                num_freq_scales=num_freq_scales,
                cfsg_kernels=cfsg_kernels,
                use_dsta=use_dsta,
                dsta_keep_ratio=dsta_keep_ratio,
                dsta_group_size=dsta_group_size,
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
        x = x[..., :h_ori * self.upscale, :w_ori * self.upscale]
        return x
