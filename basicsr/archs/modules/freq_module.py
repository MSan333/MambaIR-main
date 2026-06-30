"""多尺度频域增强模块 (Multi-Scale Frequency Enhancer)

使用 Haar 离散小波变换 (DWT) 将特征分解为多尺度频域成分,
分别处理后聚合, 以增强模型对不同频率信息的建模能力。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFreqEnhancer(nn.Module):
    """多尺度频域增强器

    对输入 2D 特征图进行多尺度 DWT 分解, 提取不同频带信息,
    经逐尺度卷积处理后聚合回原始分辨率。

    Args:
        channels (int): 输入/输出通道数。
        num_scales (int): DWT 分解尺度数。默认 3。
    """

    def __init__(self, channels, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.channels = channels

        # ---------- Haar 小波滤波器 (固定, 不参与训练) ----------
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]]).unsqueeze(0).unsqueeze(0)
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]]).unsqueeze(0).unsqueeze(0)
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]]).unsqueeze(0).unsqueeze(0)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('ll_filter', ll)
        self.register_buffer('lh_filter', lh)
        self.register_buffer('hl_filter', hl)
        self.register_buffer('hh_filter', hh)

        # ---------- 各尺度处理: 4C -> C (融合四子带) + 深度可分离卷积 ----------
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels * 4, channels, 1, bias=False),
                nn.GELU(),
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            )
            for _ in range(num_scales)
        ])

        # ---------- 多尺度聚合 ----------
        self.aggregate = nn.Sequential(
            nn.Conv2d(channels * num_scales, channels, 1, bias=False),
            nn.GELU(),
        )

    def _dwt_2d(self, x):
        """单层 2D Haar 离散小波变换

        Args:
            x: [B, C, H, W]
        Returns:
            decomposed: [B, 4C, H/2, W/2]  (LL, LH, HL, HH 沿通道拼接)
            ll_band:    [B, C, H/2, W/2]   (低频子带, 用于下一级分解)
        """
        B, C, H, W = x.shape
        # 确保尺寸为偶数
        if H % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1), mode='reflect')
        if W % 2 != 0:
            x = F.pad(x, (0, 1, 0, 0), mode='reflect')

        B, C, H, W = x.shape
        x_reshaped = x.reshape(B * C, 1, H, W)

        ll = F.conv2d(x_reshaped, self.ll_filter, stride=2)
        lh = F.conv2d(x_reshaped, self.lh_filter, stride=2)
        hl = F.conv2d(x_reshaped, self.hl_filter, stride=2)
        hh = F.conv2d(x_reshaped, self.hh_filter, stride=2)

        oH, oW = ll.shape[2], ll.shape[3]
        ll = ll.reshape(B, C, oH, oW)
        lh = lh.reshape(B, C, oH, oW)
        hl = hl.reshape(B, C, oH, oW)
        hh = hh.reshape(B, C, oH, oW)

        return torch.cat([ll, lh, hl, hh], dim=1), ll

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 输入特征
        Returns:
            out: [B, C, H, W] 频域增强后特征
        """
        target_H, target_W = x.shape[2], x.shape[3]
        scale_features = []
        current = x

        for i in range(self.num_scales):
            decomposed, ll_band = self._dwt_2d(current)  # [B, 4C, H/2^i, W/2^i]
            feat = self.scale_convs[i](decomposed)        # [B, C, H/2^i, W/2^i]

            # 上采样回原始分辨率
            if feat.shape[2] != target_H or feat.shape[3] != target_W:
                feat = F.interpolate(feat, size=(target_H, target_W),
                                     mode='bilinear', align_corners=False)
            scale_features.append(feat)
            current = ll_band  # 下一级用低频子带

        # 聚合所有尺度
        multi_scale = torch.cat(scale_features, dim=1)  # [B, C*num_scales, H, W]
        out = self.aggregate(multi_scale)                # [B, C, H, W]
        return out


class MultiScaleFreqSeparator(nn.Module):
    """多尺度频域特征分离器 (Multi-Scale Frequency Separator)

    使用 Haar 小波的 DWT 将特征逐级分解为多尺度频域成分 (LL/LH/HL/HH 子带),
    然后通过通道卷积处理并聚合回原始分辨率。

    与 ``MultiScaleFreqEnhancer`` 功能等价, 但对外提供公共 ``dwt_2d`` 接口,
    便于 CFSG 模块在需要时复用频域分解结果。

    Args:
        channels (int): 输入通道数
        num_scales (int): 分解尺度数, 默认 3
    """

    def __init__(self, channels, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.channels = channels

        # Haar 小波滤波器 (固定不可学习)
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]]).unsqueeze(0).unsqueeze(0)
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]]).unsqueeze(0).unsqueeze(0)
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]]).unsqueeze(0).unsqueeze(0)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('ll_filter', ll)
        self.register_buffer('lh_filter', lh)
        self.register_buffer('hl_filter', hl)
        self.register_buffer('hh_filter', hh)

        # 各尺度通道处理 (4 个子带合并 -> 单通道特征)
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels * 4, channels, 1, bias=False),
                nn.GELU(),
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            ) for _ in range(num_scales)
        ])

        # 多尺度聚合
        self.aggregate = nn.Sequential(
            nn.Conv2d(channels * num_scales, channels, 1, bias=False),
            nn.GELU()
        )

    def dwt_2d(self, x):
        """2D Haar 离散小波变换

        Args:
            x (Tensor): [B, C, H, W]

        Returns:
            decomposed (Tensor): [B, 4C, H/2, W/2]  (LL, LH, HL, HH 沿通道拼接)
            ll_band (Tensor): [B, C, H/2, W/2]  (低频子带, 用于下一级分解)
        """
        B, C, H, W = x.shape
        # 确保尺寸为偶数
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        B, C, H, W = x.shape
        x_flat = x.reshape(B * C, 1, H, W)

        ll = F.conv2d(x_flat, self.ll_filter, stride=2).reshape(B, C, H // 2, W // 2)
        lh = F.conv2d(x_flat, self.lh_filter, stride=2).reshape(B, C, H // 2, W // 2)
        hl = F.conv2d(x_flat, self.hl_filter, stride=2).reshape(B, C, H // 2, W // 2)
        hh = F.conv2d(x_flat, self.hh_filter, stride=2).reshape(B, C, H // 2, W // 2)

        return torch.cat([ll, lh, hl, hh], dim=1), ll  # [B, 4C, H/2, W/2], [B, C, H/2, W/2]

    def forward(self, x):
        """
        Args:
            x (Tensor): [B, C, H, W] 输入特征

        Returns:
            Tensor: [B, C, H, W] 多尺度频域聚合特征
        """
        target_H, target_W = x.shape[2], x.shape[3]
        scale_features = []
        current = x

        for i in range(self.num_scales):
            decomposed, ll_band = self.dwt_2d(current)  # [B, 4C, H/2^i, W/2^i]

            feat = self.scale_convs[i](decomposed)  # [B, C, H/2^i, W/2^i]

            # 上采样回原始分辨率
            if feat.shape[2] != target_H or feat.shape[3] != target_W:
                feat = F.interpolate(feat, size=(target_H, target_W),
                                    mode='bilinear', align_corners=False)
            scale_features.append(feat)

            # 下一级使用 LL 子带继续分解
            current = ll_band

        # 聚合所有尺度
        multi_scale = torch.cat(scale_features, dim=1)  # [B, C*num_scales, H, W]
        freq_features = self.aggregate(multi_scale)  # [B, C, H, W]

        return freq_features
