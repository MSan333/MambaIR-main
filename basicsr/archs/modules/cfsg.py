"""Cascaded Frequency-Spatial Gate (CFSG)

级联频域-空域融合门

借鉴 UniConvNet (ICCV 2025) 的 RFA (Receptive Field Aggregator) 思想:
    - Amp (Amplify): 大核深度卷积放大空域感受野
    - Dis (Discriminate): 小核卷积判别频域细粒度信息
    - 三级级联 (7x7 -> 9x9 -> 11x11) 渐进扩大感受野
"""
import torch
import torch.nn as nn


class AmpDisModule(nn.Module):
    """Amplify-Discriminate Module (放大-判别模块)

    借鉴 UniConvNet 的 Layer Operator 设计:
        - Amp 分支: 大核深度可分离卷积, 扩大前层有效感受野
        - Dis 分支: 3x3 卷积, 捕获细粒度频域判别信息
        - 门控融合: 自适应加权两个分支

    Args:
        channels (int): 输入/输出通道数
        kernel_size (int): Amp 分支的大核卷积核大小
    """

    def __init__(self, channels, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2

        # Amplify 分支: 大核深度可分离卷积 -> 扩大感受野
        self.amp = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding,
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
        )

        # Discriminate 分支: 3x3 卷积 -> 细粒度判别
        self.dis = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1,
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
        )

        # 门控权重生成
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x_spatial, x_freq):
        """
        Args:
            x_spatial (Tensor): [B, C, H, W] 空域特征 (来自前级或 ASSM 输出)
            x_freq (Tensor): [B, C, H, W] 频域特征 (来自 DWT 分解)

        Returns:
            Tensor: [B, C, H, W] 融合后特征
        """
        # Amp: 放大空域特征的感受野范围
        amp_out = self.amp(x_spatial)

        # Dis: 判别频域细节信息
        dis_out = self.dis(x_freq)

        # 自适应门控融合
        gate_weight = self.gate(torch.cat([amp_out, dis_out], dim=1))
        fused = gate_weight * amp_out + (1 - gate_weight) * dis_out

        return fused


class CascadedFSG(nn.Module):
    """Cascaded Frequency-Spatial Gate (级联频域-空域融合门)

    三级 AmpDis 模块逐步融合频域和空域信息:
        - Stage 1 (7x7): 中等感受野, 捕获局部频域-空域关系
        - Stage 2 (9x9): 较大感受野, 建立中等范围依赖
        - Stage 3 (11x11): 大感受野, 实现长距离频域-空域融合

    渐进扩展保持 AGD (Asymptotically Gaussian Distribution) 特性。

    Args:
        channels (int): 通道数
        kernel_sizes (tuple): 各级核大小, 默认 (7, 9, 11)
    """

    def __init__(self, channels, kernel_sizes=(7, 9, 11)):
        super().__init__()

        self.num_stages = len(kernel_sizes)
        self.stages = nn.ModuleList([
            AmpDisModule(channels, kernel_size=ks)
            for ks in kernel_sizes
        ])

        # 残差缩放因子 (初始化为小值, 稳定训练初期)
        self.scale = nn.Parameter(torch.ones(1) * 0.01)

    def forward(self, f_spatial, f_freq):
        """
        Args:
            f_spatial (Tensor): [B, C, H, W] ASSM 输出的空域特征
            f_freq (Tensor): [B, C, H, W] 频域模块输出的多尺度频域特征

        Returns:
            Tensor: [B, C, H, W] 融合后的增强特征
        """
        # 三级级联融合: 每级的空域输入来自上一级的融合输出
        out = self.stages[0](f_spatial, f_freq)
        for stage in self.stages[1:]:
            out = stage(out, f_freq)  # 频域特征在每级都参与融合

        # 残差连接 + 可学习缩放 (保持训练稳定性)
        output = f_spatial + self.scale * out

        return output
