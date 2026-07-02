"""TSSM: 纹理引导状态空间调制 (Texture-guided State Space Modulation)

本模块实现纹理复杂度估计器 (TCE), 用于创新点2:
通过纹理复杂度调制 C 矩阵, 使纹理区域获得更强的输出权重。

核心思想:
    - 纹理复杂区域(边缘、纹理)的调制系数 > 1 → 增强 C 矩阵输出权重
    - 平滑区域的调制系数 ≈ 1 → 保持原始 SSM 行为
    - 初始化时所有位置调制系数 = 1.0(中性值), 训练中自适应学习

架构:
    1. 多尺度深度可分离卷积(3x3 + 5x5)提取局部梯度特征
    2. 1x1 卷积压缩到 d_state 维度
    3. Sigmoid → [0,1] → 缩放到 [0.5, 1.5]

参数效率:
    - 深度可分离卷积: 仅 kernel_size^2 * channels 参数(无 pointwise)
    - 1x1 压缩: 2*channels*d_state 参数
    - 压缩卷积权重初始化为零 → 初始调制系数 = 1.0(中性)
"""

import torch
import torch.nn as nn


class TextureComplexityEstimator(nn.Module):
    """轻量纹理复杂度估计器 (TCE)

    通过多尺度深度可分离卷积提取局部梯度特征, 估计每个空间位置的纹理复杂度,
    生成调制系数用于增强状态空间模型中 C 矩阵在纹理区域的输出权重。

    输入: x [B, C, H, W] - 2D特征图
    输出: modulation [B, 1, d_state, H*W] - 纹理调制系数, 范围 [0.5, 1.5]

    原理:
        - 3x3 DW Conv: 捕获细粒度局部梯度(边缘、纹理细节)
        - 5x5 DW Conv: 捕获粗粒度结构信息
        - 1x1 Conv: 融合多尺度特征并压缩到 d_state 维度
        - Sigmoid + Scale: 确保调制系数在 [0.5, 1.5] 范围, 初始值 = 1.0

    状态空间方程中的调制位置:
        h[t] = A * h[t-1] + B[t] * x[t]      (状态转移, 不受调制)
        y[t] = C'[t] * h[t] + D * x[t]       (输出投影)
        其中 C'[t] = C[t] * texture_mod[t]   (纹理引导调制)
    """

    def __init__(self, channels, d_state):
        """
        Args:
            channels (int): 输入特征图通道数 (即 ASSM 的 dim)
            d_state (int): 状态空间维度, 调制系数的通道数
        """
        super().__init__()
        self.channels = channels
        self.d_state = d_state

        # ---- 多尺度深度可分离卷积 ----
        # 3x3 DW Conv: 捕获细粒度梯度 (边缘、纹理细节)
        self.dwconv_3x3 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1,
            groups=channels, bias=False)
        # 5x5 DW Conv: 捕获粗粒度结构信息
        self.dwconv_5x5 = nn.Conv2d(
            channels, channels, kernel_size=5, stride=1, padding=2,
            groups=channels, bias=False)

        # 激活函数: 引入非线性, 增强特征表达
        self.act = nn.GELU()

        # ---- 1x1 压缩到 d_state 维度 ----
        # 输入 2*channels (3x3 + 5x5 融合), 输出 d_state
        self.compress = nn.Conv2d(
            channels * 2, d_state, kernel_size=1, stride=1, padding=0, bias=True)

        # ---- Sigmoid + Scale ----
        self.sigmoid = nn.Sigmoid()

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重, 确保初始调制系数 ≈ 1.0 (中性值)

        策略:
            - DW Conv: Xavier 均匀初始化 (标准做法, 提取梯度特征)
            - Compress Conv: 权重置零, 偏置置零
              → 初始输出 = 0 → sigmoid(0) = 0.5 → 调制系数 = 1.0
              → 训练开始时不改变原始 ASSM 行为
        """
        nn.init.xavier_uniform_(self.dwconv_3x3.weight)
        nn.init.xavier_uniform_(self.dwconv_5x5.weight)
        # 压缩卷积零初始化: 确保初始输出为 0, sigmoid(0)=0.5, 调制系数=1.0
        nn.init.zeros_(self.compress.weight)
        nn.init.zeros_(self.compress.bias)

    def forward(self, x):
        """前向传播

        Args:
            x: [B, C, H, W] 输入特征图

        Returns:
            modulation: [B, 1, d_state, H*W] 纹理调制系数, 范围 [0.5, 1.5]
        """
        B, C, H, W = x.shape

        # 1. 多尺度局部梯度特征提取
        feat_3x3 = self.dwconv_3x3(x)    # [B, C, H, W] - 细粒度梯度
        feat_5x5 = self.dwconv_5x5(x)    # [B, C, H, W] - 粗粒度结构

        # 2. 融合多尺度特征
        feat = torch.cat([feat_3x3, feat_5x5], dim=1)  # [B, 2C, H, W]
        feat = self.act(feat)

        # 3. 压缩到 d_state 维度
        mod = self.compress(feat)  # [B, d_state, H, W]

        # 4. Sigmoid → [0, 1], 缩放到 [0.5, 1.5]
        #    sigmoid(0) = 0.5 → 0.5 * 1.0 + 0.5 = 1.0 (初始中性值)
        #    纹理区域 → sigmoid 输出大 → 调制系数 > 1 (增强)
        #    平滑区域 → sigmoid 输出小 → 调制系数 < 1 (抑制)
        mod = self.sigmoid(mod) * 1.0 + 0.5  # [B, d_state, H, W]

        # 5. 展平空间维度: [B, d_state, H*W] → [B, 1, d_state, H*W]
        #    维度 1 对应 selective_scan 中的 K=1 (单扫描方向)
        mod = mod.flatten(2).unsqueeze(1)  # [B, 1, d_state, H*W]

        return mod


if __name__ == '__main__':
    # ==================== TCE 模块测试 ====================
    print("=" * 60)
    print("TextureComplexityEstimator (TCE) 测试")
    print("=" * 60)

    B, C, H, W = 2, 48, 16, 16
    d_state = 8

    tce = TextureComplexityEstimator(channels=C, d_state=d_state)

    # 参数量统计
    params = sum(p.numel() for p in tce.parameters())
    print(f"输入: [{B}, {C}, {H}, {W}]")
    print(f"d_state: {d_state}")
    print(f"TCE 参数量: {params} ({params / 1e3:.2f}K)")

    # 前向传播
    x = torch.randn(B, C, H, W)
    mod = tce(x)

    print(f"输出: {list(mod.shape)}")
    print(f"调制系数范围: [{mod.min().item():.4f}, {mod.max().item():.4f}]")
    print(f"调制系数均值: {mod.mean().item():.4f}")

    # 验证初始调制系数 ≈ 1.0 (因 compress 权重初始化为零)
    assert abs(mod.mean().item() - 1.0) < 0.01, \
        f"初始调制系数应接近 1.0, 实际: {mod.mean().item():.4f}"
    # 验证范围 [0.5, 1.5]
    assert mod.min() >= 0.5 - 1e-6, "调制系数下限应为 0.5"
    assert mod.max() <= 1.5 + 1e-6, "调制系数上限应为 1.5"

    print("\n[OK] 所有形状和范围验证通过")

    # ==================== 参数量对比 ====================
    print("\n" + "=" * 60)
    print("参数量对比 (不同 embed_dim 配置)")
    print("=" * 60)
    for ed, ds in [(48, 8), (60, 8), (180, 16)]:
        tce_tmp = TextureComplexityEstimator(channels=ed, d_state=ds)
        p = sum(pp.numel() for pp in tce_tmp.parameters())
        print(f"  embed_dim={ed:3d}, d_state={ds:2d}: {p:6d} params ({p / 1e3:.2f}K)")

    # ==================== 梯度回传测试 ====================
    print("\n" + "=" * 60)
    print("梯度回传测试")
    print("=" * 60)
    tce_grad = TextureComplexityEstimator(channels=C, d_state=d_state)
    x_grad = torch.randn(B, C, H, W, requires_grad=True)
    mod_grad = tce_grad(x_grad)
    loss = mod_grad.sum()
    loss.backward()
    assert x_grad.grad is not None, "输入梯度不应为 None"
    assert tce_grad.compress.weight.grad is not None, "compress 权重梯度不应为 None"
    print("[OK] 梯度回传正常")
