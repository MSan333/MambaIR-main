# MFAMambaIR-light 轻量版训练指南

## 需要超越的论文指标（MambaIRv2-light, Table 3）

### x2 超分
| 测试集 | MambaIRv2-light PSNR | MambaIRv2-light SSIM | 目标 |
|--------|---------------------|---------------------|------|
| Set5   | 38.26 | 0.9615 | ≥ 38.26 |
| Set14  | 34.09 | 0.9221 | ≥ 34.09 |
| BSD100 | 32.36 | 0.9019 | ≥ 32.36 |
| Urban100 | 33.26 | 0.9378 | ≥ 33.26 |
| Manga109 | 39.35 | 0.9785 | ≥ 39.35 |

### x3 超分
| 测试集 | MambaIRv2-light PSNR | MambaIRv2-light SSIM | 目标 |
|--------|---------------------|---------------------|------|
| Set5   | 34.71 | 0.9298 | ≥ 34.71 |
| Set14  | 30.68 | 0.8483 | ≥ 30.68 |
| BSD100 | 29.26 | 0.8098 | ≥ 29.26 |
| Urban100 | 29.01 | 0.8689 | ≥ 29.01 |
| Manga109 | 34.41 | 0.9497 | ≥ 34.41 |

### x4 超分
| 测试集 | MambaIRv2-light PSNR | MambaIRv2-light SSIM | 目标 |
|--------|---------------------|---------------------|------|
| Set5   | 32.51 | 0.8992 | ≥ 32.51 |
| Set14  | 28.84 | 0.7878 | ≥ 28.84 |
| BSD100 | 27.75 | 0.7426 | ≥ 27.75 |
| Urban100 | 26.82 | 0.8079 | ≥ 26.82 |
| Manga109 | 31.24 | 0.9182 | ≥ 31.24 |

## 注意事项
- 论文 MambaIRv2-light 参数量: x2=774K, x3=781K, x4=790K
- 训练数据: DIV2K (800张)，与论文消融实验设置一致
- 训练迭代: 500K iter
- 单卡 4090 预计训练时间: 4-6小时（模型小，速度快）

## 启动命令（同样需要screen和conda）
```bash
screen -S mfa_light_x4
conda activate sr
cd ~/pyproject/MambaIR-main
export SWANLAB_API_KEY="o4MGQAOSX8rGztH69Jj5P"
python basicsr/train.py -opt options/train/mfamambaIR/train_MFAMambaIR_lightSR_x4.yml
```

## 验证模型是否正确加载
启动后应看到:
```
Network [MFAMambaIR] is created.
Network: MFAMambaIR, with parameters: ~800K-900K
```
如果参数量远大于 1M，说明配置有误。
