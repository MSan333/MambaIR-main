# MFAMambaIR 单卡训练问题记录与解决方案

## 环境信息

- GPU: NVIDIA RTX 4090 × 4，单卡 24GB
- 训练卡: GPU 0
- Python: 3.8
- PyTorch: 2.0.1, CUDA 11.7
- Conda 环境: `sr`

---

## 问题 1: CUDA Out of Memory (OOM)

### 现象

```
CUDA out of memory. Tried to allocate XX MiB (GPU 0; 23.52 GiB total capacity;
XX GiB already allocated; XX MiB free)
```

### 原因

MFAMambaIR 模型（embed_dim=174, depths=[6,6,6,6,6,6]）参数量大，训练时激活占用显存高。不同 scale 的显存需求：

| Scale | gt_size | batch_size | 预估单样本显存 | bs=8 | bs=4 | bs=2 |
|-------|---------|------------|--------------|------|------|------|
| x2 | 128 | 4 | ~5.7 GiB | ❌ | ✅ | ✅ |
| x4 | 256 | 2 | ~7-8 GiB | ❌ | ❌ | ✅ |

x4 输出 256×256，上采样中间激活是 x2 的约 4 倍，更吃显存。

### 解决方案

根据 scale 设置合适的 batch_size：

| Scale | 推荐 batch_size |
|-------|----------------|
| x2 | 4 |
| x4 | 2 |

配置文件位置：`options/train/mfamambaIR/train_MFAMambaIR_SR_x*.yml`

---

## 问题 2: AdamW 优化器不支持

### 现象

```
NotImplementedError: optimizer AdamW is not supperted yet.
```

### 原因

BasicSR 框架仅内置了 `Adam` 优化器，不支持 `AdamW`。

### 解决方案

将配置文件中的 `optim_g.type` 改为 `Adam`：

```yaml
train:
  optim_g:
    type: Adam          # 原来是 AdamW
    lr: !!float 2e-4
    weight_decay: 0
    betas: [0.9, 0.99]
```

---

## 问题 3: 数据集路径不匹配

### 现象

训练启动后找不到数据集文件。

### 原因

远端配置使用相对路径 `datasets/DF2K/HR`，本地数据在 `/home/guoshuaile/pyproject/data/` 下。

### 解决方案

修改配置文件中的数据集路径为绝对路径：

```yaml
datasets:
  train:
    dataroot_gt:
      - /home/guoshuaile/pyproject/data/DIV2K/DIV2K_train_HR
    dataroot_lq:
      - /home/guoshuaile/pyproject/data/DIV2K/DIV2K_train_LR_bicubic/X{scale}
  val:
    dataroot_gt: /home/guoshuaile/pyproject/data/SR/Set14/HR
    dataroot_lq: /home/guoshuaile/pyproject/data/SR/Set14/LR_bicubic/X{scale}
```

---

## 问题 4: SwanLab 横坐标为 iter 而非 epoch

### 现象

SwanLab 面板上 x 轴为 iteration 而非 epoch。

### 原因

`basicsr/utils/logger.py` 中 `swanlab.log` 的 `step` 参数传的是 `current_iter`。

### 解决方案

修改 `basicsr/utils/logger.py`，在 `log_metrics` 方法中记录 `self.current_epoch = epoch`，并将 `step` 改为 `self.current_epoch`。

---

## 问题 5: Git 推送网络问题

### 现象

```
fatal: unable to access 'https://github.com/...': GnuTLS recv error (-110)
ssh: connect to host github.com port 22: Connection timed out
Missing or invalid credentials (VS Code git credential helper)
```

### 原因

1. HTTPS 和 SSH 22 端口对 GitHub 直连不通
2. VS Code 的 git 凭证助手干扰认证

### 解决方案

配置 SSH 走 443 端口，并绕过 VS Code 凭证助手：

```bash
# 配置 SSH 走 443
cat >> ~/.ssh/config << 'EOF'
Host github.com
    Hostname ssh.github.com
    Port 443
    User git
EOF

# 改用 SSH 远程地址
git remote set-url origin git@github.com:MSan333/MambaIR-main.git

# 推送时绕过 VS Code 凭证助手
GIT_ASKPASS="" git push origin server/exp1
```

---

## 训练启动命令

### x2 超分

```bash
screen -S mfa_x2
conda activate sr
cd ~/pyproject/MambaIR-main
export SWANLAB_API_KEY="o4MGQAOSX8rGztH69Jj5P"
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/mfamambaIR/train_MFAMambaIR_SR_x2.yml 2>&1 | tee train_mfa_x2.log
```

### x4 超分

```bash
screen -S mfa_x4
conda activate sr
cd ~/pyproject/MambaIR-main
export SWANLAB_API_KEY="o4MGQAOSX8rGztH69Jj5P"
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/mfamambaIR/train_MFAMambaIR_SR_x4.yml 2>&1 | tee train_mfa_x4.log
```

按 `Ctrl+A D` detach screen。

### 查看训练进度

```bash
# 重新连接 screen
screen -r mfa_x4

# 查看日志
tail -f train_mfa_x4.log

# SwanLab 面板
# https://swanlab.cn/@sansan/MFAMambaIR
```

---

## 当前配置一览

| 配置项 | x2 | x4 |
|-------|----|----|
| 模型 | MFAMambaIR | MFAMambaIR |
| 训练集 | DIV2K | DIV2K |
| 验证集 | Set14 | Set14 |
| gt_size | 128 | 256 |
| batch_size | 4 | 2 |
| 优化器 | Adam | Adam |
| 学习率 | 2e-4 | 2e-4 |
| total_iter | 500K | 500K |
| SwanLab 项目 | MFAMambaIR | MFAMambaIR |
