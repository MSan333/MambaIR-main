# MFAMambaIR 训练启动指南

## 快速启动（完整流程）

### 1. 创建 screen 会话（防止SSH断开后训练中断）

```bash
# 创建新的 screen 会话
screen -S mfa_train

# 如果已有会话，恢复连接
screen -r mfa_train
```

### 2. 激活 conda 环境

```bash
conda activate sr
```

### 3. 进入项目目录

```bash
cd ~/pyproject/MambaIR-main
```

### 4. 设置环境变量

```bash
export SWANLAB_API_KEY="o4MGQAOSX8rGztH69Jj5P"
```

### 5. 启动训练

```bash
# ========================================
# MFAMambaIR（创新点版本）— 正确的启动命令
# ========================================

# x2 超分
python basicsr/train.py -opt options/train/mfamambaIR/train_MFAMambaIR_SR_x2.yml

# x3 超分
python basicsr/train.py -opt options/train/mfamambaIR/train_MFAMambaIR_SR_x3.yml

# x4 超分
python basicsr/train.py -opt options/train/mfamambaIR/train_MFAMambaIR_SR_x4.yml
```

### 6. 从 screen 会话中分离（训练在后台继续）

```
按 Ctrl+A 然后按 D（分离会话）
```

---

## 一键启动脚本

复制以下命令到终端即可一键启动 x2 训练：

```bash
screen -dmS mfa_x2 bash -c 'conda activate sr && cd ~/pyproject/MambaIR-main && export SWANLAB_API_KEY="o4MGQAOSX8rGztH69Jj5P" && python basicsr/train.py -opt options/train/mfamambaIR/train_MFAMambaIR_SR_x2.yml 2>&1 | tee logs/mfa_x2.log'
```

---

## ⚠️ 注意：不要用错配置文件！

| 配置文件 | 模型 | 用途 |
|---------|------|------|
| ✅ `options/train/mfamambaIR/train_MFAMambaIR_SR_x2.yml` | **MFAMambaIR** | 创新点训练 |
| ❌ `options/train/mambair/train_MambaIR_SR_x2.yml` | MambaIR v1 | 旧baseline |
| ❌ `options/train/mambairv2/train_MambaIRv2_SR_x2.yml` | MambaIRv2 | 论文baseline |

### 如何确认跑的是正确模型？

启动后观察日志，应该看到：
```
Network [MFAMambaIR] is created.
```

如果看到以下内容，说明用错了配置：
- `Network [MambaIR] is created.` → 用了 v1 配置
- `Network [MambaIRv2] is created.` → 用了 v2 配置（无创新点）

---

## Baseline 对照组启动命令

完成 MFAMambaIR 训练后，需要跑 baseline 对照组做对比：

```bash
# MambaIRv2 baseline（用于和MFAMambaIR对比）
screen -S baseline_x2
conda activate sr
cd ~/pyproject/MambaIR-main
export SWANLAB_API_KEY="o4MGQAOSX8rGztH69Jj5P"
python basicsr/train.py -opt options/train/mambairv2/train_MambaIRv2_SR_x2.yml
```

---

## 断点续训

如果训练中断，直接重新运行同样的命令即可（需在配置文件中添加 `auto_resume: true`）：

```bash
# 配置文件中添加:
# auto_resume: true

# 然后重新运行同样命令
python basicsr/train.py -opt options/train/mfamambaIR/train_MFAMambaIR_SR_x2.yml
```

---

## 查看训练进度

```bash
# 重新连接 screen 会话
screen -r mfa_train

# 查看 SwanLab 在线面板
# https://swanlab.cn/@sansan/MambaIR

# 查看日志文件
tail -f experiments/MFAMambaIR_SR_x2/train_MFAMambaIR_SR_x2_*.log
```

---

## Screen 常用操作

| 操作 | 命令 |
|------|------|
| 创建会话 | `screen -S name` |
| 分离会话 | `Ctrl+A, D` |
| 恢复会话 | `screen -r name` |
| 列出所有会话 | `screen -ls` |
| 终止会话 | `exit`（在会话内） |
