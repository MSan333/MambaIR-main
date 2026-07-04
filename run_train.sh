#!/bin/bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate sr
cd /home/guoshuaile/pyproject/MambaIR-main
unset PYTORCH_CUDA_ALLOC_CONF

# MFAMambaIR_v3: MFA + DSTA 集成训练
# bs=16, total_iter=300k, lr=2e-4, milestones=[150k,200k,250k,275k], gamma=0.5
# 只保存 latest 和 best 模型
screen -dmS mfa_v3_train bash -c 'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate sr && cd /home/guoshuaile/pyproject/MambaIR-main && unset PYTORCH_CUDA_ALLOC_CONF && CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/mfamambaIR/train_MFAMambaIR_v3_lightSR_x4.yml 2>&1 | tee train_mfa_v3_light_x4.log'
