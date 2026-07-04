#!/bin/bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate sr
cd /home/guoshuaile/pyproject/MambaIR-main
unset PYTORCH_CUDA_ALLOC_CONF

# 二次续训: 从 best model (30k iter, PSNR 28.8499) 以 lr=5e-5 续训 50k
# LR衰减: 0-30k@5e-5 -> 30k-40k@2.5e-5 -> 40k-50k@1.25e-5
# 只保存 latest 和 best 模型
screen -dmS mfa_train bash -c 'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate sr && cd /home/guoshuaile/pyproject/MambaIR-main && unset PYTORCH_CUDA_ALLOC_CONF && CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/mfamambaIR/train_MFAMambaIR_lightSR_x4_resume2.yml 2>&1 | tee train_mfa_light_x4_resume2.log'
