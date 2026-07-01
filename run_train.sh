#!/bin/bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate sr
cd /home/guoshuaile/pyproject/MambaIR-main
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/mambair/train_MambaIR_SR_x2_1gpu.yml 2>&1 | tee train.log
