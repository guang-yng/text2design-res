#!/bin/bash
# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
#     -t --gpus 0,1,2,3 -r logs/openimage-ldm-sr/checkpoints/last.ckpt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py \
    -t --gpus 0,1,2,3,4,5,6,7 -b configs/latent-diffusion/design_all_1208_super_13w.yaml
