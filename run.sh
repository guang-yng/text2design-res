#!/bin/bash
# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
#     -t --gpus 0,1,2,3 -r logs/openimage-ldm-sr/checkpoints/last.ckpt

CUDA_VISIBLE_DEVICES=0 python main.py \
    -t --gpus 0, -b configs/latent-diffusion/demo-ldm-sr.yaml