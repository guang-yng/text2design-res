#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    -t --gpus 0, -r logs/openimage-ldm-sr/checkpoints/last.ckpt