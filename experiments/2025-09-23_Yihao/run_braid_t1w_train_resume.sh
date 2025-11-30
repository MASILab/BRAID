#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python experiments/2025-09-23_Yihao/main.py \
    --model_name braid_t1w \
    --stage train \
    --n_epochs 20 \
    --resume_pth experiments/2025-09-23_Yihao/outputs/braid_t1w/weights/epoch-10_val-loss-36.8717.pth \
    --dataset_local_root /home-local/gaoc11/metavoxel \
    --output_dir experiments/2025-09-23_Yihao/outputs/braid_t1w_resume