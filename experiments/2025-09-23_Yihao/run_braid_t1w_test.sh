#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python experiments/2025-09-23_Yihao/main.py \
    --model_name braid_t1w \
    --stage test \
    --test_model_pth experiments/2025-09-23_Yihao/outputs/braid_t1w_resume/weights/epoch-18.pth \
    --dataset_local_root /home-local/gaoc11/metavoxel \
    --output_dir experiments/2025-09-23_Yihao/outputs/braid_t1w_resume