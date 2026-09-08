#!/usr/bin/env bash

# FairMoE Bail experiment.

set -e

CUDA_VISIBLE_DEVICES=1 python train.py \
  --dataset bail \
  --seed_num 5 \
  --epochs 1000 \
  --auto_profile \
  --acc_combo gat_mlp \
  --save_results True
