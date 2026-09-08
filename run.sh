#!/usr/bin/env bash

# FairMoE Bail experiment.
# This is the configuration recorded in the project logs for:
# AUCROC 90.49 +/- 0.58, F1 80.70 +/- 0.87.

set -e

CUDA_VISIBLE_DEVICES=1 python train.py \
  --dataset bail \
  --seed_num 5 \
  --epochs 1000 \
  --auto_profile \
  --acc_combo gat_mlp \
  --save_results True
