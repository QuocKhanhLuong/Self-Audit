#!/bin/bash
set -euo pipefail

CHECKPOINT="${1:?Usage: $0 <checkpoint_path>}"
SAVE_ROOT="./results/eval_acdc"

python eval_picie.py \
    --checkpoint "$CHECKPOINT" \
    --data_root "../../preprocessed_data/ACDC" \
    --save_root "$SAVE_ROOT" \
    --dataset acdc \
    --split_manifest "../../splits/acdc_patient_split_seed42.json" \
    --res 224 \
    --arch resnet18 \
    --in_dim 128 \
    --K_test 4 \
    --metric_test cosine \
    --batch_size 64 \
    --num_workers 4 \
    --seed 2021
