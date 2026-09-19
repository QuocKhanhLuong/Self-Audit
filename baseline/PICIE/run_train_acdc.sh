#!/bin/bash
set -euo pipefail

DATA_ROOT="../../preprocessed_data/ACDC"
SAVE_ROOT="./results/acdc"
SPLIT_MANIFEST="../../splits/acdc_patient_split_seed42.json"

python train_picie.py \
    --data_root "$DATA_ROOT" \
    --save_root "$SAVE_ROOT" \
    --dataset acdc \
    --split_manifest "$SPLIT_MANIFEST" \
    --res 224 --res1 224 --res2 224 \
    --arch resnet18 \
    --in_dim 128 \
    --K_train 4 --K_test 4 \
    --metric_train cosine --metric_test cosine \
    --batch_size_cluster 64 \
    --batch_size_train 32 \
    --batch_size_test 32 \
    --num_epoch 10 \
    --lr 1e-4 \
    --optim_type Adam \
    --seed 2021 \
    --num_workers 4 \
    --equiv \
    --h_flip --v_flip --random_crop \
    --augment \
    "$@"
