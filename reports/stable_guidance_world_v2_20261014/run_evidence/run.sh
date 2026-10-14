#!/usr/bin/env bash
set -Eeuo pipefail
source ${REPO_ROOT}/runs/stable_guidance_world_v2_acdc_4070_20261014_024155/environment.sh
cd ${REPO_ROOT}/runs/stable_guidance_world_v2_acdc_4070_20261014_024155/source
date --iso-8601=seconds > ${REPO_ROOT}/runs/stable_guidance_world_v2_acdc_4070_20261014_024155/STARTED_AT
set +e
${PYTHON_ENV}/bin/python -u ${REPO_ROOT}/runs/stable_guidance_world_v2_acdc_4070_20261014_024155/source/scripts/run_stable_guidance_wandb.py --tracking-dir ${REPO_ROOT}/runs/stable_guidance_world_v2_acdc_4070_20261014_024155/tracking --wandb-project self-audit-stable-guidance --wandb-entity nthne-hanoi-university-of-science-and-technology --wandb-run-name stable_guidance_world_v2_acdc_4070_20261014_024155 --wandb-mode online -- --dataset acdc --root ${REPO_ROOT}/data/ACDC/training --test-root ${REPO_ROOT}/data/ACDC/testing --split-manifest ${REPO_ROOT}/runs/stable_guidance_world_v2_acdc_4070_20261014_024155/SPLIT.json --config ${REPO_ROOT}/runs/stable_guidance_world_v2_acdc_4070_20261014_024155/CONFIG.json --out ${REPO_ROOT}/runs/stable_guidance_world_v2_acdc_4070_20261014_024155/run --device cuda --epochs 50 --batch-size 8 --tile-batch 1 --width 32 --lr 0.0003 --threads 4 --seed 42 --evaluation acdc --reference-root ${REPO_ROOT}/data/ACDC/training --checkpoint ${HF_CACHE}/hub/models--mathpluscode--CineMA/blobs/a841b7e9300d60ea8feb73e3a7d8c60ddedc1d1d49ec1049ebfcfa337c5d5590 --teacher-config ${HF_CACHE}/hub/models--mathpluscode--CineMA/blobs/8a465e51889187cf1056fc4feaef9a0182feabe4 --offline --orientation-policy world_left_v2 --diagnostics --allow-verified-acdc-frame-grid > ${REPO_ROOT}/runs/stable_guidance_world_v2_acdc_4070_20261014_024155/launcher.log 2>&1
job_rc=$?
printf "%s\n" "$job_rc" > ${REPO_ROOT}/runs/stable_guidance_world_v2_acdc_4070_20261014_024155/EXIT_CODE
date --iso-8601=seconds > ${REPO_ROOT}/runs/stable_guidance_world_v2_acdc_4070_20261014_024155/FINISHED_AT
exit "$job_rc"
