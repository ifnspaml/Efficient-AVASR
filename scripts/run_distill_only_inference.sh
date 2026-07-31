#!/usr/bin/env bash
PROJECT=/home/zhengyangli/work_fast/dpav_hubert_new
CKPT="${PROJECT}/exp/chapter3_distill_only/c3b_t12_layer_to_layer/seed_1337/finetune_runs/lr5e-4-0005/checkpoints/checkpoint_best.pt"

sbatch \
  --export=ALL,CH3_PROJECT_PATH="${PROJECT}" \
  scripts/run_distill_only.sh \
  --exp-name c3b_t12_layer_to_layer \
  --stage evaluate \
  --run-name lr5e-4-0005 \
  --input-checkpoint "${CKPT}" \
  --eval-name itut-screening-valid \
  --evaluation-phase screening \
  --eval-subsets valid