#!/usr/bin/env bash
PROJECT=/home/zhengyangli/work_fast/dpav_hubert_new
CKPT="${PROJECT}/exp/chapter3_distill_only/c3d_t12_historical_heads_noisy_target12/seed_1337/finetune_runs/jointdp-ft-lr5e-4-001/checkpoints/checkpoint_best.pt"

sbatch \
  --export=ALL,CH3_PROJECT_PATH="${PROJECT}" \
  scripts/run_distill_only.sh \
  --exp-name c3d_t12_historical_heads_noisy_target12 \
  --stage evaluate \
  --run-name jointdp-ft-lr5e-4-001 \
  --input-checkpoint "${CKPT}" \
  --eval-name itut-screening-valid \
  --evaluation-phase screening \
  --eval-subsets valid