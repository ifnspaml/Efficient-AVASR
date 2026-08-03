#!/usr/bin/env bash
#SBATCH --time=8-00:00:00
#SBATCH --partition=ifn
#SBATCH --gres=gpu:pro6000b_96gb:1
#SBATCH --exclude=gpu[04,05]
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48gb
#SBATCH --job-name=joint-dp-ft-lr5e-4

set -euo pipefail

PROJECT=/home/zhengyangli/work_fast/dpav_hubert_new
OUT="${PROJECT}/exp/finetune/asr/resnet_lr5e-4"
CKPT="${PROJECT}/exp/distill/resnet/final/checkpoints/pruned_checkpoint_last_final.pt"

source /home/zhengyangli/anaconda3/etc/profile.d/conda.sh
conda activate dpavhubert_pro6000
cd "${PROJECT}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/fairseq:${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_NAME=resnet-finetune-lr5e-4

fairseq-hydra-train \
  --config-dir "${PROJECT}/avhubert/conf/av-finetune" \
  --config-name base_noise_pt_noise_ft_433h.yaml \
  task.data=/beegfs/data/shared/lrs3/433h_data_avhubert \
  task.label_dir=/beegfs/data/shared/lrs3/433h_data_avhubert \
  task.tokenizer_bpe_model=/beegfs/data/shared/lrs3/spm1000/spm_unigram1000.model \
  task.noise_wav=/beegfs/data/shared/lrs3/noise/musan/tsv/all \
  task.noise_prob=0.25 \
  task.noise_snr=0 \
  model.w2v_path="${CKPT}" \
  distributed_training.distributed_world_size=1 \
  distributed_training.nprocs_per_node=1 \
  'optimization.update_freq=[8]' \
  'optimization.lr=[0.0005]' \
  dataset.num_workers=24 \
  common.seed=1337 \
  common.wandb_project=dpav-hubert \
  common.user_dir="${PROJECT}/avhubert" \
  "hydra.run.dir=${OUT}"
