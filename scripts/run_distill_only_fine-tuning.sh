#!/usr/bin/env bash
#SBATCH --time=8-00:00:00
#SBATCH --partition=ifn
#SBATCH --gres=gpu:pro6000b_96gb:1
#SBATCH --exclude=gpu[04,05]
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48gb
#SBATCH --job-name=c3a-ft-lr5e-4

set -euo pipefail

PROJECT=/home/zhengyangli/work_fast/dpav_hubert_new

INPUT_CHECKPOINT="${PROJECT}/exp/chapter3_distill_only/b1_t2_teacher_init/seed_1337/export/student.pt"

OUTPUT_DIR="${PROJECT}/exp/chapter3_distill_only/b1_t2_teacher_init/seed_1337/finetune_runs/lr5e-4-0005"

DATA=/beegfs/data/shared/lrs3/433h_data_avhubert
TOKENIZER=/beegfs/data/shared/lrs3/spm1000/spm_unigram1000.model
NOISE=/beegfs/data/shared/lrs3/noise/musan/tsv/all

source /home/zhengyangli/anaconda3/etc/profile.d/conda.sh
conda activate dpavhubert_pro6000

cd "${PROJECT}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/fairseq:${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"

export WANDB_NAME=b1_t2-teacher-init-finetune-lr5e-4-0005

if [[ ! -f "${INPUT_CHECKPOINT}" ]]; then
    echo "Missing input checkpoint: ${INPUT_CHECKPOINT}" >&2
    exit 1
fi

if [[ -d "${OUTPUT_DIR}" ]] && \
   [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Output directory is already nonempty: ${OUTPUT_DIR}" >&2
    exit 1
fi

fairseq-hydra-train \
    --config-dir "${PROJECT}/avhubert/conf/av-finetune" \
    --config-name base_noise_pt_noise_ft_433h.yaml \
    "task.data=${DATA}" \
    "task.label_dir=${DATA}" \
    "task.tokenizer_bpe_model=${TOKENIZER}" \
    "task.noise_wav=${NOISE}" \
    "task.noise_prob=0.25" \
    "task.noise_snr=0" \
    "model.w2v_path=${INPUT_CHECKPOINT}" \
    "distributed_training.distributed_world_size=1" \
    "distributed_training.nprocs_per_node=1" \
    "optimization.update_freq=[8]" \
    "optimization.lr=[0.0005]" \
    "dataset.num_workers=24" \
    "common.seed=1337" \
    "common.user_dir=${PROJECT}/chapter3_distill_only" \
    "hydra.run.dir=${OUTPUT_DIR}"
