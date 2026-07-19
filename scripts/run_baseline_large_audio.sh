#!/bin/bash
#SBATCH --time=8-00:00:00
#SBATCH --partition=ifn
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=gpu[04,05]
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --begin=now
#SBATCH --job-name=dpav_hubert_teacher_a_lg
#SBATCH --mem=48gb

# Audio-only AV-HuBERT large teacher fine-tune + clean infer + parallel ITUT eval.
# Results: exp/finetune/asr-audio/teacher_large/

PYTHON_VIRTUAL_ENVIRONMENT=dpavhubert
CONDA_ROOT=/home/zhengyangli/anaconda3/
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

set -euo pipefail

# config
exp_name=teacher_large
project_path=/beegfs/work_fast/zhengyangli/dpav_hubert
avhubert_dir=${project_path}/avhubert
cd "${project_path}"

# wandb
wandb_project=dpav-hubert
export WANDB_RUN_GROUP=${exp_name}-audio

teacher_ckpt=/home/zhengyangli/work/av_hubert_pre_trained_models/phd_thesis/large_vox_iter5.pt

gpus=1
workers=4

free_gpu=$CUDA_VISIBLE_DEVICES

# dataset config
data_path=/beegfs/data/shared/lrs3/433h_data_avhubert
tokenizer_ckpt=/beegfs/data/shared/lrs3/spm1000/spm_unigram1000.model
noise_path=/beegfs/data/shared/lrs3/noise/musan/tsv/all
noise_prob=0.25
noise_snr=0

# finetune audio-only ASR config
finetune_exp_path=${project_path}/exp/finetune/asr-audio/${exp_name}
finetune_config_path=${project_path}/avhubert/conf/a-finetune/
finetune_config_name=large_noise_pt_noise_ft_433h.yaml
finetune_update_freq=8
finetune_use_noise=true

# infer config
infer_config_path=${project_path}/avhubert/conf/
infer_config_name=s2s_decode.yaml
infer_datasets="test valid"
infer_modalities="['audio']"
infer_noise_method=${INFER_NOISE_METHOD:-itut}
infer_noise_snr="-10 -5 0 5 10"

finetune_noise_opts="task.noise_wav=Null task.noise_prob=0.0"
if $finetune_use_noise; then
    finetune_noise_opts="task.noise_wav=${noise_path} task.noise_prob=${noise_prob} task.noise_snr=${noise_snr}"
fi

# finetune ASR (audio-only)
export WANDB_NAME=${exp_name}-audio-finetune
CUDA_VISIBLE_DEVICES=$free_gpu fairseq-hydra-train \
    --config-dir ${finetune_config_path} \
    --config-name ${finetune_config_name} \
    task.data=${data_path} \
    task.label_dir=${data_path} \
    ${finetune_noise_opts} \
    task.tokenizer_bpe_model=${tokenizer_ckpt} \
    model.w2v_path=${teacher_ckpt} \
    distributed_training.distributed_world_size=${gpus} \
    distributed_training.nprocs_per_node=${gpus} \
    optimization.update_freq=[${finetune_update_freq}] \
    dataset.num_workers=${workers} \
    common.wandb_project=${wandb_project} \
    hydra.run.dir=${finetune_exp_path} \
    common.user_dir=${avhubert_dir} || exit 1;

# infer clean (sequential; 2 splits)
for dataset in $infer_datasets; do
    python -B ${avhubert_dir}/infer_s2s.py \
        --config-dir ${infer_config_path} \
        --config-name ${infer_config_name} \
        dataset.gen_subset=${dataset} \
        common_eval.path=${finetune_exp_path}/checkpoints/checkpoint_best.pt \
        common_eval.results_path=${finetune_exp_path}/infer/clean/${dataset} \
        override.modalities=${infer_modalities} \
        hydra.run.dir=${finetune_exp_path}/infer/clean/${dataset} \
        common.user_dir=${avhubert_dir} || exit 1;
done

# infer ITUT noisy in parallel: -10 -5 0 5 10 × babble/music/speech × test/valid
echo "Submitting parallel ITUT noisy inference jobs..."
EXP_ROOT=exp/finetune/asr-audio \
INFER_MODALITIES="${infer_modalities}" \
INFER_NOISE_METHOD="${infer_noise_method}" \
bash scripts/run_infer_itut_teacher_snr_ext.sh ${exp_name} ${infer_noise_snr}

echo "Done. Clean under ${finetune_exp_path}/infer/clean/; noisy jobs submitted to infer_itut/."
