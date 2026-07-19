#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --partition=ifn
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=gpu[04,05]
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --begin=now
#SBATCH --job-name=dpav_infer_itut_table2
#SBATCH --mem=48gb

# ITU-T P.56 noisy inference for Table 2 experiments 3, 4, 5.
# Results: exp/finetune/asr/<exp_name>/infer_itut/{noise}/{snr}/{dataset}/
# Overrides model.w2v_path to the copied distill final encoder (old Reichert paths are missing).

PYTHON_VIRTUAL_ENVIRONMENT=dpavhubert
CONDA_ROOT=/home/zhengyangli/anaconda3/
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

set -euo pipefail

project_path=/beegfs/work_fast/zhengyangli/dpav_hubert
avhubert_dir=${project_path}/avhubert

infer_config_path=${project_path}/avhubert/conf/
infer_config_name=s2s_decode.yaml
infer_datasets="test valid"
infer_noise_types="babble music speech"
infer_noise_snr="-10 10"
infer_noise_method=itut
infer_root=infer_itut

# Table 2 IDs 3, 4, 5: finetune_exp_name -> distill final encoder
declare -A w2v_paths=(
    [baseline_l1_weight_0.1_main_0.002_warmup_noise]=${project_path}/exp/distill/opt_baseline_noise/final/checkpoints/pruned_checkpoint_last_final.pt
    [resnet_noise]=${project_path}/exp/distill/resnet_noise/final/checkpoints/pruned_checkpoint_last_final.pt
    [resnet80_noise]=${project_path}/exp/distill/resnet80_noise/final/checkpoints/pruned_checkpoint_last_final.pt
)

exp_names=(
    baseline_l1_weight_0.1_main_0.002_warmup_noise
    resnet_noise
    resnet80_noise
)

free_gpu=$CUDA_VISIBLE_DEVICES

echo "ITUT inference for Table 2 experiments 3/4/5"
echo "project_path=${project_path}"
echo "infer_root=${infer_root}"
echo "noise_method=${infer_noise_method}"

for exp_name in "${exp_names[@]}"; do
    finetune_exp_path=${project_path}/exp/finetune/asr/${exp_name}
    ckpt=${finetune_exp_path}/checkpoints/checkpoint_best.pt
    w2v_path=${w2v_paths[$exp_name]}

    if [ ! -f "${ckpt}" ]; then
        echo "Error: missing finetune checkpoint for ${exp_name}: ${ckpt}" >&2
        exit 1
    fi
    if [ ! -f "${w2v_path}" ]; then
        echo "Error: missing distill encoder for ${exp_name}: ${w2v_path}" >&2
        exit 1
    fi

    # Override baked-in Reichert w2v_path inside checkpoint_best.pt.
    # Quote as a Hydra string so '{' is not parsed as Hydra dict grammar.
    model_overrides="{'w2v_path': '${w2v_path}'}"

    echo "=== Experiment: ${exp_name} ==="
    echo "checkpoint: ${ckpt}"
    echo "w2v_path override: ${w2v_path}"

    for noise in $infer_noise_types; do
        if [ "$noise" != speech ]; then
            infer_noise_path=/beegfs/data/shared/lrs3/noise/musan/tsv/${noise}
        else
            infer_noise_path=/beegfs/data/shared/lrs3/noise/${noise}
        fi
        for snr in $infer_noise_snr; do
            for dataset in $infer_datasets; do
                echo "  ${noise} SNR=${snr} split=${dataset}"
                python -B ${avhubert_dir}/infer_s2s.py \
                    --config-dir ${infer_config_path} \
                    --config-name ${infer_config_name} \
                    dataset.gen_subset=${dataset} \
                    common_eval.path=${ckpt} \
                    common_eval.results_path=${finetune_exp_path}/${infer_root}/${noise}/${snr}/${dataset} \
                    "common_eval.model_overrides=\"${model_overrides}\"" \
                    override.modalities=['audio','video'] \
                    override.noise_wav=${infer_noise_path} \
                    override.noise_prob=1 \
                    override.noise_snr=${snr} \
                    override.noise_method=${infer_noise_method} \
                    hydra.run.dir=${finetune_exp_path}/${infer_root}/${noise}/${snr}/${dataset} \
                    common.user_dir=${avhubert_dir} || exit 1
            done
        done
    done
done

echo "Done. Results under exp/finetune/asr/<exp_name>/${infer_root}/"
