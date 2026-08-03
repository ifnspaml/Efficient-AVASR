#!/bin/bash
# Local launcher: submit one Slurm job per condition for parallel inference.
# Also submits clean (valid/test) unless INFER_SKIP_CLEAN=1.
#
# Usage (run on login node, do NOT sbatch this wrapper):
#   bash scripts/run_infer_itut_teacher_snr_ext.sh [exp_name] [snrs...]
#
# Environment:
#   EXP_ROOT           experiment root under project (default: exp/finetune/asr)
#                      audio-only: EXP_ROOT=exp/finetune/a
#   INFER_MODALITIES   Hydra list for override.modalities (default: ['audio','video'])
#                      audio-only: INFER_MODALITIES="['audio']"
#   INFER_NOISE_METHOD rms | itut | p56 (default: itut)
#   INFER_NOISE_SNR    used when no SNR args are passed (default: -10 10)
#   INFER_SKIP_CLEAN   set to 1 to skip clean jobs
#   INFER_ONLY_CLEAN   set to 1 to submit only clean jobs (skip noisy)
#
# Examples:
#   bash scripts/run_infer_itut_teacher_snr_ext.sh teacher_large
#   bash scripts/run_infer_itut_teacher_snr_ext.sh teacher -10 10
#   INFER_SKIP_CLEAN=1 bash scripts/run_infer_itut_teacher_snr_ext.sh resnet_lr5e-4 0
#   INFER_ONLY_CLEAN=1 bash scripts/run_infer_itut_teacher_snr_ext.sh resnet_lr5e-4
#   EXP_ROOT=exp/finetune/a INFER_MODALITIES="['audio']" \
#     bash scripts/run_infer_itut_teacher_snr_ext.sh teacher -10 -5 0 5 10
#
# Results:
#   clean: ${EXP_ROOT}/<exp_name>/infer/clean/{dataset}/
#   noisy: ${EXP_ROOT}/<exp_name>/infer_itut/{noise}/{snr}/{dataset}/
#
# Set INFER_SKIP_CLEAN=1 to submit only noisy jobs.
# Set INFER_ONLY_CLEAN=1 to submit only clean jobs.

set -euo pipefail

exp_name=${1:-teacher_large}
shift || true
if [ "$#" -gt 0 ]; then
    infer_noise_snr="$*"
else
    infer_noise_snr=${INFER_NOISE_SNR:--10 -5 0 5 10}
fi

project_path=/beegfs/work_fast/zhengyangli/dpav_hubert_new
avhubert_dir=${project_path}/avhubert
exp_root=${EXP_ROOT:-exp/finetune/asr}
# Allow absolute EXP_ROOT or path relative to project_path.
case "${exp_root}" in
    /*) finetune_exp_path=${exp_root}/${exp_name} ;;
    *)  finetune_exp_path=${project_path}/${exp_root}/${exp_name} ;;
esac
ckpt=${finetune_exp_path}/checkpoints/checkpoint_best.pt

infer_config_path=${project_path}/avhubert/conf/
infer_config_name=s2s_decode.yaml
infer_datasets="test valid"
infer_noise_types="babble music speech"
infer_noise_method=${INFER_NOISE_METHOD:-itut}
infer_modalities=${INFER_MODALITIES:-"['audio','video']"}
infer_root=infer
[ "${infer_noise_method}" != "rms" ] && infer_root="infer_${infer_noise_method}"
infer_skip_clean=${INFER_SKIP_CLEAN:-0}
infer_only_clean=${INFER_ONLY_CLEAN:-0}
if [ "${infer_only_clean}" = "1" ]; then
    infer_skip_clean=0
fi

PYTHON_VIRTUAL_ENVIRONMENT=dpavhubert_pro6000
CONDA_ROOT=/home/zhengyangli/anaconda3/

slurm_gres=${SLURM_GRES:-gpu:a100:1}
slurm_partition=${SLURM_PARTITION:-ifn}
slurm_qos=${SLURM_QOS:-low}

if [ ! -f "${ckpt}" ]; then
    echo "Error: missing finetune checkpoint: ${ckpt}" >&2
    exit 1
fi

sbatch_qos_args=()
if [ -n "${slurm_qos}" ]; then
    sbatch_qos_args=(--qos="${slurm_qos}")
fi

echo "Submitting parallel inference jobs"
echo "exp_name=${exp_name}"
echo "exp_root=${exp_root}"
echo "project_path=${project_path}"
echo "checkpoint=${ckpt}"
echo "infer_root=${infer_root}"
echo "modalities=${infer_modalities}"
echo "noise_method=${infer_noise_method}"
echo "noise_snr=${infer_noise_snr}"
echo "skip_clean=${infer_skip_clean}"
echo "only_clean=${infer_only_clean}"
echo "gres=${slurm_gres} partition=${slurm_partition} qos=${slurm_qos:-none}"

n_submitted=0

# Clean: no noise overrides (same layout as joint-DP / run_pruning.sh).
if [ "${infer_skip_clean}" != "1" ]; then
    for dataset in $infer_datasets; do
        job_name="clean_${exp_name}_${dataset}"
        results_path=${finetune_exp_path}/infer/clean/${dataset}

        echo "  sbatch ${job_name} -> ${results_path}"
        sbatch --job-name="${job_name}" \
            --time=1-00:00:00 \
            --partition="${slurm_partition}" \
            "${sbatch_qos_args[@]}" \
            --gres="${slurm_gres}" \
            --exclude=gpu[04,05] \
            --cpus-per-task=2 \
            --ntasks-per-node=1 \
            --mem=48gb \
            --export=ALL \
            <<EOF
#!/bin/bash
set -euo pipefail
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate ${PYTHON_VIRTUAL_ENVIRONMENT}

echo "Job: ${job_name}"
echo "exp=${exp_name} condition=clean dataset=${dataset}"
echo "modalities=${infer_modalities}"
echo "CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES}"

python -B ${avhubert_dir}/infer_s2s.py \\
    --config-dir ${infer_config_path} \\
    --config-name ${infer_config_name} \\
    dataset.gen_subset=${dataset} \\
    common_eval.path=${ckpt} \\
    common_eval.results_path=${results_path} \\
    override.modalities=${infer_modalities} \\
    hydra.run.dir=${results_path} \\
    common.user_dir=${avhubert_dir}
EOF
        n_submitted=$((n_submitted + 1))
    done
fi

if [ "${infer_only_clean}" != "1" ]; then
    for noise in $infer_noise_types; do
        if [ "$noise" != speech ]; then
            infer_noise_path=/beegfs/data/shared/lrs3/noise/musan/tsv/${noise}
        else
            infer_noise_path=/beegfs/data/shared/lrs3/noise/${noise}
        fi
        for snr in $infer_noise_snr; do
            for dataset in $infer_datasets; do
                # Short job name; keep unique across exp/noise/snr/split.
                job_name="itut_${exp_name}_${noise}_${snr}_${dataset}"
                results_path=${finetune_exp_path}/${infer_root}/${noise}/${snr}/${dataset}

                echo "  sbatch ${job_name} -> ${results_path}"
                sbatch --job-name="${job_name}" \
                    --time=1-00:00:00 \
                    --partition="${slurm_partition}" \
                    "${sbatch_qos_args[@]}" \
                    --gres="${slurm_gres}" \
                    --exclude=gpu[04,05] \
                    --cpus-per-task=2 \
                    --ntasks-per-node=1 \
                    --mem=48gb \
                    --export=ALL \
                    <<EOF
#!/bin/bash
set -euo pipefail
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate ${PYTHON_VIRTUAL_ENVIRONMENT}

echo "Job: ${job_name}"
echo "exp=${exp_name} noise=${noise} snr=${snr} dataset=${dataset}"
echo "modalities=${infer_modalities}"
echo "CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES}"

python -B ${avhubert_dir}/infer_s2s.py \\
    --config-dir ${infer_config_path} \\
    --config-name ${infer_config_name} \\
    dataset.gen_subset=${dataset} \\
    common_eval.path=${ckpt} \\
    common_eval.results_path=${results_path} \\
    override.modalities=${infer_modalities} \\
    override.noise_wav=${infer_noise_path} \\
    override.noise_prob=1 \\
    override.noise_snr=${snr} \\
    override.noise_method=${infer_noise_method} \\
    hydra.run.dir=${results_path} \\
    common.user_dir=${avhubert_dir}
EOF
                n_submitted=$((n_submitted + 1))
            done
        done
    done
fi

echo "Submitted ${n_submitted} jobs."
echo "  clean -> ${finetune_exp_path}/infer/clean/"
if [ "${infer_only_clean}" != "1" ]; then
    echo "  noisy -> ${finetune_exp_path}/${infer_root}/"
fi
