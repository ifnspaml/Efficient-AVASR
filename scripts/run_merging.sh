#!/bin/bash
#SBATCH --time=4-00:00:00
#SBATCH --partition=ifn
#SBATCH --gres=gpu:pro6000b_96gb:1
#SBATCH --exclude=gpu[04,05]
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --begin=now
#SBATCH --job-name=dpav_hubert
#SBATCH --mem=48gb



PYTHON_VIRTUAL_ENVIRONMENT=dpavhubert_pro6000
CONDA_ROOT=/home/zhengyangli/anaconda3/
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT


# config
exp_name=merging
project_path=/beegfs/work_fast/zhengyangli/dpav_hubert_new

exp_path=${project_path}/exp/distill/${exp_name}

teacher_ckpt=${project_path}/avhubert/pretrained/base_vox_iter5.pt
student_ckpt=${teacher_ckpt}

gpus=1
workers=4

free_gpu=$CUDA_VISIBLE_DEVICES

log_level=INFO

# dataset config
data_path=/beegfs/data/shared/lrs3/433h_data_avhubert
tokenizer_ckpt=/beegfs/data/shared/lrs3/spm1000/spm_unigram1000.model
noise_path=/beegfs/data/shared/lrs3/noise/musan/tsv/all
noise_prob=0.25
noise_snr=0

# dm config
config_path=${project_path}/avhubert/conf/distill/
config_name=merging.yaml
distill_config_path=${config_path}
distill_config_name=distill.yaml
update_freq=1
use_noise=false
merge_type=QKV

# finetune asr config
finetune_exp_path=${project_path}/exp/finetune/asr/${exp_name}
finetune_config_path=${project_path}/avhubert/conf/av-finetune/
finetune_config_name=base_noise_pt_noise_ft_433h.yaml
finetune_update_freq=8
finetune_use_noise=true

# infer config
infer_config_path=${project_path}/avhubert/conf/
infer_config_name=s2s_decode.yaml
infer_datasets="test valid"
infer_noise_types="babble music speech"
infer_noise_snr="-5 0 5"
infer_noise_method=${INFER_NOISE_METHOD:-rms}
infer_root=infer
[ "${infer_noise_method}" != "rms" ] && infer_root="infer_${infer_noise_method}"

noise_opts="task.noise_wav=Null task.noise_prob=0.0"
if $use_noise; then
    noise_opts="task.noise_wav=${noise_path} task.noise_prob=${noise_prob} task.noise_snr=${noise_snr}"
fi

finetune_noise_opts="task.noise_wav=Null task.noise_prob=0.0"
if $finetune_use_noise; then
    finetune_noise_opts="task.noise_wav=${noise_path} task.noise_prob=${noise_prob} task.noise_snr=${noise_snr}"
fi

# distillation and pruning
CUDA_VISIBLE_DEVICES=$free_gpu fairseq-hydra-train \
    --config-dir ${config_path} \
    --config-name ${config_name} \
    task.data=${data_path} \
    task.label_dir=${data_path} \
    ${noise_opts} \
    task.tokenizer_bpe_model=${tokenizer_ckpt} \
    model.teacher_path=${teacher_ckpt} \
    model.student_path=${student_ckpt} \
    distributed_training.distributed_world_size=${gpus} \
    distributed_training.nprocs_per_node=${gpus} \
    optimization.update_freq=[${update_freq}] \
    dataset.num_workers=${workers} \
    hydra.run.dir=${exp_path} \
    common.user_dir=../avhubert || exit 1;

# merge
python ../avhubert/merge.py \
    --distilled_ckpt ${exp_path}/checkpoints/checkpoint_last.pt\
    --original_ckpt ${teacher_ckpt} \
    --merge_type ${merge_type} \
    --log-level ${log_level} \
    2>&1 | tee ${exp_path}/merge.log || exit 1;

# final distillation
CUDA_VISIBLE_DEVICES=$free_gpu fairseq-hydra-train \
    --config-dir ${distill_config_path} \
    --config-name ${distill_config_name} \
    task.data=${data_path} \
    task.label_dir=${data_path} \
    ${noise_opts} \
    task.tokenizer_bpe_model=${tokenizer_ckpt} \
    model.teacher_path=${teacher_ckpt} \
    model.student_path=${exp_path}/checkpoints/merged_checkpoint_last.pt \
    distributed_training.distributed_world_size=${gpus} \
    distributed_training.nprocs_per_node=${gpus} \
    optimization.update_freq=[${update_freq}] \
    dataset.num_workers=${workers} \
    hydra.run.dir=${exp_path}/final \
    common.user_dir=../avhubert || exit 1;

# save final model and config
python ../avhubert/save_final_ckpt.py \
    --distilled_ckpt ${exp_path}/final/checkpoints/checkpoint_last.pt\
    --original_ckpt ${exp_path}/checkpoints/merged_checkpoint_last.pt  \
    --log-level ${log_level} || exit 1;

# finetune asr
CUDA_VISIBLE_DEVICES=$free_gpu fairseq-hydra-train \
    --config-dir ${finetune_config_path} \
    --config-name ${finetune_config_name} \
    task.data=${data_path} \
    task.label_dir=${data_path} \
    ${finetune_noise_opts} \
    task.tokenizer_bpe_model=${tokenizer_ckpt} \
    model.w2v_path=${exp_path}/final/checkpoints/merged_checkpoint_last_final.pt \
    distributed_training.distributed_world_size=${gpus} \
    distributed_training.nprocs_per_node=${gpus} \
    optimization.update_freq=[${finetune_update_freq}] \
    dataset.num_workers=${workers} \
    hydra.run.dir=${finetune_exp_path} \
    common.user_dir=../avhubert || exit 1;

# infer clean
for dataset in $datasets; do
    python -B ../avhubert/infer_s2s.py \
        --config-dir ${infer_config_path} \
        --config-name ${infer_config_name} \
        dataset.gen_subset=${dataset} \
        common_eval.path=${finetune_exp_path}/checkpoints/checkpoint_best.pt \
        common_eval.results_path=${finetune_exp_path}/infer/clean/${dataset} \
        override.modalities=['audio','video'] \
        hydra.run.dir=${finetune_exp_path}/infer/clean/${dataset} \
        common.user_dir=../avhubert || exit 1;
done

# infer noise
for noise in $infer_noise_types; do
    if [ $noise != speech ]; then
        infer_noise_path=/beegfs/data/shared/lrs3/noise/musan/tsv/${noise}
    else
        infer_noise_path=/beegfs/data/shared/lrs3/noise/${noise}
    fi
    for snr in $infer_noise_snr; do
        for dataset in $datasets; do
            python -B ../avhubert/infer_s2s.py \
                --config-dir ${infer_config_path} \
                --config-name ${infer_config_name} \
                dataset.gen_subset=${dataset} \
                common_eval.path=${finetune_exp_path}/checkpoints/checkpoint_best.pt \
                common_eval.results_path=${finetune_exp_path}/${infer_root}/${noise}/${snr}/${dataset} \
                override.modalities=['audio','video'] \
                override.noise_wav=${infer_noise_path} \
                override.noise_prob=1 \
                override.noise_snr=${snr} \
                override.noise_method=${infer_noise_method} \
                hydra.run.dir=${finetune_exp_path}/${infer_root}/${noise}/${snr}/${dataset} \
                common.user_dir=../avhubert || exit 1;
        done
    done
done
