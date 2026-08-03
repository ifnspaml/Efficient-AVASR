#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "${script_dir}/common.sh"

project_path="${CH3_PROJECT_PATH:-$(cd "${script_dir}/../.." && pwd)}"
input_checkpoint=""
output_dir=""
seed=1337
finetune_lr=0.0005
user_dir=""
resume=false
dry_run=false

usage() {
    cat <<'EOF'
Usage: scripts_v3/distill_only/run_finetune.sh [options]

Required:
  --input-checkpoint PATH
  --output-dir PATH

Options:
  --project-path PATH
  --user-dir PATH
  --seed INTEGER
  --finetune-lr FLOAT       default: 0.0005
  --resume
  --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project-path) ch3_need_value "$@"; project_path=$2; shift 2 ;;
        --input-checkpoint) ch3_need_value "$@"; input_checkpoint=$2; shift 2 ;;
        --output-dir) ch3_need_value "$@"; output_dir=$2; shift 2 ;;
        --user-dir) ch3_need_value "$@"; user_dir=$2; shift 2 ;;
        --seed) ch3_need_value "$@"; seed=$2; shift 2 ;;
        --finetune-lr) ch3_need_value "$@"; finetune_lr=$2; shift 2 ;;
        --resume) resume=true; shift ;;
        --dry-run) dry_run=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) ch3_fail "unknown option: $1" ;;
    esac
done

[[ -n "$input_checkpoint" ]] || ch3_fail "--input-checkpoint is required"
[[ -n "$output_dir" ]] || ch3_fail "--output-dir is required"
[[ "$seed" =~ ^[0-9]+$ ]] || ch3_fail "seed must be a nonnegative integer"
ch3_validate_positive_float "$finetune_lr" ||
    ch3_fail "--finetune-lr must be a positive finite number"

project_path=$(cd "$project_path" && pwd)
user_dir=${user_dir:-"${project_path}/chapter3_distill_only"}
data_path="${CH3_DATA_PATH:-/beegfs/data/shared/lrs3/433h_data_avhubert}"
tokenizer="${CH3_TOKENIZER_PATH:-/beegfs/data/shared/lrs3/spm1000/spm_unigram1000.model}"
noise_path="${CH3_NOISE_PATH:-/beegfs/data/shared/lrs3/noise/musan/tsv/all}"
workers="${CH3_WORKERS:-24}"
gpus="${CH3_GPUS:-1}"
last_checkpoint="${output_dir}/checkpoints/checkpoint_last.pt"

if [[ "$dry_run" != true ]]; then
    ch3_require_checkpoint "$input_checkpoint" "fine-tuning input checkpoint"
fi
ch3_reject_nonempty "$output_dir" "fine-tuning" "$resume"
if [[ "$resume" == true && "$dry_run" != true ]]; then
    ch3_require_loadable_checkpoint "$last_checkpoint" "fine-tuning resume checkpoint"
fi

command=(
    fairseq-hydra-train
    --config-dir "${project_path}/avhubert/conf/av-finetune"
    --config-name base_noise_pt_noise_ft_433h.yaml
    "task.data=${data_path}"
    "task.label_dir=${data_path}"
    "task.tokenizer_bpe_model=${tokenizer}"
    "task.noise_wav=${noise_path}"
    "task.noise_prob=0.25"
    "task.noise_snr=0"
    "model.w2v_path=${input_checkpoint}"
    "model.freeze_finetune_updates=48000"
    "optimization.max_update=60000"
    "optimization.lr=[${finetune_lr}]"
    "optimization.clip_norm=0.0"
    "optimization.update_freq=[8]"
    "distributed_training.distributed_world_size=${gpus}"
    "distributed_training.nprocs_per_node=${gpus}"
    "dataset.num_workers=${workers}"
    "common.seed=${seed}"
    "common.user_dir=${user_dir}"
    "hydra.run.dir=${output_dir}"
)
if [[ "$resume" == true ]]; then
    command+=("checkpoint.restore_file=${last_checkpoint}")
fi

printf 'Scientific context: %s\n' "$CH3_CONTEXT_URL"
printf 'Fine-tuning contract: updates=60000 freeze=48000 lr=%s clip=0.0 update_freq=8 noise_prob=0.25 noise_snr=0\n' "$finetune_lr"
ch3_print_command "${command[@]}"
[[ "$dry_run" == true ]] && exit 0

ch3_activate_environment
cd "$project_path"
mkdir -p "$output_dir"
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="${project_path}/fairseq:${project_path}/avhubert:${project_path}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${command[@]}"
