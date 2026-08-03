#!/usr/bin/env bash
#SBATCH --time=8-00:00:00
#SBATCH --partition=ifn
#SBATCH --qos=low
#SBATCH --gres=gpu:pro6000b_96gb:1
#SBATCH --exclude=gpu[04,05]
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64gb
#SBATCH --job-name=ch3-v3-distill

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=distill_only/common.sh
source "${script_dir}/distill_only/common.sh"

exp_name=""
stage=all
seed=1337
run_name=lr5e-4
finetune_lr=0.0005
input_checkpoint=""
eval_name=""
eval_subsets=valid
evaluation_phase=screening
resume=false
dry_run=false
project_path="${CH3_PROJECT_PATH:-$(cd "${script_dir}/.." && pwd)}"

usage() {
    cat <<'EOF'
Usage: scripts_v3/run_ch3_distill_only.sh --exp-name NAME [options]

  --stage encoder|export|finetune|evaluate|all
  --seed INTEGER
  --run-name NAME
  --finetune-lr FLOAT
  --input-checkpoint PATH
  --eval-name NAME
  --eval-subsets valid|test|valid,test
  --evaluation-phase screening|final
  --resume
  --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp-name) ch3_need_value "$@"; exp_name=$2; shift 2 ;;
        --stage) ch3_need_value "$@"; stage=$2; shift 2 ;;
        --seed) ch3_need_value "$@"; seed=$2; shift 2 ;;
        --run-name) ch3_need_value "$@"; run_name=$2; shift 2 ;;
        --finetune-lr) ch3_need_value "$@"; finetune_lr=$2; shift 2 ;;
        --input-checkpoint) ch3_need_value "$@"; input_checkpoint=$2; shift 2 ;;
        --eval-name) ch3_need_value "$@"; eval_name=$2; shift 2 ;;
        --eval-subsets) ch3_need_value "$@"; eval_subsets=$2; shift 2 ;;
        --evaluation-phase) ch3_need_value "$@"; evaluation_phase=$2; shift 2 ;;
        --resume) resume=true; shift ;;
        --dry-run) dry_run=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) ch3_fail "unknown option: $1" ;;
    esac
done

[[ -n "$exp_name" ]] || ch3_fail "--exp-name is required"
ch3_validate_name "experiment name" "$exp_name"
ch3_validate_name "run name" "$run_name"
[[ "$exp_name" != base_* && "$exp_name" != l_base_* && "$exp_name" != _* ]] ||
    ch3_fail "shared protocol/base configs are not runnable experiments"
[[ "$seed" =~ ^[0-9]+$ ]] || ch3_fail "seed must be a nonnegative integer"
ch3_validate_positive_float "$finetune_lr" ||
    ch3_fail "--finetune-lr must be a positive finite number"
case "$stage" in encoder|export|finetune|evaluate|all) ;; *) ch3_fail "unsupported stage: $stage" ;; esac
case "$eval_subsets" in valid|test|valid,test) ;; *) ch3_fail "invalid evaluation subsets" ;; esac
case "$evaluation_phase" in screening|final) ;; *) ch3_fail "invalid evaluation phase" ;; esac
[[ "$evaluation_phase" != screening || "$eval_subsets" == valid ]] ||
    ch3_fail "screening evaluation supports only valid"
if [[ -n "$input_checkpoint" ]]; then
    [[ "$stage" != all && "$stage" != encoder ]] ||
        ch3_fail "--input-checkpoint is valid only for an individual downstream stage"
fi

project_path=$(cd "$project_path" && pwd)
config_dir="${project_path}/avhubert/conf/distill_v3/distill_only"
config_path="${config_dir}/${exp_name}.yaml"
[[ -f "$config_path" ]] || ch3_fail "missing v3 experiment config: $config_path"
output_root="${project_path}/exp/chapter3_distill_only"
root_path="${output_root}/${exp_name}/seed_${seed}"
encoder_path="${root_path}/encoder"
export_path="${root_path}/export"
export_checkpoint="${export_path}/student.pt"
finetune_path="${root_path}/finetune_runs/${run_name}"
finetune_checkpoint="${finetune_path}/checkpoints/checkpoint_best.pt"
eval_name=${eval_name:-"${evaluation_phase}-${eval_subsets//,/-}"}
ch3_validate_name "evaluation name" "$eval_name"
evaluation_path="${finetune_path}/evaluations/${eval_name}"
case "$root_path" in "${output_root}"/*) ;; *) ch3_fail "output escaped Chapter 3 root" ;; esac

teacher="${CH3_TEACHER_CHECKPOINT:-/home/zhengyangli/work/av_hubert_pre_trained_models/phd_thesis/base_vox_iter5.pt}"
data="${CH3_DATA_PATH:-/beegfs/data/shared/lrs3/433h_data_avhubert}"
tokenizer="${CH3_TOKENIZER_PATH:-/beegfs/data/shared/lrs3/spm1000/spm_unigram1000.model}"
workers="${CH3_WORKERS:-24}"
gpus="${CH3_GPUS:-1}"
user_dir="${project_path}/chapter3_distill_only"

execute() {
    ch3_print_command "$@"
    [[ "$dry_run" == true ]] && return
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${project_path}/fairseq:${project_path}/avhubert:${project_path}${PYTHONPATH:+:${PYTHONPATH}}" \
        "$@"
}

run_encoder() {
    local last="${encoder_path}/checkpoints/checkpoint_last.pt"
    ch3_reject_nonempty "$encoder_path" "encoder" "$resume"
    if [[ "$resume" == true ]]; then ch3_require_loadable_checkpoint "$last" "encoder resume checkpoint"; fi
    local -a command=(
        fairseq-hydra-train --config-dir "$config_dir" --config-name "${exp_name}.yaml"
        "task.data=${data}" "task.label_dir=${data}"
        "task.tokenizer_bpe_model=${tokenizer}" "model.teacher_path=${teacher}"
        "distributed_training.distributed_world_size=${gpus}"
        "distributed_training.nprocs_per_node=${gpus}"
        "dataset.num_workers=${workers}" "common.seed=${seed}"
        "common.user_dir=${user_dir}" "hydra.run.dir=${encoder_path}"
    )
    [[ "$resume" == true ]] && command+=("checkpoint.restore_file=${last}")
    [[ "$dry_run" == true ]] || mkdir -p "$encoder_path"
    execute "${command[@]}"
}

run_export() {
    local parent=${input_checkpoint:-"${encoder_path}/checkpoints/checkpoint_last.pt"}
    if [[ "$resume" == true ]]; then
        [[ "$dry_run" == true ]] || ch3_require_checkpoint "$export_checkpoint" "export checkpoint"
        execute python -m chapter3_distill_only.exporter --verify-existing "$export_checkpoint" --no-digest-metadata
        return
    fi
    ch3_reject_nonempty "$export_path" "export" false
    [[ "$dry_run" == true ]] || ch3_require_checkpoint "$parent" "encoder checkpoint"
    [[ "$dry_run" == true ]] || mkdir -p "$export_path"
    execute python -m chapter3_distill_only.exporter --distilled-checkpoint "$parent" --output "$export_checkpoint" --no-digest-metadata
}

run_finetune() {
    local parent=${input_checkpoint:-$export_checkpoint}
    local -a command=(
        "${script_dir}/distill_only/run_finetune.sh"
        --project-path "$project_path" --input-checkpoint "$parent"
        --output-dir "$finetune_path" --user-dir "$user_dir"
        --seed "$seed" --finetune-lr "$finetune_lr"
    )
    [[ "$resume" == true ]] && command+=(--resume)
    [[ "$dry_run" == true ]] && command+=(--dry-run)
    ch3_print_command "${command[@]}"
    "${command[@]}"
}

run_evaluate() {
    local parent=${input_checkpoint:-$finetune_checkpoint}
    ch3_run_evaluation "$project_path" "$parent" "$evaluation_path" \
        "$evaluation_phase" "$eval_subsets" "$user_dir" "$resume" "$dry_run"
}

printf 'Scientific context: %s\n' "$CH3_CONTEXT_URL"
printf 'Experiment: %s\nStage: %s\nSeed: %s\nConfig: %s\nOutput: %s\n' \
    "$exp_name" "$stage" "$seed" "$config_path" "$root_path"
printf 'Fine-tuning LR: %s\n' "$finetune_lr"

if [[ "$dry_run" != true ]]; then
    ch3_activate_environment
    cd "$project_path"
fi
case "$stage" in
    encoder) run_encoder ;;
    export) run_export ;;
    finetune) run_finetune ;;
    evaluate) run_evaluate ;;
    all) run_encoder; run_export; run_finetune; run_evaluate ;;
esac
