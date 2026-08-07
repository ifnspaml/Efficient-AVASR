#!/usr/bin/env bash
#SBATCH --time=8-00:00:00
#SBATCH --partition=ifn
#SBATCH --qos=low
#SBATCH --gres=gpu:pro6000b_96gb:1
#SBATCH --exclude=gpu[04,05]
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64gb
#SBATCH --job-name=ch3-v3-joint-dp


PYTHON_VIRTUAL_ENVIRONMENT=dpavhubert_pro6000
CONDA_ROOT=/home/zhengyangli/anaconda3/
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

set -euo pipefail

# Under Slurm, the batch file is copied to …/slurm_script, so BASH_SOURCE no
# longer sits next to distill_only/. Recover via SLURM_SUBMIT_DIR when needed.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ ! -f "${script_dir}/distill_only/common.sh" ]]; then
    if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts_v3/distill_only/common.sh" ]]; then
        script_dir="${SLURM_SUBMIT_DIR}/scripts_v3"
    elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/distill_only/common.sh" ]]; then
        script_dir="${SLURM_SUBMIT_DIR}"
    else
        printf 'error: cannot locate scripts_v3/distill_only/common.sh (got script_dir=%s)\n' \
            "$script_dir" >&2
        exit 2
    fi
fi
# shellcheck source=distill_only/common.sh
source "${script_dir}/distill_only/common.sh"

exp_name=""
stage=all
seed=1337
run_name=lr5e-4
finetune_lr=0.0005
cosine_type=""
l1_weight=""
input_checkpoint=""
eval_name=""
eval_subsets=valid
evaluation_phase=screening
resume=false
dry_run=false
project_path="${CH3_PROJECT_PATH:-$(cd "${script_dir}/.." && pwd)}"

usage() {
    cat <<'EOF'
Usage: scripts_v3/run_ch3_joint_dp.sh --exp-name NAME [options]

  --stage joint_dp|prune|post_distill|export|finetune|evaluate|all
  --seed INTEGER
  --run-name NAME
  --finetune-lr FLOAT
  --cosine-type raw|log_sig
  --l1-weight FLOAT
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
        --cosine-type) ch3_need_value "$@"; cosine_type=$2; shift 2 ;;
        --l1-weight) ch3_need_value "$@"; l1_weight=$2; shift 2 ;;
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
[[ "$seed" =~ ^[0-9]+$ ]] || ch3_fail "seed must be a nonnegative integer"
ch3_validate_positive_float "$finetune_lr" || ch3_fail "invalid fine-tuning LR"
case "$stage" in joint_dp|prune|post_distill|export|finetune|evaluate|all) ;; *) ch3_fail "unsupported stage: $stage" ;; esac
case "$eval_subsets" in valid|test|valid,test) ;; *) ch3_fail "invalid evaluation subsets" ;; esac
case "$evaluation_phase" in screening|final) ;; *) ch3_fail "invalid evaluation phase" ;; esac
[[ "$evaluation_phase" != screening || "$eval_subsets" == valid ]] || ch3_fail "screening supports only valid"
if [[ -n "$input_checkpoint" && "$stage" == all ]]; then ch3_fail "--input-checkpoint is ambiguous with all"; fi

project_path=$(cd "$project_path" && pwd)
stage1_dir="${project_path}/avhubert/conf/distill_v3/joint_dp_stage1"
stage2_dir="${project_path}/avhubert/conf/distill_v3/joint_dp_stage2"
stage1_config="${stage1_dir}/${exp_name}.yaml"
stage2_config="${stage2_dir}/${exp_name}.yaml"
validator=(python "${script_dir}/distill_only/joint_dp_config.py" --experiment "$exp_name" --stage1 "$stage1_config" --stage2 "$stage2_config")
[[ -n "$cosine_type" ]] && validator+=(--cosine-type "$cosine_type")
[[ -n "$l1_weight" ]] && validator+=(--l1-weight "$l1_weight")
loss=$("${validator[@]}") || exit $?
IFS=$'\t' read -r resolved_cosine resolved_l1 <<<"$loss"

output_root="${project_path}/exp/chapter3_distill_only"
root_path="${output_root}/${exp_name}/seed_${seed}"
joint_path="${root_path}/joint_dp"
joint_checkpoint="${joint_path}/checkpoints/checkpoint_last.pt"
pruned_path="${root_path}/pruned"
pruned_checkpoint="${pruned_path}/student.pt"
post_path="${root_path}/post_distill"
post_checkpoint="${post_path}/checkpoints/checkpoint_last.pt"
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
noise="${CH3_NOISE_PATH:-/beegfs/data/shared/lrs3/noise/musan/tsv/all}"
workers="${CH3_WORKERS:-24}"
gpus="${CH3_GPUS:-1}"
user_dir="${project_path}/avhubert"

execute() {
    ch3_print_command "$@"
    [[ "$dry_run" == true ]] && return
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${project_path}/fairseq:${project_path}/avhubert:${project_path}${PYTHONPATH:+:${PYTHONPATH}}" \
        "$@"
}

run_joint_dp() {
    local student=${input_checkpoint:-$teacher}
    ch3_reject_nonempty "$joint_path" "joint-DP" "$resume"
    if [[ "$resume" == true ]]; then ch3_require_loadable_checkpoint "$joint_checkpoint" "joint-DP resume checkpoint"; fi
    [[ "$dry_run" == true ]] || ch3_require_checkpoint "$student" "student initialization checkpoint"
    local -a command=(
        fairseq-hydra-train --config-dir "$stage1_dir" --config-name "${exp_name}.yaml"
        "task.data=${data}" "task.label_dir=${data}" "task.noise_wav=${noise}"
        "task.tokenizer_bpe_model=${tokenizer}" "model.teacher_path=${teacher}"
        "model.student_path=${student}" "criterion.cos_type=${resolved_cosine}"
        "criterion.l1_weight=${resolved_l1}" "common.seed=${seed}"
        "common.user_dir=${user_dir}" "dataset.num_workers=${workers}"
        "distributed_training.distributed_world_size=${gpus}"
        "distributed_training.nprocs_per_node=${gpus}"
        "hydra.run.dir=${joint_path}"
    )
    [[ "$resume" == true ]] && command+=("checkpoint.restore_file=${joint_checkpoint}")
    [[ "$dry_run" == true ]] || mkdir -p "$joint_path"
    execute "${command[@]}"
}

run_prune() {
    local parent=${input_checkpoint:-$joint_checkpoint}
    if [[ "$resume" == true ]]; then
        execute python "${script_dir}/distill_only/joint_dp_artifacts.py" verify --project-path "$project_path" --checkpoint "$pruned_checkpoint"
        return
    fi
    ch3_reject_nonempty "$pruned_path" "physical pruning" false
    [[ "$dry_run" == true ]] || ch3_require_checkpoint "$parent" "joint-DP checkpoint"
    execute python "${script_dir}/distill_only/joint_dp_artifacts.py" prune --project-path "$project_path" --distilled-checkpoint "$parent" --original-checkpoint "$teacher" --output "$pruned_checkpoint"
}

run_post_distill() {
    local parent=${input_checkpoint:-$pruned_checkpoint}
    ch3_reject_nonempty "$post_path" "post-distillation" "$resume"
    if [[ "$resume" == true ]]; then ch3_require_loadable_checkpoint "$post_checkpoint" "post-distillation resume checkpoint"; fi
    [[ "$dry_run" == true ]] || ch3_require_checkpoint "$parent" "pruned student checkpoint"
    local -a command=(
        fairseq-hydra-train --config-dir "$stage2_dir" --config-name "${exp_name}.yaml"
        "task.data=${data}" "task.label_dir=${data}" "task.noise_wav=${noise}"
        "task.tokenizer_bpe_model=${tokenizer}" "model.teacher_path=${teacher}"
        "model.student_path=${parent}" "criterion.cos_type=${resolved_cosine}"
        "criterion.l1_weight=${resolved_l1}" "common.seed=${seed}"
        "common.user_dir=${user_dir}" "dataset.num_workers=${workers}"
        "distributed_training.distributed_world_size=${gpus}"
        "distributed_training.nprocs_per_node=${gpus}"
        "hydra.run.dir=${post_path}"
    )
    [[ "$resume" == true ]] && command+=("checkpoint.restore_file=${post_checkpoint}")
    [[ "$dry_run" == true ]] || mkdir -p "$post_path"
    execute "${command[@]}"
}

run_export() {
    local parent=${input_checkpoint:-$post_checkpoint}
    if [[ "$resume" == true ]]; then
        execute python "${script_dir}/distill_only/joint_dp_artifacts.py" verify --project-path "$project_path" --checkpoint "$export_checkpoint"
        return
    fi
    ch3_reject_nonempty "$export_path" "export" false
    [[ "$dry_run" == true ]] || ch3_require_checkpoint "$parent" "post-distillation checkpoint"
    execute python "${script_dir}/distill_only/joint_dp_artifacts.py" export --project-path "$project_path" --distilled-checkpoint "$parent" --original-checkpoint "$pruned_checkpoint" --output "$export_checkpoint"
}

run_finetune() {
    local parent=${input_checkpoint:-$export_checkpoint}
    local -a command=("${script_dir}/distill_only/run_finetune.sh" --project-path "$project_path" --input-checkpoint "$parent" --output-dir "$finetune_path" --user-dir "$user_dir" --seed "$seed" --finetune-lr "$finetune_lr")
    [[ "$resume" == true ]] && command+=(--resume)
    [[ "$dry_run" == true ]] && command+=(--dry-run)
    ch3_print_command "${command[@]}"
    "${command[@]}"
}

run_evaluate() {
    local parent=${input_checkpoint:-$finetune_checkpoint}
    ch3_run_evaluation "$project_path" "$parent" "$evaluation_path" "$evaluation_phase" "$eval_subsets" "$user_dir" "$resume" "$dry_run"
}

printf 'Scientific context: %s\n' "$CH3_CONTEXT_URL"
printf 'Experiment: %s\nStage: %s\nSeed: %s\nLoss: cosine=%s l1=%s\nOutput: %s\n' "$exp_name" "$stage" "$seed" "$resolved_cosine" "$resolved_l1" "$root_path"
printf 'Fine-tuning LR: %s\n' "$finetune_lr"
if [[ "$dry_run" != true ]]; then ch3_activate_environment; cd "$project_path"; fi
case "$stage" in
    joint_dp) run_joint_dp ;;
    prune) run_prune ;;
    post_distill) run_post_distill ;;
    export) run_export ;;
    finetune) run_finetune ;;
    evaluate) run_evaluate ;;
    all) run_joint_dp; run_prune; run_post_distill; run_export; run_finetune; run_evaluate ;;
esac
