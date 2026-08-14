#!/usr/bin/env bash
#SBATCH --time=8-00:00:00
#SBATCH --partition=ifn
#SBATCH --gres=gpu:pro6000b_96gb:1
#SBATCH --exclude=gpu[04,05]
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64gb
#SBATCH --job-name=ch3-v3-stage-j-joint-dp


PYTHON_VIRTUAL_ENVIRONMENT=dpavhubert_pro6000
CONDA_ROOT=/home/zhengyangli/anaconda3/
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ ! -f "${script_dir}/distill_only/common.sh" ]]; then
    if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts_v3/distill_only/common.sh" ]]; then
        script_dir="${SLURM_SUBMIT_DIR}/scripts_v3"
    elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/distill_only/common.sh" ]]; then
        script_dir="${SLURM_SUBMIT_DIR}"
    else
        printf 'error: cannot locate scripts_v3/distill_only/common.sh\n' >&2
        exit 2
    fi
fi
# shellcheck source=distill_only/common.sh
source "${script_dir}/distill_only/common.sh"

experiment=""
stage=all
seed=1337
run_name=lr5e-4
finetune_lr=0.0005
evaluation_phase=screening
evaluation_subsets=valid
evaluation_subsets_explicit=false
evaluation_name=""
resume=false
dry_run=false
teacher_override=""
stage1_student_override=""
prune_input_override=""
stage2_input_override=""
export_input_override=""
finetune_input_override=""
evaluation_input_override=""
k0_root_override=""
project_path="${CH3_PROJECT_PATH:-$(cd "${script_dir}/.." && pwd)}"

usage() {
    cat <<'EOF'
Usage: scripts_v3/run_ch3_joint_dp_stage_j.sh --experiment j1|j2|j3 [options]

  --stage joint_dp|prune|post_distill|export|save|finetune|evaluate|all
  --seed INTEGER                       default: 1337
  --finetune-run-name NAME             default: lr5e-4
  --finetune-lr FLOAT                  default: 0.0005
  --evaluation-phase screening|final   default: screening
  --evaluation-subsets valid|test|valid,test  default: valid
  --evaluation-name NAME
  --k0-root PATH                       J2 source; default: exp/chapter3_joint_dp/k0/seed_1337
  --teacher-checkpoint PATH
  --stage1-student-checkpoint PATH
  --prune-input-checkpoint PATH
  --stage2-input-checkpoint PATH
  --export-input-checkpoint PATH
  --finetune-input-checkpoint PATH
  --evaluation-input-checkpoint PATH
  --resume
  --dry-run

J2 always reuses the strictly audited completed K0 encoder. Its joint_dp,
prune, post_distill, and export stages cannot be launched from this script.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment) ch3_need_value "$@"; experiment=$2; shift 2 ;;
        --stage) ch3_need_value "$@"; stage=$2; shift 2 ;;
        --seed) ch3_need_value "$@"; seed=$2; shift 2 ;;
        --finetune-run-name|--run-name) ch3_need_value "$@"; run_name=$2; shift 2 ;;
        --finetune-lr) ch3_need_value "$@"; finetune_lr=$2; shift 2 ;;
        --evaluation-phase) ch3_need_value "$@"; evaluation_phase=$2; shift 2 ;;
        --evaluation-subsets|--eval-subsets) ch3_need_value "$@"; evaluation_subsets=$2; evaluation_subsets_explicit=true; shift 2 ;;
        --evaluation-name|--eval-name) ch3_need_value "$@"; evaluation_name=$2; shift 2 ;;
        --k0-root) ch3_need_value "$@"; k0_root_override=$2; shift 2 ;;
        --teacher-checkpoint) ch3_need_value "$@"; teacher_override=$2; shift 2 ;;
        --stage1-student-checkpoint) ch3_need_value "$@"; stage1_student_override=$2; shift 2 ;;
        --prune-input-checkpoint) ch3_need_value "$@"; prune_input_override=$2; shift 2 ;;
        --stage2-input-checkpoint) ch3_need_value "$@"; stage2_input_override=$2; shift 2 ;;
        --export-input-checkpoint) ch3_need_value "$@"; export_input_override=$2; shift 2 ;;
        --finetune-input-checkpoint) ch3_need_value "$@"; finetune_input_override=$2; shift 2 ;;
        --evaluation-input-checkpoint) ch3_need_value "$@"; evaluation_input_override=$2; shift 2 ;;
        --resume) resume=true; shift ;;
        --dry-run) dry_run=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) ch3_fail "unknown option: $1" ;;
    esac
done

[[ -n "$experiment" ]] || ch3_fail "--experiment is required"
case "$experiment" in j1|j2|j3) ;; *) ch3_fail "unsupported Stage-J experiment: $experiment" ;; esac
case "$stage" in joint_dp|prune|post_distill|export|save|finetune|evaluate|all) ;; *) ch3_fail "unsupported stage: $stage" ;; esac
[[ "$stage" != save ]] || stage=export
[[ "$seed" =~ ^[0-9]+$ ]] || ch3_fail "seed must be a nonnegative integer"
[[ "$seed" == 1337 ]] || ch3_fail "Stage-J scientific runs require seed 1337"
ch3_validate_name "fine-tuning run name" "$run_name"
ch3_validate_positive_float "$finetune_lr" || ch3_fail "invalid fine-tuning LR"
case "$evaluation_phase" in screening|final) ;; *) ch3_fail "invalid evaluation phase" ;; esac
if [[ "$evaluation_phase" == final && "$evaluation_subsets_explicit" == false ]]; then evaluation_subsets=valid,test; fi
case "$evaluation_subsets" in valid|test|valid,test) ;; *) ch3_fail "invalid evaluation subsets" ;; esac
[[ "$evaluation_phase" != screening || "$evaluation_subsets" == valid ]] || ch3_fail "screening is validation-only"

project_path=$(cd "$project_path" && pwd)
stage1_root="${project_path}/avhubert/conf/distill_v3/joint_dp_stage1"
stage2_root="${project_path}/avhubert/conf/distill_v3/joint_dp_stage2"
config_helper="${script_dir}/joint_dp/stage_j_config.py"
metadata_helper="${script_dir}/joint_dp/stage_j_metadata.py"
artifact_helper="${script_dir}/joint_dp/stage_k_artifacts.py"
[[ -f "$config_helper" && -f "$metadata_helper" && -f "$artifact_helper" ]] || ch3_fail "missing Stage-J helper"
k0_root="${k0_root_override:-${project_path}/exp/chapter3_joint_dp/k0/seed_1337}"
[[ "$k0_root" == /* ]] || k0_root="${project_path}/${k0_root}"
config_command=(python "$config_helper" --experiment "$experiment" --stage1-root "$stage1_root" --stage2-root "$stage2_root")
[[ "$experiment" != j2 ]] || config_command+=(--k0-root "$k0_root")
resolved=$("${config_command[@]}")
IFS=$'\t' read -r output_id stage1_name stage2_name matching targets cosine stage1_l1 stage2_l1 pruning_units target_sparsity encoder_source stage1_config stage2_config <<<"$resolved"

output_root="${project_path}/exp/chapter3_joint_dp"
root_path="${output_root}/${output_id}/seed_${seed}"
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
evaluation_name=${evaluation_name:-"${evaluation_phase}-${evaluation_subsets//,/-}"}
ch3_validate_name "evaluation name" "$evaluation_name"
evaluation_path="${finetune_path}/evaluations/${evaluation_name}"
metadata_path="${root_path}/run_metadata.json"
stage1_resolved="${joint_path}/.hydra/config.yaml"
stage2_resolved="${post_path}/.hydra/config.yaml"
teacher="${teacher_override:-${CH3_TEACHER_CHECKPOINT:-/home/zhengyangli/work/av_hubert_pre_trained_models/phd_thesis/base_vox_iter5.pt}}"
stage1_student="${stage1_student_override:-$teacher}"

reuse_k0=false
if [[ "$experiment" == j2 ]]; then
    reuse_k0=true
    case "$stage" in joint_dp|prune|post_distill|export)
        ch3_fail "J2 is the audited completed K0 encoder; use stage finetune, evaluate, or all"
    esac
    [[ -z "$teacher_override$stage1_student_override$prune_input_override$stage2_input_override$export_input_override$finetune_input_override" ]] ||
        ch3_fail "J2 encoder provenance is immutable; checkpoint overrides are not allowed"
    joint_checkpoint="${k0_root}/joint_dp/checkpoints/checkpoint_last.pt"
    pruned_checkpoint="${k0_root}/pruned/student.pt"
    post_checkpoint="${k0_root}/post_distill/checkpoints/checkpoint_last.pt"
    export_checkpoint="${k0_root}/export/student.pt"
    stage1_resolved="${k0_root}/joint_dp/.hydra/config.yaml"
    stage2_resolved="${k0_root}/post_distill/.hydra/config.yaml"
    teacher=$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["model"]["teacher_path"])' "$stage1_resolved")
    stage1_student=$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["model"]["student_path"])' "$stage1_resolved")
fi

prune_input="${prune_input_override:-$joint_checkpoint}"
stage2_input="${stage2_input_override:-$pruned_checkpoint}"
export_input="${export_input_override:-$post_checkpoint}"
finetune_input="${finetune_input_override:-$export_checkpoint}"
evaluation_input="${evaluation_input_override:-$finetune_checkpoint}"
data="${CH3_DATA_PATH:-/beegfs/data/shared/lrs3/433h_data_avhubert}"
tokenizer="${CH3_TOKENIZER_PATH:-/beegfs/data/shared/lrs3/spm1000/spm_unigram1000.model}"
noise="${CH3_NOISE_PATH:-/beegfs/data/shared/lrs3/noise/musan/tsv/all}"
workers="${CH3_WORKERS:-24}"
gpus="${CH3_GPUS:-1}"
user_dir="${project_path}/avhubert"
finetune_config="${project_path}/avhubert/conf/av-finetune/base_noise_pt_noise_ft_433h.yaml"
evaluation_protocol="${project_path}/scripts/distill_only/evaluation_protocol_itut.yaml"
source_branch=$(git -C "$project_path" branch --show-current)
source_commit=$(git -C "$project_path" rev-parse HEAD)
source_dirty=false
[[ -z "$(git -C "$project_path" status --porcelain=v1 --untracked-files=all)" ]] || source_dirty=true
[[ "$source_branch" == ch3-distill-only-v3 ]] || ch3_fail "must run from ch3-distill-only-v3, got $source_branch"
case "$root_path" in "${output_root}"/*) ;; *) ch3_fail "output escaped Stage-J root" ;; esac

selected_stage() {
    if [[ "$stage" == all && "$reuse_k0" == true ]]; then [[ "$1" == finetune || "$1" == evaluate ]]; return; fi
    [[ "$stage" == "$1" || "$stage" == all ]]
}

preflight_outputs() {
    if ch3_directory_nonempty "$root_path" && [[ ! -f "$metadata_path" ]]; then
        ch3_fail "Stage-J root is nonempty without compatible metadata: $root_path"
    fi
    [[ "$resume" == true ]] && return
    selected_stage joint_dp && ch3_reject_nonempty "$joint_path" "joint-DP" false
    selected_stage prune && ch3_reject_nonempty "$pruned_path" "physical pruning" false
    selected_stage post_distill && ch3_reject_nonempty "$post_path" "post-distillation" false
    selected_stage export && ch3_reject_nonempty "$export_path" "export" false
    selected_stage finetune && ch3_reject_nonempty "$finetune_path" "fine-tuning" false
    selected_stage evaluate && ch3_reject_nonempty "$evaluation_path" "evaluation" false
}

execute() {
    ch3_print_command "$@"
    [[ "$dry_run" == true ]] && return
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${project_path}/fairseq:${project_path}/avhubert:${project_path}${PYTHONPATH:+:${PYTHONPATH}}" "$@"
}

record_metadata() {
    local -a command=(python "$metadata_helper" record --metadata "$metadata_path"
        --experiment "$experiment" --encoder-source "$encoder_source" --seed "$seed" --stage "$stage"
        --branch "$source_branch" --commit "$source_commit" --stage1-config "$stage1_config" --stage2-config "$stage2_config"
        --matching "$matching" --targets "$targets" --cosine "$cosine" --stage1-l1 "$stage1_l1" --stage2-l1 "$stage2_l1"
        --pruning-units "$pruning_units" --target-sparsity "$target_sparsity"
        --teacher "$teacher" --student-input "$stage1_student" --joint-checkpoint "$joint_checkpoint"
        --pruned-checkpoint "$pruned_checkpoint" --stage2-checkpoint "$post_checkpoint" --export-checkpoint "$export_checkpoint"
        --finetune-checkpoint "$finetune_checkpoint" --stage1-resolved "$stage1_resolved" --stage2-resolved "$stage2_resolved"
        --finetune-config "$finetune_config" --run-name "$run_name" --finetune-lr "$finetune_lr"
        --evaluation-protocol "$evaluation_protocol" --evaluation-phase "$evaluation_phase" --evaluation-subsets "$evaluation_subsets")
    [[ "$source_dirty" != true ]] || command+=(--dirty)
    [[ "$reuse_k0" != true ]] || command+=(--reuse-k0-root "$k0_root")
    execute "${command[@]}"
}

run_joint_dp() {
    [[ "$dry_run" == true ]] || ch3_require_checkpoint "$stage1_student" "Stage-1 student checkpoint"
    local -a command=(fairseq-hydra-train --config-dir "$stage1_root" --config-name "${stage1_name}.yaml"
        "task.data=${data}" "task.label_dir=${data}" "task.noise_wav=${noise}" "task.tokenizer_bpe_model=${tokenizer}"
        "model.teacher_path=${teacher}" "model.student_path=${stage1_student}" "common.seed=${seed}" "common.user_dir=${user_dir}"
        "dataset.num_workers=${workers}" "distributed_training.distributed_world_size=${gpus}"
        "distributed_training.nprocs_per_node=${gpus}" "hydra.run.dir=${joint_path}")
    if [[ "$resume" == true ]] && ch3_directory_nonempty "$joint_path"; then command+=("checkpoint.restore_file=${joint_checkpoint}"); fi
    [[ "$dry_run" == true ]] || mkdir -p "$joint_path"
    execute "${command[@]}"
}

run_prune() {
    execute python "$metadata_helper" expected-sparsity --metadata "$metadata_path" --stage1-log "${joint_path}/hydra_train.log"
    if [[ "$resume" == true && -f "$pruned_checkpoint" ]]; then
        execute python "$artifact_helper" verify --project-path "$project_path" --checkpoint "$pruned_checkpoint"
    else
        [[ "$dry_run" == true ]] || ch3_require_checkpoint "$prune_input" "joint-DP checkpoint"
        execute python "$artifact_helper" prune --project-path "$project_path" --distilled-checkpoint "$prune_input" --original-checkpoint "$teacher" --output "$pruned_checkpoint"
    fi
    execute python "$metadata_helper" realized-sparsity --metadata "$metadata_path" --project "$project_path" --teacher "$teacher" --pruned "$pruned_checkpoint"
}

run_post_distill() {
    [[ "$dry_run" == true ]] || ch3_require_checkpoint "$stage2_input" "pruned student checkpoint"
    local -a command=(fairseq-hydra-train --config-dir "$stage2_root" --config-name "${stage2_name}.yaml"
        "task.data=${data}" "task.label_dir=${data}" "task.noise_wav=${noise}" "task.tokenizer_bpe_model=${tokenizer}"
        "model.teacher_path=${teacher}" "model.student_path=${stage2_input}" "common.seed=${seed}" "common.user_dir=${user_dir}"
        "dataset.num_workers=${workers}" "distributed_training.distributed_world_size=${gpus}"
        "distributed_training.nprocs_per_node=${gpus}" "hydra.run.dir=${post_path}")
    if [[ "$resume" == true ]] && ch3_directory_nonempty "$post_path"; then command+=("checkpoint.restore_file=${post_checkpoint}"); fi
    [[ "$dry_run" == true ]] || mkdir -p "$post_path"
    execute "${command[@]}"
}

run_export() {
    if [[ "$resume" == true && -f "$export_checkpoint" ]]; then
        execute python "$artifact_helper" verify --project-path "$project_path" --checkpoint "$export_checkpoint"
    else
        [[ "$dry_run" == true ]] || ch3_require_checkpoint "$export_input" "post-distillation checkpoint"
        execute python "$artifact_helper" export --project-path "$project_path" --distilled-checkpoint "$export_input" --original-checkpoint "$stage2_input" --output "$export_checkpoint"
    fi
}

run_finetune() {
    local -a command=("${script_dir}/distill_only/run_finetune.sh" --project-path "$project_path" --input-checkpoint "$finetune_input"
        --output-dir "$finetune_path" --user-dir "$user_dir" --seed "$seed" --finetune-lr "$finetune_lr")
    [[ "$resume" != true ]] || command+=(--resume)
    [[ "$dry_run" != true ]] || command+=(--dry-run)
    ch3_print_command "${command[@]}"
    "${command[@]}"
}

run_evaluate() {
    if [[ "$evaluation_phase" == screening ]]; then
        printf 'Selection guard: validation-only clean, babble 0 dB, and speech 0 dB.\n'
    else
        printf 'Selection guard: test WER is reporting-only and cannot select a checkpoint.\n'
    fi
    ch3_run_evaluation "$project_path" "$evaluation_input" "$evaluation_path" "$evaluation_phase" "$evaluation_subsets" "$user_dir" "$resume" "$dry_run"
}

printf 'Scientific context: %s\nBranch: %s\nGit commit: %s\nGit dirty: %s\n' "$CH3_CONTEXT_URL" "$source_branch" "$source_commit" "$source_dirty"
printf 'Experiment: %s\nStage: %s\nEncoder source: %s\nSeed: %s\n' "$experiment" "$stage" "$encoder_source" "$seed"
[[ "$reuse_k0" != true ]] || printf 'J2 K0 audit: PASSED\nK0 read-only root: %s\nEncoder stages: skipped\n' "$k0_root"
printf 'Stage-1 config: %s\nStage-2 config: %s\nMatching: %s\nTeacher targets: {%s}\nCosine: %s\n' "$stage1_config" "$stage2_config" "$matching" "$targets" "$cosine"
printf 'L1: Stage 1=%s Stage 2=%s\nPruning units: %s\nConfigured target sparsity: %s\n' "$stage1_l1" "$stage2_l1" "$pruning_units" "$target_sparsity"
printf 'Stage-1: 50000 updates, clip_norm=10.0, update_freq=4, sparsity_warmup=10000, noise_prob=0.25, noise_snr=0dB\n'
printf 'Stage-2: 25000 updates, clip_norm=10.0, update_freq=4, noise_prob=0.25, noise_snr=0dB\n'
printf 'Teacher: %s\nStage-1 input: %s\nStage-1 checkpoint: %s\nPruned checkpoint: %s\nStage-2 checkpoint: %s\nExported checkpoint: %s\n' "$teacher" "$stage1_student" "$joint_checkpoint" "$pruned_checkpoint" "$post_checkpoint" "$export_checkpoint"
printf 'Resolved Hydra configs: %s ; %s\nOutput root: %s\nMetadata: %s\n' "$stage1_resolved" "$stage2_resolved" "$root_path" "$metadata_path"
printf 'Fine-tuning: input=%s run=%s updates=60000 freeze=48000 lr=%s clip_norm=0.0 update_freq=8 noise_prob=0.25 noise_snr=0dB\n' "$finetune_input" "$run_name" "$finetune_lr"
printf 'Evaluation: phase=%s subsets=%s protocol=%s output=%s\n' "$evaluation_phase" "$evaluation_subsets" "$evaluation_protocol" "$evaluation_path"
[[ "$finetune_lr" == 0.0005 ]] || printf 'Warning: non-comparable fine-tuning LR override is recorded for debugging.\n' >&2

preflight_outputs
if [[ "$dry_run" != true ]]; then ch3_activate_environment; cd "$project_path"; fi
record_metadata
case "$stage" in
    joint_dp) run_joint_dp ;;
    prune) run_prune ;;
    post_distill) run_post_distill ;;
    export) run_export ;;
    finetune) run_finetune ;;
    evaluate) run_evaluate ;;
    all)
        if [[ "$reuse_k0" != true ]]; then run_joint_dp; run_prune; run_post_distill; run_export; fi
        run_finetune
        run_evaluate
        ;;
esac
