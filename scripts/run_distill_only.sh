#!/usr/bin/env bash
#SBATCH --time=8-00:00:00
#SBATCH --partition=ifn
#SBATCH --gres=gpu:pro6000b_96gb:1
#SBATCH --exclude=gpu[04,05]
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48gb
#SBATCH --job-name=ch3-distill-only

set -euo pipefail

exp_name=""
stage="all"
seed=1337
run_name="default"
input_checkpoint=""
eval_name=""
eval_subsets="valid"
evaluation_phase="screening"
resume=false
dry_run=false

project_path="${CH3_PROJECT_PATH:-/beegfs/work_fast/zhengyangli/dpav_hubert_new}"
teacher_ckpt="/home/zhengyangli/work/av_hubert_pre_trained_models/phd_thesis/base_vox_iter5.pt"
data_path="/beegfs/data/shared/lrs3/433h_data_avhubert"
tokenizer_ckpt="/beegfs/data/shared/lrs3/spm1000/spm_unigram1000.model"
finetune_noise_path="/beegfs/data/shared/lrs3/noise/musan/tsv/all"
workers=4
gpus=1
max_tokens=4000
encoder_update_freq=4
finetune_update_freq=8
finetune_lr=0.001

fail() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

usage() {
    cat <<'EOF'
Usage:
  scripts/run_distill_only.sh --exp-name NAME [options]

Options:
  --stage encoder|stage1|stage2|export|finetune|evaluate|all
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

need_value() {
    [[ $# -ge 2 && -n "${2:-}" ]] || fail "$1 requires a value"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --exp-name) need_value "$@"; exp_name=$2; shift 2 ;;
            --stage) need_value "$@"; stage=$2; shift 2 ;;
            --seed) need_value "$@"; seed=$2; shift 2 ;;
            --run-name) need_value "$@"; run_name=$2; shift 2 ;;
            --finetune-lr) need_value "$@"; finetune_lr=$2; shift 2 ;;
            --input-checkpoint) need_value "$@"; input_checkpoint=$2; shift 2 ;;
            --eval-name) need_value "$@"; eval_name=$2; shift 2 ;;
            --eval-subsets) need_value "$@"; eval_subsets=$2; shift 2 ;;
            --evaluation-phase) need_value "$@"; evaluation_phase=$2; shift 2 ;;
            --resume) resume=true; shift ;;
            --dry-run) dry_run=true; shift ;;
            --help|-h) usage; exit 0 ;;
            *) fail "unknown option: $1" ;;
        esac
    done
}

validate_name() {
    local label=$1 value=$2
    [[ -n "$value" ]] || fail "$label must not be empty"
    [[ "$value" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] ||
        fail "$label contains unsafe characters: $value"
    [[ "$value" != *".."* ]] || fail "$label must not contain '..': $value"
}

validate_args() {
    [[ -n "$exp_name" ]] || fail "--exp-name is required"
    validate_name "experiment name" "$exp_name"
    validate_name "run name" "$run_name"
    [[ "$seed" =~ ^[0-9]+$ && "$seed" -gt 0 ]] ||
        fail "seed must be a positive integer"
    if ! python - "$finetune_lr" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit(1)
raise SystemExit(0 if math.isfinite(value) and value > 0 else 1)
PY
    then
        fail "--finetune-lr must be a positive finite number"
    fi
    case "$stage" in
        encoder|stage1|stage2|export|finetune|evaluate|all) ;;
        *) fail "unsupported stage: $stage" ;;
    esac
    case "$eval_subsets" in
        valid|test|valid,test) ;;
        *) fail "--eval-subsets must be valid, test, or valid,test" ;;
    esac
    case "$evaluation_phase" in
        screening|final) ;;
        *) fail "--evaluation-phase must be screening or final" ;;
    esac
    if [[ "$evaluation_phase" == "screening" && "$eval_subsets" != "valid" ]]; then
        fail "screening evaluation supports only --eval-subsets valid"
    fi
    if [[ -n "$input_checkpoint" ]]; then
        case "$stage" in
            finetune|evaluate) ;;
            *) fail "--input-checkpoint is valid only for finetune or evaluate" ;;
        esac
    fi
    if [[ "$stage" == "stage1" || "$stage" == "stage2" ]]; then
        [[ "$exp_name" == "s2_optional_two_stage" ]] ||
            fail "$stage is supported only for s2_optional_two_stage"
    fi
    if [[ "$stage" == "encoder" && "$exp_name" == "s2_optional_two_stage" ]]; then
        fail "S2 requires explicit stage1 and stage2; encoder is not valid"
    fi
}

resolve_paths() {
    project_path=$(cd "$project_path" && pwd)
    avhubert_dir="${project_path}/avhubert"
    output_root="${project_path}/exp/chapter3_distill_only"
    config_path="${avhubert_dir}/conf/distill_only/${exp_name}.yaml"
    evaluation_protocol="${project_path}/scripts/distill_only/evaluation_protocol_itut.yaml"
    condition_helper="${project_path}/scripts/distill_only/evaluation_conditions.py"
    [[ -f "$config_path" ]] || fail "missing experiment config: $config_path"
    [[ -f "$evaluation_protocol" ]] || fail "missing evaluation protocol"

    eval_name=${eval_name:-"${evaluation_phase}-${eval_subsets//,/-}"}
    validate_name "evaluation name" "$eval_name"
    root_path="${output_root}/${exp_name}/seed_${seed}"
    encoder_path="${root_path}/encoder"
    export_path="${root_path}/export"
    export_checkpoint="${export_path}/student.pt"
    finetune_path="${root_path}/finetune_runs/${run_name}"
    finetune_checkpoint="${finetune_path}/checkpoints/checkpoint_best.pt"
    evaluation_path="${finetune_path}/evaluations/${eval_name}"
    case "$root_path" in "${output_root}"/*) ;; *) fail "output path escaped root" ;; esac
}

activate_environment() {
    local conda_root="${CH3_CONDA_ROOT:-/home/zhengyangli/anaconda3}"
    local conda_environment="${CH3_CONDA_ENV:-dpavhubert_pro6000}"
    # shellcheck disable=SC1091
    source "${conda_root}/etc/profile.d/conda.sh"
    conda activate "$conda_environment"
}

print_command() {
    printf '$'
    printf ' %q' "$@"
    printf '\n'
}

sha256_if_file() {
    if [[ -f "$1" ]]; then sha256sum "$1" | awk '{print $1}'; fi
}

source_state() {
    source_commit=$(git -C "$project_path" rev-parse HEAD)
    source_tree=$(git -C "$project_path" rev-parse HEAD^{tree})
    source_branch=$(git -C "$project_path" branch --show-current)
    source_dirty=false
    if [[ -n "$(git -C "$project_path" status --porcelain=v1 --untracked-files=all)" ]]; then
        source_dirty=true
        printf 'warning: source checkout is dirty; provenance will record this state\n' >&2
    fi
}

runtime_preflight() {
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${project_path}/fairseq:${project_path}${PYTHONPATH:+:${PYTHONPATH}}" \
        python -c \
        'import fairseq.data.data_utils_fast, fairseq.libbleu, chapter3_distill_only'
}

require_checkpoint() {
    [[ -f "$1" ]] || fail "$2 is not a file: $1"
}

check_output() {
    local path=$1 label=$2 require_last=${3:-false}
    if $resume && [[ "$require_last" == "true" ]]; then
        require_checkpoint "${path}/checkpoints/checkpoint_last.pt" \
            "$label resume checkpoint"
        return
    fi
    if [[ -d "$path" && -n "$(find "$path" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        $resume || fail "$label output is nonempty: $path (use --resume)"
    fi
}

write_provenance() {
    local action=$1 directory=$2 stage_name=$3 config=$4 input=$5 output=$6 exit_code=$7
    shift 7
    python - "$action" "$directory" "$stage_name" "$config" "$input" "$output" \
        "$exit_code" "$project_path" "$source_branch" "$source_commit" \
        "$source_tree" "$source_dirty" "$@" <<'PY'
import datetime, hashlib, json, os, pathlib, sys, tempfile

(action, directory, stage, config, input_path, output_path, exit_code,
 repo, branch, commit, tree, dirty, *command) = sys.argv[1:]
directory = pathlib.Path(directory)
directory.mkdir(parents=True, exist_ok=True)
target = directory / "provenance.json"

def digest(value):
    path = pathlib.Path(value) if value else None
    if path is None or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

now = datetime.datetime.now(datetime.timezone.utc).isoformat()
if action == "start":
    payload = {
        "schema_version": "chapter3-stage-provenance/v1",
        "stage": stage,
        "status": "running",
        "start_timestamp": now,
        "source": {
            "repository": repo, "branch": branch, "commit": commit,
            "tree": tree, "dirty": dirty == "true",
        },
        "command": command,
        "config": {"path": config, "sha256": digest(config)},
        "resolved_config": None,
        "input_checkpoint": {"path": input_path, "sha256": digest(input_path)},
        "output_checkpoint": {"path": output_path, "sha256": None},
    }
else:
    payload = json.loads(target.read_text(encoding="utf-8"))
    resolved = pathlib.Path(directory) / ".hydra" / "config.yaml"
    payload.update({
        "status": "complete" if exit_code == "0" else "failed",
        "end_timestamp": now,
        "exit_code": int(exit_code),
        "resolved_config": {
            "path": str(resolved), "sha256": digest(str(resolved))
        } if resolved.is_file() else None,
    })
    payload["output_checkpoint"]["sha256"] = digest(output_path)
    if command:
        payload["command"] = command
with tempfile.NamedTemporaryFile(
    "w", dir=directory, prefix=".provenance.", delete=False, encoding="utf-8"
) as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
    temp = stream.name
os.replace(temp, target)
PY
}

run_command() {
    local stage_name=$1 directory=$2 config=$3 input=$4 output=$5
    shift 5
    local -a command=("$@")
    print_command "${command[@]}"
    $dry_run && return
    mkdir -p "$directory"
    write_provenance start "$directory" "$stage_name" "$config" "$input" \
        "$output" 0 "${command[@]}"
    set +e
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${project_path}/fairseq:${project_path}${PYTHONPATH:+:${PYTHONPATH}}" \
        "${command[@]}"
    local code=$?
    set -e
    write_provenance finish "$directory" "$stage_name" "$config" "$input" \
        "$output" "$code" "${command[@]}"
    [[ "$code" -eq 0 ]] || fail "$stage_name failed with exit code $code"
}

encoder_command() {
    local output=$1
    ENCODER_COMMAND=(
        fairseq-hydra-train
        --config-dir "${avhubert_dir}/conf/distill_only"
        --config-name "${exp_name}.yaml"
        "task.data=${data_path}"
        "task.label_dir=${data_path}"
        "task.tokenizer_bpe_model=${tokenizer_ckpt}"
        "model.teacher_path=${teacher_ckpt}"
        "distributed_training.distributed_world_size=${gpus}"
        "distributed_training.nprocs_per_node=${gpus}"
        "dataset.max_tokens=${max_tokens}"
        "optimization.update_freq=[${encoder_update_freq}]"
        "dataset.num_workers=${workers}"
        "common.seed=${seed}"
        "common.user_dir=${project_path}/chapter3_distill_only"
        "hydra.run.dir=${output}"
    )
}

run_encoder() {
    check_output "$encoder_path" "encoder" true
    encoder_command "$encoder_path"
    run_command encoder "$encoder_path" "$config_path" "$teacher_ckpt" \
        "${encoder_path}/checkpoints/checkpoint_last.pt" "${ENCODER_COMMAND[@]}"
}

run_stage1() {
    local output="${encoder_path}/stage1"
    check_output "$output" "stage1" true
    encoder_command "$output"
    ENCODER_COMMAND+=(
        "optimization.max_update=50000" "optimization.lr=[0.002]"
        "lr_scheduler.warmup_updates=15000"
        "lr_scheduler.total_num_update=50000"
        "model.schedule_mode=two_stage_50k_25k" "model.max_update=50000"
        "model.warmup_updates=15000"
    )
    run_command stage1 "$output" "$config_path" "$teacher_ckpt" \
        "${output}/checkpoints/checkpoint_last.pt" "${ENCODER_COMMAND[@]}"
}

run_stage2() {
    local output="${encoder_path}/stage2"
    local parent="${encoder_path}/stage1/checkpoints/checkpoint_last.pt"
    if ! $dry_run; then require_checkpoint "$parent" "stage-1 checkpoint"; fi
    check_output "$output" "stage2" true
    encoder_command "$output"
    ENCODER_COMMAND+=(
        "optimization.max_update=25000" "optimization.lr=[0.0001]"
        "lr_scheduler.warmup_updates=5000"
        "lr_scheduler.total_num_update=25000"
        "model.schedule_mode=two_stage_50k_25k" "model.max_update=25000"
        "model.warmup_updates=5000"
        "model.initialization_policy=warm_start_distilled"
        "checkpoint.finetune_from_model=${parent}"
    )
    run_command stage2 "$output" "$config_path" "$parent" \
        "${output}/checkpoints/checkpoint_last.pt" "${ENCODER_COMMAND[@]}"
}

run_export() {
    local parent="${encoder_path}/checkpoints/checkpoint_last.pt"
    [[ "$exp_name" == "s2_optional_two_stage" ]] &&
        parent="${encoder_path}/stage2/checkpoints/checkpoint_last.pt"
    if $resume; then
        require_checkpoint "$export_checkpoint" "export resume checkpoint"
        local -a verify=(python -m chapter3_distill_only.exporter
            --verify-existing "$export_checkpoint")
        run_command export "$export_path" "$config_path" "$parent" \
            "$export_checkpoint" "${verify[@]}"
        return
    fi
    check_output "$export_path" "export"
    if ! $dry_run; then require_checkpoint "$parent" "encoder checkpoint"; fi
    [[ ! -e "$export_checkpoint" ]] ||
        fail "export exists; pass --resume to verify and reuse it"
    local -a command=(python -m chapter3_distill_only.exporter
        --distilled-checkpoint "$parent" --output "$export_checkpoint")
    run_command export "$export_path" "$config_path" "$parent" \
        "$export_checkpoint" "${command[@]}"
}

run_finetune() {
    local parent="${input_checkpoint:-$export_checkpoint}"
    if ! $dry_run; then require_checkpoint "$parent" "fine-tuning input"; fi
    check_output "$finetune_path" "fine-tuning" true
    local config="${avhubert_dir}/conf/av-finetune/base_noise_pt_noise_ft_433h.yaml"
    local -a command=(
        fairseq-hydra-train
        --config-dir "${avhubert_dir}/conf/av-finetune"
        --config-name base_noise_pt_noise_ft_433h.yaml
        "task.data=${data_path}" "task.label_dir=${data_path}"
        "task.tokenizer_bpe_model=${tokenizer_ckpt}"
        "task.noise_wav=${finetune_noise_path}" "task.noise_prob=0.25"
        "task.noise_snr=0" "model.w2v_path=${parent}"
        "distributed_training.distributed_world_size=${gpus}"
        "distributed_training.nprocs_per_node=${gpus}"
        "optimization.lr=[${finetune_lr}]"
        "optimization.update_freq=[${finetune_update_freq}]"
        "dataset.num_workers=${workers}" "common.seed=${seed}"
        "common.user_dir=${project_path}/chapter3_distill_only"
        "hydra.run.dir=${finetune_path}"
    )
    run_command finetune "$finetune_path" "$config" "$parent" \
        "$finetune_checkpoint" "${command[@]}"
}

condition_complete() {
    [[ -d "$1" ]] && find "$1" -maxdepth 1 -type f -name 'wer.*' -print -quit |
        grep -q .
}

run_evaluate() {
    local parent="${input_checkpoint:-$finetune_checkpoint}"
    if ! $dry_run; then require_checkpoint "$parent" "evaluation checkpoint"; fi
    check_output "$evaluation_path" "evaluation"
    local rows
    rows=$(python "$condition_helper" --protocol "$evaluation_protocol" \
        --phase "$evaluation_phase" --subsets "$eval_subsets")
    local -a commands=()
    local -a marker=(python "$condition_helper" --protocol "$evaluation_protocol"
        --phase "$evaluation_phase" --subsets "$eval_subsets")
    if ! $dry_run; then
        mkdir -p "$evaluation_path"
        write_provenance start "$evaluation_path" evaluate "$evaluation_protocol" \
            "$parent" "" 0 "${marker[@]}"
    fi
    while IFS=$'\t' read -r name subset noise_type noise_root snr method eval_seed relative; do
        [[ -n "$name" ]] || continue
        [[ "$noise_type" == "-" ]] && noise_type=""
        [[ "$noise_root" == "-" ]] && noise_root=""
        [[ "$snr" == "-" ]] && snr=""
        [[ "$method" == "-" ]] && method=""
        local result="${evaluation_path}/${relative}"
        if $resume && condition_complete "$result"; then
            printf 'Skipping completed evaluation: %s/%s\n' "$subset" "$name"
            continue
        fi
        local -a command=(
            python -B "${avhubert_dir}/infer_s2s.py"
            --config-dir "${avhubert_dir}/conf" --config-name s2s_decode.yaml
            "dataset.gen_subset=${subset}" "common_eval.path=${parent}"
            "common_eval.results_path=${result}"
            "override.modalities=['audio','video']"
            "hydra.run.dir=${result}"
            "common.user_dir=${project_path}/chapter3_distill_only"
            "common.seed=${eval_seed}"
        )
        if [[ -n "$noise_type" ]]; then
            command+=(
                "override.noise_wav=${noise_root}" "override.noise_prob=1"
                "override.noise_snr=${snr}" "override.noise_method=${method}"
            )
        fi
        print_command "${command[@]}"
        local command_text
        printf -v command_text '%q ' "${command[@]}"
        commands+=("${command_text% }")
        if ! $dry_run; then
            mkdir -p "$result"
            set +e
            PYTHONDONTWRITEBYTECODE=1 \
            PYTHONPATH="${project_path}/fairseq:${project_path}${PYTHONPATH:+:${PYTHONPATH}}" \
                "${command[@]}"
            local code=$?
            set -e
            if [[ "$code" -ne 0 ]]; then
                write_provenance finish "$evaluation_path" evaluate \
                    "$evaluation_protocol" "$parent" "" "$code" "${commands[@]}"
                fail "evaluation condition $subset/$name failed with exit code $code"
            fi
        fi
    done <<<"$rows"
    $dry_run && return
    write_provenance finish "$evaluation_path" evaluate "$evaluation_protocol" \
        "$parent" "" 0 "${commands[@]}"
}

dispatch() {
    case "$stage" in
        encoder) run_encoder ;;
        stage1) run_stage1 ;;
        stage2) run_stage2 ;;
        export) run_export ;;
        finetune) run_finetune ;;
        evaluate) run_evaluate ;;
        all)
            if [[ "$exp_name" == "s2_optional_two_stage" ]]; then
                run_stage1; run_stage2
            else
                run_encoder
            fi
            run_export; run_finetune; run_evaluate
            ;;
    esac
}

print_plan() {
    printf 'Experiment: %s\nStage: %s\nSeed: %s\n' "$exp_name" "$stage" "$seed"
    printf 'Config: %s\nRoot output: %s\n' "$config_path" "$root_path"
    printf 'Encoder output: %s\nExport checkpoint: %s\n' \
        "$encoder_path" "$export_checkpoint"
    printf 'Fine-tuning output: %s\nEvaluation output: %s\n' \
        "$finetune_path" "$evaluation_path"
    printf 'Fine-tuning LR: %s\n' "$finetune_lr"
    printf 'Source: %s %s (dirty=%s)\n' \
        "$source_branch" "$source_commit" "$source_dirty"
}

main() {
    parse_args "$@"
    validate_args
    resolve_paths
    source_state
    cd "$project_path"
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONPATH="${project_path}/fairseq:${project_path}${PYTHONPATH:+:${PYTHONPATH}}"
    print_plan
    if ! $dry_run; then
        activate_environment
        runtime_preflight
    fi
    dispatch
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
