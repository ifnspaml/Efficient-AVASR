#!/usr/bin/env bash

CH3_CONTEXT_URL="https://github.com/joyolee/PhD-Thesis-Zhengyang-Li-2026/issues/12#issue-4905908061"

ch3_fail() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

ch3_need_value() {
    [[ $# -ge 2 && -n "${2:-}" ]] || ch3_fail "$1 requires a value"
}

ch3_validate_name() {
    local label=$1 value=$2
    [[ -n "$value" ]] || ch3_fail "$label must not be empty"
    [[ "$value" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] ||
        ch3_fail "$label contains unsafe characters: $value"
    [[ "$value" != *".."* ]] || ch3_fail "$label must not contain '..'"
}

ch3_validate_positive_float() {
    python - "$1" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit(1)
raise SystemExit(0 if math.isfinite(value) and value > 0 else 1)
PY
}

ch3_print_command() {
    printf '$'
    printf ' %q' "$@"
    printf '\n'
}

ch3_activate_environment() {
    local conda_root="${CH3_CONDA_ROOT:-/home/zhengyangli/anaconda3}"
    local conda_environment="${CH3_CONDA_ENV:-dpavhubert_pro6000}"
    # shellcheck disable=SC1091
    source "${conda_root}/etc/profile.d/conda.sh"
    conda activate "$conda_environment"
}

ch3_require_checkpoint() {
    local checkpoint=$1 label=$2
    [[ -f "$checkpoint" ]] || ch3_fail "$label is not a file: $checkpoint"
}

ch3_require_loadable_checkpoint() {
    local checkpoint=$1 label=$2
    ch3_require_checkpoint "$checkpoint" "$label"
    python - "$checkpoint" <<'PY'
import sys
import torch

value = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if not isinstance(value, dict) or "model" not in value:
    raise SystemExit("checkpoint is not a Fairseq model checkpoint")
PY
}

ch3_directory_nonempty() {
    [[ -d "$1" ]] && [[ -n "$(find "$1" -mindepth 1 -maxdepth 1 -print -quit)" ]]
}

ch3_reject_nonempty() {
    local directory=$1 label=$2 resume=$3
    if ch3_directory_nonempty "$directory" && [[ "$resume" != true ]]; then
        ch3_fail "$label output is nonempty: $directory (use --resume)"
    fi
}

ch3_condition_complete() {
    [[ -d "$1" ]] && find "$1" -maxdepth 1 -type f -name 'wer.*' -print -quit |
        grep -q .
}

ch3_run_evaluation() {
    local project_path=$1 checkpoint=$2 output=$3 phase=$4 subsets=$5
    local user_dir=$6 resume=$7 dry_run=$8
    local protocol="${project_path}/scripts/distill_only/evaluation_protocol_itut.yaml"
    local helper="${project_path}/scripts/distill_only/evaluation_conditions.py"
    [[ -f "$protocol" ]] || ch3_fail "missing evaluation protocol: $protocol"
    [[ -f "$helper" ]] || ch3_fail "missing evaluation helper: $helper"
    if [[ "$dry_run" != true ]]; then
        ch3_require_checkpoint "$checkpoint" "evaluation checkpoint"
    fi
    ch3_reject_nonempty "$output" "evaluation" "$resume"
    local rows
    rows=$(python "$helper" --protocol "$protocol" --phase "$phase" --subsets "$subsets")
    while IFS=$'\t' read -r name subset noise_type noise_root snr method eval_seed relative; do
        [[ -n "$name" ]] || continue
        [[ "$noise_type" == "-" ]] && noise_type=""
        [[ "$noise_root" == "-" ]] && noise_root=""
        [[ "$snr" == "-" ]] && snr=""
        [[ "$method" == "-" ]] && method=""
        local result="${output}/${relative}"
        if [[ "$resume" == true ]] && ch3_condition_complete "$result"; then
            printf 'Skipping completed evaluation: %s/%s\n' "$subset" "$name"
            continue
        fi
        local -a command=(
            python -B "${project_path}/avhubert/infer_s2s.py"
            --config-dir "${project_path}/avhubert/conf"
            --config-name s2s_decode.yaml
            "dataset.gen_subset=${subset}"
            "common_eval.path=${checkpoint}"
            "common_eval.results_path=${result}"
            "override.modalities=['audio','video']"
            "hydra.run.dir=${result}"
            "common.user_dir=${user_dir}"
            "common.seed=${eval_seed}"
        )
        if [[ -n "$noise_type" ]]; then
            command+=(
                "override.noise_wav=${noise_root}"
                "override.noise_prob=1"
                "override.noise_snr=${snr}"
                "override.noise_method=${method}"
            )
        fi
        ch3_print_command "${command[@]}"
        if [[ "$dry_run" != true ]]; then
            mkdir -p "$result"
            PYTHONDONTWRITEBYTECODE=1 \
            PYTHONPATH="${project_path}/fairseq:${project_path}/avhubert:${project_path}${PYTHONPATH:+:${PYTHONPATH}}" \
                "${command[@]}"
        fi
    done <<<"$rows"
}
