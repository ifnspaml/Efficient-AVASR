#!/usr/bin/env bash
#SBATCH --time=2-00:00:00
#SBATCH --partition=ifn
#SBATCH --qos=low
#SBATCH --gres=gpu:pro6000b_96gb:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48gb
#SBATCH --job-name=ch3-v3-final-eval

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=distill_only/common.sh
source "${script_dir}/distill_only/common.sh"

project_path="${CH3_PROJECT_PATH:-$(cd "${script_dir}/.." && pwd)}"
registry=""
models=""
eval_name=itut-final
seed=1337
resume=false
dry_run=false

usage() {
    cat <<'EOF'
Usage: scripts_v3/run_ch3_final_evaluation.sh --registry PATH --models ID[,ID...] [options]

  --eval-name NAME
  --seed INTEGER
  --resume
  --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --registry) ch3_need_value "$@"; registry=$2; shift 2 ;;
        --models) ch3_need_value "$@"; models=$2; shift 2 ;;
        --eval-name) ch3_need_value "$@"; eval_name=$2; shift 2 ;;
        --seed) ch3_need_value "$@"; seed=$2; shift 2 ;;
        --resume) resume=true; shift ;;
        --dry-run) dry_run=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) ch3_fail "unknown option: $1" ;;
    esac
done

[[ -n "$registry" ]] || ch3_fail "--registry is required"
[[ -n "$models" ]] || ch3_fail "--models is required"
ch3_validate_name "evaluation name" "$eval_name"
[[ "$seed" =~ ^[0-9]+$ ]] || ch3_fail "seed must be a nonnegative integer"
project_path=$(cd "$project_path" && pwd)
rows=$(python "${script_dir}/distill_only/final_registry.py" --registry "$registry" --models "$models")
printf 'Scientific context: %s\n' "$CH3_CONTEXT_URL"
printf 'Final evaluation registry: %s\nModels: %s\n' "$registry" "$models"
if [[ "$dry_run" != true ]]; then ch3_activate_environment; cd "$project_path"; fi
while IFS=$'\t' read -r identifier checkpoint; do
    [[ -n "$identifier" ]] || continue
    ch3_validate_name "model identifier" "$identifier"
    output="${project_path}/exp/chapter3_distill_only/stage_f/${identifier}/seed_${seed}/evaluations/${eval_name}"
    ch3_run_evaluation "$project_path" "$checkpoint" "$output" final valid,test "${project_path}/chapter3_distill_only" "$resume" "$dry_run"
done <<<"$rows"
