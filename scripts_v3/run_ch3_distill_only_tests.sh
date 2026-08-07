#!/usr/bin/env bash
#SBATCH --time=0-01:00:00
#SBATCH --partition=ifn
#SBATCH --gres=gpu:pro6000b_24gb:1
#SBATCH --qos=low
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32gb
#SBATCH --job-name=ch3-v3-tests
# These unit tests are CPU-only; a GPU is requested only because partition ifn
# typically requires --gres.

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
        printf 'error: cannot locate scripts_v3/distill_only/common.sh (got script_dir=%s)\n' \
            "$script_dir" >&2
        exit 2
    fi
fi
# shellcheck source=distill_only/common.sh
source "${script_dir}/distill_only/common.sh"

project_path="${CH3_PROJECT_PATH:-$(cd "${script_dir}/.." && pwd)}"
# Project root must come first. fairseq/ also has a top-level tests/ package;
# with project tests/__init__.py present, ours wins when listed first.
export PYTHONPATH="${project_path}:${project_path}/fairseq:${project_path}/avhubert${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=
export PYTHONDONTWRITEBYTECODE=1

ch3_activate_environment
cd "$project_path"

if [[ ! -f tests/__init__.py || ! -f tests/distill_only/__init__.py ]]; then
    ch3_fail "missing tests package markers under ${project_path}/tests"
fi
if [[ ! -f tests/distill_only/test_noise_isolation.py ]]; then
    ch3_fail "missing tests/distill_only/test_noise_isolation.py under ${project_path}"
fi

# Default: a4 noise-isolation + A-matrix config checks. Pass modules after --.
if [[ $# -eq 0 || "${1:-}" == "--" ]]; then
    [[ "${1:-}" == "--" ]] && shift
    set -- \
        tests.distill_only.test_noise_isolation \
        tests.distill_only.test_v3_suite.V3ConfigTest.test_a_matrix_and_fixed_protocol \
        "$@"
fi

printf 'Project: %s\nPython: %s\nPYTHONPATH[0]=%s\n' \
    "$project_path" "$(command -v python)" "${project_path}"
python - <<'PY'
import tests, tests.distill_only
print("tests ->", tests.__file__)
print("tests.distill_only ->", tests.distill_only.__file__)
PY
ch3_print_command python -m unittest "$@" -v
python -m unittest "$@" -v
