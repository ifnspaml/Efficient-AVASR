#!/usr/bin/env bash
#SBATCH --time=8-00:00:00
#SBATCH --partition=ifn
#SBATCH --gres=gpu:pro6000b_96gb:1
#SBATCH --exclude=gpu[04,05]
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48gb

set -euo pipefail

# Slurm copies this script into /var/spool/slurmd/job*/ before execution, so
# BASH_SOURCE no longer points at the repository copy. Prefer the directory
# from which sbatch was invoked, then fall back to the interactive path.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  if [[ -f "${SLURM_SUBMIT_DIR}/scripts/distill_only/launch.py" ]]; then
    repo_root="$(cd -- "${SLURM_SUBMIT_DIR}" && pwd)"
    script_dir="${repo_root}/scripts/distill_only"
  elif [[ -f "${SLURM_SUBMIT_DIR}/launch.py" ]]; then
    script_dir="$(cd -- "${SLURM_SUBMIT_DIR}" && pwd)"
    repo_root="$(cd -- "${script_dir}/../.." && pwd)"
  else
    echo "error: cannot locate scripts/distill_only/launch.py from SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}" >&2
    exit 1
  fi
else
  script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  repo_root="$(cd -- "${script_dir}/../.." && pwd)"
fi

if [[ ! -f "${script_dir}/launch.py" ]]; then
  echo "error: launch.py not found at ${script_dir}/launch.py" >&2
  exit 1
fi

conda_root="${CH3_CONDA_ROOT:-/home/zhengyangli/anaconda3}"
conda_environment="${CH3_CONDA_ENV:-dpavhubert_pro6000}"

# Match the existing pro6000 launch environment while keeping the choice explicit
# and recordable through CH3_CONDA_ENV.
source "${conda_root}/etc/profile.d/conda.sh"
conda activate "${conda_environment}"

cd "${repo_root}"
exec python "${script_dir}/launch.py" "$@"
