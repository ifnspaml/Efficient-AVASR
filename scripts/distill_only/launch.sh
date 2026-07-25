#!/usr/bin/env bash
#SBATCH --time=8-00:00:00
#SBATCH --partition=ifn
#SBATCH --gres=gpu:pro6000b_96gb:1
#SBATCH --exclude=gpu[04,05]
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48gb

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
conda_root="${CH3_CONDA_ROOT:-/home/zhengyangli/anaconda3}"
conda_environment="${CH3_CONDA_ENV:-dpavhubert_pro6000}"

# Match the existing pro6000 launch environment while keeping the choice explicit
# and recordable through CH3_CONDA_ENV.
source "${conda_root}/etc/profile.d/conda.sh"
conda activate "${conda_environment}"

cd "${repo_root}"
exec python "${script_dir}/launch.py" "$@"
