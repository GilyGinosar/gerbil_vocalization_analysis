#!/bin/bash
#SBATCH --job-name=average-audio
#SBATCH --output=/mnt/home/gginosar/repos/gerbil_vocalization_analysis/slurm/slurm-%A_%a.out
#SBATCH --error=/mnt/home/gginosar/repos/gerbil_vocalization_analysis/slurm/slurm-%A_%a.err
#SBATCH -c 8
#SBATCH --mem=24GB
#SBATCH --time=24:00:00
#SBATCH --array=115

set -euo pipefail

PROJECT_ROOT="/mnt/home/gginosar/repos/gerbil_vocalization_analysis"
EXPERIMENT_ID="${SLURM_ARRAY_TASK_ID}"

source "${PROJECT_ROOT}/.venv/bin/activate"
export UV_LINK_MODE=copy

mkdir -p "${PROJECT_ROOT}/slurm"

cd "${PROJECT_ROOT}"
python scripts/run_average_audio_batch.py --experiment-id "${EXPERIMENT_ID}"
