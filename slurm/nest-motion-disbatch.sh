#!/bin/bash
#SBATCH --job-name=nest-motion
#SBATCH --output=/mnt/home/gginosar/repos/gerbil_vocalization_analysis/slurm/slurm-%A.out
#SBATCH --error=/mnt/home/gginosar/repos/gerbil_vocalization_analysis/slurm/slurm-%A.err
#SBATCH --partition=gen
#SBATCH -N 1
#SBATCH --exclusive           # gen is OverSubscribe=EXCLUSIVE, same as burrow-scan:
                              # take the whole node rather than idle half of it.
#SBATCH --time=1:00:00        # 7.6 core-hours of work: ~7 min on 64 cores. An hour is
                              # generous cover for a slow filesystem, not an estimate.
#
# Pre-entry nest motion for every to_nest traverse of a date folder. CPU only --
# this is OpenCV frame differencing over decoded video, nothing touches a GPU, so
# do NOT send it to a GPU partition where it would queue behind real GPU work.
#
#   python slurm/make_nest_motion_tasks.py --date 2026_02 \
#       --scan /mnt/ceph/users/gginosar/burrow_scan_2026_02 \
#       --out-dir /mnt/ceph/users/gginosar/nest_motion_2026_02 \
#       > slurm/nest_motion_2026_02.tasks
#   sbatch slurm/nest-motion-disbatch.sh slurm/nest_motion_2026_02.tasks
#
# Then pool the per-shard CSVs into one table:
#   python scripts/analysis/pool_nest_motion.py \
#       --in-dir /mnt/ceph/users/gginosar/nest_motion_2026_02 \
#       --out exports/burrow/nest_motion/nest_motion_full.csv

set -euo pipefail

PROJECT_ROOT="/mnt/home/gginosar/repos/gerbil_vocalization_analysis"
TASKFILE="${1:?usage: sbatch slurm/nest-motion-disbatch.sh <taskfile>}"

source "${PROJECT_ROOT}/.venv/bin/activate"
export UV_LINK_MODE=copy
module -q load disBatch

cd "${PROJECT_ROOT}"
disBatch -p "${PROJECT_ROOT}/slurm/" --status-header "${TASKFILE}"
