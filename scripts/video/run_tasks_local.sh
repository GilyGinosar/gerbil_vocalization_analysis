#!/bin/bash
# Run a burrow_scan task file locally, N at a time -- no Slurm, no GPU.
# The work is I/O-bound on ceph rather than CPU-bound, so a modest N still
# gets most of the benefit without occupying the whole workstation.
#
#   bash scripts/video/run_tasks_local.sh slurm/burrow_scan_sample.tasks 6
set -euo pipefail
TASKFILE="${1:?usage: run_tasks_local.sh <taskfile> [workers]}"
WORKERS="${2:-6}"
TOTAL=$(wc -l < "$TASKFILE")
echo "running $TOTAL tasks, $WORKERS at a time"
xargs -a "$TASKFILE" -d '\n' -P "$WORKERS" -I{} bash -c '{}' > /dev/null
echo "done"
