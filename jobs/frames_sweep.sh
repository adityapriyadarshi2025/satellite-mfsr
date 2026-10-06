#!/usr/bin/env bash
# Submit the frames-count sweep: gsmisr with N = 2, 4, 8, 16, 32 frames at x4 and x8, as a chain of
# jobs that run one after another on one CPU node (job name mfsr-sweep). Run on the login node.
#   AUG=none bash jobs/frames_sweep.sh       (then AUG=light)
# Results: results/gsmisr_n2 ... gsmisr_n32 (n16 is re-run at the same iteration count for a fair curve).
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f slurm/site.sh ]; then source slurm/site.sh; fi   # site settings, e.g. SBATCH_ACCOUNT (see README)
AUG=${AUG:-none}; IT=${ITERS:-600}
for F in 4 8; do
    for N in 2 4 8 16 32; do
        GS_ARGS="--num_frames $N --iters $IT" sbatch -J mfsr-sweep --dependency=singleton slurm/stage.sbatch "gsmisr_n$N" "$F" "$AUG"
    done
done
squeue -u "$USER" -n mfsr-sweep -o "%.10i %.14j %.8T %.20R"
