#!/usr/bin/env bash
# Submit the v0 ablation ladder (x4 light, all 20 scenes, 600 iterations each) as a chain of jobs that
# run one after another on one CPU node (job name mfsr-abl). Run on the login node: it only calls sbatch.
#   bash jobs/ablation.sh            (FACTOR=8 bash jobs/ablation.sh for x8)
# Each step adds one thing; the results land in results/gsmisr_a1 ... gsmisr_a9 and the report lists them.
#   a1  s2-superres-like: fixed grid of round, fixed-size Gaussians, shifts only, phase-correlation start,
#       no colour model, no noise weighting
#   a2  + exhaustive registration search (+30-iteration warm-up)
#   a3  + per-frame colour transform
#   a4  + per-frame noise weighting (from residuals)
#   a5  + movable centres
#   a6  + free sizes (still round)
#   a7  + anisotropic, rotated Gaussians = full v0 (at 600 iterations)
#   a8  full v0 but the pixel box approximated by a Gaussian on the LR grid (tests exact pixel integration)
#   a9  full v0 with the true shifts fixed (diagnostic upper bound: no alignment error)
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f slurm/site.sh ]; then source slurm/site.sh; fi   # site settings, e.g. SBATCH_ACCOUNT (see README)
F=${FACTOR:-4}; AUG=${AUG:-light}; IT=${ITERS:-600}
A1="--max_off 0 --isotropic true --s_lo 0.4 --s_hi 0.4 --rotate false --color none --noise none --init_shifts phasecorr --align_warmup 0"
A2="--max_off 0 --isotropic true --s_lo 0.4 --s_hi 0.4 --rotate false --color none --noise none"
A3="--max_off 0 --isotropic true --s_lo 0.4 --s_hi 0.4 --rotate false --noise none"
A4="--max_off 0 --isotropic true --s_lo 0.4 --s_hi 0.4 --rotate false"
A5="--isotropic true --s_lo 0.4 --s_hi 0.4 --rotate false"
A6="--isotropic true --rotate false"
A7=""
A8="--sub 1 --sub_box true"
A9="--oracle_shifts"
for i in 1 2 3 4 5 6 7 8 9; do
    eval "args=\$A$i"
    GS_ARGS="--iters $IT $args" sbatch -J mfsr-abl --dependency=singleton slurm/stage.sbatch "gsmisr_a$i" "$F" "$AUG"
done
squeue -u "$USER" -n mfsr-abl -o "%.10i %.14j %.8T %.20R"
