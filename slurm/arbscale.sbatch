#!/usr/bin/env bash
#SBATCH --job-name=mfsr-arb
#SBATCH -p cpup
#SBATCH -N 1 -n 1 -c 32
#SBATCH -t 04:00:00
#SBATCH -o slurm_logs/%x-%j.out
# Arbitrary (including non-integer) scales. Builds data/arbscale from the official SatSynthBurst if it
# isn't there yet (same 20 HR images and shifts, scales 2.56 3.2 4 5.12 6.4 plus 2 and 8), then runs
# bicubic, bilinear and gsmisr on the non-integer scales and the x4 control, then scores into results_arb/.
# Resumable: finished bursts are skipped, so just resubmit if it hits the time limit.
#   sbatch slurm/arbscale.sbatch none          (then: sbatch slurm/arbscale.sbatch light)
#   a v0 variant instead of the default:  GS_TAG=gsmisr_tv GS_EXTRA="--tv 3e-4" sbatch slurm/arbscale.sbatch none
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1
source slurm/config.sh
set +u; activate_env || exit 1; set -u
AUG=${1:-none}
export DATA_ROOT="$DATA/arbscale" RES=results_arb
mkdir -p "$RES"
if [ ! -f "$DATA_ROOT/.complete" ]; then
    echo "=== [$(date '+%H:%M:%S')] building $DATA_ROOT"
    python setup/make_arbscale.py --src "$DATA/satsynthburst" --out "$DATA_ROOT" || { echo "!!! data build failed"; exit 1; }
    python setup/make_arbscale.py --src "$DATA/satsynthburst" --check | tee "$RES/fidelity.txt"
    touch "$DATA_ROOT/.complete"
fi
for F in 2.56 3.2 5.12 6.4 4; do
    for M in bicubic bilinear; do
        bash slurm/stage.sbatch "$M" "$F" "$AUG" > /dev/null || echo "!!! $M x$F failed"
    done
    GS_ARGS="--iters ${ITERS:-600} ${GS_EXTRA:-}" bash slurm/stage.sbatch "${GS_TAG:-gsmisr}" "$F" "$AUG" \
        || echo "!!! ${GS_TAG:-gsmisr} x$F failed"
done
bash slurm/report.sbatch
echo "=== [$(date '+%H:%M:%S')] arbscale $AUG done"
