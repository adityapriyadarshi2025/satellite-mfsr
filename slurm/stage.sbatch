#!/usr/bin/env bash
#SBATCH --job-name=mfsr-stage
#SBATCH -p cpup
#SBATCH -N 1 -n 1 -c 32
#SBATCH -t 04:00:00
#SBATCH -o slurm_logs/%x-%j.out
# One method, one factor, one dataset version; 4 bursts at a time, 8 threads each.
# Resumable: finished bursts are skipped.
#   sbatch slurm/stage.sbatch superf_mse 4 none
#   args: METHOD (superf_mse | superf_gnll | superf_mse_exact | nir | s2superres | gsmisr[_variant])  FACTOR (2 | 4 | 8)  AUG (light | none | ...)
#   gsmisr variants: the method name is the results tag; extra flags via GS_ARGS, e.g.
#     GS_ARGS="--iters 1000 --frames_per_iter 8" sbatch slurm/stage.sbatch gsmisr 4 light
#   s2superres variants likewise via S2_ARGS, e.g.
#     S2_ARGS="--init search" sbatch slurm/stage.sbatch s2superres_search 4 light
#   On a GPU node: MFSR_DEVICE=cuda THREADS=4 bash slurm/stage.sbatch superf_mse 4 none
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1
source slurm/config.sh
set +u; activate_env || exit 1; set -u
METHOD=${1:?method}; FACTOR=${2:?factor}; AUG=${3:?aug}
W=${WORKERS:-4}; TH=${THREADS:-8}; SHIFT=${SHIFT:-1.0}; RES=${RES:-results}; DEV=${MFSR_DEVICE:-cpu}
DATA_ROOT="${DATA_ROOT:-$DATA/satsynthburst}"   # non-integer scales: DATA_ROOT=$PWD/data/arbscale RES=results_arb
case "$METHOD" in
    superf_mse|superf_gnll|nir) RUN=(python runners/run_superf.py --variant "$METHOD") ;;
    superf_mse_exact|superf_gnll_exact)   # SuperF with an exact pixel-average downsampler (fairness check at non-integer scales)
        RUN=(python runners/run_superf.py --variant "${METHOD%_exact}" --tag "$METHOD" --exact_area) ;;
    bilinear|bicubic) RUN=(python runners/run_bilinear.py --interp "$METHOD") ;;
    s2superres) RUN=(python runners/run_s2superres.py) ;;
    s2superres_*) RUN=(python runners/run_s2superres.py --tag "$METHOD" ${S2_ARGS:-}) ;;
    gsmisr*) RUN=(python runners/run_gsmisr.py --tag "$METHOD" --threads "$TH" ${GS_ARGS:-}) ;;
    *) echo "unknown method $METHOD"; exit 1 ;;
esac
echo "=== [$(date '+%H:%M:%S')] $METHOD x$FACTOR [$AUG] on $(hostname) ($DEV), $W bursts at a time x $TH threads"
python jobs/pool.py --workers "$W" --threads "$TH" --data_root "$DATA_ROOT" --df "$FACTOR" --aug "$AUG" \
    --shift "$SHIFT" --logdir "$RES/logs/${METHOD}_${AUG}" -- "${RUN[@]}" --data_root "$DATA_ROOT" \
    --results "$RES" --aug "$AUG" --shift "$SHIFT" --skip_existing --device "$DEV" --df "{df}" --scenes "{scene}"
echo "=== [$(date '+%H:%M:%S')] done"
