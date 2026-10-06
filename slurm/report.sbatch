#!/usr/bin/env bash
#SBATCH --job-name=mfsr-report
#SBATCH -p cpup
#SBATCH -N 1 -n 1 -c 8
#SBATCH -t 01:00:00
#SBATCH -o slurm_logs/%x-%j.out
# Re-score every prediction and write results/report.md. Read-only on predictions, so it is
# safe to run while other MFSR jobs are still adding results; rerun it any time.
#   sbatch slurm/report.sbatch      then: cat results/report.md
#   non-integer scales: DATA_ROOT=$PWD/data/arbscale RES=results_arb sbatch slurm/report.sbatch
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1
source slurm/config.sh
set +u; activate_env || exit 1; set -u
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
D="${DATA_ROOT:-$DATA/satsynthburst}"; RES="${RES:-results}"
mkdir -p "$RES"
echo "=== [$(date '+%H:%M:%S')] scoring (code + paper protocols)"
python eval/evaluate.py --data_root "$D" --results "$RES" --protocol both > "$RES"/evaluate.log 2>&1 \
    || { echo "!!! evaluate failed"; tail -5 "$RES"/evaluate.log; }
echo "=== [$(date '+%H:%M:%S')] scoring with the superf_compat colour match"
python eval/evaluate.py --data_root "$D" --results "$RES" --protocol paper --colormatch superf_compat \
    --out "$RES"/scores_compat.csv > "$RES"/evaluate_compat.log 2>&1 || echo "!!! compat scoring failed"
echo "=== [$(date '+%H:%M:%S')] LPIPS (a failure here doesn't matter)"
python eval/evaluate.py --data_root "$D" --results "$RES" --protocol paper --lpips \
    --out "$RES"/scores_lpips.csv > "$RES"/evaluate_lpips.log 2>&1 || echo "!!! LPIPS failed (see $RES/evaluate_lpips.log)"
echo "=== [$(date '+%H:%M:%S')] sub-pixel coverage vs gain"
python eval/coverage.py --data_root "$D" --results "$RES" --aug light > "$RES"/coverage.log 2>&1 \
    || echo "!!! coverage failed (see $RES/coverage.log)"
echo "=== [$(date '+%H:%M:%S')] report"
python eval/report.py --results "$RES" > "$RES"/report.md 2> "$RES"/report.err \
    || { echo "!!! report failed"; tail -20 "$RES"/report.err; }
cat "$RES"/report.md
cat "$RES"/coverage_light.md 2>/dev/null
echo "=== [$(date '+%H:%M:%S')] done"
