#!/usr/bin/env bash
# SuperF MSE with N frames on clean bursts (x4 `none`), next to our frames sweep (gsmisr_tv_n*).
# SuperF with all 16 frames is results/superf_mse. Run it on a compute node; resumable.
#   bash jobs/superf_frames.sh 4     ->  results/superf_mse_n4
#   on a GPU node: MFSR_DEVICE=cuda THREADS=4 bash jobs/superf_frames.sh 8
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1
source slurm/config.sh
set +u; activate_env || exit 1; set -u
N=${1:?number of frames}; F=${2:-4}; D="$DATA/satsynthburst"
echo "=== [$(date '+%H:%M:%S')] SuperF MSE with $N frames, x$F none (${MFSR_DEVICE:-cpu})"
python jobs/pool.py --workers "${WORKERS:-4}" --threads "${THREADS:-8}" --data_root "$D" --df "$F" --aug none --shift 1.0 \
    --logdir "results/logs/superf_mse_n${N}_none" -- python runners/run_superf.py --variant superf_mse \
    --tag "superf_mse_n$N" --num_frames "$N" --data_root "$D" --results results --aug none --shift 1.0 \
    --skip_existing --device "${MFSR_DEVICE:-cpu}" --df "{df}" --scenes "{scene}"
echo "=== [$(date '+%H:%M:%S')] done"
