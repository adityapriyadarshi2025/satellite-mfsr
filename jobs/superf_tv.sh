#!/usr/bin/env bash
# Control for "is the gain from the prior or from the Gaussians?": SuperF GNLL + the same TV prior as our
# TV version (runners/run_superf_tv.py). Run it on a compute node; resumable.
#   bash jobs/superf_tv.sh 4      (first call: picks the weight, then all 20 scenes at x4)
#   bash jobs/superf_tv.sh 8      (later calls reuse the weight picked at x4)
# Weight: same tuning budget our TV weight got. Pilot on the 2 scenes ours was picked on (smoothest and
# most detailed at x4): weights 0.1 1 10 100 (SuperF's GNLL divides residuals by learned variances, so its
# scale is far from gsmisr's 1e-3), one more decade if the best is at an edge, then best x3 and best /3;
# best mean PSNR wins; saved in results/superf_tv_weight.txt.
# Results: results/superf_gnll_tv (pilot runs: results/superf_gnll_tvp<w>).
# Overrides (env): TV_WEIGHT (skip the pilot, use this weight), TV_TAG (results folder), MFSR_DEVICE (cuda on a
# GPU node), THREADS, WORKERS. Both arms of the control on one device:
#   MFSR_DEVICE=cuda THREADS=4 TV_TAG=superf_gnll_gpu TV_WEIGHT=0 bash jobs/superf_tv.sh 4   (SuperF GNLL, no prior)
#   MFSR_DEVICE=cuda THREADS=4 bash jobs/superf_tv.sh 4                                      (+ prior, pilot weight)
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1
source slurm/config.sh
set +u; activate_env || exit 1; set -u
F=${1:-4}; D="$DATA/satsynthburst"; TH=${THREADS:-8}; TAG=${TV_TAG:-superf_gnll_tv}
PILOT=("Landcover-242541_rgb" "Landcover-770726_rgb")
WFILE=results/superf_tv_weight.txt
mkdir -p results/logs
common=(--data_root "$D" --results results --aug light --shift 1.0 --skip_existing --device "${MFSR_DEVICE:-cpu}")

pilot() {   # pilot <weight>...: one process (8 threads) per weight and scene, 4 at a time
    local w sc
    for w in "$@"; do
        for sc in "${PILOT[@]}"; do
            while [ "$(jobs -rp | wc -l)" -ge 4 ]; do wait -n; done
            OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH python runners/run_superf_tv.py --variant superf_gnll --tv "$w" \
                --tag "superf_gnll_tvp$w" --df 4 --scenes "$sc" "${common[@]}" \
                > "results/logs/superf_tv_pilot_${w}_${sc}.log" 2>&1 &
        done
    done
    wait
}
best_weight() {   # score all pilot runs, print the weight with the best mean PSNR (needs both scenes)
    local tags
    tags=$(ls -d results/superf_gnll_tvp* 2>/dev/null | xargs -n1 basename)
    [ -n "$tags" ] || return 1
    python eval/evaluate.py --data_root "$D" --results results --protocol paper --methods $tags \
        --out results/scores_superf_tv_pilot.csv > results/logs/superf_tv_pilot_eval.log 2>&1 || return 1
    python - <<'PY'
import pandas as pd
d = pd.read_csv("results/scores_superf_tv_pilot.csv", dtype={"aug": str, "shift": str})
d = d[(d.df == 4) & (d.aug == "light")]
g = d.groupby("method").psnr.agg(["mean", "size"])
g = g[g["size"] >= 2]
for m, r in g.sort_values("mean").iterrows():
    print(f"pilot {m}: {r['mean']:.2f} dB on {int(r['size'])} scenes", flush=True)
print("BEST", g["mean"].idxmax().replace("superf_gnll_tvp", ""))
PY
}

if [ -z "${TV_WEIGHT:-}" ] && [ ! -f "$WFILE" ]; then
    echo "=== [$(date '+%H:%M:%S')] SuperF+TV pilot at x4 on ${PILOT[*]}"
    GRID=(0.1 1 10 100)
    pilot "${GRID[@]}"
    out=$(best_weight) || { echo "!!! pilot scoring failed"; tail -5 results/logs/superf_tv_pilot_eval.log; exit 1; }
    echo "$out"; W=$(echo "$out" | awk '/^BEST/ {print $2}')
    for step in 1 2; do              # best at an edge of the grid: one more decade beyond it
        hi=$(printf '%s\n' "${GRID[@]}" | sort -g | tail -1); lo=$(printf '%s\n' "${GRID[@]}" | sort -g | head -1)
        if [ "$W" = "$hi" ]; then NEW=$(python -c "print(f'{float(\"$W\") * 10:g}')")
        elif [ "$W" = "$lo" ]; then NEW=$(python -c "print(f'{float(\"$W\") / 10:g}')")
        else break; fi
        echo "=== best weight $W is at the edge of the grid; trying $NEW"
        GRID+=("$NEW"); pilot "$NEW"
        out=$(best_weight) || exit 1
        echo "$out"; W=$(echo "$out" | awk '/^BEST/ {print $2}')
    done
    UP=$(python -c "print(f'{float(\"$W\") * 3:g}')"); DN=$(python -c "print(f'{float(\"$W\") / 3:g}')")
    echo "=== refining around $W: $DN and $UP"
    pilot "$DN" "$UP"
    out=$(best_weight) || exit 1
    echo "$out"; W=$(echo "$out" | awk '/^BEST/ {print $2}')
    echo "$W" > "$WFILE"
    echo "=== SuperF+TV weight picked: $W"
fi
W=${TV_WEIGHT:-$(cat "$WFILE")}
echo "=== [$(date '+%H:%M:%S')] SuperF GNLL + TV (weight $W) x$F, all scenes -> results/$TAG (${MFSR_DEVICE:-cpu})"
python jobs/pool.py --workers "${WORKERS:-4}" --threads "$TH" --data_root "$D" --df "$F" --aug light --shift 1.0 \
    --logdir "results/logs/${TAG}_light" -- python runners/run_superf_tv.py --variant superf_gnll --tv "$W" \
    --tag "$TAG" "${common[@]}" --df "{df}" --scenes "{scene}"
echo "=== [$(date '+%H:%M:%S')] done"
