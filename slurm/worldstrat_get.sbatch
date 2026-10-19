#!/usr/bin/env bash
#SBATCH --job-name=mfsr-wsget
#SBATCH -p cpup
#SBATCH -N 1 -n 1 -c 2
#SBATCH -t 08:00:00
#SBATCH -o slurm_logs/%x-%j.out
# WorldStrat (Zenodo record 6810792): the Sentinel-2 L2A revisits and the SPOT HR images, then only the AOIs in
# setup/worldstrat_aois.txt extracted to data/worldstrat/aoi (see setup/get_worldstrat.py). Needs internet, so it
# runs on a compute node. Resumable: rerun it to continue an interrupted download.
#   cd ~/aditya/mfsr && sbatch -A kosh2-aidrive-acc slurm/worldstrat_get.sbatch
# Done when the log ends with "WORLDSTRAT OK".
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1
source slurm/config.sh
set +u; activate_env || exit 1; set -u
echo "=== [$(date '+%H:%M:%S')] WorldStrat on $(hostname)"
python setup/get_worldstrat.py --out "$DATA/worldstrat" --aois setup/worldstrat_aois.txt
echo "=== [$(date '+%H:%M:%S')] done"
