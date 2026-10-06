# shellcheck shell=bash
# Sourced by every job script (stage, report, setup, ...). Everything stays inside the repo folder.
# Paths come from where this file is, not from the environment: sbatch copies the submitting shell's
# environment into every job, so a PROJ, DATA or VENV set in that shell must not override them.
# Do not `source` this file in a login shell.
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJ
export VENV="$PROJ/.venv"
export DATA="$PROJ/data"
export BASELINES="$PROJ/baselines"
# site settings kept out of git (e.g. the SLURM account): slurm/site.sh, see README
# shellcheck disable=SC1091
if [ -f "$PROJ/slurm/site.sh" ]; then source "$PROJ/slurm/site.sh"; fi
export PIP_USER=0   # a global pip "user = true" setting would break venv installs
# keep caches inside the project folder, not in ~/.cache
export PIP_CACHE_DIR="$PROJ/.cache/pip" TORCH_HOME="$PROJ/.cache/torch" MPLCONFIGDIR="$PROJ/.cache/matplotlib"

activate_env() {
    if [ ! -f "$VENV/bin/activate" ]; then
        echo "activate_env: no venv at $VENV - run 'bash slurm/setup.sh' on a compute node first" >&2
        return 1
    fi
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
}
