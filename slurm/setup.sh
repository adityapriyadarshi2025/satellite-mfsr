#!/usr/bin/env bash
# One-time setup, safe to rerun (every step skips what is already done).
# Needs internet. On a cluster where only compute nodes have it, run it inside an interactive job:
#   srun -p cpup -N1 -n1 -c 8 -t 01:00:00 --pty bash   (add -A ACCOUNT if you have several)
#
#   bash slurm/setup.sh                 # env repos data check
#   bash slurm/setup.sh data check      # only some steps
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"
STEPS="${*:-env repos data check}"
has() { [[ " $STEPS " == *" $1 "* ]]; }
cd "$PROJ" || exit 1
echo "setup: PROJ=$PROJ  host=$(hostname)  steps: $STEPS"

# if the HTTP proxy fails, retry without the proxy variables (compute nodes often have direct internet)
noproxy() { env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY "$@"; }
needs_net=0; for s in env repos data gsasr; do has "$s" && needs_net=1; done
if [ "$needs_net" = 1 ] && ! curl -s -m 20 -o /dev/null https://github.com && ! noproxy curl -s -m 20 -o /dev/null https://github.com; then
    echo "!! no internet on $(hostname). This must run on a compute node, not the login node." >&2
    exit 1
fi

if has env; then
    echo "== env: uv venv at $VENV (Python 3.12, torch 2.6.0 cu124)"
    UV="$(command -v uv || true)"
    [ -z "$UV" ] && [ -x "$HOME/.local/bin/uv" ] && UV="$HOME/.local/bin/uv"
    if [ -z "$UV" ]; then echo "!! uv not found (expected ~/.local/bin/uv)" >&2; exit 1; fi
    [ -x "$VENV/bin/python" ] || "$UV" venv --python 3.12 --seed "$VENV" || exit 1
    set +u; source "$VENV/bin/activate"; set -u
    if ! python -c "import torch" 2>/dev/null; then
        pip install -q torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 \
          || noproxy pip install -q torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 \
          || exit 1
    fi
    pip install -q -r "$PROJ/requirements.txt" || noproxy pip install -q -r "$PROJ/requirements.txt" || exit 1
    python -c "import torch, torchmetrics, lpips, cv2, einops, tifffile, pandas; print('  torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '| threads:', torch.get_num_threads())" || exit 1
fi

if has repos; then
    echo "== repos -> $BASELINES (pinned commits)"
    mkdir -p "$BASELINES"
    clone() {  # name url commit
        local d="$BASELINES/$1"
        if [ ! -d "$d/.git" ]; then
            git clone -q "$2" "$d" 2>/dev/null \
              || { rm -rf "$d"; noproxy git clone -q "$2" "$d"; } \
              || { echo "!! clone failed: $1"; return 1; }
        fi
        git -C "$d" -c advice.detachedHead=false checkout -q "$3" && echo "  $1 @ $(git -C "$d" rev-parse --short HEAD)"
    }
    clone superf                https://github.com/Sjyhne/superf.git                 dff884c
    clone s2-superres           https://github.com/calebrob6/s2-superres.git         6543e25
    clone gsasr                 https://github.com/ChrisDud0257/GSASR.git            9d2eb64
fi

if has gsasr; then
    echo "== gsasr: the paper's weights (EDSR, RDN; trained on DIV2K) from Hugging Face -> $BASELINES/gsasr_weights"
    for m in EDSR RDN; do
        d="$BASELINES/gsasr_weights/$m"; mkdir -p "$d"
        for f in encoder decoder; do
            [ -s "$d/$f.pth" ] && continue
            u="https://huggingface.co/mutou0308/GSASR/resolve/main/GSASR_paper/$m/$f.pth"
            if curl -sSfL --retry 3 -o "$d/$f.pth.part" "$u" || noproxy curl -sSfL --retry 3 -o "$d/$f.pth.part" "$u"; then
                mv "$d/$f.pth.part" "$d/$f.pth"
            else
                echo "!! download failed: $m/$f.pth"
            fi
        done
        ls -l "$d"
    done
fi

if has data; then
    echo "== data: SatSynthBurst -> $DATA/satsynthburst"
    set +u; activate_env || exit 1; set -u
    python "$PROJ/setup/get_satsynthburst.py" "$DATA" || echo "!! SatSynthBurst download failed (see above)"
fi

if has check; then
    echo "== check"
    set +u; activate_env || exit 1; set -u
    python "$PROJ/jobs/variants.py" --data_root "$DATA/satsynthburst" report || true
fi
echo "setup done"
