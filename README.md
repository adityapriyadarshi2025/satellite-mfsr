# satellite-mfsr

Multi-frame super-resolution of satellite image bursts with per-scene 2D Gaussian splatting.

Aditya Priyadarshi, IIT Gandhinagar

Each burst is fitted on its own, without training data. The scene is a set of anisotropic 2D Gaussians, and a differentiable sensor model renders every low-resolution frame from it: sub-pixel shift and rotation, an analytic Gaussian PSF, integration over each LR pixel and a per-frame colour transform. Frames are registered by exhaustive NCC search and refined jointly with the scene, noisy frames are down-weighted (IRLS), and a total-variation prior regularises the HR image. The rasteriser is pure PyTorch with a hand-written backward pass. Because the output is a render, any scale works, including non-integer ones.

## Results

SatSynthBurst `light` (20 scenes, 16 frames). PSNR in dB with DBSR colour matching and a 16 px border crop.

| Method | ×2 | ×4 | ×8 |
|---|---|---|---|
| Bicubic | 37.55 | 30.64 | 27.09 |
| SuperF (MSE) | 36.99 | 33.63 | 29.24 |
| SuperF (GNLL) | 37.65 | 34.19 | 29.82 |
| Ours | **40.47** | **35.57** | **30.55** |

The SuperF rows are our runs of the released code and are above the numbers reported in the paper. Ours is better than SuperF (GNLL) on all 20 scenes at every factor, and adding the same TV prior to SuperF changes it by only +0.02 dB at ×4.

## Setup

Linux, Python 3.12 (set up with `uv`), PyTorch 2.6. The script creates `.venv`, clones the baselines into `baselines/` at pinned commits and downloads SatSynthBurst into `data/`.

```bash
bash slurm/setup.sh
```

## Usage

```bash
source .venv/bin/activate

# ours (the configuration in the table) and a baseline, at x4
python runners/run_gsmisr.py --df 4 --aug light --iters 600 --tv 1e-3 --tag gsmisr_tv
python runners/run_superf.py --variant superf_gnll --df 4 --aug light

# score every prediction in results/ and write the table
python eval/evaluate.py
python eval/make_table.py --aug light
```

On a SLURM cluster, `slurm/stage.sbatch` runs one method over all bursts and `slurm/report.sbatch` scores everything. Set the partition in the `#SBATCH -p` lines and, if needed, an account in an untracked `slurm/site.sh` (`export SBATCH_ACCOUNT=<account> SLURM_ACCOUNT=<account>`).

```bash
mkdir -p slurm_logs
GS_ARGS="--iters 600 --tv 1e-3" sbatch slurm/stage.sbatch gsmisr_tv 4 light
sbatch slurm/stage.sbatch superf_gnll 4 light
sbatch slurm/report.sbatch
```

Other experiments: `jobs/superf_tv.sh` (SuperF with the TV prior), `jobs/ablation.sh`, `jobs/frames_sweep.sh` and `jobs/superf_frames.sh` (number of frames), `slurm/arbscale.sbatch` (non-integer scales).

## Tests

```bash
python gsmisr/tests/test_render.py
python eval/tests/test_colormatch.py
```

## Third-party

The baselines are not included; `slurm/setup.sh` clones SuperF and s2-superres at pinned commits. `eval/metrics.py` contains a numpy port of DBSR's colour matching. SatSynthBurst (SuperF, ICLR 2026) is downloaded from Zenodo (CC BY-NC 4.0).
