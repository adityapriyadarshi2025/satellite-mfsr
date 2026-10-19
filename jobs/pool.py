#!/usr/bin/env python3
"""Run one command per burst, several at a time, on a CPU node.

    python jobs/pool.py --workers 4 --threads 8 --data_root data/satsynthburst \
        --df 4 --aug light --shift 1.0 --logdir results/logs/superf_mse -- \
        python runners/run_superf.py --variant superf_mse --data_root data/satsynthburst \
            --results results --aug light --shift 1.0 --skip_existing --device cpu --df {df} --scenes {scene}

{scene} and {df} are filled in per burst. Each burst logs to <logdir>/<scene>_x<df>.log.
One failed burst doesn't stop the rest; the summary at the end lists failures.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.bursts import discover  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8, help="OMP/MKL threads per burst")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--df", type=float, nargs="+", default=[4])
    ap.add_argument("--aug", default=None)
    ap.add_argument("--shift", default="1.0")
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--stride", type=int, default=1, help="only every k-th burst in sorted order (a spread-out subset)")
    ap.add_argument("--max_bursts", type=int, default=0, help="stop after the first N bursts (0 = all)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        sys.exit("no command given after --")

    refs = [r for df in a.df for r in discover(a.data_root, df=df, shift=a.shift, aug=a.aug)]
    if not refs:
        sys.exit(f"no bursts for df={a.df} aug={a.aug} shift={a.shift} under {a.data_root}")
    refs = refs[::max(1, a.stride)]
    if a.max_bursts > 0:
        refs = refs[:a.max_bursts]
    logdir = Path(a.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads))

    todo = list(refs)
    running, failed, done = {}, [], 0
    t0 = time.time()
    print(f"[pool] {len(todo)} bursts, {a.workers} workers x {a.threads} threads, logs in {logdir}", flush=True)
    while todo or running:
        while todo and len(running) < a.workers:
            r = todo.pop(0)
            argv = [c.replace("{scene}", r.scene).replace("{df}", str(r.df)) for c in cmd]
            log = open(logdir / f"{r.scene}_x{r.df}.log", "w")
            running[subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env)] = (r, log, time.time())
        time.sleep(2)
        for p in [p for p in running if p.poll() is not None]:
            r, log, ts = running.pop(p)
            log.close()
            done += 1
            status = "ok" if p.returncode == 0 else f"FAILED rc={p.returncode}"
            if p.returncode != 0:
                failed.append((r, log.name))
            print(f"[pool] {done}/{len(refs)} {r.scene} x{r.df} {status} ({time.time() - ts:.0f}s, "
                  f"total {(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"[pool] finished: {len(refs) - len(failed)} ok, {len(failed)} failed, {(time.time() - t0) / 60:.1f} min")
    for r, name in failed[:3]:
        print(f"[pool] last lines of {name}:")
        print("".join(open(name).readlines()[-12:]))


if __name__ == "__main__":
    main()
