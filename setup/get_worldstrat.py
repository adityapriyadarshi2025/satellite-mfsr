#!/usr/bin/env python3
"""WorldStrat (Cornebise et al., NeurIPS 2022 Datasets & Benchmarks; Zenodo record 6810792): download the
Sentinel-2 L2A revisits and the SPOT HR images, then extract only the AOIs we use.

    python setup/get_worldstrat.py --out data/worldstrat --aois setup/worldstrat_aois.txt
    python3 setup/get_worldstrat.py --out data/worldstrat --connections 16     (parallel, e.g. on the Mac)
    python setup/get_worldstrat.py --list-only            (sizes and checksums of every file, no download)

Zenodo serves each connection slowly (~1 MB/s) but allows many at once, so --connections N splits each tarball
into 64 MB byte ranges fetched N at a time (stdlib only, Python >= 3.8). Without it, one curl stream is used.
On the cluster it runs inside a job (compute nodes have internet, the login node doesn't):
slurm/worldstrat_get.sbatch. Resumable: an interrupted download continues where it stopped (finished ranges are
kept in <tarball>.ranges.json; a sequential .part from the curl path is reused), finished files are checked
against Zenodo's md5 and skipped on a rerun, and extraction skips files that already exist.

Layout:
    <out>/raw/<tarball>                the downloads (kept, so more AOIs can be extracted later without a re-download)
    <out>/aoi/<path inside the tar>    the extracted files of the selected AOIs
    <out>/metadata.csv, <out>/stratified_train_val_test_split.csv
    <out>/aoi/EXTRACTED.txt            members extracted per AOI and tarball (an AOI with 0 files is flagged)
A member belongs to an AOI when one of the folders in its path has exactly the AOI's name (no prefix matching,
so "Landcover-777318" never picks up "Landcover-7773180").
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

RECORD = "6810792"
WANT = ["metadata.csv", "stratified_train_val_test_split.csv", "lr_dataset_l2a.tar.gz", "hr_dataset.tar.gz"]
PROXY_VARS = ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]


def log(*a):
    print(f"[worldstrat {time.strftime('%H:%M:%S')}]", *a, flush=True)


def fetch_json(url):
    last = None
    for use_proxy in (True, False):           # compute nodes have direct internet; the cluster proxy is unreliable
        handler = urllib.request.ProxyHandler() if use_proxy else urllib.request.ProxyHandler({})
        try:
            with urllib.request.build_opener(handler).open(url, timeout=60) as r:
                return json.load(r)
        except Exception as e:                    # noqa: BLE001
            last = e
    raise SystemExit(f"cannot reach {url}: {last}")


def md5sum(path, block=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest: Path, size: int, md5: str | None, tries=30):
    if dest.exists() and dest.stat().st_size == size and (md5 is None or md5sum(dest) == md5):
        log(f"have {dest.name} ({size / 1e9:.2f} GB, md5 ok)")
        return
    part = dest.with_suffix(dest.suffix + ".part")
    env_direct = {k: v for k, v in os.environ.items() if k not in PROXY_VARS}
    for i in range(tries):
        have = part.stat().st_size if part.exists() else 0
        if have >= size:
            break
        log(f"downloading {dest.name}: {have / 1e9:.2f} of {size / 1e9:.2f} GB (try {i + 1})")
        cmd = ["curl", "-sS", "-L", "--fail", "-C", "-", "--retry", "5", "--retry-delay", "20",
               "--speed-limit", "100000", "--speed-time", "300", "-o", str(part), url]
        rc = subprocess.call(cmd, env=env_direct if i % 2 == 0 else os.environ.copy())
        if rc not in (0, 18, 28, 56, 92):
            log(f"curl exited {rc}")
        time.sleep(5 if rc else 0)
    if not part.exists() or part.stat().st_size != size:
        raise SystemExit(f"!!! {dest.name}: download incomplete ({part.stat().st_size if part.exists() else 0} of "
                         f"{size} bytes); rerun the job to continue")
    if md5:
        got = md5sum(part)
        if got != md5:
            part.rename(part.with_suffix(".bad"))
            raise SystemExit(f"!!! {dest.name}: md5 {got} != {md5}; moved to .bad, rerun to download again")
    part.rename(dest)
    log(f"done {dest.name}")


CHUNK = 64 << 20


def _fetch_range(url, fd, start, end, tries=12):
    """Bytes [start, end] of url written at the same offset of fd. True when every byte arrived."""
    want = end - start + 1
    for i in range(tries):
        handler = urllib.request.ProxyHandler() if i % 2 == 0 else urllib.request.ProxyHandler({})
        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "mfsr-get-worldstrat/1.0"})
        got = 0
        try:
            with urllib.request.build_opener(handler).open(req, timeout=60) as r:
                if r.status != 206:
                    raise RuntimeError(f"server ignored the byte range (HTTP {r.status})")
                while got < want:
                    buf = r.read(min(1 << 20, want - got))
                    if not buf:
                        break
                    os.pwrite(fd, buf, start + got)
                    got += len(buf)
            if got == want:
                return True
        except RuntimeError:
            raise
        except Exception:                         # noqa: BLE001  (timeouts, resets, DNS hiccups: retry)
            pass
        time.sleep(min(60, 5 * (i + 1)))
    return False


def download_parallel(url, dest: Path, size: int, md5: str | None, conns: int):
    if dest.exists() and dest.stat().st_size == size and (md5 is None or md5sum(dest) == md5):
        log(f"have {dest.name} ({size / 1e9:.2f} GB, md5 ok)")
        return
    part = dest.with_suffix(dest.suffix + ".part")
    state = dest.with_suffix(dest.suffix + ".ranges.json")
    n = (size + CHUNK - 1) // CHUNK
    done = set()
    if state.exists():
        done = set(json.loads(state.read_text()))
    elif part.exists():                           # a sequential .part from the curl path: its prefix is valid
        done = set(range(part.stat().st_size // CHUNK))
    if not part.exists():
        part.touch()
    if part.stat().st_size != size:
        os.truncate(part, size)                   # sparse on Linux/macOS; ranges are written in place
    todo = [i for i in range(n) if i not in done]
    log(f"downloading {dest.name}: {len(done)} of {n} ranges already done, {len(todo)} to go, {conns} connections")
    fd = os.open(part, os.O_RDWR)
    lock = threading.Lock()
    t0, got0, failed = time.time(), len(done), []
    try:
        with ThreadPoolExecutor(max_workers=conns) as pool:
            futs = {pool.submit(_fetch_range, url, fd, i * CHUNK, min(size, (i + 1) * CHUNK) - 1): i for i in todo}
            for k, fut in enumerate(as_completed(futs), 1):
                i = futs[fut]
                if fut.result():
                    with lock:
                        done.add(i)
                        if k % 8 == 0 or k == len(todo):
                            state.write_text(json.dumps(sorted(done)))
                else:
                    failed.append(i)
                if k % 16 == 0 or k == len(todo):
                    mbs = (len(done) - got0) * CHUNK / max(1e-6, time.time() - t0) / 1e6
                    log(f"  {dest.name}: {min(size, len(done) * CHUNK) / 1e9:.2f} of {size / 1e9:.2f} GB "
                        f"({mbs:.1f} MB/s)")
    finally:
        os.close(fd)
        state.write_text(json.dumps(sorted(done)))
    if failed or len(done) != n:
        raise SystemExit(f"!!! {dest.name}: {n - len(done)} of {n} ranges still missing; rerun to continue")
    if md5:
        log(f"checking md5 of {dest.name}")
        got = md5sum(part)
        if got != md5:
            part.rename(part.with_suffix(".bad"))
            state.unlink(missing_ok=True)
            raise SystemExit(f"!!! {dest.name}: md5 {got} != {md5}; moved to .bad, rerun to download again")
    part.rename(dest)
    state.unlink(missing_ok=True)
    log(f"done {dest.name}")


def read_aois(path):
    names = []
    for line in Path(path).read_text().splitlines():
        s = line.split("#", 1)[0].strip()
        if s and s not in names:
            names.append(s)
    return names


def extract(tar_path: Path, aois, dest: Path, counts: Counter):
    aoiset = set(aois)
    shown = 0
    log(f"extracting {len(aoiset)} AOIs from {tar_path.name} (one pass over the whole archive)")
    with tarfile.open(tar_path, mode="r|gz") as tar:
        for m in tar:
            if shown < 12:
                log(f"  member: {m.name}")
                shown += 1
            hit = next((p for p in m.name.split("/")[:-1] if p in aoiset), None)
            if hit is None or not (m.isfile() or m.isdir()):
                continue
            target = dest / m.name
            if m.isfile():
                counts[(hit, tar_path.name)] += 1
                if target.exists() and target.stat().st_size == m.size:
                    continue
            try:
                tar.extract(m, dest, filter="data")
            except TypeError:                     # Python without tar filters (e.g. macOS's system 3.9)
                if m.name.startswith("/") or ".." in Path(m.name).parts:
                    continue
                tar.extract(m, dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/worldstrat")
    ap.add_argument("--aois", default="setup/worldstrat_aois.txt")
    ap.add_argument("--files", nargs="+", default=WANT)
    ap.add_argument("--max_gb", type=float, default=400.0, help="refuse to download more than this")
    ap.add_argument("--connections", type=int, default=1,
                    help="parallel byte-range connections per file (1 = a single curl stream)")
    ap.add_argument("--list-only", action="store_true")
    a = ap.parse_args()

    rec = fetch_json(f"https://zenodo.org/api/records/{RECORD}")
    files = {f["key"]: f for f in rec.get("files", [])}
    log(f"record {RECORD}: {rec.get('metadata', {}).get('title', '?')}")
    for k, f in sorted(files.items()):
        log(f"  {k:45s} {f['size'] / 1e9:8.2f} GB  {f.get('checksum', '')}")
    if a.list_only:
        return
    missing = [k for k in a.files if k not in files]
    if missing:
        raise SystemExit(f"!!! not in the record: {missing}")
    total = sum(files[k]["size"] for k in a.files) / 1e9
    log(f"to download: {', '.join(a.files)} = {total:.1f} GB")
    if total > a.max_gb:
        raise SystemExit(f"!!! {total:.1f} GB is more than --max_gb {a.max_gb}; not downloading")

    out = Path(a.out)
    raw, aoi_dir = out / "raw", out / "aoi"
    raw.mkdir(parents=True, exist_ok=True)
    aoi_dir.mkdir(parents=True, exist_ok=True)
    for k in a.files:
        f = files[k]
        md5 = f.get("checksum", "").split(":", 1)[1] if f.get("checksum", "").startswith("md5:") else None
        url = f.get("links", {}).get("self") or f"https://zenodo.org/records/{RECORD}/files/{k}?download=1"
        dest = (out if k.endswith(".csv") else raw) / k
        if a.connections > 1 and f["size"] > CHUNK:
            download_parallel(url, dest, f["size"], md5, a.connections)
        else:
            download(url, dest, f["size"], md5)

    aois = read_aois(a.aois)
    counts = Counter()
    for k in a.files:
        if k.endswith((".tar.gz", ".tgz")):
            extract(raw / k, aois, aoi_dir, counts)
    tars = [k for k in a.files if k.endswith((".tar.gz", ".tgz"))]
    lines = [f"{'AOI':32s} " + " ".join(f"{t:>24s}" for t in tars)]
    empty = []
    for n in aois:
        c = [counts[(n, t)] for t in tars]
        if 0 in c:
            empty.append(n)
        lines.append(f"{n:32s} " + " ".join(f"{v:24d}" for v in c))
    (aoi_dir / "EXTRACTED.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if empty:
        log(f"!!! {len(empty)} AOIs with no files in at least one tarball: {empty}")
    log(f"WORLDSTRAT OK: {len(aois) - len(empty)} of {len(aois)} AOIs complete under {aoi_dir}")


if __name__ == "__main__":
    sys.exit(main())
