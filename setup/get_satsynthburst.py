#!/usr/bin/env python3
"""Download SatSynthBurst (SuperF, ICLR 2026; Zenodo record 19593026, CC BY-NC 4.0),
check md5 against Zenodo's own checksum, unzip into <data>/satsynthburst.

    python setup/get_satsynthburst.py data

stdlib only. Tries the normal network path first, then again with the proxy
variables ignored (for clusters whose compute nodes have direct internet).
"""
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

REC = "19593026"
# from the Zenodo record page; used only if the API is unreachable
FALLBACK = [{"key": "data.zip", "checksum": "md5:ee6243387768b1077206de375a37c4d4"}]


def opener(direct: bool):
    return urllib.request.build_opener(urllib.request.ProxyHandler({})) if direct else urllib.request.build_opener()


def get(url, direct, timeout=120):
    return opener(direct).open(url, timeout=timeout)


def normalise(files):
    """Zenodo has returned files as a list, or as {"entries": {...} | [...]}; accept both."""
    if isinstance(files, dict):
        files = files.get("entries", files)
        if isinstance(files, dict):
            files = list(files.values())
    out = []
    for f in files or []:
        if isinstance(f, dict) and "key" in f:
            out.append({"key": f["key"], "checksum": f.get("checksum", ""), "size": f.get("size", 0)})
    return out


def record_files():
    for direct in (False, True):
        for url in (f"https://zenodo.org/api/records/{REC}", f"https://zenodo.org/api/records/{REC}/files"):
            try:
                with get(url, direct, 60) as r:
                    rec = json.load(r)
                files = normalise(rec.get("files", rec.get("entries")))
                if files:
                    print("  record:", rec.get("metadata", {}).get("title", "?"), "| files:", [f["key"] for f in files])
                    return files, direct
            except Exception as e:  # noqa: BLE001
                print(f"  {url} {'(no proxy) ' if direct else ''}failed: {e}")
    print("  using the recorded file list (data.zip)")
    return FALLBACK, True


def download(key, out: Path, direct):
    url = f"https://zenodo.org/records/{REC}/files/{key}?download=1"
    part = out.with_name(out.name + ".part")
    for d in (direct, not direct):
        try:
            with get(url, d) as r, open(part, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            part.rename(out)
            return
        except Exception as e:  # noqa: BLE001
            print(f"  download {'(no proxy) ' if d else ''}failed: {e}")
    raise SystemExit(f"!! could not download {key}: download it manually and copy it to {out}")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    dest, dl = root / "satsynthburst", root / "downloads"
    dest.mkdir(parents=True, exist_ok=True)
    dl.mkdir(parents=True, exist_ok=True)
    files, direct = record_files()
    for f in files:
        key, chk = f["key"], f.get("checksum", "")
        out = dl / key
        if not out.exists():
            print(f"  downloading {key} ({f.get('size', 0) / 1e6:.1f} MB)")
            download(key, out, direct)
        h = md5(out)
        ok = not chk.startswith("md5:") or h == chk[4:]
        print(f"  {key}: md5 {h} {'OK' if ok else 'MISMATCH, expected ' + chk}")
        if not ok:
            raise SystemExit("!! md5 mismatch: delete the file and rerun")
        if key.lower().endswith(".zip"):
            marker = dest / f".unzipped_{key}"
            if not marker.exists():
                with zipfile.ZipFile(out) as z:
                    z.extractall(dest)
                marker.touch()
                print(f"  unzipped {key} -> {dest}")
    print("  done")


if __name__ == "__main__":
    main()
