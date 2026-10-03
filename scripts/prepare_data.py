#!/usr/bin/env python
"""Download and verify the DORA data release (tasks, source layers, perception checkpoints).

    python scripts/prepare_data.py                      # Hugging Face Hub -> ./data  (~18 GB)
    python scripts/prepare_data.py --no-checkpoints     # skip the perception weights (~12.7 GB): enough to score runs with evaluate.py, not to run the agents
    python scripts/prepare_data.py --source kaggle      # Kaggle mirror (needs `pip install kagglehub`)
    python scripts/prepare_data.py --source /path/to/DORA_v1.0      # local copy / offline mirror
    python scripts/prepare_data.py --gvlm /path/to/GVLM             # GVLM from a local copy instead of downloading it
    python scripts/prepare_data.py --rescuenet /path/to/RescueNet   # RescueNet originals from a local copy
    python scripts/prepare_data.py --verify-only        # re-check an existing data directory

Every file is checked against the release manifest (size + SHA-256). The data directory can
live anywhere; point the code to it with ``export DORA_DATA=/path/to/data``.

Two sources are not redistributed and are fetched from their official releases instead: the GVLM
scenes (the official archive, 1.05 GB, is downloaded once into the cache directory and the scenes
are copied as released) and the ten downsampled RescueNet images (rebuilt from the originals, read
from the authors' figshare record with HTTP range requests). ``--gvlm`` / ``--rescuenet`` use local
copies instead.
"""
import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

HF_REPO = "Junjue-Wang/DORA"              # Hugging Face dataset repo
HF_REVISION = "v1.0"                       # release tag
KAGGLE_HANDLE = "doradataset/dora-benchmark"
MANIFEST = "manifest.json"
DOCS = ["README.md", "LICENSE.md"]
DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data"
GVLM_URL = "https://github.com/zxk688/GVLM"
GVLM_ARCHIVE = "GVLM_CD.7z"   # official release, Google Drive link in the GVLM README (1.05 GB)
GVLM_ARCHIVE_URL = ("https://drive.usercontent.google.com/download?id=1R6U5GmBHVDi9g3XM09jYCnaqWSwEpBj-"
                    "&export=download&confirm=t")
RESCUENET_DOI = "https://doi.org/10.6084/m9.figshare.c.6647354.v1"
RESCUENET_ZIP = "https://ndownloader.figshare.com/files/40581458"   # RescueNet segmentation train set (18.7 GB)
RESCUENET_MEMBER = "train-org-img/{}.jpg"
RESCUENET_SIZE = (896, 672)


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download_hf(data_dir: Path, parts):
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=HF_REPO, repo_type="dataset", revision=HF_REVISION, local_dir=data_dir,
                      allow_patterns=[MANIFEST, *DOCS, *(f"{part}/*" for part in parts)], max_workers=8)


def download_kaggle(data_dir: Path, parts):
    try:
        import kagglehub
    except ImportError:
        sys.exit("kagglehub is not installed: pip install kagglehub")
    copy_tree(Path(kagglehub.dataset_download(KAGGLE_HANDLE)), data_dir, parts)


def copy_tree(src: Path, data_dir: Path, parts):
    """Copy the selected parts of a local release copy into ``data_dir`` (skips files already there)."""
    files = [src / f for f in [MANIFEST, *DOCS] if (src / f).is_file()]
    for part in parts:
        files += [p for p in (src / part).rglob("*") if p.is_file()]
    for f in files:
        dst = data_dir / f.relative_to(src)
        if dst.exists() and dst.stat().st_size == f.stat().st_size:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)


def import_gvlm(gvlm_dir: Path, data_dir: Path, manifest):
    """Copy the GVLM scenes DORA uses (``<region>/im1.png, im2.png``) from a local copy of the official release."""
    for rel, meta in manifest["files"].items():
        if meta.get("license", "") != "restricted:gvlm":
            continue
        region, name = Path(rel).parts[-2:]
        match = next((p for p in gvlm_dir.rglob(name) if p.parent.name == region), None)
        if match is None:
            print(f"  GVLM: {region}/{name} not found under {gvlm_dir}")
            continue
        dst = data_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(match, dst)


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file; every read is one HTTP Range request.

    figshare redirects to S3 URLs that are signed for GET and expire within seconds, so every
    request goes through the stable download URL again."""

    def __init__(self, url):
        import requests

        self.session, self.url, self.pos = requests.Session(), url, 0
        r = self.session.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
        r.raise_for_status()
        self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, b):
        end = min(self.size, self.pos + len(b))
        if end <= self.pos:
            return 0
        r = self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{end - 1}"}, timeout=300)
        r.raise_for_status()
        data = r.content
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def build_rescuenet(data_dir: Path, manifest, local_dir: Path = None):
    """Rebuild DORA's downsampled RescueNet images from the originals, as DORA made them: OpenCV
    bicubic resize to 896x672, one JPEG round trip at OpenCV's default quality, saved as PNG."""
    todo = [rel for rel, meta in manifest["files"].items()
            if meta.get("license", "") == "restricted:rescuenet" and not (data_dir / rel).is_file()]
    if not todo:
        return
    import cv2
    import numpy as np
    from PIL import Image

    archive = None
    if local_dir is None:
        print(f"RescueNet: reading {len(todo)} originals from figshare ({RESCUENET_DOI})")
        archive = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(RESCUENET_ZIP), buffer_size=1 << 20))
    for rel in todo:
        stem = Path(rel).stem
        if archive is not None:
            data = archive.read(RESCUENET_MEMBER.format(stem))
        else:
            match = next(iter(local_dir.rglob(f"{stem}.jpg")), None)
            if match is None:
                print(f"  RescueNet: {stem}.jpg not found under {local_dir}")
                continue
            data = match.read_bytes()
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        small = cv2.resize(image, RESCUENET_SIZE, interpolation=cv2.INTER_CUBIC)
        _, jpeg = cv2.imencode(".jpg", small)
        rgb = cv2.cvtColor(cv2.imdecode(jpeg, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        dst = data_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(dst, "PNG")


def download_ranged(url: str, dst: Path, workers: int = 8) -> Path:
    """Download ``url`` to ``dst`` in ``workers`` parallel byte ranges; interrupted parts resume."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import requests

    r = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
    r.raise_for_status()
    if r.status_code != 206 or "Content-Range" not in r.headers:
        raise RuntimeError(f"{url} does not serve byte ranges (status {r.status_code}, "
                           f"{r.headers.get('Content-Type')}); the host may be refusing the download")
    size = int(r.headers["Content-Range"].rsplit("/", 1)[1])
    if dst.is_file() and dst.stat().st_size == size:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    step = -(-size // workers)
    done, lock = [0], threading.Lock()

    def part_path(k):
        return dst.with_name(f"{dst.name}.part{k}")

    def fetch(k):
        start, end = k * step, min(size, (k + 1) * step) - 1
        part = part_path(k)
        have = part.stat().st_size if part.exists() else 0
        with lock:
            done[0] += have
        if start + have > end:
            return
        with requests.get(url, headers={"Range": f"bytes={start + have}-{end}"}, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            if resp.status_code != 206:
                raise RuntimeError(f"range request refused (status {resp.status_code})")
            with open(part, "ab") as f:
                for block in resp.iter_content(1 << 20):
                    f.write(block)
                    with lock:
                        done[0] += len(block)
                        print(f"\r  {dst.name}: {done[0] / 1e9:.2f}/{size / 1e9:.2f} GB", end="", flush=True)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(fetch, range(workers)))
    print()
    tmp = dst.with_name(dst.name + ".tmp")
    with open(tmp, "wb") as out:
        for k in range(workers):
            with open(part_path(k), "rb") as f:
                shutil.copyfileobj(f, out, 1 << 24)
    if tmp.stat().st_size != size:
        raise RuntimeError(f"{dst.name}: got {tmp.stat().st_size} bytes, expected {size}")
    os.replace(tmp, dst)
    for k in range(workers):
        part_path(k).unlink()
    return dst


def fetch_gvlm(data_dir: Path, manifest, cache_dir: Path):
    """Download the official GVLM archive and import the scenes DORA uses (the files are kept as released)."""
    if all((data_dir / rel).is_file() for rel, meta in manifest["files"].items() if meta.get("license", "") == "restricted:gvlm"):
        return
    import py7zr

    print(f"GVLM: downloading the official release ({GVLM_URL}) to {cache_dir}")
    archive = download_ranged(GVLM_ARCHIVE_URL, cache_dir / GVLM_ARCHIVE)
    wanted = {Path(rel).parts[-2:] for rel, meta in manifest["files"].items() if meta.get("license", "") == "restricted:gvlm"}
    out = cache_dir / "GVLM"
    with py7zr.SevenZipFile(archive) as z:
        targets = [n for n in z.getnames() if tuple(n.replace("\\", "/").split("/")[-2:]) in wanted]
        z.extract(path=out, targets=targets)
    import_gvlm(out, data_dir, manifest)


def verify(data_dir: Path, manifest, parts, check_hash: bool) -> bool:
    files = {k: v for k, v in manifest["files"].items() if k.split("/", 1)[0] in parts}
    bad, absent_restricted = [], {}
    for i, (rel, meta) in enumerate(sorted(files.items()), 1):
        path = data_dir / rel
        restricted = meta.get("license", "").startswith("restricted:")
        if not path.is_file():
            if restricted:
                absent_restricted.setdefault(meta.get("license", ""), []).append(rel)
            else:
                bad.append((rel, "missing"))
        elif path.stat().st_size != meta["bytes"]:
            bad.append((rel, "size mismatch"))
        elif check_hash and sha256(path) != meta["sha256"]:
            bad.append((rel, "sha256 mismatch"))
        if i % 100 == 0 or i == len(files):
            print(f"\r  verified {i}/{len(files)} files", end="", flush=True)
    print()
    for rel, why in bad[:20]:
        print(f"  [{why}] {rel}")
    total = sum(v["bytes"] for v in files.values())
    n_absent = sum(len(v) for v in absent_restricted.values())
    print(f"{len(files) - len(bad) - n_absent}/{len(files)} files OK ({total / 1e9:.2f} GB) in {data_dir}")
    if absent_restricted:
        print(f"\nNot redistributed (tasks using these layers cannot be solved without them, "
              f"see {data_dir / 'LICENSE.md'}):")
        hints = {"restricted:gvlm": f" -- re-run with network access, or download GVLM ({GVLM_URL}, e.g. its "
                                    "Baidu link) and re-run with --gvlm <dir>",
                 "restricted:rescuenet": f" -- re-run with network access, or download RescueNet ({RESCUENET_DOI}) "
                                         "and re-run with --rescuenet <dir>"}
        for kind, rels in sorted(absent_restricted.items()):
            print(f"  {kind}: {len(rels)} files{hints.get(kind, '')}")
    return not bad


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path(os.environ.get("DORA_DATA", DEFAULT_DATA)))
    p.add_argument("--source", default="hf", help="'hf' (default), 'kaggle', or a local release directory")
    p.add_argument("--no-checkpoints", action="store_true", help="skip the ~12.7 GB of perception weights")
    p.add_argument("--gvlm", type=Path, default=None,
                   help="local copy of the official GVLM dataset (default: download its official archive)")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="where downloaded source archives are kept (default: <data-dir>/.cache)")
    p.add_argument("--rescuenet", type=Path, default=None,
                   help="local copy of the RescueNet originals (default: read them from figshare)")
    p.add_argument("--no-hash", action="store_true", help="verify file sizes only (faster)")
    p.add_argument("--verify-only", action="store_true", help="do not download, only verify")
    args = p.parse_args()

    parts = ["tasks", "images"] + ([] if args.no_checkpoints else ["checkpoints"])
    data_dir = args.data_dir.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    if not args.verify_only:
        print(f"Downloading DORA ({', '.join(parts)}) from {args.source} to {data_dir}")
        if args.source == "hf":
            download_hf(data_dir, parts)
        elif args.source == "kaggle":
            download_kaggle(data_dir, parts)
        else:
            copy_tree(Path(args.source).expanduser().resolve(), data_dir, parts)

    manifest = json.loads((data_dir / MANIFEST).read_text(encoding="utf-8"))
    if args.gvlm is not None:
        import_gvlm(args.gvlm.expanduser().resolve(), data_dir, manifest)
    if "images" in parts and not args.verify_only:
        try:
            fetch_gvlm(data_dir, manifest, (args.cache_dir or data_dir / ".cache").expanduser().resolve())
        except Exception as e:  # e.g. Google Drive download quota: report it, verification lists what is missing
            print(f"  GVLM: could not download the official archive ({e})")
        try:
            build_rescuenet(data_dir, manifest, args.rescuenet.expanduser().resolve() if args.rescuenet else None)
        except Exception as e:  # network or archive problem: report it, verification lists what is missing
            print(f"  RescueNet: could not rebuild the images ({e})")

    ok = verify(data_dir, manifest, parts, check_hash=not args.no_hash)
    if ok:
        print(f"\nDORA data is ready. The release combines several public sources; see {data_dir / 'LICENSE.md'}\n"
              "for their licenses and required attributions (non-commercial research use).")
        if data_dir != DEFAULT_DATA.resolve():
            print(f"Point the code to it with:  export DORA_DATA={data_dir}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
