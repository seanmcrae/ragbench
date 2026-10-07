"""Download BEIR SciFact for optional larger runs (never needed by the tests).

Source:  https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
         (BEIR benchmark, Thakur et al., NeurIPS 2021 Datasets and Benchmarks)
Dataset: SciFact, Wadden et al., EMNLP 2020 -- https://github.com/allenai/scifact
License: CC BY-NC 2.0 (non-commercial). By downloading you agree to the SciFact terms.

SciFact ships claims, abstracts and relevance judgements but no reference answers, so answer
F1 / exact match are skipped and only retrieval, groundedness, latency and cost are scored.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
MD5 = "5f7d1de60b170fc8027bb7898e2efca1"  # published in the BEIR README
DEFAULT_DEST = Path(__file__).resolve().parent.parent / "data" / "downloads"


def md5sum(path: Path) -> str:
    digest = hashlib.md5()  # integrity check against the published checksum, not security
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()
    target = args.dest / "scifact"
    if (target / "corpus.jsonl").exists():
        print(f"already present: {target}")
        return
    args.dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "scifact.zip"
        print(f"downloading {URL}")
        with urllib.request.urlopen(URL, timeout=120) as response, archive.open("wb") as out:
            shutil.copyfileobj(response, out)
        if (actual := md5sum(archive)) != MD5:
            raise SystemExit(f"checksum mismatch: expected {MD5}, got {actual}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(args.dest)
    print(f"extracted to {target}; run with configs/scifact.yaml")


if __name__ == "__main__":
    main()
