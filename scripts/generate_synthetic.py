"""Regenerate the bundled SYNTHETIC help-center dataset under data/synthetic_helpcenter/."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_eval.datasets.base import save_beir_dir
from rag_eval.datasets.synthetic import DEFAULT_SEED, build_synthetic_dataset

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "synthetic_helpcenter"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    dataset = build_synthetic_dataset(args.seed)
    save_beir_dir(dataset, args.out)
    print(
        f"wrote {len(dataset.corpus)} documents, {len(dataset.queries)} queries "
        f"to {args.out} (fingerprint {dataset.fingerprint()})"
    )


if __name__ == "__main__":
    main()
