# Writes the validation split: a seeded random 5% of the kept HME100K training images.

import argparse
from pathlib import Path

import numpy as np

from ours.build_vocab import read_labels
from ours.config import HME100K_EXCLUDED, HME100K_VAL_SPLIT


def main():
    parser = argparse.ArgumentParser(description="Write the HME100K validation split.")
    parser.add_argument("--out", type=Path, default=HME100K_VAL_SPLIT)
    parser.add_argument("--fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    lines = HME100K_EXCLUDED.read_text(encoding="utf-8").splitlines()[1:]
    excluded = {line.split("\t")[1] for line in lines if line.strip()}
    paths = sorted(p for p, _ in read_labels("train") if p not in excluded)
    n = round(len(paths) * args.fraction)
    picked = sorted(np.random.default_rng(args.seed).permutation(paths)[:n].tolist())

    args.out.write_text("\n".join(picked) + "\n", encoding="utf-8", newline="\n")
    print(f"kept train images: {len(paths)}, validation: {n}; wrote {args.out}")


if __name__ == "__main__":
    main()
