# Builds the HME100K dictionary (CROHME symbols first, in order) and the image exclusion list.

import argparse
import re
from collections import Counter
from pathlib import Path

from ours.config import (
    CROHME_DICT,
    HME100K_DICT,
    HME100K_EXCLUDED as EXCLUDED,
    HME100K_FILES as FILES,
    HME100K_HF_REPO,
    HME100K_REVISION,
)

CHINESE = re.compile(r"[一-鿿]")


# Reads (image_path, label) pairs remotely, one row group at a time with retries.
def read_labels(split, retries=5):
    import fsspec
    import pyarrow.parquet as pq

    columns = ["image_path", "label"]
    rows = []
    for name in FILES[split]:
        url = (
            f"https://huggingface.co/datasets/{HME100K_HF_REPO}"
            f"/resolve/{HME100K_REVISION}/{name}"
        )
        with fsspec.open(url, block_size=2**20) as f:
            n_groups = pq.ParquetFile(f).metadata.num_row_groups
        for i in range(n_groups):
            for attempt in range(retries):
                try:
                    with fsspec.open(url, block_size=2**20) as f:
                        group = pq.ParquetFile(f).read_row_group(i, columns=columns)
                    break
                except Exception as e:
                    if attempt == retries - 1:
                        raise
                    print(f"  {name} chunk {i}: {type(e).__name__}, retrying")
            rows += zip(*(group.column(c).to_pylist() for c in columns))
    return rows


# Returns old symbols in order followed by new ones sorted, plus symbol counts.
def build_dictionary(crohme_symbols, train_labels):
    counts = Counter(sym for label in train_labels for sym in label.split())
    known = set(crohme_symbols)
    new = sorted(sym for sym in counts if sym not in known)
    return list(crohme_symbols) + new, counts


def main():
    parser = argparse.ArgumentParser(description="Build the HME100K dictionary and exclusion list.")
    parser.add_argument("--out", type=Path, default=HME100K_DICT)
    parser.add_argument("--excluded", type=Path, default=EXCLUDED)
    args = parser.parse_args()

    crohme = CROHME_DICT.read_text(encoding="utf-8").split()
    excluded = []

    train = read_labels("train")
    kept = []
    for path, label in train:
        if CHINESE.search(label):
            excluded.append(("train", path, "chinese_text"))
        else:
            kept.append(label)
    lines, counts = build_dictionary(crohme, kept)

    known = set(lines)
    test = read_labels("test")
    missing = Counter()
    for path, label in test:
        unknown = sorted({s for s in label.split() if s not in known})
        if unknown:
            missing.update(unknown)
            excluded.append(("test", path, "symbol_not_in_train:" + ",".join(unknown)))

    unused = [s for s in crohme if s not in counts]
    n_train_ex = len(train) - len(kept)
    n_test_ex = len(excluded) - n_train_ex
    print(f"train labels: {len(train)}, excluded {n_train_ex} with Chinese text, kept {len(kept)}")
    print(f"test labels: {len(test)}, excluded {n_test_ex} using symbols not in train: {dict(missing)}")
    print(f"CROHME symbols: {len(crohme)} ({len(unused)} unused in kept train labels: {unused})")
    print(f"new symbols: {len(lines) - len(crohme)}")
    print(f"dictionary size: {len(lines)} (+3 for <pad> <sos> <eos> = {len(lines) + 3})")

    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    rows = ["split\timage_path\treason"] + ["\t".join(r) for r in excluded]
    args.excluded.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {args.out} and {args.excluded}")


if __name__ == "__main__":
    main()
