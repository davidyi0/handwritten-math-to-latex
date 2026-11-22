# The test subset: the 2,000 HME100K test images every model is scored on
# (run: python -m ours.test_splits [--data DIR] [--export DIR]).


import argparse
import re
from pathlib import Path

from ours.config import HME100K_EXCLUDED, HME100K_FILES, REPO_ROOT

TEST_SUBSET = REPO_ROOT / "ours/hme100k_test_subset.tsv"
SUBSET_SIZE, SEED = 2000, 7
DIFFICULTY = {"easy": 14, "medium": 22, "hard": None}
COLUMNS = ["id", "image_path", "label"]


def image_id(image_path):
    return re.split(r"[\\/]", image_path)[-1].rsplit(".", 1)[0]


def difficulty(label):
    n = len(label.split())
    return next(name for name, longest in DIFFICULTY.items() if longest is None or n <= longest)


def read_excluded():
    lines = HME100K_EXCLUDED.read_text(encoding="utf-8").splitlines()[1:]
    return {line.split("\t")[1] for line in lines if line.strip()}


def read_test_subset(path=TEST_SUBSET):
    """The test subset as a list of dicts: id, image_path and label, plus the difficulty worked out from the label."""
    header, *lines = Path(path).read_text(encoding="utf-8").splitlines()
    rows = [dict(zip(header.split("\t"), line.split("\t"))) for line in lines if line]
    for r in rows:
        r["difficulty"] = difficulty(r["label"])
    return rows


def write_test_subset(rows, path=TEST_SUBSET):
    lines = ["\t".join(COLUMNS)] + ["\t".join(r[c] for c in COLUMNS) for r in rows]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def slice_ids(rows, per_difficulty):
    """Ids of a fixed smaller slice of the subset (e.g. for Gemini on degraded images): `per_difficulty` images of
    each difficulty, picked with a fixed seed. A bigger slice always contains every smaller one."""
    import numpy as np

    rng = np.random.default_rng(SEED)
    picked = set()
    for name in DIFFICULTY:
        group = sorted(r["image_path"] for r in rows if difficulty(r["label"]) == name)
        picked |= {image_id(group[k]) for k in rng.permutation(len(group))[:per_difficulty]}
    return picked


def build(data_dir):
    """Picks the subset from the kept test images. Returns its rows sorted by image_path."""
    import numpy as np
    import pyarrow.parquet as pq

    excluded = read_excluded()
    pool = []
    for name in HME100K_FILES["test"]:
        table = pq.read_table(Path(data_dir) / name, columns=["image_path", "label"])
        pool += [(p, l) for p, l in zip(table.column("image_path").to_pylist(), table.column("label").to_pylist())
                 if p not in excluded]
    pool.sort()
    rng = np.random.default_rng(SEED)
    sizes = [SUBSET_SIZE // 3 + (k < SUBSET_SIZE % 3) for k in range(3)]
    picked = []
    for name, size in zip(DIFFICULTY, sizes):
        group = [(p, l) for p, l in pool if difficulty(l) == name]
        print(f"{name}: {size} of {len(group)} kept test images")
        picked += [group[k] for k in rng.permutation(len(group))[:size]]
    rows = [{"id": image_id(p), "image_path": p, "label": l} for p, l in sorted(picked)]
    assert len({r["id"] for r in rows}) == len(rows), "image ids are not unique"
    assert not any("\t" in r["label"] or "\n" in r["label"] for r in rows), "a label contains a tab or newline"
    return rows


def export_images(data_dir, rows, out_dir):
    import pyarrow.parquet as pq

    wanted = {r["image_path"]: r["id"] for r in rows}
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for name in HME100K_FILES["test"]:
        pf = pq.ParquetFile(Path(data_dir) / name)
        for g in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(g, columns=["image_path", "image"])
            for p, image in zip(table.column("image_path").to_pylist(), table.column("image").to_pylist()):
                if p in wanted:
                    data = image["bytes"]
                    ext = ".png" if data.startswith(b"\x89PNG") else ".jpg" if data.startswith(b"\xff\xd8") else ""
                    if not ext:
                        raise ValueError(f"{p} is neither PNG nor JPEG")
                    (out_dir / f"{wanted[p]}{ext}").write_bytes(data)
                    written += 1
    print(f"wrote {written} images to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Write the HME100K test subset, or export its images.")
    parser.add_argument("--data", type=Path, default=Path("/content/hme100k"), help="the --local folder of ours.fetch_data")
    parser.add_argument("--subset", type=Path, default=TEST_SUBSET, help="the subset file to write (or read, with --export)")
    parser.add_argument("--export", type=Path, help="write the images listed in the subset file to this folder")
    args = parser.parse_args()

    if args.export:
        export_images(args.data, read_test_subset(args.subset), args.export)
        return
    rows = build(args.data)
    write_test_subset(rows, args.subset)
    print(f"wrote {len(rows)} images to {args.subset}")


if __name__ == "__main__":
    main()
